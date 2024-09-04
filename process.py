import numpy as np
import h5py as h5
from datetime import datetime, timedelta
import xarray as xr
import glob
from scipy import signal
import argparse
import os
import multiprocessing as mp
from functools import partial

def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta

class FileCollection:
    def __init__(self):
        self.files = {}

    def add_file(self, file_path):
        if file_path not in self.files:
            self.files[file_path] = h5.File(file_path, driver="core", mode='r', backing_store=False)

    def get_file(self, file_path):
        return self.files.get(file_path)

    def close_all(self):
        for file in self.files.values():
            file.close()
        self.files.clear()

file_collection = FileCollection()

def preload_files(base_path, start_time, end_time):
    file_metadata = []
    for di in datetime_range(start_time, end_time, timedelta(minutes=1)):
        path = base_path + 'decimator_' + datetime.strftime(di, '%Y-%m-%d_%H.%M') + '*.h5'
        files = glob.glob(path)
        for file_path in sorted(files):
            file_collection.add_file(file_path)
            f = file_collection.get_file(file_path)
            metadata = {
                'file_path': file_path,
                'start_time': di,
                'pulse_rate': f['Acquisition'].attrs['PulseRate'],
                'spatial_sampling_interval': f['Acquisition'].attrs['SpatialSamplingInterval'],
                'shape': f['Acquisition']['Raw[0]']['RawData'].shape
            }
            file_metadata.append(metadata)
    return sorted(file_metadata, key=lambda x: x['start_time'])

def load_h5_into_xr_chunk(file_metadata_chunk):
    ds_DAS_chunk = None
    attrs = None

    for metadata in file_metadata_chunk:
        f = file_collection.get_file(metadata['file_path'])
        f_data = f['Acquisition']['Raw[0]']['RawData']
        channels = np.arange(0, metadata['shape'][1]) * metadata['spatial_sampling_interval']

        file_timestr = os.path.basename(metadata['file_path'])[10:-7]
        file_datetime = datetime.strptime(file_timestr, '%Y-%m-%d_%H.%M.%S')
        f_seconds = metadata['shape'][0] / metadata['pulse_rate']
        dt_ms = 1000000 / metadata['pulse_rate']

        f_time = [dt for dt in datetime_range(file_datetime, file_datetime + timedelta(seconds=f_seconds), timedelta(microseconds=dt_ms))]

        data_DAS = {'strain': (['time', 'channels'], f_data[:], {'units': '', 'long_name': 'strain data'})}

        coords = {'time': (['time'], f_time), 'channels': (['channels'], channels)}
        
        attrs = {k: v.decode("utf-8") if isinstance(v, bytes) else v for k, v in f['Acquisition'].attrs.items()}

        ds_DAS = xr.Dataset(data_vars=data_DAS, coords=coords)

        if ds_DAS_chunk is None:
            ds_DAS_chunk = ds_DAS
        else:
            ds_DAS_chunk = xr.concat([ds_DAS_chunk, ds_DAS], dim='time')

    if ds_DAS_chunk is not None:
        ds_DAS_chunk = ds_DAS_chunk.assign_attrs(attrs)

    return ds_DAS_chunk

def process_channel(channel_data, b_butter, a_butter, t_inc):
    filtered_channel = signal.filtfilt(b_butter, a_butter, channel_data)
    return filtered_channel[::t_inc]

def das_butterworth_decimate_xarray(ds_DAS_chunk, fs_target, n_jobs=None):
    fs = ds_DAS_chunk.attrs['PulseRate']
    t_inc = int(fs/fs_target)

    cutoff = fs_target
    nyq = 0.5 * fs
    order = 1
    normal_cutoff = cutoff / nyq
    b_butter, a_butter = signal.butter(order, normal_cutoff, btype='low', analog=False)

    strain_data = ds_DAS_chunk.strain.values

    # Determine the number of processes to use
    if n_jobs is None:
        n_jobs = mp.cpu_count()

    # Create a partial function with fixed parameters
    process_func = partial(process_channel, b_butter=b_butter, a_butter=a_butter, t_inc=t_inc)

    # Use multiprocessing to apply the filter to each channel
    with mp.Pool(processes=n_jobs) as pool:
        results = pool.map(process_func, strain_data.T)

    # Combine the results
    ds_DAS_deci = np.array(results).T

    attrs_deci = ds_DAS_chunk.attrs.copy()
    attrs_deci['PulseRateDecimated'] = fs_target
    attrs_deci['DecimationFilterType'] = 'butterworth'

    time_values = ds_DAS_chunk.time.values[::t_inc]
    
    coords = {
        'time': ('time', time_values),
        'channels': ('channels', ds_DAS_chunk.channels.values)
    }

    data_deci = {
        'strain': (['time', 'channels'], ds_DAS_deci, {'units': '', 'long_name': 'decimated strain data'})
    }
    
    strain_deci_butter_all = xr.Dataset(data_vars=data_deci, coords=coords, attrs=attrs_deci)

    return strain_deci_butter_all


def process_data(onyx_path, start_time, end_time, time_chunk, fs_target, output_dir, n_jobs):
    if not onyx_path.endswith('/'):
        onyx_path += '/'
    if not output_dir.endswith('/'):
        output_dir += '/'

    print("Preloading files...")
    all_file_metadata = preload_files(onyx_path, start_time, end_time)
    print(f"Preloaded {len(all_file_metadata)} files.")

    try:
        for di in datetime_range(start_time, end_time, timedelta(minutes=time_chunk)):
            print(str(di)+', current time '+str(datetime.now()))
            
            chunk_end = di + timedelta(minutes=time_chunk)
            file_metadata_chunk = [metadata for metadata in all_file_metadata 
                                   if di <= metadata['start_time'] < chunk_end]
            
            if not file_metadata_chunk:
                print(f"No files found for time chunk starting at {di}")
                continue

            ds_DAS_chunk = load_h5_into_xr_chunk(file_metadata_chunk)
            
            if ds_DAS_chunk is None:
                print(f"No data found for time chunk starting at {di}")
                continue

            # Select exactly the time_chunk minutes from the full combined array
            ds_DAS_chunk = ds_DAS_chunk.sel(time=slice(di, chunk_end))
            
            if len(ds_DAS_chunk.time) < time_chunk * ds_DAS_chunk.attrs['PulseRate'] * 60:
                print('Warning: Missing data: '+str(len(ds_DAS_chunk.time)) + ' should be ' + str(time_chunk*ds_DAS_chunk.attrs['PulseRate']*60))

            strain_deci_butter_all = das_butterworth_decimate_xarray(ds_DAS_chunk, fs_target, n_jobs)

            output_file = f'{output_dir}Onyx_{di.strftime("%Y-%m-%d_%H.%M")}_{fs_target}hz.nc'
            strain_deci_butter_all.to_netcdf(output_file)
    finally:
        print("Closing all files...")
        file_collection.close_all()

def main():
    parser = argparse.ArgumentParser(description="Process DAS data and decimate it.")
    parser.add_argument("onyx_path", type=str, help="Path to the Onyx DAS data")
    parser.add_argument("start_time", type=str, help="Start time in format YYYY-MM-DD HH:MM")
    parser.add_argument("end_time", type=str, help="End time in format YYYY-MM-DD HH:MM")
    parser.add_argument("output_dir", type=str, help="Output directory for processed files")
    parser.add_argument("--time_chunk", type=int, default=30, help="Time chunk in minutes")
    parser.add_argument("--fs_target", type=int, default=5, help="Target frequency in Hz")
    parser.add_argument("--jobs", type=int, default=None, help="Number of parallel jobs to run")

    args = parser.parse_args()

    start_time = datetime.strptime(args.start_time, "%Y-%m-%d %H:%M")
    end_time = datetime.strptime(args.end_time, "%Y-%m-%d %H:%M")

    process_data(args.onyx_path, start_time, end_time, args.time_chunk, args.fs_target, args.output_dir, args.jobs)

if __name__ == "__main__":
    main()
