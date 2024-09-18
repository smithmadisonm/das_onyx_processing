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

def get_file_metadata(file_path):
    with h5.File(file_path, 'r') as f:
        return {
            'file_path': file_path,
            'start_time': datetime.strptime(os.path.basename(file_path)[10:26], '%Y-%m-%d_%H.%M'),
            'pulse_rate': f['Acquisition'].attrs['PulseRate'],
            'spatial_sampling_interval': f['Acquisition'].attrs['SpatialSamplingInterval'],
            'shape': f['Acquisition']['Raw[0]']['RawData'].shape
        }

def preload_files(base_path, start_time, end_time):
    file_metadata = []
    for di in datetime_range(start_time, end_time, timedelta(minutes=1)):
        path = base_path + 'decimator_' + datetime.strftime(di, '%Y-%m-%d_%H.%M') + '*.h5'
        files = glob.glob(path)
        for file_path in sorted(files):
            file_metadata.append(get_file_metadata(file_path))
    return sorted(file_metadata, key=lambda x: x['start_time'])

def load_h5_into_xr_chunk(file_metadata_chunk):
    ds_DAS_chunk = None
    attrs = None

    for metadata in file_metadata_chunk:
        with h5.File(metadata['file_path'], 'r') as f:
            f_data = f['Acquisition']['Raw[0]']['RawData'][:]
            channels = np.arange(0, metadata['shape'][1]) * metadata['spatial_sampling_interval']

            file_timestr = os.path.basename(metadata['file_path'])[10:-7]
            file_datetime = datetime.strptime(file_timestr, '%Y-%m-%d_%H.%M.%S')
            f_seconds = metadata['shape'][0] / metadata['pulse_rate']
            dt_ms = 1000000 / metadata['pulse_rate']

            f_time = [dt for dt in datetime_range(file_datetime, file_datetime + timedelta(seconds=f_seconds), timedelta(microseconds=dt_ms))]

            data_DAS = {'strain': (['time', 'channels'], f_data, {'units': '', 'long_name': 'strain data'})}

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

def das_butterworth_decimate_xarray(ds_DAS_chunk, fs_target):
    fs = ds_DAS_chunk.attrs['PulseRate']
    t_inc = int(fs/fs_target)

    cutoff = fs_target
    nyq = 0.5 * fs
    order = 1
    normal_cutoff = cutoff / nyq
    b_butter, a_butter = signal.butter(order, normal_cutoff, btype='low', analog=False)

    strain_data = ds_DAS_chunk.strain.values

    # Apply the filter to each channel sequentially
    ds_DAS_deci = np.array([signal.filtfilt(b_butter, a_butter, channel)[::t_inc] for channel in strain_data.T]).T

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

def process_time_chunk(di, time_chunk, all_file_metadata, fs_target, output_dir):
    print(f"Processing chunk starting at {di}, current time {datetime.now()}")
    
    chunk_end = di + timedelta(minutes=time_chunk)
    file_metadata_chunk = [metadata for metadata in all_file_metadata 
                           if di <= metadata['start_time'] < chunk_end]
    
    if not file_metadata_chunk:
        print(f"No files found for time chunk starting at {di}")
        return

    ds_DAS_chunk = load_h5_into_xr_chunk(file_metadata_chunk)
    
    if ds_DAS_chunk is None:
        print(f"No data found for time chunk starting at {di}")
        return

    # Select exactly the time_chunk minutes from the full combined array
    ds_DAS_chunk = ds_DAS_chunk.sel(time=slice(di, chunk_end))
    
    if len(ds_DAS_chunk.time) < time_chunk * ds_DAS_chunk.attrs['PulseRate'] * 60:
        print(f'Warning: Missing data: {len(ds_DAS_chunk.time)} should be {time_chunk*ds_DAS_chunk.attrs["PulseRate"]*60}')

    strain_deci_butter_all = das_butterworth_decimate_xarray(ds_DAS_chunk, fs_target)

    output_file = f'{output_dir}Onyx_{di.strftime("%Y-%m-%d_%H.%M")}_{fs_target}hz.nc'
    strain_deci_butter_all.to_netcdf(output_file)

def process_data(onyx_path, start_time, end_time, time_chunk, fs_target, output_dir, n_jobs):
    if not onyx_path.endswith('/'):
        onyx_path += '/'
    if not output_dir.endswith('/'):
        output_dir += '/'

    print("Preloading files...")
    all_file_metadata = preload_files(onyx_path, start_time, end_time)
    print(f"Preloaded {len(all_file_metadata)} files.")

    # Create a pool of workers
    with mp.Pool(processes=n_jobs) as pool:
        # Create a partial function with fixed parameters
        process_func = partial(process_time_chunk, 
                               time_chunk=time_chunk, 
                               all_file_metadata=all_file_metadata, 
                               fs_target=fs_target, 
                               output_dir=output_dir)

        # Generate time chunks
        time_chunks = list(datetime_range(start_time, end_time, timedelta(minutes=time_chunk)))

        # Map the processing function to the time chunks
        pool.map(process_func, time_chunks)

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
