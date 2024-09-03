import numpy as np
import h5py as h5
from datetime import datetime, timedelta
import xarray as xr
import glob
from scipy import signal
import argparse
import os

def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta


def select_files(base_path, time_target, time_chunk):
    files_target_chunk = []
    chunk_range = datetime_range(time_target, time_target + timedelta(minutes=time_chunk), timedelta(minutes=1))
    for time_temp in chunk_range:
        path = base_path + 'decimator_' + datetime.strftime(time_temp,'%Y-%m-%d_%H.%M') +'*.h5'
        files_target_chunk.extend(glob.glob(path))
    return sorted(files_target_chunk)

def load_h5_into_xr_chunk(files_target_chunk):
    ds_DAS_chunk = None
    attrs = None

    for file_path in files_target_chunk:
        with h5.File(file_path, 'r') as f:
            f_data = f['Acquisition']['Raw[0]']['RawData']
            channels = np.arange(0, f_data.shape[1]) * f['Acquisition'].attrs['SpatialSamplingInterval']

            file_timestr = os.path.basename(file_path)[10:-7]
            file_datetime = datetime.strptime(file_timestr, '%Y-%m-%d_%H.%M.%S')
            f_seconds = f_data.shape[0] / f['Acquisition'].attrs['PulseRate']
            dt_ms = 1000000 / f['Acquisition'].attrs['PulseRate']

            f_time = [dt for dt in datetime_range(file_datetime, file_datetime + timedelta(seconds=f_seconds), timedelta(microseconds=dt_ms))]

            # Use memory mapping for large datasets
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
    #“t_inc" parameter is the an integer representing the multiples you need to downsample
    #where fs is the original sampling rate and 50 is the frequency you want
    fs = ds_DAS_chunk.attrs['PulseRate']
    t_inc = int(fs/fs_target)

    # Define butterworth filter
    cutoff = fs_target
    nyq = 0.5 * fs
    order = 1
    normal_cutoff = cutoff / nyq
    b_butter, a_butter = signal.butter(order, normal_cutoff, btype='low', analog=False)

    strain_data = ds_DAS_chunk.strain.values
    strain_butter = signal.filtfilt(b_butter, a_butter, strain_data, axis=0)
    
    # Decimate the filtered data
    ds_DAS_deci = strain_butter[::t_inc, :]

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


def process_data(onyx_path, start_time, end_time, time_chunk, fs_target, output_dir):
    if not onyx_path.endswith('/'):
        onyx_path += '/'
    if not output_dir.endswith('/'):
        output_dir += '/'

    for di in datetime_range(start_time, end_time, timedelta(minutes=30)):
        print(str(di)+', current time '+str(datetime.now()))
        files_target_chunk = select_files(onyx_path, di, time_chunk)
        
        if not files_target_chunk:
            print(f"No files found for time chunk starting at {di}")
            continue

        ds_DAS_chunk = load_h5_into_xr_chunk(files_target_chunk)
        
        if ds_DAS_chunk is None:
            print(f"No data found for time chunk starting at {di}")
            continue

        # Select exactly the 30 minutes from the full combined array
        ds_DAS_chunk = ds_DAS_chunk.sel(time=slice(di, di+timedelta(minutes=time_chunk)))
        
        if len(ds_DAS_chunk.time) < time_chunk * ds_DAS_chunk.attrs['PulseRate'] * 60:
            print('Warning: Missing data: '+str(len(ds_DAS_chunk.time)) + ' should be ' + str(time_chunk*ds_DAS_chunk.attrs['PulseRate']*60))

        strain_deci_butter_all = das_butterworth_decimate_xarray(ds_DAS_chunk, fs_target)

        output_file = f'{output_dir}Onyx_{di.strftime("%Y-%m-%d_%H.%M")}_{fs_target}hz.nc'
        strain_deci_butter_all.to_netcdf(output_file)

def main():
    parser = argparse.ArgumentParser(description="Process DAS data and decimate it.")
    parser.add_argument("onyx_path", type=str, help="Path to the Onyx DAS data")
    parser.add_argument("start_time", type=str, help="Start time in format YYYY-MM-DD HH:MM")
    parser.add_argument("end_time", type=str, help="End time in format YYYY-MM-DD HH:MM")
    parser.add_argument("output_dir", type=str, help="Output directory for processed files")
    parser.add_argument("--time_chunk", type=int, default=30, help="Time chunk in minutes")
    parser.add_argument("--fs_target", type=int, default=5, help="Target frequency in Hz")

    args = parser.parse_args()

    start_time = datetime.strptime(args.start_time, "%Y-%m-%d %H:%M")
    end_time = datetime.strptime(args.end_time, "%Y-%m-%d %H:%M")

    process_data(args.onyx_path, start_time, end_time, args.time_chunk, args.fs_target, args.output_dir)

if __name__ == "__main__":
    main()
