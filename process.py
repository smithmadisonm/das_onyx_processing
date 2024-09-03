import pandas as pd
import numpy as np
import h5py as h5
from matplotlib import pyplot as plt
from datetime import date, datetime, timedelta
import xarray as xr
import glob
from scipy import signal
from joblib import Parallel, delayed
import time as tm
import argparse

def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta

def load_h5_into_xr_chunk(base_path, time_target, time_chunk):
    #file names for all files including times within the 30 min chunk
    files_target_chunk = [glob.glob(base_path + 'decimator_' + datetime.strftime(time_temp,'%Y-%m-%d_%H.%M') +'*.h5') for time_temp in 
           datetime_range(time_target, time_target + timedelta(minutes=time_chunk), 
           timedelta(minutes=1))]
    
    ds_DAS_chunk = None
    attrs = None
    
    for fi in files_target_chunk:
        if not fi:
            print(f"No files found for time: {time_target}")
            continue
        else:
            f = h5.File(fi[0],'r')

            #data and key parameters
            f_data = np.array(f['Acquisition']['Raw[0]']['RawData'])
            f_sampcount = np.array(f['Acquisition']['Raw[0]']['RawDataSampleCount'])
            channels = np.arange(0,f_data.shape[1])*f['Acquisition'].attrs['SpatialSamplingInterval']

            #create actual times, where file time is in microseconds from #
            file_timestr = fi[0].split('/')[-1][10:-7]
            file_datetime = datetime.strptime(file_timestr, '%Y-%m-%d_%H.%M.%S')
            f_seconds = f_data.shape[0]/f['Acquisition'].attrs['PulseRate'] #length of time, in seconds, of array
            dt_ms = 1000000/f['Acquisition'].attrs['PulseRate'] #time in ms between each ...at 250 Hz, 4000 micros between each timestep

            f_time = [dt for dt in 
                   datetime_range(file_datetime, file_datetime + timedelta(seconds=f_seconds), 
                   timedelta(microseconds=dt_ms))]

            data_DAS = {'strain':(['time','channels'], f_data, 
                                {'units':'',
                               'long_name':'strain data'})}

            # define coordinates
            coords = {'time': (['time'], f_time),
                      'channels': (['channels'], channels)}
            #define attributes, all from hdf5 file
            attrs = dict()
            for fi,fi_attr in enumerate(f['Acquisition'].attrs.keys()):
                if isinstance(f['Acquisition'].attrs[fi_attr], bytes):
                    attrs[fi_attr] = f['Acquisition'].attrs[fi_attr].decode("utf-8")
                else:
                    attrs[fi_attr] = f['Acquisition'].attrs[fi_attr] 

            #create dataset
            ds_DAS = xr.Dataset(data_vars=data_DAS, 
                            coords=coords)

            if ds_DAS_chunk is None:
                ds_DAS_chunk = ds_DAS
            else:
                ds_DAS_chunk = xr.merge([ds_DAS_chunk, ds_DAS])
    
    if ds_DAS_chunk is None:
        print(f"No data found for time chunk starting at {time_target}")
        return None
    
    ds_DAS_chunk = ds_DAS_chunk.assign_attrs(attrs)

    #select exactly the 30 minutes from the full combined array
    fs = ds_DAS_chunk.attrs['PulseRate']
    ds_DAS_chunk = ds_DAS_chunk.sel(time=slice(time_target, time_target+timedelta(minutes=time_chunk)))
    if len(ds_DAS_chunk.time) < time_chunk*fs*60:
        print('Warning: Missing data: '+str(len(ds_DAS_chunk.time)) + ' should be ' + str(time_chunk*ds_DAS_chunk.attrs['PulseRate']*60))

    return ds_DAS_chunk

def das_butterworth_decimate_xarray(ds_DAS_chunk, fs_target):
    
    #“t_inc" parameter is the an integer representing the multiples you need to downsample
    #where fs is the original sampling rate and 50 is the frequency you want
    fs = ds_DAS_chunk.attrs['PulseRate']
    t_inc = int(fs/fs_target) 

    #initialize empty nan array for decimated data
    ds_DAS_deci = np.empty((len(ds_DAS_chunk.time[0::t_inc]),len(ds_DAS_chunk.channels)))
    ds_DAS_deci[:] = np.nan

    #butterworth filter, use for surface waves

    #define butterworth filter 
    cutoff = fs_target #desire cutoff frequency of filter, Hz
    nyq = 0.5*fs #nyquist frequency
    order = 1
    normal_cutoff = cutoff/nyq
    b_butter, a_butter = signal.butter(order,normal_cutoff,btype='low',analog=False)

    for i, ci in enumerate(ds_DAS_chunk.channels.values):
        strain = ds_DAS_chunk.strain.transpose().values[i]
        strain_butter = signal.filtfilt(b_butter, a_butter, strain)
        strain_deci_butter = strain_butter[::t_inc]
        ds_DAS_deci[:,i] = strain_deci_butter

    attrs_deci = ds_DAS_chunk.attrs.copy()
    attrs_deci['PulseRateDecimated'] = fs_target
    attrs_deci['DecimationFilterType'] = 'butterworth'

    # Ensure time values are numpy array
    time_values = ds_DAS_chunk.time.values[0::t_inc]
    
    coords = {
        'time': ('time', time_values),
        'channels': ('channels', ds_DAS_chunk.channels.values)
    }

    data_deci = {
        'strain': (['time','channels'], ds_DAS_deci, {'units':'', 'long_name':'decimated strain data'})
    }
    
    strain_deci_butter_all = xr.Dataset(data_vars=data_deci, coords=coords, attrs=attrs_deci)

    return strain_deci_butter_all


def process_data(onyx_path, start_time, end_time, time_chunk, fs_target, output_dir):
    # Add trailing slash if not present
    if not onyx_path.endswith('/'):
        onyx_path += '/'
    for di in datetime_range(start_time, end_time, timedelta(minutes=30)):
        print(str(di)+', current time '+str(datetime.now()))
        ds_DAS_chunk = load_h5_into_xr_chunk(onyx_path, di, time_chunk)
        if ds_DAS_chunk is None:
            print(f"Skipping time chunk starting at {di}")
            continue
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
