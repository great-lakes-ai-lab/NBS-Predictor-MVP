# Download/Pre-process CFS forecast data
# Lindsay Fitzpatrick
# ljob@umich.edu
# 08/28/2024

# This script:
# 1. Downloads CFS forecast data from the AWS as grib2 files. 
# 2. Opens the grib2 files, calculates total basin, lake, and land, precipitation, evaporation, and average 2m air temperature. 
# 3. These calculations are then added to the CSV files. 

# This script needs the following files:

# - GL_mask.nc
# - CFS_EVAP_forecasts_Sums_CMS.csv
# - CFS_PCP_forecasts_Sums_CMS.csv
# - CFS_TMP_forecasts_Avgs_K.csv

from datetime import datetime, timedelta
import os
import sys
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import cfgrib
import pandas as pd
import netCDF4 as nc
import numpy as np
import shutil

# Path to download data to
dir = 'C:/Users/fitzpatrick/Desktop/Data/'
# Location of the mask file
mask_file = dir + 'Input/GL_mask.nc'

# Location of existing CSV files or path/name to new CSV files
tmp_csv = dir + 'CFS_TMP_forecasts_Avgs_K.csv'
evap_csv = dir + 'CFS_EVAP_forecasts_Sums_CMS.csv'
pcp_csv = dir + 'CFS_PCP_forecasts_Sums_CMS.csv'

# IF YOU ARE CREATING NEW CSV FILES:
# Then you need to define the start and end dates
# IF YOU ARE ADDING TO EXISTING CSV FILES:
# Then these dates will be ignored and the script will automatically pull
# the last date from the existing CSV files and continue the forecast from there.
start_date = '2024-08-31'
end_date = '2024-08-31'

## Presets ##
products = ['pgb','flx']
utc = ['00','06','12','18']

# Define mask variables
mask_variables = ['eri_basin','eri_lake','eri_land',
                 'hur_basin','hur_lake','hur_land',
                 'ont_basin','ont_lake','ont_land',
                 'mic_basin','mic_lake','mic_land',
                 'sup_basin','sup_lake','sup_land']

#AWS bucket name to locate the CFS forecast
bucket_name = 'noaa-cfs-pds'

def download_grb2_aws(product, bucket_name, folder_path, download_dir):
    """
    Download the CFS forecast from AWS

    Parameters:
    - product: 'flx' or 'pgb'
    - bucket_name: for CFS data it is 'noaa-cfs-pds'
    - folder_path: the url path to data
    - download_dir: location to download data to
    """
    num_files_downloaded = 0

    # Create a boto3 client for S3
    s3_config = Config(signature_version=UNSIGNED)
    s3 = boto3.client('s3', config=s3_config)

    # List all objects in the specified folder path
    continuation_token = None
    objects = []

    # Use a loop to handle pagination
    while True:
        list_objects_args = {'Bucket': bucket_name, 'Prefix': folder_path}
        if continuation_token:
            list_objects_args['ContinuationToken'] = continuation_token

        list_objects_response = s3.list_objects_v2(**list_objects_args)

        objects.extend(list_objects_response.get('Contents', []))

        if not list_objects_response.get('IsTruncated', False):
            break

        continuation_token = list_objects_response.get('NextContinuationToken')

    # Iterate over each object and download if it ends with '.grb2'
    for obj in objects:
        key = obj['Key']
        if product in key and key.endswith('grib.grb2'): #if key.endswith('.grb2'):
            local_file_path = os.path.join(download_dir, os.path.relpath(key, folder_path))

            # Ensure the directory structure exists
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            # Download the file
            s3.download_file(bucket_name, key, local_file_path)
            num_files_downloaded += 1

            print(f"Downloaded: {key}")

def initialize_dataframes(tmp_csv, evap_csv, pcp_csv):
    """
    Initialize new DataFrames if CSV files do not exist.
    """
    if os.path.exists(tmp_csv):
        df_tmp_forecasts = pd.read_csv(tmp_csv)
    else:
        df_tmp_forecasts = pd.DataFrame(columns=['cfs_run', 'forecast_year', 'forecast_month'] + mask_variables)
    
    if os.path.exists(evap_csv):
        df_evap_forecasts = pd.read_csv(evap_csv)
    else:
        df_evap_forecasts = pd.DataFrame(columns=['cfs_run', 'forecast_year', 'forecast_month'] + mask_variables)
    
    if os.path.exists(pcp_csv):
        df_pcp_forecasts = pd.read_csv(pcp_csv)
    else:
        df_pcp_forecasts = pd.DataFrame(columns=['cfs_run', 'forecast_year', 'forecast_month'] + mask_variables)
    
    return df_tmp_forecasts, df_evap_forecasts, df_pcp_forecasts

def get_files(directory, affix, identifier):
    """
    Get a list of all GRIB2 files in the specified directory.

    Parameters:
    - directory (str): Path to the directory containing files.
    - affix (str): 'prefix' or 'suffix'
    - identifier (str):  (ie. 'pgb', 'flx', '.grb2', or '.nc')
    Returns:
    - List of file paths to the GRIB2 files.
    """
    files = []
    for file_name in os.listdir(directory):
        if affix == 'suffix': # ends with
            if file_name.endswith(identifier):
                file_path = os.path.join(directory, file_name)
                files.append(file_path)
        elif affix == 'prefix': # begins with
            if file_name.startswith(identifier):
                file_path = os.path.join(directory, file_name)
                files.append(file_path)
    return files

def delete_directory(directory_path):
    # Check if the directory exists
    if not os.path.isdir(directory_path):
        print(f"The directory {directory_path} does not exist.")
        return
    try:
        # Remove the entire directory tree
        shutil.rmtree(directory_path)
        print(f"Successfully deleted the directory and all its contents: {directory_path}")
    except Exception as e:
        print(f"Error deleting {directory_path}: {e}")

def calculate_grid_cell_areas(lon, lat):
    # Calculate grid cell areas
    # Assuming lat and lon are 1D arrays
    # Convert latitude to radians

    R = 6371000.0  # Radius of Earth in meters
    lat_rad = np.radians(lat)

    # Calculate grid cell width in radians
    dlat = np.radians(lat[1] - lat[0])
    dlon = np.radians(lon[1] - lon[0])

    # Calculate area of each grid cell in square kilometers
    area = np.zeros((len(lat), len(lon)))
    for i in range(len(lat)):
        for j in range(len(lon)):
            area[i, j] = R**2 * dlat * dlon * np.cos(lat_rad[i])

    return area

def calculate_evaporation(temperature_K, latent_heat):
    lamda=(2.501-(0.002361*(temperature_K-273.15)))
    evaporation_rate=((latent_heat)*0.000001)/lamda

    return evaporation_rate # kg/m2 per s

def process_grib_files(download_dir, df_tmp_forecasts, df_evap_forecasts, df_pcp_forecasts, mask_lat, mask_lon, mask_ds, mask_variables, area, calculate_evaporation):
    # Find all the .grb2 files in the directory
    file_list = get_files(download_dir, 'suffix', '.grb2')
    index = len(df_tmp_forecasts) if not df_tmp_forecasts.empty else 0  # Picks up on the last line of the CSV

    for grib2_file in file_list:

        filename = os.path.basename(grib2_file)
        parts = filename.split('.')
        cfs_run = parts[2]
        date_part = parts[3]  # Assuming parts[2] is in the format YYYYMM
        forecast_year = date_part[:4]
        forecast_month = date_part[4:6]

        if filename.startswith('flxf'):

            # Open the flx file at the 2m level to pull the 2m air temperature
            flx_2mabove = cfgrib.open_dataset(grib2_file, engine='cfgrib', filter_by_keys={'typeOfLevel': 'heightAboveGround', 'level': 2})
            df_tmp_forecasts.loc[index, 'cfs_run'] = cfs_run
            df_tmp_forecasts.loc[index, 'forecast_year'] = forecast_year
            df_tmp_forecasts.loc[index, 'forecast_month'] = forecast_month
            mean2t = flx_2mabove['mean2t']

            # Cut the variable to the mask domain
            mean2t_cut = mean2t.sel(
                latitude=slice(mask_lat.max(), mask_lat.min()),
                longitude=slice(mask_lon.min(), mask_lon.max())
            )
            # Remap and upscale the variable to match the mask domain
            mean2t_remap = mean2t_cut.interp(latitude=mask_lat, longitude=mask_lon, method='linear')
            
            # Calculate mean2t for each of the mask variables (i.e., eri_lake, eri_basin, etc.)
            for mask_var in mask_variables:

                mask = mask_ds.variables[mask_var][:]
                # Take the mean over the mask area
                tmp_avg = np.mean(mean2t_remap * mask)

                df_tmp_forecasts.loc[index, mask_var] = tmp_avg.data

            ###############################################################################

            # Open the flx file again but at the surface level to pull the latent heat flux
            flx_surface = cfgrib.open_dataset(grib2_file, engine='cfgrib', filter_by_keys={'typeOfLevel': 'surface'})
            df_evap_forecasts.loc[index, 'cfs_run'] = cfs_run
            df_evap_forecasts.loc[index, 'forecast_year'] = forecast_year
            df_evap_forecasts.loc[index, 'forecast_month'] = forecast_month
            mslhf = flx_surface['mslhf']
            
            # Cut the variable to the mask domain
            mslhf_cut = mslhf.sel(
                latitude=slice(mask_lat.max(), mask_lat.min()),
                longitude=slice(mask_lon.min(), mask_lon.max())
            )
            # Remap and upscale the variable to match the mask domain
            mslhf_remap = mslhf_cut.interp(latitude=mask_lat, longitude=mask_lon, method='linear')
            
            # Calculate evaporation across the entire domain using air temp and latent heat flux
            evap = calculate_evaporation(mean2t_remap, mslhf_remap)
            
            # Calculate evaporation for each of the mask variables (i.e., eri_lake, eri_basin, etc.)
            for mask_var in mask_variables:
                
                mask = mask_ds.variables[mask_var][:]
                total_evap = (np.sum(evap * area * mask)) # Converts kg/s/m2 to kg/s
                # Convert kg/s to m³/s (assuming density of water ≈ 1000 kg/m³)
                evap_cms = total_evap / 1000.0

                df_evap_forecasts.loc[index, mask_var] = evap_cms.data

        ###############################################################################

        elif filename.startswith('pgbf'):

            # Open the pgb file at the surface level to pull the precipitation
            pgb_surface = cfgrib.open_dataset(grib2_file, engine='cfgrib', filter_by_keys={'typeOfLevel': 'surface'})
            df_pcp_forecasts.loc[index, 'cfs_run'] = cfs_run
            df_pcp_forecasts.loc[index, 'forecast_year'] = forecast_year
            df_pcp_forecasts.loc[index, 'forecast_month'] = forecast_month
            pcp = pgb_surface['tp']  # Total precipitation
            
            # Cut the variable to the mask domain
            pcp_cut = pcp.sel(
                latitude=slice(mask_lat.max(), mask_lat.min()),
                longitude=slice(mask_lon.min(), mask_lon.max())
            )
            # Remap and upscale the variable to match the mask domain
            pcp_remap = pcp_cut.interp(latitude=mask_lat, longitude=mask_lon, method='linear')
            
            for mask_var in mask_variables:
                mask = mask_ds.variables[mask_var][:]
                
                # Convert precipitation from kg/m² per 6 hours to kg/m² per second
                pcp_per_s = pcp_remap / 21600.0 # seconds in 6hrs
                total_pcp_kg_per_s = (np.sum(pcp_per_s * area * mask)) # kg/s

                # Convert kg/s to m³/s (assuming density of water ≈ 1000 kg/m³)
                total_pcp_cms = total_pcp_kg_per_s / 1000.0
                df_pcp_forecasts.loc[index, mask_var] = total_pcp_cms.data

        print(f'Done with {filename}')

        index += 1

# Open existing CSVs or create empty dataframes to save to new CSVs
df_tmp_forecasts, df_evap_forecasts, df_pcp_forecasts = initialize_dataframes(tmp_csv, evap_csv, pcp_csv)

# If we are starting a new CSV, then user must input dates above to pull data
if df_tmp_forecasts.empty:
    print("Creating new files.")
    start_date = datetime.strptime(start_date, "%Y-%m-%d") # User input above
    end_date = datetime.strptime(end_date, "%Y-%m-%d") # User input above
else:
    # If we are adding to an existing CSV, then pull the last date from the CSV
    # and continue from there
    last_cfs = df_tmp_forecasts['cfs_run'].astype(str).iloc[-1][:8]
    start_date = datetime.strptime(last_cfs, '%Y%m%d') + timedelta(days=1)
    # Pull all the forecasts days up to yesterday (the most complete forecast)
    end_date = datetime.now() - timedelta(days=1)

# Check if start_date is equal to or after end_date
if start_date > end_date:
    print("The files are up-to-date.")
    sys.exit()  # Stop the script

print(f"Starting from: {start_date.strftime('%Y-%m-%d')} and continuing through: {end_date.strftime('%Y-%m-%d')}")

# Create a date range
date_range = pd.date_range(start=start_date, end=end_date)
# Convert to integer format YYYYMMDD
dates_array = date_range.strftime('%Y%m%d').astype(int)

# Open the mask file and calculate the grid cell areas
mask_ds = nc.Dataset(mask_file)
mask_lat = mask_ds.variables['latitude'][:]
mask_lon = mask_ds.variables['longitude'][:]
area = calculate_grid_cell_areas(mask_lon, mask_lat)

for date in dates_array:
    print(f"Beginning {date}.")
    download_dir = f'{dir}{date}/CFS/'
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # Uses AWS to download the grib2 files
    for utc_time in utc:
        for product in products:
            folder_path = f'cfs.{date}/{utc_time}/monthly_grib_01/'
            download_grb2_aws(product, bucket_name, folder_path, download_dir)

    process_grib_files(download_dir, df_tmp_forecasts, df_evap_forecasts, df_pcp_forecasts, mask_lat, mask_lon, mask_ds, mask_variables, area, calculate_evaporation)   
    
    # Save the updated DataFrames to CSV files
    df_tmp_forecasts.to_csv(tmp_csv, sep=',', index=False)
    df_evap_forecasts.to_csv(evap_csv, sep=',', index=False)
    df_pcp_forecasts.to_csv(pcp_csv, sep=',', index=False)

    # Delete downloaded grib2 files
    #delete_directory(download_dir)
    
    print(f"Done with {date}.")

mask_ds.close()