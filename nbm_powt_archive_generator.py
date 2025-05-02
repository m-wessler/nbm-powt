# Standard library imports
import os
import pygrib
import swifter
import time
import json
import zipfile
import warnings
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
from functools import partial
from glob import glob
from typing import List, Optional
from multiprocessing import Pool, cpu_count, set_start_method, get_context
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party library imports
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import requests
import itertools
from pydantic import BaseModel, Field

def mkdir_p(check_dir):
    from pathlib import Path
    Path(check_dir).mkdir(parents=True, exist_ok=True)
    return check_dir

# Set global configurations
warnings.filterwarnings('ignore')

# Multiprocess settings
mp_workers = cpu_count()*2
print(f'Process Pool Size: {mp_workers}')

# Pandas IndexSlice
idx = pd.IndexSlice

output_dir = mkdir_p('/data/powt/nbm')
nbm_dir = mkdir_p('/data/powt/nbm_grib2')

aws_bucket_nbm = 'noaa-nbm-grib2-pds'
nbm_set = 'core'
nbm_area = 'co'
query_vars = ['TMP', 'PTYPE', 'SNOWLVL', 'RH'] #'APTMP',

lead_days_selection = 1
fhr = int(lead_days_selection) * 24

# Tolerance +/- top of the hour obs (minutes)
tolerance = 30 #min:0, max:90, step:15
tolerance = pd.Timedelta(f"{tolerance}min")

# For Which Region?
region_selection = "WR" #"WR", "SR", "CR", "ER", "CONUS", "CWA", "RFC"

# If CWA/RFC selected, which one? (i.e. "SLC" for Salt Lake City, "CBRFC" for Colorado Basin)
cwa_selection = None

#network_selection = "NWS+RAWS", "NWS+RAWS+HADS", "NWS", "RAWS", "HADS", "SNOTEL", "ALL"
network_selection = 'NWS'
element = "powt"

ob_hours = {
    'powt':['0000']}

# Define mappings for precipitation type (Code Table 4.201)
ptype_lookup = {
    1: "Rain",
    3: "Freezing rain",
    5: "Snow",
    6: "Snow",
    7: "Snow",
    8: "Ice pellets",
}

# Define general variable lookups
variable_lookup = {
    "Apparent temperature": "Apparent Temperature",
    "Temperature": "Surface Temperature",
    "2 metre temperature": "2 Meter Temperature",
    "236": "Snow Level",  # Map 236 to "Snow Level"
    "2 metre relative humidity": "Relative Humidity"
}

# Define column renaming mappings
column_rename_mapping = {
    'Precipitation Type (Snow) (Surface)': 'PSN',
    'Precipitation Type (Rain) (Surface)': 'PRA',
    'Precipitation Type (Ice pellets) (Surface)': 'PPL',
    'Precipitation Type (Freezing rain) (Surface)': 'PZR',
    'Snow Level (0 m above sea level)': 'snowlvl',
    'Surface Temperature (Surface)': 'tsfc',
    '2 Meter Temperature (2 m)': 't2m',
    'Apparent Temperature (2 m)': 'tapp',
    'Relative Humidity (2 m)':'FXRH'
}

def cwa_list(input_region):

    input_region = input_region.upper()

    region_dict ={
        "WR":["BYZ", "BOI", "LKN", "EKA", "FGZ", "GGW", "TFX", "VEF", "LOX", "MFR",
            "MSO", "PDT", "PSR", "PIH", "PQR", "REV", "STO", "SLC", "SGX", "MTR",
            "HNX", "SEW", "OTX", "TWC"],

        "CR":["ABR", "BIS", "CYS", "LOT", "DVN", "BOU", "DMX", "DTX", "DDC", "DLH",
            "FGF", "GLD", "GJT", "GRR", "GRB", "GID", "IND", "JKL", "EAX", "ARX",
            "ILX", "LMK", "MQT", "MKX", "MPX", "LBF", "APX", "IWX", "OAX", "PAH",
            "PUB", "UNR", "RIW", "FSD", "SGF", "LSX", "TOP", "ICT"],

        "ER":["ALY", "LWX", "BGM", "BOX", "BUF", "BTV", "CAR", "CTP", "RLX", "CHS",
            "ILN", "CLE", "CAE", "GSP", "MHX", "OKX", "PHI", "PBZ", "GYX", "RAH",
            "RNK", "AKQ", "ILM"],

        "SR":["ABQ", "AMA", "FFC", "EWX", "BMX", "BRO", "CRP", "EPZ", "FWD", "HGX",
            "HUN", "JAN", "JAX", "KEY", "MRX", "LCH", "LZK", "LUB", "MLB", "MEG",
            "MAF", "MFL", "MOB", "MRX", "OHX", "LIX", "OUN", "SJT", "SHV", "TAE",
            "TBW", "TSA"]}
    
def cwa_list_rfc(input_rfc):

    metadata_api = 'https://api.synopticdata.com/v2/stations/metadata?'

    network_query = (f"&network={synoptic_networks[network_selection]}"
                    if synoptic_networks[network_selection] is not None else '')

    # Assemble the API query
    api_query = (f"{metadata_api}&token={user_token}" + network_query +
                f"&complete=1&sensorvars=1,obrange=20230118") #hardcoded for NBM4.1+

    # Print the API query to output
    # print(api_query)

    # Get the data from the API
    response = requests.get(api_query)
    metadata = pd.DataFrame(response.json()['STATION'])

    # Remove NaNs and index by network, station ID
    metadata = metadata[metadata['MNET_SHORTNAME'].notna()]
    metadata = metadata.set_index(['MNET_SHORTNAME', 'STID'])

    metadata['LATITUDE'] = metadata['LATITUDE'].astype(float)
    metadata['LONGITUDE'] = metadata['LONGITUDE'].astype(float)
    metadata['ELEVATION'] = metadata['ELEVATION'].astype(float)

    metadata = metadata[metadata['LATITUDE'] >= 31]
    metadata = metadata[metadata['LONGITUDE'] <= -103.00]
    metadata = metadata[metadata['STATUS'] == 'ACTIVE']

    geometry = gpd.points_from_xy(metadata.LONGITUDE, metadata.LATITUDE)
    metadata = gpd.GeoDataFrame(metadata, geometry=geometry)

    req = requests.get(
        'https://www.weather.gov/source/gis/Shapefiles/Misc/rf05mr24.zip',

    allow_redirects=True)
    open('rf05mr24.zip', 'wb').write(req.content)

    with zipfile.ZipFile('rf05mr24.zip', 'r') as zip_ref:
        zip_ref.extractall()

    rfc_shp = gpd.read_file('rf05mr24.shp').set_index('BASIN_ID')

    metadata = metadata[metadata.geometry.within(rfc_shp.geometry.loc[input_rfc])]

    rfc_site_list = metadata.index.get_level_values(1).unique()
    rfc_cwa_list = metadata['CWA'].unique()

    return metadata

def create_init_times(start_date, end_date, frequency):
    """
    Creates a list of datetime objects representing initialization times.

    Args:
        start_date: The start date (inclusive).
        end_date: The end date (inclusive).

    Returns:
        A list of datetime objects.
    """
    init_times = []
    current_date = start_date
    while current_date <= end_date:
        init_times.append(current_date + timedelta(hours=0))  # 00:00
        #init_times.append(current_date + timedelta(hours=12)) # 12:00
        #current_date += timedelta(days=1)
        current_date += timedelta(hours=frequency)  # Increment by the specified frequency

    return init_times

class NBMGribFetcher:
    def __init__(self, aws_bucket, element, nbm_set, nbm_area, query_vars, save_dir):
        self.aws_bucket = aws_bucket
        self.element = element
        self.nbm_set = nbm_set
        self.nbm_area = nbm_area
        self.query_vars = query_vars
        self.save_dir = save_dir
        self.produced_files = []

    def get_nbm_grib_aws(self, aws_request):
        yyyymmdd = datetime.strftime(aws_request['init_time'], '%Y%m%d')
        hh = datetime.strftime(aws_request['init_time'], '%H')

        os.makedirs(self.save_dir, exist_ok=True)

        output_file = (self.save_dir +
                       f"/{yyyymmdd}.t{hh}z.fhr{aws_request['fhr']:03d}.{aws_request['element']}.grib2")

        if os.path.isfile(output_file):
            self.produced_files.append(output_file)
            return output_file

        client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

        bucket_dir = f'blend.{yyyymmdd}/{hh}/{self.nbm_set}/'
        grib_file = f'{bucket_dir}blend.t{hh}z.{self.nbm_set}.f{aws_request["fhr"]:03d}.{self.nbm_area}.grib2'
        index_file = f'{grib_file}.idx'

        try:
            print(self.aws_bucket, index_file)

            index_data_raw = client.get_object(
                Bucket=self.aws_bucket, Key=index_file)['Body'].read().decode().split('\n')

        except Exception as e:
            print(f"Error fetching index file: {e}")
            client.close()
            return

        cols = ['num', 'byte', 'date', 'var', 'level', 'forecast', 'fthresh', 'ftype']
        index_data = []
        for item in index_data_raw:
            item_split = item.split(':')
            if len(item_split) >= 8:
                index_data.append(item_split[:8])
            elif len(item_split) == 7:  # Handle cases where columns are missing
                item_split.append('')
                index_data.append(item_split[:8])

        index_data = pd.DataFrame(index_data, columns=cols)

        index_data = index_data[index_data['num'] != '']
        index_data['num'] = index_data['num'].astype(int)
        index_data = index_data.set_index('num')
        index_data.loc[index_data.shape[0] + 1] = [''] * index_data.shape[1]

        index_subset = index_data[index_data['var'].isin(self.query_vars)]
        index_subset = index_subset[~index_subset['fthresh'].str.contains(
            '% level|std dev', na=False)]

        for i in index_subset.index:
            if (int(i) + 1) in index_data.index:
                index_subset.at[i, 'byte'] = (
                    index_data.at[i, 'byte'],
                    index_data.at[int(i) + 1, 'byte'])
            else:
                index_subset.at[i, 'byte'] = (
                    index_data.at[i, 'byte'], '')

        # Fetch only the selected variables
        with open(output_file, 'wb') as wfp:
            for index, item in index_subset.iterrows():
                byte_range = f"bytes={item['byte'][0]}-{item['byte'][1]}"
                output_bytes = client.get_object(
                    Bucket=self.aws_bucket, Key=grib_file, Range=byte_range)
                for chunk in output_bytes['Body'].iter_chunks(chunk_size=4096):
                    wfp.write(chunk)

        client.close()
        self.produced_files.append(output_file)
        return output_file

    def fetch_for_init_times(self, init_times, fhr):
        with ThreadPoolExecutor(max_workers=mp_workers) as executor:
            futures = [executor.submit(self.get_nbm_grib_aws, {
                'init_time': init_time,
                'fhr': fhr,
                'element': self.element
            }) for init_time in init_times]

            for future in as_completed(futures):
                result = future.result()
                print(f"Downloaded: {result}")

        return self.produced_files
    
def relabel_grib(file_path):
    grbs = pygrib.open(file_path)
    relabeled = []

    for grb in grbs:
        # Try to use grb.name first
        name = grb.name

        # If grb.name is unknown or doesn't match, fall back to message string
        if name not in variable_lookup:
            try:
                # Extract name from the message string
                message_parts = grb.__str__().split(":")
                if len(message_parts) > 1:
                    name = message_parts[1].strip()  # Use the second part (after the first ':')
                else:
                    name = "Unknown Variable"
            except Exception:
                name = "Unknown Variable"

        # Relabel known variables
        if name in variable_lookup:
            label = variable_lookup[name]
        elif name == "Precipitation type":
            # Handle Precipitation Type using lowerLimit
            try:
                lower_limit = grb.lowerLimit  # Extract lowerLimit metadata
                label = f"Precipitation Type ({ptype_lookup.get(lower_limit, 'Unknown')})"
            except AttributeError:
                label = "Precipitation Type (Unknown)"
        else:
            label = f"Unknown Variable ({name})"

        # Add level information if applicable
        level = grb.level
        level_type = grb.typeOfLevel
        if level_type == "heightAboveGround":
            label += f" ({level} m)"
        elif level_type == "heightAboveSea":
            label += f" ({level} m above sea level)"
        elif level_type == "surface":
            label += " (Surface)"

        relabeled.append(label)

    grbs.close()
    return relabeled

def ll_to_index(lat, lon, datalats, datalons):
    """
    Find the nearest grid point index for the given latitude and longitude.
    """
    dist = np.sqrt((datalats - lat)**2 + (datalons - lon)**2)
    return np.unravel_index(np.argmin(dist), datalats.shape)

def relabel_grib_message(grb):
    """
    Relabel a GRIB message to create a human-readable column name.
    """
    # Try to use grb.name first
    name = grb.name

    if name not in variable_lookup:
        try:
            message_parts = grb.__str__().split(":")
            if len(message_parts) > 1:
                name = message_parts[1].strip()
            else:
                name = "Unknown Variable"
        except Exception:
            name = "Unknown Variable"

    if name in variable_lookup:
        label = variable_lookup[name]
    elif name == "Precipitation type":
        try:
            lower_limit = grb.lowerLimit
            label = f"Precipitation Type ({ptype_lookup.get(lower_limit, 'Unknown')})"
        except AttributeError:
            label = "Precipitation Type (Unknown)"
    else:
        label = f"Unknown Variable ({name})"

    level = grb.level
    level_type = grb.typeOfLevel
    if level_type == "heightAboveGround":
        label += f" ({level} m)"
    elif level_type == "heightAboveSea":
        label += f" ({level} m above sea level)"
    elif level_type == "surface":
        label += " (Surface)"

    return label

def convert_kelvin_to_fahrenheit(df, columns):
    """
    Convert the specified columns from Kelvin to Fahrenheit.
    """
    for column in columns:
        if column in df.columns:
            print(f"Converting column {column} from Kelvin to Fahrenheit.")
            df[column] = (df[column] - 273.15) * 9 / 5 + 32
    return df

def calculate_wet_bulb_temperature(T, RH):
    """
    Calculate the wet bulb temperature using the empirical formula.
    T: Temperature in Fahrenheit
    RH: Relative Humidity in percentage (0-100)
    Returns: Wet bulb temperature in Fahrenheit or None if inputs are invalid
    """
    # Check for None or NaN values
    if T is None or RH is None or np.isnan(T) or np.isnan(RH):
        return None

    # Convert temperature from Fahrenheit to Celsius
    T_celsius = (T - 32) * 5.0 / 9.0

    # Calculate wet bulb temperature in Celsius
    T_w_celsius = (
        T_celsius * np.arctan(0.151977 * np.sqrt(RH + 8.313659)) +
        np.arctan(T_celsius + RH) -
        np.arctan(RH - 1.676331) +
        0.00391838 * (RH ** (3/2)) * np.arctan(0.023101 * RH) -
        4.686035
    )

    # Convert wet bulb temperature back to Fahrenheit
    T_w_fahrenheit = T_w_celsius * 9.0 / 5.0 + 32
    return round(T_w_fahrenheit, 2)

def extract_nbm_index(index, nbm_data):

    y, x = index
    return nbm_data[y, x]

def grib_indexer(df, grib_files):
    """
    Process GRIB files dynamically to extract relevant grid point values for the DataFrame.
    """
    for grib_file in grib_files:
        if os.path.isfile(grib_file):
            print(f"Generating index from GRIB file: {grib_file}")
            nbm = pygrib.open(grib_file)

            # If not yet indexed, create grib_index for matching grid points
            if 'grib_index' not in df.columns:
                # print("Creating grib_index for latitude/longitude points.")
                nbmlats, nbmlons = nbm.message(1).latlons()
                # print(f"GRIB file lat/lon grid shape: {nbmlats.shape}")

                df_indexed = df.reset_index()[['stid', 'latitude', 'longitude']].drop_duplicates()

                ll_to_index_mapped = partial(ll_to_index, datalats=nbmlats, datalons=nbmlons)

                df_indexed['grib_index'] = df_indexed.swifter.apply(
                    lambda x: ll_to_index_mapped(x.latitude, x.longitude), axis=1)

                # Extract the grid lat/lon for validation
                extract_nbm_lats_mapped = partial(extract_nbm_index, nbm_data=nbmlats)
                extract_nbm_lons_mapped = partial(extract_nbm_index, nbm_data=nbmlons)

                df_indexed['grib_lat'] = df_indexed['grib_index'].apply(extract_nbm_lats_mapped)
                df_indexed['grib_lon'] = df_indexed['grib_index'].apply(extract_nbm_lons_mapped)

                df_indexed.set_index('stid', inplace=True)

                return df_indexed
        else:
            print(f"GRIB file not found: {grib_file}")

    return df

def process_grib_files(df, grib_file):

    with pygrib.open(grib_file) as nbm:
        print(f"Processing: {grib_file}")

        for idx, msg in enumerate(nbm):
            name = relabel_grib_message(msg)

            if name not in df.columns:
                df[name] = np.nan

            extract_nbm_index_mapped = partial(extract_nbm_index, nbm_data=msg.values)

            valid_date = msg.validDate
            if valid_date in df.index.get_level_values('timestamp'):
                df.loc[valid_date, name] = df.loc[valid_date]['grib_index'].apply(
                    extract_nbm_index_mapped
                ).values
    
    gc.collect()
    return df

if __name__ == "__main__":

    start_date = "2024-10-01" 
    end_date = "2025-04-28" 

    # Convert user input to datetime objects
    start_date, end_date = [datetime.strptime(date+' 0000', '%Y-%m-%d %H%M')
        for date in [start_date, end_date]]

    # Matching Observation File
    cwa_reg_sel = region_selection if region_selection != 'CWA' else cwa_selection
    obs_file = f"/data/powt/obs/{cwa_reg_sel}_{start_date.strftime('%Y%m%d%H')}_{end_date.strftime('%Y%m%d%H')}.powt-obs.csv"

    if not os.path.isfile(obs_file):
        print(f"Observation file not found: {obs_file}")
        exit(1)
    else:
        print(f"Observation file found: {obs_file}")

    obs = pd.read_csv(obs_file)

    obs['timestamp'] = obs['timestamp'].astype('datetime64[ns]')
    obs['ob_timestamp'] = obs['timestamp'].copy()
    obs['timestamp'] = obs['timestamp'].dt.round('1h')

    obs.set_index(['timestamp', 'stid'], inplace=True)
    obs.sort_index(inplace=True)

    columns_order = ["ob_timestamp"] + [col for col in obs.columns if col != "ob_timestamp"]
    obs = obs[columns_order]

    init_times = create_init_times(start_date, end_date, frequency=1)
    # valid_times = [t+timedelta(hours=fhr) for t in init_times]

    fetcher = NBMGribFetcher(aws_bucket_nbm, element, nbm_set, nbm_area, query_vars, nbm_dir)
    nbm_files = fetcher.fetch_for_init_times(init_times, fhr)
    
    nbm_files = sorted(nbm_files)#[:10]  # Limit

    print(f"{len(nbm_files)} NBM GRIB files to process.")

    grib_index = grib_indexer(obs, nbm_files)
    obs = obs.join(grib_index[['grib_index', 'grib_lat', 'grib_lon']], on='stid')

    for nbm_file in nbm_files:
        obs = process_grib_files(obs, nbm_file)

    processed_df = obs

    # processed_df = [process_grib_files(obs, f) for f in nbm_files]
    # processed_df = pd.concat(processed_df, axis=0)

    # Rename columns based on the mapping
    processed_df.rename(columns=column_rename_mapping, inplace=True)

    # Convert specified columns from Kelvin to Fahrenheit
    columns_to_convert = ['t2m', 'tsfc', 'tapp']
    processed_df = convert_kelvin_to_fahrenheit(processed_df, columns_to_convert)

    processed_df.rename(columns={'observed_air_temp':'T', 'observed_wet_bulb_temp':'Tw', 'relative_humidity':'RH', 't2m':'FXT'}, inplace=True)
    processed_df['snowlvl'] = processed_df['snowlvl'] * 3.28084 # m to ft

    subset = processed_df[['T', 'FXT', 'RH', 'FXRH', 'Tw', 'elevation', 'snowlvl', 'RA', 'SN', 'ZR', 'PL', 'UP', 'PRA', 'PZR', 'PSN', 'PPL']]

    # Must drop NA before applying the Tw function
    subset.dropna(how='any', inplace=True)

    # Apply the Tw function to the dataframe
    subset['FXTw'] = subset.apply(lambda row: calculate_wet_bulb_temperature(row['FXT'], row['FXRH']), axis=1)
    subset = subset[['T', 'FXT', 'RH', 'FXRH', 'Tw', 'FXTw', 'elevation', 'snowlvl', 'RA', 'SN', 'ZR', 'PL', 'UP', 'PRA', 'PZR', 'PSN', 'PPL']]

    subset = subset.apply(lambda col: col.round(2) if np.issubdtype(col.dtype, np.number) else col)

    subset_file = f"{output_dir}/{cwa_reg_sel}_{start_date.strftime('%Y%m%d%H')}_{end_date.strftime('%Y%m%d%H')}.powt-nbm-obs.csv"
    subset.to_csv(subset_file, index=True)
    print(f"NBM/OBS Subset DataFrame saved to {subset_file}")