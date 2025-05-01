# Standard library imports
import os
import time
import json
import zipfile
import warnings
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
from functools import partial
from glob import glob
from typing import List, Optional
from multiprocessing import Pool, cpu_count, set_start_method, get_context

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

# Concurrent and parallel processing imports
# from concurrent.futures import ThreadPoolExecutor, as_completed

# Set global configurations
warnings.filterwarnings('ignore')

# Pandas IndexSlice
idx = pd.IndexSlice

output_dir = '/data/powt/'

user_token = "a2386b75ecbc4c2784db1270695dde73" 

start_date = "2024-10-01" 
end_date = "2025-04-28" 

lead_days_selection = 1

# For Which Region?
region_selection = "WR" #"WR", "SR", "CR", "ER", "CONUS", "CWA", "RFC"

# If CWA/RFC selected, which one? (i.e. "SLC" for Salt Lake City, "CBRFC" for Colorado Basin)
cwa_selection = None

#network_selection = "NWS+RAWS", "NWS+RAWS+HADS", "NWS", "RAWS", "HADS", "SNOTEL", "ALL"
network_selection = 'NWS'
element = "powt"

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# GLOBAL VARIABLES AND GENERAL CONFIG                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Multiprocess settings
# process_pool_size = 20 #cpu_count()*16
# print(f'Process Pool Size: {process_pool_size}')

# Backend APIs
metadata_api = "https://api.synopticdata.com/v2/stations/metadata?"
qc_api = "https://api.synopticdata.com/v2/stations/qcsegments?"

# Data Query APIs
timeseries_api = "https://api.synopticdata.com/v2/stations/timeseries?"
statistics_api = "https://api.synopticlabs.org/v2/stations/statistics?"
precipitation_api = "https://api.synopticdata.com/v2/stations/precipitation?"

# Assign API to element name
synoptic_apis = {
    'qpf':precipitation_api,
    'maxt':statistics_api,
    'mint':statistics_api,
    'powt':timeseries_api}

synoptic_networks = {"NWS+RAWS+HADS":"1,2,106",
                     "NWS+RAWS":"1,2",
                     "NWS":"1",
                     "RAWS": "2",
                     "HADS": "106",
                     "SNOTEL":"25",
                     "ALL":None}
                    #  "CUSTOM": "&network="+network_input,
                    #  "LIST": "&stid="+network_input}

# Assign synoptic variable to element name
synoptic_vars = {'powt':'weather_condition,weather_summary,' \
        'weather_cond_code_synop,past_weather_code,weather_cond_code,' \
        'air_temp,relative_humidity'}

synoptic_vars_out = {
    'powt':''}

# Assign stat type to element name
stat_type = {
    'powt':None}

ob_hours = {
    'powt':['0000']}

# Convert user input to datetime objects
start_date, end_date = [datetime.strptime(date+' 0000', '%Y-%m-%d %H%M')
    for date in [start_date, end_date]]

# Build synoptic arg dict
synoptic_api_args = {
    'ob_stat':stat_type[element],
    'api':synoptic_apis[element],
    'element':element,
    'interval':24, #24h to poll daily powt data
    #interval_selection if element == 'qpf' else False,
    'region':region_selection,
    'network_query':synoptic_networks[network_selection], # add config feature later
    'vars_query':None if element == 'qpf'
        else f'{synoptic_vars[element]}',}

class PeriodOfRecord(BaseModel):
    start: str
    end: str

class Providers(BaseModel):
    name: str
    url: str

class Units(BaseModel):
    position: str
    elevation: str

class SensorVariables(BaseModel):
    air_temp: dict = Field(default_factory=dict)
    relative_humidity: dict = Field(default_factory=dict)
    weather_summary: dict = Field(default_factory=dict)
    weather_condition: dict = Field(default_factory=dict)
    weather_cond_code: dict = Field(default_factory=dict)
    past_weather_code: dict = Field(default_factory=dict)
    weather_cond_code_synop: dict = Field(default_factory=dict)

class Observations(BaseModel):
    date_time: List[str]

    air_temp_set_1: List[Optional[float]] = Field(default_factory=list)
    air_temp_set_1d: List[Optional[float]] = Field(default_factory=list)

    relative_humidity_set_1: List[Optional[float]] = Field(default_factory=list)
    relative_humidity_set_1d: List[Optional[float]] = Field(default_factory=list)

    weather_summary_set_1: List[Optional[str]] = Field(default_factory=list)
    weather_summary_set_1d: List[Optional[str]] = Field(default_factory=list)

    weather_condition_set_1: List[Optional[str]] = Field(default_factory=list)
    weather_condition_set_1d: List[Optional[str]] = Field(default_factory=list)

    weather_cond_code_set_1: List[Optional[float]] = Field(default_factory=list)
    weather_cond_code_set_1d: List[Optional[float]] = Field(default_factory=list)

    weather_cond_code_synop_set_1: List[Optional[object]] = Field(default_factory=list)
    weather_cond_code_synop_set_1d: List[Optional[object]] = Field(default_factory=list)

    past_weather_code_set_1: List[Optional[object]] = Field(default_factory=list)
    past_weather_code_set_1d: List[Optional[object]] = Field(default_factory=list)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        strict = False

class Station(BaseModel):
    ID: str
    STID: str
    NAME: str
    ELEVATION: float
    LATITUDE: float
    LONGITUDE: float
    STATUS: str
    MNET_ID: str
    STATE: str
    TIMEZONE: str
    ELEV_DEM: object
    # NWSZONE: str
    # NWSFIREZONE: str
    # GACC: str
    SHORTNAME: str
    # SGID: str
    COUNTY: str
    # COUNTRY: str
    # WIMS_ID: str
    CWA: str
    PERIOD_OF_RECORD: PeriodOfRecord
    # PROVIDERS: Providers
    UNITS: Units
    SENSOR_VARIABLES: SensorVariables
    OBSERVATIONS: Observations
    QC_FLAGGED: bool
    RESTRICTED: bool

class Data(BaseModel):
    STATION: List[Station]

    def to_dataframe(self) -> pd.DataFrame:
        records = []

        for station in self.STATION:
            for i, date_time in enumerate(station.OBSERVATIONS.date_time):
                record = {
                    'stid': station.STID,
                    'latitude': station.LATITUDE,
                    'longitude': station.LONGITUDE,
                    'elevation': station.ELEVATION,
                    'state': station.STATE,
                    'cwa': station.CWA,
                    'timestamp':date_time,

                    'air_temp_set_1': station.OBSERVATIONS.air_temp_set_1[i] if len(station.OBSERVATIONS.air_temp_set_1) > i else None,
                    'relative_humidity_set_1': station.OBSERVATIONS.relative_humidity_set_1[i] if len(station.OBSERVATIONS.relative_humidity_set_1) > i else None,
                    'weather_summary_set_1': station.OBSERVATIONS.weather_summary_set_1[i] if len(station.OBSERVATIONS.weather_summary_set_1) > i else None,
                    'weather_condition_set_1': station.OBSERVATIONS.weather_condition_set_1[i] if len(station.OBSERVATIONS.weather_condition_set_1) > i else None,
                    'weather_cond_code_set_1': station.OBSERVATIONS.weather_cond_code_set_1[i] if len(station.OBSERVATIONS.weather_cond_code_set_1) > i else None,
                    'past_weather_code_set_1': station.OBSERVATIONS.past_weather_code_set_1[i] if len(station.OBSERVATIONS.past_weather_code_set_1) > i else None,
                    'weather_cond_code_synop_set_1': station.OBSERVATIONS.weather_cond_code_synop_set_1[i] if len(station.OBSERVATIONS.weather_cond_code_synop_set_1) > i else None,

                    'air_temp_set_1d': station.OBSERVATIONS.air_temp_set_1d[i] if len(station.OBSERVATIONS.air_temp_set_1d) > i else None,
                    'relative_humidity_set_1d': station.OBSERVATIONS.relative_humidity_set_1d[i] if len(station.OBSERVATIONS.relative_humidity_set_1d) > i else None,
                    'weather_summary_set_1d': station.OBSERVATIONS.weather_summary_set_1d[i] if len(station.OBSERVATIONS.weather_summary_set_1d) > i else None,
                    'weather_condition_set_1d': station.OBSERVATIONS.weather_condition_set_1d[i] if len(station.OBSERVATIONS.weather_condition_set_1d) > i else None,
                    'weather_cond_code_set_1d': station.OBSERVATIONS.weather_cond_code_set_1d[i] if len(station.OBSERVATIONS.weather_cond_code_set_1d) > i else None,
                    'past_weather_code_set_1d': station.OBSERVATIONS.past_weather_code_set_1d[i] if len(station.OBSERVATIONS.past_weather_code_set_1d) > i else None,
                    'weather_cond_code_synop_set_1d': station.OBSERVATIONS.weather_cond_code_synop_set_1d[i] if len(station.OBSERVATIONS.weather_cond_code_synop_set_1d) > i else None
                }
                records.append(record)

        df = pd.DataFrame(records)
        return df

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# FUNCTIONS AND METHODS (GENERAL)                                             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def mkdir_p(check_dir):
    from pathlib import Path
    check_dir = output_dir + check_dir
    Path(check_dir).mkdir(parents=True, exist_ok=True)
    return check_dir

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

    if input_region == "CONUS":
        return np.hstack([region_dict[region] for region in region_dict.keys()])
    else:
        return region_dict[input_region]

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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# FUNCTIONS AND METHODS (SYNOPTIC API)                                        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def fetch_obs_from_API(valid_datetime, cwa='', output_type='json',
                       use_saved=True, **req):

    start_adjusted = (datetime.strptime(valid_datetime, '%Y%m%d%H%M')
                        - timedelta(hours=req["interval"]))

    end_adjusted = datetime.strptime(valid_datetime, '%Y%m%d%H%M')

    valid = True
    cwa_filename = (region_selection if region_selection != 'CWA'
                    else cwa_selection)

    element_label = req['element'] if req['element'] != 'qpf' else \
                        'qpe' + f'{req["interval"]:02d}'


    output_file = mkdir_p(f'obs_{output_type}/') +\
        f'obs.{element_label}.{req["ob_stat"]}' +\
        f'.{valid_datetime}.{cwa_filename}.{output_type}'

    if os.path.isfile(output_file) & use_saved:
        print(f'Output file exists for:{iter_item}')

        with open(output_file, 'r') as f:
            data = Data.model_validate_json(f.read())

            # Convert to DataFrame
            df = data.to_dataframe()
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)

            return df

    else:
        json_file = mkdir_p('obs_json/') +\
            f'obs.{element_label}.{req["ob_stat"]}' +\
            f'.{valid_datetime}.{cwa_filename}.json'

        # if os.path.isfile(json_file) & use_saved:
        #     # print(f'Polling archived JSON for: {iter_item}')

        #     with open(json_file, 'rb+') as rfp:
        #         response_dataframe = pd.json_normalize(json.load(rfp)['STATION'])

        # else:
        api_query_args = {
            'api_token':f'&token={user_token}',
            'station_query':f'&cwa={cwa}',
            'network_query':(f'&network={req["network_query"]}'
                                if req["network_query"] is not None else ''),

            'start_date_query':f'&start={start_adjusted.strftime("%Y%m%d%H%M")}',
            'end_date_query':f'&end={end_adjusted.strftime("%Y%m%d%H%M")}',

            'vars_query':(f'&pmode=intervals&interval={req["interval"]}'
                            if req["element"] == 'qpf'
                            else f'&vars={req["vars_query"]}'),
            'stats_query':f'&type={req["ob_stat"]}',
            'timezone_query':'&obtimezone=utc',
            'api_extras':'&units=temp|f&complete=True'}
                #'&fields=name,status,latitude,longitude,elevation'

        api_query = req['api'] + ''.join(
            [api_query_args[k] for k in api_query_args.keys()])

        print(f'Polling API for: {iter_item}\n{api_query}')

        status_code, response_count = None, 0
        while (status_code != 200) & (response_count <= 10):
            print(f'{iter_item}, HTTP:{status_code}, #:{response_count}')

            # Don't sleep first try, sleep increasing amount for each retry
            time.sleep(2*response_count)

            response = requests.get(api_query)
            # response.raise_for_status()

            status_code = response.status_code
            response_count += 1

        try:
            response_dataframe = pd.json_normalize(
                response.json()['STATION'])
        except:
            valid = False
        else:
            with open(json_file, 'wb+') as wfp:
                wfp.write(response.content)

    if valid:
        # Example usage
        # with open('sample.json', 'r') as f:
        data = Data.model_validate_json(response.content)

        # Convert to DataFrame
        df = data.to_dataframe()
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)

        return df

    else:
        return None

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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# INPUT-BASED GLOBAL VARIABLES AND CONFIG                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Build an iterable date list from range
iter_date = start_date
valid_date_iterable = []
valid_datetime_iterable = []
forecast_datetime_iterable = []

while iter_date <= end_date:

    valid_date_iterable.append(iter_date.strftime('%Y%m%d'))

    for hour_range in ob_hours[element]:
        end_hour = f'{int(hour_range[-1]):02d}'

        valid_datetime_iterable.append(iter_date.strftime('%Y%m%d') + end_hour)

        forecast_datetime_iterable.append(
                (iter_date-timedelta(days=lead_days_selection)
            ).strftime('%Y%m%d') + end_hour)

    iter_date += timedelta(days=1)

# Assign the fixed kwargs to the function
if region_selection == 'CWA':
    cwa_query = cwa_selection
elif region_selection == 'RFC':
    rfc_metadata = cwa_list_rfc(cwa_selection)
    cwa_query = ','.join([str(cwa) for cwa in rfc_metadata['CWA'].unique()
                if cwa is not None])
else:
    cwa_query = ','.join(cwa_list(region_selection))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# DATA ACQUISITION                                                            #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
multiprocess_function = partial(fetch_obs_from_API,
                                cwa=cwa_query,
                                **synoptic_api_args)

df_list = []
# Multithreaded requests currently not supported by the Synoptic API
for iter_item in valid_datetime_iterable:
    df_list.append(multiprocess_function(iter_item))

print('Data acquisition complete. Beginning concatenation.')

df = pd.concat(df_list).set_index(['timestamp', 'stid']).sort_index().drop_duplicates()

print('Data concatenation complete.')

# Combine similarly named columns
for col in df.columns:
    if col.endswith('_1'):
        new_col_name = col[:-6]  # Remove the last two characters (_1)
        df[new_col_name] = df[col].combine_first(df[col + 'd'])  # Keep non-NaN values

# Drop the original _1 and _1d columns if needed
obs = df.drop(columns=[col for col in df.columns if col.endswith(('_1', '_1d'))])

# Apply the function to the dataframe
obs['wet_bulb_temp'] = obs.apply(
    lambda row: calculate_wet_bulb_temperature(row['air_temp'], row['relative_humidity']), axis=1
)
# Drop rows with NaN values in the 'wet_bulb_temp' column
obs.dropna(subset=['wet_bulb_temp'], inplace=True)

obs.rename(columns={'air_temp':'observed_air_temp', 'wet_bulb_temp':'observed_wet_bulb_temp'}, inplace=True)
obs['weather_condition'] = obs['weather_condition'].astype(str)

# Extract unique weather conditions
raw_wx_conds = [v.split(',') for v in obs['weather_condition'].values if v is not None]
wx_conds = np.hstack([v.split('/') for v in np.unique(np.hstack(raw_wx_conds))])

wx_counts = obs['weather_condition'].str.split(',|/').explode().value_counts()
wx_counts = wx_counts.sort_values(ascending=False)

# RA: Contains "rain" or "drizzle" but not "frz"
obs['RA'] = obs['weather_condition'].str.contains(r'\brain\b|\bdrizzle\b', case=False, na=False) & \
             ~obs['weather_condition'].str.contains(r'\bfrz\b', case=False, na=False)

# SN: Contains "snow", "grains", or "graupel"
obs['SN'] = obs['weather_condition'].str.contains(r'\bsnow\b|\bgrains\b|\bgraupel\b', case=False, na=False)

# ZR: Contains "frz" but not "fog"
obs['ZR'] = obs['weather_condition'].str.contains(r'\bfrz\b', case=False, na=False) & \
             ~obs['weather_condition'].str.contains(r'\bfog\b', case=False, na=False)

# PL: Contains "ice" or "pellets" but not "fog"
obs['PL'] = obs['weather_condition'].str.contains(r'\bice\b|\bpellets\b', case=False, na=False) & \
             ~obs['weather_condition'].str.contains(r'\bfog\b', case=False, na=False)

# UP: Contains "unknown"
obs['UP'] = obs['weather_condition'].str.contains(r'\bunknown\b', case=False, na=False)& \
             ~obs['weather_condition'].str.contains(r'\bfog\b', case=False, na=False)

# Create a dictionary to store combination counts
combination_counts = {}

# Iterate through all possible combinations of conditions
for i in range(1, 5):  # Iterate through combinations of 1 to 4 conditions
    for combo in itertools.combinations(['RA', 'SN', 'ZR', 'PL', 'UP'], i):
        # Create a boolean series indicating if the combination is present
        combo_series = obs[list(combo)].all(axis=1)

        # Count the occurrences of the combination
        combination_counts[','.join(combo)] = combo_series.sum()

# determine which selection to use based on the condition
selection = cwa_selection.replace(',', '-') if region_selection.lower() == "cwa" else region_selection

# generate the filename using the globals
filename = f"{selection}_{start_date.strftime('%Y%m%d%H')}_{end_date.strftime('%Y%m%d%H')}.powt-obs.csv"

# full path
output_path = os.path.join(output_dir, filename)

obs.to_csv(output_path)
print('Saved', output_path)