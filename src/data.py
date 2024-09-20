from pathlib import Path
import requests

from typing import Optional, List

import pandas as pd
import numpy as np

from src.paths import RAW_DATA_DIR, TRANSFORMED_DATA_DIR

def download_one_file_of_raw_data(
    year: int, 
    month: int
    ) -> Path:
    """
    Downloads Parquet file with historical taxi rides for the given year and month
    """
    URL = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
    response = requests.get(url=URL)
    
    if response.status_code == 200:
        path = RAW_DATA_DIR / f"rides_{year}-{month:02d}.parquet"
        with open(path, "wb") as f:
            f.write(response.content)
        return path
    else:
        raise Exception(f"{URL} is not available.")
    
def validate_raw_data(
    rides: pd.DataFrame,
    year: int,
    month: int, 
) -> pd.DataFrame:
    """
    Removes rows with pickup_datetime outside of year, month
    """
    
    this_month_start = f"{year}-{month:02d}-01"
    next_month_start = f"{year}-{month+1:02d}-01" if month < 12 else f"{year}-01-01"
    
    mask = (rides['pickup_datetime'] >= this_month_start) & (rides['pickup_datetime'] < next_month_start)

    return rides[mask]
    
def load_raw_data(
    year: int, 
    months: Optional[List[int]] = None  
) -> pd.DataFrame:
    
    rides = pd.DataFrame()
    
    if months is None:
        # Download data for entire year
        months = list(range(1, 13))
    elif isinstance(months, int):
        # Download data for just month specified
        months = [months]
        
    for month in months:
        
        local_file = RAW_DATA_DIR / f'rides_{year}-{month:02d}.parquet'
        if not local_file.exists():
            try:
                # Download the file 
                print(f"Downloading file {year}-{month:02d}")
                download_one_file_of_raw_data(year, month)
            except:
                print(f"{year}-{month:02d} file is not available")
                continue
        else:
            print(f"File {year}-{month:02d} was already in local storage")
                
        # Load file into pandas df
        rides_one_month = pd.read_parquet(local_file)
        
        # Rename columns
        rides_one_month = rides_one_month[['tpep_pickup_datetime', 'PULocationID']]
        rides_one_month.rename(columns={
            'tpep_pickup_datetime': 'pickup_datetime',
            'PULocationID': 'pickup_location_id',
        }, inplace=True)
        
        # Validate file
        rides_one_month = validate_raw_data(rides_one_month, year, month)
        
        # Append to existing data
        rides = pd.concat([rides, rides_one_month])
        
    # Keep only time and location columns
    rides = rides[['pickup_datetime', 'pickup_location_id']]
    
    return rides

from tqdm import tqdm

def add_missing_slots(agg_rides: pd.DataFrame) -> pd.DataFrame:
    
    location_ids = agg_rides['pickup_location_id'].unique()
    # Makde a full range of datetime indexes.
    full_range = pd.date_range(
        agg_rides['pickup_hour'].min(), agg_rides['pickup_hour'].max(), freq='h'
    )
    # For storing the modified time series'
    output = pd.DataFrame()
    
    for location_id in tqdm(location_ids):
        
        # Temporary df holding rides with current location id
        tmp = agg_rides.loc[agg_rides['pickup_location_id'] == location_id, ['pickup_hour', 'rides']]
        
        # Quick way of adding missing dates and filling with 0 in a Series; from https://stackoverflow.com/a/19324591
        tmp.set_index('pickup_hour', inplace=True)
        tmp.index = pd.DatetimeIndex(tmp.index)
        tmp = tmp.reindex(full_range, fill_value=0)
        
        # Add back in the location id
        tmp['pickup_location_id'] = location_id
        
        output = pd.concat([output, tmp])
    
    # Move purchase date from index to column
    output = output.reset_index().rename(columns={'index': 'pickup_hour'})
    
    return output

def transform_raw_data_into_ts_data(
    rides: pd.DataFrame
    ) -> pd.DataFrame:
    
    # Sum rides per location and pickup hour
    rides['pickup_hour'] = rides['pickup_datetime'].dt.floor('h')
    agg_rides = rides.groupby(['pickup_hour', 'pickup_location_id']).size().reset_index()
    agg_rides.rename(columns={0: 'rides'}, inplace=True)
    
    # Add rows for (location, hour) pairs with zero rides
    agg_rides_all_slots = add_missing_slots(agg_rides)
    
    return agg_rides_all_slots

def get_cutoff_indices(
    data: pd.DataFrame,
    n_features: int, 
    step_size: int) -> list:
    
    stop_position = len(data) - 1
    
    # start first slice at index 0
    subseq_first_idx = 0
    subseq_mid_idx = n_features
    subseq_last_idx = n_features + 1
    indices = []
    
    while subseq_last_idx <= stop_position:
        indices.append((subseq_first_idx, subseq_mid_idx, subseq_last_idx))
        
        subseq_first_idx += step_size
        subseq_mid_idx += step_size
        subseq_last_idx += step_size
        
    return indices
  
def transform_ts_into_features_and_targets(
    ts_data: pd.DataFrame,
    n_features: int,
    step_size: int  
) -> pd.DataFrame:
    """
    Slices and transposes data from time-series format to (features, target) for training
    """
    assert set(ts_data.columns) == {'pickup_hour', 'rides', 'pickup_location_id'}
    
    location_ids = ts_data['pickup_location_id'].unique()
    features = pd.DataFrame()
    targets = pd.DataFrame()
    
    for location_id in tqdm(location_ids):
        
        # keep only data for this location_id
        ts_data_one_location = ts_data.loc[
            ts_data['pickup_location_id'] == location_id,
            ['pickup_hour', 'rides']
        ]
        
        # pre-compute cutoff indices to split dataframe rows
        indices = get_cutoff_indices(
            ts_data_one_location,
            n_features,
            step_size
        )
        
        # Slice and transpose data into np arrays
        n_examples = len(indices)
        x = np.ndarray(shape=(n_examples, n_features), dtype=np.float32)
        y = np.ndarray(shape=n_examples, dtype=np.float32)
        pickup_hours = []

        for i, idx in enumerate(indices):
            x[i, :] = ts_data_one_location.iloc[idx[0]:idx[1]]['rides'].values
            y[i] = ts_data_one_location.iloc[idx[1]]['rides']
            pickup_hours.append(ts_data_one_location.iloc[idx[1]]['pickup_hour'])
            
        # features np -> pd
        features_one_location = pd.DataFrame(
            x,
            columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(n_features))]
            )
        features_one_location['pickup_hour'] = pickup_hours
        features_one_location['pickup_location_id'] = location_id
        
        # target np -> pd
        targets_one_location = pd.DataFrame(
            y,
            columns=['target_rides_next_hour']
            )
        
        # concatenate results
        features = pd.concat([features, features_one_location])
        targets = pd.concat([targets, targets_one_location])
        
    features.reset_index(inplace=True, drop=True)
    targets.reset_index(inplace=True, drop=True) 
    
    return features, targets['target_rides_next_hour']          
