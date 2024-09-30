import zipfile
from datetime import datetime

import requests
import numpy as np
import pandas as pd

# Plotting libraries
import streamlit as st
import geopandas as gpd
import pydeck as pdk

from src.inference import (
    load_batch_of_features_from_store,
    load_model_from_registry,
    get_model_predictions
)

from src.paths import DATA_DIR
from src.plot import plot_one_sample

st.set_page_config(layout='wide')

# title
current_date = pd.to_datetime(datetime.utcnow()).floor('H')
st.title(f"Taxi demand prediction")
st.header(f'{current_date}')

# Progress bar on the left
progress_bar = st.sidebar.header('Work in progress...')
progress_bar = st.sidebar.progress(0)
N_STEPS = 7

def load_shape_data_file():
    URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
    response = requests.get(URL)
    path = DATA_DIR / f'taxi_zones.zip'
    if response.status_code == 200:
        with open(path, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f'{URL} is not available')
    
    # Unzip file
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR / 'taxi_zones')
        
    # load and return shape file
    return gpd.read_file(DATA_DIR / 'taxi_zones/taxi_zones.shp').to_crs('epsg:4326')

with st.spinner(text="Downloading shape file to plot taxi zones"):
    geo_df = load_shape_data_file()
    st.sidebar.write('Shape file was downloaded')
    progress_bar.progress(1/N_STEPS)
    
