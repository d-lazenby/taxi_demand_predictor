import os
from dotenv import load_dotenv

from src.paths import PARENT_DIR

load_dotenv(PARENT_DIR / '.env')

HOPSWORKS_PROJECT_NAME = 'taxi_demand_DL'
try: 
    HOPSWORKS_API_KEY = os.getenv('HOPSWORKS_API_KEY')
except:
    raise Exception('Create a .env file on the project root with the HOPSWORKS_API_KEY')

FEATURE_GROUP_NAME = 'time_series_hourly_feature_group'
FEATURE_GROUP_VERSION = 1