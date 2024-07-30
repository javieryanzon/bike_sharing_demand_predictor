import os
from dotenv import load_dotenv

from src.paths import PARENT_DIR

# load key-value pairs from .env file located in the parent directory
load_dotenv(PARENT_DIR / '.env')

HOPSWORKS_PROJECT_NAME = 'bike_sharing_demand'
try:
    HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
except:
    raise Exception('Create an .env file on the project root with the HOPSWORKS_API_KEY')

FEATURE_GROUP_NAME = 'time_series_hourly_feature_group'
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = 'time_series_hourly_feature_view'
FEATURE_VIEW_VERSION = 1
MODEL_NAME = "bike_demand_predictor_next_hour"
MODEL_VERSION = 1

#Agrego esto para que se consulte al feature store la latitud y longitud
FEATURE_GROUP_LAT_LONG = 'latitud_y_longitud_group' 
FEATURE_VIEW_LAT_LONG = 'latitud_y_longitud_view'
#FEATURE_VIEW_LAT_LONG_VERSION = 1


# added for monitoring purposes
FEATURE_GROUP_MODEL_PREDICTIONS = 'model_predictions_feature_group_'
FEATURE_VIEW_MODEL_PREDICTIONS = 'model_predictions_feature_view_'
FEATURE_VIEW_MONITORING = 'predictions_vs_actuals_for_monitoring_feature_view'

# number of historical values our model needs to generate predictions
N_FEATURES = 24 * 28

# maximum Mean Absolute Error we allow our production model to have
MAX_MAE = 4.0