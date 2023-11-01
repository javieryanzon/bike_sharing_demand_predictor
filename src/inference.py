from datetime import datetime, timedelta

import hopsworks
#from hsfs.feature_store import FeatureStore
import pandas as pd
import numpy as np

import src.config as config
from src.feature_store_api import get_feature_store
#, get_or_create_feature_view
#from src.config import FEATURE_VIEW_METADATA

def get_hopsworks_project() -> hopsworks.project.Project:

    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )

# def get_feature_store() -> FeatureStore:
    
#     project = get_hopsworks_project()
#     return project.get_feature_store()


def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    """"""
    # past_rides_columns = [c for c in features.columns if c.startswith('rides_')]
    predictions = model.predict(features)
    predictions = predictions.round(0)

    results = pd.DataFrame(predictions,
            columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(36))]
        ) #son 36 horas de prediccion
    results['pickup_location_id'] = features['pickup_location_id'].values
    #results['predicted_demand'] = predictions.round(0) #esto estaba antes
    
    return results


def load_batch_of_features_from_store(
    current_date: datetime,    
) -> pd.DataFrame:
    """Fetches the batch of features used by the ML system at `current_date`

    Args:
        current_date (datetime): datetime of the prediction for which we want
        to get the batch of features

    Returns:
        pd.DataFrame: 3 columns:
            - `pickup_hour`
            - `rides`
            - `pickup_location_id`
    """
    n_features = config.N_FEATURES

    feature_store = get_feature_store()

    # read time-series data from the feature store
    fetch_data_to = pd.to_datetime(current_date - timedelta(hours=1), utc=True)
    fetch_data_from = pd.to_datetime(current_date - timedelta(days=28), utc=True)
    print(f'Fetching data from {fetch_data_from} to {fetch_data_to}')
    feature_view = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_NAME,
        version=config.FEATURE_VIEW_VERSION
    )
    ts_data = feature_view.get_batch_data(
        start_time=pd.to_datetime(fetch_data_from - timedelta(days=1), utc=True),
        end_time=pd.to_datetime(fetch_data_to + timedelta(days=1), utc=True)
    )
    
    # Convert to UTC aware datetime
    ts_data['pickup_hour'] = pd.to_datetime(ts_data['pickup_hour'], utc=True)

    # filter data to the time period we are interested in
    ts_data = ts_data[ts_data.pickup_hour.between(fetch_data_from, fetch_data_to)]

    # validate we are not missing data in the feature store
    location_ids = ts_data['pickup_location_id'].unique()
    assert len(ts_data) == n_features*len(location_ids), \
        "Time-series data is not complete. Make sure your feature pipeline is up and runnning."
    
    # sort data by location and time
    ts_data.sort_values(by=['pickup_location_id', 'pickup_hour'], inplace=True)
    # print(f'{ts_data=}')

    # transpose time-series data as a feature vector, for each `pickup_location_id`
    x = np.ndarray(shape=(len(location_ids), n_features), dtype=np.float32)
    for i, location_id in enumerate(location_ids):
        ts_data_i = ts_data.loc[ts_data.pickup_location_id == location_id, :]
        ts_data_i = ts_data_i.sort_values(by=['pickup_hour'])
        x[i, :] = ts_data_i['rides'].values

    # numpy arrays to Pandas dataframes
    features = pd.DataFrame(
        x,
        columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(n_features))]
    )
    
    features['pickup_hour'] = pd.to_datetime(current_date, utc=True)
    features['pickup_location_id'] = location_ids
    features.sort_values(by=['pickup_location_id'], inplace=True)

    return features
    

def load_model_from_registry():
    
    import joblib
    from pathlib import Path

    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    model = model_registry.get_model(
        name=config.MODEL_NAME,
        version=config.MODEL_VERSION,
    )  
    
    model_dir = model.download()
    model = joblib.load(Path(model_dir)  / 'model.pkl')
       
    return model

def load_predictions_from_store(
        from_pickup_hour: datetime,
        to_pickup_hour: datetime) -> pd.DataFrame:
    """
    Connects to the feature store and retrieves model predictions for all
    `pickup_location_id`s and for the time period from `from_pickup_hour`
    to `to_pickup_hour`

    Args:
        from_pickup_hour (datetime): min datetime (rounded hour) for which we want to get
        predictions

        to_pickup_hour (datetime): max datetime (rounded hour) for which we want to get
        predictions

    Returns:
        pd.DataFrame: 3 columns:
            - `pickup_location_id`
            - `predicted_demand`
            - `pickup_hour`
    """
    from src.feature_store_api import get_feature_store
    import src.config as config

    feature_store = get_feature_store()

    predictiong_fg = feature_store.get_feature_group(
        name=config.FEATURE_GROUP_MODEL_PREDICTIONS,
        version=1,
    )

    try:
        # create feature view as it does not exist yet
        feature_store.create_feature_view(
            name=config.FEATURE_VIEW_MODEL_PREDICTIONS,
            version=1,
            query=predictiong_fg.select_all()
        )
    except:
        print(f'Feature view {config.FEATURE_VIEW_MODEL_PREDICTIONS} \
              already existed. Skipped creation.')
        
    predictions_fv = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_MODEL_PREDICTIONS,
        version=1
    )
    
    print(f'Fetching predictions for `pickup_hours` between {from_pickup_hour}  and {to_pickup_hour}')
    predictions = predictions_fv.get_batch_data(
        start_time=from_pickup_hour - timedelta(days=1),
        end_time=to_pickup_hour + timedelta(days=1)
    )
    
    # make sure datetimes are UTC aware
    predictions['pickup_hour'] = pd.to_datetime(predictions['pickup_hour'], utc=True)
    from_pickup_hour = pd.to_datetime(from_pickup_hour, utc=True)
    to_pickup_hour = pd.to_datetime(to_pickup_hour, utc=True)

    predictions = predictions[predictions.pickup_hour.between(
        from_pickup_hour, to_pickup_hour)]

    # sort by `pick_up_hour` and `pickup_location_id`
    predictions.sort_values(by=['pickup_hour', 'pickup_location_id'], inplace=True)

    return predictions