import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.multioutput import MultiOutputRegressor
from src.paths import RAW_DATA_DIR

import lightgbm as lgb

def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    """
    Adds one column with the average rides from
    - 7 days ago
    - 14 days ago
    - 21 days ago
    - 28 days ago
    """
    X['average_rides_last_4_weeks'] = 0.25*(
        X[f'rides_previous_{7*24}_hour'] + \
        X[f'rides_previous_{2*7*24}_hour'] + \
        X[f'rides_previous_{3*7*24}_hour'] + \
        X[f'rides_previous_{4*7*24}_hour']
    )
    return X

def latitude_and_longitude(X: pd.DataFrame) -> pd.DataFrame:
    """
    Adds two columns with the latitude and longitude from pickup_location_id
    
    """
    raw_data_rides = pd.read_parquet(RAW_DATA_DIR / 'rides_2022.parquet')

    #Nos quedamos sólo con las columnas que nos interesan y las renombramos
    raw_data_rides = raw_data_rides[['id_estacion_origen', 'lat_estacion_origen', 'long_estacion_origen']]
    raw_data_rides['id_estacion_origen'] = raw_data_rides['id_estacion_origen'].str.replace('BAEcobici', '').astype(int)
    raw_data_rides = raw_data_rides.drop_duplicates().reset_index(drop=True)
    raw_data_rides.rename(columns={
    'id_estacion_origen': 'pickup_location_id',
    'lat_estacion_origen': 'latitude',
    'long_estacion_origen': 'longitude'
    }, inplace=True)

    # Combinar la información de latitud y longitud en X
    X = X.merge(raw_data_rides, on='pickup_location_id', how='left')

    # Eliminar la columna 'pickup_location_id'
    #X.drop('pickup_location_id', axis=1, inplace=True)

    return X


class TemporalFeaturesEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn data transformation that adds 2 columns
    - hour
    - day_of_week
    and removes the `pickup_hour` datetime column.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        X_ = X.copy()
        
        # Generate numeric columns from datetime
        X_["hour"] = X_['pickup_hour'].dt.hour
        X_["day_of_week"] = X_['pickup_hour'].dt.dayofweek
        
        return X_.drop(columns=['pickup_hour'])

def get_pipeline(**hyperparams) -> Pipeline:

    # sklearn transform
    add_feature_average_rides_last_4_weeks = FunctionTransformer(
        average_rides_last_4_weeks, validate=False)
    
    # sklearn transform
    add_feature_latitude_and_longitude = FunctionTransformer(
    latitude_and_longitude, validate=False)

    # sklearn transform
    add_temporal_features = TemporalFeaturesEngineer()

    # sklearn pipeline
    return make_pipeline(
        add_feature_average_rides_last_4_weeks,
        add_feature_latitude_and_longitude,
        add_temporal_features,
        MultiOutputRegressor(lgb.LGBMRegressor(**hyperparams, force_col_wise=True))
    )