from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from tqdm import tqdm

import src.config as config
from src.feature_store_api import get_feature_store, get_feature_group
from src.data import get_cutoff_indices_features_and_target

from datetime import datetime, timedelta
import pandas as pd
from src.data import transform_raw_data_into_ts_data
from src.data import transform_ts_data_into_features_and_target
from src.data import transform_ts_data_into_dataset_comparable_with_predictions


def load_predictions_and_actual_values_from_store(
    from_date: datetime,
    to_date: datetime,
) -> pd.DataFrame:
    """Fetches model predictions and actuals values from
    `from_date` to `to_date` from the Feature Store and returns a dataframe

    Args:
        from_date (datetime): min datetime for which we want predictions and
        actual values

        to_date (datetime): max datetime for which we want predictions and
        actual values

    Returns:
        pd.DataFrame: 4 columns
            - `pickup_location_id`
            - `predicted_demand`
            - `pickup_hour`
            - `rides`
    """
    current_date = pd.to_datetime(datetime.utcnow(), utc=True).floor('H')

    fetch_data_from = pd.Timestamp('2023-01-01 0:00:00+0000', tz='UTC') #quizas cambiarlo y que sea solo el año en curso
    fetch_data_to = pd.to_datetime(current_date - timedelta(hours=1), utc=True)

    feature_store_1 = get_feature_store()
    predictions_fg = feature_store_1.get_feature_view(name=config.FEATURE_VIEW_MODEL_PREDICTIONS)
    ts_data_1 = predictions_fg.get_batch_data(
        start_time=pd.to_datetime(fetch_data_from, utc=True),
        end_time=pd.to_datetime(fetch_data_to, utc=True)
    )

    feature_store_2 = get_feature_store()
    actuals_fg = feature_store_2.get_feature_view(name=config.FEATURE_VIEW_NAME)
    ts_data_2 = actuals_fg.get_batch_data(
        start_time=pd.to_datetime(fetch_data_from, utc=True),
        end_time=pd.to_datetime(fetch_data_to, utc=True)
    )



    # # 2 feature groups we need to merge
    # predictions_fg = get_feature_group(name=config.FEATURE_GROUP_MODEL_PREDICTIONS)
    # actuals_fg = get_feature_group(name=config.FEATURE_GROUP_NAME)

    # # query to join the 2 features groups by `pickup_hour` and `pickup_location_id`
    # query = predictions_fg.select_all() \
    #     .join(actuals_fg.select_all(), on=['pickup_hour', 'pickup_location_id']) \
    #     .filter(predictions_fg.pickup_hour >= from_date) \
    #     .filter(predictions_fg.pickup_hour <= to_date)
    
    # # create the feature view `config.FEATURE_VIEW_MONITORING` if it does not
    # # exist yet
    # feature_store = get_feature_store()
    # try:
    #     # create feature view as it does not exist yet
    #     feature_store.create_feature_view(
    #         name=config.FEATURE_VIEW_MONITORING,
    #         version=1,
    #         query=query
    #     )
    # except:
    #     print('Feature view already existed. Skip creation.')

    # # feature view
    # monitoring_fv = feature_store.get_feature_view(
    #     name=config.FEATURE_VIEW_MONITORING,
    #     version=1
    # )

    # # fetch data form the feature view
    # # fetch predicted and actual values for the last 30 days
    # monitoring_df = monitoring_fv.get_batch_data(
    #     start_time=pd.to_datetime(from_date - timedelta(days=7), utc=True),
    #     end_time=pd.to_datetime(to_date + timedelta(days=7), utc=True)
    # )
    # monitoring_df = monitoring_df[monitoring_df.pickup_hour.between(from_date, to_date)]

    return ts_data_1, ts_data_2


# def transform_ts_data_hopsworks_into_df_comparable_with_predictions(
#     ts_data: pd.DataFrame,
#     input_seq_len: int,
#     step_size: int,
#     output_seq_len: int #Lo que agregué nuevo
# ) -> pd.DataFrame:
#     """
#     Slices and transposes data from time-series format into a (features, target)
#     format that we can use to train Supervised ML models
#     """
#     assert set(ts_data.columns) == {'pickup_hour', 'rides', 'pickup_location_id'}

#     location_ids = ts_data['pickup_location_id'].unique()
#     #features = pd.DataFrame()
#     real_rides = pd.DataFrame()
    
#     for location_id in tqdm(location_ids):
        
#         # keep only ts data for this `location_id`
#         ts_data_one_location = ts_data.loc[
#             ts_data.pickup_location_id == location_id, 
#             ['pickup_hour', 'rides']
#         ].sort_values(by=['pickup_hour'])

#         # pre-compute cutoff indices to split dataframe rows
#         indices = get_cutoff_indices_features_and_target(
#             ts_data_one_location,
#             input_seq_len,
#             step_size,
#             output_seq_len #Lo que agregué nuevo
#         )

#         # slice and transpose data into numpy arrays for features and targets
#         n_examples = len(indices)
#         #x = np.ndarray(shape=(n_examples, input_seq_len), dtype=np.float32)
#         y = np.ndarray(shape=(n_examples, output_seq_len), dtype=np.float32) #Agregué el (output_seq_len) porque quiero esa cantidad de horas
#         pickup_hours = []
#         for i, idx in enumerate(indices):
#             #x[i, :] = ts_data_one_location.iloc[idx[0]:idx[1]]['rides'].values
#             y[i] = ts_data_one_location.iloc[idx[1]:idx[2]]['rides'].values
#             pickup_hours.append(ts_data_one_location.iloc[idx[1]]['pickup_hour'])

#         # numpy -> pandas
#         # features_one_location = pd.DataFrame(
#         #     x,
#         #     columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(input_seq_len))]
#         # )
#         # features_one_location['pickup_hour'] = pickup_hours
#         # features_one_location['pickup_location_id'] = location_id

#         # numpy -> pandas
#         real_rides_one_location = pd.DataFrame(y, columns=[f'real_rides_next_{i+1}_hour' for i in range(output_seq_len)])
#         real_rides_one_location['pickup_hour'] = pickup_hours
#         real_rides_one_location['pickup_location_id'] = location_id

#         # concatenate results
#         #features = pd.concat([features, features_one_location])
#         real_rides = pd.concat([real_rides, real_rides_one_location])

#     #features.reset_index(inplace=True, drop=True)
#     real_rides.reset_index(inplace=True, drop=True)

#     return  real_rides #,features  #['target_rides_next_hour']