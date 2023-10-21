from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
from pdb import set_trace as stop

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import pyarrow as pa
import zipfile
import pyarrow.parquet as pq
import subprocess

from src.paths import RAW_DATA_DIR, TRANSFORMED_DATA_DIR


def download_one_file_of_raw_data(year: int) -> Path: #, month: int) -> Path:
    """
    Downloads Parquet file with historical bike rides for the given `year` and
    `month`
    """
    URL = f'https://cdn.buenosaires.gob.ar/datosabiertos/datasets/transporte-y-obras-publicas/bicicletas-publicas/recorridos-realizados-{year}.zip'
    
    # Ruta de destino para guardar el archivo descargado
    destination_path = RAW_DATA_DIR / f'recorridos-realizados-{year}.zip'

    try:
        # Utiliza wget para descargar el archivo en la ubicación deseada
        subprocess.run(['wget', URL, '-O', destination_path])

        # Verifica si el archivo se descargó correctamente
        if destination_path.is_file():
            print(f'Descargado año {year}')
            return destination_path
        else:
            raise Exception(f'Error al descargar {URL}: El archivo no se descargó correctamente.')

    except Exception as e:
        raise Exception(f'Error al descargar {URL}: {str(e)}')
    
      
        # response = requests.get(URL)

    # if response.status_code == 200:
    #     path = RAW_DATA_DIR / f'recorridos-realizados-{year}.zip'
    #     open(path, "wb").write(response.content)
    #     print(f'descargado año {year}')
    #     # time.sleep(2)
    #     return path
    # else:
    #     raise Exception(f'{URL} is not available')

def unzip_and_convert_csv_to_parquet(year: int) -> Path:
    nombre_archivo_zip = RAW_DATA_DIR / f"recorridos-realizados-{year}.zip"
        # Descomprimir el archivo zip
    with zipfile.ZipFile(nombre_archivo_zip, 'r') as archivo_zip:

        # Extraer el archivo CSV del zip
        nombre_archivo_csv = archivo_zip.namelist()[0]  # Suponiendo que el archivo CSV es el primer archivo en el zip
        archivo_zip.extractall(RAW_DATA_DIR) #(f"../data/raw/")

        # Leer el archivo CSV con pandas
        df = pd.read_csv(nombre_archivo_csv, delimiter=',', decimal=".") #RAW_DATA_DIR /

        # Convertir el DataFrame a formato parquet
        nombre_archivo_parquet = f"rides_{year}.parquet"
        table = pa.Table.from_pandas(df)
        pq.write_table(table, RAW_DATA_DIR / nombre_archivo_parquet)

        path = RAW_DATA_DIR / f'rides_{year}.parquet'
    return path


def validate_raw_data(
    rides: pd.DataFrame,
    year: int,
    #month: int,
) -> pd.DataFrame:
    """
    Removes rows with pickup_datetimes outside their valid range
    """
    # keep only rides for this month
    # this_month_start = f'{year}-{month:02d}-01'
    # next_month_start = f'{year}-{month+1:02d}-01' if month < 12 else f'{year+1}-01-01'
    this_year_start = f'{year}-01-01'
    next_year_start = f'{year+1}-01-01'
    rides = rides[rides.pickup_datetime >= this_year_start]
    rides = rides[rides.pickup_datetime < next_year_start]
    
    return rides


def fetch_ride_events_from_data_warehouse(
    from_date: datetime,
    to_date: datetime
) -> pd.DataFrame:
    """
    This function is used to simulate production data by sampling historical data
    from 52 weeks ago (i.e. 1 year)
    """
    from_date_ = from_date - timedelta(days=7*52)
    to_date_ = to_date - timedelta(days=7*52)
    print(f'Fetching ride events from {from_date} to {to_date}')

    if (from_date_.year == to_date_.year) and (from_date_.month == to_date_.month):
        # download 1 file of data only
        rides = load_raw_data(year=from_date_.year, months=from_date_.month)
        rides = rides[rides.pickup_datetime >= from_date_]
        rides = rides[rides.pickup_datetime < to_date_]

    else:
        # download 2 files from website
        rides = load_raw_data(year=from_date_.year, months=from_date_.month)
        rides = rides[rides.pickup_datetime >= from_date_]
        rides_2 = load_raw_data(year=to_date_.year, months=to_date_.month)
        rides_2 = rides_2[rides_2.pickup_datetime < to_date_]
        rides = pd.concat([rides, rides_2])

    # shift the pickup_datetime back 1 year ahead, to simulate production data
    # using its 7*52-days-ago value
    rides['pickup_datetime'] += timedelta(days=7*52)

    rides.sort_values(by=['pickup_location_id', 'pickup_datetime'], inplace=True)

    return rides


def load_raw_data(
    year: int
    #months: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Loads raw data from local storage or downloads it from the BsAs website, and
    then loads it into a Pandas DataFrame

    Args:
        year: year of the data to download
        #months: months of the data to download. If `None`, download all months

    Returns:
        pd.DataFrame: DataFrame with the following columns:
            - pickup_datetime: datetime of the pickup
            - pickup_location_id: ID of the pickup location
    """  
    rides = pd.DataFrame()
    
    # if months is None:
    #     # download data for the entire year (all months)
    #     months = list(range(1, 13))
    # elif isinstance(months, int):
    #     # download data only for the month specified by the int `month`
    #     months = [months]

    #for month in months:
        
    local_file = RAW_DATA_DIR / f'rides_{year}.parquet' #-{month:02d}.parquet'
    if not local_file.exists():
        try:
            # download the file from the BsAs website
            print(f'Downloading file {year}') #-{month:02d}
            download_one_file_of_raw_data(year)
            unzip_and_convert_csv_to_parquet(year)
        except:
            print(f'{year} file is not available')
            #continue
    else:
        print(f'File {year} was already in local storage') 

    # load the file into Pandas
    rides_one_year = pd.read_parquet(local_file)

    # rename columns
    rides_one_year = rides_one_year[['fecha_origen_recorrido', 'id_estacion_origen']]
    rides_one_year.rename(columns={
        'fecha_origen_recorrido': 'pickup_datetime',
        'id_estacion_origen': 'pickup_location_id',
        }, inplace=True)
    
    # eliminate "BAEcobici" and convert it to int type
    rides_one_year['pickup_location_id'] = rides_one_year['pickup_location_id'].str.replace('BAEcobici', '').astype(int)
    # transform "pickup_datetime" to datetime
    rides_one_year['pickup_datetime'] = pd.to_datetime(rides_one_year['pickup_datetime'],format='%Y-%m-%d %H:%M:%S')

    # validate the file
    rides_one_year = validate_raw_data(rides_one_year, year)

    # append to existing data
    rides = pd.concat([rides, rides_one_year])

    if rides.empty:
        # no data, so we return an empty dataframe
        return pd.DataFrame()
    else:
        # keep only time and origin of the ride
        rides = rides[['pickup_datetime', 'pickup_location_id']]
        return rides


def add_missing_slots(ts_data: pd.DataFrame) -> pd.DataFrame:
    """
    Add necessary rows to the input 'ts_data' to make sure the output
    has a complete list of
    - pickup_hours
    - pickup_location_ids
    """
    #Estaba generando más locations id por tanto lo modifique. Esta es la version antigua
    #location_ids = range(1, ts_data['pickup_location_id'].max() + 1)
    
    #Esta es la línea modificada !!!!!!!
    location_ids = ts_data['pickup_location_id'].unique()

    full_range = pd.date_range(ts_data['pickup_hour'].min(),
                               ts_data['pickup_hour'].max(),
                               freq='H')
    output = pd.DataFrame()
    for location_id in tqdm(location_ids):

        # keep only rides for this 'location_id'
        ts_data_i = ts_data.loc[ts_data.pickup_location_id == location_id, ['pickup_hour', 'rides']]
        
        if ts_data_i.empty:
            # add a dummy entry with a 0
            ts_data_i = pd.DataFrame.from_dict([
                {'pickup_hour': ts_data['pickup_hour'].max(), 'rides': 0}
            ])

        # quick way to add missing dates with 0 in a Series
        # taken from https://stackoverflow.com/a/19324591
        ts_data_i.set_index('pickup_hour', inplace=True)
        ts_data_i.index = pd.DatetimeIndex(ts_data_i.index)
        ts_data_i = ts_data_i.reindex(full_range, fill_value=0)
        
        # add back `location_id` columns
        ts_data_i['pickup_location_id'] = location_id

        output = pd.concat([output, ts_data_i])
    
    # move the pickup_hour from the index to a dataframe column
    output = output.reset_index().rename(columns={'index': 'pickup_hour'})
    
    return output


def transform_raw_data_into_ts_data(
    rides: pd.DataFrame
) -> pd.DataFrame:
    """"""
    # sum rides per location and pickup_hour
    rides['pickup_hour'] = rides['pickup_datetime'].dt.floor('H')
    agg_rides = rides.groupby(['pickup_hour', 'pickup_location_id']).size().reset_index()
    agg_rides.rename(columns={0: 'rides'}, inplace=True)

    # add rows for (locations, pickup_hours)s with 0 rides
    agg_rides_all_slots = add_missing_slots(agg_rides)

    return agg_rides_all_slots


def transform_ts_data_into_features_and_target(
    ts_data: pd.DataFrame,
    input_seq_len: int,
    step_size: int,
    output_seq_len: int #Lo que agregué nuevo
) -> pd.DataFrame:
    """
    Slices and transposes data from time-series format into a (features, target)
    format that we can use to train Supervised ML models
    """
    assert set(ts_data.columns) == {'pickup_hour', 'rides', 'pickup_location_id'}

    location_ids = ts_data['pickup_location_id'].unique()
    features = pd.DataFrame()
    targets = pd.DataFrame()
    
    for location_id in tqdm(location_ids):
        
        # keep only ts data for this `location_id`
        ts_data_one_location = ts_data.loc[
            ts_data.pickup_location_id == location_id, 
            ['pickup_hour', 'rides']
        ].sort_values(by=['pickup_hour'])

        # pre-compute cutoff indices to split dataframe rows
        indices = get_cutoff_indices_features_and_target(
            ts_data_one_location,
            input_seq_len,
            step_size,
            output_seq_len #Lo que agregué nuevo
        )

        # slice and transpose data into numpy arrays for features and targets
        n_examples = len(indices)
        x = np.ndarray(shape=(n_examples, input_seq_len), dtype=np.float32)
        y = np.ndarray(shape=(n_examples, output_seq_len), dtype=np.float32) #Agregué el (output_seq_len) porque quiero esa cantidad de horas
        pickup_hours = []
        for i, idx in enumerate(indices):
            x[i, :] = ts_data_one_location.iloc[idx[0]:idx[1]]['rides'].values
            y[i] = ts_data_one_location.iloc[idx[1]:idx[2]]['rides'].values
            pickup_hours.append(ts_data_one_location.iloc[idx[1]]['pickup_hour'])

        # numpy -> pandas
        features_one_location = pd.DataFrame(
            x,
            columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(input_seq_len))]
        )
        features_one_location['pickup_hour'] = pickup_hours
        features_one_location['pickup_location_id'] = location_id

        # numpy -> pandas
        targets_one_location = pd.DataFrame(y, columns=[f'rides_next_{i+1}_hour' for i in range(output_seq_len)])

        # concatenate results
        features = pd.concat([features, features_one_location])
        targets = pd.concat([targets, targets_one_location])

    features.reset_index(inplace=True, drop=True)
    targets.reset_index(inplace=True, drop=True)

    return features, targets #['target_rides_next_hour']


def get_cutoff_indices_features_and_target(
    data: pd.DataFrame,
    input_seq_len: int,
    step_size: int,
    output_seq_len: int #Lo que agregué nuevo
    ) -> list:

        stop_position = len(data) - 1
        
        # Start the first sub-sequence at index position 0
        subseq_first_idx = 0
        subseq_mid_idx = input_seq_len
        subseq_last_idx = input_seq_len + output_seq_len #le agrego "output_seq_len" para introducirlo como variable
        indices = []
        
        while subseq_last_idx <= stop_position:
            indices.append((subseq_first_idx, subseq_mid_idx, subseq_last_idx))
            subseq_first_idx += step_size
            subseq_mid_idx += step_size
            subseq_last_idx += step_size

        return indices
