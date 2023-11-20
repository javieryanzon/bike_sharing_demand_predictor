import zipfile 
from datetime import datetime, timedelta

import requests
import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
import pydeck as pdk
import numpy as np

from src.inference import (
    load_predictions_from_store,
    load_batch_of_features_from_store
)
from src.paths import DATA_DIR
from src.plot import plot_one_sample

st.set_page_config(layout="wide")

# title
# current_date = datetime.strptime('2023-01-05 12:00:00', '%Y-%m-%d %H:%M:%S')
current_date = pd.to_datetime(datetime.utcnow(), utc=True).floor('H')
current_date_str = str(current_date.strftime('%Y-%m-%d %H:%M'))
st.title(f'Bike demand prediction 🚲')
st.header(f'{current_date_str} UTC')

progress_bar = st.sidebar.header('⚙️ Working Progress')
progress_bar = st.sidebar.progress(0)
N_STEPS = 6

def load_shape_data_file() -> gpd.geodataframe.GeoDataFrame:
    """
    Fetches remote file with shape data, that we later use to plot the
    different pickup_location_ids on the map of NYC.

    Raises:
        Exception: when we cannot connect to the external server where
        the file is.

    Returns:
        GeoDataFrame: columns -> (OBJECTID	Shape_Leng	Shape_Area	zone	LocationID	borough	geometry)
    """
    # download zip file
    URL = 'https://cdn.buenosaires.gob.ar/datosabiertos/datasets/transporte-y-obras-publicas/estaciones-bicicletas-publicas/estaciones-de-bicicletas-zip.zip'
    response = requests.get(URL)
    path = DATA_DIR / f'IE-Estaciones.zip'
    if response.status_code == 200:
        open(path, "wb").write(response.content)
    else:
        raise Exception(f'{URL} is not available')

    # unzip file
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR / 'IE-Estaciones')

    # load and return shape file
    return gpd.read_file(DATA_DIR / 'IE-Estaciones/IE-Estaciones.shp').to_crs('epsg:4326') # 3857

@st.cache_data
def _load_batch_of_features_from_store(current_date: datetime) -> pd.DataFrame:
    """Wrapped version of src.inference.load_batch_of_features_from_store, so
    we can add Streamlit caching

    Args:
        current_date (datetime): _description_

    Returns:
        pd.DataFrame: n_features + 2 columns:
            - `rides_previous_N_hour`
            - `rides_previous_{N-1}_hour`
            - ...
            - `rides_previous_1_hour`
            - `pickup_hour`
            - `pickup_location_id`
    """
    return load_batch_of_features_from_store(current_date)

@st.cache_data
def _load_predictions_from_store(
    from_pickup_hour: datetime,
    to_pickup_hour: datetime
    ) -> pd.DataFrame:
    """
    Wrapped version of src.inference.load_predictions_from_store, so we
    can add Streamlit caching

    Args:
        from_pickup_hour (datetime): min datetime (rounded hour) for which we want to get
        predictions

        to_pickup_hour (datetime): max datetime (rounded hour) for which we want to get
        predictions

    Returns:
        pd.DataFrame: 2 columns: pickup_location_id, predicted_demand
    """
    return load_predictions_from_store(from_pickup_hour, to_pickup_hour)

with st.spinner(text="Downloading shape file to plot bike stations"):
    geo_df = load_shape_data_file()
    st.sidebar.write('✅ Shape file was downloaded ')
    progress_bar.progress(1/N_STEPS)

with st.spinner(text="Fetching model predictions from the store"):
    predictions_df = _load_predictions_from_store(
        from_pickup_hour=current_date - timedelta(hours=1),
        to_pickup_hour=current_date
        )
    predictions_df = predictions_df.reset_index(drop=True)
    #predictions_df=predictions_df.set_index("pickup_location_id")
    #predictions_df.index.name = None
    st.sidebar.write('✅ Model predictions arrived')
    progress_bar.progress(2/N_STEPS)

# Here we are checking the predictions for the current hour have already been computed
# and are available
next_hour_predictions_ready = \
    False if predictions_df[predictions_df.pickup_hour == current_date].empty else True
prev_hour_predictions_ready = \
    False if predictions_df[predictions_df.pickup_hour == (current_date - timedelta(hours=1))].empty else True

if next_hour_predictions_ready:
    # predictions for the current hour are available
    predictions_df = predictions_df[predictions_df.pickup_hour == current_date]
elif prev_hour_predictions_ready:
    # predictions for current hour are not available, so we use previous hour predictions
    predictions_df = predictions_df[predictions_df.pickup_hour == (current_date - timedelta(hours=1))]
    current_date = current_date - timedelta(hours=1)
    st.subheader('⚠️ The most recent data is not yet available. Using last hour predictions')
else:
    raise Exception('Features are not available for the last 2 hours. Is your feature \
                    pipeline up and running? 🤔')


with st.spinner(text="Preparing data to plot"):

    def pseudocolor(val, minval, maxval, startcolor, stopcolor):
        """
        Convert value in the range minval...maxval to a color in the range
        startcolor to stopcolor. The colors passed and the the one returned are
        composed of a sequence of N component values.

        Credits to https://stackoverflow.com/a/10907855
        """
        f = float(val-minval) / (maxval-minval)
        return tuple(f*(b-a)+a for (a, b) in zip(startcolor, stopcolor))
    
    df = pd.merge(geo_df, predictions_df,
                  right_on='pickup_location_id',
                  left_on='ID',
                  how='inner')
    
    BLACK, ORANGE = (0, 0, 0), (255, 128, 0)
    selected_columns = [c for c in df.columns if c.startswith('rides_next_')]
    df['max_hour'] = df[selected_columns].idxmax(axis=1)
    df['color_scaling'] = df[selected_columns].max(axis=1) 
    max_pred, min_pred = df['color_scaling'].max(), df['color_scaling'].min()
    df['fill_color'] = df['color_scaling'].apply(lambda x: pseudocolor(x, min_pred, max_pred, BLACK, ORANGE))
    
    progress_bar.progress(3/N_STEPS)

with st.spinner(text="Generating BsAs Map"):

    INITIAL_VIEW_STATE = pdk.ViewState(
        latitude=-34.60280869220721,
        longitude=-58.42827362585887,
        zoom=11,
        max_zoom=16,
        pitch=45,
        bearing=0
    )
    layer = pdk.Layer("ColumnLayer",
                          data=df,
                          get_position=["Lon", "Lat"],
                          get_elevation=['color_scaling'],
                          auto_highlight=True,
                          radius=50,
                          elevation_scale=300,                          
                          get_fill_color="fill_color",
                          get_line_color=[255, 255, 255],
                          pickable=True,
                          extruded=True,
                          coverage=1)
    

    tooltip = {"html": "<b>Zone:</b> [{ID} - {DIRECCION}] <br /> <b>Max: </b> {color_scaling} rides - {max_hour}"}

    r = pdk.Deck(
        layers=[layer],
        initial_view_state=INITIAL_VIEW_STATE,
        tooltip=tooltip
    )

    st.pydeck_chart(r)
    progress_bar.progress(4/N_STEPS)

with st.spinner(text="Fetching batch of features used in the last run"):
    features_df = _load_batch_of_features_from_store(current_date)
    features_df=features_df.reset_index(drop=True)
    #features_df=features_df.set_index("pickup_location_id")
    #features_df.index.name = None
    st.sidebar.write('✅ Inference features fetched from the store')
    progress_bar.progress(5/N_STEPS)

with st.spinner(text="Plotting time-series data"):
    
    predictions_df = np.clip(predictions_df[selected_columns], 0, None) #Hago esto para limitar los valores a cero y que no de ninguno negativo 
    

    predictions_df['max'] = predictions_df[selected_columns].max(axis=1) 
    sorted_indices = predictions_df['max'].sort_values(ascending=False).index
    predictions_max = predictions_df.copy()
    predictions_max['max_hour'] = predictions_max[selected_columns].idxmax(axis=1)
    predictions_df = predictions_df.drop('max', axis=1)

    # Selecciona las 10 filas principales 
    top_10_indices = sorted_indices[:10]
    #st.sidebar.write(top_10_indices)
    #st.sidebar.write(len(predictions_df))

    # plot each time-series with the prediction 
    for row_id in top_10_indices:
        #if row_id < len(predictions_df):
            # title
            location_id = df['pickup_location_id'].iloc[row_id]
            location_name = df['DIRECCION'].iloc[row_id]
            st.header(f'Direction: {location_id} - {location_name}')

            # plot predictions
            prediction = predictions_max['max'].iloc[row_id] #df['color_scaling'].iloc[row_id]
            max_hour_prediction = predictions_max['max_hour'].iloc[row_id]
            max_hour_prediction_int = int(max_hour_prediction.replace('rides_next_', '').replace('_hour', ''))
            max_hour_prediction_str =str(pd.to_datetime(current_date + timedelta(hours=max_hour_prediction_int-1), utc=True).strftime('%Y-%m-%d %H:%M'))+ " UTC " + " - " + str(pd.to_datetime(current_date + timedelta(hours=max_hour_prediction_int), utc=True).strftime('%Y-%m-%d %H:%M') + " UTC")
            st.metric(label="Max rides predicted in 36 hours", value=int(prediction))
            st.metric(label="Approximate Hour of max prediction", value=max_hour_prediction_str)

            fig = plot_one_sample(
                example_id=row_id,
                features=features_df,
                targets=predictions_df,
                predictions=predictions_df
                #directions=geo_df[['ID', 'DIRECCION']]
            )
            st.plotly_chart(fig, theme="streamlit", use_container_width=True, width=1000)

    progress_bar.progress(6/N_STEPS)