from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error
import plotly.express as px

from src.monitoring import load_predictions_and_actual_values_from_store
from src.data import transform_ts_data_into_dataset_comparable_with_predictions

st.set_page_config(layout="wide")

# title
current_date = pd.to_datetime(datetime.utcnow(), utc=True).floor('H')
st.title(f'Monitoring dashboard 🔎')

progress_bar = st.sidebar.header('⚙️ Working Progress')
progress_bar = st.sidebar.progress(0)
N_STEPS = 3


@st.cache_data
def _load_predictions_and_actuals_from_store(
    from_date: datetime,
    to_date: datetime
    ) -> pd.DataFrame:
    """Wrapped version of src.monitoring.load_predictions_and_actual_values_from_store, so
    we can add Streamlit caching

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
    return load_predictions_and_actual_values_from_store(from_date, to_date)

# with st.spinner(text="Fetching model predictions and actual values from the store"):
    
#     ts_data_1, ts_data_2 = _load_predictions_and_actuals_from_store(
#         from_date=current_date - timedelta(days=14),
#         to_date=current_date
#     )
    # real_rides = transform_ts_data_into_dataset_comparable_with_predictions(
    # ts_data_2,
    # input_seq_len=0, # one month
    # step_size=24,
    # output_seq_len=36
    # )
    # st.sidebar.write('✅ Model predictions and actual values arrived')
    # progress_bar.progress(1/N_STEPS)

while True:
        try:
            with st.spinner(text="Fetching model predictions and actual values from the store"):
                ts_data_1, ts_data_2 = _load_predictions_and_actuals_from_store(
                from_date=current_date - timedelta(days=14),
                to_date=current_date
                )
                real_rides = transform_ts_data_into_dataset_comparable_with_predictions(
                ts_data_2,
                input_seq_len=0, # one month
                step_size=24,
                output_seq_len=36
                )
                st.sidebar.write('✅ Model predictions and actual values arrived')
                progress_bar.progress(1/N_STEPS)
                break
        except Exception as e:
            # Captura el error
            st.error(f"An error occurred: {str(e)}")

            # Intenta nuevamente
            st.warning(f"Retrying...")

with st.spinner(text="Plotting aggregate MAE hour-by-hour"):

    monitoring_df = pd.merge(ts_data_1, real_rides, on=['pickup_hour', 'pickup_location_id'], how='inner')
    st.header('Mean Absolute Error (MAE) hour-by-hour')
    selected_columns_pred = [c for c in monitoring_df.columns if c.startswith('rides_next_')] #####
    selected_columns_real = [c for c in monitoring_df.columns if c.startswith('real_rides_next_')]

    # MAE per pickup_hour
    # https://stackoverflow.com/a/47914634
    mae_per_hour = (
        monitoring_df
        .groupby('pickup_hour')
        .apply(lambda g: mean_absolute_error(g[selected_columns_real], g[selected_columns_pred])) ####
        .reset_index()
        .rename(columns={0: 'mae'})
        .sort_values(by='pickup_hour')
    )

    fig = px.bar(
        mae_per_hour,
        x='pickup_hour', y='mae',
        template='plotly_dark',
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True, width=1000)

    progress_bar.progress(2/N_STEPS)


with st.spinner(text="Plotting MAE hour-by-hour for top locations"):
    
    st.header('Mean Absolute Error (MAE) per location and hour')

    top_locations_by_demand = (
        monitoring_df
        .groupby('pickup_location_id')[selected_columns_real].sum()
        .sum(axis=1)
        .sort_values(ascending=False)
        .reset_index()
        .head(10)['pickup_location_id']
    )

    for location_id in top_locations_by_demand:
        
        mae_per_hour = (
            monitoring_df[monitoring_df.pickup_location_id == location_id]
            .groupby('pickup_hour')
            .apply(lambda g: mean_absolute_error(g[selected_columns_real], g[selected_columns_pred]))
            .reset_index()
            .rename(columns={0: 'mae'})
            .sort_values(by='pickup_hour')
        )

        fig = px.bar(
            mae_per_hour,
            x='pickup_hour', y='mae',
            template='plotly_dark',
        )
        st.subheader(f'{location_id=}')
        st.plotly_chart(fig, theme="streamlit", use_container_width=True, width=1000)

    progress_bar.progress(3/N_STEPS)