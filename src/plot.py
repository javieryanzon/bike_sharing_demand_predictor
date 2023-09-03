from typing import Optional, List
from datetime import timedelta

import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go

def plot_one_sample(
    features: pd.DataFrame,
    targets: pd.DataFrame, #pd.Series, 
    example_id: int,
    predictions: Optional[pd.DataFrame] = None,
):
    """"""
    features_ = features.iloc[example_id]
    target_ = targets.iloc[example_id]
    
    # ts_columns = [c for c in features.columns if c.startswith('rides_previous_')]
    # ts_values = [features_[c] for c in ts_columns] + [target_]
    ts_columns_features = [c for c in features.columns if c.startswith('rides_previous_')]
    ts_columns_targets = [c for c in targets.columns if c.startswith('rides_next_')]
    ts_values_features = [features_[c] for c in ts_columns_features] 
    ts_values_targets = [target_[c] for c in ts_columns_targets] 
    # ts_dates = pd.date_range(
    #     features_['pickup_hour'] - timedelta(hours=len(ts_columns)),
    #     features_['pickup_hour'],
    #     freq='H'
    # )
    ts_dates_features = pd.date_range(
        features_['pickup_hour'] - timedelta(hours=len(ts_columns_features)),
        features_['pickup_hour'] - timedelta(hours=1),
        freq='H'
    )
    ts_dates_targets = pd.date_range(
        features_['pickup_hour'],
        features_['pickup_hour'] + timedelta(hours=len(ts_columns_targets)-1),
        freq='H'
    )
    
    # line plot with past values
    # title = f'Pick up hour={features_["pickup_hour"]}, location_id={features_["pickup_location_id"]}'
    # fig = px.line(
    #     x=ts_dates, y=ts_values,
    #     template='plotly_dark',
    #     markers=True, title=title
    # )
    fig = go.Figure()
    title = f'Pick up hour={features_["pickup_hour"]}, location_id={features_["pickup_location_id"]}'
    fig = px.line( x=ts_dates_features, y=ts_values_features,
                template='plotly_dark',
                markers=True, title=title)
    #features_fig.update_traces(line_color='blue')
    #fig.add_traces(features_fig.data)
    
    # green line for the values we wanna predict
    # fig.add_scatter(x=ts_dates[-1:], y=[target_],
    #                 line_color='green',
    #                 mode='markers', marker_size=10, name='actual value') 
    targets_fig = px.line(x=ts_dates_targets, y=target_.values.tolist(),
                template='plotly_dark', 
                markers=True, title='actual values')
    targets_fig.update_traces(line_color='green')
    fig.add_traces(targets_fig.data)
    #fig.show()

    if predictions is not None:
        # red line for the predicted values, if passed
        # prediction_ = predictions.iloc[example_id]
        # fig.add_scatter(x=ts_dates[-1:], y=[prediction_],
        #                 line_color='red',
        #                 mode='markers', marker_symbol='x', marker_size=15,
        #                 name='prediction')  
        prediction_ = predictions.iloc[example_id]
        prediction_fig = px.line(x=ts_dates_targets, y=prediction_.values.tolist(),
                template='plotly_dark', 
                markers=True, title='predicted values')
        prediction_fig.update_traces(line_color='red')
        fig.add_traces(prediction_fig.data)   

    return fig


def plot_ts(
    ts_data: pd.DataFrame,
    locations: Optional[List[int]] = None
    ):
    """
    Plot time-series data
    """
    ts_data_to_plot = ts_data[ts_data.pickup_location_id.isin(locations)] if locations else ts_data

    fig = px.line(
        ts_data,
        x="pickup_hour",
        y="rides",
        color='pickup_location_id',
        template='none',
    )

    fig.show()
