from typing import Optional, List
from datetime import timedelta

import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go

def plot_one_sample(
    example_id: int,
    features: pd.DataFrame,
    targets: Optional[pd.DataFrame] = None, #pd.Series, 
    predictions: Optional[pd.DataFrame] = None,
    directions: Optional[pd.DataFrame] = None
):
    """"""
    if directions is not None:
        features_ = pd.merge(features, directions, left_on='pickup_location_id', right_on='ID', how='left')
        features_ = features_.iloc[example_id] #Lo modifique porque en otros casos no filtrando^
    else:
        features_ = features.iloc[example_id] #Lo modifique porque en otros casos no filtrando^ 
    
    
    if targets is not None:
        target_ = targets.iloc[example_id] #Lo modifique porque en otros casos no filtrando
        ts_columns_targets = [c for c in targets.columns if c.startswith('rides_next_')]
        ts_values_targets = [target_[c] for c in ts_columns_targets] 
        ts_dates_targets = pd.date_range(
        features_['pickup_hour'],
        features_['pickup_hour'] + timedelta(hours=len(ts_columns_targets)-1),
        freq='H'
    )
    else:
        target_ = None
    
    # features_ = features[features['pickup_location_id'] == example_id]
    # target_ = targets[targets['pickup_location_id'] == example_id]

   

    ts_columns_features = [c for c in features.columns if c.startswith('rides_previous_')]
    
    ts_values_features = [features_[c] for c in ts_columns_features] 
    

    ts_dates_features = pd.date_range(
        features_['pickup_hour'] - timedelta(hours=len(ts_columns_features)), #Agregp .iloc[0] `porque quiero tomar el unico valor de esa serie`
        features_['pickup_hour'] - timedelta(hours=1),
        freq='H'
    )


    fig = go.Figure()
    if directions is not None:
        title = f'Pick up hour= {features_["pickup_hour"]}, location_id= {features_["pickup_location_id"]}, direction= {features_["DIRECCION"]}'
    else:
        title = f'Pick up hour= {features_["pickup_hour"]}, location_id= {features_["pickup_location_id"]}'
    fig = px.line( x=ts_dates_features, y=ts_values_features,
                template='plotly_dark',
                markers=True, title=title)

    if targets is not None:
        targets_fig = px.line(x=ts_dates_targets, y=ts_values_targets,  
                    template='plotly_dark', 
                    markers=True, title='actual values')
        targets_fig.update_traces(line_color='green')
        fig.add_traces(targets_fig.data)
    

    if predictions is not None:
 
        prediction_ = predictions.iloc[example_id]
        #prediction_ = predictions[predictions['pickup_location_id'] == example_id]
        ts_columns_predictions = [c for c in predictions.columns if c.startswith('rides_next_')]       
        ts_values_predictions = [prediction_[c] for c in ts_columns_predictions]
        ts_dates_predictions = pd.date_range(
        features_['pickup_hour'],
        features_['pickup_hour'] + timedelta(hours=len(ts_columns_predictions)-1),
        freq='H'
        )
 
        prediction_fig = px.line(x=ts_dates_predictions, y=ts_values_predictions, 
                template='plotly_dark', 
                markers=True, title='predicted values')
        prediction_fig.update_traces(line_color='darkorange')
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
