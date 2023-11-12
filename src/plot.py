from typing import Optional, List
from datetime import timedelta

import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go

# def plot_one_sample(
#     example_id: int,
#     features: pd.DataFrame,
#     targets: Optional[pd.DataFrame] = None, #pd.Series, 
#     predictions: Optional[pd.DataFrame] = None,
#     directions: Optional[pd.DataFrame] = None
# ):
#     """"""
#     if directions is not None:
#         features_ = pd.merge(features, directions, left_on='pickup_location_id', right_on='ID', how='left')
#         features_[features_.index == example_id].iloc[-1] #para traer la ultima fila del dataset en caso de que la consulta a la feature store traiga más de una hora
#         #features_ = features_.iloc[example_id] #Lo modifique porque en otros casos no filtrando^
#     else:
#         #features_ = features.iloc[example_id] #Lo modifique porque en otros casos no filtrando^ 
#         features_[features_.index == example_id].iloc[-1] #para traer la ultima fila del dataset en caso de que la consulta a la feature store traiga más de una hora
    
    
#     if targets is not None:
#         target_ = targets[targets.index == example_id].iloc[-1] #para traer la ultima fila del dataset en caso de que la consulta a la feature store traiga más de una hora
#         #target_ = targets.iloc[example_id] #Lo modifique porque en otros casos no filtrando
#         ts_columns_targets = [c for c in targets.columns if c.startswith('rides_next_')]
#         ts_values_targets = [target_[c] for c in ts_columns_targets] 
#         ts_dates_targets = pd.date_range(
#         features_['pickup_hour'].iloc[0],
#         features_['pickup_hour'].iloc[0] + timedelta(hours=len(ts_columns_targets)-1),
#         freq='H'
#     )
#     else:
#         target_ = None
    
#     #features_df[features_df.index == 327].iloc[-1] #para traer la ultima fila del dataset en caso de que la consulta a la feature store traiga más de una hora
#     # features_ = features[features['pickup_location_id'] == example_id]
#     # target_ = targets[targets['pickup_location_id'] == example_id]

   

#     ts_columns_features = [c for c in features.columns if c.startswith('rides_previous_')]
    
#     ts_values_features = [features_[c] for c in ts_columns_features] 
    

#     ts_dates_features = pd.date_range(
#         features_['pickup_hour'].iloc[0] - timedelta(hours=len(ts_columns_features)), #Agregp .iloc[0] `porque quiero tomar el unico valor de esa serie`
#         features_['pickup_hour'].iloc[0] - timedelta(hours=1),
#         freq='H'
#     )


#     fig = go.Figure()
#     if directions is not None:
#         title = f'Pick up hour= {features_["pickup_hour"]}, location_id= {features_.index}, direction= {features_["DIRECCION"]}'
#     else:
#         title = f'Pick up hour= {features_["pickup_hour"]}, location_id= {features_.index}'
#     fig = px.line( x=ts_dates_features, y=ts_values_features,
#                 template='plotly_dark',
#                 markers=True, title=title)

#     if targets is not None:
#         targets_fig = px.line(x=ts_dates_targets, y=ts_values_targets,  
#                     template='plotly_dark', 
#                     markers=True, title='actual values')
#         targets_fig.update_traces(line_color='green')
#         fig.add_traces(targets_fig.data)
    

#     if predictions is not None:
 
#         prediction_ = predictions.iloc[example_id]
#         #prediction_ = predictions[predictions['pickup_location_id'] == example_id]
#         ts_columns_predictions = [c for c in predictions.columns if c.startswith('rides_next_')]       
#         ts_values_predictions = [prediction_[c] for c in ts_columns_predictions]
#         ts_dates_predictions = pd.date_range(
#         features_['pickup_hour'].iloc[0],
#         features_['pickup_hour'].iloc[0] + timedelta(hours=len(ts_columns_predictions)-1),
#         freq='H'
#         )
 
#         prediction_fig = px.line(x=ts_dates_predictions, y=ts_values_predictions, 
#                 template='plotly_dark', 
#                 markers=True, title='predicted values')
#         prediction_fig.update_traces(line_color='darkorange')
#         fig.add_traces(prediction_fig.data)   

#     return fig

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
