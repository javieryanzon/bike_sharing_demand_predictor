"""
Versiones más robustas de las funciones de inferencia
"""
import pandas as pd
from datetime import datetime, timedelta
import time
from src.feature_store_api import get_feature_store
from src.logger import get_logger
import src.config as config

logger = get_logger(__name__)

def load_batch_of_features_from_store_robust(current_date: pd.Timestamp, max_retries: int = 3) -> pd.DataFrame:
    """
    Versión robusta para cargar features del feature store con múltiples estrategias de retry
    """
    logger.info(f"Loading features for inference at {current_date}")
    
    # Calcular el rango de fechas
    fetch_data_from = current_date - timedelta(days=28)
    fetch_data_to = current_date - timedelta(hours=1)
    
    logger.info(f'Fetching data from {fetch_data_from} to {fetch_data_to}')
    
    # Obtener feature store y feature view
    feature_store = get_feature_store()
    feature_view = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_NAME,
        version=config.FEATURE_VIEW_VERSION
    )
    
    # Asegurar que batch scoring esté inicializado
    if not feature_view._batch_scoring_server._serving_initialized:
        logger.info("Initializing batch scoring server...")
        feature_view.init_batch_scoring()
    
    # Definir estrategias de lectura
    read_strategies = [
        {
            "name": "Arrow Flight optimized",
            "options": {
                "arrow_flight_config": {
                    "timeout": 300,  # 5 minutos
                    "max_retries": 3,
                    "retry_delay": 2.0,
                    "chunk_size": 10000,
                }
            }
        },
        {
            "name": "Arrow Flight conservative",
            "options": {
                "arrow_flight_config": {
                    "timeout": 180,
                    "max_retries": 2,
                }
            }
        },
        {
            "name": "Arrow Flight minimal",
            "options": {
                "arrow_flight_config": {}
            }
        },
        {
            "name": "Default",
            "options": {}
        }
    ]
    
    # Intentar cada estrategia
    for attempt in range(max_retries):
        for strategy in read_strategies:
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries} - Strategy: {strategy['name']}")
                
                ts_data = feature_view.get_batch_data(
                    start_time=pd.to_datetime(fetch_data_from - timedelta(days=1), utc=True),
                    end_time=pd.to_datetime(fetch_data_to + timedelta(days=1), utc=True),
                    read_options=strategy['options']
                )
                
                logger.info(f"✅ Successfully loaded {len(ts_data)} rows with strategy: {strategy['name']}")
                
                # Procesar los datos
                return process_batch_data(ts_data, current_date, fetch_data_from, fetch_data_to)
                
            except Exception as e:
                logger.warning(f"Strategy '{strategy['name']}' failed: {str(e)}")
                if "timeout" in str(e).lower():
                    logger.info("Timeout detected, trying next strategy...")
                    continue
                elif "hoodie.properties" in str(e):
                    logger.error("HUDI table structure issue detected")
                    raise ValueError("Feature store table structure needs to be initialized. Run the feature pipeline first.")
                else:
                    logger.warning(f"Unexpected error: {str(e)}")
                    continue
        
        # Esperar antes del siguiente intento
        if attempt < max_retries - 1:
            wait_time = (attempt + 1) * 30  # 30s, 60s, 90s...
            logger.info(f"Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
    
    # Si llegamos aquí, todas las estrategias fallaron
    raise Exception(f"Failed to load data after {max_retries} attempts with all strategies")

def process_batch_data(ts_data: pd.DataFrame, current_date: pd.Timestamp, 
                      fetch_data_from: pd.Timestamp, fetch_data_to: pd.Timestamp) -> pd.DataFrame:
    """
    Procesar los datos obtenidos del feature store
    """
    logger.info(f"Processing batch data: {len(ts_data)} rows")
    
    # Convertir a UTC aware datetime
    ts_data['pickup_hour'] = pd.to_datetime(ts_data['pickup_hour'], utc=True)
    
    # Filtrar por el rango de fechas requerido
    ts_data = ts_data[
        (ts_data['pickup_hour'] >= fetch_data_from) & 
        (ts_data['pickup_hour'] <= fetch_data_to)
    ]
    
    logger.info(f"After date filtering: {len(ts_data)} rows")
    
    # Verificar que tenemos datos
    if len(ts_data) == 0:
        logger.warning("No data found in the specified date range")
        raise ValueError(f"No data available between {fetch_data_from} and {fetch_data_to}")
    
    # Ordenar por pickup_hour y pickup_location_id
    ts_data = ts_data.sort_values(['pickup_location_id', 'pickup_hour'])
    
    # Obtener las características más recientes para cada location_id
    # (tomamos el último registro disponible para cada location)
    ts_data = ts_data.groupby('pickup_location_id').last().reset_index()
    
    logger.info(f"Final processed data: {len(ts_data)} rows (unique locations)")
    
    return ts_data

def get_model_predictions_robust(model, features: pd.DataFrame) -> pd.DataFrame:
    """
    Versión robusta para obtener predicciones del modelo
    """
    logger.info(f"Making predictions for {len(features)} samples")
    
    try:
        # Preparar las features (remover columnas no necesarias)
        feature_columns = [col for col in features.columns 
                          if col not in ['pickup_hour', 'pickup_location_id']]
        
        X = features[feature_columns]
        
        logger.info(f"Using {len(feature_columns)} features for prediction")
        
        # Hacer predicciones
        predictions_array = model.predict(X)
        
        # Crear DataFrame con predicciones
        predictions_df = pd.DataFrame({
            'pickup_location_id': features['pickup_location_id'],
            'predicted_demand': predictions_array.flatten()
        })
        
        # Asegurar que las predicciones no sean negativas
        predictions_df['predicted_demand'] = predictions_df['predicted_demand'].clip(lower=0)
        
        logger.info(f"Generated {len(predictions_df)} predictions")
        logger.info(f"Prediction stats - Min: {predictions_df['predicted_demand'].min():.2f}, "
                   f"Max: {predictions_df['predicted_demand'].max():.2f}, "
                   f"Mean: {predictions_df['predicted_demand'].mean():.2f}")
        
        return predictions_df
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise