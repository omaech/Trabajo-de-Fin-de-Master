import os
import pandas as pd

def load_and_aggregate_sensors(datos_dir: str, sensor_files: list) -> pd.DataFrame:
    """
    Lee cada archivo de sensor .txt y extrae métricas estadísticas 
    (media, desviación estándar, máximo, mínimo) por cada ciclo.
    """
    aggregated_features = {}
    
    for sensor in sensor_files:
        file_path = os.path.join(datos_dir, f"{sensor}.txt")
        if os.path.exists(file_path):
            df_sensor = pd.read_csv(file_path, sep='\t', header=None)
            
            # Extraemos múltiples métricas por ciclo
            aggregated_features[f"{sensor}_mean"] = df_sensor.mean(axis=1)
            aggregated_features[f"{sensor}_std"]  = df_sensor.std(axis=1)
            aggregated_features[f"{sensor}_max"]  = df_sensor.max(axis=1)
            aggregated_features[f"{sensor}_min"]  = df_sensor.min(axis=1)
        else:
            print(f"Advertencia: El archivo {file_path} no fue encontrado.")
            
    return pd.DataFrame(aggregated_features)

def load_target(datos_dir: str, target_column_idx: int = 0) -> pd.Series:
    """
    Carga profile.txt y extrae la columna objetivo deseada.
    """
    profile_path = os.path.join(datos_dir, "profile.txt")
    df_profile = pd.read_csv(profile_path, sep='\t', header=None)
    return df_profile[target_column_idx]