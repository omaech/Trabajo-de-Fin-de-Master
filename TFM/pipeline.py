import os
import warnings
import logging
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Importaciones modulares internas (asegúrate de que src/ tenga estos módulos)
from src.data_loader import load_and_aggregate_sensors, load_target
from src.feature_engineering import split_and_scale
from src.models import get_candidate_models, evaluate_models_cv, get_feature_importances

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline():
    logging.info("=========================================================")
    logging.info(" TFM: SISTEMA PREDICTIVO DE MERMAS E INDUSTRIA 4.0 ")
    logging.info("=========================================================")
    
    # Configuración de Rutas Estructuradas
    base_dir = os.path.dirname(os.path.abspath(__file__))
    datos_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')
    reports_dir = os.path.join(base_dir, 'reports')
    figures_dir = os.path.join(reports_dir, 'figures')

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # 11 Sensores estándar alineados con el resto del proyecto
    sensors = ['PS1', 'PS5', 'PS6', 'FS1', 'FS2', 'TS1', 'TS4', 'VS1', 'EPS1', 'CE', 'CP']
    
    logging.info("1. Cargando y agregando características de los 11 sensores clave...")
    X_df = load_and_aggregate_sensors(datos_dir, sensors)
    y_series = load_target(datos_dir, target_column_idx=0)
    
    logging.info(f"   Matriz consolidada: {X_df.shape[0]} ciclos x {X_df.shape[1]} características.")
    
    logging.info("2. Preparando datos y escalado con StandardScaler...")
    df = X_df.copy()
    df['target'] = y_series
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df, target_col='target') 
    
    # Guardar Scaler para la API
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    logging.info("   StandardScaler guardado correctamente en 'models/scaler.pkl'.")
    
    logging.info("3. Ejecutando Benchmark Multimodelo (Stratified 5-Fold Cross-Validation)...")
    models = get_candidate_models()
    
    X_scaled = scaler.fit_transform(X_df)
    results_df = evaluate_models_cv(models, X_scaled, y_series, cv_folds=5)
    
    print("\n" + "="*80)
    print(" TABLA COMPARATIVA DE MODELOS (BENCHMARK TFM)")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80 + "\n")
    
    logging.info("4. Análisis de Importancia de Variables (Top 10 Sensores más críticos)...")
    feat_imp = get_feature_importances(X_df, y_series)
    print("\nTOP 10 VARIABLES MÁS INFLUYENTES EN EL PROCESO:")
    print(feat_imp.head(10).to_string(index=False))

    # 5. Exportación de Resultados e Imágenes para la Memoria
    results_df.to_csv(os.path.join(reports_dir, 'benchmark_results.csv'), index=False)
    feat_imp.to_csv(os.path.join(reports_dir, 'feature_importance.csv'), index=False)

    plt.figure(figsize=(10, 6))
    plt.barh(feat_imp['Sensor_Metrica'].head(10)[::-1], feat_imp['Importancia'].head(10)[::-1], color='#1f77b4')
    plt.xlabel('Importancia Relativa')
    plt.title('Top 10 Sensores Críticos para la Predicción de Mermas')
    plt.tight_layout()
    
    fig_path = os.path.join(figures_dir, 'feature_importance.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()

    logging.info(f" Pipeline finalizado. Resultados en '{reports_dir}' y artefactos en '{models_dir}'.")

if __name__ == '__main__':
    run_pipeline()