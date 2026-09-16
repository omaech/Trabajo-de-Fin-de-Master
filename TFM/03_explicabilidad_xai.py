import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split

# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def main():
    # ==============================================================================
    # 0. CONFIGURACIÓN DE RUTAS RELATIVAS Y CARPETAS
    # ==============================================================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ==============================================================================
    # 1. CARGA DE DATOS Y DIVISIÓN TEMPORAL
    # ==============================================================================
    print("⏳ Cargando datos para el análisis XAI...")

    archivos_sensores = {
        'PS1': os.path.join(DATA_DIR, 'PS1.txt'),
        'PS5': os.path.join(DATA_DIR, 'PS5.txt'),
        'PS6': os.path.join(DATA_DIR, 'PS6.txt'),
        'FS1': os.path.join(DATA_DIR, 'FS1.txt'),
        'FS2': os.path.join(DATA_DIR, 'FS2.txt'),
        'TS1': os.path.join(DATA_DIR, 'TS1.txt'),
        'TS4': os.path.join(DATA_DIR, 'TS4.txt'),
        'VS1': os.path.join(DATA_DIR, 'VS1.txt'),
        'EPS1': os.path.join(DATA_DIR, 'EPS1.txt'),
        'CE': os.path.join(DATA_DIR, 'CE.txt'),
        'CP': os.path.join(DATA_DIR, 'CP.txt')
    }

    ruta_profile = os.path.join(DATA_DIR, 'profile.txt')

    df_features = pd.DataFrame()
    for nombre_sensor, archivo in archivos_sensores.items():
        if os.path.exists(archivo):
            data_sensor = pd.read_csv(archivo, sep='\t', header=None)
            df_features[f'{nombre_sensor}_mean'] = data_sensor.mean(axis=1)
            df_features[f'{nombre_sensor}_std']  = data_sensor.std(axis=1)
            df_features[f'{nombre_sensor}_min']  = data_sensor.min(axis=1)
            df_features[f'{nombre_sensor}_max']  = data_sensor.max(axis=1)

    if df_features.empty or not os.path.exists(ruta_profile):
        print("❌ Error: No se encontraron los archivos de datos en 'data/'. Abortando.")
        return

    df_profile = pd.read_csv(ruta_profile, sep='\t', header=None)
    y = df_profile[0].astype('category').cat.codes.values
    X = df_features.values

    # Extraer exactamente la misma partición de test que en el entrenamiento
    _, X_test_raw, _, _ = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_test = pd.DataFrame(X_test_raw, columns=df_features.columns)

    # Cargar modelo serializado
    path_modelo = os.path.join(MODELS_DIR, 'modelo_random_forest.pkl')
    if not os.path.exists(path_modelo):
        print(f"❌ Error: No existe el modelo en '{path_modelo}'. Ejecuta primero 02_entrenamiento_modelos.py")
        return

    modelo_ganador = joblib.load(path_modelo)

    # ==============================================================================
    # 2. FIGURA 6.1: IMPORTANCIA GLOBAL DE VARIABLES (GINI)
    # ==============================================================================
    importances = modelo_ganador.feature_importances_
    indices = np.argsort(importances)[::-1][:10]

    plt.figure(figsize=(9, 5))
    plt.barh(range(10), importances[indices][::-1], align='center', color='#1f77b4')
    plt.yticks(range(10), [df_features.columns[i] for i in indices][::-1], fontsize=10, fontweight='bold')
    plt.xlabel('Importancia Relativa (Impureza de Gini)', fontsize=11)
    plt.title('Top 10 Variables Predictoras Globales (Random Forest)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    fig1_path = os.path.join(FIGURES_DIR, 'figura_6_1_feature_importance.png')
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"📸 Guardada '{fig1_path}'.")

    # ==============================================================================
    # 3. FIGURA 6.2: EXPLICACIÓN LOCAL CON SHAP (BEESWARM)
    # ==============================================================================
    print("⏳ Generando valores SHAP y gráfico Beeswarm...")
    explainer = shap.TreeExplainer(modelo_ganador)
    shap_values = explainer(X_test)

    # Seleccionar la clase correspondiente a Estado Crítico / Fallo
    if len(shap_values.shape) == 3:
        shap_vals_class = shap_values[:, :, 0]
    else:
        shap_vals_class = shap_values

    plt.close('all')
    shap.plots.beeswarm(shap_vals_class, max_display=10, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 5)
    plt.title('Impacto Local de Sensórica en Estado Crítico/Fallo (SHAP)', fontsize=11, fontweight='bold', pad=12)
    plt.tight_layout()
    
    fig2_path = os.path.join(FIGURES_DIR, 'figura_6_2_shap_local.png')
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📸 Guardada '{fig2_path}' con éxito.")

if __name__ == '__main__':
    main()