import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Configuración de estilo gráfico
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def main():
    # ==============================================================================
    # 0. CONFIGURACIÓN DE RUTAS RELATIVAS Y CARPETAS DE SALIDA
    # ==============================================================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')

    # Crear carpetas de salida si no existen
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ==============================================================================
    # 1. CARGA DE DATOS Y PREPARACIÓN DE MATRICES
    # ==============================================================================
    print("⏳ Cargando datos y generando características para modelado...")

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
        else:
            print(f"⚠️ Advertencia: No se encontró el archivo {archivo}")

    if df_features.empty or not os.path.exists(ruta_profile):
        print("❌ Error: Faltan archivos de datos en la carpeta 'data/'. Abortando.")
        return

    # Cargar variable objetivo (Cooler Condition) y codificar a (0, 1, 2)
    df_profile = pd.read_csv(ruta_profile, sep='\t', header=None)
    y = df_profile[0].astype('category').cat.codes.values
    X = df_features.values

    # ==============================================================================
    # 2. DIVISIÓN DE DATOS Y ESCALADO (PREVENCIÓN DE DATA LEAKAGE)
    # ==============================================================================
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    print(f"✅ Datos divididos con estratificación de clases:")
    print(f"   - Conjunto Entrenamiento (Train): {X_train_raw.shape[0]} muestras")
    print(f"   - Conjunto Prueba (Test):        {X_test_raw.shape[0]} muestras")

    # Normalización: fit solo sobre train
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # ==============================================================================
    # 3. OPTIMIZACIÓN DE HIPERPARÁMETROS CON GRIDSEARCHCV
    # ==============================================================================
    print("\n🔍 Iniciando optimización de hiperparámetros (GridSearchCV)...")

    param_grids = {
        'SVM (RBF Kernel)': (
            SVC(probability=True),
            {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['rbf']}
        ),
        'Random Forest': (
            RandomForestClassifier(random_state=42),
            {'n_estimators': [50, 100], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}
        ),
        'KNN': (
            KNeighborsClassifier(),
            {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}
        ),
        'XGBoost': (
            XGBClassifier(eval_metric='mlogloss', random_state=42),
            {'n_estimators': [50, 100], 'max_depth': [3, 6], 'learning_rate': [0.01, 0.1]}
        )
    }

    resultados = []
    mejores_modelos = {}

    for nombre_modelo, (model, grid) in param_grids.items():
        print(f"   -> Entrenando y evaluando {nombre_modelo}...")
        clf = GridSearchCV(model, grid, cv=5, scoring='f1_macro', n_jobs=-1)
        clf.fit(X_train, y_train)
        
        best_model = clf.best_estimator_
        mejores_modelos[nombre_modelo] = best_model
        
        start_time = time.time()
        y_pred = best_model.predict(X_test)
        end_time = time.time()
        
        tiempo_inferencia_ms = ((end_time - start_time) / len(X_test)) * 1000
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        
        resultados.append({
            'Modelo': nombre_modelo,
            'Accuracy': round(acc, 4),
            'Precision (Macro)': round(prec, 4),
            'Recall (Macro)': round(rec, 4),
            'F1-Score (Macro)': round(f1, 4),
            'Tiempo Inferencia (ms/ciclo)': round(tiempo_inferencia_ms, 3)
        })

    # ==============================================================================
    # 4. TABLA COMPARATIVA
    # ==============================================================================
    df_resultados = pd.DataFrame(resultados).sort_values(by='F1-Score (Macro)', ascending=False)

    print("\n" + "="*80)
    print("--- TABLA COMPARATIVA DE MODELOS EN TEST (CAPÍTULO 5.4) ---")
    print("="*80)
    print(df_resultados.to_string(index=False))
    print("="*80)

    # ==============================================================================
    # 5. MATRIZ DE CONFUSIÓN Y EXPORTACIÓN DEL MODELO Y SCALER
    # ==============================================================================
    modelo_ganador_nombre = df_resultados.iloc[0]['Modelo']
    modelo_ganador = mejores_modelos[modelo_ganador_nombre]
    y_pred_ganador = modelo_ganador.predict(X_test)

    plt.figure(figsize=(7, 5))
    cm = confusion_matrix(y_test, y_pred_ganador)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', 
        xticklabels=['Fallo (3%)', 'Degradado (20%)', 'Óptimo (100%)'],
        yticklabels=['Fallo (3%)', 'Degradado (20%)', 'Óptimo (100%)']
    )

    plt.title(f'Matriz de Confusión - Modelo Ganador ({modelo_ganador_nombre})', fontsize=12, fontweight='bold')
    plt.xlabel('Predicción del Modelo')
    plt.ylabel('Estado Real del Enfriador')
    plt.tight_layout()
    
    fig_path = os.path.join(FIGURES_DIR, 'figura_5_1_matriz_confusion.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"\n📸 Guardada '{fig_path}'.")

    # Exportar artefactos del modelo
    path_modelo = os.path.join(MODELS_DIR, 'modelo_random_forest.pkl')
    path_scaler = os.path.join(MODELS_DIR, 'scaler.pkl')

    joblib.dump(modelo_ganador, path_modelo)
    joblib.dump(scaler, path_scaler)

    print(f"💾 Modelo óptimo exportado a: '{path_modelo}'")
    print(f"💾 Escalador (StandardScaler) exportado a: '{path_scaler}'")

if __name__ == '__main__':
    main()