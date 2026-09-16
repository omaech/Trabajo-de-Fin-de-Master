import os
import joblib
import pandas as pd
import numpy as np

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')

    path_modelo = os.path.join(MODELS_DIR, 'modelo_random_forest.pkl')

    if not os.path.exists(path_modelo):
        print(f"❌ Error: No se encuentra el modelo en '{path_modelo}'. Asegúrate de ejecutar 02_entrenamiento_modelos.py primero.")
        return

    modelo = joblib.load(path_modelo)

    print("🔍 INSPECCIÓN DEL MODELO CARGADO:")
    print(f"  * Clases detectadas por el modelo (classes_): {modelo.classes_}")

    # Cargar características
    archivos_sensores = ['PS1', 'PS5', 'PS6', 'FS1', 'FS2', 'TS1', 'TS4', 'VS1', 'EPS1', 'CE', 'CP']
    
    df_features = pd.DataFrame()
    for s in archivos_sensores:
        archivo = os.path.join(DATA_DIR, f"{s}.txt")
        if os.path.exists(archivo):
            data = pd.read_csv(archivo, sep='\t', header=None)
            df_features[f'{s}_mean'] = data.mean(axis=1)
            df_features[f'{s}_std']  = data.std(axis=1)
            df_features[f'{s}_min']  = data.min(axis=1)
            df_features[f'{s}_max']  = data.max(axis=1)

    ruta_profile = os.path.join(DATA_DIR, 'profile.txt')
    if not os.path.exists(ruta_profile):
        print("❌ Error: No se encontró 'profile.txt' en la carpeta data.")
        return

    df_profile = pd.read_csv(ruta_profile, sep=r'\s+', header=None)
    y_real = df_profile[0].values

    print("\n🧪 PREDICCIONES BRUTAS DE SAMPLES REALES (SIN MAPEAR):")
    for target in [3, 20, 100]:
        idxs = np.where(y_real == target)[0]
        if len(idxs) > 0:
            idx = idxs[0]
            sample = df_features.iloc[idx].values.reshape(1, -1)
            pred_raw = modelo.predict(sample)[0]
            probs_raw = modelo.predict_proba(sample)[0]
            print(f"  * Ciclo real de {target:>3}% -> Output directo de predict(): {pred_raw} | Probabilidades: {probs_raw}")

if __name__ == '__main__':
    main()