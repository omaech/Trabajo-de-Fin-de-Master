import os
import joblib
import pandas as pd
import numpy as np

def main():
    # ==============================================================================
    # 0. CONFIGURACIÓN DE RUTAS RELATIVAS
    # ==============================================================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')

    path_modelo = os.path.join(MODELS_DIR, 'modelo_random_forest.pkl')
    path_scaler = os.path.join(MODELS_DIR, 'scaler.pkl')

    if not os.path.exists(path_modelo):
        print(f"❌ Error: No se encuentra el modelo en '{path_modelo}'. Asegúrate de ejecutar 02_entrenamiento_modelos.py primero.")
        return

    modelo = joblib.load(path_modelo)
    scaler_guardado = joblib.load(path_scaler) if os.path.exists(path_scaler) else None

    archivos_sensores = ['PS1', 'PS5', 'PS6', 'FS1', 'FS2', 'TS1', 'TS4', 'VS1', 'EPS1', 'CE', 'CP']

    # ==============================================================================
    # 1. RECONSTRUCCIÓN DE FORMAS DE ORDENACIÓN DE COLUMNAS
    # ==============================================================================
    # Forma A: Intercalado por sensor (PS1_mean, PS1_std, PS1_min, PS1_max, PS5_mean...)
    df_forma_a = pd.DataFrame()
    for s in archivos_sensores:
        arch = os.path.join(DATA_DIR, f"{s}.txt")
        if os.path.exists(arch):
            d = pd.read_csv(arch, sep='\t', header=None)
            df_forma_a[f'{s}_mean'] = d.mean(axis=1)
            df_forma_a[f'{s}_std']  = d.std(axis=1)
            df_forma_a[f'{s}_min']  = d.min(axis=1)
            df_forma_a[f'{s}_max']  = d.max(axis=1)

    # Forma B: Agrupado por métrica (Todos los means, todos los stds, etc.)
    df_forma_b = pd.DataFrame()
    for s in archivos_sensores:
        d = pd.read_csv(os.path.join(DATA_DIR, f"{s}.txt"), sep='\t', header=None)
        df_forma_b[f'{s}_mean'] = d.mean(axis=1)
    for s in archivos_sensores:
        d = pd.read_csv(os.path.join(DATA_DIR, f"{s}.txt"), sep='\t', header=None)
        df_forma_b[f'{s}_std'] = d.std(axis=1)
    for s in archivos_sensores:
        d = pd.read_csv(os.path.join(DATA_DIR, f"{s}.txt"), sep='\t', header=None)
        df_forma_b[f'{s}_min'] = d.min(axis=1)
    for s in archivos_sensores:
        d = pd.read_csv(os.path.join(DATA_DIR, f"{s}.txt"), sep='\t', header=None)
        df_forma_b[f'{s}_max'] = d.max(axis=1)

    # Cargar etiquetas
    ruta_profile = os.path.join(DATA_DIR, 'profile.txt')
    if not os.path.exists(ruta_profile):
        print("❌ Error: No se encontró 'profile.txt' en la carpeta data.")
        return

    df_profile = pd.read_csv(ruta_profile, sep=r'\s+', header=None)
    y_real = df_profile[0].values

    # ==============================================================================
    # 2. EVALUACIÓN CON SCALER GUARDADO O AJUSTADO
    # ==============================================================================
    print("🧪 PRUEBA DE DIAGNÓSTICO DE ORDEN DE CARACTERÍSTICAS:")
    
    for target, label in [(3, "Fallo"), (20, "Degradado"), (100, "Óptimo")]:
        idxs = np.where(y_real == target)[0]
        if len(idxs) == 0:
            continue
        idx = idxs[0]

        # Inferencia Forma A
        if scaler_guardado is not None:
            X_a = scaler_guardado.transform(df_forma_a.iloc[[idx]].values)
        else:
            X_a = df_forma_a.iloc[[idx]].values

        pred_a = modelo.predict(X_a)[0]
        prob_a = modelo.predict_proba(X_a)[0]

        # Inferencia Forma B
        if scaler_guardado is not None:
            X_b = scaler_guardado.transform(df_forma_b.iloc[[idx]].values)
        else:
            X_b = df_forma_b.iloc[[idx]].values

        pred_b = modelo.predict(X_b)[0]
        prob_b = modelo.predict_proba(X_b)[0]

        print(f"\n📌 Ciclo Real {label} ({target}%):")
        print(f"   * Forma A (Intercalada por sensor) -> Predicción: {pred_a} | Probs: {prob_a}")
        print(f"   * Forma B (Agrupada por métrica)  -> Predicción: {pred_b} | Probs: {prob_b}")

if __name__ == '__main__':
    main()