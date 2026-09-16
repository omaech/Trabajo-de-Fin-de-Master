import os
import requests
import numpy as np
import pandas as pd

# ==============================================================================
# 1. CARGA DE DATOS EN ESTRUCTURA DE 44 CARACTERÍSTICAS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

SENSORES = ['PS1', 'PS5', 'PS6', 'FS1', 'FS2', 'TS1', 'TS4', 'VS1', 'EPS1', 'CE', 'CP']

df_features = pd.DataFrame()
for s in SENSORES:
    arch = os.path.join(DATA_DIR, f"{s}.txt")
    if os.path.exists(arch):
        data = pd.read_csv(arch, sep='\t', header=None)
        df_features[f'{s}_mean'] = data.mean(axis=1)
        df_features[f'{s}_std']  = data.std(axis=1)
        df_features[f'{s}_min']  = data.min(axis=1)
        df_features[f'{s}_max']  = data.max(axis=1)

# Cargar target real
ruta_profile = os.path.join(DATA_DIR, 'profile.txt')
if not os.path.exists(ruta_profile):
    print("❌ Error: No se encontró 'profile.txt' en la carpeta data.")
    exit()

df_profile = pd.read_csv(ruta_profile, sep=r'\s+', header=None)
y_real = df_profile[0].values

URL = "http://127.0.0.1:8000/predict"

print("=" * 80)
print(" 🚨 PRUEBA DE DIAGNÓSTICO PRESCRIPTIVO CON FALLOS Y DEGRADACIONES REALES")
print("=" * 80)

# ==============================================================================
# 2. PRUEBA DE CICLOS REALES DEL HISTÓRICO VIA API
# ==============================================================================
casos_prueba = [
    (3, "Fallo Crítico (3%)"),
    (20, "Degradado (20%)"),
    (100, "Óptimo (100%)")
]

for estado_target, etiqueta in casos_prueba:
    indices = np.where(y_real == estado_target)[0]
    
    if len(indices) > 0:
        idx = indices[0]
        row = df_features.iloc[idx]
        
        # Construir diccionario con las 44 variables numéricas exactas
        dict_44_features = {col: float(row[col]) for col in df_features.columns}
        
        payload = {
            "ciclo_id": int(idx),
            "features": dict_44_features
        }
        
        try:
            response = requests.post(URL, json=payload, timeout=5)
            if response.status_code == 200:
                r = response.json()
                diag = r.get("diagnostico_xai", {})
                alertas = r.get("motor_alertas", {})
                
                print(f"\n📌 Ciclo Histórico Real #{idx} | Estado Real Esperado: {etiqueta}")
                print(f"   * Estado Predicho API : {r.get('estado_predicho')}")
                print(f"   * Probabilidades      : {r.get('probabilidades')}")
                print(f"   * Salud Estimada      : {r.get('salud_enfriador_pct')}%")
                print(f"   * Causa Raíz (XAI)    : {diag.get('causa_raiz')}")
                print(f"   * Acción Prescriptiva : {diag.get('accion_prescriptiva_recomendada')}")
                print(f"   * Nivel de Alerta     : {alertas.get('etiqueta_alerta')} (Nivel {alertas.get('nivel_alerta')})")
            else:
                print(f"\n❌ Error HTTP {response.status_code} en Ciclo #{idx}: {response.text}")
        except Exception as e:
            print(f"  ❌ Error conectando con la API: {e}")

print("\n" + "=" * 80)
print(" ✅ PRUEBA FINALIZADA")
print("=" * 80)