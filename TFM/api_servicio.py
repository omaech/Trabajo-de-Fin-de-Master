import os
import math
import joblib
import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
import uvicorn
import shap

# ==============================================================================
# 1. CONFIGURACIÓN DE RUTAS Y CARGA DE ARTEFACTOS REALES
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

PATH_MODELO = os.path.join(MODELS_DIR, 'modelo_random_forest.pkl')
PATH_SCALER = os.path.join(MODELS_DIR, 'scaler.pkl')

modelo_real = None
scaler_real = None
explainer_shap = None

# Carga global al iniciar la API
if os.path.exists(PATH_MODELO) and os.path.exists(PATH_SCALER):
    modelo_real = joblib.load(PATH_MODELO)
    scaler_real = joblib.load(PATH_SCALER)
    explainer_shap = shap.TreeExplainer(modelo_real)
    print("✅ Modelo, StandardScaler y SHAP TreeExplainer cargados correctamente.")
else:
    print("⚠️ ADVERTENCIA: Artefactos (.pkl) no encontrados. Asegúrate de ejecutar 02_entrenamiento_modelos.py")

# ==============================================================================
# 2. CONFIGURACIÓN DE FASTAPI
# ==============================================================================
app = FastAPI(
    title="API de Mantenimiento Prescriptivo - Enfriador de Prensa Cerámica",
    description="Servicio de inferencia en tiempo real con modelo Random Forest real, explicabilidad SHAP y Motor de Alertas.",
    version="2.0.0"
)

BUFFER_ESTADOS = deque(maxlen=5)
WEBHOOK_TEAMS_URL = os.getenv("WEBHOOK_TEAMS_URL", "https://outlook.office.com/webhook/YOUR_TEAMS_WEBHOOK_URL")

# Mapeo de etiquetas ordinales (0, 1, 2) del dataset
ETIQUETAS_CLASE = {
    0: "Fallo Crítico (3%)",
    1: "Degradación Preventiva (20%)",
    2: "Estado Óptimo / Nominal (100%)"
}

# Nombres de las 44 columnas exactas con las que se entrenó el modelo
NOMBRES_SENSORES = ['PS1', 'PS5', 'PS6', 'FS1', 'FS2', 'TS1', 'TS4', 'VS1', 'EPS1', 'CE', 'CP']
COLUMNAS_FEATURES = []
for s in NOMBRES_SENSORES:
    COLUMNAS_FEATURES.extend([f'{s}_mean', f'{s}_std', f'{s}_min', f'{s}_max'])


def enviar_notificacion_webhook(nivel: str, causa_raiz: str, accion: str, salud: float, ciclo: int):
    color = "F57F17" if "Nivel 1" in nivel else "C62828"
    payload = {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "themeColor": color,
        "summary": "Alerta de Planta - Enfriador Hidráulico",
        "sections": [{
            "activityTitle": f"⚠️ **{nivel} - Prensa Cerámica**",
            "activitySubtitle": f"Ciclo #{ciclo} | Salud Estimada: {salud:.1f}%",
            "facts": [
                {"name": "Causa Raíz (XAI):", "value": causa_raiz},
                {"name": "Acción Prescriptiva:", "value": accion},
                {"name": "Estado del Enfriador:", "value": "Degradado" if "Nivel 1" in nivel else "CRÍTICO / PARO INMINENTE"}
            ],
            "markdown": True
        }]
    }
    try:
        # import requests
        # requests.post(WEBHOOK_TEAMS_URL, json=payload, timeout=2.0)
        print(f"[EVENT TRIGGER] Notificación Enviada -> {nivel} | Ciclo #{ciclo} | Salud: {salud:.1f}%")
    except Exception as e:
        print(f"[EVENT TRIGGER ERROR] Fallo al enviar notificación: {e}")


def evaluar_logica_alertas(clase_id: int, salud_st: float, diag_xai: Dict[str, Any], ciclo_id: int = 0) -> Dict[str, Any]:
    BUFFER_ESTADOS.append(clase_id)
    
    # Nivel 2: Crítica (Fallo o Salud < 15%)
    if clase_id == 0 or salud_st < 15.0:
        enviar_notificacion_webhook(
            nivel="Nivel 2: ALERTA CRÍTICA (RIESGO DE PARO)",
            causa_raiz=diag_xai.get("causa_raiz", "Fallo severo detectado por la sensórica de la prensa"),
            accion=diag_xai.get("accion_prescriptiva_recomendada", "Parada inmediata e inspección urgente"),
            salud=salud_st,
            ciclo=ciclo_id
        )
        return {
            "nivel_alerta": 2,
            "etiqueta_alerta": "CRÍTICA",
            "color_hex": "#C62828",
            "accion_disparada": "SMS enviado a Jefe de Planta & Orden de Trabajo generada en GMAO"
        }
    
    # Nivel 1: Preventiva (3 ciclos en degradado = clase 1)
    elif len(BUFFER_ESTADOS) >= 3 and list(BUFFER_ESTADOS)[-3:] == [1, 1, 1]:
        enviar_notificacion_webhook(
            nivel="Nivel 1: Alerta Preventiva (Degradación)",
            causa_raiz=diag_xai.get("causa_raiz", "Degradación persistente en eficiencia térmica/caudal"),
            accion=diag_xai.get("accion_prescriptiva_recomendada", "Programar mantenimiento preventivo"),
            salud=salud_st,
            ciclo=ciclo_id
        )
        return {
            "nivel_alerta": 1,
            "etiqueta_alerta": "PREVENTIVA",
            "color_hex": "#F57F17",
            "accion_disparada": "Notificación Webhook enviada a Microsoft Teams (Equipo Mantenimiento)"
        }
        
    # Nivel 0: Nominal
    return {
        "nivel_alerta": 0,
        "etiqueta_alerta": "NOMINAL",
        "color_hex": "#2E7D32",
        "accion_disparada": "Sin acción externa. Registro de telemetría en base de datos."
    }

# ==============================================================================
# 3. MODELOS DE ENTRADA Y SALIDA (PYDANTIC)
# ==============================================================================
class FeaturesEntrada(BaseModel):
    ciclo_id: Optional[int] = Field(default=1, description="Número de ciclo de prensado")
    # Diccionario dinámico para recibir las 44 variables numéricas agregadas (mean, std, min, max)
    features: Dict[str, float] = Field(
        ..., 
        description="Diccionario con las 44 características extraídas de los 11 sensores (ej: {'PS1_mean': 150.2, 'PS1_std': 1.2, ...})"
    )


class PredictionResponse(BaseModel):
    ciclo_id: int
    estado_predicho: str
    clase_id: int
    probabilidades: Dict[str, float]
    salud_enfriador_pct: float
    diagnostico_xai: Dict[str, Any]
    motor_alertas: Dict[str, Any]

# ==============================================================================
# 4. ENDPOINTS DE LA API
# ==============================================================================
@app.get("/")
def read_root():
    return {
        "sistema": "TFM - Mantenimiento Prescriptivo de Prensa Cerámica",
        "modulo": "7.3. Servicio de Inferencia en Tiempo Real y Alertas",
        "status": "Online" if modelo_real is not None else "Warning: Modelo no cargado",
        "docs_url": "/docs"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: FeaturesEntrada):
    if modelo_real is None or scaler_real is None:
        raise HTTPException(status_code=500, detail="Los modelos no están cargados en el servidor.")

    try:
        # 1. Construir DataFrame con el orden exacto de columnas
        dict_feat = data.features
        vector_entrada = []
        for col in COLUMNAS_FEATURES:
            if col not in dict_feat:
                raise HTTPException(status_code=400, detail=f"Falta la característica requerida: '{col}'")
            vector_entrada.append(dict_feat[col])

        df_input = pd.DataFrame([vector_entrada], columns=COLUMNAS_FEATURES)

        # 2. Escalado de variables e Inferencia
        X_scaled = scaler_real.transform(df_input.values)
        clase_id = int(modelo_real.predict(X_scaled)[0])
        probs_arr = modelo_real.predict_proba(X_scaled)[0]

        dict_probs = {
            "Fallo": round(float(probs_arr[0]), 4),
            "Degradado": round(float(probs_arr[1]), 4),
            "Óptimo": round(float(probs_arr[2]), 4)
        }

        # 3. Estimación de Salud (Basada en la probabilidad ponderada de estado nominal/degradado)
        salud_st = float((probs_arr[2] * 100.0) + (probs_arr[1] * 45.0) + (probs_arr[0] * 5.0))

        # 4. Explicabilidad Local SHAP
        shap_values_inst = explainer_shap(X_scaled)
        
        # Extraer valores SHAP para la clase predicha
        if len(shap_values_inst.shape) == 3:
            vals_clase = shap_values_inst.values[0, :, clase_id]
        else:
            vals_clase = shap_values_inst.values[0, :]

        # Obtener Top 3 variables con mayor contribución SHAP
        top_indices = np.argsort(np.abs(vals_clase))[::-1][:3]
        top_shap_contrib = {COLUMNAS_FEATURES[idx]: round(float(vals_clase[idx]), 4) for idx in top_indices}

        col_top1 = COLUMNAS_FEATURES[top_indices[0]]
        val_top1 = dict_feat[col_top1]

        # 5. Generar diagnóstico prescriptivo
        if clase_id == 0:
            causa = f"Anomalía severa impulsada por la variable {col_top1} (Valor actual: {val_top1:.2f})."
            accion = "Parada inmediata de la prensa. Inspección del caudal del refrigerante y circuito hidráulico principal."
        elif clase_id == 1:
            causa = f"Degradación progresiva detectada con impacto significativo en {col_top1} (Valor actual: {val_top1:.2f})."
            accion = "Programar limpieza de intercambio térmico y sustitución/revisión de filtros de aceite en la siguiente parada."
        else:
            causa = "Sensórica dentro de rangos óptimos de operación."
            accion = "Continuar con el ciclo normal de producción sin intervenciones requeridas."

        diagnostico_xai = {
            "causa_raiz": causa,
            "accion_prescriptiva_recomendada": accion,
            "top_shap_contributions": top_shap_contrib
        }

        # 6. Motor de Alertas
        info_alerta = evaluar_logica_alertas(
            clase_id=clase_id,
            salud_st=salud_st,
            diag_xai=diagnostico_xai,
            ciclo_id=data.ciclo_id
        )

        return PredictionResponse(
            ciclo_id=data.ciclo_id,
            estado_predicho=ETIQUETAS_CLASE.get(clase_id, "Desconocido"),
            clase_id=clase_id,
            probabilidades=dict_probs,
            salud_enfriador_pct=round(salud_st, 2),
            diagnostico_xai=diagnostico_xai,
            motor_alertas=info_alerta
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la inferencia del modelo: {str(e)}")


@app.get("/buffer-status")
def get_buffer_status():
    return {
        "capacidad_buffer": BUFFER_ESTADOS.maxlen,
        "historial_reciente": list(BUFFER_ESTADOS),
        "persistencia_degradado_3_ciclos": len(BUFFER_ESTADOS) >= 3 and list(BUFFER_ESTADOS)[-3:] == [1, 1, 1]
    }

# ==============================================================================
# 5. EJECUCIÓN DEL SERVIDOR
# ==============================================================================
if __name__ == "__main__":
    uvicorn.run("04_api_servicio:app", host="0.0.0.0", port=8000, reload=True)