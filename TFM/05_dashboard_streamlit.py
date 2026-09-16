import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import os
import time

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA (WIDE LAYOUT PARA SALA DE CONTROL)
# ==============================================================================
st.set_page_config(
    page_title="Cuadro de Mando - Prensa Cerámica (TFM)",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .metric-card {
        background-color: #1e222a;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
        text-align: center;
    }
    .status-badge {
        font-size: 22px;
        font-weight: bold;
        padding: 10px 15px;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin-bottom: 15px;
    }
    .status-verde { background-color: #2e7d32; }
    .status-amarillo { background-color: #f57f17; }
    .status-rojo { background-color: #c62828; }
</style>
""", unsafe_allow_html=True)

# URL de la API local y ruta relativa para los datos
API_URL = "http://127.0.0.1:8000/predict"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

SENSORES = ['PS1', 'PS5', 'PS6', 'FS1', 'FS2', 'TS1', 'TS4', 'VS1', 'EPS1', 'CE', 'CP']

# ==============================================================================
# CARGA Y PREPARACIÓN DE DATOS
# ==============================================================================
@st.cache_data
def cargar_datos_historicos():
    df_feat = pd.DataFrame()
    for s in SENSORES:
        arch = os.path.join(DATA_DIR, f"{s}.txt")
        if os.path.exists(arch):
            d = pd.read_csv(arch, sep='\t', header=None)
            df_feat[f'{s}_mean'] = d.mean(axis=1)
            df_feat[f'{s}_std']  = d.std(axis=1)
            df_feat[f'{s}_min']  = d.min(axis=1)
            df_feat[f'{s}_max']  = d.max(axis=1)
            
    ruta_profile = os.path.join(DATA_DIR, 'profile.txt')
    if os.path.exists(ruta_profile):
        df_prof = pd.read_csv(ruta_profile, sep='\t', header=None)
        df_feat['target_real'] = df_prof[0]
    return df_feat

try:
    df_historico = cargar_datos_historicos()
    if df_historico.empty:
        st.error("❌ No se pudieron cargar los datos desde la carpeta 'data/'. Verifica que los archivos .txt estén en su lugar.")
        st.stop()
except Exception as e:
    st.error(f"Error cargando archivos de sensores: {e}")
    st.stop()

total_ciclos = len(df_historico)

# ==============================================================================
# INICIALIZACIÓN DEL ESTADO DE SESIÓN (SESSION STATE)
# ==============================================================================
if 'ciclo_actual' not in st.session_state:
    st.session_state.ciclo_actual = 0

if 'historial_alertas' not in st.session_state:
    st.session_state.historial_alertas = []

# ==============================================================================
# BARRA LATERAL: CONTROL DE PLANTILLA
# ==============================================================================
st.sidebar.image("https://img.icons8.com/color/96/factory.png", width=80)
st.sidebar.title("🎛️ Control de Planta")
st.sidebar.markdown("---")

modo_sim = st.sidebar.radio("Modo de Inspección:", ["Ciclo Específico", "Simulación Continua"])

if modo_sim == "Ciclo Específico":
    ciclo_sel = st.sidebar.slider("Seleccionar Ciclo de Producción:", 0, total_ciclos - 1, st.session_state.ciclo_actual)
    st.session_state.ciclo_actual = ciclo_sel
else:
    st.sidebar.write(f"**Ciclo en ejecución:** #{st.session_state.ciclo_actual}")
    velocidad = st.sidebar.slider("Intervalo de Actualización (s):", 0.1, 2.0, 0.5)
    auto_play = st.sidebar.checkbox("▶️ Iniciar Monitorización en Vivo", value=True)
    
    if st.sidebar.button("🔄 Reiniciar al Ciclo #0"):
        st.session_state.ciclo_actual = 0
        st.session_state.historial_alertas = []
        st.rerun()
        
    ciclo_sel = st.session_state.ciclo_actual

st.sidebar.markdown("---")
st.sidebar.info("💡 **TFM Big Data & AI**\nSistema Prescriptivo de Enfriador Hidráulico para Prensa Cerámica.")

# ==============================================================================
# CONSULTA A LA API EN TIEMPO REAL (PAYLOAD AJUSTADO AL ESQUEMA 44 FEATURES)
# ==============================================================================
fila_ciclo = df_historico.iloc[ciclo_sel]

# Construir diccionario con las 44 características exactas requeridas por la API
dict_44_features = {}
for s in SENSORES:
    dict_44_features[f'{s}_mean'] = float(fila_ciclo.get(f'{s}_mean', 0.0))
    dict_44_features[f'{s}_std']  = float(fila_ciclo.get(f'{s}_std', 0.0))
    dict_44_features[f'{s}_min']  = float(fila_ciclo.get(f'{s}_min', 0.0))
    dict_44_features[f'{s}_max']  = float(fila_ciclo.get(f'{s}_max', 0.0))

payload = {
    "ciclo_id": int(ciclo_sel),
    "features": dict_44_features
}

try:
    response = requests.post(API_URL, json=payload, timeout=3)
    if response.status_code == 200:
        res = response.json()
        estado_predicho = res.get("estado_predicho", "Desconocido")
        clase_id = res.get("clase_id", 2)
        probs = res.get("probabilidades", {"Fallo": 0.0, "Degradado": 0.0, "Óptimo": 1.0})
        diag_xai = res.get("diagnostico_xai", {})
        salud_st = res.get("salud_enfriador_pct", 100.0)
    else:
        st.error(f"⚠️ La API devolvió un código de error {response.status_code}: {response.text}")
        st.stop()
except Exception as e:
    st.error(f"⚠️ No se pudo conectar con la API en {API_URL}. Asegúrate de ejecutar la API primero (`python 04_api_servicio.py`). Error: {e}")
    st.stop()

if clase_id == 2:
    rttm_texto = "> 120 Horas"
elif clase_id == 1:
    rttm_texto = "~ 18 Horas"
else:
    rttm_texto = "¡PARADA INMEDIATA!"

# Registrar alerta en el historial si el estado es Degradado (1) o Fallo (0)
if clase_id != 2:
    nueva_alerta = {
        "Ciclo": f"#{ciclo_sel}",
        "Estado": estado_predicho.upper(),
        "Salud (St)": f"{salud_st:.1f}%",
        "Causa Raíz": diag_xai.get("causa_raiz", "N/A"),
        "Acción Prescriptiva": diag_xai.get("accion_prescriptiva_recomendada", "N/A")
    }
    if not st.session_state.historial_alertas or st.session_state.historial_alertas[0]["Ciclo"] != f"#{ciclo_sel}":
        st.session_state.historial_alertas.insert(0, nueva_alerta)

# ==============================================================================
# PANEL SUPERIOR: KPIs GLOBALES
# ==============================================================================
st.title("🏭 Sala de Control: Monitorización de Prensa Cerámica")
st.markdown(f"**Ciclo Actual de Inspección:** #{ciclo_sel} / {total_ciclos - 1}")

col_kpi1, col_kpi2, col_kpi3 = st.columns(3)

with col_kpi1:
    if clase_id == 2:
        badge_html = f'<div class="status-badge status-verde">🟢 ESTADO: {estado_predicho.upper()}</div>'
    elif clase_id == 1:
        badge_html = f'<div class="status-badge status-amarillo">🟡 ESTADO: {estado_predicho.upper()}</div>'
    else:
        badge_html = f'<div class="status-badge status-rojo">🔴 ESTADO: {estado_predicho.upper()}</div>'
    st.markdown(badge_html, unsafe_allow_html=True)

with col_kpi2:
    st.markdown("<h4 style='text-align: center; color: #aaa; margin-bottom: 0px;'>Índice de Salud (S_t)</h4>", unsafe_allow_html=True)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=salud_st,
        number={'suffix': "%", 'font': {'size': 28, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 15], 'color': "#ff4d4d"},
                {'range': [15, 60], 'color': "#ffaa00"},
                {'range': [60, 100], 'color': "#2ca02c"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 3},
                'thickness': 0.75,
                'value': salud_st
            }
        }
    ))
    fig_gauge.update_layout(
        height=180, 
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_kpi3:
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="color:#aaa; margin:0;">Mantenimiento Prescrito (RTTM)</h4>
        <h2 style="color:#ffffff; margin:10px 0;">{rttm_texto}</h2>
        <p style="color:#888; margin:0;">Basado en tasa de degradación</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==============================================================================
# PANEL CENTRAL: SENSORES CRÍTICOS
# ==============================================================================
st.subheader("📈 Línea Temporal de Sensores Críticos")

window_size = 50
start_idx = max(0, ciclo_sel - window_size)
end_idx = min(total_ciclos, ciclo_sel + window_size + 1)
df_window = df_historico.iloc[start_idx:end_idx].copy()
df_window['ciclo'] = df_window.index

col_g1, col_g2, col_g3 = st.columns(3)

with col_g1:
    fig_ce = px.line(df_window, x='ciclo', y='CE_mean', title="Eficiencia Enfriador (CE_mean)")
    fig_ce.add_vline(x=ciclo_sel, line_dash="dash", line_color="orange")
    fig_ce.add_hline(y=45.0, line_dash="dot", line_color="red", annotation_text="Umbral Crítico")
    fig_ce.update_layout(height=240, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_ce, use_container_width=True)

with col_g2:
    fig_cp = px.line(df_window, x='ciclo', y='CP_mean', title="Capacidad Térmica (CP_mean)")
    fig_cp.add_vline(x=ciclo_sel, line_dash="dash", line_color="orange")
    fig_cp.add_hline(y=1.5, line_dash="dot", line_color="yellow", annotation_text="Aviso Preventivo")
    fig_cp.update_layout(height=240, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_cp, use_container_width=True)

with col_g3:
    fig_ps6 = px.line(df_window, x='ciclo', y='PS6_mean', title="Presión de Retorno (PS6_mean)")
    fig_ps6.add_vline(x=ciclo_sel, line_dash="dash", line_color="orange")
    fig_ps6.add_hline(y=9.5, line_dash="dot", line_color="red", annotation_text="Límite Presión")
    fig_ps6.update_layout(height=240, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_ps6, use_container_width=True)

st.markdown("---")

# ==============================================================================
# PANEL INFERIOR: DIAGNÓSTICO XAI
# ==============================================================================
st.subheader("🔍 Módulo de Diagnóstico Prescriptivo de Causa Raíz (XAI Local)")

col_xai_left, col_xai_right = st.columns([1.2, 1])

with col_xai_left:
    st.markdown("##### 📊 Top Variables Contribuyentes al Riesgo (Impacto SHAP)")
    
    # Extraer las contribuciones dinámicas devueltas por la API
    shap_dict = diag_xai.get("top_shap_contributions", {})
    if shap_dict:
        top_vars = list(shap_dict.keys())
        top_impacts = [abs(v) for v in shap_dict.values()]
    else:
        top_vars = ["CE_min", "CP_min", "PS6_mean"]
        top_impacts = [0.10, 0.05, 0.02]
        
    df_shap = pd.DataFrame({
        "Variable": top_vars,
        "Contribución al Riesgo": top_impacts
    }).sort_values(by="Contribución al Riesgo", ascending=True)

    fig_bar = px.bar(
        df_shap, 
        x="Contribución al Riesgo", 
        y="Variable", 
        orientation='h',
        color="Contribución al Riesgo",
        color_continuous_scale="Reds" if clase_id != 2 else "Greens"
    )
    fig_bar.update_layout(height=220, showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_bar, use_container_width=True)

with col_xai_right:
    st.markdown("##### 🛠️ Diagnóstico y Recomendación Operativa")
    causa = diag_xai.get('causa_raiz', 'Operación Nominal sin anomalías detectadas.')
    accion = diag_xai.get('accion_prescriptiva_recomendada', 'Continuar monitorización rutinaria.')
    
    if clase_id == 0:
        st.error(f"**Causa Raíz:** {causa}")
        st.error(f"**Acción Prescriptiva:** {accion}")
    elif clase_id == 1:
        st.warning(f"**Causa Raíz:** {causa}")
        st.warning(f"**Acción Prescriptiva:** {accion}")
    else:
        st.success(f"**Diagnóstico:** {causa}")
        st.info(f"**Acción Recomendada:** {accion}")

st.markdown("---")

# ==============================================================================
# HISTORIAL DE ALERTAS
# ==============================================================================
st.subheader("📋 Registro Histórico de Eventos y Paradas")

if st.session_state.historial_alertas:
    df_alertas = pd.DataFrame(st.session_state.historial_alertas)
    st.dataframe(
        df_alertas, 
        use_container_width=True, 
        hide_index=True
    )
else:
    st.info("🟢 No se han registrado alertas ni eventos anómalos hasta el momento.")

# ==============================================================================
# BUCLE DE INCREMENTO AUTOMÁTICO EN SIMULACIÓN CONTINUA
# ==============================================================================
if modo_sim == "Simulación Continua" and auto_play:
    time.sleep(velocidad)
    st.session_state.ciclo_actual = (st.session_state.ciclo_actual + 1) % total_ciclos
    st.rerun()