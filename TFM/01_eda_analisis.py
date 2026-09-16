import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo global para gráficos académicos
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'

def main():
    # ==============================================================================
    # 0. CONFIGURACIÓN DE RUTAS RELATIVAS Y CARPETAS DE SALIDA
    # ==============================================================================
    # Define la ruta raíz del proyecto respecto a este script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')

    # Crear directorio para guardar figuras si no existe
    os.makedirs(FIGURES_DIR, exist_ok=True)

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

    # ==============================================================================
    # 1. CARGA Y CREACIÓN DEL DATAFRAME COMPLETO
    # ==============================================================================
    df_features = pd.DataFrame()
    print("⏳ Cargando archivos de sensores y calculando métricas agregadas...")

    archivos_encontrados = False
    for nombre_sensor, archivo in archivos_sensores.items():
        if os.path.exists(archivo):
            archivos_encontrados = True
            data_sensor = pd.read_csv(archivo, sep='\t', header=None)
            df_features[f'{nombre_sensor}_mean'] = data_sensor.mean(axis=1)
            df_features[f'{nombre_sensor}_std']  = data_sensor.std(axis=1)
            df_features[f'{nombre_sensor}_min']  = data_sensor.min(axis=1)
            df_features[f'{nombre_sensor}_max']  = data_sensor.max(axis=1)
        else:
            print(f"⚠️ Advertencia: No se encontró el archivo {archivo}")

    if not archivos_encontrados:
        print("❌ Error: No se encontraron los archivos en la carpeta '/data'. Comprueba que existen.")
        return

    # Cargar etiquetas del sistema
    if os.path.exists(ruta_profile):
        df_profile = pd.read_csv(ruta_profile, sep='\t', header=None)
        df_features['Cooler_Condition'] = df_profile[0]
        df_features['Valve_Condition'] = df_profile[1]
        df_features['Internal_Leakage'] = df_profile[2]

    print(f"✅ Dataset construido correctamente: {df_features.shape[0]} muestras y {df_features.shape[1]} columnas.")

    # ==============================================================================
    # GRÁFICA 1: HISTOGRAMAS Y DISTRIBUCIONES (EDA 4.1)
    # ==============================================================================
    fig1, axes1 = plt.subplots(2, 2, figsize=(11, 7))
    fig1.suptitle('Distribución Estadística de Variables Críticas de la Prensa', fontsize=13, fontweight='bold')

    if 'PS1_mean' in df_features.columns:
        sns.histplot(df_features['PS1_mean'], kde=True, ax=axes1[0, 0], color='#1f77b4')
        axes1[0, 0].set_title('Presión Principal (PS1_mean) - [bar]')

    if 'CE_mean' in df_features.columns:
        sns.histplot(df_features['CE_mean'], kde=True, ax=axes1[0, 1], color='#2ca02c')
        axes1[0, 1].set_title('Eficiencia Enfriador (CE_mean) - [%]')

    if 'TS1_mean' in df_features.columns:
        sns.histplot(df_features['TS1_mean'], kde=True, ax=axes1[1, 0], color='#d62728')
        axes1[1, 0].set_title('Temperatura Aceite (TS1_mean) - [°C]')

    if 'VS1_mean' in df_features.columns:
        sns.histplot(df_features['VS1_mean'], kde=True, ax=axes1[1, 1], color='#9467bd')
        axes1[1, 1].set_title('Vibración (VS1_mean) - [mm/s]')

    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, 'figura_4_1_distribuciones.png')
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"📸 Guardada '{fig1_path}'")

    # ==============================================================================
    # GRÁFICA 2: MATRIZ DE CORRELACIÓN DE PEARSON (EDA 4.2)
    # ==============================================================================
    vars_correlacion = [
        'PS1_mean', 'PS5_mean', 'FS1_mean', 'TS1_mean', 
        'TS4_max', 'CE_mean', 'CP_min', 'EPS1_mean', 'VS1_mean'
    ]
    vars_corr_existentes = [v for v in vars_correlacion if v in df_features.columns]

    if vars_corr_existentes:
        fig2 = plt.figure(figsize=(9, 7))
        corr_matrix = df_features[vars_corr_existentes].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(
            corr_matrix, 
            mask=mask,
            annot=True, 
            fmt=".2f", 
            cmap='coolwarm', 
            vmax=1, 
            vmin=-1, 
            center=0,
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .8}
        )

        plt.title('Matriz de Correlación de Pearson (Variables Clave)', fontsize=13, fontweight='bold', pad=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        fig2_path = os.path.join(FIGURES_DIR, 'figura_4_2_matriz_correlacion.png')
        plt.savefig(fig2_path, dpi=300)
        plt.close()
        print(f"📸 Guardada '{fig2_path}'")

    # ==============================================================================
    # GRÁFICA 3: BOXPLOTS SEGÚN ESTADO DE SALUD DEL ENFRIADOR
    # ==============================================================================
    if 'Cooler_Condition' in df_features.columns:
        fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
        fig3.suptitle('Impacto del Estado del Enfriador en Parámetros Térmicos y de Eficiencia', fontsize=13, fontweight='bold')

        sns.boxplot(x='Cooler_Condition', y='TS1_mean', data=df_features, ax=axes3[0], palette='Reds')
        axes3[0].set_title('Temperatura Media del Aceite (TS1) vs Estado Enfriador')
        axes3[0].set_xlabel('Estado Enfriador (3=Fallo, 20=Degradado, 100=Óptimo)')
        axes3[0].set_ylabel('Temperatura [°C]')

        sns.boxplot(x='Cooler_Condition', y='CE_mean', data=df_features, ax=axes3[1], palette='Greens')
        axes3[1].set_title('Eficiencia Enfriador (CE) vs Estado Enfriador')
        axes3[1].set_xlabel('Estado Enfriador (3=Fallo, 20=Degradado, 100=Óptimo)')
        axes3[1].set_ylabel('Eficiencia [%]')

        plt.tight_layout()
        fig3_path = os.path.join(FIGURES_DIR, 'figura_4_3_boxplots_falla.png')
        plt.savefig(fig3_path, dpi=300)
        plt.close()
        print(f"📸 Guardada '{fig3_path}'")

    # ==============================================================================
    # GRÁFICA 4: EVOLUCIÓN TEMPORAL / DERIVA TÉRMICA
    # ==============================================================================
    if 'TS1_mean' in df_features.columns and 'TS4_max' in df_features.columns:
        fig4 = plt.figure(figsize=(11, 4))
        plt.plot(df_features['TS1_mean'].iloc[:200], label='Temperatura Aceite (TS1_mean)', color='#d62728', linewidth=1.5)
        plt.plot(df_features['TS4_max'].iloc[:200], label='Temperatura Máx (TS4_max)', color='#ff7f0e', linestyle='--', linewidth=1.5)

        plt.title('Inercia y Deriva Térmica en los Primeros 200 Ciclos de Prensado', fontsize=13, fontweight='bold')
        plt.xlabel('Número de Ciclo de Trabajo')
        plt.ylabel('Temperatura [°C]')
        plt.legend(loc='upper left')
        plt.tight_layout()
        fig4_path = os.path.join(FIGURES_DIR, 'figura_4_4_deriva_temporal.png')
        plt.savefig(fig4_path, dpi=300)
        plt.close()
        print(f"📸 Guardada '{fig4_path}'")

    print("\n🚀 ¡PROCESO COMPLETO! Las imágenes se han guardado en 'reports/figures/'.")

if __name__ == '__main__':
    main()