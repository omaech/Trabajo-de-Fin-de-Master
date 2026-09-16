import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_and_prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y valida la matriz de características antes de la división/escalado.
    Elimina nulos si existieran y asegura el formato numérico.
    """
    df_clean = df.copy()
    df_clean = df_clean.dropna()
    return df_clean

def split_and_scale(
    df: pd.DataFrame, 
    target_col: str = 'target', 
    test_size: float = 0.2, 
    random_state: int = 42
):
    """
    Realiza la división estratificada Train/Test y ajusta el StandardScaler 
    exclusivamente sobre el conjunto de entrenamiento para evitar Data Leakage.
    
    Retorna:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # División estratificada según las clases del objetivo
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )

    # Ajuste e inferencia del escalador
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convertir de nuevo a DataFrames conservando los nombres de columnas
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler