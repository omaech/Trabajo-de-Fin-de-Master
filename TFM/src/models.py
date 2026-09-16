import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold

def get_candidate_models():
    """
    Devuelve un diccionario con la batería de modelos a comparar 
    para la memoria del TFM.
    """
    svm_base = SVC(C=10, kernel='rbf', gamma='scale', probability=True)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes': GaussianNB(),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'SVM (RBF)': svm_base,
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Bagging (SVM)': BaggingClassifier(estimator=svm_base, n_estimators=10, random_state=42),
        'Stacking Classifier': StackingClassifier(
            estimators=[
                ('svm', svm_base),
                ('dt', DecisionTreeClassifier(max_depth=5, random_state=42)),
                ('knn', KNeighborsClassifier(n_neighbors=5))
            ],
            final_estimator=LogisticRegression(C=0.1, random_state=42),
            passthrough=True
        )
    }
    return models

def evaluate_models_cv(models: dict, X: np.ndarray, y: pd.Series, cv_folds: int = 5) -> pd.DataFrame:
    """
    Evalúa todos los modelos mediante Stratified K-Fold Cross Validation
    y devuelve un DataFrame con el ranking completo y tiempos de ejecución.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = []

    for name, model in models.items():
        start_time = time.time()
        
        # Validación cruzada multiclase
        cv_results = cross_validate(
            model, X, y, cv=skf, 
            scoring=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'],
            n_jobs=-1
        )
        
        elapsed_time = time.time() - start_time
        
        results.append({
            'Modelo': name,
            'Accuracy Mean': cv_results['test_accuracy'].mean(),
            'Accuracy Std': cv_results['test_accuracy'].std(),
            'F1-Score (Macro)': cv_results['test_f1_macro'].mean(),
            'Precision (Macro)': cv_results['test_precision_macro'].mean(),
            'Recall (Macro)': cv_results['test_recall_macro'].mean(),
            'Tiempo Entren. (s)': elapsed_time
        })

    df_results = pd.DataFrame(results).sort_values(by='F1-Score (Macro)', ascending=False)
    return df_results

def get_feature_importances(X_df: pd.DataFrame, y: pd.Series, top_n: int = None) -> pd.DataFrame:
    """
    Calcula la importancia de las variables usando Random Forest 
    para identificar los sensores más críticos en el proceso.
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_df, y)
    
    importances = pd.DataFrame({
        'Sensor_Metrica': X_df.columns,
        'Importancia': rf.feature_importances_
    }).sort_values(by='Importancia', ascending=False)
    
    if top_n is not None:
        return importances.head(top_n)
        
    return importances