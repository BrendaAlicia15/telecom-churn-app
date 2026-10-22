import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Importamos nuestra función del módulo de preprocesamiento
from preprocessing import cargar_y_limpiar_datos

def entrenar_modelo():
    ruta_dataset = "data/telco.csv"
    if not os.path.exists(ruta_dataset):
        print(f"⚠️ No se encontró el dataset en {ruta_dataset}.")
        return

    print("Cargando y limpiando datos para el entrenamiento...")
    df = cargar_y_limpiar_datos(ruta_dataset)
    
    # 1. Codificar todas las variables categóricas restantes a numéricas
    cat_columns = df.select_dtypes(include=['object']).columns
    for col in cat_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        
    # 2. Separar variables predictoras (X) y objetivo (y)
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # 3. División train/test (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Entrenar el modelo (Random Forest)
    print("Entrenando el modelo Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluar precisión
    score = model.score(X_test, y_test)
    print(f"¡Modelo entrenado con éxito! Precisión (Accuracy) en test: {score * 100:.2f}%")
    
    # 5. Guardar el modelo y las columnas exactas (crucial para Streamlit)
    os.makedirs('src', exist_ok=True)
    joblib.dump(model, 'src/modelo_churn.pkl')
    joblib.dump(list(X.columns), 'src/columnas_modelo.pkl')
    print("✨ Modelo y estructura de columnas guardados exitosamente en la carpeta 'src/'.")

if __name__ == "__main__":
    entrenar_modelo()