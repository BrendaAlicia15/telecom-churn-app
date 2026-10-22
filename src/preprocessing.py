# Script inicial de Limpieza y Preprocesamiento

import pandas as pd
import numpy as np

def cargar_y_limpiar_datos(filepath):
    # Cargar el dataset
    df = pd.read_csv(filepath)
    
    print("Columnas encontradas en el CSV:", df.columns.tolist())
    
    # 1. Eliminar identificadores que no aportan al modelo
    if 'Customer ID' in df.columns:
        df = df.drop(columns=['Customer ID'])
        
    # 2. Limpiar la columna 'Total Charges' (convertir a numérico y rellenar nulos)
    if 'Total Charges' in df.columns:
        df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
        df['Total Charges'] = df['Total Charges'].fillna(df['Total Charges'].median())
    
    # 3. Definir la variable objetivo (Target)
    # En este dataset extendido, 'Customer Status' suele tener valores como 'Stayed', 'Churned', 'Joined'
    # Vamos a predecir si es 'Churned' (1) o no (0). O podemos usar 'Churn Label' si está disponible.
    if 'Customer Status' in df.columns:
        # Convertimos 'Churned' a 1, y cualquier otro estado ('Stayed', 'Joined') a 0
        df['Churn'] = df['Customer Status'].apply(lambda x: 1 if x == 'Churned' else 0)
        # Eliminamos columnas redundantes de estado o churn para evitar fuga de datos (data leakage)
        df = df.drop(columns=['Customer Status', 'Churn Label', 'Churn Score', 'Churn Category', 'Churn Reason'], errors='ignore')
    elif 'Churn Label' in df.columns:
        df['Churn'] = df['Churn Label'].apply(lambda x: 1 if x == 'Yes' else 0)
        
    print(f"Datos limpiados con éxito. Forma final del dataset: {df.shape}")
    return df

if __name__ == "__main__":
    ruta_dataset = "data/telco.csv"
    df_limpio = cargar_y_limpiar_datos(ruta_dataset)
    print(df_limpio.head(3))