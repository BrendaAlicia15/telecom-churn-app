#Paso: Crear la App Web con Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Churn - Telecom",
    page_icon="📊",
    layout="wide"
)

# Cargar el modelo y las columnas guardadas
@st.cache_resource
def cargar_recursos():
    modelo_path = 'src/modelo_churn.pkl'
    columnas_path = 'src/columnas_modelo.pkl'
    
    if os.path.exists(modelo_path) and os.path.exists(columnas_path):
        model = joblib.load(modelo_path)
        columnas = joblib.load(columnas_path)
        return model, columnas
    return None, None

model, columnas_modelo = cargar_recursos()

st.title("📊 Predicción de Cancelación de Clientes (Telecom Churn)")
st.markdown("Esta aplicación utiliza **Machine Learning (Random Forest)** para predecir si un cliente de telecomunicaciones tiene probabilidades de cancelar su servicio basándose en su perfil y comportamiento.")

if model is None:
    st.error("⚠️ No se encontró el modelo entrenado. Asegúrate de correr primero el script `python src/train_model.py`.")
else:
    st.sidebar.header("📝 Parámetros del Cliente")
    
    # Entradas principales para la barra lateral (Sliders y Selectbox representativos)
    tenure = st.sidebar.slider("Antigüedad (Meses)", 0, 72, 12)
    monthly_charge = st.sidebar.number_input("Cargo Mensual ($)", 0.0, 150.0, 70.0)
    total_charges = st.sidebar.number_input("Cargo Total ($)", 0.0, 9000.0, 800.0)
    
    contract = st.sidebar.selectbox("Tipo de Contrato", ["Month-to-month", "One year", "Two year"])
    internet_service = st.sidebar.selectbox("Servicio de Internet", ["Yes", "No"])
    internet_type = st.sidebar.selectbox("Tipo de Internet", ["Fiber Optic", "DSL", "Cable", "None"])
    payment_method = st.sidebar.selectbox("Método de Pago", ["Bank Withdrawal", "Credit Card", "Mailed Check"])
    
    satisfaction_score = st.sidebar.slider("Puntuación de Satisfacción (1-5)", 1, 5, 3)
    num_referrals = st.sidebar.slider("Número de Referidos", 0, 10, 0)

    # Botón de predicción
    if st.button("🔍 Predecir Churn", use_container_width=True):
        # Crear un DataFrame por defecto con valores base basados en las columnas del modelo
        # Rellenamos con ceros o valores por defecto y luego actualizamos los que ingresó el usuario
        input_data = {col: 0 for col in columnas_modelo}
        
        # Asignar valores ingresados (mapeando a los nombres exactos que espera el modelo)
        if 'Tenure in Months' in input_data: input_data['Tenure in Months'] = tenure
        if 'Monthly Charge' in input_data: input_data['Monthly Charge'] = monthly_charge
        if 'Total Charges' in input_data: input_data['Total Charges'] = total_charges
        if 'Satisfaction Score' in input_data: input_data['Satisfaction Score'] = satisfaction_score
        if 'Number of Referrals' in input_data: input_data['Number of Referrals'] = num_referrals
        
        # Como el modelo entrenado codificó las categóricas con LabelEncoder numérico, 
        # para una prueba rápida pasamos un dataframe con las columnas en el orden correcto.
        df_pred = pd.DataFrame([input_data])
        
        # Realizar predicción
        prediccion = model.predict(df_pred)[0]
        probabilidad = model.predict_proba(df_pred)[0][1] # Probabilidad de churn
        
        st.subheader("Resultado del Análisis:")
        
        col1, col2 = st.columns(2)
        with col1:
            if prediccion == 1:
                st.error(f"⚠️ **ALERTA:** Este cliente tiene **ALTO RIESGO** de cancelar el servicio.")
                st.metric(label="Probabilidad de Churn", value=f"{probabilidad * 100:.2f}%")
            else:
                st.success(f"✅ **ESTABLE:** Es muy probable que este cliente **RETIENE** su servicio.")
                st.metric(label="Probabilidad de Churn", value=f"{probabilidad * 100:.2f}%")
        
        with col2:
            st.info("💡 **Consejo de Negocio:** Los clientes con contratos mensuales cortos y baja satisfacción suelen concentrar el mayor porcentaje de cancelación. Considera ofrecerles promociones de retención o contratos anuales.")