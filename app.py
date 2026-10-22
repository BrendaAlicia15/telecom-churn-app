#Paso: Crear la App Web con Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(
    page_title="Dashboard Ejecutivo - Telecom Churn",
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

# Encabezado Ejecutivo
st.title("📊 Dashboard Ejecutivo: Predicción y Prevención de Churn")
st.markdown("Herramienta de análisis predictivo y retención de clientes basada en **Machine Learning** y analítica de datos avanzada.")
st.divider()

if model is None:
    st.error("⚠️ No se encontró el modelo entrenado. Ejecuta primero `python src/train_model.py`.")
else:
    # Sidebar: Controles de entrada de datos
    st.sidebar.header("🎛️ Panel de Control del Cliente")
    st.sidebar.markdown("Modifica los parámetros para simular el comportamiento del cliente:")
    
    tenure = st.sidebar.slider("Antigüedad (Meses)", 0, 72, 12)
    monthly_charge = st.sidebar.number_input("Cargo Mensual ($)", 0.0, 150.0, 70.0)
    total_charges = st.sidebar.number_input("Cargo Total ($)", 0.0, 9000.0, 800.0)
    
    contract = st.sidebar.selectbox("Tipo de Contrato", ["Month-to-month", "One year", "Two year"])
    internet_type = st.sidebar.selectbox("Tipo de Internet", ["Fiber Optic", "DSL", "Cable", "None"])
    payment_method = st.sidebar.selectbox("Método de Pago", ["Bank Withdrawal", "Credit Card", "Mailed Check"])
    
    satisfaction_score = st.sidebar.slider("Puntuación de Satisfacción (1-5)", 1, 5, 3)
    num_referrals = st.sidebar.slider("Número de Referidos", 0, 10, 0)

    # Preparación de datos para la predicción
    input_data = {col: 0 for col in columnas_modelo}
    if 'Tenure in Months' in input_data: input_data['Tenure in Months'] = tenure
    if 'Monthly Charge' in input_data: input_data['Monthly Charge'] = monthly_charge
    if 'Total Charges' in input_data: input_data['Total Charges'] = total_charges
    if 'Satisfaction Score' in input_data: input_data['Satisfaction Score'] = satisfaction_score
    if 'Number of Referrals' in input_data: input_data['Number of Referrals'] = num_referrals
    
    df_pred = pd.DataFrame([input_data])
    
    # Inferencia del Modelo
    prediccion = model.predict(df_pred)[0]
    probabilidad = model.predict_proba(df_pred)[0][1] # Probabilidad de churn
    
    # Layout Principal en Columnas
    col_left, col_right = st.columns([1.2, 1.8])
    
    with col_left:
        st.subheader("🎯 Diagnóstico del Modelo")
        
        # Gráfico de Tacómetro (Gauge Chart) con Plotly
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probabilidad * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Riesgo de Cancelación (%)", 'font': {'size': 18}},
            number = {'suffix': "%"},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkslategray"},
                'steps': [
                    {'range': [0, 40], 'color': "rgba(40, 167, 69, 0.6)"},
                    {'range': [40, 70], 'color': "rgba(255, 193, 7, 0.6)"},
                    {'range': [70, 100], 'color': "rgba(220, 53, 69, 0.6)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        if prediccion == 1 or probabilidad > 0.5:
            st.error("🚨 **Estado: ALTO RIESGO DE FUGA**")
        else:
            st.success("✅ **Estado: CLIENTE ESTABLE / LEAL**")

    with col_right:
        st.subheader("💡 Insights y Estrategia de Retención")
        
        # Métricas clave resumidas
        m1, m2, m3 = st.columns(3)
        m1.metric("Antigüedad", f"{tenure} meses")
        m2.metric("Gasto Mensual", f"${monthly_charge:.2f}")
        m3.metric("Satisfacción", f"{satisfaction_score} / 5 ⭐")
        
        st.markdown("---")
        
        # Recomendación automatizada basada en reglas de negocio inteligentes
        if probabilidad > 0.6:
            st.markdown("""
            > **Acción Comercial Recomendada:**
            > 1. **Oferta de Contrato:** Incentivar el cambio a contrato anual ofreciendo un **20% de descuento** en los próximos 3 meses.
            > 2. **Atención Prioritaria:** Derivar este caso al equipo de retención sénior debido a su baja satisfacción y riesgo crítico.
            """)
        elif probabilidad > 0.3:
            st.markdown("""
            > **Acción Comercial Recomendada:**
            > 1. **Monitoreo Preventivo:** Enviar una encuesta de satisfacción personalizada y ofrecer mejoras gratuitas en velocidad de internet o canales de streaming.
            """)
        else:
            st.markdown("""
            > **Acción Comercial Recomendada:**
            > 1. **Fidelización:** Cliente estable. Considerar para programas de referidos (aprovechando los incentivos de recomendación actuales).
            """)

    # Sección inferior: Análisis analítico avanzado
    st.divider()
    st.subheader("📈 Resumen de Variables Clave del Perfil Analizado")
    
    # Pequeña tabla comparativa simulada para enriquecer el dashboard visual
    resumen_df = pd.DataFrame({
        "Métrica Evaluada": ["Antigüedad del Cliente", "Cargo Mensual Actual", "Nivel de Satisfacción", "Índice de Riesgo Asignado"],
        "Valor del Perfil": [f"{tenure} meses", f"${monthly_charge:.2f}", f"{satisfaction_score} / 5", f"{probabilidad*100:.1f}%"],
        "Estado del Indicador": ["Normal" if tenure > 12 else "Bajo (Crítico)", "Estándar", "Óptimo" if satisfaction_score >= 4 else "Alerta", "Crítico" if probabilidad > 0.5 else "Saludable"]
    })
    st.dataframe(resumen_df, use_container_width=True, hide_index=True)
    