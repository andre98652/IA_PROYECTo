import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar el modelo entrenado
model = joblib.load("modelo_estres.pkl")

# Título de la app
st.title("🧠 Predicción del Nivel de Estrés en Estudiantes")

st.markdown("""
Esta aplicación predice el nivel de estrés de un estudiante universitario 
en base a cinco factores: calidad del sueño, dolores de cabeza, rendimiento académico, 
carga de estudio y actividades extracurriculares.
""")

# Inputs del usuario
sleep_quality = st.slider("Calidad del sueño (1 = muy mala, 5 = excelente)", 1, 5, 3)
headache_freq = st.slider("Dolores de cabeza por semana", 0, 7, 2)
academic_perf = st.slider("Rendimiento académico (1 = muy bajo, 5 = excelente)", 1, 5, 3)
study_load = st.slider("Carga académica (1 = baja, 5 = muy alta)", 1, 5, 3)
extracurricular = st.slider("Actividades extracurriculares por semana", 0, 7, 2)

# Botón para predecir
if st.button("Predecir nivel de estrés"):
    entrada = pd.DataFrame(np.array([[sleep_quality, headache_freq, academic_perf, study_load, extracurricular]]),
                           columns=["sleep_quality", "headache_freq", "academic_perf", "study_load", "extracurricular"])
    pred = model.predict(entrada)[0]

    st.success(f"🧠 Nivel de estrés estimado: {pred}")
    st.markdown("""
    **Interpretación sugerida**:
    - 1 a 2 → Estrés bajo  
    - 3 → Estrés moderado  
    - 4 a 5 → Estrés alto
    """)
