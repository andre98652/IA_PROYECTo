import streamlit as st
import numpy as np, joblib

st.set_page_config(page_title="Stress Predictor", page_icon="🧠", layout="centered")

# ---------- carga del modelo ----------
@st.cache_resource
def load_model():
    return joblib.load("stress_model.joblib")

model = load_model()
classes = ["High", "Low", "Moderate"]  # asegura mismo orden

# ---------- UI ----------
st.title("🧠 Predicción de Estrés Académico")

study  = st.slider("Horas de estudio por día",       5.0, 10.0, 7.5, 0.1)
sleep  = st.slider("Horas de sueño por día",         5.0, 10.0, 7.5, 0.1)
phys   = st.slider("Actividad física (h/día)",       0.0, 13.0, 4.0, 0.1)
social = st.slider("Horas sociales (h/día)",         0.0,  6.0, 2.5, 0.1)
extra  = st.slider("Extracurriculares (h/día)",      0.0,  4.0, 1.0, 0.1)
gpa    = st.slider("GPA (0-4)",                      2.24, 4.00, 3.10, 0.01)

if st.button("Predecir"):
    X = np.array([[study, extra, sleep, social, phys, gpa]])
    probs = model.predict_proba(X)[0]
    label = classes[int(np.argmax(probs))]

    st.subheader(f"Nivel de estrés: **{label}**")
    st.write("Probabilidades:")
    st.bar_chart({c: p for c, p in zip(classes, probs)})

    st.info(
        "⚠️ Esta herramienta es orientativa y no sustituye la valoración de un profesional."
    )
