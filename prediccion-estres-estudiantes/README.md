# Predicción del Estrés en Estudiantes Universitarios

Este proyecto utiliza un modelo de aprendizaje automático para predecir el nivel de estrés de estudiantes universitarios a partir de variables como calidad del sueño, dolores de cabeza, rendimiento académico, carga de estudio y actividades extracurriculares.

## 📊 Dataset

Basado en datos reales disponibles en línea recolectados mediante encuestas a estudiantes.

## 🧠 Modelo

Se entrena un Árbol de Decisión (`scikit-learn`) con los siguientes pasos:
- Carga y limpieza de datos
- División entrenamiento/prueba
- Evaluación con métricas (accuracy, matriz de confusión)
- Visualización de importancia de variables

## ⚙️ Tecnologías utilizadas

- Python 3.x
- Pandas
- scikit-learn
- Matplotlib
- Seaborn

## ▶️ Ejecución

```bash
python main.py
