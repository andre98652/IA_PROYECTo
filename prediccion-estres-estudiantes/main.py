import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Paso 1: Cargar el dataset
df = pd.read_csv("data/student_stress.csv")

# Paso 2: Renombrar columnas
df.rename(columns={
    "Kindly Rate your Sleep Quality 😴": "sleep_quality",
    "How many times a week do you suffer headaches 🤕?": "headache_freq",
    "How would you rate you academic performance 👩‍🎓?": "academic_perf",
    "how would you rate your study load?": "study_load",
    "How many times a week you practice extracurricular activities 🎾?": "extracurricular",
    "How would you rate your stress levels?": "stress_level"
}, inplace=True)

# Paso 3: Análisis estadístico básico
print("\n--- ANÁLISIS ESTADÍSTICO DEL DATASET ---")
print(df.describe())

plt.figure()
sns.countplot(data=df, x="stress_level")
plt.title("Distribución de niveles de estrés")
plt.tight_layout()
plt.savefig("distribucion_estres.png")
plt.show()

# Paso 4: Definir variables predictoras (X) y objetivo (y)
X = df[["sleep_quality", "headache_freq", "academic_perf", "study_load", "extracurricular"]]
y = df["stress_level"]

# Paso 5: Separar datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 6: Crear y entrenar el modelo
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluación de bias y varianza
train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_pred)
print(f"\nAccuracy en entrenamiento: {train_accuracy:.2f}")

# Paso 7: Guardar modelo entrenado
joblib.dump(model, "modelo_estres.pkl")

# Paso 8: Evaluar el modelo
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy en prueba: {test_accuracy:.2f}")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Paso 9: Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Matriz de Confusión - Árbol de Decisión")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.tight_layout()
plt.savefig("matriz_confusion.png")
plt.show()

# Paso 10: Importancia de variables
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(6, 4))
sns.barplot(x=importances, y=features)
plt.title("Importancia de cada variable en la predicción")
plt.xlabel("Importancia")
plt.ylabel("Características")
plt.tight_layout()
plt.savefig("importancia_variables.png")
plt.show()

# Paso 11: Comparación de accuracy (bias/varianza visual)
plt.figure()
plt.bar(["Entrenamiento", "Prueba"], [train_accuracy, test_accuracy], color=["green", "blue"])
plt.title("Comparación de precisión: entrenamiento vs prueba")
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("comparacion_accuracy.png")
plt.show()

# Paso 12: Predicción con nueva entrada simulada
nueva_entrada = [[3, 2, 4, 3, 1]]
prediccion = model.predict(nueva_entrada)
print("\nPredicción para nueva entrada (simulada):", prediccion[0])

# Paso 13: Predicción con valores ingresados por consola
print("\n--- PREDICCIÓN MANUAL ---")
try:
    sueño = int(input("Calidad del sueño (1-5): "))
    cabeza = int(input("Dolores de cabeza por semana: "))
    rendimiento = int(input("Desempeño académico (1-5): "))
    carga = int(input("Carga académica (1-5): "))
    extra = int(input("Actividades extracurriculares por semana: "))

    entrada_manual = pd.DataFrame(np.array([[sueño, cabeza, rendimiento, carga, extra]]),
                                   columns=X.columns)
    prediccion_manual = model.predict(entrada_manual)
    print(f"\nNivel de estrés predicho: {prediccion_manual[0]}")
except:
    print("❌ Error: Ingreso inválido.")
