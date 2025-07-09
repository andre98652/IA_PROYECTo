from joblib import load
import numpy as np

model = load('models/student_stress_model.joblib')

# Ejemplo de input nuevo (ya preprocesado y escalado)
new_data = np.array([[6, 2, 7, 3, 1]])  # Ejemplo

prediction = model.predict(new_data)
print(f"Nivel de estrés predicho: {prediction[0]}")
