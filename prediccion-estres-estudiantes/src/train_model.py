from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from data_preprocessing import preprocess_data

X_train, X_test, y_train, y_test = preprocess_data('data/student_lifestyle_dataset.csv')

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Guardar modelo entrenado
dump(model, 'models/student_stress_model.joblib')
