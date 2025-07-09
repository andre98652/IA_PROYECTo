from joblib import load
from data_preprocessing import preprocess_data
from sklearn.metrics import classification_report

model = load('models/student_stress_model.joblib')
X_train, X_test, y_train, y_test = preprocess_data('data/student_lifestyle_dataset.csv')

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
