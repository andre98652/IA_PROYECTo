# ----------------------------------------------
# main.py
# ----------------------------------------------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------------------------
# 1) Cargar dataset
# ----------------------------------------------
df = pd.read_csv('data/student_lifestyle_dataset.csv')


# ----------------------------------------------
# 2) Diagramas de dispersión: GPA vs hábitos
# ----------------------------------------------

fig, axs = plt.subplots(3, 2, figsize=(15, 15))

# 1) Study Hours vs GPA
sns.regplot(x='Study_Hours_Per_Day', y='GPA', data=df, ax=axs[0, 0], scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
axs[0, 0].set_title('Study Hours Per Day vs. GPA')

# 2) Sleep Hours vs GPA
sns.regplot(x='Sleep_Hours_Per_Day', y='GPA', data=df, ax=axs[0, 1], scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
axs[0, 1].set_title('Sleep Hours Per Day vs. GPA')

# 3) Physical Activity Hours vs GPA
sns.regplot(x='Physical_Activity_Hours_Per_Day', y='GPA', data=df, ax=axs[1, 0], scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
axs[1, 0].set_title('Physical Activity Hours Per Day vs. GPA')

# 4) Social Hours vs GPA
sns.regplot(x='Social_Hours_Per_Day', y='GPA', data=df, ax=axs[1, 1], scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
axs[1, 1].set_title('Social Hours Per Day vs. GPA')

# 5) Extracurricular Hours vs GPA
sns.regplot(x='Extracurricular_Hours_Per_Day', y='GPA', data=df, ax=axs[2, 0], scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
axs[2, 0].set_title('Extracurricular Hours Per Day vs. GPA')

# Quitar panel vacío
fig.delaxes(axs[2, 1])

plt.tight_layout()
plt.show()

# ----------------------------------------------
# 3) Preparar datos para clasificación multiclase
# ----------------------------------------------

# Features: elimina columnas que no se usan
X = df.drop(['Stress_Level', 'Student_ID'], axis=1)
y = df['Stress_Level']

# Escalar features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Codificar y binarizar clases para ROC multiclase
lb = LabelBinarizer()
y_binarized = lb.fit_transform(y)

# Para multiclass OvR debe ser array (n_samples, n_classes)
print(f"Clases: {lb.classes_}")  # ['High', 'Low', 'Moderate']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binarized, test_size=0.2, random_state=42)

# ----------------------------------------------
# 4) Entrenar modelo OvR
# ----------------------------------------------
clf = OneVsRestClassifier(RandomForestClassifier(random_state=42))
clf.fit(X_train, y_train)

# Probabilidades de cada clase
y_score = clf.predict_proba(X_test)

# ----------------------------------------------
# 5) Curvas ROC para cada clase
# ----------------------------------------------

fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y_binarized.shape[1]

# Calcular ROC para cada clase
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# ----------------------------------------------
# 6) Dibujar todas las curvas ROC juntas
# ----------------------------------------------
plt.figure(figsize=(8, 6))
colors = ['blue', 'red', 'green']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve for class {lb.classes_[i]} (AUC = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Stress_Level (One-vs-Rest)')
plt.legend(loc='lower right')
plt.show()

# ----------------------------------------------
# 7) Reporte final
# ----------------------------------------------
# Predicciones finales para clasificación
y_pred = clf.predict(X_test)
y_test_labels = lb.inverse_transform(y_test)
y_pred_labels = lb.inverse_transform(y_pred)

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test_labels, y_pred_labels))

print("\nReporte de Clasificación:")
print(classification_report(y_test_labels, y_pred_labels, target_names=lb.classes_))

# ----------------------------------------------
# FIN
# ----------------------------------------------
