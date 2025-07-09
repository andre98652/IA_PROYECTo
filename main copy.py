# -------------------------------------------------
# main.py
# -------------------------------------------------
"""
Visualiza diagramas de dispersión (hábitos vs. GPA)
y, además, entrena un clasificador multiclase
para predecir Stress_Level.  Se construyen
curvas ROC one-vs-rest para cada clase.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
# ─ Si prefieres un modelo menos potente descomenta:
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

# -------------------------------------------------
# 1) Cargar dataset
# -------------------------------------------------
df = pd.read_csv("data/student_lifestyle_dataset.csv")      # <- ajusta ruta si es necesario

# -------------------------------------------------
# 2) Diagramas de dispersión: GPA vs hábitos
# -------------------------------------------------
fig, axs = plt.subplots(3, 2, figsize=(15, 15))

pairs = [
    ("Study_Hours_Per_Day",              "Study Hours Per Day vs. GPA"),
    ("Sleep_Hours_Per_Day",              "Sleep Hours Per Day vs. GPA"),
    ("Physical_Activity_Hours_Per_Day",  "Physical Activity Hours Per Day vs. GPA"),
    ("Social_Hours_Per_Day",             "Social Hours Per Day vs. GPA"),
    ("Extracurricular_Hours_Per_Day",    "Extracurricular Hours Per Day vs. GPA"),
]

for ax, (col, title) in zip(axs.flat, pairs):
    sns.regplot(x=col, y="GPA", data=df,
                scatter_kws={"alpha": .5}, line_kws={"color": "red"}, ax=ax)
    ax.set_title(title)

# Quitar panel vacío (posición 6 del grid 3×2)
fig.delaxes(axs[2, 1])
plt.tight_layout()
plt.show()

# -------------------------------------------------
# 3) Preparar datos para clasificación multiclase
#    (split estratificado, binarización post-split)
# -------------------------------------------------
X = df.drop(["Stress_Level", "Student_ID"], axis=1)
y = df["Stress_Level"]

# ► Split 80-20 manteniendo proporciones de cada clase
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ► Binarizar etiquetas DESPUÉS del split
lb = LabelBinarizer()
y_train_bin = lb.fit_transform(y_train)
y_test_bin  = lb.transform(y_test)

n_classes = y_train_bin.shape[1]
print("Clases:", lb.classes_)          # ['High' 'Low' 'Moderate']

# -------------------------------------------------
# 4) Pipeline: escalado + clasificador OvR
# -------------------------------------------------
pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    #("clf",    OneVsRestClassifier(RandomForestClassifier(
    #                n_estimators=300,
    #                random_state=42,
    #                n_jobs=-1)))
    # ► Si quieres un modelo lineal, comenta la línea anterior
    #    y descomenta la siguiente:
    ("clf", OneVsRestClassifier(
                LogisticRegression(max_iter=1000,
                                   multi_class="ovr",
                                   solver="lbfgs")))
])

pipe.fit(X_train, y_train_bin)

# -------------------------------------------------
# 5) Curvas ROC (one-vs-rest)
# -------------------------------------------------
y_score = pipe.predict_proba(X_test)

fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
colors = ["blue", "red", "green"]
for i, c in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=c, lw=2,
             label=f'ROC curve for class {lb.classes_[i]} (AUC = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Stress_Level (One-vs-Rest)")
plt.legend(loc="lower right")
plt.show()

# -------------------------------------------------
# 6) Métricas finales
# -------------------------------------------------
y_pred_bin = pipe.predict(X_test)
y_test_lbl = lb.inverse_transform(y_test_bin)
y_pred_lbl = lb.inverse_transform(y_pred_bin)

print("\nMatriz de confusión")
print(confusion_matrix(y_test_lbl, y_pred_lbl))

print("\nReporte de clasificación")
print(classification_report(y_test_lbl, y_pred_lbl, target_names=lb.classes_))

# -------------------- FIN ------------------------
