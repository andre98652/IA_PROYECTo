# ----------------------------------------------
# main.py  (versión mejorada)
# ----------------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report
)

# ----------------------------------------------
# 1) Cargar dataset
# ----------------------------------------------
df = pd.read_csv("data/student_lifestyle_dataset.csv")

# ----------------------------------------------
# 2) Diagramas de dispersión GPA vs. hábitos
# (sin cambios)
# ----------------------------------------------
fig, axs = plt.subplots(3, 2, figsize=(15, 15))
pairs = [
    ("Study_Hours_Per_Day",      "Study Hours Per Day vs. GPA"),
    ("Sleep_Hours_Per_Day",      "Sleep Hours Per Day vs. GPA"),
    ("Physical_Activity_Hours_Per_Day", "Physical Activity Hours Per Day vs. GPA"),
    ("Social_Hours_Per_Day",     "Social Hours Per Day vs. GPA"),
    ("Extracurricular_Hours_Per_Day",   "Extracurricular Hours Per Day vs. GPA"),
]
for (col, title), ax in zip(pairs, axs.flat):
    sns.regplot(x=col, y="GPA", data=df,
                ax=ax, scatter_kws={"alpha": .5}, line_kws={"color": "red"})
    ax.set_title(title)
fig.delaxes(axs[2, 1])            # quita el panel vacío
plt.tight_layout(); plt.show()

# ----------------------------------------------
# 3) Preparar X-y
# ----------------------------------------------
X = df.drop(columns=["Stress_Level", "Student_ID"])
y = df["Stress_Level"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lb = LabelBinarizer()
y_bin = lb.fit_transform(y)
print(f"Clases: {lb.classes_}")         # ['High' 'Low' 'Moderate']

X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y_bin, test_size=0.20,
        random_state=42, stratify=y_bin)

# ----------------------------------------------
# 4) Entrenar modelo (LR + OvR)
# ----------------------------------------------
base_lr = LogisticRegression(max_iter=2000, solver="liblinear")
clf = OneVsRestClassifier(base_lr)
clf.fit(X_tr, y_tr)
y_score = clf.predict_proba(X_te)

# ----------------------------------------------
# 5) ROC por clase  + micro y macro
# ----------------------------------------------
n_classes = y_bin.shape[1]
fpr, tpr, roc_auc = {}, {}, {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_te[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# micro-average
fpr_micro, tpr_micro, _ = roc_curve(y_te.ravel(), y_score.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)

# macro-average
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes
roc_auc_macro = auc(all_fpr, mean_tpr)

# ----------------------------------------------
# 6) Graficar ROC
# ----------------------------------------------
plt.figure(figsize=(8, 6))
colors = ["blue", "red", "green"]
for i, c in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=c, lw=2,
             label=f"Class {lb.classes_[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot(fpr_micro, tpr_micro, "k--", lw=3,
         label=f"micro-avg (AUC = {roc_auc_micro:.2f})")
plt.plot(all_fpr, mean_tpr, "k-", lw=3,
         label=f"macro-avg (AUC = {roc_auc_macro:.2f})")

plt.plot([0, 1], [0, 1], "grey", linestyle="--")
plt.xlim([0, 1]); plt.ylim([0, 1.05])
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves – Stress_Level (One-vs-Rest)")
plt.legend(); plt.tight_layout(); plt.show()

# ----------------------------------------------
# 7) Reporte final
# ----------------------------------------------
y_pred = clf.predict(X_te)
print("\nMatriz de confusión")
print(confusion_matrix(lb.inverse_transform(y_te),
                       lb.inverse_transform(y_pred)))

print("\nReporte de clasificación")
print(classification_report(lb.inverse_transform(y_te),
                            lb.inverse_transform(y_pred),
                            target_names=lb.classes_))
# ----------------------------------------------
# FIN
# ----------------------------------------------
