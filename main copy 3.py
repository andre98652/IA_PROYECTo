# -------------------------------------------------
# main.py
# -------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    StratifiedKFold, cross_validate, train_test_split
)
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    roc_auc_score,
    roc_curve, auc, confusion_matrix, classification_report
)

# -------------------------------------------------
# 1) Cargar dataset
# -------------------------------------------------
df = pd.read_csv("data/student_lifestyle_dataset.csv")

# -------------------------------------------------
# 2) Diagramas GPA vs hábitos (opcional, sin cambios)
# -------------------------------------------------
pairs = [
    ("Study_Hours_Per_Day",      "Study Hours vs GPA"),
    ("Sleep_Hours_Per_Day",      "Sleep Hours vs GPA"),
    ("Physical_Activity_Hours_Per_Day", "Physical Activity vs GPA"),
    ("Social_Hours_Per_Day",     "Social Hours vs GPA"),
    ("Extracurricular_Hours_Per_Day",   "Extracurricular vs GPA"),
]
fig, axs = plt.subplots(3, 2, figsize=(15, 15))
for (col, title), ax in zip(pairs, axs.flat):
    sns.regplot(x=col, y="GPA", data=df,
                ax=ax, scatter_kws={"alpha": .5}, line_kws={"color": "red"})
    ax.set_title(title)
fig.delaxes(axs[2, 1]); plt.tight_layout(); plt.show()

# -------------------------------------------------
# 3) Preparar X-y
# -------------------------------------------------
X = df.drop(columns=["Stress_Level", "Student_ID"])
y = df["Stress_Level"]

scaler = StandardScaler()
lb = LabelBinarizer()
y_bin = lb.fit_transform(y)
print("Clases:", lb.classes_)

# -------------------------------------------------
# 4) Definir modelos a comparar
# -------------------------------------------------
lr = OneVsRestClassifier(LogisticRegression(
        max_iter=2000, solver="liblinear"))
rf = OneVsRestClassifier(RandomForestClassifier(
        n_estimators=400, class_weight="balanced", random_state=42))

pipe_lr = Pipeline([("scaler", scaler), ("clf", lr)])
pipe_rf = Pipeline([("scaler", scaler), ("clf", rf)])

models = {"LogisticRegression": pipe_lr,
          "RandomForest":      pipe_rf}

# -------------------------------------------------
# 5) K-Fold cross-validation
# -------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scorers = {
    "accuracy":      "accuracy",
    "f1_macro":      "f1_macro",
    "roc_auc_macro": "roc_auc_ovr"          # ✅ NUEVO
}

cv_results = {}
for name, model in models.items():
    scores = cross_validate(model, X, y, cv=cv, scoring=scorers)
    cv_results[name] = {m: scores[f"test_{m}"].mean()
                        for m in scorers}
    print(f"\n{name}  – 5-fold promedio")
    for met, val in cv_results[name].items():
        print(f"  {met:15s}: {val:0.3f}")

# -------------------------------------------------
# 6) Elegir el mejor modelo (mayor AUC macro)
# -------------------------------------------------
best_name = max(cv_results,
                key=lambda k: cv_results[k]["roc_auc_macro"])
best_pipe = models[best_name]
print(f"\n*** Mejor modelo según AUC macro: {best_name} ***")

# -------------------------------------------------
# 7) Hold-out final (20 %)  → ROC, matriz, reporte
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, test_size=0.20, random_state=42, stratify=y_bin)

best_pipe.fit(X_train, y_train)
y_score = best_pipe.predict_proba(X_test)
y_pred  = best_pipe.predict(X_test)

# ---- ROC por clase + micro/macro ----------------
n_classes = y_bin.shape[1]
fpr, tpr, roc_auc = {}, {}, {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr_mic, tpr_mic, _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc_mic = auc(fpr_mic, tpr_mic)

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes
roc_auc_mac = auc(all_fpr, mean_tpr)

plt.figure(figsize=(8, 6))
colors = ["blue", "red", "green"]
for i, c in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=c, lw=2,
             label=f"{lb.classes_[i]} (AUC = {roc_auc[i]:.2f})")
plt.plot(fpr_mic, tpr_mic, "k--", lw=3,
         label=f"micro-avg (AUC = {roc_auc_mic:.2f})")
plt.plot(all_fpr, mean_tpr, "k-", lw=3,
         label=f"macro-avg (AUC = {roc_auc_mac:.2f})")
plt.plot([0, 1], [0, 1], "grey", ls="--")
plt.xlim([0, 1]); plt.ylim([0, 1.05])
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title(f"ROC Curves – Stress_Level ({best_name})")
plt.legend(); plt.tight_layout(); plt.show()

# ---- Matriz + reporte ---------------------------
print("\nMatriz de confusión")
print(confusion_matrix(lb.inverse_transform(y_test),
                       lb.inverse_transform(y_pred)))

print("\nReporte de clasificación")
print(classification_report(lb.inverse_transform(y_test),
                            lb.inverse_transform(y_pred),
                            target_names=lb.classes_))

# -------------------------------------------------
# FIN
# -------------------------------------------------
