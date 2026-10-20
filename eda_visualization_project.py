"""
EDA & Visualization Case Study
Dataset: Wisconsin Diagnostic Breast Cancer (WDBC)
Run: python eda_visualization_project.py
Requirements: see requirements.txt
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
                             classification_report)
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PLOT_DIR = ROOT / "visualizations"
DATA_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42

# 1. Load dataset; use bundled CSV if available, otherwise download from sklearn package.
csv_path = DATA_DIR / "breast_cancer_wisconsin.csv"
if csv_path.exists():
    df = pd.read_csv(csv_path)
else:
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()
    df["diagnosis"] = df["target"].map({0: "malignant", 1: "benign"})
    df.to_csv(csv_path, index=False)

# sklearn's dataset has 30 numeric predictors; target 0=malignant, 1=benign.
feature_cols = [c for c in df.columns if c not in ("target", "diagnosis")]
X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df["target"], errors="coerce").astype(int)

print("Shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))
print(df["diagnosis"].value_counts())

# 2. EDA visualizations
df["diagnosis"].value_counts().reindex(["malignant", "benign"]).plot(kind="bar")
plt.title("Class Distribution"); plt.xlabel("Diagnosis"); plt.ylabel("Cases")
plt.xticks(rotation=0); plt.tight_layout()
plt.savefig(PLOT_DIR / "01_class_distribution.png", dpi=160); plt.close()

df["mean radius"].hist(bins=25, edgecolor="black")
plt.title("Distribution of Mean Radius"); plt.xlabel("Mean radius"); plt.ylabel("Frequency")
plt.tight_layout(); plt.savefig(PLOT_DIR / "02_mean_radius_histogram.png", dpi=160); plt.close()

df.boxplot(column="mean area", by="diagnosis")
plt.title("Mean Area by Diagnosis"); plt.suptitle("")
plt.xlabel("Diagnosis"); plt.ylabel("Mean area")
plt.tight_layout(); plt.savefig(PLOT_DIR / "03_mean_area_boxplot.png", dpi=160); plt.close()

corr = df[feature_cols[:10]].corr()
plt.figure(figsize=(8, 6))
im = plt.imshow(corr, aspect="auto", vmin=-1, vmax=1); plt.colorbar(im, label="Correlation")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
plt.yticks(range(len(corr.index)), corr.index, fontsize=7)
plt.title("Correlation Heatmap (First 10 Features)")
plt.tight_layout(); plt.savefig(PLOT_DIR / "04_correlation_heatmap.png", dpi=160); plt.close()

for label, name in [(0, "malignant"), (1, "benign")]:
    subset = df[df["target"] == label]
    plt.scatter(subset["mean radius"], subset["mean texture"], s=18, alpha=.6, label=name)
plt.title("Mean Radius vs Mean Texture"); plt.xlabel("Mean radius"); plt.ylabel("Mean texture")
plt.legend(); plt.tight_layout()
plt.savefig(PLOT_DIR / "05_radius_texture_scatter.png", dpi=160); plt.close()

# 3. Preprocessing and stratified train/validation/test split (70/15/15).
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp)
medians = X_train.median()
X_train = X_train.fillna(medians); X_val = X_val.fillna(medians); X_test = X_test.fillna(medians)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# 4. Basic hyperparameter comparison using validation accuracy.
candidates = [(32,), (64, 32), (64, 32, 16)]
best_model, best_layers, best_val = None, None, -1
for layers in candidates:
    model = MLPClassifier(hidden_layer_sizes=layers, activation="relu", solver="adam",
                          alpha=0.001, learning_rate_init=0.001, max_iter=500,
                          early_stopping=True, validation_fraction=0.15,
                          n_iter_no_change=25, random_state=RANDOM_STATE)
    model.fit(X_train_s, y_train)
    val_acc = accuracy_score(y_val, model.predict(X_val_s))
    print(f"Hidden layers {layers}: validation accuracy={val_acc:.4f}")
    if val_acc > best_val:
        best_model, best_layers, best_val = model, layers, val_acc

# 5. Final test evaluation.
pred = best_model.predict(X_test_s)
prob = best_model.predict_proba(X_test_s)[:, 1]
fpr, tpr, _ = roc_curve(y_test, prob)
results = {
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision": float(precision_score(y_test, pred, zero_division=0)),
    "recall": float(recall_score(y_test, pred, zero_division=0)),
    "f1": float(f1_score(y_test, pred, zero_division=0)),
    "auc": float(auc(fpr, tpr)),
    "best_hidden_layers": str(best_layers),
    "validation_accuracy": float(best_val),
    "train_rows": int(len(X_train)), "validation_rows": int(len(X_val)),
    "test_rows": int(len(X_test))
}
with open(ROOT / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
print(classification_report(y_test, pred, target_names=["malignant", "benign"], zero_division=0))

# Loss curve
plt.plot(range(1, len(best_model.loss_curve_) + 1), best_model.loss_curve_)
plt.title("Training Loss vs Iteration"); plt.xlabel("Iteration"); plt.ylabel("Loss")
plt.tight_layout(); plt.savefig(PLOT_DIR / "06_loss_curve.png", dpi=160); plt.close()

ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["malignant", "benign"], values_format="d")
plt.title("Confusion Matrix"); plt.tight_layout()
plt.savefig(PLOT_DIR / "07_confusion_matrix.png", dpi=160); plt.close()

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {results['auc']:.3f}"); plt.plot([0, 1], [0, 1], "--")
plt.title("ROC Curve"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.legend(); plt.tight_layout(); plt.savefig(PLOT_DIR / "08_roc_curve.png", dpi=160); plt.close()

plt.hist(y_test.to_numpy() - prob, bins=20, edgecolor="black")
plt.title("Prediction Error Distribution"); plt.xlabel("Actual class - predicted probability")
plt.ylabel("Frequency"); plt.tight_layout()
plt.savefig(PLOT_DIR / "09_error_distribution.png", dpi=160); plt.close()

# Permutation importance (interpret cautiously on small held-out sample).
importance = permutation_importance(best_model, X_test_s, y_test, n_repeats=5,
                                    random_state=RANDOM_STATE, scoring="accuracy")
imp = pd.Series(importance.importances_mean, index=feature_cols).sort_values(ascending=False).head(10)
imp.sort_values().plot(kind="barh")
plt.title("Top 10 Permutation Importances"); plt.xlabel("Mean accuracy decrease")
plt.tight_layout(); plt.savefig(PLOT_DIR / "10_feature_importance.png", dpi=160); plt.close()
print("Done. Charts saved to:", PLOT_DIR)
