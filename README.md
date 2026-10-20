# EDA & Visualization Case Study: Breast Cancer Diagnosis

**Course:** U21ADP05 – Exploratory Data Analysis & Visualization  
**Project type:** Real-world tabular dataset + EDA + neural network classification

## Problem statement
Explore diagnostic measurements from the Wisconsin Diagnostic Breast Cancer dataset and build a Multi-Layer Perceptron (MLP) classifier to distinguish benign and malignant cases. This is an educational case study, not a medical diagnostic tool.

## Dataset
- **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)
- **Rows:** 569
- **Predictor features:** 30 numeric features
- **Target:** diagnosis (`malignant` / `benign`)
- **Included file:** `data/breast_cancer_wisconsin.csv`
- **Original source:** UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- scikit-learn dataset documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

## What is included
- `eda_visualization_project.py` — complete runnable pipeline
- `data/breast_cancer_wisconsin.csv` — dataset
- `visualizations/` — EDA and model result charts
- `metrics.json` — measured test metrics from the generated run
- `Project_Report.pdf` — report with methods, charts, results, conclusion and references
- `requirements.txt` — Python dependencies

## Methods
1. Inspect shape, missing values, duplicates and target distribution.
2. Use stratified 70/15/15 train/validation/test split.
3. Median imputation is included as a safeguard; standardization is fitted only on training data.
4. Compare three MLP hidden-layer configurations using validation accuracy.
5. Report final metrics on the held-out test split.
6. Generate EDA and model evaluation charts.

## Run locally
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python eda_visualization_project.py
```

## Results from the fixed random-seed run
- Test accuracy: 0.942
- Precision (benign class): 0.930
- Recall (benign class): 0.981
- F1 score (benign class): 0.955
- ROC-AUC (benign class): 0.992
- Selected hidden layers: `(64, 32)`

Results can vary slightly across package versions. Because this is a medical dataset, do not use the model for real clinical decisions.

## Seven-stage visualization workflow
1. Acquire data and understand the context.
2. Define the question and intended audience.
3. Select variables and data types.
4. Clean and transform the data.
5. Choose suitable visual encodings/charts.
6. Label, format and validate the charts.
7. Interpret findings and communicate limitations.

## Repository checklist
- [x] Source code
- [x] Dataset
- [x] Project report PDF
- [x] README
- [x] Requirements
- [x] Visualization outputs
