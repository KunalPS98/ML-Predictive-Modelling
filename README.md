# 📦 Predictive Modelling of Delivery Timeliness Using Machine Learning on E-Commerce Logistics Data

## Overview

This project builds a binary classification model to predict whether an e-commerce shipment will be **delayed** or **delivered on time**, using structured logistics data. The pipeline covers the full ML lifecycle — from data loading and exploratory analysis through to model evaluation, feature importance, and a deployable Streamlit dashboard.

**Target Variable:** `Reached.on.Time_Y.N`
- `1` → Delayed (did not reach on time)
- `0` → On time

---

## 📁 Repository Structure

```
ML-Predictive-Modelling/
│
├── Predictive_Modelling_Using_ML_on_E-Commerce_Logistics_Data.ipynb   # Main notebook
├── Train.csv                          # Training dataset
├── app.py                             # Streamlit web application
├── delivery_delay_best_model.joblib   # Saved best-performing model
└── model_results.json                 # Final evaluation metrics (JSON)
```

---

## 🏆 Results — Best Model: Gradient Boosting

| Metric | Score |
|--------|-------|
| Accuracy | 68.3% |
| Precision | 91.4% |
| Recall | 51.7% |
| F1-Score | 66.1% |
| ROC-AUC | 0.751 |
| PR-AUC | 0.858 |

> **High precision (91.4%)** means the model rarely flags a shipment as delayed when it isn't — making it well-suited for operational alerting without false alarms.

### All Models — Test Set Comparison

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| **Gradient Boosting** ⭐ | 68.3% | 0.661 | 0.751 |
| Stacking Classifier | 67.6% | 0.680 | 0.748 |
| SVM | 65.5% | 0.632 | 0.734 |
| Random Forest | 65.6% | 0.686 | 0.733 |
| Gaussian NB | 64.5% | 0.586 | 0.731 |
| Logistic Regression | 63.9% | 0.690 | 0.718 |
| KNN | 61.7% | 0.668 | 0.692 |

---

## 🔄 ML Pipeline

The notebook is structured as a clean, end-to-end reproducible workflow across 20 sections:

| Step | Description |
|------|-------------|
| 1 | Data Loading & Validation |
| 2 | Data Quality Check (missing values, duplicates, types) |
| 3 | Train / Validation / Test Split (70 / 10 / 20) |
| 4 | Exploratory Data Analysis (EDA) |
| 5 | Nested Preprocessing Pipeline (leakage-safe) |
| 6 | Baseline Model Training & Cross-Validation |
| 7 | Advanced Model Training & Cross-Validation |
| 8 | Combined Model Comparison |
| 9 | Hyperparameter Tuning (RandomizedSearchCV) |
| 10 | Ensemble Methods (Voting & Stacking) |
| 11 | Final Model Evaluation on Test Set |
| 12–18 | Visualisations (Confusion Matrices, ROC, PR Curves, Learning Curve, Feature Importance) |
| 19 | Save Best Model & Results |
| 20 | Final Summary |

---

## ⚙️ Preprocessing Architecture

A nested, leakage-safe `sklearn` Pipeline with `ColumnTransformer`:

```
ColumnTransformer
├── Numeric:      RobustScaler → SelectKBest (top 10 features, ANOVA F-score)
└── Categorical:  OneHotEncoder (handle_unknown='ignore')
```

All preprocessing is encapsulated inside the pipeline, preventing data leakage across folds.

---

## 🧪 Models Benchmarked

**Baseline:** Logistic Regression, K-Nearest Neighbours (KNN), Gaussian Naive Bayes

**Advanced:** Random Forest, Gradient Boosting, Support Vector Machine (SVM)

**Ensembles:** Voting Classifier, Stacking Classifier

All models evaluated using **Repeated Stratified K-Fold Cross-Validation** (5 splits × 3 repeats, F1-Weighted scoring).

---

## 📊 Evaluation Metrics

Each model is assessed on the held-out test set using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and PR-AUC. Results are saved to `model_results.json`.

---

## 📈 Visualisations Generated

| File | Description |
|------|-------------|
| `01_target_distribution.png` | Class balance (bar + pie chart) |
| `02_numeric_distributions.png` | Feature histograms |
| `03_correlation_heatmap.png` | Pearson correlation between numeric features |
| `04_anomaly_detection.png` | Outlier detection using Elliptic Envelope |
| `05_baseline_models_comparison.png` | Baseline CV performance |
| `06_advanced_models_comparison.png` | Advanced model CV performance |
| `07_model_comparison.png` | All models combined |
| `08_confusion_matrices.png` | Confusion matrices for all models |
| `09_roc_curves.png` | ROC curves with AUC scores |
| `10_pr_curves.png` | Precision-Recall curves |
| `11_learning_curve.png` | Learning curve for best model |
| `12_feature_importance.png` | Permutation feature importance (top 20) |

---

## 🚀 Running the Streamlit App

The `app.py` file provides an interactive dark-mode dashboard with:
- **Overview page** with model metrics and project context
- **EDA page** with interactive Plotly charts (distributions, correlation heatmap, anomaly detection, scatter plots)
- **Prediction page** supporting both single (manual entry) and batch (CSV upload) predictions, with downloadable results

```bash
# Install dependencies
pip install streamlit scikit-learn pandas joblib plotly seaborn streamlit-option-menu

# Run the app
python -m streamlit run app.py
```

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| Data | `pandas`, `numpy` |
| Modelling | `scikit-learn` |
| Visualisation | `matplotlib`, `seaborn`, `plotly` |
| App | `streamlit`, `streamlit-option-menu` |
| Persistence | `joblib`, `json` |

---

## 📌 Key Configuration

```python
RANDOM_STATE = 42
TEST_SIZE    = 0.20
K_BEST       = 10          # Top features via SelectKBest
N_ESTIMATORS = 100
CV_STRATEGY  = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
```

---

## 👤 Author

**Kunal**
MSc Data Analytics
[GitHub: KunalPS98](https://github.com/KunalPS98)
