# Supplier Risk & Quality Classification

A machine learning project evaluating supply chain delivery and vendor risks using the **DataCo Smart Supply Chain Dataset**. We compare two contrasting machine learning paradigms implemented **from scratch using NumPy** and benchmarked against `scikit-learn`:

1. **Discriminative**: Vectorized Logistic Regression with Gradient Descent (Person A)
2. **Generative / Probabilistic**: Gaussian Naive Bayes with Log-Likelihood Estimation (Person B - @aakritishakya9)

---

## 🎯 Objective & Business Impact
In procurement and supply chain management, **False Negatives (failing to identify a late delivery or risky vendor)** cause factory shutdowns, broken SLA commitments, and high financial losses. This project optimizes for **High-Risk Recall** to catch late deliveries before they happen.

---

## 📁 Repository Structure

```
supplier-risk-classification/
├── data/
│   ├── raw/                      # Raw DataCo dataset (git-ignored)
│   └── processed/                # Preprocessed train/test datasets
├── notebooks/
│   ├── 01_dataco_eda.ipynb       # Exploratory Data Analysis & Leakage Check (Colab-ready)
│   └── 02_model_experiments.ipynb # Model benchmarking & comparison
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          # Feature selection, encoding & scaling
│   └── models/
│       ├── __init__.py
│       ├── logistic_regression.py # Person A: Vectorized Logistic Regression
│       └── naive_bayes.py         # Person B: Scratch Gaussian Naive Bayes
├── app/
│   └── streamlit_app.py          # Interactive procurement risk assessor
├── requirements.txt
└── README.md
```

---

## 👥 Division of Labor

| Component | Lead | Reviewer |
| :--- | :---: | :---: |
| **Data Cleaning & Colab EDA** | Joint | Joint |
| **Gaussian Naive Bayes (NumPy Scratch)** | **Person B (@aakritishakya9)** | Person A |
| **Logistic Regression (NumPy Scratch)** | **Person A** | Person B (@aakritishakya9) |
| **Evaluation Suite & Recall Benchmarks** | Joint | Joint |
| **Streamlit Deployment Dashboard** | Joint | Joint |

---

## 🚀 Getting Started

### 1. Run EDA in Google Colab
Open [`notebooks/01_dataco_eda.ipynb`](notebooks/01_dataco_eda.ipynb) directly in Google Colab to download the DataCo dataset, inspect class distributions, and verify target leakage safeguards.

### 2. Local Setup
```bash
git clone https://github.com/aakritishakya9/supplier-risk-classification.git
cd supplier-risk-classification
pip install -r requirements.txt
```
