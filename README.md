# 🚚 Supplier Risk & Quality Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/Implemented_from_Scratch-NumPy-013243.svg)](https://numpy.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App_Ready-FF4B4B.svg)](https://streamlit.io/)
[![Google Colab](https://img.shields.io/badge/Google_Colab-Ready-F9AB00.svg)](https://colab.research.google.com/)

An end-to-end machine learning system evaluating vendor reliability and delivery risks using the **DataCo Smart Supply Chain Dataset** (180,519 records). We compare two contrasting machine learning paradigms implemented **from scratch in NumPy** and benchmarked against `scikit-learn`:

1. **Generative / Probabilistic**: Gaussian Naive Bayes with log-likelihood estimation (**Person B: @aakritishakya9**)
2. **Discriminative**: Vectorized Logistic Regression with gradient descent (**Person A**)

---

## 🎯 Business Context: Procurement Risk Asymmetry
In manufacturing and supply chain operations, prediction errors have asymmetric costs:
- **False Negative (FN)**: Classifying a high-risk vendor as safe $\rightarrow$ **Catastrophic** (assembly line stoppages, stockouts, contractual breach penalties, customer churn).
- **False Positive (FP)**: Classifying a safe vendor as risky $\rightarrow$ **Low impact** (minor administrative overhead of conducting an extra supplier audit).

Therefore, this project prioritizes **High-Risk Recall ($TP / (TP + FN)$)** and decision threshold tuning over raw accuracy.

---

## 📊 Dataset & Leakage Prevention Audit

- **Dataset**: [DataCo Smart Supply Chain for Big Data Analysis](https://huggingface.co/idoyaaran/dataco-supply-chain-model) (180,519 rows $\times$ 53 columns)
- **Target Variable**: `Late_delivery_risk`
  - **Late / High Risk ($Y = 1$)**: **54.83%** (98,977 shipments)
  - **On-Time / Safe ($Y = 0$)**: **45.17%** (81,542 shipments)
  - *Class Balance*: Naturally balanced (~55/45).

### ⚠️ Preventing Circular Target Leakage
Cross-tabulation during exploratory data analysis proved that `Delivery Status == 'Late delivery'` maps 1-to-1 with `Late_delivery_risk = 1`:
```text
Delivery Status vs Late_delivery_risk:
Late_delivery_risk      0      1
Delivery Status                 
Advance shipping    41592      0
Late delivery           0  98977
Shipping canceled    7754      0
Shipping on time    32196      0
```
> **Enforced Rule**: Post-shipment outcome columns (`Delivery Status` and `Days for shipping (real)`) are strictly excluded from input features. Models are trained **strictly on variables available at the time of order placement** (22 transformed features: scheduled days, shipping mode, market, order item price, discount, profit ratio, etc.).

---

## 🔬 Benchmark Results: Scratch Gaussian Naive Bayes vs. Scikit-Learn

Evaluated on **36,104 test samples** (stratified 80/20 train/test split on 180,519 records):

| Metric | NumPy Scratch Gaussian NB | Scikit-Learn Gaussian NB | Parity Check |
| :--- | :---: | :---: | :---: |
| **Prediction Agreement** | **`100.00%`** | **`100.00%`** | **Exact 1-to-1 Match** |
| **Max Prob Difference** | **`0.000000`** | **`0.000000`** | **Machine Precision Convergence** |
| **Accuracy** | **`69.63%`** | **`69.63%`** | Identical |
| **Precision (Late Deliveries)** | **`84.85%`** | **`84.85%`** | Identical |
| **Recall (Late Deliveries)** | **`54.30%`** | **`54.30%`** | Identical |
| **F1-Score (Late Deliveries)** | **`66.22%`** | **`66.22%`** | Identical |
| **Training Time (144,415 samples)** | **`0.099 s`** | `0.103 s` | **Scratch is faster** |

### 🎚️ Decision Threshold Tuning for Procurement
By shifting the classification threshold from the default $0.50$ down to $0.30$, we increase sensitivity to delayed deliveries:
- **Default Threshold (0.50)**: Recall = **$54.3\%$** | Precision = **$84.8\%$**
- **Optimized Threshold (0.30)**: Recall = **$58.7\%$** | Precision = **$79.7\%$** *(Catches more high-risk shipments with minimal false alarms)*

---

## 👥 Division of Labor & Progress Status

| Component | Lead | Status | Notes |
| :--- | :---: | :---: | :--- |
| **Cloud EDA & Leakage Audit** | Person B (@aakritishakya9) | ✅ Complete | Run on Google Colab |
| **Leakage-Safe Preprocessing Pipeline** | Person B (@aakritishakya9) | ✅ Complete | Centralized in `src/preprocessing.py` |
| **Gaussian Naive Bayes (NumPy Scratch)** | Person B (@aakritishakya9) | ✅ Complete | Verified at 100% parity with sklearn |
| **Interactive Streamlit Web Dashboard** | Person B (@aakritishakya9) | ✅ Complete | Live UI in `app/streamlit_app.py` |
| **Technical Handover Report for Person A** | Person B (@aakritishakya9) | ✅ Complete | See [`PROJECT_PROGRESS_AND_HANDOVER_REPORT.md`](PROJECT_PROGRESS_AND_HANDOVER_REPORT.md) |
| **Logistic Regression (NumPy Scratch)** | **Person A** | ⏳ In Progress | Scaffolding in `src/models/logistic_regression.py` |
| **Gradient Descent Loss Curve & Tuning** | **Person A** | ⏳ In Progress | Learning rate & epoch tuning |
| **Feature Importance (Odds Ratios)** | **Person A** | ⏳ In Progress | Extract $\exp(w_j)$ weights |
| **Final Comparative Writeup** | Joint | ⏳ Pending | Generative vs Discriminative trade-off analysis |

---

## 📁 Repository Structure

```text
supplier-risk-classification/
├── data/
│   ├── raw/                      # Downloaded dataset (git-ignored)
│   └── processed/                # Preprocessed feature arrays
├── notebooks/
│   ├── 01_dataco_eda.ipynb       # Cloud EDA, priors & target balance in Colab
│   └── 02_model_experiments.ipynb # Scratch GNB vs. sklearn benchmarks & tuning
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          # Leakage-safe feature extraction & scaling
│   └── models/
│       ├── __init__.py
│       ├── naive_bayes.py         # Person B: Scratch NumPy Gaussian Naive Bayes
│       └── logistic_regression.py # Person A: Vectorized Scratch Logistic Regression
├── app/
│   └── streamlit_app.py          # Interactive procurement risk assessment dashboard
├── requirements.txt
├── README.md
└── PROJECT_PROGRESS_AND_HANDOVER_REPORT.md # Technical handover documentation for Person A
```

---

## 💻 Quickstart & Running the Project

### 1. Run in Google Colab (Zero Local Setup)
- 📊 **[Open `01_dataco_eda.ipynb` in Colab](https://colab.research.google.com/github/aakritishakya9/supplier-risk-classification/blob/main/notebooks/01_dataco_eda.ipynb)**: Inspect raw dataset distributions and leakage checks.
- 🧪 **[Open `02_model_experiments.ipynb` in Colab](https://colab.research.google.com/github/aakritishakya9/supplier-risk-classification/blob/main/notebooks/02_model_experiments.ipynb)**: Train `ScratchGaussianNB`, reproduce parity benchmarks, and view confusion matrices.

### 2. Run the Interactive Dashboard Locally (VS Code / Terminal)
```bash
# Clone the repository
git clone https://github.com/aakritishakya9/supplier-risk-classification.git
cd supplier-risk-classification

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit dashboard
python -m streamlit run app/streamlit_app.py
```
Open **`http://localhost:8501`** in your browser to interact with the model selectors and threshold sliders live!
