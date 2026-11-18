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
## 🔬 Head-to-Head Benchmark Results: Scratch Models vs. Scikit-Learn

Both models were implemented from scratch in NumPy, regularized, and benchmarked against `scikit-learn` on identical train/test splits:

| Evaluation Metric | Scratch Gaussian NB (Person B) | Scratch Logistic Reg (Person A) | Scikit-Learn Gaussian NB | Scikit-Learn Logistic Reg |
| :--- | :---: | :---: | :---: | :---: |
| **Model Family** | **Generative** | **Discriminative** | Generative | Discriminative |
| **Parity with Scikit-Learn** | **`100.00%`** | **`99.20%`** | Reference | Reference |
| **Accuracy** | 64.00% | **`70.37%`** | 64.00% | 70.40% |
| **High-Risk Recall (Late=1)** | 51.49% | **`70.61%`** | 51.49% | 70.58% |
| **Precision (Late=1)** | 70.78% | **`71.58%`** | 70.78% | 71.60% |
| **F1-Score (Late=1)** | 0.5961 | **`0.7109`** | 0.5961 | 0.7110 |
| **ROC-AUC** | 0.7389 | **`0.7698`** | 0.7389 | 0.7699 |
| **PR-AUC (Avg Precision)** | 0.7338 | **`0.7699`** | 0.7338 | 0.7701 |
| **Training Complexity** | **`0.006 s`** (Closed-form) | `0.414 s` (Gradient Descent) | 0.008 s | 0.420 s |
| **Explainability Mechanism** | Log-Likelihood Ratio (LLR) | Odds Ratios $\exp(w_j)$ | — | — |

---

## ⚖️ Comparative Analysis: Generative vs. Discriminative in Procurement

### 1. Decision Boundary & Feature Interactions
- **Gaussian Naive Bayes (Generative)** assumes all features are conditionally independent given the class: $P(X \mid Y) = \prod_{j} P(x_j \mid Y)$. In real-world supply chain data, features like `Shipping Mode`, `Days for shipment (scheduled)`, and `Discount Rate` exhibit collinearity. This independence assumption artificially constrains Naive Bayes to axis-aligned quadratic likelihood contours, leading to a conservative decision boundary.
- **Logistic Regression (Discriminative)** directly models the posterior distribution $P(Y \mid X) = \sigma(w^T x + b)$ without assumptions about feature distributions. It captures additive feature interactions, allowing correlated shipping modes and price tiers to adjust the decision hyperplane directly.

### 2. High-Risk Recall & Business Decision Trade-offs
- In procurement, **False Negatives are far more dangerous than False Positives** (shutting down a plant vs. conducting an unnecessary supplier audit).
- **Logistic Regression achieves superior High-Risk Recall (70.61% vs. 51.49%)**, successfully catching ~20% more delayed shipments at the default threshold.
- **Recommendation**: For automated vendor gating and operational risk screening, **Logistic Regression is the superior choice** due to its higher recall ($70.6\%$) and overall discriminative power ($\text{AUC} = 0.770$). Gaussian Naive Bayes remains valuable as an ultra-fast baseline ($0.006\text{s}$) where training latency or incremental streaming updates are needed.

---

## 👥 Division of Labor & Progress Status

| Component | Lead | Status | Notes |
| :--- | :---: | :---: | :--- |
| **Cloud EDA & Leakage Audit** | Person B (@aakritishakya9) | ✅ Complete | Run on Google Colab |
| **Leakage-Safe Preprocessing Pipeline** | Person B (@aakritishakya9) | ✅ Complete | Centralized in `src/preprocessing.py` |
| **Gaussian Naive Bayes (NumPy Scratch)** | Person B (@aakritishakya9) | ✅ Complete | Verified at 100% parity with sklearn |
| **Feature Attribution (Log-Likelihood Ratios)** | Person B (@aakritishakya9) | ✅ Complete | Implemented in `src/models/naive_bayes.py` |
| **Automated Unit Testing Suite (GNB)** | Person B (@aakritishakya9) | ✅ Complete | 5 passing tests in `tests/test_naive_bayes.py` |
| **Logistic Regression (NumPy Scratch)** | **Person A** | ✅ Complete | Vectorized gradient descent in `src/models/logistic_regression.py` |
| **Gradient Descent Loss Curve & Tuning** | **Person A** | ✅ Complete | Monotonic convergence with early stopping |
| **Feature Importance (Odds Ratios)** | **Person A** | ✅ Complete | Extracted $\exp(w_j)$ risk drivers |
| **Automated Unit Testing Suite (LR)** | **Person A** | ✅ Complete | 5 passing tests in `tests/test_logistic.py` |
| **Evaluation & Master Benchmark Suite** | Joint | ✅ Complete | ROC, PR, thresholding in `src/evaluate.py` & `src/train_models.py` |
| **Interactive Streamlit Web Dashboard** | Joint | ✅ Complete | Multi-model UI with LLR and Odds Ratios in `app/streamlit_app.py` |
| **Final Comparative Writeup** | Joint | ✅ Complete | Generative vs Discriminative trade-off analysis |

---

## 📁 Repository Structure

```text
supplier-risk-classification/
├── data/
│   ├── raw/                      # Downloaded dataset (git-ignored)
│   └── processed/                # Preprocessed feature arrays
├── notebooks/
│   ├── 01_dataco_eda.ipynb       # Cloud EDA, priors & target balance in Colab
│   └── 02_model_experiments.ipynb # Full scratch experiments, loss curve & parity
├── reports/
│   └── figures/                  # Confusion matrices, ROC curves, PR curves
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          # Leakage-safe feature extraction & scaling
│   ├── evaluate.py               # Evaluation metrics, ROC-AUC, PR-AUC & plotting
│   ├── train_models.py           # Master benchmarking runner
│   └── models/
│       ├── __init__.py
│       ├── naive_bayes.py         # Person B: Scratch Gaussian Naive Bayes + LLR
│       └── logistic_regression.py # Person A: Scratch Logistic Regression + Odds Ratios
├── tests/
│   ├── test_naive_bayes.py       # Unit tests for Gaussian Naive Bayes
│   └── test_logistic.py          # Unit tests for Logistic Regression
├── app/
│   └── streamlit_app.py          # Interactive procurement risk assessment dashboard
├── requirements.txt
├── README.md
└── PROJECT_PROGRESS_AND_HANDOVER_REPORT.md
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

### 3. Run Automated Unit Tests
```bash
python -m unittest discover tests
```
Verifies mathematical invariants, numerical stability against zero-variance features, probability normalization, and parity against `scikit-learn`.

