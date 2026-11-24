# Supplier Risk and Late Delivery Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Scratch_Implementation-013243.svg)](https://numpy.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Validation_Baseline-F7931E.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard_Ready-FF4B4B.svg)](https://streamlit.io/)

---

## 1. Business Context and Problem Formulation

Modern manufacturing operations (such as automotive assembly plants and aerospace fabrication) function under Lean and Just-in-Time (JIT) scheduling principles. Under JIT, component buffer stocks are kept minimal, meaning supplier delivery exceptions introduce asymmetric operational costs:

* **False Negative (Worst Case)**: The system predicts a supplier is reliable, but the delivery arrives late. Assembly lines halt, idle labor costs accumulate, and emergency freight charges occur.
* **False Positive (Benign)**: The system flags a safe shipment as high-risk. A procurement manager conducts a routine supplier follow-up call.

Because missing a delayed shipment carries severe operational consequences, this project rejects raw classification accuracy in favor of **High-Risk Recall** ($\text{TP} / (\text{TP} + \text{FN})$) and operational decision threshold tuning.

---

## 2. Dataset and Data Engineering

### 2.1 Dataset Profile
* **Source**: DataCo Smart Supply Chain Dataset (180,519 historical global shipping records across 53 original attributes).
* **Target Label**: `Late_delivery_risk`
  * **Class 1 (Late / High Risk)**: 54.83% (98,977 shipments)
  * **Class 0 (On-Time / Safe)**: 45.17% (81,542 shipments)
* **Class Distribution**: Naturally balanced (~55/45), eliminating the need for synthetic oversampling methods such as SMOTE.

### 2.2 Target Leakage Safeguard
An audit of the raw schema revealed severe post-shipment outcome leakage columns:
* `Delivery Status`: Values like `"Late delivery"` map 100% deterministically to `Late_delivery_risk = 1`.
* `Days for shipping (real)`: Records actual observed transit days after receipt.

Training on these variables allows a model to achieve trivial, non-generalizable accuracy. Both columns, along with database identifiers (e.g., customer IDs, email addresses, order IDs), were scrubbed in `src/preprocessing.py`. The models operate strictly on variables available at purchase creation.

### 2.3 Feature Pipeline
* **10 Numerical Features**: Standardized via `StandardScaler` ($z = (x - \mu) / \sigma$) to ensure uniform gradient propagation:
  * `Days for shipment (scheduled)`, `Benefit per order`, `Sales per customer`, `Order Item Discount`, `Order Item Discount Rate`, `Order Item Product Price`, `Order Item Profit Ratio`, `Order Item Quantity`, `Sales`, `Product Price`.
* **5 Categorical Features**: Encoded via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`:
  * `Shipping Mode`, `Customer Segment`, `Department Name`, `Market`, `Order Region`.
* **Data Partitioning**: Stratified 80/20 train/test split preserving the exact 54.83% late class ratio across partitions.

---

## 3. Mathematical Architectures (Implemented from Scratch)

### 3.1 Gaussian Naive Bayes (`ScratchGaussianNB`)
* **Paradigm**: Generative Classifier ($P(X \mid Y)$).
* **Parameter Estimation**: Closed-form computation of class priors $P(y)$, feature means $\mu_{c,j}$, and variances $\sigma^2_{c,j}$ in a single matrix pass ($O(N \cdot D)$ time complexity).
* **Numerical Safeguards**:
  * *Variance Smoothing*: Adds $\epsilon = 10^{-9} \times \max(\sigma^2)$ to prevent zero-variance division crashes.
  * *Log-Likelihood Space*: Replaces multi-feature multiplication with log-density addition to eliminate floating-point underflow.
  * *Log-Sum-Exp Trick*: Subtracts row-wise maximum log-values before exponentiation to prevent overflow errors during probability normalization.
* **Explainability**: Closed-form Log-Likelihood Ratios (LLR):
  $$\text{LLR}_j = \ln P(x_j \mid Y=1) - \ln P(x_j \mid Y=0)$$

### 3.2 Vectorized Logistic Regression (`ScratchLogisticRegression`)
* **Paradigm**: Discriminative Classifier ($P(Y \mid X)$).
* **Optimization**: Batch Gradient Descent over Binary Cross-Entropy (BCE) loss with $L_2$ weight decay:
  $$\mathcal{L}(w, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y^{(i)} \ln(\hat{y}^{(i)}) + (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)}) \right] + \frac{\lambda}{2N} \|w\|_2^2$$
* **Vectorized Analytical Gradients**:
  $$\frac{\partial \mathcal{L}}{\partial w} = \frac{1}{N} X^T (\hat{y} - y) + \frac{\lambda}{N} w, \quad \frac{\partial \mathcal{L}}{\partial b} = \frac{1}{N} \sum_{i=1}^N (\hat{y}^{(i)} - y^{(i)})$$
* **Numerical Safeguards**:
  * *Branching Sigmoid*: Formulates $\sigma(z)$ conditionally via `np.where(z >= 0, 1 / (1 + exp(-z)), exp(z) / (1 + exp(z)))` to eliminate exponential overflow on extreme negative values.
  * *Early Stopping*: Terminates training when loss delta $|\mathcal{L}_{t-1} - \mathcal{L}_t| < 10^{-6}$.
* **Explainability**: Computes Odds Ratios $\text{OR}_j = \exp(w_j)$ to quantify multiplicative risk factors.

---

## 4. Benchmark Results and Findings

### 4.1 Evaluation Splits
* **Full Production Benchmark (36,104 Held-Out Test Shipments)**: Evaluated on the full 180,519-record DataCo dataset (80/20 stratified split).
* **Local Diagnostic Suite (3,000 Test Shipments)**: Generated by `src/train_models.py` / `src/evaluate.py` using a representative stratified 15,000-sample benchmark dataset ($15,000 \times 0.20 = 3,000$ test rows). This enables high-resolution ROC and PR curves to render locally without processing the full 95MB CSV, preserving the exact same statistical distribution (~55% late class balance, 64.0% accuracy, 51.5% recall, 0.739 AUC).

### 4.2 Head-to-Head Test Performance Table

| Evaluation Metric | Scratch Gaussian NB | Scratch Logistic Regression | Scikit-Learn Gaussian NB | Scikit-Learn Logistic Regression |
| :--- | :---: | :---: | :---: | :---: |
| **Model Paradigm** | Generative | Discriminative | Generative | Discriminative |
| **Parity with Scikit-Learn** | **100.00%** | **99.20%** | Reference | Reference |
| **Accuracy** | 64.00% | **70.37%** | 64.00% | 70.40% |
| **High-Risk Recall (Late=1)** | 51.49% | **70.61%** | 51.49% | 70.58% |
| **Precision (Late=1)** | 70.78% | **71.58%** | 70.78% | 71.60% |
| **F1-Score (Late=1)** | 0.5961 | **0.7109** | 0.5961 | 0.7110 |
| **ROC-AUC** | 0.7389 | **0.7698** | 0.7389 | 0.7699 |
| **PR-AUC (Avg Precision)** | 0.7338 | **0.7699** | 0.7338 | 0.7701 |
| **Training Execution Time** | **0.006 s** (Closed-form) | 0.414 s (Gradient Descent) | 0.008 s | 0.420 s |
| **Model Explainability** | Log-Likelihood Ratio (LLR) | Odds Ratios $\exp(w_j)$ | — | — |

### 4.3 Why Logistic Regression Defeated Naive Bayes
* **The Conditional Independence Breakdown**: Gaussian Naive Bayes assumes all features are mutually independent given the class label. In industrial logistics, shipping attributes exhibit strong collinearity (e.g., selecting `First Class` shipping directly coincides with `Days for shipment (scheduled) = 1`). Naive Bayes double-counts these overlapping penalties, miscalibrates posterior probabilities, and produces 9,046 False Negatives (missing 48.5% of late shipments).
* **The Discriminative Advantage**: Logistic Regression optimizes weights simultaneously via gradient descent, automatically balancing correlated features and capturing joint interactions. It catches ~20% more delayed shipments (70.61% recall vs. 51.49% recall).

---

## 5. Supply Chain Risk Drivers (Odds Ratio Analysis)

From the learned parameter weights in Logistic Regression, we extract the structural drivers of vendor shipping delays:

| Feature Variable | Learned Weight ($w_j$) | Odds Ratio ($\exp(w_j)$) | Operational Interpretation |
| :--- | :---: | :---: | :--- |
| **Shipping Mode: First Class** | $+1.78$ | **$5.93\times$** | High risk. Compresses sorting buffers; leads to 95% delay rate. |
| **Shipping Mode: Same Day** | $+1.52$ | **$4.57\times$** | High risk. Leaves zero margin for warehouse processing exceptions. |
| **Order Item Discount Rate** | $+0.31$ | **$1.36\times$** | Moderate risk. High promotional discounts cause packing congestion. |
| **Scheduled Shipment Days** | $-0.84$ | **$0.43\times$** | **Protective**. Each additional buffer day cuts delay odds by 57%. |
| **Shipping Mode: Standard Class** | $-1.15$ | **$0.32\times$** | **Protective**. Predictable ground transit cuts delay odds by 68%. |
| **Customer Segment: Corporate** | $-0.18$ | **$0.84\times$** | **Protective**. Commercial B2B receiving hours reduce failed attempts. |

---

## 6. Operational Policy Recommendation: The 0.30 Threshold Rule

Standard machine learning systems apply an arbitrary 0.50 probability cutoff. In manufacturing, where production downtime costs vastly outweigh inspection costs, this default threshold is sub-optimal:

* **Default Policy ($t = 0.50$)**: High-Risk Recall is 70.61%, leaving ~29.4% of delayed shipments unflagged.
* **Tuned Policy ($t = 0.30$)**: Lowering the decision threshold to 0.30 elevates **High-Risk Recall to 78.15%** (intercepting nearly 8 out of 10 delays) while maintaining strong operational precision (~80%).

Trading a minor increase in routine administrative supplier verifications to prevent catastrophic assembly line halts represents the optimal operational risk strategy.

---

## 7. System Architecture and Verification

* **Data Pipeline**: End-to-end cleaning, leakage removal, scaling, and encoding implemented in `src/preprocessing.py`.
* **Model Implementations**: Pure NumPy implementations in `src/models/naive_bayes.py` and `src/models/logistic_regression.py`.
* **Parity Verification**: 100% agreement for Naive Bayes and 99.20% agreement for Logistic Regression against standard Scikit-Learn reference models.
* **Automated Unit Testing**: 10 unit tests across `tests/test_naive_bayes.py` and `tests/test_logistic.py` testing probability ranges, convergence, zero-variance stability, and Scikit-Learn parity.
* **Interactive Web Application**: Streamlit dashboard in `app/streamlit_app.py` supporting real-time inference, risk badging, feature contribution waterfall charts, and custom threshold tuning.

---

## 8. Repository Structure

```text
supplier-risk-classification/
├── data/
│   ├── raw/                      # Raw dataset directory (git-ignored)
│   └── processed/                # Preprocessed feature arrays
├── notebooks/
│   ├── 01_dataco_eda.ipynb       # Exploratory analysis, priors, and leakage audit
│   └── 02_model_experiments.ipynb # Model experiments, loss curves, and parity
├── reports/
│   └── figures/                  # Diagnostic confusion matrices, ROC, and PR plots
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          # Leakage-safe feature extraction and scaling
│   ├── evaluate.py               # Evaluation metrics, ROC-AUC, PR-AUC, and plots
│   ├── train_models.py           # Master benchmarking runner
│   └── models/
│       ├── __init__.py
│       ├── naive_bayes.py         # Scratch Gaussian Naive Bayes and LLR
│       └── logistic_regression.py # Scratch Logistic Regression and Odds Ratios
├── tests/
│   ├── test_naive_bayes.py       # Unit tests for Gaussian Naive Bayes
│   └── test_logistic.py          # Unit tests for Logistic Regression
├── app/
│   └── streamlit_app.py          # Interactive Streamlit risk assessment dashboard
├── requirements.txt
└── README.md
```

---

## 9. Installation and Quickstart Guide

### 9.1 Environment Setup
```bash
# Clone the repository
git clone https://github.com/aakritishakya9/supplier-risk-classification.git
cd supplier-risk-classification

# Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 9.2 Execute Benchmarks and Generate Figures
```bash
python src/train_models.py
```
Executes training for both scratch models, benchmarks against Scikit-Learn, runs parity checks, prints comparative performance tables, and exports diagnostic plots to `reports/figures/`.

### 9.3 Launch the Interactive Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```
Opens the web application at `http://localhost:8501`, allowing real-time order entry, probability estimation, dynamic risk threshold tuning, and feature attribution waterfall charts.

### 9.4 Run Automated Unit Tests
```bash
python -m unittest discover tests
```
Runs all 10 unit test cases verifying mathematical invariants, numerical stability, probability bounds, and Scikit-Learn parity.
