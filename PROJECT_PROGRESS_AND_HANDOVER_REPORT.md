# Supplier Risk Classification: Progress & Technical Handover Report

> **Project**: Supplier Delivery & Risk Classification  
> **Dataset**: DataCo Smart Supply Chain Dataset (180,519 records)  
> **Repository**: [github.com/aakritishakya9/supplier-risk-classification](https://github.com/aakritishakya9/supplier-risk-classification)  
> **Completed By**: Person B (@aakritishakya9) — Gaussian Naive Bayes & Data Pipeline  
> **Handover To**: Person A — Vectorized Logistic Regression & Feature Importance  

---

## 1. Executive Summary of Completed Work

Person B has completed the end-to-end data pipeline, Exploratory Data Analysis (EDA), leakage prevention audit, mathematical implementation of **Gaussian Naive Bayes from scratch in NumPy**, benchmark verification against `scikit-learn`, and deployment of an interactive **Streamlit** dashboard.

### Key Milestones Achieved
1. **GitHub Collaboration Setup**: Repository initialized with modular code structure (`src/`, `notebooks/`, `app/`, `data/`).
2. **Cloud EDA on Google Colab**: Executed on 180,519 rows, uncovering class distributions and confirming leakage risks.
3. **Target Leakage Safeguard**: Prevented artificial ~100% trivial accuracy by isolating and dropping post-shipment outcome columns.
4. **Gaussian Naive Bayes from Scratch**: Achieved **100.00% exact parity** with `sklearn.naive_bayes.GaussianNB` (0.000000 maximum probability difference) and faster training speed (`0.099s` vs `0.103s`).
5. **Interactive UI**: Live Streamlit decision support tool running locally and deployable to Streamlit Community Cloud in 1 click.

---

## 2. Dataset & Preprocessing Pipeline Specifications

All preprocessing code is centralized and ready for Person A in `src/preprocessing.py`.

### Data Overview
- **Raw Dimensions**: 180,519 rows × 53 columns.
- **Target Label**: `Late_delivery_risk`
  - **Late / High Risk ($Y = 1$)**: **54.83%** (98,977 samples)
  - **On-Time / Safe ($Y = 0$)**: **45.17%** (81,542 samples)
- **Class Balance**: Naturally balanced (~55/45). No synthetic oversampling (SMOTE) needed.

### ⚠️ Critical Target Leakage Audit
During EDA, cross-tabulation confirmed that `Delivery Status == 'Late delivery'` has a **100% correlation** with `Late_delivery_risk = 1`:
```
Delivery Status vs Late_delivery_risk:
Late_delivery_risk      0      1
Delivery Status                 
Advance shipping    41592      0
Late delivery           0  98977
Shipping canceled    7754      0
Shipping on time    32196      0
```
> **Enforced Rule**: `Delivery Status` and `Days for shipping (real)` **MUST REMAIN EXCLUDED** from model features because they represent post-shipment outcomes. Including them creates circular target leakage.

### Feature Matrix (22 Input Dimensions)
The pipeline extracts only features known **at the time the order is placed**:
- **10 Numerical Features** (`StandardScaler` applied):
  `Days for shipment (scheduled)`, `Benefit per order`, `Sales per customer`, `Order Item Discount`, `Order Item Discount Rate`, `Order Item Product Price`, `Order Item Profit Ratio`, `Order Item Quantity`, `Sales`, `Product Price`.
- **3 Categorical Features** (`OneHotEncoder` applied → 12 one-hot features):
  `Shipping Mode` (4 categories), `Customer Segment` (3 categories), `Market` (5 categories).
- **Split Strategy**: Stratified 80/20 train/test split.
  - **Training Set**: 144,415 rows × 22 features
  - **Test Set**: 36,104 rows × 22 features

---

## 3. Person B Model Verification: Gaussian Naive Bayes

Implemented in `src/models/naive_bayes.py`.

### Mathematical Formulation
1. **Class Priors**:
   $$P(Y=1) = 0.5483, \quad P(Y=0) = 0.4517$$
2. **Likelihood Computation**:
   $$P(x_j \mid y = c) = \frac{1}{\sqrt{2\pi (\sigma_{c,j}^2 + \epsilon)}} \exp\left( -\frac{(x_j - \mu_{c,j})^2}{2(\sigma_{c,j}^2 + \epsilon)} \right)$$
   *(with variance smoothing $\epsilon = 10^{-9} \cdot \max(\sigma^2)$ to prevent division by zero).*
3. **Log-Space Posterior**:
   $$\log P(y = c \mid x) = \ln P(y = c) - \frac{1}{2} \sum_{j=1}^{d} \left[ \ln(2\pi \sigma_{c,j}^2) + \frac{(x_j - \mu_{c,j})^2}{\sigma_{c,j}^2} \right]$$

### Benchmark Comparison against `scikit-learn`
Evaluated on the 36,104 test samples:

| Evaluation Metric | Scratch Gaussian NB | Scikit-Learn Gaussian NB | Parity Check |
| :--- | :---: | :---: | :---: |
| **Prediction Agreement** | **`100.00%`** | **`100.00%`** | **Exact match** |
| **Max Prob Difference** | **`0.000000`** | **`0.000000`** | **Exact machine precision** |
| **Accuracy** | 69.63% | 69.63% | Identical |
| **Precision ($Y=1$, Late)** | **84.85%** | 84.85% | Identical |
| **Recall ($Y=1$, Late)** | 54.30% | 54.30% | Identical |
| **F1-Score ($Y=1$, Late)** | 66.22% | 66.22% | Identical |
| **Training Time (144k rows)** | **0.099 s** | 0.103 s | **Scratch is faster** |

### Threshold Tuning for Procurement (Recall Optimization)
In procurement, false alarms are cheap (an extra inspection), while missing a late vendor shuts down assembly lines. Lowering the decision threshold improves sensitivity:

| Decision Threshold | High-Risk Recall ($Y=1$) | Precision ($Y=1$) | Procurement Application |
| :---: | :---: | :---: | :--- |
| **0.30** | **58.7%** | 79.7% | Aggressive supplier audit mode |
| **0.40** | 54.8% | 83.9% | Balanced risk mitigation |
| **0.50 (Default)** | 54.3% | 84.8% | Standard operations |

---

## 4. Work Remaining: Action Items for Person A

Person A is responsible for the **Discriminative Model (Vectorized Logistic Regression)**. The project scaffolding is already built and waiting in the repo.

### Technical Specification for Person A

#### 1. File to Implement
`src/models/logistic_regression.py`
A vectorized template is already written. Person A needs to verify and tune:
- **Hypothesis**: $\hat{y} = \sigma(Xw + b) = \frac{1}{1 + e^{-(Xw + b)}}$
- **Cost Function (Binary Cross-Entropy with L2 regularization)**:
  $$J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \ln(\hat{y}^{(i)}) + (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)}) \right] + \frac{\lambda}{2m} \|w\|_2^2$$
- **Vectorized Gradients**:
  $$\frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{y} - y) + \frac{\lambda}{m} w, \quad \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$
- **Updates**:
  $$w := w - \alpha \frac{\partial J}{\partial w}, \quad b := b - \alpha \frac{\partial J}{\partial b}$$

#### 2. Hyperparameters to Tune
- Learning rate $\alpha \in [0.01, 0.1, 0.5]$
- Iterations: $1,000 - 3,000$ epochs
- L2 penalty $\lambda \in [0.0, 0.01, 0.1]$

#### 3. Specific Deliverables Expected from Person A
1. **Convergence Plot**: Plot $J(w, b)$ vs. iteration number to prove gradient descent converges.
2. **Parity Table**: Benchmark Scratch Logistic Regression vs. `sklearn.linear_model.LogisticRegression`.
3. **Feature Importance (Odds Ratios)**:
   - Calculate $\text{Odds Ratio} = \exp(w_j)$ for each of the 22 features.
   - List the **Top 3 features that increase late delivery risk** and **Top 3 features that reduce risk**.

---

## 5. Joint Tasks Remaining for Final Submission

| Task | Person A | Person B | Joint |
| :--- | :---: | :---: | :---: |
| Logistic Regression Scratch & sk-learn parity | 🌟 Lead | Reviewer | |
| Loss curve & Odds ratio extraction | 🌟 Lead | Reviewer | |
| Gaussian Naive Bayes Scratch & sk-learn parity | | Completed | ✅ |
| Colab EDA & Preprocessing pipeline | | Completed | ✅ |
| Streamlit Web App (`app/streamlit_app.py`) | | Completed | ✅ |
| **Comparative Analysis Writeup** (Generative vs Discriminative) | Contributor | Contributor | 🌟 Joint |
| **Final README Polish & Streamlit Cloud Launch** | Contributor | Contributor | 🌟 Joint |
