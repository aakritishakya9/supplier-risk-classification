# 🎓 Master Study & Defense Notes: Supplier Risk & Quality Classification

> **Purpose of these Notes**: Read this document to thoroughly master the project concepts, understand the code line-by-line, explain the mathematical algorithms from scratch, and justify your experimental results in any academic evaluation, viva, or technical interview.

---

## 📌 MODULE 1: The 30-Second Executive Summary

If an examiner or interviewer asks: *"What is your project about, and what did you find?"*, here is your concise summary:

> *"We built an end-to-end machine learning system to predict whether a supplier order will be delayed (**High Risk**) or delivered on schedule (**Low Risk**) using 180,000 real global supply chain transactions from the DataCo dataset.*  
>  
> *Because a late delivery can halt a factory assembly line, we prioritized **High-Risk Recall** over simple accuracy and eliminated post-shipment target leakages.*  
>  
> *We implemented two contrasting machine learning paradigms completely from scratch in NumPy: **Gaussian Naive Bayes** (Generative) and **Vectorized Logistic Regression** (Discriminative), benchmarked against Scikit-Learn.*  
>  
> *Our key finding: **Logistic Regression significantly outperformed Naive Bayes on High-Risk Recall ($70.6\%$ vs $51.5\%$) and ROC-AUC ($0.770$ vs $0.739$)**. This occurred because real supply chain features (such as rush shipping modes and short scheduled lead times) are strongly collinear, violating Naive Bayes' independence assumption. However, Naive Bayes trained over $60\times$ faster ($0.006\text{s}$ vs $0.414\text{s}$)."*

---

## 📌 MODULE 2: Your Algorithm — Gaussian Naive Bayes (Person B)

### 1. Intuitive Mental Model
Imagine a detective with two suspect files:
- **File 0 (On-Time Shipments)**
- **File 1 (Late Shipments)**

When a new order comes in, the model doesn't draw a line between them; instead, it asks:  
👉 *"Does this order look more like the historical profile of an On-Time shipment, or a Late shipment?"*

---

### 2. The Core Mathematics

#### A. Class Prior Probabilities
The baseline probability of an event before looking at any features:
$$P(Y = 1) = \frac{N_{\text{late}}}{N_{\text{total}}} \approx 0.5483 \quad (54.83\%)$$
$$P(Y = 0) = \frac{N_{\text{on-time}}}{N_{\text{total}}} \approx 0.4517 \quad (45.17\%)$$

#### B. The Gaussian Likelihood Function
For continuous numbers (like price or scheduled days), we model the feature as a normal bell curve $\mathcal{N}(\mu, \sigma^2)$:
$$P(x_j \mid Y = c) = \frac{1}{\sqrt{2\pi (\sigma_{c,j}^2 + \epsilon)}} \exp\left( -\frac{(x_j - \mu_{c,j})^2}{2(\sigma_{c,j}^2 + \epsilon)} \right)$$
- $\mu_{c,j}$: The average value of feature $j$ for class $c$.
- $\sigma_{c,j}^2$: The variance (spread) of feature $j$ for class $c$.
- $\epsilon = 10^{-9} \cdot \max(\sigma^2)$: **Variance smoothing** added to the denominator to prevent division-by-zero on constant columns.

#### C. The "Naive" Assumption & Log-Space Computation
Naive Bayes assumes all features are conditionally independent:
$$P(X \mid Y = c) = \prod_{j=1}^{d} P(x_j \mid Y = c)$$
Multiplying hundreds of small decimal probabilities causes **floating-point arithmetic underflow** (the number becomes $0.000000000$ and vanishes).  
To prevent this, we take the natural logarithm ($\ln$), converting multiplication into **addition**:
$$\ln P(Y = c \mid X) \propto \ln P(Y = c) + \sum_{j=1}^{d} \ln P(x_j \mid Y = c)$$
Expanding the Gaussian formula in log-space gives:
$$\ln P(Y = c \mid X) \propto \ln P(Y = c) - \frac{1}{2} \sum_{j=1}^{d} \left[ \ln(2\pi \sigma_{c,j}^2) + \frac{(x_j - \mu_{c,j})^2}{\sigma_{c,j}^2} \right]$$

#### D. Feature Explainability via Log-Likelihood Ratio (LLR)
To explain to a procurement manager why a shipment was flagged, we calculate:
$$\text{LLR}_j(x_j) = \ln \frac{P(x_j \mid Y=1)}{P(x_j \mid Y=0)} = -\frac{1}{2}\ln\left(\frac{\sigma_{1,j}^2}{\sigma_{0,j}^2}\right) - \frac{(x_j - \mu_{1,j})^2}{2\sigma_{1,j}^2} + \frac{(x_j - \mu_{0,j})^2}{2\sigma_{0,j}^2}$$
- If $\text{LLR}_j > 0$: Feature $j$ pushed the probability towards **Late**.
- If $\text{LLR}_j < 0$: Feature $j$ pulled the probability towards **On-Time**.

---

### 3. Step-by-Step Code Walkthrough (`src/models/naive_bayes.py`)

```python
class ScratchGaussianNB:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
```
- Stores the hyperparameter $\epsilon$ for numerical safety.

```python
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y) # [0, 1]
        ...
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]                                  # Filter rows belonging to class c
            self.class_priors_[idx] = X_c.shape[0] / n_samples # Calculate P(Y=c)
            self.means_[idx, :] = np.mean(X_c, axis=0)       # Mean vector mu
            self.variances_[idx, :] = np.var(X_c, axis=0) + epsilon # Variance sigma^2
```
- **Why this is fast**: It only scans the dataset **once** ($O(N \cdot D)$ time complexity). No gradient loops, no epochs.

```python
    def predict_proba(self, X):
        log_posteriors = self._calculate_log_likelihood(X)
        # Log-Sum-Exp Trick: subtract max to prevent overflow when exponentiating
        max_log = np.max(log_posteriors, axis=1, keepdims=True)
        exp_post = np.exp(log_posteriors - max_log)
        return exp_post / np.sum(exp_post, axis=1, keepdims=True)
```
- Converts unnormalized log scores back into genuine probabilities that sum to $1.0$.

---

## 📌 MODULE 3: The Competitor — Vectorized Logistic Regression (Person A)

### 1. Intuitive Mental Model
Logistic Regression doesn't build profiles. Instead, it acts like a scale:
- Each feature is given a weight $w_j$.
- It multiplies each order parameter by its weight, adds them up ($z = Xw + b$), and passes the score through an **S-curve (Sigmoid)** to output a probability between $0\%$ and $100\%$.

---

### 2. The Core Mathematics

#### A. Sigmoid Activation Function
$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad \text{where } z = Xw + b$$
- If $z \gg 0 \rightarrow \hat{y} \to 1.0$ (Late).
- If $z \ll 0 \rightarrow \hat{y} \to 0.0$ (On-Time).

#### B. Binary Cross-Entropy Loss with L2 Regularization
$$J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \ln(\hat{y}^{(i)}) + (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)}) \right] + \frac{\lambda}{2m} \|w\|_2^2$$
- Penalizes the model heavily when it is confident and wrong.
- $\frac{\lambda}{2m} \|w\|_2^2$: Shrinks weights to prevent overfitting on outliers.

#### C. Vectorized Gradients
$$\frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{y} - y) + \frac{\lambda}{m} w, \quad \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$
- Computed across all 144,000 samples simultaneously using matrix multiplication (`np.dot`).
- Parameters updated by step size $\alpha$:  
  $$w := w - \alpha \frac{\partial J}{\partial w}, \quad b := b - \alpha \frac{\partial J}{\partial b}$$

#### D. Feature Explainability via Odds Ratios
$$\text{Odds Ratio}_j = \exp(w_j)$$
- If $w_j = +0.69 \rightarrow \text{Odds Ratio} = e^{0.69} \approx 2.0$.  
  *Meaning: Having this feature **doubles** the odds of delivery delay.*

---

## 📌 MODULE 4: The Scientific Analysis: "WHY Did We Get These Results?"

This is the most critical section for oral exams and reports.

```
                         HEAD-TO-HEAD RESULTS SUMMARY
┌──────────────────────────────┬────────────────────────┬────────────────────────┐
│ Metric                       │ Scratch Naive Bayes    │ Scratch Logistic Reg   │
├──────────────────────────────┼────────────────────────┼────────────────────────┤
│ Accuracy                     │ 64.00%                 │ 70.37% (Winner)        │
│ High-Risk Recall             │ 51.49%                 │ 70.61% (Winner)        │
│ Precision                    │ 70.78%                 │ 71.58%                 │
│ ROC-AUC                      │ 0.7389                 │ 0.7698 (Winner)        │
│ Training Time                │ 0.006 seconds (Fastest)│ 0.414 seconds          │
│ Sklearn Parity Match         │ 100.00% exact match    │ 99.20% match           │
└──────────────────────────────┴────────────────────────┴────────────────────────┘
```

### Question 1: "Why did Logistic Regression get much higher Recall (70.6% vs 51.5%)?"
*The Answer*: **Feature Collinearity (Interdependence)**.
- Naive Bayes assumes $P(X \mid Y) = P(x_1 \mid Y) \cdot P(x_2 \mid Y) \cdots P(x_d \mid Y)$.
- In supply chain procurement, features are **not independent**:
  - `Shipping Mode = First Class` almost always correlates with `Scheduled Days = 1 or 2`.
  - When features correlate, Naive Bayes double-counts evidence or artificially distorts probability distributions into axis-aligned ellipses.
- **Logistic Regression makes no independence assumptions**. It directly optimizes a single linear decision hyperplane $\sigma(w^T x + b)$ that accounts for collinearity between shipping tiers and lead times. Therefore, it caught **$70.6\%$ of all late deliveries (~20% more than Naive Bayes)**.

---

### Question 2: "Why was Gaussian Naive Bayes so much faster (0.006s vs 0.414s)?"
*The Answer*: **Closed-Form Statistics vs. Iterative Optimization**.
- Naive Bayes has a **closed-form solution**: it computes the arithmetic mean and variance of columns in a single matrix pass ($O(N \cdot D)$).
- Logistic Regression has **no closed-form formula**: it must run an iterative loop of **400 epochs of gradient descent**, evaluating matrix multiplications and cost gradients on every iteration ($O(\text{epochs} \cdot N \cdot D)$).

---

### Question 3: "Why did both models achieve ~100% parity with Scikit-Learn?"
*The Answer*: **Exact Numerical Formulation**.
- For Naive Bayes, our log-posterior Gaussian density formula matches Scikit-Learn's `GaussianNB` implementation down to machine floating-point precision ($0.000000$ difference).
- For Logistic Regression, our binary cross-entropy gradient matches Scikit-Learn's L2-regularized objective ($99.20\%$ agreement, with minor differences only due to solver algorithms: our vanilla Gradient Descent vs Scikit-Learn's L-BFGS quasi-Newton solver).

---

### Question 4: "Why does lowering the threshold to 0.30 help the business?"
*The Answer*: **Risk-Averse Procurement Policy**.
- Default classification cuts off at probability $0.50$.
- In procurement, missing a bad vendor (False Negative) can shut down an entire factory.
- By shifting the threshold slider down to **$0.30$**, the model sounds the alarm if it is even $30\%$ suspicious. This boosted late delivery detection up to **$\sim 78\%$**, trading off a few unnecessary 10-minute supplier audits in exchange for preventing factory stockouts.

---

## 📌 MODULE 5: Code Architecture Summary

| File Path | Role & Purpose | Key Concepts Inside |
| :--- | :--- | :--- |
| `src/preprocessing.py` | Cleaning & transforms | Stripping `Delivery Status` & `Days for shipping (real)` leakages; `StandardScaler`; `OneHotEncoder`. |
| `src/models/naive_bayes.py` | Generative Classifier | Class priors, Gaussian likelihoods, variance smoothing $\epsilon$, Log-Sum-Exp trick, Log-Likelihood Ratio (LLR). |
| `src/models/logistic_regression.py` | Discriminative Classifier | Sigmoid, Binary Cross-Entropy loss, L2 penalty, vectorized gradient updates, early stopping, Odds Ratios $\exp(w_j)$. |
| `src/evaluate.py` | Metrics & Diagnostics | Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix, Threshold Curve generation. |
| `src/train_models.py` | Master Benchmark Runner | Trains both models, computes head-to-head scorecard, verifies Sklearn parity, saves plots in `reports/figures/`. |
| `app/streamlit_app.py` | Interactive Web Dashboard | UI sliders for vendor parameters, model selector dropdown, dynamic risk badge (🟢/🔴), live feature attribution bar chart. |
| `tests/` | Automated Test Suite | 10 unit tests in `test_naive_bayes.py` and `test_logistic.py` testing probability normalization, parity, and numerical stability. |

---

## 📌 MODULE 6: Fast Revision Cheat Sheet (Memorize Before Your Viva)

1. **Dataset**: DataCo Smart Supply Chain ($180,519$ rows, $54.8\%$ Late vs $45.2\%$ On-Time).
2. **Leakage**: Dropped `Delivery Status` and `Days for shipping (real)` to prevent artificial 100% accuracy.
3. **Primary Metric**: **High-Risk Recall** (catching late shipments).
4. **Best Model**: **Logistic Regression** ($70.6\%$ Recall, $70.4\%$ Accuracy, $0.770$ ROC-AUC).
5. **Fastest Model**: **Gaussian Naive Bayes** ($0.006$ seconds).
6. **Code Proof**: 10 unit tests passing in $0.27$ seconds with over $99\%$ Scikit-Learn parity.
7. **Business Recommendation**: Deploy Logistic Regression with a decision threshold of **$0.30$** to catch ~8 out of 10 late suppliers.
