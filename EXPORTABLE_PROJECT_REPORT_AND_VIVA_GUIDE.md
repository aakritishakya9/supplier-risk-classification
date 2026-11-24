# Supplier Risk & Late Delivery Classification
## Comprehensive Project Report, Code Walkthrough, and Defense Guide

> **Document Type**: Project Report & Viva Defense Handbook  
> **Authors**: Person A (Logistic Regression Lead) & Person B (Gaussian Naive Bayes Lead)  
> **Repository**: `github.com/aakritishakya9/supplier-risk-classification`  
> **Dataset**: DataCo Smart Supply Chain Dataset (~180,519 records)  
> **Tech Stack**: Python 3.12, NumPy (Scratch Implementations), Pandas, Scikit-Learn, Streamlit, Matplotlib, Seaborn  

---

## Table of Contents
1. [Executive Summary (Simple Explanation)](#1-executive-summary-simple-explanation)
2. [Business Problem & Procurement Context](#2-business-problem--procurement-context)
3. [Dataset & Target Leakage Safeguard](#3-dataset--target-leakage-safeguard)
4. [Tool & Technology Stack Justifications](#4-tool--technology-stack-justifications)
5. [In-Depth Code Walkthrough](#5-in-depth-code-walkthrough)
   - [5.1 `src/preprocessing.py` (Data Pipeline)](#51-srcpreprocessingpy-the-data-pipeline)
   - [5.2 `src/models/naive_bayes.py` (Scratch Gaussian Naive Bayes)](#52-srcmodelsnaive_bayespy-gaussian-naive-bayes-from-scratch)
   - [5.3 `src/models/logistic_regression.py` (Scratch Logistic Regression)](#53-srcmodelslogistic_regressionpy-logistic-regression-from-scratch)
   - [5.4 `src/evaluate.py` & `src/train_models.py` (Benchmarking & Visuals)](#54-srcevaluatepy--srctrain_modelspy-evaluation--plotting)
   - [5.5 `app/streamlit_app.py` (Interactive Web Dashboard)](#55-appstreamlit_apppy-the-interactive-web-dashboard)
6. [Experimental Findings & Model Benchmark](#6-experimental-findings--model-benchmark)
7. [Why Logistic Regression Defeated Naive Bayes](#7-why-logistic-regression-defeated-naive-bayes)
8. [Master Viva & Defense Q&A Cheatsheet](#8-master-viva--defense-qa-cheatsheet)

---

## 1. Executive Summary (Simple Explanation)

### 1.1 What is this project and why did we build it?
Imagine running a smartphone assembly plant. To keep the factory running, you order thousands of sensitive microchips, displays, and battery modules from vendors across the globe daily.
* If a delivery arrives on schedule, the assembly line hums along smoothly.
* If a delivery is delayed by even 48 hours, the entire assembly line freezes. Workers sit idle, customer delivery promises are broken, and the firm bleeds thousands of dollars per hour.

**Our Solution**: We engineered an operational Machine Learning decision-support system for procurement managers. Before a purchase order is dispatched, our system evaluates parameters known at the moment of order placement (shipping mode, scheduled transit days, item value, discount level, destination market) and answers one critical question:
> **"Is this shipment at High Risk (🔴 Late) or Low Risk (🟢 On-Time)?"**

### 1.2 The Two AI Paradigms Built from Scratch in NumPy
Rather than relying on black-box pre-built libraries, we engineered both machine learning paradigms from mathematical first principles in pure `NumPy`:
1. **Gaussian Naive Bayes (The Profile Matcher - Generative Paradigm)**:  
   Constructs two independent probability profiles: *"What does a typical on-time order look like?"* vs. *"What does a typical late order look like?"* When an incoming order arrives, it measures which profile's distribution it matches closer.
2. **Logistic Regression (The Scorekeeper - Discriminative Paradigm)**:  
   Directly seeks an optimal multidimensional decision hyperplane separating safe orders from risky ones. It assigns weights to every clue (e.g., *Rush First-Class shipping adds risk points; ample scheduled days subtract risk points*), runs the sum through a sigmoid function, and computes a calibrated risk percentage ($0\%$ to $100\%$).

### 1.3 Who Won and What Did We Learn?
On a test set of **36,104 real shipments**:
* **Logistic Regression defeated Naive Bayes convincingly**:
  * **High-Risk Recall**: **70.61%** vs. **51.49%** (Caught $\sim 20\%$ more late deliveries!).
  * **Overall Accuracy**: **70.37%** vs. **64.00%**.
  * **ROC-AUC**: **0.770** vs. **0.739**.
* **The Scientific Reason**: In supply chain networks, features are **collinear** (not independent). Shipping mode and scheduled transit days are strongly linked (e.g., First Class is paired with short 1-day windows). Naive Bayes' fundamental assumption of conditional independence broke down, causing it to miscalibrate probabilities and miss late shipments. Logistic Regression modeled these interrelated weights simultaneously without assuming independence.

---

## 2. Business Problem & Procurement Context

### 2.1 The Asymmetric Cost of Errors
In enterprise supply chains, prediction errors have profoundly asymmetric financial costs:

```
                    ┌────────────────────────────────────────────────────────┐
                    │               Actual Ground Truth Delivery             │
                    │               LATE (1)         │      ON-TIME (0)      │
┌───────────────────┼────────────────────────────────┼───────────────────────┤
│ Predicted LATE (1)│       True Positive (TP)       │  False Positive (FP)  │
│                   │   Late correctly flagged;      │  Safe order flagged;  │
│                   │   contingency plan activated.  │  minor audit cost.    │
├───────────────────┼────────────────────────────────┼───────────────────────┤
│ Predicted SAFE (0)│      False Negative (FN)       │  True Negative (TN)   │
│                   │   🚨 CATASTROPHIC LOSS:        │  Smooth operations;   │
│                   │   Factory halts, stockouts.    │  zero friction.       │
└───────────────────┴────────────────────────────────┴───────────────────────┘
```

* **False Negative (FN)**: Model says *"Safe"*, but the order arrives late. Assembly lines shut down, emergency air freight is booked at 5× cost, and SLAs are breached.
* **False Positive (FP)**: Model says *"Risky"*, but the order would have arrived on time. A procurement specialist spends 10 minutes making a confirmation phone call.

$$\text{Financial Loss}(\text{FN}) \gg \text{Administrative Cost}(\text{FP})$$

Therefore, standard **Accuracy is misleading**. Our primary optimization metric is **High-Risk Recall**:

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

---

## 3. Dataset & Target Leakage Safeguard

### 3.1 Dataset Profile
* **Dataset**: DataCo Smart Supply Chain Dataset
* **Observations**: 180,519 rows across 53 features
* **Target Feature**: `Late_delivery_risk` ($1 = \text{Late}, 0 = \text{On-Time}$)
* **Class Balance**: 54.83% Late vs. 45.17% On-Time (Naturally balanced; no synthetic SMOTE oversampling required)

### 3.2 Target Leakage Safeguard
A major pitfall in predictive modeling is **Target Leakage**—introducing features into training that contain information generated *after* the target event occurred.
In DataCo, two features caused 100% artificial accuracy:
1. `Days for shipping (real)`: Records the exact transit duration observed *after* delivery.
2. `Delivery Status`: Explicitly categorized as *"Late delivery"*, *"Advance shipping"*, etc.

If left in, any classifier trivially learns:
$$\text{If } \text{Days for shipping (real)} > \text{Days for shipment (scheduled)} \implies \text{Late}$$

We excised both columns entirely, restricting the feature space solely to parameters known at purchase order creation:
* `Days for shipment (scheduled)`
* `Shipping Mode` (Standard, First Class, Second Class, Same Day)
* `Order Item Product Price`, `Order Item Discount Rate`, `Order Item Quantity`, `Sales`
* `Order Item Profit Ratio`
* `Customer Segment` (Consumer, Corporate, Home Office)
* `Market` (Pacific Asia, USCA, Europe, LATAM, Africa)

---

## 4. Tool & Technology Stack Justifications

| Tool / Technology | Role in Project | Engineering Justification |
| :--- | :--- | :--- |
| **Python 3.12** | Core Language | Universal standard for scientific computing and machine learning. |
| **NumPy** | Scratch ML Engine | Provides C-speed vectorized array operations. Critical for writing gradient descent, log-sum-exp, and covariance matrices without library shortcuts. |
| **Pandas** | Data Wrangling | High-performance tabular data manipulation, leakage removal, and preprocessing verification. |
| **Scikit-Learn** | Pipeline & Benchmark | Used strictly for data scaling (`StandardScaler`), one-hot encoding (`OneHotEncoder`), and stratified splitting, as well as the gold-standard baseline to verify scratch model correctness. |
| **Matplotlib & Seaborn** | Visualization | Generated publication-quality figures: Confusion matrices, ROC curves, Precision-Recall curves, and training loss convergence curves. |
| **Streamlit** | UI Deployment | Allowed rapid construction of an interactive web application with sliders and dynamic attribution plots in pure Python. |
| **Google Colab** | Cloud EDA & Training | Executed heavy exploratory data analysis on the full 180,000-row dataset in the cloud without burdening local machine RAM. |
| **Hugging Face Hub** | Dataset Mirror | Provided a permanent public download URL bypassing Kaggle API credential restrictions in cloud environments. |
| **Python `unittest`** | Automated Testing | 10 automated unit tests guaranteeing numerical stability, parameter shapes, and mathematical parity against Scikit-Learn. |
| **Git & GitHub** | Version Control | Maintained clean commit history, role separation (Person A vs. Person B), and code reproducibility. |

---

## 5. In-Depth Code Walkthrough

```
supplier-risk-classification/
├── src/
│   ├── preprocessing.py         # Leakage removal, scaling & one-hot encoding
│   ├── evaluate.py              # Metric calculation, ROC/PR curves & threshold sweep
│   ├── train_models.py          # Master training, parity benchmarking & figure generation
│   └── models/
│       ├── naive_bayes.py       # Scratch Gaussian Naive Bayes + Log-Likelihood Ratio
│       └── logistic_regression.py # Scratch Vectorized Logistic Regression + Odds Ratios
├── tests/
│   ├── test_naive_bayes.py      # Automated unit tests for Naive Bayes
│   └── test_logistic.py         # Automated unit tests for Logistic Regression
└── app/
    └── streamlit_app.py         # Interactive risk dashboard with real-time attribution
```

---

### 5.1 `src/preprocessing.py` (The Data Pipeline)

#### Key Responsibilities
1. **Target Leakage Removal**: Drops `Days for shipping (real)` and `Delivery Status`.
2. **Column Categorization**: Separates numeric inputs from categorical inputs.
3. **ColumnTransformer Construction**:
   - `StandardScaler()` applied to numeric features:
     $$z = \frac{x - \mu}{\sigma}$$
     *Why?* Prevents high-magnitude values (like \$500 Sales) from skewing gradient steps compared to small values (like 2 scheduled days).
   - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` applied to categories.
4. **Stratified Split**: 80/20 train-test split maintaining identical 55% late class proportions in both splits.

```python
# Snippet from src/preprocessing.py
def build_preprocessor():
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor
```

---

### 5.2 `src/models/naive_bayes.py` (Gaussian Naive Bayes from Scratch)

#### Mathematical Formulation
Bayes' Theorem for classification:

$$P(Y = c \mid X) = \frac{P(Y = c) \prod_{j=1}^{d} P(x_j \mid Y = c)}{P(X)}$$

For continuous features, each feature likelihood follows a Gaussian (normal) distribution:

$$P(x_j \mid Y = c) = \frac{1}{\sqrt{2\pi\sigma_{c,j}^2}} \exp\left( -\frac{(x_j - \mu_{c,j})^2}{2\sigma_{c,j}^2} \right)$$

#### Numerical Stability Safeguards in Code
1. **Variance Smoothing ($\epsilon = 10^{-9}$)**:
   Added to feature variances to prevent division by zero for features with zero spread:
   ```python
   self.variances_[idx, :] = np.var(X_c, axis=0) + self.epsilon
   ```
2. **Log-Space Computation**:
   Multiplying hundreds of small probabilities causes IEEE-754 floating-point underflow to zero. We convert products into sums of logarithms:
   $$\ln P(Y = c \mid X) \propto \ln P(Y = c) + \sum_{j=1}^{d} \left[ -\frac{1}{2} \ln(2\pi\sigma_{c,j}^2) - \frac{(x_j - \mu_{c,j})^2}{2\sigma_{c,j}^2} \right]$$
   ```python
   # Vectorized log-likelihood calculation
   log_likelihood = -0.5 * np.sum(np.log(2.0 * np.pi * var)) - 0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
   log_posteriors[:, idx] = log_prior + log_likelihood
   ```
3. **Log-Sum-Exp Normalization for Probabilities**:
   Converts log-posteriors back into proper $[0, 1]$ probabilities safely without overflow:
   ```python
   max_log = np.max(log_posteriors, axis=1, keepdims=True)
   exp_posteriors = np.exp(log_posteriors - max_log)
   return exp_posteriors / np.sum(exp_posteriors, axis=1, keepdims=True)
   ```
4. **Log-Likelihood Ratio (LLR) Feature Attribution**:
   Measures the log-evidence that feature $j$ contributes toward the late class ($c=1$) versus on-time ($c=0$):
   $$\text{LLR}_j = \ln P(x_j \mid Y = 1) - \ln P(x_j \mid Y = 0)$$
   If $\text{LLR}_j > 0$, feature $j$ pushed the model toward predicting Late.

---

### 5.3 `src/models/logistic_regression.py` (Logistic Regression from Scratch)

#### Mathematical Formulation
Linear score:

$$z = Xw + b$$

Sigmoid activation:

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Binary Cross-Entropy Loss with $L_2$ Regularization:

$$J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \ln(\hat{y}^{(i)}) + (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)}) \right] + \frac{\lambda}{2m} \|w\|_2^2$$

Analytical Gradients:

$$\frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{y} - y) + \frac{\lambda}{m} w$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$

#### Numerical Stability Safeguards in Code
1. **Branching Sigmoid (`np.where`)**:
   Standard $\frac{1}{1 + e^{-z}}$ overflows when $z < -709$. We branch dynamically:
   ```python
   def _sigmoid(self, z):
       z = np.clip(z, -500.0, 500.0)
       return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
   ```
2. **Loss Clipping**:
   Probabilities are clipped to $[\epsilon, 1-\epsilon]$ ($\epsilon = 10^{-15}$) before passing to $\ln()$ to eliminate $\ln(0) = -\infty$.
3. **Vectorized Training Loop with Early Stopping**:
   Computes matrix operations across all samples simultaneously with convergence monitoring:
   ```python
   for epoch in range(self.n_iters):
       z = np.dot(X, self.weights) + self.bias
       y_pred = self._sigmoid(z)
       # Gradient update
       dw = (1.0 / n_samples) * np.dot(X.T, (y_pred - y)) + (self.l2_penalty / n_samples) * self.weights
       db = (1.0 / n_samples) * np.sum(y_pred - y)
       self.weights -= self.learning_rate * dw
       self.bias -= self.learning_rate * db
       # Early stopping check
       if abs(prev_loss - current_loss) < self.tol:
           break
   ```
4. **Odds Ratios Feature Interpretability**:
   $$\text{Odds Ratio}_j = \exp(w_j)$$
   * $\text{Odds Ratio} = 1.80 \implies$ increases odds of late delivery by $80\%$.
   * $\text{Odds Ratio} = 0.40 \implies$ reduces odds of late delivery by $60\%$ (protective factor).

---

### 5.4 `src/evaluate.py` & `src/train_models.py` (Evaluation & Plotting)

* Evaluates accuracy, precision, recall, F1-score, and ROC-AUC.
* Automatically plots and saves to `reports/figures/`:
  1. `confusion_matrices_scratch.png` (Side-by-side heatmaps)
  2. `roc_curves_comparison.png`
  3. `precision_recall_curves.png`
  4. `logistic_loss_curve.png` (Proves gradient descent convergence)
* Generates a threshold tuning table sweeping thresholds from $0.20$ to $0.70$.

---

### 5.5 `app/streamlit_app.py` (The Interactive Web Dashboard)

* **Architecture**: Fully modular pure-Python web interface.
* **Model Toggle**: Allows switching between Scratch Logistic Regression, Scratch Gaussian Naive Bayes, and Scikit-Learn.
* **Threshold Slider**: Lets users adjust decision cutoff from $0.20$ to $0.80$ (default $0.50$).
* **Real-time Attribution**: Displays an interactive horizontal bar chart depicting exactly which factors pushed the score up or down.

---

## 6. Experimental Findings & Model Benchmark

### 6.1 Head-to-Head Comparison

| Metric | Scratch Gaussian Naive Bayes | Scratch Logistic Regression | Scikit-Learn Baseline | Winner & Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Model Type** | Generative | Discriminative | Industry Standard | — |
| **Scikit-Learn Parity** | **100.00%** | **99.20%** | Benchmark | Proves scratch code correctness |
| **Accuracy** | 64.00% | **70.37%** | 70.41% | 🏆 **Logistic Regression** (+6.4% higher) |
| **High-Risk Recall** | 51.49% | **70.61%** | 70.64% | 🏆 **Logistic Regression** (+19.1% higher) |
| **Precision** | 70.78% | **71.58%** | 71.60% | 🏆 **Logistic Regression** |
| **F1-Score** | 0.5961 | **0.7109** | 0.7112 | 🏆 **Logistic Regression** |
| **ROC-AUC** | 0.7389 | **0.7698** | 0.7701 | 🏆 **Logistic Regression** |
| **Training Speed** | **0.006s** ⚡ | 0.414s | 0.280s | ⚡ **Naive Bayes** (60× faster) |
| **Explainability** | Log-Likelihood Ratio | Odds Ratios $\exp(w)$ | Coefficients | Both provide clear attributions |

### 6.2 Top Feature Drivers of Supplier Risk

| Feature | Model Weight ($w_j$) | Odds Ratio $\exp(w_j)$ | Practical Interpretation |
| :--- | :---: | :---: | :--- |
| **Shipping Mode: First Class** | $+1.78$ | **$5.93\times$** | Massive risk driver; rush shipping severely compresses handling time. |
| **Shipping Mode: Same Day** | $+1.52$ | **$4.57\times$** | Extreme risk driver; zero operational margin for logistics hiccups. |
| **Order Item Discount Rate** | $+0.31$ | **$1.36\times$** | High discount orders correlate with flash sales and warehouse congestion. |
| **Days for shipment (scheduled)** | $-0.84$ | **$0.43\times$** | Strong protective factor; each extra scheduled day cuts delay odds by 57%. |
| **Shipping Mode: Standard Class** | $-1.15$ | **$0.32\times$** | Strong protective factor; predictable carrier routing schedules. |

---

## 7. Why Logistic Regression Defeated Naive Bayes

### The Core Flaw in Naive Bayes: Violation of Feature Independence
Gaussian Naive Bayes relies on the foundational assumption:

$$P(x_1, x_2, \dots, x_d \mid Y) = \prod_{j=1}^{d} P(x_j \mid Y)$$

In supply chain operations, this assumption is **demonstrably false**:
* `Shipping Mode = First Class` directly implies `Days for shipment (scheduled) = 1 day`.
* Because these features are **collinear**, Naive Bayes counts the same underlying evidence twice:
  1. It penalizes the shipment for being *First Class*.
  2. It penalizes the shipment again for having *1 Scheduled Day*.

This leads to severe probability miscalibration, pulling predictions toward extremes and causing the model to miss **48.5% of all late shipments** (51.49% recall).

### Why Logistic Regression Succeeded
Logistic Regression is a **discriminative** model that does not model feature distributions or assume independence:
* It solves a joint optimization problem across all features simultaneously.
* If two features share redundant information, gradient descent naturally splits and scales their weights ($w_j$).
* As a result, it accurately captured the interaction between rush delivery modes and tight schedules, catching **70.61% of late shipments**.

---

## 8. Master Viva & Defense Q&A Cheatsheet

### Q1: Why did you implement these models from scratch instead of just importing Scikit-Learn?
> **Answer**: Implementing from scratch in pure NumPy proves an exact mathematical understanding of how machine learning works under the hood—including vectorized matrix calculus, numerical stability (Log-Sum-Exp, branching sigmoid), and optimization (gradient descent). We used Scikit-Learn as an objective verification benchmark, achieving over 99.2% parity.

### Q2: What is the fundamental difference between Generative and Discriminative models?
> **Answer**: A **Generative model** (Gaussian Naive Bayes) learns the joint probability distribution $P(X, Y)$ by modeling how each class generates features: *"What does an on-time shipment look like vs. a late shipment?"*  
> A **Discriminative model** (Logistic Regression) directly models the posterior boundary $P(Y \mid X)$, focusing strictly on drawing an optimal separating hyperplane between classes without modeling how the inputs were generated.

### Q3: What is Target Leakage and how did you prevent it?
> **Answer**: Target leakage occurs when data used during training includes information that would not yet exist at inference time. In DataCo, `Days for shipping (real)` and `Delivery Status` are only recorded *after* delivery. Including them produces a 100% accurate model in training that is completely useless in production. We identified and dropped both columns during EDA.

### Q4: Why did you prioritize High-Risk Recall over Accuracy?
> **Answer**: In procurement, errors have asymmetric business costs. A False Negative (failing to predict a late delivery) shuts down factory assembly lines, causing tens of thousands of dollars in downtime. A False Positive (flagging an on-time delivery as risky) results in a brief confirmation phone call. Because False Negatives are vastly more destructive, maximizing High-Risk Recall is our north-star objective.

### Q5: How does decision threshold tuning benefit supply chain managers?
> **Answer**: Standard classifiers use a static threshold of $0.50$. In high-consequence operations, a manager cannot tolerate late arrivals. By lowering the threshold to $0.30$ in our Streamlit dashboard, we classify any order with $\ge 30\%$ risk as high risk. This increases Recall from $70.6\%$ to nearly $78\%$, capturing almost every at-risk delivery.

### Q6: How did you ensure numerical stability in your scratch implementations?
> **Answer**: 
> 1. In Naive Bayes, we performed all likelihood calculations in log-space to prevent floating-point underflow, added variance smoothing ($\epsilon = 10^{-9}$), and used the Log-Sum-Exp trick.
> 2. In Logistic Regression, we implemented a branching sigmoid using `np.where(z >= 0, ...)` with `np.clip(z, -500, 500)` to eliminate overflow errors in $e^{-z}$, and clipped predicted probabilities to $[\epsilon, 1-\epsilon]$ to prevent $\ln(0)$.

### Q7: How do your models provide explainability for non-technical managers?
> **Answer**: 
> * In Logistic Regression, we compute **Odds Ratios** ($\exp(w_j)$). An Odds Ratio of $5.93$ for First Class clearly communicates that choosing First Class multiplies the odds of a delay by nearly $6\times$.
> * In Naive Bayes, we calculate the **Log-Likelihood Ratio (LLR)** for each feature, showing whether individual parameters provided positive or negative evidence toward a delay. Both are rendered as live bar charts in our Streamlit dashboard.

### Q8: What was your team division of labor?
> **Answer**: 
> * **Person A** served as the Discriminative Lead: Formulated the mathematical derivation of vectorized binary cross-entropy, implemented the scratch Logistic Regression with $L_2$ regularization, built the Odds Ratio explainability module, and wrote automated unit tests.
> * **Person B** served as the Generative & Systems Lead: Formulated the Gaussian likelihood equations with variance smoothing, implemented scratch Naive Bayes with LLR attribution, conducted cloud EDA on Google Colab, and built the interactive Streamlit dashboard.

---
*Report generated for the Supplier Risk & Late Delivery Classification System. All code, tests, and documentation are verified and reproducible.*
