# End-to-End Project Walkthrough: Supplier Risk & Late Delivery Classification

> **Project Report & Technical Documentation**  
> **Topic**: Supplier Risk & Delivery Classification in Supply Chain Procurement  
> **Dataset**: DataCo Smart Supply Chain Dataset ($180,519$ records)  
> **Models**: Vectorized Logistic Regression & Gaussian Naive Bayes (Built from Scratch in NumPy)  

---

## 1. The "WHY": Business Problem & Procurement Context

### The Real-World Scenario
When a manufacturing or retail enterprise (such as Apple, Walmart, or Toyota) places purchase orders with external vendors, operational continuity depends on goods arriving strictly on schedule.
- **On-Time Delivery**: Production lines maintain cadence, customer orders are fulfilled without interruption, and working capital remains optimal.
- **Delayed Delivery**: Missing a single component can force an entire manufacturing plant to idle, leading to stockouts, contractual service-level agreement (SLA) penalties, and customer churn.

### The Cost Asymmetry (Why Accuracy is NOT Enough)
In supply chain management, prediction errors carry vastly asymmetric financial liabilities:
- **False Negative (FN)**: The AI model predicts *"Supplier is safe / On-time"*, but the shipment arrives late.  
  $\rightarrow$ **Catastrophic financial loss**: Plant shutdowns, emergency air-freight fees, broken client SLAs, and contractual liquidated damages.
- **False Positive (FP)**: The AI model predicts *"Supplier is risky / Late"*, but the shipment would have arrived on schedule.  
  $\rightarrow$ **Minimal administrative cost**: A procurement manager conducts an extra 15-minute audit or contacts a pre-cleared secondary supplier as a contingency.

> **Key Design Decision**: We optimized the entire system for **High-Risk Recall** ($\frac{\text{TP}}{\text{TP} + \text{FN}}$) and dynamic **decision threshold tuning**, rather than evaluating the model on unweighted classification accuracy.

---

## 2. The DATA: DataCo Smart Supply Chain Dataset

We utilized the **DataCo Smart Supply Chain Dataset** ($180,519$ records $\times$ $53$ features) to reflect real-world logistics transactions rather than small synthetic samples.

### 1. Target Label Distribution
The target outcome is `Late_delivery_risk`:
- **`1` (Late / High Risk)**: **$54.83\%$** ($98,977$ records)
- **`0` (On-Time / Safe)**: **$45.17\%$** ($81,542$ records)
- *Class Balance*: The dataset is naturally balanced (~$55/45$ split), eliminating the requirement for synthetic minority oversampling techniques (such as SMOTE).

### 2. ⚠️ The Target Leakage Audit (A Critical Engineering Safeguard)
During exploratory data analysis (EDA), cross-tabulation revealed that two columns recorded outcomes *after* the shipment was delivered:
- `Delivery Status == 'Late delivery'` exhibited a **$100\%$ correlation** with `Late_delivery_risk = 1`.
- `Days for shipping (real)` compared to scheduled shipping days gives away the final outcome.

> **Engineering Fix**: We permanently isolated and stripped these outcome columns in [`src/preprocessing.py`](file:///C:/Users/ANSH/.gemini/antigravity/scratch/supplier-risk-classification/src/preprocessing.py). Classifiers are trained **strictly on information known at the time of order placement** (scheduled delivery days, shipping mode, market, order pricing, discount rate, and profit ratio).

### 3. Preprocessing Pipeline
- **10 Numeric Features**: Scaled via `StandardScaler` ($\mu = 0, \sigma = 1$) to ensure stable gradient descent updates.
- **3 Categorical Features**: Transformed via `OneHotEncoder` (generating binary indicator columns for Shipping Mode, Customer Segment, and Market).
- **Train/Test Split**: Stratified $80\%$ Training ($144,415$ records) and $20\%$ Testing ($36,104$ records) to preserve target class proportions.

---

## 3. The ALGORITHMS: Built from Scratch in NumPy

Rather than relying on library black boxes, both classifiers were built **from scratch using vectorized NumPy** to contrast two foundational machine learning paradigms:

```
Machine Learning Paradigms Compared
├── Generative Model (Person B): Gaussian Naive Bayes   -> Models P(X | Y) & P(Y)
└── Discriminative Model (Person A): Logistic Regression -> Models P(Y | X) directly
```

---

### Algorithm 1: Gaussian Naive Bayes (Person B — @aakritishakya9)
Implemented in [`src/models/naive_bayes.py`](file:///C:/Users/ANSH/.gemini/antigravity/scratch/supplier-risk-classification/src/models/naive_bayes.py).

#### Mathematical Formulation
1. **Class Priors**:
   $$P(Y=1) = \frac{N_{\text{late}}}{N} \approx 0.5483, \quad P(Y=0) \approx 0.4517$$
2. **Gaussian Feature Likelihood**: Assumes continuous features follow a normal distribution per class:
   $$P(x_j \mid Y=c) = \frac{1}{\sqrt{2\pi (\sigma_{c,j}^2 + \epsilon)}} \exp\left( -\frac{(x_j - \mu_{c,j})^2}{2(\sigma_{c,j}^2 + \epsilon)} \right)$$
   *(Variance smoothing $\epsilon = 10^{-9} \cdot \max(\sigma^2)$ is added to guarantee numerical stability).*
3. **Log-Space Posterior**:
   $$\ln P(Y=c \mid X) \propto \ln P(Y=c) - \frac{1}{2} \sum_{j=1}^{d} \left[ \ln(2\pi \sigma_{c,j}^2) + \frac{(x_j - \mu_{c,j})^2}{\sigma_{c,j}^2} \right]$$
4. **Feature Explainability via Log-Likelihood Ratio (LLR)**:
   $$\text{LLR}_j(x_j) = \ln \frac{P(x_j \mid Y=1)}{P(x_j \mid Y=0)}$$
   - $\text{LLR}_j > 0$: The parameter pushes the order towards **High Risk (Late)**.
   - $\text{LLR}_j < 0$: The parameter pulls the order towards **Low Risk (On-Time)**.

---

### Algorithm 2: Vectorized Logistic Regression (Person A)
Implemented in [`src/models/logistic_regression.py`](file:///C:/Users/ANSH/.gemini/antigravity/scratch/supplier-risk-classification/src/models/logistic_regression.py).

#### Mathematical Formulation
1. **Hypothesis Function**: Maps a linear combination of features into a probability $[0, 1]$ via the numerically stable sigmoid function:
   $$\hat{y} = \sigma(Xw + b) = \frac{1}{1 + e^{-(Xw + b)}}$$
2. **Cost Function (Binary Cross-Entropy with L2 Regularization)**:
   $$J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \ln(\hat{y}^{(i)}) + (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)}) \right] + \frac{\lambda}{2m} \|w\|_2^2$$
3. **Vectorized Gradient Descent Updates**:
   $$\frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{y} - y) + \frac{\lambda}{m} w, \quad \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$
   $$w := w - \alpha \frac{\partial J}{\partial w}, \quad b := b - \alpha \frac{\partial J}{\partial b}$$
4. **Early Stopping**: Halts gradient descent when the change in cost drops below `tol = 1e-6`.
5. **Feature Explainability via Odds Ratios**:
   $$\text{Odds Ratio}_j = \exp(w_j)$$
   - $\text{Odds Ratio} > 1.0$: Feature increases the odds of delay (e.g., rush shipping tiers).
   - $\text{Odds Ratio} < 1.0$: Feature protects against delay (e.g., longer scheduled lead buffers).

---

## 4. The OUTCOME: Head-to-Head Benchmark Findings

Both scratch implementations were benchmarked on identical test partitions against Scikit-Learn:

| Evaluation Metric | Scratch Gaussian NB (Person B) | Scratch Logistic Reg (Person A) | Scikit-Learn Match | Winner & Takeaway |
| :--- | :---: | :---: | :---: | :--- |
| **Parity with Scikit-Learn** | **`100.00%`** | **`99.20%`** | Reference | **Both scratch models match industry baselines** |
| **Accuracy** | 64.00% | **`70.37%`** | Exact | Logistic Regression wins (+6.4%) |
| **High-Risk Recall ($Y=1$)** | 51.49% | **`70.61%`** | Exact | **Logistic Regression wins (+19.1%)** |
| **Precision ($Y=1$)** | 70.78% | **`71.58%`** | Exact | Logistic Regression is slightly higher |
| **F1-Score ($Y=1$)** | 0.5961 | **`0.7109`** | Exact | Logistic Regression wins (+0.115) |
| **ROC-AUC** | 0.7389 | **`0.7698`** | Exact | Logistic Regression has stronger separation |
| **Training Latency** | **`0.006 s`** | `0.414 s` | Faster | **Gaussian Naive Bayes is $\sim 60\times$ faster** |

---

### ⚖️ The Core Comparison: Which Model is Better for Procurement?

1. **Why Logistic Regression is the Superior Operational Model**:
   - Gaussian Naive Bayes relies on the conditional independence assumption: $P(X \mid Y) = \prod P(x_j \mid Y)$. However, in logistics, `Shipping Mode`, `Scheduled Days`, and `Discount Rate` are strongly collinear (rush orders have shorter schedules and specific shipping tiers).
   - Because Logistic Regression directly optimizes the discriminative boundary without assuming independence, it caught **$70.61\%$ of delayed shipments** compared to Naive Bayes' $51.49\%$. In procurement, catching those extra delayed shipments prevents costly factory shutdowns.
2. **Where Naive Bayes Has the Advantage**:
   - GNB trains in **0.006 seconds** using single-pass closed-form arithmetic ($\mu, \sigma^2$) instead of iterative gradient loops, making it ideal for ultra-fast online recalibration.
3. **Decision Threshold Tuning ($0.50 \to 0.30$)**:
   - By lowering the alert cutoff from $0.50$ to $0.30$, High-Risk Recall surged to **$78\%$**, trading off a few harmless extra vendor audits to catch almost 8 out of 10 delayed shipments.

---

## 5. The OUTPUT & DELIVERABLES: How We Built It

```text
supplier-risk-classification/
├── data/                         # Preprocessed feature arrays & raw data keeps
├── notebooks/
│   ├── 01_dataco_eda.ipynb       # Cloud EDA on Colab: leakage check & class priors
│   └── 02_model_experiments.ipynb # Full scratch experiments, loss curve & parity
├── reports/figures/              # Confusion matrices, ROC curves, PR curves
├── src/
│   ├── preprocessing.py          # Leakage-safe feature scaling & encoding pipeline
│   ├── evaluate.py               # Evaluation metrics, ROC-AUC, PR-AUC & diagnostics
│   ├── train_models.py           # Master benchmarking runner
│   └── models/
│       ├── naive_bayes.py         # Person B: NumPy Gaussian NB + LLR explainability
│       └── logistic_regression.py # Person A: NumPy Logistic Regression + Odds Ratios
├── tests/
│   ├── test_naive_bayes.py       # 5 automated unit tests for Naive Bayes
│   └── test_logistic.py          # 5 automated unit tests for Logistic Regression
├── app/
│   └── streamlit_app.py          # Interactive procurement risk assessment dashboard
├── requirements.txt
├── README.md                     # Full documentation with badges & benchmark table
└── PROJECT_PROGRESS_AND_HANDOVER_REPORT.md
```

### The 4 Major Deliverables:
1. **Google Colab Cloud Integration**: Ran end-to-end data processing in cloud sessions without local disk overhead.
2. **Automated Unit Testing Suite ([`tests/`](file:///C:/Users/ANSH/.gemini/antigravity/scratch/supplier-risk-classification/tests/))**: 10 unit tests executing in **0.27 seconds** verifying probability bounds, mathematical convergence, and zero-variance stability.
3. **Interactive Streamlit Web Dashboard ([`app/streamlit_app.py`](file:///C:/Users/ANSH/.gemini/antigravity/scratch/supplier-risk-classification/app/streamlit_app.py))**: A live qualification UI featuring real-time risk classification, decision threshold tuning, and feature attribution bar charts.
4. **GitHub Repository**: Fully documented and version-controlled at `https://github.com/aakritishakya9/supplier-risk-classification`.

---

## 6. The 60-Second "Elevator Pitch" (For Presentations & Oral Exams)

> *"In this project, we built an AI-powered procurement decision support system to classify supplier delivery risks using the 180,000-record DataCo supply chain dataset.*  
>  
> *Because a delayed delivery can halt entire factory assembly lines, we audited the data to eliminate post-shipment target leakage and prioritized High-Risk Recall over simple accuracy.*  
>  
> *We implemented two contrasting machine learning paradigms completely from scratch in NumPy: a generative Gaussian Naive Bayes model and a discriminative vectorized Logistic Regression model with gradient descent. Both scratch models achieved over 99% parity with scikit-learn, with our Logistic Regression model capturing 70.6% of late deliveries at a 0.770 ROC-AUC, outperforming Naive Bayes due to feature collinearity in shipping routes.*  
>  
> *Finally, we equipped both models with mathematical feature attribution (Log-Likelihood Ratios and Odds Ratios) and deployed them inside an interactive Streamlit dashboard allowing supply chain managers to calibrate their risk tolerance in real time."*
