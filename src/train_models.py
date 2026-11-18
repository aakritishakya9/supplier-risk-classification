"""
End-to-End Training and Benchmarking Runner for Both Models:
- Model A: Scratch Vectorized Logistic Regression (Person A)
- Model B: Scratch Gaussian Naive Bayes (Person B)
Benchmarks both models against scikit-learn and outputs comparative metrics.
"""

import numpy as np
import pandas as pd
import time
import os
import sys

# Ensure UTF-8 console output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB as SklearnGaussianNB
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.naive_bayes import ScratchGaussianNB
from src.models.logistic_regression import ScratchLogisticRegression
from src.evaluate import compute_classification_metrics, evaluate_thresholds, plot_model_diagnostics
from src.preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def generate_benchmark_dataset(n_samples=10000, random_state=42):
    """
    Generates realistic procurement order data matching DataCo distributions
    when raw 95MB dataset is not present locally.
    """
    np.random.seed(random_state)
    
    # 10 Numerical features
    sched_days = np.random.choice([1, 2, 3, 4, 5, 6], size=n_samples, p=[0.1, 0.2, 0.35, 0.2, 0.1, 0.05])
    product_price = np.random.exponential(scale=120, size=n_samples) + 15
    quantity = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.4, 0.3, 0.15, 0.1, 0.05])
    discount_rate = np.random.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.25], size=n_samples)
    sales = product_price * quantity * (1.0 - discount_rate)
    profit_ratio = np.random.normal(loc=0.12, scale=0.15, size=n_samples)
    benefit = sales * profit_ratio

    X_num = np.column_stack([
        sched_days,
        benefit,
        sales,
        sales * discount_rate,
        discount_rate,
        product_price,
        profit_ratio,
        quantity,
        sales,
        product_price
    ])

    shipping_modes = np.random.choice(['Standard Class', 'First Class', 'Second Class', 'Same Day'], size=n_samples, p=[0.60, 0.18, 0.17, 0.05])
    customer_segments = np.random.choice(['Consumer', 'Corporate', 'Home Office'], size=n_samples, p=[0.52, 0.30, 0.18])
    markets = np.random.choice(['Europe', 'LATAM', 'Pacific Asia', 'USCA', 'Africa'], size=n_samples)
    departments = np.random.choice(['Apparel', 'Fan Shop', 'Footwear', 'Golf', 'Outdoors'], size=n_samples)
    regions = np.random.choice(['Western Europe', 'Central America', 'South America', 'Northern Europe', 'Southern Asia'], size=n_samples)

    df = pd.DataFrame(X_num, columns=NUMERIC_FEATURES)
    df['Shipping Mode'] = shipping_modes
    df['Customer Segment'] = customer_segments
    df['Market'] = markets
    df['Department Name'] = departments
    df['Order Region'] = regions


    # Target: First Class, Same Day, and short scheduled days have higher delay probability
    logit = (
        -0.8 * (sched_days - 3.0) +
        1.5 * (shipping_modes == 'First Class') +
        1.2 * (shipping_modes == 'Second Class') +
        1.8 * (shipping_modes == 'Same Day') +
        0.3 * (discount_rate > 0.15) - 0.5
    )
    prob_late = 1.0 / (1.0 + np.exp(-logit))
    y = (np.random.rand(n_samples) < prob_late).astype(int)

    return df, y


def run_benchmark():
    print("🚀 Initializing Procurement Risk Classification Benchmark...")
    raw_data_path = "data/raw/DataCoSupplyChainDataset.csv"

    if os.path.exists(raw_data_path):
        print(f"📦 Loading DataCo dataset from {raw_data_path}...")
        df_raw = pd.read_csv(raw_data_path, encoding="latin-1")
        from src.preprocessing import get_train_test_data
        X_train_trans, X_test_trans, y_train, y_test, preprocessor = get_train_test_data(df_raw)
    else:
        print("💡 Raw DataCo CSV not stored locally; generating representative DataCo distribution for local evaluation...")
        df_bench, y = generate_benchmark_dataset(n_samples=15000)
        X_train, X_test, y_train, y_test = train_test_split(df_bench, y, test_size=0.2, random_state=42, stratify=y)
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUMERIC_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)
            ]
        )
        X_train_trans = preprocessor.fit_transform(X_train)
        X_test_trans = preprocessor.transform(X_test)

    print(f"✅ Training set: {X_train_trans.shape[0]:,} samples, {X_train_trans.shape[1]} features")
    print(f"✅ Test set:     {X_test_trans.shape[0]:,} samples, {X_test_trans.shape[1]} features\n")

    # 1. Train Person B: Scratch Gaussian Naive Bayes
    t0 = time.time()
    scratch_gnb = ScratchGaussianNB().fit(X_train_trans, y_train)
    gnb_train_time = time.time() - t0
    y_pred_gnb = scratch_gnb.predict(X_test_trans)
    y_prob_gnb = scratch_gnb.predict_proba(X_test_trans)[:, 1]

    # 2. Train Person A: Scratch Logistic Regression
    t0 = time.time()
    scratch_lr = ScratchLogisticRegression(learning_rate=0.1, n_iters=400, l2_penalty=0.1).fit(X_train_trans, y_train)
    lr_train_time = time.time() - t0
    y_pred_lr = scratch_lr.predict(X_test_trans)
    y_prob_lr = scratch_lr.predict_proba(X_test_trans)[:, 1]

    # 3. Train Scikit-Learn Benchmarks
    sk_gnb = SklearnGaussianNB().fit(X_train_trans, y_train)
    sk_lr = SklearnLogisticRegression(C=10.0, max_iter=400).fit(X_train_trans, y_train)

    y_pred_sk_gnb = sk_gnb.predict(X_test_trans)
    y_pred_sk_lr = sk_lr.predict(X_test_trans)

    # Parity checks
    gnb_parity = np.mean(y_pred_gnb == y_pred_sk_gnb) * 100
    lr_parity = np.mean(y_pred_lr == y_pred_sk_lr) * 100

    print("=" * 70)
    print("🎯 PARITY VERIFICATION AGAINST SCIKIT-LEARN")
    print(f"  • Gaussian Naive Bayes Parity:   {gnb_parity:.2f}% agreement")
    print(f"  • Logistic Regression Parity:     {lr_parity:.2f}% agreement")
    print("=" * 70)

    # Compute Metrics
    metrics_gnb = compute_classification_metrics(y_test, y_pred_gnb, y_prob_gnb)
    metrics_lr = compute_classification_metrics(y_test, y_pred_lr, y_prob_lr)

    metrics_df = pd.DataFrame([
        {"Model": "Scratch Gaussian NB (Person B)", "Family": "Generative", "Training Time (s)": gnb_train_time, **metrics_gnb},
        {"Model": "Scratch Logistic Reg (Person A)", "Family": "Discriminative", "Training Time (s)": lr_train_time, **metrics_lr}
    ])

    print("\n📊 HEAD-TO-HEAD PERFORMANCE METRICS:")
    print(metrics_df.to_string(index=False))

    # Save Diagnostic Plots
    plot_model_diagnostics(y_test, y_prob_gnb, model_name="Gaussian_Naive_Bayes")
    plot_model_diagnostics(y_test, y_prob_lr, model_name="Logistic_Regression")

    print("\n✅ Diagnostic plots generated in reports/figures/")
    return metrics_df


if __name__ == "__main__":
    run_benchmark()
