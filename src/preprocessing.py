"""
Data preprocessing & leakage-safe feature selection pipeline for DataCo dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# Target variable in DataCo
TARGET_COL = "Late_delivery_risk"

# Columns to DROP to avoid data leakage or irrelevant IDs
# 'Days for shipping (real)' and 'Delivery Status' are outcome leakages!
LEAKAGE_AND_ID_COLS = [
    "Days for shipping (real)",
    "Delivery Status",
    "Customer Email",
    "Customer Fname",
    "Customer Lname",
    "Customer Password",
    "Customer Street",
    "Product Description",
    "Product Image",
    "Order Id",
    "Order Item Id",
    "Customer Id",
    "Order Customer Id",
    "Order Zipcode",
    "Product Status"
]

# Baseline numerical features available at order time
NUMERIC_FEATURES = [
    "Days for shipment (scheduled)",
    "Benefit per order",
    "Sales per customer",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Product Price",
    "Order Item Profit Ratio",
    "Order Item Quantity",
    "Sales",
    "Product Price"
]

# Categorical features available at order time
CATEGORICAL_FEATURES = [
    "Shipping Mode",
    "Customer Segment",
    "Department Name",
    "Market",
    "Order Region"
]


def clean_dataco_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw DataCo dataset by removing leakage columns and handling missing values.
    """
    # Drop known leakages and IDs if present
    cols_to_drop = [c for c in LEAKAGE_AND_ID_COLS if c in df.columns]
    cleaned = df.drop(columns=cols_to_drop).copy()

    # Fill or drop missing values
    cleaned = cleaned.dropna(subset=[TARGET_COL])
    return cleaned


def build_preprocessor():
    """
    Creates a scikit-learn ColumnTransformer for scaling numeric features
    and one-hot encoding categorical variables.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )
    return preprocessor


def get_train_test_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Preprocesses dataset and returns stratified train and test sets.
    """
    cleaned_df = clean_dataco_dataframe(df)

    X = cleaned_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = cleaned_df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    return X_train_trans, X_test_trans, y_train, y_test, preprocessor
