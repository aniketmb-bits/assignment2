import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

# ==========================================
# 1. STEP 1: DEFINE THE EMBEDDED CUSTOM CLASS
# ==========================================
# This must remain exactly identical to your training class structure
class IncomeDataTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.country_mode_ = None
        self.occupation_imputer_ = None
        self.expected_columns_ = None
        self.categorical_features_ = None

    def _stateless_clean(self, df):
        df = df.copy()
        df["workclass"] = np.where(
            df["workclass"] == "?", "Unknown", df["workclass"]
        )
        df["fnlwgt_log"] = np.log1p(df["fnlwgt"])
        df["capital_gain_log"] = np.log1p(df["capital.gain"])
        df["capital_loss_log"] = np.log1p(df["capital.loss"])
        df["native.country"] = df["native.country"].str.strip().replace("?", np.nan)
        df["occupation"] = df["occupation"].replace("?", np.nan)
        if "income" in df.columns:
            df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})
        return df

    def fit(self, X, y=None):
        X_clean = self._stateless_clean(X)
        self.country_mode_ = X_clean["native.country"].mode()[0]
        X_clean["native.country"] = X_clean["native.country"].fillna(
            self.country_mode_
        )
        self.categorical_features_ = [
            col
            for col in X_clean.select_dtypes(include="object").columns
            if col != "occupation"
        ]
        self.occupation_imputer_ = SimpleImputer(strategy="most_frequent")
        self.occupation_imputer_.fit(X_clean[["occupation"]])
        X_clean["occupation"] = self.occupation_imputer_.transform(
            X_clean[["occupation"]]
        ).ravel()
        dummy_cols = self.categorical_features_ + ["occupation"]
        X_encoded = pd.get_dummies(X_clean, columns=dummy_cols, drop_first=True)
        cols_to_drop = ["fnlwgt", "capital.gain", "capital.loss", "income"]
        self.expected_columns_ = [
            c for c in X_encoded.columns if c not in cols_to_drop
        ]
        return self

    def transform(self, X):
        X_clean = self._stateless_clean(X)
        X_clean["native.country"] = X_clean["native.country"].fillna(
            self.country_mode_
        )
        X_clean["occupation"] = self.occupation_imputer_.transform(
            X_clean[["occupation"]]
        ).ravel()
        dummy_cols = self.categorical_features_ + ["occupation"]
        X_encoded = pd.get_dummies(X_clean, columns=dummy_cols, drop_first=True)
        X_final = X_encoded.reindex(
            columns=self.expected_columns_, fill_value=0
        )
        return X_final


# Force Python to map any unpickling lookups to this file's main module
import sys

sys.modules["preprocessors"] = sys.modules[__name__]

# ==========================================
# 2. STEP 2: STREAMLIT APP LOGIC
# ==========================================
st.title("🚀 Model Inference Dashboard")

# Look for available .pkl files dynamically
all_files = os.listdir(".")
model_options = [
    f for f in all_files if f.endswith(".pkl") and "transformer" not in f.lower()
]
if not model_options:
    model_options = ["trained_model.pkl"]

transformer_options = [
    f for f in all_files if f.endswith(".pkl") and "transformer" in f.lower()
]
if not transformer_options:
    transformer_options = ["data_transformer.pkl"]

# Add dropdown configuration sidebars
st.sidebar.subheader("⚙️ Configuration")
selected_model_file = st.sidebar.selectbox(
    "Select the trained model file to use:", options=model_options, index=0
)
selected_transformer_file = st.sidebar.selectbox(
    "Select data transformer file:", options=transformer_options, index=0
)


# Cached loader
@st.cache_resource
def load_artifacts(transformer_path, model_path):
    transformer = joblib.load(transformer_path)
    model = joblib.load(model_path)
    return transformer, model


try:
    transformer, model = load_artifacts(
        selected_transformer_file, selected_model_file
    )
    st.success(
        f"Successfully loaded: **{selected_model_file}** using **{selected_transformer_file}**"
    )
except Exception as e:
    st.error(
        f"Error loading files. Ensure your pkl files are uploaded. Error: {e}"
    )
    st.stop()

# Display Feature names
st.subheader("📋 Expected Model Features")
trained_features = transformer.expected_columns_
with st.expander(f"View all {len(trained_features)} feature columns"):
    st.write(trained_features)

# Upload Target Test Data
st.subheader("📤 Upload Test Data")
uploaded_file = st.file_uploader(
    "Choose a CSV file containing your test data", type=["csv"]
)

if uploaded_file is not None:
    X_test = pd.read_csv(uploaded_file)
    st.write("### Raw Uploaded Data Preview", X_test.head())

    if st.button("Run Predictions"):
        with st.spinner(f"Processing data with {selected_model_file}..."):
            try:
                X_test_processed = transformer.transform(X_test)
                predictions = model.predict(X_test_processed)
                X_test["Predicted_Income"] = predictions
                X_test["Predicted_Income_Label"] = X_test[
                    "Predicted_Income"
                ].map({0: "<=50K", 1: ">50K"})

                st.success("Processing complete!")
                st.write("### Predictions Output Preview", X_test.head())

                csv_data = X_test.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv_data,
                    file_name=f"predictions_output.csv",
                    mime="text/csv",
                )
            except Exception as transform_error:
                st.error(f"Prediction failed: {transform_error}")
