import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

# ==========================================
# 1. SETUP ENVIRONMENT AND RESOLVE PATHS
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

# Force Python to map any unpickling lookups to this file's main module immediately
sys.modules["preprocessors"] = sys.modules[__name__]


# ==========================================
# 2. STEP 1: DEFINE THE EMBEDDED CUSTOM CLASS
# ==========================================
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
            if col not in ["occupation", "income"]
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


# ==========================================
# 3. STEP 2: STREAMLIT APP LOGIC & STATE
# ==========================================
st.title("🚀 Model Inference Dashboard")

# Initialize Session State values to track prediction states between dropdown changes
if "processed_results" not in st.session_state:
    st.session_state.processed_results = None
if "last_model" not in st.session_state:
    st.session_state.last_model = None
if "last_transformer" not in st.session_state:
    st.session_state.last_transformer = None

# Safe folder scanning: scans model directory if it exists, otherwise root workspace
if MODEL_DIR.exists():
    all_files = os.listdir(MODEL_DIR)
else:
    all_files = os.listdir(BASE_DIR)

model_options = [
    f for f in all_files if f.endswith(".pkl") and "transformer" not in f.lower()
]
if not model_options:
    model_options = ["logistic_regression_l1.pkl"]

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

# If user flips dropdown settings, invalidate the old results to trigger an automatic re-run
if (
    st.session_state.last_model != selected_model_file
    or st.session_state.last_transformer != selected_transformer_file
):
    st.session_state.processed_results = None
    st.session_state.last_model = selected_model_file
    st.session_state.last_transformer = selected_transformer_file


# Cached loader that builds explicit paths based on directory location
@st.cache_resource
def load_artifacts(transformer_filename, model_filename):
    t_path = (
        (MODEL_DIR / transformer_filename)
        if MODEL_DIR.exists()
        else (BASE_DIR / transformer_filename)
    )
    m_path = (
        (MODEL_DIR / model_filename)
        if MODEL_DIR.exists()
        else (BASE_DIR / model_filename)
    )

    transformer = joblib.load(t_path)
    model = joblib.load(m_path)
    return transformer, model


try:
    transformer, model = load_artifacts(
        selected_transformer_file, selected_model_file
    )
    st.success(
        f"Loaded: **{selected_model_file}** using **{selected_transformer_file}**"
    )
except Exception as e:
    st.error(f"Error loading files. Check paths. System Error: {e}")
    st.stop()

# Display Feature names
st.subheader("📋 Expected Model Features")
trained_features = transformer.expected_columns_
with st.expander(f"View all {len(trained_features)} feature columns"):
    st.write(trained_features)

# ==========================================
# 4. STEP 3: PREDICTION ENGINE
# ==========================================
st.subheader("📤 Upload Test Data")
uploaded_file = st.file_uploader(
    "Choose a CSV file containing your test data", type=["csv"]
)

if uploaded_file is not None:
    X_test_raw = pd.read_csv(uploaded_file)
    st.write("### Raw Uploaded Data Preview", X_test_raw.head())

    run_button = st.button("Run Predictions")

    # Auto-trigger execution if button is clicked OR if state requires a recalculation run
    if run_button or (st.session_state.processed_results is None):
        with st.spinner(f"Processing data with {selected_model_file}..."):
            try:
                # ------------------------------------------------------------------
                # PROACTIVE FIX: Drop 'income' BEFORE passing to the frozen pickle
                # ------------------------------------------------------------------
                X_input_clean = X_test_raw.copy()
            
                # Transform the safely scrubbed features matrix
                X_test_processed = transformer.transform(X_input_clean)
                predictions = model.predict(X_test_processed.drop(columns=['income']))

                # Attach outputs back to a visual copy dataframe for user interface
                display_df = X_test_raw.copy()
                display_df["Predicted_Income"] = predictions
                display_df["Predicted_Income_Label"] = display_df[
                    "Predicted_Income"
                ].map({0: "<=50K", 1: ">50K"})

                # Store matrix inside Session State memory
                st.session_state.processed_results = display_df
            except Exception as transform_error:
                st.error(f"Prediction failed: {transform_error}")
                st.session_state.processed_results = None

    # Render results dynamically out of Session State memory
    if st.session_state.processed_results is not None:
        st.success(f"Processing complete using **{selected_model_file}**!")
        st.write(
            "### Predictions Output Preview",
            st.session_state.processed_results.head(),
        )

        csv_data = st.session_state.processed_results.to_csv(index=False).encode(
            "utf-8"
        )
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv_data,
            file_name=f"predictions_{selected_model_file.split('.')[0]}.csv",
            mime="text/csv",
        )
