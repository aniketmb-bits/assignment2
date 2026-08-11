import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, 
    f1_score, matthews_corrcoef, RocCurveDisplay,
    confusion_matrix, ConfusionMatrixDisplay
) 

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
        self.country_mode_ = X_clean["native.country"].mode()
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
# 3. EVALUATION AND PLOTTING FUNCTIONS
# ==========================================
def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    return accuracy, recall, precision, f1, mcc


def get_metrics_dataframe(model_dict, accuracy, recall, precision, f1, mcc):
    rows = []
    for model_name in model_dict:
        rows.append({
            'Model Name': model_name,
            'Accuracy': accuracy,
            'Recall': recall,
            'Precision': precision,
            'F1 Score': f1,
            'MCC': mcc
        })
    return pd.DataFrame(rows)


def plot_roc_curve(model_dict, X_test, y_test):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for model_name, model in model_dict.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=model_name)

    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess (AUC = 0.5)')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Adult Income Model (All Models)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    st.pyplot(fig)


# ==========================================
# 4. STEP 2: STREAMLIT APP LOGIC & STATE
# ==========================================
st.title("🚀 Model Inference & Evaluation Dashboard")

# Initialize Session State values to track prediction states between dropdown changes
if "processed_results" not in st.session_state:
    st.session_state.processed_results = None
if "last_model" not in st.session_state:
    st.session_state.last_model = None
if "last_transformer" not in st.session_state:
    st.session_state.last_transformer = None

# Scan folder for files
if MODEL_DIR.exists():
    all_files = os.listdir(MODEL_DIR)
else:
    all_files = os.listdir(BASE_DIR)

model_files = [f for f in all_files if f.endswith(".pkl") and "transformer" not in f.lower()]
if not model_files:
    model_files = ["logistic_regression_l1.pkl"]

transformer_options = [f for f in all_files if f.endswith(".pkl") and "transformer" in f.lower()]
if not transformer_options:
    transformer_options = ["data_transformer.pkl"]

# Sidebar settings
st.sidebar.subheader("⚙️ Configuration")
selected_transformer_file = st.sidebar.selectbox("Select data transformer file:", options=transformer_options, index=0)

# Load Selected Transformer
@st.cache_resource
def load_transformer(transformer_path):
    t_path = (MODEL_DIR / transformer_path) if MODEL_DIR.exists() else (BASE_DIR / transformer_path)
    return joblib.load(t_path)

try:
    transformer = load_transformer(selected_transformer_file)
except Exception as e:
    st.error(f"Error loading transformer: {e}")
    st.stop()


# Load ALL available models into a dictionary for evaluation comparisons
@st.cache_resource
def load_all_models(files):
    model_dict = {}
    for f in files:
        m_path = (MODEL_DIR / f) if MODEL_DIR.exists() else (BASE_DIR / f)
        try:
            model_dict[f.replace('.pkl', '')] = joblib.load(m_path)
        except:
            pass
    return model_dict

model_dict = load_all_models(model_files)

if not model_dict:
    st.error("No valid models found to load.")
    st.stop()

# Let users pick one active model for preview table predictions
selected_model_name = st.sidebar.selectbox("Active Preview Model:", options=list(model_dict.keys()))
active_model = model_dict[selected_model_name]

# If user flips dropdown settings, invalidate the old results to trigger an automatic re-run
if (
    st.session_state.last_model != selected_model_name
    or st.session_state.last_transformer != selected_transformer_file
):
    st.session_state.processed_results = None
    st.session_state.last_model = selected_model_name
    st.session_state.last_transformer = selected_transformer_file

# Display Feature names
st.subheader("📋 Expected Model Features")
with st.expander(f"View all {len(transformer.expected_columns_)} feature columns"):
    st.write(transformer.expected_columns_)


# ==========================================
# 5. STEP 3: PREDICTION & EVALUATION ENGINE
# ==========================================
st.subheader("📤 Upload Test Data")
uploaded_file = st.file_uploader("Choose a CSV file containing your test data", type=["csv"])

if uploaded_file is not None:
    X_test_raw = pd.read_csv(uploaded_file)
    st.write("### Raw Uploaded Data Preview", X_test_raw.head())

    # Check if 'income' exists in the uploaded dataset to enable performance benchmarking
    has_ground_truth = "income" in X_test_raw.columns

    if has_ground_truth:
        st.info("📊 **Ground truth label ('income') detected!** The app will display model performance comparisons and ROC curves.")
    else:
        st.warning("⚠️ **No 'income' column found.** Running in pure production inference mode (no metrics or charts).")

    run_button = st.button("Run Diagnostics & Predictions")

    if run_button or (st.session_state.processed_results is None):
        with st.spinner("Processing computations..."):
            try:
                # 1. Isolate the target label text safely if it exists
                y_test = None
                X_input_clean = X_test_raw.copy()

                # 2. Preprocess features using the transformer
                X_test_processed = transformer.transform(X_input_clean)
                y_test = X_test_processed['income']
                # 3. Predict using active selection model for table preview
                active_preds = active_model.predict(X_test_processed.drop(columns=['income']))
                active_pred_prob = active_model.predict_proba(X_test_processed.drop(columns=['income']))
                accuracy, recall, precision, f1, mcc = evaluate_model(y_test, active_preds)
                display_df = X_test_raw.copy()
                display_df["Predicted_Income"] = active_preds
                display_df["Predicted_Income_Label"] = display_df["Predicted_Income"].map({0: "<=50K", 1: ">50K"})
                metrics_df = get_metrics_dataframe({selected_model_name: active_model}, accuracy, recall, precision, f1, mcc)
                st.dataframe(metrics_df)
                plot_roc_curve({selected_model_name: active_model}, X_test_processed.drop(columns=['income']), y_test)

                fig, ax = plt.subplots()
                cm = confusion_matrix(y_test, active_preds)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['<=50k', '>50K'])
                disp.plot(cmap=plt.cm.Blues)
                st.subheader("📊 Confusion Matrix Chart")
                st.pyplot(fig)

                # Store matrix inside Session State memory
                st.session_state.processed_results = (display_df, X_test_processed, y_test)
                
            except Exception as e:
                st.error(f"Inference Engine failure: {e}")
                st.session_state.processed_results = None

