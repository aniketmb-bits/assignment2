import joblib
import pandas as pd
import streamlit as st
# Import your custom class so joblib can unpickle it successfully
from preprocessors import IncomeDataTransformer

# Set up page title
st.title("🚀 Model Inference Dashboard")

model_options = ['logistic_regression_l1',
                 'logistic_regression_l2',
                 'decision_tree',
                 'knn',
                 'naive_bayes_classifier',
                 'random_forest']

st.sidebar.subheader("⚙️ Configuration")
selected_model_file = st.sidebar.selectbox(
    "Select the trained model file to use:",
    options=model_options,
    index=0
)
# 1. Load your saved pipeline components
@st.cache_resource
def load_artifacts():
    model_path = "model/" + selected_model_file + "_model.pkl"
    transformer = joblib.load("data_transformer.pkl")
    model = joblib.load("trained_mo.pkl")
    return transformer, model


try:
    transformer, model = load_artifacts()
    st.success("Preprocessors and Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# 2. Extract and display expected training feature names
st.subheader("📋 Expected Model Features")
# The transformer object holds the training column names we saved earlier
trained_features = transformer.expected_columns_

with st.expander(f"View all {len(trained_features)} feature columns"):
    st.write(trained_features)

# 3. Add option to provide test.csv
st.subheader("📤 Upload Test Data")
uploaded_file = st.file_uploader(
    "Choose a CSV file containing your test data", type=["csv"]
)

if uploaded_file is not None:
    # Read the uploaded CSV file into a pandas DataFrame
    X_test = pd.read_csv(uploaded_file)
    st.write("### Raw Uploaded Data Preview", X_test.head())

    # Add a processing button
    if st.button("Run Predictions"):
        with st.spinner("Processing data and generating predictions..."):
            try:
                # 4. Transform data safely with zero leakage
                X_test_processed = transformer.transform(X_test)

                # 5. Make predictions using the loaded model
                predictions = model.predict(X_test_processed)

                # 6. Display results
                X_test["Predicted_Income"] = predictions
                # Map binary outputs back to readable text if desired
                X_test["Predicted_Income_Label"] = X_test[
                    "Predicted_Income"
                ].map({0: "<=50K", 1: ">50K"})

                st.success("Processing complete!")
                st.write("### Predictions Output Preview", X_test.head())

                # 7. Optional: Allow user to download the final CSV
                csv_data = X_test.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv_data,
                    file_name="predictions_output.csv",
                    mime="text/csv",
                )

            except Exception as transform_error:
                st.error(
                    f"Preprocessing or Prediction failed. Verify your columns match requirements. Error: {transform_error}"
                )
