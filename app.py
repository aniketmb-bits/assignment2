import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("🍷 Wine Quality Predictor App")
st.write("Adjust the physicochemical properties below to classify the wine quality tier.")

wine_display_type = st.selectbox("Select Wine Variety", ["White Wine", "Red Wine"])

# Map the string to a number (Matches your training step, e.g., White=0, Red=1)
if wine_display_type == "Red Wine":
    wine_type_encoded = 1
else:
    wine_type_encoded = 0

col1, col2 = st.columns(2)

with col1:
    alcohol = st.slider("Alcohol Content (%)", min_value=8.0, max_value=15.0, value=10.5, step=0.1)
    volatile_acidity = st.slider("Volatile Acidity (g/dm³)", min_value=0.0, max_value=1.5, value=0.3, step=0.01)
    citric_acid = st.slider("Citric Acid (g/dm³)", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    residual_sugar = st.slider("Residual Sugar (g/dm³)", min_value=0.0, max_value=20.0, value=2.5, step=0.1)
    chlorides = st.slider("Chlorides (g/dm³)", min_value=0.0, max_value=0.2, value=0.05, step=0.001)

with col2:
    free_so2 = st.slider("Free Sulfur Dioxide (mg/dm³)", min_value=1.0, max_value=100.0, value=30.0, step=1.0)
    total_so2 = st.slider("Total Sulfur Dioxide (mg/dm³)", min_value=5.0, max_value=300.0, value=120.0, step=1.0)
    density = st.slider("Density (g/cm³)", min_value=0.980, max_value=1.005, value=0.994, step=0.001)
    pH = st.slider("pH Level", min_value=2.5, max_value=4.0, value=3.2, step=0.01)
    sulphates = st.slider("Sulphates (g/dm³)", min_value=0.2, max_value=2.0, value=0.6, step=0.01)

input_data = pd.DataFrame([{
    'fixed_acidity': 7.0, # Using baseline constants for unlisted features
    'volatile_acidity': volatile_acidity,
    'citric_acid': citric_acid,
    'residual_sugar': residual_sugar,
    'chlorides': chlorides,
    'free_sulfur_dioxide': free_so2,
    'total_sulfur_dioxide': total_so2,
    'density': density,
    'pH': pH,
    'sulphates': sulphates,
    'alcohol': alcohol
}])

@st.cache_resource
def load_scalar_model():
    loaded_scaler = joblib.load("model/wine_scaler.pkl")
    return loaded_scaler
    
def load_rf_saved_model():
    # Make sure these filenames match your exported files exactly
    loaded_model = joblib.load("model/wine_rf_model.pkl")
    return loaded_model

def load_lg_saved_model():
    # Make sure these filenames match your exported files exactly
    loaded_model = joblib.load("model/wine_lg_model.pkl")
    return loaded_model

def load_knn_saved_model():
    # Make sure these filenames match your exported files exactly
    loaded_model = joblib.load("model/wine_knn_model.pkl")
    return loaded_model

def load_nb_saved_model():
    # Make sure these filenames match your exported files exactly
    loaded_model = joblib.load("model/wine_nb_model.pkl")
    return loaded_model

def load_dt_saved_model():
    # Make sure these filenames match your exported files exactly
    loaded_model = joblib.load("model/wine_dt_model.pkl")
    return loaded_model

# Call the function to fetch your pipeline
model_rf = load_rf_saved_model()
model_lg = load_lg_saved_model()
model_knn = load_knn_saved_model()
model_dt = load_dt_saved_model()
model_nb = load_nb_saved_model()
scaler =   load_scaler_model()

wine_model_type = st.selectbox("Select model", ["Logistic regression", "Decision tree", "K Nearest neighbours", "naive bayes classifier", "Random Forest"])

if st.button("🔮 Predict Wine Quality Tier"):
    if (wine_model_type == "Logistic regression"):
        simulated_prediction = model_lg.predict(scaler.transform(input_data))
    elif (wine_model_type == "Decision tree"):
        simulated_prediction = model_dt.predict(scaler.transform(input_data))
    elif (wine_model_type == "K Nearest neighbours"):
        simulated_prediction = model_knn.predict(scaler.transform(input_data))
    elif (wine_model_type == "naive bayes classifier"):
        simulated_prediction == model_nb.predict(scaler.transform(input_data))
    elif (wine_model_type == "Random Forest"):
        simulated_prediction = model_rf.predict(scaler.transform(input_data))
    
    # 5. Display the output cleanly with status markers
    st.markdown("---")
    if simulated_prediction == 1:
        st.error("📉 Prediction: **Low Quality** (Tier 1)")
    elif simulated_prediction == 2:
        st.warning("📊 Prediction: **Medium Quality** (Tier 2)")
    else:
        st.success("✨ Prediction: **High Quality / Premium** (Tier 3)")
