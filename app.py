import streamlit as st
import pandas as pd
import numpy as np
import joblib
# (Optional) Import your trained model file here, e.g., using joblib
# import joblib
# model = joblib.load("random_forest_wine_model.pkl")

# 1. Set App Title and Subheading
st.title("🍷 Wine Quality Predictor App")
st.write("Adjust the physicochemical properties below to classify the wine quality tier.")

wine_display_type = st.selectbox("Select Wine Variety", ["White Wine", "Red Wine"])

# Map the string to a number (Matches your training step, e.g., White=0, Red=1)
if wine_display_type == "Red Wine":
    wine_type_encoded = 1
else:
    wine_type_encoded = 0

# 2. Organize User Input Controls into Columns
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

# 3. Create Dataframe from Inputs
input_data = pd.DataFrame([{
    'type': wine_type_encoded,  # Using the encoded value
    'fixed acidity': 7.0, # Using baseline constants for unlisted features
    'volatile acidity': volatile_acidity,
    'citric acid': citric_acid,
    'residual sugar': residual_sugar,
    'chlorides': chlorides,
    'free sulfur dioxide': free_so2,
    'total sulfur dioxide': total_so2,
    'density': density,
    'pH': pH,
    'sulphates': sulphates,
    'alcohol': alcohol
}])

@st.cache_resource
def load_saved_model():
    # Make sure these filenames match your exported files exactly
    loaded_model = joblib.load("wine_rf_model.pkl")
    loaded_scaler = joblib.load("wine_scaler.pkl")
    return loaded_model, loaded_scaler

# Call the function to fetch your pipeline
model, scaler = load_saved_model()

# 4. Trigger Prediction Button
if st.button("🔮 Predict Wine Quality Tier"):
    # Simulated prediction output logic 
    # (Replace this with model.predict(input_data) once your pkl file is linked)
    simulated_prediction = model.predict(scaler.transform(input_data))
    
    # 5. Display the output cleanly with status markers
    st.markdown("---")
    if simulated_prediction == 1:
        st.error("📉 Prediction: **Low Quality** (Tier 1)")
    elif simulated_prediction == 2:
        st.warning("📊 Prediction: **Medium Quality** (Tier 2)")
    else:
        st.success("✨ Prediction: **High Quality / Premium** (Tier 3)")
