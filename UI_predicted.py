import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib

# Application Settings
st.title("Traffic Accident Severity Prediction")
st.write("Enter the features to obtain the severity prediction.")

# Load pre-trained model and encoders
try:
    model = joblib.load("random_forest_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    st.success("Modelo cargado correctamente.")
except Exception as e:
    st.error("Error al cargar el modelo o los codificadores. Verifica los archivos.")
    st.stop()

categorical_cols = ['Wind_Direction', 'Weather_Condition', 'Sunrise_Sunset', 
                    'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight']
numerical_cols = ['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 
                  'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']

user_input = {}
# Split into two columns
col1, col2 = st.columns(2)

# Collect categorical inputs
with col1:
    st.subheader("Categorical Variables")
    for col in categorical_cols:
        categories = label_encoders[col].classes_ 
        user_input[col] = st.selectbox(f"{col}:", categories)

# Collect numerical inputs
with col2:
    st.subheader("Numeric Variables")
    for col in numerical_cols:
        min_val, max_val = 0, 100  # Adjust range if necessary
        user_input[col] = st.slider(f"{col}:", min_val, max_val, step=1)

# Prepare the input data
input_data = pd.DataFrame([user_input])
for col in categorical_cols:
    input_data[col] = label_encoders[col].transform(input_data[col])

if st.button("Predict Severity"):
    try:
        prediction_proba = model.predict_proba(input_data)
        predicted_class = np.argmax(prediction_proba) + 1 

        # Display probabilities rounded to three decimals
        st.write("**Probabilities for each class:**")
        probabilities = {f"Clase {i+1}": round(prob, 3) for i, prob in enumerate(prediction_proba[0])}
        st.write(probabilities)

        # Alert box with updated styles
        alert_color = {
            1: "#4CAF50",  # Green
            2: "#FFEB3B",  # Yellow
            3: "#FF9800",  # Orange
            4: "#F44336"   # Red
        }
        st.markdown(
            f"<div style='padding: 10px; border-radius: 5px; background-color: {alert_color[predicted_class]}; color: white; text-align: center;'>"
            f"<strong>Severity Prediction:</strong> Clase {predicted_class}</div>",
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error("An error occurred while making the prediction. Please check your entries.")