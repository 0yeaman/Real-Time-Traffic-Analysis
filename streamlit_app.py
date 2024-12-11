import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    h1 {
        color: #4CAF50;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    h2, h3, .stSelectbox, .stNumberInput {
        color: #333333;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        font-size: 16px;
        font-family: 'Arial', sans-serif;
    }
    .result-box {
        border: 2px solid #4CAF50;
        padding: 15px;
        border-radius: 5px;
        background-color: #eaf7ea;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Load combined model file
model_dir = r"C:\Users\c0937432\OneDrive - Lambton College\Desktop\Aman Lambton\Bhavik Application Design for Big Data"
scaler_file = os.path.join(model_dir, 'scaler (1).pkl')


# Check if the file exists
if not os.path.exists(scaler_file):
    raise FileNotFoundError(f"Scaler file not found at: {scaler_file}")

# Load the scaler
scaler = joblib.load(scaler_file)


# Initialize session state for prediction history
if "history" not in st.session_state:
    st.session_state["history"] = []

# Streamlit UI
st.title("🚦 Real-Time Traffic Severity Prediction")
st.write("Input relevant traffic and weather features to predict traffic severity levels.")

# Input fields for features
st.write("### Input Features")
city = st.text_input("City", value="Toronto")
lon = st.number_input("Longitude", value=-79.3832, step=0.0001)
lat = st.number_input("Latitude", value=43.6532, step=0.0001)
weather_id = st.number_input("Weather ID", value=800, step=1)
weather_main = st.text_input("Weather Main", value="Clear")
weather_description = st.text_input("Weather Description", value="clear sky")
temp = st.number_input("Temperature (K)", value=293.15, step=0.1)
feels_like = st.number_input("Feels Like (K)", value=293.15, step=0.1)
temp_min = st.number_input("Minimum Temperature (K)", value=289.15, step=0.1)
temp_max = st.number_input("Maximum Temperature (K)", value=297.15, step=0.1)
pressure = st.number_input("Pressure (hPa)", value=1013, step=1)
humidity = st.number_input("Humidity (%)", value=50, step=1)
visibility = st.number_input("Visibility (m)", value=10000, step=100)
wind_speed = st.number_input("Wind Speed (m/s)", value=3.5, step=0.1)
wind_deg = st.number_input("Wind Direction (°)", value=270, step=1)
rain_1h = st.number_input("Rain Volume (mm)", value=0.0, step=0.1)
clouds_all = st.number_input("Cloudiness (%)", value=0, step=1)
sunrise = st.number_input("Sunrise Time (UTC, seconds)", value=1627477200, step=1)
sunset = st.number_input("Sunset Time (UTC, seconds)", value=1627527600, step=1)

# Select the model to use
st.write("### Select Prediction Model")
selected_model = st.selectbox("Choose a model for prediction:", ['Random Forest', 'XGBoost', 'Decision Tree', 'KNN'])

# Predict button
if st.button("Predict Traffic Severity"):
    # Prepare the input
    features = np.array([[lon, lat, weather_id, temp, feels_like, temp_min, temp_max, pressure, humidity, visibility, 
                          wind_speed, wind_deg, rain_1h, clouds_all, sunrise, sunset]])

    # Predict using the selected model
    model = models[selected_model]
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    # Add prediction to history
    st.session_state["history"].append({
        "Model": selected_model,
        "Prediction": int(prediction),
        "Probabilities": probabilities.tolist(),
        "Features": {
            "city": city,
            "lon": lon,
            "lat": lat,
            "weather_id": weather_id,
            "weather_main": weather_main,
            "weather_description": weather_description,
            "temp": temp,
            "feels_like": feels_like,
            "temp_min": temp_min,
            "temp_max": temp_max,
            "pressure": pressure,
            "humidity": humidity,
            "visibility": visibility,
            "wind_speed": wind_speed,
            "wind_deg": wind_deg,
            "rain_1h": rain_1h,
            "clouds_all": clouds_all,
            "sunrise": sunrise,
            "sunset": sunset
        }
    })

    # Display results in a styled box
    st.markdown(f"""
        <div class="result-box">
            <h3>Prediction using {selected_model}:</h3>
            <p><b>Predicted Severity Level:</b> {int(prediction)}</p>
            <h4>Probabilities for each level:</h4>
            <ul>
                <li><b>Low:</b> {probabilities[0]:.2f}</li>
                <li><b>Medium:</b> {probabilities[1]:.2f}</li>
                <li><b>High:</b> {probabilities[2]:.2f}</li>
                <li><b>Severe:</b> {probabilities[3]:.2f}</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # Plot probabilities
    st.write("### Probability Distribution")
    labels = ["Low", "Medium", "High", "Severe"]
    plt.bar(labels, probabilities, color=['green', 'yellow', 'orange', 'red'])
    plt.xlabel("Severity Level")
    plt.ylabel("Probability")
    plt.title("Probability Distribution for Severity Levels")
    st.pyplot(plt)

    # Explain the model
    st.write("### Model Explanation")
    if selected_model == "Random Forest":
        st.write("Random Forest uses an ensemble of decision trees to make predictions based on majority voting.")
    elif selected_model == "XGBoost":
        st.write("XGBoost is an optimized gradient-boosting framework known for high prediction accuracy.")
    elif selected_model == "Decision Tree":
        st.write("Decision Tree models classify data by splitting it into subsets based on feature values.")
    elif selected_model == "KNN":
        st.write("K-Nearest Neighbors classifies data based on the majority vote of its nearest neighbors.")

# Display prediction history
st.write("### Prediction History")
if st.session_state["history"]:
    history_df = pd.DataFrame(st.session_state["history"])
    st.write(history_df)
    st.download_button("Download History as CSV", data=history_df.to_csv(index=False), file_name="prediction_history.csv")
else:
    st.write("No predictions yet.")
