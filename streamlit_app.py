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
        background-color: #f0f0f0;
    }
    h1 {
        color: #007BFF;
        text-align: center;
        font-family: 'Verdana', sans-serif;
    }
    h2, h3, .stSelectbox, .stNumberInput {
        color: #333333;
        font-family: 'Verdana', sans-serif;
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
        border-radius: 5px;
        font-size: 16px;
        font-family: 'Verdana', sans-serif;
    }
    .result-box {
        border: 2px solid #007BFF;
        padding: 15px;
        border-radius: 5px;
        background-color: #e8f4ff;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Load models and scaler
model_dir = r"C:\\Users\\brian\\RTTA_Models"
models = {
    'Random Forest': joblib.load(os.path.join(model_dir, 'RTTA_RandomForest.pkl')),
    'Gradient Boosting': joblib.load(os.path.join(model_dir, 'RTTA_GradientBoosting.pkl')),
    'SVM': joblib.load(os.path.join(model_dir, 'RTTA_SVM.pkl')),
    'Logistic Regression': joblib.load(os.path.join(model_dir, 'RTTA_LogisticRegression.pkl')),
}
scaler = joblib.load(os.path.join(model_dir, 'rtta_scaler.pkl'))

# Initialize session state for prediction history
if "history" not in st.session_state:
    st.session_state["history"] = []

# Streamlit UI
st.title("🚗 Real-Time Traffic Severity Analysis")
st.write("Enter the values for the following features to predict traffic severity:")

# Input fields for features
st.write("### Input Features")
city = st.text_input("City")
lon = st.number_input("Longitude", value=-79.0, step=0.1)
lat = st.number_input("Latitude", value=43.0, step=0.1)
weather_id = st.number_input("Weather ID", value=800, step=1)
temp = st.number_input("Temperature (°C)", value=20.0, step=1.0)
feels_like = st.number_input("Feels Like Temperature (°C)", value=20.0, step=1.0)
temp_min = st.number_input("Minimum Temperature (°C)", value=18.0, step=1.0)
temp_max = st.number_input("Maximum Temperature (°C)", value=22.0, step=1.0)
pressure = st.number_input("Pressure (hPa)", value=1013, step=1)
humidity = st.number_input("Humidity (%)", value=50, step=1)
visibility = st.number_input("Visibility (m)", value=10000, step=100)
wind_speed = st.number_input("Wind Speed (m/s)", value=3.0, step=0.1)
wind_deg = st.number_input("Wind Direction (°)", value=180, step=1)
rain_1h = st.number_input("Rain Volume (1 hour, mm)", value=0.0, step=0.1)
clouds_all = st.number_input("Cloud Coverage (%)", value=20, step=1)
sunrise = st.number_input("Sunrise (Unix Time)", value=1672531200, step=1)
sunset = st.number_input("Sunset (Unix Time)", value=1672574400, step=1)

# Validation for input values
if temp < -50 or temp > 60:
    st.error("Temperature must be between -50 and 60 degrees Celsius!")
elif humidity < 0 or humidity > 100:
    st.error("Humidity must be between 0 and 100%!")
else:
    # Select the model to use
    st.write("### Select Prediction Model")
    selected_model = st.selectbox("Choose a model for prediction:", list(models.keys()))

    # Predict button
    if st.button("Predict Traffic Severity"):
        # Prepare the input
        features = np.array([[lon, lat, weather_id, temp, feels_like, temp_min, temp_max, pressure, humidity, visibility, wind_speed, wind_deg, rain_1h, clouds_all, sunrise, sunset]])
        scaled_features = scaler.transform(features)

        # Predict using the selected model
        model = models[selected_model]
        prediction = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]

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
                <p><b>Predicted Traffic Severity:</b> {int(prediction)}</p>
                <h4>Probabilities for each severity level:</h4>
                <ul>
                    <li><b>Low:</b> {probabilities[0]:.2f}</li>
                    <li><b>Moderate:</b> {probabilities[1]:.2f}</li>
                    <li><b>High:</b> {probabilities[2]:.2f}</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        # Plot probabilities
        st.write("### Probability Distribution")
        labels = ["Low", "Moderate", "High"]
        plt.bar(labels, probabilities, color=['green', 'orange', 'red'])
        plt.xlabel("Traffic Severity Level")
        plt.ylabel("Probability")
        plt.title("Probability Distribution for Traffic Severity Levels")
        st.pyplot(plt)

        # Explain the model
        st.write("### Model Explanation")
        if selected_model == "Random Forest":
            st.write("Random Forest combines multiple decision trees to improve prediction accuracy.")
        elif selected_model == "Gradient Boosting":
            st.write("Gradient Boosting builds sequential trees to optimize the prediction outcome.")
        elif selected_model == "SVM":
            st.write("SVM separates data into classes using a hyperplane for optimal classification.")
        elif selected_model == "Logistic Regression":
            st.write("Logistic Regression predicts probabilities of categorical outcomes.")

# Display prediction history
st.write("### Prediction History")
if st.session_state["history"]:
    history_df = pd.DataFrame(st.session_state["history"])
    st.write(history_df)
    st.download_button("Download History as CSV", data=history_df.to_csv(index=False), file_name="traffic_prediction_history.csv")
else:
    st.write("No predictions yet.")
