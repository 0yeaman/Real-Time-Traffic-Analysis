import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import joblib

# Page configuration
st.set_page_config(page_title="Real-Time Traffic Analysis", layout="wide")

# Title
st.title("Real-Time Traffic Analysis (RTTA)")

# Sidebar inputs
st.sidebar.header("User Input")

def fetch_real_time_data(api_url):
    """Fetch real-time traffic data from an API (placeholder function)."""
    # Replace with actual API logic
    try:
        data = pd.DataFrame({
            "Location": ["Location A", "Location B", "Location C"],
            "Traffic_Flow": [120, 200, 180],
            "Time": ["08:00 AM", "08:15 AM", "08:30 AM"]
        })
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Upload local traffic data
uploaded_file = st.sidebar.file_uploader("Upload Traffic Data (CSV):", type="csv")
if uploaded_file is not None:
    traffic_data = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")
else:
    # Fetch data from API (use your real API endpoint)
    traffic_data = fetch_real_time_data("ttps://api.openweathermap.org/data/2.5/weather?q=Toronto&appid=0daa089ec817d4921f4a1df8478dc369")

# Display raw data
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Traffic Data")
    st.dataframe(traffic_data)

# Traffic Flow Visualization
st.subheader("Traffic Flow Analysis")
if not traffic_data.empty:
    try:
        fig = px.bar(traffic_data, x="Location", y="Traffic_Flow", color="Traffic_Flow",
                     title="Traffic Flow at Different Locations")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error visualizing data: {e}")
else:
    st.warning("No data available for visualization.")

# Real-Time Prediction
if st.sidebar.button("Run Prediction"):
    try:
        # Paths to models and scaler
        model_path = "/content/combined_models.pkl"
        scaler_path = "/content/scaler.pkl"

        # Load the model and scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        # Mock input features
        sample_features = pd.DataFrame({
            "Feature1": [10],
            "Feature2": [20],
            "Feature3": [30]
        })

        # Scale and predict
        scaled_features = scaler.transform(sample_features)
        prediction = model.predict(scaled_features)

        st.sidebar.success(f"Prediction: {prediction[0]}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Real-time clock
st.sidebar.text(f"Current Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
