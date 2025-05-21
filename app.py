import streamlit as st
import pandas as pd
import statsmodels.api as sm
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
from prophet.plot import plot_plotly, plot_components_plotly

# Set page title
st.title("Air Passengers Forecasting App")

# Load the dataset
st.subheader("Step 1: View the Air Passengers Dataset")
data = sm.datasets.get_rdataset("AirPassengers", "datasets").data
data['ds'] = pd.date_range(start='1949-01-01', periods=len(data), freq='M')
data = data.rename(columns={'value': 'y'})[['ds', 'y']]

# Show the dataset
st.write("This dataset shows monthly air passengers from 1949 to 1960.")
st.dataframe(data)

# Plot the raw data
st.subheader("Historical Passenger Data")
fig = px.line(data, x='ds', y='y', title="Monthly Air Passengers (1949–1960)")
st.plotly_chart(fig)

# Let user choose forecast horizon
st.subheader("Step 2: Choose Forecast Horizon")
horizon = st.selectbox("How many months to forecast?", [6, 12, 24])

# Button to trigger forecast
if st.button("Generate Forecast"):
    st.subheader("Step 3: Forecast Results")
    
    # Create and train the Prophet model
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(data)
    
    # Create future dates
    future = model.make_future_dataframe(periods=horizon, freq='ME')
    forecast = model.predict(future)
    
    # Show forecast plot
    st.write(f"Forecast for the next {horizon} months:")
    fig_forecast = plot_plotly(model, forecast)
    st.plotly_chart(fig_forecast)
    
    # Show components (trend, seasonality)
    st.subheader("Forecast Components (Trend and Seasonality)")
    fig_components = plot_components_plotly(model, forecast)
    st.plotly_chart(fig_components)

st.write("Note: The forecast shows predicted passenger numbers (blue line) with uncertainty bands (shaded area). The components plot breaks down the forecast into trend (overall growth) and yearly seasonality (monthly patterns).")