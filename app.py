import streamlit as st
from statsmodels.tsa.arima.model import ARIMAResults
import pandas as pd
import matplotlib.pyplot as plt

st.title("Household Power Consumption Forecast")
st.write("Forecasts daily average Global Active Power using an ARIMA(1,0,0) model.")

# Load the trained model
model = ARIMAResults.load('arima_model.pkl')

# User input
days = st.slider("How many days ahead do you want to forecast?", 1, 60, 7)

if st.button("Forecast"):
    forecast = model.forecast(steps=days)
    
    st.subheader(f"Forecast for the next {days} days")
    st.line_chart(forecast)
    
    st.write(forecast)