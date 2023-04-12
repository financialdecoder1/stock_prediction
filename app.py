import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
import plotly.graph_objs as go
# Set page title
st.set_page_config(page_title='Stock Prediction App')

# Define function to get stock data
@st.cache
def load_data(ticker):
    data = yf.download(ticker, start='2023-01-01')
    return data

# Define function to make stock predictions
@st.cache
# Define function to make stock predictions
@st.cache
def make_prediction(data, years):
    # Rename columns to fit Prophet format
    data = data[['Close']]
    data = data.rename(columns={'Close': 'y'})
    data['ds'] = data.index

    # Train Prophet model
    model = Prophet()
    model.fit(data)

    # Make predictions
    future = model.make_future_dataframe(periods=365*years)
    forecast = model.predict(future)

    # Get last date and prediction for that date
    last_date = forecast['ds'].iloc[-1]
    prediction = forecast.loc[forecast['ds'] == last_date, 'yhat'].iloc[0]

    return prediction

# Define Streamlit app
# Define Streamlit app
def main():
    # Set app title
    st.title('Stock Prediction App')

    # Get user input
    stocks = ('ADANIENT.NS', 'APOLLOHOSP.NS','BAJAJ-AUTO.NS', 'BAJAJFINSV.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS', 'ITC.NS', 'KOTAKBANK.NS', 'LT.NS', 'MARUTI.NS', 'MM.NS', 'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS', 'RELIANCE.NS', 'TATACONSUM.NS', 'TATASTEEL.NS', 'TCS.NS', 'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS')
    ticker = st.selectbox("Select dataset for prediction", stocks)
    years = st.slider('Select number of years to predict', min_value=1, max_value=10, value=2)

    # Load data
    data = load_data(ticker)

    # Display data
    st.subheader('Historical Data')
    st.write(data)

    # Make prediction
    future_dates = get_future_dates(data, years)
    predicted_values = make_prediction(data, years)
    predicted_values_list = list(predicted_values)

    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Historical Values'))
    fig.add_trace(go.Scatter(x=future_dates, y=predicted_values_list, name='Predicted Values'))

    st.plotly_chart(fig)

    # Display prediction
    st.subheader(f'Predicted closing price for {ticker} on the last day of {years} years from now')
    st.write(f'Predicted closing price: {round(prediction, 2)}')
    
# Run the app
if __name__ == '__main__':
    main()
