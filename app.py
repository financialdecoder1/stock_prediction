import streamlit as st
from datetime import date
import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import dates as mdates

START = "2023-01-01"
TODAY = "2023-04-11"
st.title("Stock Prediction App")
stocks = ('ADANIENT.NS', 'APOLLOHOSP.NS','BAJAJ-AUTO.NS', 'BAJAJFINSV.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS', 'ITC.NS', 'KOTAKBANK.NS', 'LT.NS', 'MARUTI.NS', 'MM.NS', 'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS', 'RELIANCE.NS', 'TATACONSUM.NS', 'TATASTEEL.NS', 'TCS.NS', 'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS')
selected_stock = st.selectbox("Select dataset for prediction", stocks)
n_years = st.slider("Years of predictions: ", 1, 4)
period = n_years*365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data... Done!")

st.subheader('Raw Data')
st.write(data.tail())

def plot_raw_data():
    fig, ax = plt.subplots()
    ax.plot(data['Date'], data['Close'], label='stock_close')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Time Series Data')
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    ax.legend()
    st.pyplot(fig)

plot_raw_data()

# Linear Regression
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date":"ds","Close":"y"})
df_train['ds'] = pd.to_datetime(df_train['ds'])
df_train['ds'] = df_train['ds'].map(mdates.date2num)

X_train = np.array(df_train['ds']).reshape(-1, 1)
y_train = np.array(df_train['y'])

model = LinearRegression()
model.fit(X_train, y_train)

last_date = mdates.num2date(df_train['ds'].iloc[-1])
future_dates = np.array([pd.to_datetime(str(df_train['ds'].iloc[-1])) + pd.DateOffset(days=x) for x in range(1, period+1)])
future_dates = np.vectorize(mdates.date2num)(future_dates)
future_dates = future_dates.reshape(-1, 1)


y_pred = model.predict(future_dates)
forecast = pd.DataFrame({'ds': future_dates.flatten(), 'y': y_pred})

forecast['ds'] = pd.to_datetime(forecast['ds'])
forecast['ds'] = forecast['ds'].map(mdates.date2num)

st.subheader('Predicted Data')

fig, ax = plt.subplots()
ax.plot(data['Date'], data['Close'], label='Actual')
ax.plot(forecast['ds'], forecast['y'], label='Predicted')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('Predicted Data')
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
ax.legend()
st.pyplot(fig)

st.subheader('Predicted Data')
st.write(forecast.tail())

st.subheader('Prediction Data Statistics')
st.write(forecast.describe())

st.subheader('Predicted Data')
st.write(forecast)
