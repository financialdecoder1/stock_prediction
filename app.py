import streamlit as st
from datetime import date
import yfinance as yf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

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

# LSTM Model
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date":"ds","Close":"y"})
df_train['ds'] = pd.to_datetime(df_train['ds'])

# Add more features
df_train['year'] = df_train['ds'].dt.year
df_train['month'] = df_train['ds'].dt.month
df_train['day'] = df_train['ds'].dt.day
df_train['dayofweek'] = df_train['ds'].dt.dayofweek
df_train['dayofyear'] = df_train['ds'].dt.dayofyear
df_train['weekofyear'] = df_train['ds'].dt.isocalendar().week.astype(int)

X_train = df_train.drop(['ds', 'y'], axis=1)
y_train = df_train['y']

# Normalize data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
y_train = scaler.fit_transform(y_train.values.reshape(-X_train.shape[1], 1))
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
    
# Early stopping callback to prevent overfitting
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=10)

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=32, callbacks=[early_stop])
# Prepare the test data
test_start_date = date.today()
test_end_date = test_start_date.replace(year=test_start_date.year + n_years)
test_data = yf.download(selected_stock, test_start_date.strftime('%Y-%m-%d'), test_end_date.strftime('%Y-%m-%d'))
test_data.reset_index(inplace=True)
test_data = test_data.rename(columns={"Date":"ds","Close":"y"})
test_data['ds'] = pd.to_datetime(test_data['ds'])
test_data['year'] = test_data['ds'].dt.year
test_data['month'] = test_data['ds'].dt.month
test_data['day'] = test_data['ds'].dt.day
test_data['dayofweek'] = test_data['ds'].dt.dayofweek
test_data['dayofyear'] = test_data['ds'].dt.dayofyear
test_data['weekofyear'] = test_data['ds'].dt.isocalendar().week.astype(int)
X_test = test_data.drop(['ds', 'y'], axis=1)
y_test = test_data['y']
X_test = scaler.transform(X_test)

# Predict the stock prices
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)

# Plot the predictions
fig, ax = plt.subplots()
ax.plot(test_data['ds'], y_test, label='Actual')
ax.plot(test_data['ds'], y_pred, label='Predicted')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('Predictions')
ax.legend()
st.pyplot(fig)

