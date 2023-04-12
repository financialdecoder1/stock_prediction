import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
stocks = ('ADANIENT.NS', 'APOLLOHOSP.NS','BAJAJ-AUTO.NS', 'BAJAJFINSV.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS', 'ITC.NS', 'KOTAKBANK.NS', 'LT.NS', 'MARUTI.NS', 'MM.NS', 'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS', 'RELIANCE.NS', 'TATACONSUM.NS', 'TATASTEEL.NS', 'TCS.NS', 'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS')
# Define function to get stock data
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start_date, end_date)
    return data

# Define function to create dataset
def create_dataset(dataset, time_steps=1):
    X, y = [], []
    for i in range(len(dataset)-time_steps):
        X.append(dataset[i:(i+time_steps)])
        y.append(dataset[i+time_steps])
    return np.array(X), np.array(y)

# Define app
def main():
    # Set app title
    st.title("Stock Price Prediction App")

    # Define sidebar
    st.sidebar.title("Enter Stock Information")

    # Get user input
    ticker = st.selectbox("Select dataset for prediction", stocks)
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2022-01-01"))

    # Get stock data
    data = get_stock_data(ticker, start_date, end_date)

    # Display stock data
    st.write("### Stock Data")
    st.write(data)

    # Create dataset
    df = pd.DataFrame(data["Close"])
    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(df)
    time_steps = 6
    X, y = create_dataset(dataset, time_steps)

    # Split dataset into training and testing sets
    split_ratio = 0.8
    split_index = int(split_ratio * len(dataset))
    X_train, y_train = X[:split_index], y[:split_index]
    X_test, y_test = X[split_index:], y[split_index:]

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")

    # Fit model
    early_stop = EarlyStopping(monitor="val_loss", patience=5, verbose=1)
    history = model.fit(X_train.reshape((X_train.shape[0], X_train.shape[1], 1)), y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stop])

    # Plot training and validation loss
    fig, ax = plt.subplots()
    ax.plot(history.history["loss"], label="Training Loss")
    ax.plot(history.history["val_loss"], label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    st.pyplot(fig)

    # Make predictions
    y_train_pred = model.predict(X_train.reshape((X_train.shape[0], X_train.shape[1], 1)))
    y_test_pred = model.predict(X_test.reshape((X_test.shape[0], X_test.shape[1], 1)))

    # Transform predictions back to original scale
    y_train_pred = scaler.inverse_transform(y_train_pred)
    y_train_true = scaler.inverse_transform(y_train[:len(y_train_pred)])
    y_test_pred = scaler.inverse_transform(y_test_pred
