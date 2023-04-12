import streamlit as st
from datetime import date

import yfinance as yf 
from fbprophet import Prophet 
from fbprophet.plot import plot_plotly 
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = "2023-04-11"
st.title("Stock Prediciton App")
stocks = ('ADANIENT.NS', 'APOLLOHOSP.NS','BAJAJ-AUTO.NS', 'BAJAJFINSV.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS', 'ITC.NS', 'KOTAKBANK.NS', 'LT.NS', 'MARUTI.NS', 'MM.NS', 'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS', 'RELIANCE.NS', 'TATACONSUM.NS', 'TATASTEEL.NS', 'TCS.NS', 'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS')
selected_stock = st.selectbox("Select dataset for prediction",stocks)
n_years = st.slider("Years of predictions: ",1,4)
period = n_years*365

@st.cache_data
def load_data(ticker):
  data = yf.download(ticker,START,TODAY) #returns pd df
  data.reset_index(inplace = True) #puts the date in 1st column
  return data 

data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data... Done!")

st.subheader('Raw Data')
st.write(data.tail())

def plot_raw_data():
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='stock_open'))
  fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='stock_open'))
  fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
  st.plotly_chart(fig)

plot_raw_data()

# Forcasting using fbprophet
df_train=data[['Date','Close']]
df_train = df_train.rename(columns={"Date":"ds","Close":"y"})

# create model
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)

forecast = m.predict(future)
st.subheader('Forcast Data')
st.write(forecast.tail())

st.write('forcast data')
fig1 = plot_plotly(m,forecast)
st.plotly_chart(fig1)

st.write('forcast components')
fig2=m.plot_components(forecast)
st.write(fig2)
