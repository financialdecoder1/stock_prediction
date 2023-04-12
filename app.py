import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import pypfopt
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from datetime import date
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
st.set_page_config(page_title="Portfolio Creator", page_icon=":chart_with_upwards_trend:")
st.header("Portfolio Creator")
st.sidebar.header("Portfolio Configuration")
portfolio_val = st.sidebar.number_input("Enter Investment Amount:" ,min_value=1000, max_value=100000, step=1000)
# portfolio_val = int(st.sidebar.selectbox("Select Investment Amount " ,('1000','2000','3000','15000','45000'))
risk = st.sidebar.selectbox("Select your risk level", ('low', 'medium', 'high'))
def fun():
  START = "2023-01-01"
  TODAY = "2023-04-11"
  stocks = ('ADANIENT.NS', 'APOLLOHOSP.NS','BAJAJ-AUTO.NS', 'BAJAJFINSV.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS', 'ITC.NS', 'KOTAKBANK.NS', 'LT.NS', 'MARUTI.NS', 'MM.NS', 'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS', 'RELIANCE.NS', 'TATACONSUM.NS', 'TATASTEEL.NS', 'TCS.NS', 'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS')
  df=[]
  for i in stocks:
    data = yf.download(i,START,TODAY) #returns pd df
    data.reset_index(inplace = True) #puts the date in 1st column
    df.append(data)
  new_df = pd.DataFrame(columns=['Stock', 'Date', 'Close'])
  for i in stocks:
      data = yf.download(i,START,TODAY)
      data.reset_index(inplace=True)
      data = data[['Date', 'Close']]
      data.columns = ['Date', 'Close']
      data['Stock'] = i
      new_df = new_df.append(data)
  new_df = new_df.pivot(index='Date', columns='Stock', values='Close')
  new_df = new_df.dropna()
  new_df.reset_index(inplace=True)
  new_df.columns.name = None
  df=new_df
  df = df.set_index(pd.DatetimeIndex(df['Date'].values))
#Remove the Date column
  df.drop(columns=['Date'], axis=1, inplace=True)
  def risk_factor(risk):
#mu =expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)
    if(risk=='low'):
      mu=pypfopt.expected_returns.capm_return(df, market_prices=None, returns_data=False, risk_free_rate=0.1, compounding=True, frequency=252, log_returns=False)
    elif(risk=='medium'):
      mu=pypfopt.expected_returns.capm_return(df, market_prices=None, returns_data=False, risk_free_rate=0.08, compounding=True, frequency=252, log_returns=False)
    elif(risk=='high'):
      mu=pypfopt.expected_returns.capm_return(df, market_prices=None, returns_data=False, risk_free_rate=0.05, compounding=True, frequency=252, log_returns=False)
      #mu =expected_returns.mean_historical_return(df)
    return S,mu
  assets = df.columns
  S,mu=risk_factor(risk)
  ef = EfficientFrontier(mu, S) #Create the Efficient Frontier Object
  weights = ef.max_sharpe()
  cleaned_weights = ef.clean_weights()
  ef.portfolio_performance(verbose=True)
  latest_prices = get_latest_prices(df)
  weights = cleaned_weights
  da = DiscreteAllocation(weights, latest_prices, total_portfolio_value = portfolio_val)
  allocation, leftover = da.lp_portfolio()
  r=portfolio_val-leftover
  portfolio_df=pd.DataFrame(columns=['Company_name','Discrete_Value_'+str(r)])
  l=[]
  m=[]
  for i,j in allocation.items():
    l.append(i)
    m.append(j)
  portfolio_df['Company_name']=l
  portfolio_df['Discrete_Value_'+str(r)]=m
  st.write("Stock Holdings")
  st.write(portfolio_df.tail())
  fig,ax=plt.subplots()
  ax.pie(portfolio_df['Discrete_Value_'+str(r)], autopct='%1.1f%%', startangle=90, labels=portfolio_df['Company_name'])
  ax.axis('equal')
  st.pyplot(fig)
  crypto = ( 'SAFEMOON-INR', 'HOT-INR', 'RVN-INR', 'CHZ-INR', 'MATIC-INR', 'ALGO-INR', 'ONE-INR', 'DASH-INR', 'BTG-INR', 'KAVA-INR', 'ZIL-INR','BTC-INR', 'ETH-INR', 'ADA-INR', 'BNB-INR', 'XRP-INR')
  df1=[]
  crypto_names = {'SAFEMOON-INR': 'Safemoon','HOT-INR': 'Holo','RVN-INR': 'Ravencoin','CHZ-INR': 'Chiliz','MATIC-INR': 'Polygon','ALGO-INR': 'Algorand','ONE-INR': 'Harmony','DASH-INR': 'Dash','BTG-INR': 'Bitcoin Gold','KAVA-INR': 'Kava','ZIL-INR': 'Zilliqa','BTC-INR': 'Bitcoin','ETH-INR': 'Ethereum','ADA-INR': 'Cardano','BNB-INR': 'Binance Coin','XRP-INR': 'XRP'}
  for i in crypto:
   data1 = yf.download(i,START,TODAY) #returns pd df
   data1.reset_index(inplace = True) #puts the date in 1st column
   df1.append(data1)
  new_df1 = pd.DataFrame(columns=['Crypto', 'Date', 'Close'])
  for i in crypto:
    data1 = yf.download(i,START,TODAY)
    data1.reset_index(inplace=True)
    data1 = data1[['Date', 'Close']]
    data1.columns = ['Date', 'Close']
    data1['Crypto'] = i
    new_df1 = new_df1.append(data1)
  new_df1 = new_df1.pivot(index='Date', columns='Crypto', values='Close')
  new_df1 = new_df1.dropna()
  new_df1.reset_index(inplace=True)
  new_df1.columns.name = None
  df1=new_df1
  df1 = df1.set_index(pd.DatetimeIndex(df1['Date'].values))
#Remove the Date column
  df1.drop(columns=['Date'], axis=1, inplace=True)
  def risk_factor1(risk):
#mu1 =expected_returns.mean_historical_return(df1)
    S1 = risk_models.sample_cov(df1)
    if(risk=='low'):
      mu1=pypfopt.expected_returns.capm_return(df1, market_prices=None, returns_data=False, risk_free_rate=0.1, compounding=True, frequency=252, log_returns=False)
    elif(risk=='medium'):
      mu1=pypfopt.expected_returns.capm_return(df1, market_prices=None, returns_data=False, risk_free_rate=0.08, compounding=True, frequency=252, log_returns=False)
    elif(risk=='high'):
      mu1=pypfopt.expected_returns.capm_return(df1, market_prices=None, returns_data=False, risk_free_rate=0.05, compounding=True, frequency=252, log_returns=False)
      #mu1 =expected_returns.mean_historical_return(df1)
    return S1,mu1
  assets1 = df1.columns
  S1,mu1=risk_factor1(risk)
  ef1 = EfficientFrontier(mu1, S1) #Create the Efficient Frontier Object
  weights1 = ef1.max_sharpe()
  cleaned_weights1 = ef1.clean_weights()
  ef1 .portfolio_performance(verbose=True)
  portfolio_val1 = leftover
  latest_prices1 = get_latest_prices(df1)
  weights1 = cleaned_weights1
  da1 = DiscreteAllocation(weights1, latest_prices1, total_portfolio_value = portfolio_val1)
  allocation1, leftover1 = da1.lp_portfolio()
  r1=portfolio_val1-leftover1
  portfolio_df1=pd.DataFrame(columns=['Company_Name','Company_Sym','Discrete_'+str(r1)])
  l1=[]
  m1=[]
  k1=[]
  for i,j in allocation1.items():
    l1.append(i)
    m1.append(j)
  portfolio_df1['Company_Sym']=l1
  portfolio_df1['Discrete_'+str(r1)]=m1
  for i in l1:
    if i in crypto_names:
      k1.append(crypto_names[i])
  portfolio_df1['Company_Name']=k1
  st.write("Crypto Holdings")
  st.write(portfolio_df1.tail())
  fig1,ax1=plt.subplots()
  ax1.pie(portfolio_df1['Discrete_'+str(r1)], autopct='%1.1f%%', startangle=90, labels=portfolio_df1['Company_Name'])
  ax1.axis('equal')
  st.pyplot(fig1)
if st.sidebar.button("Create Portfolio"):
  fun()