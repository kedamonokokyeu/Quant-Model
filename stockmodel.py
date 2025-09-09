import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import yfinance as yf

print("Enter the company ticker you want me to predict. e.g. AAPL")
ticker_symbol = input()
ticker = yf.Ticker(ticker_symbol)

def OHLCV_data_analyzer(): # here we find what's important from OHLCV data
    df = yf.download(
        ticker_symbol, 
        start="2015-01-01", 
        end="2025-01-01", 
        interval="1d"
    )
    # the most important part of OHLC data, it is standardized and you want a positive % change
    df["Return_1m"] = df["Close"].pct_change(21)   # 21 trading days ≈ 1 month
    df["Return_3m"] = df["Close"].pct_change(63)   # 63 trading days ≈ 1 quarter
    df["Return_12m"] = df["Close"].pct_change(252) # 252 trading days ≈ 1 year

    # how the averages move across long periods of time
    df["SMA_50"] = df["Close"].rolling(50).mean()   # 50-day ≈ 2 months
    df["SMA_200"] = df["Close"].rolling(200).mean() # 200-day ≈ 10 months
    df["SMA_Ratio"] = df["SMA_50"] / df["SMA_200"]

    # calculates sd from the mean of volatility for the past 63 and 252 days
    df["Volatility_3m"] = df["Close"].pct_change().rolling(63).std()
    df["Volatility_12m"] = df["Close"].pct_change().rolling(252).std()

    # average volume
    df["AvgVolume_3m"] = df["Volume"].rolling(63).mean()

def valuation_ratios():
    pe_ratio = ticker.info.get("trailingPE")

    column = ticker.quarterly_financials.T["Total Revenue"]
    revenue_growth = column.pct_change(4).iloc[-1] # take the most recent quarter’s revenue and compare it to revenue 4 quarters ago, and report that year-over-year growth rate

    financial_q = ticker.quarterly_financials.T
    profit_margin = financial_q["Net Income"].iloc[-1] / financial_q["Total Revenue"].iloc[-1]

    balance_sheet = ticker.quarterly_balance_sheet.T
    debt_equity = balance_sheet["Total Liab"].iloc[-1] / balance_sheet["Total Stockholder Equity"].iloc[-1]

    current_ratio = balance_sheet["Total Current Assets"].iloc[-1] /  balance_sheet["Total Current Liabilities"].iloc[-1]
    dividend_yield = ticker.info.get("dividendYield")

    return df  

df = OHLCV_data_analyzer()

# display last five rows, we only get NaN for the first 200 because those have no past days to "roll" back upon
print(df.tail())
