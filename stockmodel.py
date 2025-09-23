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

    return df

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

    return pe_ratio, revenue_growth, profit_margin, debt_equity, dividend_yield

#RSI: measures the speed and change of price movements --> a high value suggest a stock is overbought, a low value suggest its oversold
def rsi_calculation(data, window = 21): # usually 14, but we want long term so...
    close_difference = data['Close'].diff()

    gain = close_difference.where(close_difference > 0, 0) # we need average gain / average loss to calculate RSI line
    loss = -close_difference.where(close_difference < 0, 0) #negative sign so we can calculate ratio for RS, don't want negative ratio

    avg_gain = gain.ewm(com = window - 1, min_periods = window).mean() #can't just calculate average gain and loss simply, because recent values matter MORE
    avg_loss = loss.ewm(com = window - 1, min_periods = window).mean()

    relative_strength = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi

#MACD: indicates how strong and in what direction a trend for a stock is --> measures momentum
def macd_calculation(data, short_window = 21, long_window = 50, signal_window = 15): # CANNOT preset window data, since we want to try different windows later

    # calculate exponential moving averages --> we need them for the MACD line
    short_ema = data['Close'].ewm(com = short_window, adjust = False).mean() #adjust = False because we want weighted averages of the time frame, not of all history
    long_ema = data['Close'].ewm(com = long_window, adjust = False).mean()
    macd = short_ema - long_ema

    # calculate the signal line (used to smooth the MACD line)
    signal_line = macd.ewm(span = signal_window, adjust=False).mean() # if MACD > signal, bullish. if MACD < signal, bearish
    indicator = macd - signal_line
    return macd, signal_line, indicator

# Bollinger Bands: measure volatility using the upper band and lower band
def bollinger_bands_calculation(data, window = 20):

    #calculate the bands (look at google doc for mathematical notation)
    middle_band = data['Close'].rolling(window).mean()
    #upper band is middle band + (2 * standard deviations), and lower band is middle band - that
    #standard deviation changes with rolling window, so we must change the rolling window as well
    std = data['Close'].rolling(window).std()
    upper_band = middle_band + (2 * std)
    lower_band = middle_band - (2 * std)
    return upper_band, lower_band

def schocastic_oscillator(data, window = 21, smoothing_window = 3):
    lowest = data['Low'].rolling(window).min()
    highest = data['High'].rolling(window).max()
    percent_k = (data["Close"] - lowest)/(highest - lowest) * 100

    # traders smooth percent_k with %D, a 3 day simple-moving average to "smooth out the noise"
    percent_d = percent_k.rolling(smoothing_window).mean()
    return percent_d, percent_k

# OBV: the On-Balance Volume (OBV) is a technical indicator that reflects the buying or selling pressure of a stock
def calculate_OBV(data):
    # if close of today > close of yesterday you add volume from the yfinance, and vice versa
    obv = [0] # we have to change OBV into dataframe later

    for i in range(1, len(data)):
        if data['Close'].iloc[i] > data['Close'].iloc[i - 1]: # our data is from oldest to newest, so this is today - yesterday
            obv.append(obv[-1] + data["Volume"].iloc[i])  # price up so add volume
        elif data["Close"].iloc[i] < data["Close"].iloc[i - 1]:
            obv.append(obv[-1] - data["Volume"].iloc[i])  # price down so minus volume
        else:
            obv.append(obv[-1])  # price unchanged so no change in OBV
    return obv



