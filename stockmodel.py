import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
        interval="1d",
        group_by="column"  # prevents multi-level columns
    )

    # --- Fix for MultiIndex or multi-column issue ---
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Sometimes yfinance still leaves sub-DataFrames — force select first column
    if isinstance(df["Close"], pd.DataFrame):
        df["Close"] = df["Close"].iloc[:, 0]
        df["High"]  = df["High"].iloc[:, 0]
        df["Low"]   = df["Low"].iloc[:, 0]
        df["Open"]  = df["Open"].iloc[:, 0]
        df["Volume"] = df["Volume"].iloc[:, 0]

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

    # rolling Sharpe ratio
    rolling_return = df["Close"].pct_change().rolling(63).mean()
    rolling_volume = df["Close"].pct_change().rolling(63).std()
    df["Sharpe_3m"] = rolling_return / (rolling_volume + 1e-6) # add the 1e-6 value because of rolling volume is super small then we compiler might think we divide by 0, so we throw that in instead

    # rolling Max Drawdown
    roll_max = df["Close"].rolling(63, min_periods=1).max()
    daily_drawdown = df["Close"] / roll_max - 1.0
    df["MaxDrawdown_3m"] = daily_drawdown.rolling(63, min_periods=1).min()

    # average volume
    df["AvgVolume_3m"] = df["Volume"].rolling(63).mean()
    df["VolumeVolatility"] = df["AvgVolume_3m"] / df["Volatility_3m"]

    # adding lagged returns columns to df (just some extra info for the RandomForestClassifier)
    df["1day_lagged"] = df["Close"].pct_change(1)
    df["5day_lagged"] = df["Close"].pct_change(5)
    df["10day_lagged"] = df["Close"].pct_change(10)

    # STATISTICS --- rolling skewedness and kurtosis (check doc documentation if unsure)
    df["Skewness_3months"] = df["Close"].pct_change().rolling(63).skew()
    df["Kurtosis_3months"] = df["Close"].pct_change().rolling(63).kurt()

    # Moving Average Convergence Ratios
    df["Price_vs_SMA50"] = df["Close"] / df["SMA_50"]
    df["Price_vs_SMA200"] = df["Close"] / df["SMA_200"]

    # Target Variable Design
    df["Target_3m"] = df["Close"].shift(-63) / df["Close"] - 1
    df["Target_3m"] = (df["Target_3m"] > 0.05).astype(int)  # classification

    df = df.dropna() # b/c we used rolling for some of these df so we just get NaN before the window is filled
    
    return df

def valuation_ratios():
    financial_q = ticker.quarterly_financials.T.copy()
    balance_sheet = ticker.quarterly_balance_sheet.T.copy()

    COLUMN_ALIASES = { # remember to add into debug documentation---yfinance can change column labels depending on the ticker
        "Total Liab": ["Total Liab", "Total Liabilities Net Minority Interest"],
        "Total Stockholder Equity": ["Total Stockholder Equity", "Ordinary Shares", "Total Equity Gross Minority Interest"],
        "Total Current Assets": ["Total Current Assets"],
        "Total Current Liabilities": ["Total Current Liabilities"]
    }

    # written so whichever column alias shows up first is used
    def get_first_available(df, candidates):
        for c in candidates:
            if c in df.columns:
                return df[c]
        return None

    if "Net Income" in financial_q.columns and "Total Revenue" in financial_q.columns:
        financial_q["ProfitMargin"] = financial_q["Net Income"] / financial_q["Total Revenue"]
        financial_q["Revenue_YoY"] = financial_q["Total Revenue"].pct_change(4)
        financial_q["ProfitMargin_YoY"] = financial_q["ProfitMargin"].pct_change(4)
    else:
        financial_q["ProfitMargin"] = None
        financial_q["Revenue_YoY"] = None
        financial_q["ProfitMargin_YoY"] = None

    total_liab = get_first_available(balance_sheet, COLUMN_ALIASES["Total Liab"])
    equity = get_first_available(balance_sheet, COLUMN_ALIASES["Total Stockholder Equity"])
    if total_liab is not None and equity is not None:
        balance_sheet["DebtEquity"] = total_liab / equity
        balance_sheet["DebtEquity_QoQ"] = balance_sheet["DebtEquity"].pct_change()
        balance_sheet["DebtEquity_YoY"] = balance_sheet["DebtEquity"].pct_change(4)
    else:
        balance_sheet["DebtEquity"] = None
        balance_sheet["DebtEquity_QoQ"] = None
        balance_sheet["DebtEquity_YoY"] = None

    current_assets = get_first_available(balance_sheet, COLUMN_ALIASES["Total Current Assets"])
    current_liabilities = get_first_available(balance_sheet, COLUMN_ALIASES["Total Current Liabilities"])
    if current_assets is not None and current_liabilities is not None:
        balance_sheet["CurrentRatio"] = current_assets / current_liabilities
        balance_sheet["CurrentRatio_QoQ"] = balance_sheet["CurrentRatio"].pct_change()
    else:
        balance_sheet["CurrentRatio"] = None
        balance_sheet["CurrentRatio_QoQ"] = None

    # --- Combine ---
    fundamentals = pd.concat([
        financial_q[["Revenue_YoY", "ProfitMargin", "ProfitMargin_YoY"]],
        balance_sheet[["DebtEquity", "DebtEquity_QoQ", "DebtEquity_YoY",
                       "CurrentRatio", "CurrentRatio_QoQ"]]
    ], axis=1)

    fundamentals = fundamentals.resample("D").ffill()

    return fundamentals

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

def stochastic_oscillator(data, window = 21, smoothing_window = 3):
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

def relative_strength_comparison(df, comparisons, start =  None, end = None): # implement user input for comparison later, which will take in a list of ETF and index fund tickers to compare the stock to
    relative_strength = pd.DataFrame(index = df.index) # index = df.index to align timeline with stock df
    stock_return = df["Close"].pct_change()

    for ticker in comparisons:
        etf_ticker = yf.download(ticker, start = start, end = end)
        etf = etf_ticker["Close"]
        etf = etf.reindex(df.index).fillna(method = "ffill") # if there are empty values in the etf_ticker, fill it with last known value and match it with stock data timeline
        etf_return = etf.pct_change()
        relative_strength[f"RS_{ticker}"] = stock_return - etf_return # f-string so that the dataframe's columns 

    return relative_strength
   
def combine_features_dataset(comparisons = None): # just in case the user doesn't input ETFs or index funsd to compare the stock to

    df = OHLCV_data_analyzer()
    fundamentals = valuation_ratios()
    df = df.merge(fundamentals, how = "left", left_index = True, right_index = True) # left_index and right_index true because the indexes represent dates from the ticker, and we do how = "left" b/c fundamentals data is only for quarters, we fill in later for the daily data
    
    df["RSI"] = rsi_calculation(df)

    macd, signal, indicator = macd_calculation(df) # remember return line of macd_calculation(), so macd = macd, signal = signal_line, indicator = indicator
    df["MACD"] = macd
    df["MACD_Signal"] = signal
    df["MACD_Indicator"] = indicator

    upper, lower = bollinger_bands_calculation(df)
    df["Upper_Bollinger"] = upper
    df["Lower_Bollinger"] = lower

    percent_d, percent_k = stochastic_oscillator(df)
    df["Stochastic_%D"] = percent_d
    df["Stochastic_%K"] = percent_k

    df["OBV"] = calculate_OBV(df)

    if comparisons != None:
        relative_strength_df = relative_strength_comparison(df, comparisons) # remember relative_strength_comparison returns a dataframe of these comparisons
        df = df.join(relative_strength_df) # add to our big dataframe

    df = df.ffill().dropna() # so quarterly values will be applied to every day in that quarter until updated instead of being NaN values
    
    return df

# ---------------------- BUILDING THE MODEL AND TUNING ---------------------- #

df = combine_features_dataset()

X = df.drop(columns = ["Target_3m"]) # given all the data from the dataset, we want to predict Target_3m, 3 months into the future, but we drop it so the model cannot see the future
y = df["Target_3m"] # what we want to predict

# can't shuffle the data around for training and testing since finance is time-series
split_date = "2021-01-01"
X_train, X_test = X.loc[:split_date].iloc[:-1], X.loc[split_date:] # added iloc[:-1] all rows except the very last row to prevent data leakage, since both the testing and the training include the split date
y_train, y_test = y.loc[:split_date].iloc[:-1], y.loc[split_date:] 

clf = RandomForestClassifier(random_state = 42)
clf.fit(X_train, y_train) # give the model these parameters to learn off X_train and learn to predict y_train, the labels we want to predict
y_pred = clf.predict(X_test) # features from the test period from 2021 to 2025 and predict labels of 0 to 1 for y_pred, and we'll compare to y_test which is the answer key

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))



   # user_input = input("Enter the ETFs and Index Fund Tickers that you'd like to compare to, and separate them by a space.") #SAVE FOR LATER
