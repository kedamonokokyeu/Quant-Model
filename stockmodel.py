import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer, roc_auc_score, f1_score
import joblib
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
import requests
from pandas_datareader import data as pdr

ALPHA_VANTAGE_API_KEY = "JI2DKJUN7HDY1I7Y"

# -------------------- MACRO FEATURES -------------------- #

def fetch_vix(start_date="2010-01-01", end_date=None):

    df = yf.download(
        "^VIX",
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,   # prevent silent behavior changes
        progress=False
    )

    if df.empty:
        raise ValueError("VIX download returned empty data.")

    # extract and rename
    vix = df["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
        vix = vix.rename('VIX')

    vix_change_20d = vix.pct_change(20, fill_method = None).rename("VIX_change20d")

    roll_mean = vix.rolling(20).mean()
    roll_std = vix.rolling(20).std()
    vix_z = ((vix - roll_mean) / (roll_std + 1e-9)).rename("VIX_z20")

    # combine into one dataframe
    vix_df = pd.concat(
        [vix, vix_change_20d, vix_z],
        axis=1
    )

    return vix_df.dropna()


def fetch_10Year_Treasury(start_date="2010-01-01", end_date=None):
    df = yf.download(
        "^TNX",
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        return pd.DataFrame()

    df = df[["Close"]].rename(columns={"Close": "10Y_Yield"})
    df["10Y_Yield"] /= 10.0  # TNX scaling

    df["10Y_Yield_Change"] = df["10Y_Yield"].diff()
    df["10Y_Yield_ROC"] = df["10Y_Yield"].pct_change()

    roll_mean = df["10Y_Yield"].rolling(60).mean()
    roll_std = df["10Y_Yield"].rolling(60).std()
    df["10Y_Yield_Z"] = (df["10Y_Yield"] - roll_mean) / (roll_std + 1e-9)

    return df.dropna()

def fetch_cpi(start_date="2010-01-01", end_date=None): # because yfinance does not provide CPI data, requires federal reserve's database
    cpi = pdr.DataReader("CPIAUCSL", "fred", start_date, end_date)

    cpi = cpi.rename(columns={"CPIAUCSL": "CPI"})
    cpi["CPI_YoY"] = cpi["CPI"].pct_change(periods = 12, fill_method = None)
    cpi["CPI_MoM"] = cpi["CPI"].pct_change(periods = 1, fill_method = None)
    cpi["CPI_YoY_Trend"] = cpi["CPI_YoY"].rolling(6).mean()

    roll_mean = cpi["CPI_YoY"].rolling(24).mean()
    roll_std = cpi["CPI_YoY"].rolling(24).std()
    cpi["CPI_YoY_Z"] = (cpi["CPI_YoY"] - roll_mean) / (roll_std + 1e-9)

    cpi = cpi.resample("D").ffill()
    return cpi.dropna()

# merge all our macrofeatures into one dataframe so we can later merge it into the main one
def build_macro_features(start, end):
    macro_frames = []

    try:
        macro_frames.append(fetch_vix(start, end))
    except Exception as e:
        print("VIX failed:", e)

    try:
        macro_frames.append(fetch_cpi(start, end))
    except Exception as e:
        print("CPI failed:", e)

    try:
        macro_frames.append(fetch_10Year_Treasury(start, end))
    except Exception as e:
        print("10Y failed:", e)

    if not macro_frames:
        return pd.DataFrame()

    macro_df = pd.concat(macro_frames, axis=1)
    macro_df = macro_df.sort_index().ffill()

    return macro_df


print("Enter the company ticker(s) you want to predict. (e.g. AAPL MSFT NVDA)")
ticker_symbol = input()
tickers = ticker_symbol.upper().split()
results = {}

GLOBAL_MACRO_DF = build_macro_features(
    start="2010-01-01",
    end=pd.Timestamp.today().strftime('%Y-%m-%d')
)

print("Enter ETF or Index Fund tickers for comparison (e.g. SPY QQQ ^GSPC IWM), or press Enter for none:")
etf_input = input().strip().upper()
comparisons = etf_input.split() if etf_input else None

for ticker_symbol in tickers:
    print(f"\n========== Analyzing {ticker_symbol} ==========")
    ticker = yf.Ticker(ticker_symbol)

    def OHLCV_data_analyzer(): # here we find what's important from OHLCV data 
        df = yf.download(
            ticker_symbol,
            start="2015-01-01",
            end = pd.Timestamp.today().strftime("%Y-%m-%d"),
            interval="1d",
            group_by="column"  # prevents multi-level columns
        )

        # fix for MultiIndex or multi-column issue 
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # sometimes yfinance still leaves sub-DataFrames, so we force select first column
        if isinstance(df["Close"], pd.DataFrame):
            df["Close"] = df["Close"].iloc[:, 0]
            df["High"]  = df["High"].iloc[:, 0]
            df["Low"]   = df["Low"].iloc[:, 0]
            df["Open"]  = df["Open"].iloc[:, 0]
            df["Volume"] = df["Volume"].iloc[:, 0]

        # the most important part of OHLC data, it is standardized and you want a positive % change
        df["Return_1m"] = df["Close"].pct_change(21).shift(1)   # 21 trading days ≈ 1 month
        df["Return_3m"] = df["Close"].pct_change(63).shift(1)   # 63 trading days ≈ 1 quarter
        df["Return_12m"] = df["Close"].pct_change(252).shift(1) # 252 trading days ≈ 1 year

        # how the averages move across long periods of time
        df["SMA_50"] = df["Close"].rolling(50).mean().shift(1)   # 50-day ≈ 2 months
        df["SMA_200"] = df["Close"].rolling(200).mean().shift(1) # 200-day ≈ 10 months
        df["SMA_Ratio"] = df["SMA_50"] / df["SMA_200"]

        df = df.drop(columns=["Return_3m", "Return_12m"], errors="ignore")

        # calculates sd from the mean of volatility for the past 63 and 252 days
        df["Volatility_3m"] = df["Close"].pct_change().rolling(63).std().shift(1)
        df["Volatility_12m"] = df["Close"].pct_change().rolling(252).std().shift(1)

        # rolling Sharpe ratio
        rolling_return = df["Close"].pct_change().rolling(63).mean().shift(1)
        rolling_volume = df["Close"].pct_change().rolling(63).std().shift(1)
        df["Sharpe_3m"] = rolling_return / (rolling_volume + 1e-6) # add the 1e-6 value because of rolling volume is super small then we compiler might think we divide by 0, so we throw that in instead

        # rolling Max Drawdown
        roll_max = df["Close"].rolling(63, min_periods=1).max()
        daily_drawdown = df["Close"] / roll_max - 1.0
        df["MaxDrawdown_3m"] = daily_drawdown.rolling(63, min_periods=1).min().shift(1)

        # average volume
        df["AvgVolume_3m"] = df["Volume"].rolling(63).mean().shift(1)   
        df["VolumeVolatility"] = df["AvgVolume_3m"] / df["Volatility_3m"]

        # adding lagged returns columns to df (just some extra info for the RandomForestClassifier)
        df["1day_lagged"] = df["Close"].pct_change(1)
        df["5day_lagged"] = df["Close"].pct_change(5)
        df["10day_lagged"] = df["Close"].pct_change(10)

        # STATISTICS --- rolling skewedness and kurtosis (check doc documentation if unsure)
        df["Skewness_3months"] = df["Close"].pct_change().rolling(63).skew()
        df["Kurtosis_3months"] = df["Close"].pct_change().rolling(63).kurt()

        # Moving Average Convergence Ratios
        df["Price_vs_SMA50"] = df["Close"].shift(1) / df["SMA_50"]
        df["Price_vs_SMA200"] = df["Close"].shift(1) / df["SMA_200"]

        # compute raw 3-month forward return
        # compute raw 3-month forward return
        df["Target_3m_raw"] = df["Close"].shift(-63) / df["Close"] - 1

        # yearly rolling quantile labeling
        df["Year"] = df.index.year
        
        def yearly_threshold(x): # let x be the sub-DataFrame for one year
            threshold = x["Target_3m_raw"].quantile(0.6)
            return (x["Target_3m_raw"] > threshold).astype(int) # turn into true or false, then into 0s and 1s
        
        df["Target_3m"] = df.groupby("Year", group_keys = False).apply(yearly_threshold) # group_keys to prevent a group label
        df = df.drop(columns=["Year"])

        df = df.ffill()   # forward fill missing fundamentals
        df = df.dropna(subset=["Target_3m"])   # only drop rows where target is missing
        # b/c we used rolling for some of these df so we just get NaN before the window is filled
        
        return df

    def get_alpha_fundamentals(ticker):
        base = "https://www.alphavantage.co/query"
        endpoints = {
        "income": f"{base}?function=INCOME_STATEMENT&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}",
        "balance": f"{base}?function=BALANCE_SHEET&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}",
        "cashflow": f"{base}?function=CASH_FLOW&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        }
        
        all_frames = [] # will collect the three dataframes from the endpoints above

        for name, url in endpoints.items():
            try:
                print(f"Fetching {name} statement for {ticker}...")
                response = requests.get(url, timeout = 15)
                response.raise_for_status()
                data = response.json()
                reports = data.get("quarterlyReports", []) # alphaVantage provides fundamental data in "quarterlyReports" column
                # empty list returned if there are no 'quarterlyReports

                if not reports:
                    print(f"No {name} data returned for {ticker}.")
                    continue
                
                df = pd.DataFrame(reports)
                df["fiscalDateEnding"] = pd.to_datetime(df["fiscalDateEnding"], errors = "coerce") # fill with NaN to avoid errors
                df = df.set_index("fiscalDateEnding").sort_index() # align the fiscalDateEnding columns with our dates
                df = df.apply(pd.to_numeric, errors = "coerce")

                # converting the scale-dependent values to ratios
                if "totalRevenue" in df.columns and "netIncome" in df.columns:
                    df["ProfitMargin"] = df["netIncome"] / df["totalRevenue"]

                if "totalLiabilities" in df.columns and "totalShareholderEquity" in df.columns:
                    df["DebtEquity"] = df["totalLiabilities"] / df["totalShareholderEquity"]

                if "totalCurrentAssets" in df.columns and "totalCurrentLiabilities" in df.columns:
                    df["CurrentRatio"] = df["totalCurrentAssets"] / df["totalCurrentLiabilities"]

                if "operatingCashflow" in df.columns and "totalRevenue" in df.columns:
                    df["FreeCashFlowMargin"] = df["operatingCashflow"] / df["totalRevenue"]

                if "netIncome" in df.columns and "totalAssets" in df.columns:
                    df["ROA"] = df["netIncome"] / df["totalAssets"]

                nan_threshold = 0.7
                drop_cols = df.columns[df.isna().mean() > nan_threshold]
                if len(drop_cols) > 0:
                    df = df.drop(columns = drop_cols)

                # log-transform large positive fundamentals to stabilize scale
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if df[col].max() > 1e6:  # only log-transform large-magnitude columns
                        df[col] = df[col].apply(lambda x: np.log1p(x) if x > 0 else x)


                all_frames.append(df)

            except Exception as error:
                print(f"Error fetching {name} for {ticker}: {error}")

        if not all_frames: # case checking
            print(f"No quarterly data for {ticker}")
            return pd.DataFrame()
        
        fundamentals = pd.concat(all_frames, axis = 1)
        fundamentals = fundamentals.loc[:, ~fundamentals.columns.duplicated()] # check debug documentation
        fundamentals = fundamentals.resample("D").ffill()
        fundamentals = fundamentals.shift(90, freq="D")  # move values *forward* in time (simulate delay)


        print(f"Retrieved {fundamentals.shape[1]} fundamental features for {ticker}.")
        return fundamentals            

    #RSI: measures the speed and change of price movements --> a high value suggest a stock is overbought, a low value suggest its oversold
    def rsi_calculation(data, window = 21): # usually 14, but we want long term so...
        close_difference = data['Close'].diff() # a Series of difference between each row and the previous one

        # .where() can perform scalar operations on Series, which is why it works here
        gain = close_difference.where(close_difference > 0, 0) # where this condition is true, keep the value, otherwise replace with 0
        loss = -close_difference.where(close_difference < 0, 0) # Same, except keep the value and turn it negative. 

        avg_gain = gain.ewm(com = window - 1, min_periods = window).mean() # can't just calculate average gain and loss simply, because recent values matter MORE
        avg_loss = loss.ewm(com = window - 1, min_periods = window).mean() # our com value comes from some convoluted mathematical proof

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
        signal_line = macd.ewm(span = signal_window, adjust = False).mean() # if MACD > signal, bullish. if MACD < signal, bearish
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
            etf_ticker = yf.download(ticker, start = start, end = end, progress = False)

            # --- ensure 'Close' is a single Series, even if multiple tickers or levels 
            if isinstance(etf_ticker.columns, pd.MultiIndex):
                etf_ticker.columns = etf_ticker.columns.get_level_values(0)
            if isinstance(etf_ticker["Close"], pd.DataFrame):
                etf = etf_ticker["Close"].iloc[:, 0]  # take first column if multi
            else:
                etf = etf_ticker["Close"]

            etf = etf.reindex(df.index).ffill()  # align with stock timeline
            etf_return = etf.pct_change(fill_method=None)  # avoid future deprecation warning

            relative_strength[f"RS_{ticker}"] = stock_return - etf_return
        return relative_strength
    
    def find_best_threshold(model, X, y, metric = f1_score):

        y_proba = model.predict_proba(X)[:, 1]

        thresholds = np.linspace(0.05, 0.95, 50)
        scores = []

        for t in thresholds:
            preds = (y_proba >= t).astype(int)
            scores.append(metric(y, preds))

        best_idx = np.argmax(scores)
        return thresholds[best_idx], scores[best_idx]

    def combine_features_dataset(ticker_symbol, comparisons = None): # just in case the user doesn't input ETFs or index funsd to compare the stock to

        df = OHLCV_data_analyzer()

        df = df.merge(
            GLOBAL_MACRO_DF.loc[df.index.min():df.index.max()],
            how = "left",
            left_index = True,
            right_index = True
        )

        fundamentals = get_alpha_fundamentals(ticker_symbol)
        if fundamentals is None or fundamentals.empty:
            print(f"No fundamentals for {ticker_symbol}, skipping.")
            fundamentals = pd.DataFrame(index=df.index)  # placeholder to avoid merge error

        print("Fundamentals preview:")
        print(fundamentals.head(3))

        df = df.merge(fundamentals, how = "left", left_index = True, right_index = True) # left_index and right_index true because the indexes represent dates from the ticker, and we do how = "left" b/c fundamentals data is only for quarters, we fill in later for the daily data
        df = df.dropna(subset=["Target_3m"])  # only drop if label is missing

        if not fundamentals.empty:
            df = df.loc[fundamentals.index.min():]

        df["RSI"] = rsi_calculation(df).shift(1)

        macd, signal, indicator = macd_calculation(df) # remember return line of macd_calculation(), so macd = macd, signal = signal_line, indicator = indicator
        df["MACD"] = macd.shift(1)
        df["MACD_Signal"] = signal.shift(1)
        df["MACD_Indicator"] = indicator.shift(1)

        upper, lower = bollinger_bands_calculation(df)
        df["Upper_Bollinger"] = upper.shift(1)
        df["Lower_Bollinger"] = lower.shift(1)

        percent_d, percent_k = stochastic_oscillator(df)
        df["Stochastic_%D"] = percent_d.shift(1)
        df["Stochastic_%K"] = percent_k.shift(1)

        df["OBV"] = pd.Series(calculate_OBV(df), index=df.index).shift(1)

        if comparisons != None:
            relative_strength_df = relative_strength_comparison(df, comparisons) # remember relative_strength_comparison returns a dataframe of these comparisons
            df = df.join(relative_strength_df) # add to our big dataframe

        # --- sanity check: how imbalanced are our labels? ---
        print("\n===== Label Distribution Check for", ticker_symbol, "=====")
        print(df["Target_3m"].value_counts())
        print(df["Target_3m"].value_counts(normalize=True))
        print("Ratio (1s / 0s):", df["Target_3m"].sum() / len(df["Target_3m"]))
        print("Number of samples:", len(df))
        print("Period covered:", df.index.min().date(), "to", df.index.max().date())

        # so quarterly values will be applied to every day in that quarter until updated instead of being NaN values
        return df

    # ---------------------- BUILDING THE MODEL AND TUNING ---------------------- #

    df = combine_features_dataset(ticker_symbol, comparisons)

    X = df.drop(columns=["Target_3m", "Target_3m_raw"], errors="ignore") # given all the data from the dataset, we want to predict Target_3m, 3 months into the future, but we drop it so the model cannot see the future
    y = df["Target_3m"] # what we want to predict

    dates = pd.to_datetime(X.index) # X.index refers to the rows of dates 

    # can't shuffle the data around for training and testing since finance is time-series
    split_date = "2021-01-01"
    X_train, X_test = X.loc[:split_date].iloc[:-1], X.loc[split_date:] # added iloc[:-1] all rows except the very last row to prevent data leakage, since both the testing and the training include the split date
    y_train, y_test = y.loc[:split_date].iloc[:-1], y.loc[split_date:] 

    train_dates = pd.to_datetime(X_train.index)
    recent_date = train_dates.max()
    age_days = (recent_date - train_dates).days # days converts this value into integers of how many days past

    # implement decaying logic for weighting
    decay = 0.99
    weights_train = pd.Series(decay ** (age_days / 30), index=X_train.index) # also added .iloc[:-1] to drop the last row so that the split date data isn't in both training and testing

    positives = y_train.sum()  # looks at all "1" values in y_train, and we'll apply weighting on this later
    negatives = len(y_train) - positives # these are our "0" values through binary classification, this is to see how many there are
    positive_scaling = negatives / max(positives, 1) # we divide by max of positives or 1 to avoid division by 0 -- this is just how XGBoost does weighting

# ----------- XGBoost FOR EACH TICKER (we'll look at this performance and the aggregate performance to see what we need to change) --------- #

    xgb = XGBClassifier( # REFER TO DOC
        objective = "binary:logistic", # written to indicate binary classification with a logistic loss
        eval_metric = "logloss", # a penalty based off of confidence and how wrong the model was
        tree_method = "hist", # creates histogram with bins, tests which values until tree splits off into child
        random_state = 42,
        n_jobs = -1,
        scale_pos_weight = positive_scaling
    )

    param_distributions = { # what we'll grid search for optimization
        "n_estimators": [300, 500, 700, 900],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth": [3, 4, 5, 6, 8],
        "min_child_weight": [1, 3, 5, 7],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0, 0.5, 1],
        "reg_alpha": [0, 0.1, 0.5],
        "reg_lambda": [0.5, 1, 2]
    }

    # time series cross validation (or tcsv)
    tscv = TimeSeriesSplit(n_splits = 5)
    scorer = make_scorer(f1_score)

    search = RandomizedSearchCV( # check documentation
        estimator = xgb,
        param_distributions = param_distributions,
        n_iter = 40, 
        scoring = scorer,
        cv = tscv,
        verbose = 1,
        n_jobs = -1,
        refit = True,
        random_state = 42
    )

    print("")
    print("Currently tuning XGBoost Hyperparameters . . .")
    search.fit(X_train, y_train, sample_weight = weights_train)
    
    # mark the best parameters for each ticker
    best_xgb = search.best_estimator_ # a new XGB model, but instead of dictionaries for its parameters, it just has the best ones  

    best_threshold, best_train_f1 = find_best_threshold(
        best_xgb,
        X_train,
        y_train,
        metric=f1_score
    )

    print(f"Optimized threshold (train only): {best_threshold:.3f}")
    print(f"Train F1 at optimal threshold: {best_train_f1:.3f}")


    print("")
    print("Best Parameters:", search.best_params_)  
    print("Best Cross-val F1:", search.best_score_)

    y_proba = best_xgb.predict_proba(X_test)[:, 1] # get the probaiblity of growth values
    y_pred = (y_proba >= best_threshold).astype(int) # convert to true or false array, then back to 0 and 1s

    print("F1:", f1_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))  # see how accurate we are in ranking positives
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Report:\n", classification_report(y_test, y_pred, digits=3))

    # save model for this ticker
    joblib.dump(best_xgb, f"{ticker_symbol}_xgb_model.pkl")

    results[ticker_symbol] = accuracy_score(y_test, y_pred)

print("\n===== Final Results Across Tickers =====")
for t, acc in results.items():
    print(f"{t}: Accuracy = {acc:.3f}")

# --------------------- AGGREGATE PLOTS -------------------- #
# collect all confusion matrices and correlations across tickers
print("\n===== Building Aggregate Plots =====")

all_cm = np.zeros((2,2)) # 2 by 2 confusion matrix
all_corrs = [] # series meant to hold the correlations

for ticker_symbol in tickers:
    ticker = yf.Ticker(ticker_symbol)
    df = combine_features_dataset(ticker_symbol, comparisons)

    X = df.drop(columns=["Target_3m"])
    y = df["Target_3m"]

    split_date = "2021-01-01"
    X_train, X_test = X.loc[:split_date].iloc[:-1], X.loc[split_date:]
    y_train, y_test = y.loc[:split_date].iloc[:-1], y.loc[split_date:]

    positives = y_train.sum()
    negatives = len(y_train) - max(positives, 1)
    positive_scaling = negatives / max(positives, 1)

    # ---------- AGGREGATE XGBoost FOR ALL TICKERS ---------- #
    xgb = XGBClassifier(
        objective = "binary:logistic", 
        eval_metric = "logloss", 
        tree_method = "hist", 
        random_state = 42,
        n_jobs = -1,
        scale_pos_weight = positive_scaling,
        n_estimators = best_xgb.get_params().get("n_estimators", 500), # get_params is built in function that just gets the parameters from the best_xgb model
        learning_rate = best_xgb.get_params().get("learning_rate", 0.05), # the second value in the parameters is just default if none is found, just chose them without much basis
        max_depth = best_xgb.get_params().get("max_depth", 5),
        min_child_weight = best_xgb.get_params().get("min_child_weight", 3),
        subsample = best_xgb.get_params().get("subsample", 0.8),
        colsample_bytree = best_xgb.get_params().get("colsample_bytree", 0.8),
        gamma = best_xgb.get_params().get("gamma", 0.5),
        reg_alpha = best_xgb.get_params().get("reg_alpha", 0.1),
        reg_lambda = best_xgb.get_params().get("reg_lambda", 1.0)
    )

    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)

    # accumulate confusion matrices for aggregate view
    all_cm += confusion_matrix(y_test, y_pred)

    df = df.drop(columns=["Target_3m_raw"], errors="ignore")
    corrs = X_train.join(y_train).corr()["Target_3m"].drop("Target_3m")

    all_corrs.append(corrs)

# ---- final confusion matrix ----
plt.figure(figsize=(5,4))
sns.heatmap(all_cm.astype(int), annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Growth", "Growth"],
            yticklabels=["No Growth", "Growth"])
plt.title("Aggregate Confusion Matrix (All Tickers)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ---- average correlations across tickers ----
mean_corrs = pd.concat(all_corrs, axis=1).mean(axis=1).sort_values(ascending=False)
plt.figure(figsize=(8,6))
sns.barplot(x=mean_corrs.values, y=mean_corrs.index, palette="coolwarm")
plt.title("Average Feature Correlation with Stock Growth (All Tickers)")
plt.xlabel("Correlation")
plt.ylabel("Feature")
plt.show()

