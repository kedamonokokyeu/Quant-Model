                                                 QUANT-Model

This repository contains a machine learning pipeline for stock trend forecasting, evaluating whether a stock's price will rise over the next 3 months using financial and time-series features derived from the Yahoo Finanace and Alpha Vantage APIs. It combines technical indicators, fundamental data, and benchmark comparisons using XGBoost.

If you want to see the mathematical derivations as to how the features were calculated:
https://docs.google.com/document/d/1Pd--ssGqXhYoqMyuCInbOMeH1jBITANIfHPYS2MOVrc/edit?usp=sharing

                                       ---------- 📊 FEATURES: ----------

Automated Data Retrieval
  - OHLCV Data via yfinance
  - Company fundamentals via Alpha Vantage API

Feature Engineering
  - Rolling Sharpe Ratios
  - Drawdowns, Volatility Metrics
  - Relative Strength Index
  - Moving Average Convergence Divergence
  - Bollinger Bands
  - Stochastic Oscillator
  - On Balance Volume
  - Relative Strength vs ETFs and Index Funds
  - Quarterly Financial Ratios (Debt/Equity, Profit Margin, etc.)

Machine Learning & Hyperparameter Tuning
  - XGBoost Classifier for binary prediction (0 for no growth, 1 for growth)
  - TimeSeriesSplit validation
  - Hyperparameter tuning through RandomizedSearchCV
  - Weighted training to place more emphasis on recent years (also due to limited fundamentals data)
    
Visualization
  - Confusiong Matrices and Correlation graphs

                                    --------- 🧠 HOW IT WORKS ---------

1. Enter in one or more stock tickers (e.g., AAPL MSFT)
2. Optionally, you can add ETF/index tickers for benchmarking (specifically ones that pertain to the tickers you want to analyze) (eg., SPY, QQQ, ^GSPC)
3. Daily OHLC data and Quarterly Fundamentals downloaded
4. Aligns and forward-fills missing fundamental data
5. Feature engineering, with 30+ predictive indicators
6. Model Training --> Data is split by date --> hyperparameters tuned via cross-validation --> exponential decay weighting for recency bias
7. Prints classificaiton metrics such as F1 score, ROC-AUC, Confusion Matrices
8. Generates plots showing aggregate model behavior across tickers

                                      -----  EXAMPLE OUTPUT:  -----
 
Fetching income statement for AAPL...
Fetching balance statement for AAPL...
Retrieved 87 fundamental features for AAPL.

Currently tuning XGBoost Hyperparameters . . .
Best Parameters: {'learning_rate': 0.03, 'max_depth': 5, 'subsample': 0.8, ...}
F1: 0.61
ROC AUC: 0.72

├── stockmodel.py        # Main pipeline for feature engineering + XGBoost model
├── test.py              # Quick Alpha Vantage API check
├── README.md            # Documentation (this file)
├── AAPL_xgb_model.pkl   # Example trained model (Apple)
├── MSFT_xgb_model.pkl   # Example trained model (Microsoft)
├── NVDA_xgb_model.pkl   # Example trained model (NVIDIA)

                               ---------- ⚙️ INSTALLATION AND SETUP ---------- 

Clone the repository: 
git clone https://github.com/<kedamonokokyeu>/Stock-Prediction-Model.git
cd Stock-Prediction-Model

Install: 
pip install -r requirements.txt
pip install numpy pandas scikit-learn xgboost yfinance seaborn matplotlib requests joblib

Open stockmodel.py and replace:
ALPHA_VANTAGE_API_KEY = "YOUR_API_KEY_HERE"
You can get your API key for free here: https://www.alphavantage.co/

Run the model!

Enter the tickers you'd like to predict, and enter the ETF and index funds tickers you'd like to compare to.

                                 ----------  😣 LIMITATIONS  ----------

Accuracy depends heavily on data quality and feature completeness
- With only a free API, much of the fundamentals data is missing

Market regime shifts (e.g., COVID, inflation cycles)
- These extenuating shifts go beyond the pure numbers that the model relies on

Limited Alpha Vantage free-tier data frequency
Does not predict exact prices — only growth direction.

Designed for educational and experimental use, not financial advice (unless you wanna go broke).

                                    ---------- TAKEAWAYS ----------

- Learned how to apply RandomForestClassifier, XGBoost, Hyperparameter tuning, Grid Searching
- Applied Pandas, visualization tools
- Coding as a medium for math; how built-in functions, parameters, etc. can be used to interpret mathematical formulas of the technical features being applied
- Data cleaning and analysis in order to tweak XGB parameter lists and optimize so Grid Search can test more effective combinations
- that this wasn't easy and I am now contemplating jumping off the clock tower

👨‍💻 Author

Tristan Pham

Contact: tristanpham@berkeley.edu






