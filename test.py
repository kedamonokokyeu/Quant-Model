import numpy as np
import pandas as pd
import yfinance as yf

# Pick a ticker to inspect
ticker = yf.Ticker("AAPL")  # <-- quotes are required

# --- Quarterly Financials (Income Statement) ---
fin_q = ticker.quarterly_financials
print("=== Quarterly financials (income statement) ===")
print(fin_q.head())  # first few line items (rows) across recent quarters (columns)
print("\nFinancials line items:", list(fin_q.index))
print("Financials quarter columns:", list(fin_q.columns))

# --- Quarterly Balance Sheet ---
bs_q = ticker.quarterly_balance_sheet
print("\n=== Quarterly balance sheet ===")
print(bs_q.head())
print("\nBalance sheet line items:", list(bs_q.index))
print("Balance sheet quarter columns:", list(bs_q.columns))

# --- Quarterly Cash Flow ---
cf_q = ticker.quarterly_cashflow
print("\n=== Quarterly cashflow ===")
print(cf_q.head())
print("\nCashflow line items:", list(cf_q.index))
print("Cashflow quarter columns:", list(cf_q.columns))

# --- Info dict keys (valuation, shares, etc.) ---
info = ticker.info
print("\n=== info() keys (metadata/valuation) ===")
print(list(info.keys()))
