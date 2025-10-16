import pandas as pd
import requests

API_KEY = "JI2DKJUN7HDY1I7Y"
ticker = "AAPL"
url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={API_KEY}"

r = requests.get(url, timeout=15)
print("Status:", r.status_code)
data = r.json()
print("Sample fields:", list(data.items())[:10])