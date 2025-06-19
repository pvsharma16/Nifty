import yfinance as yf

tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
data = yf.download(tickers, start="2024-06-01", end="2024-06-10")

print(data.head())
