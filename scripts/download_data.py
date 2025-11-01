import os

import yfinance as yf


def fetch_gold_data(symbol="GC=F", start="2020-01-01", end="2025-01-01"):
    df = yf.download(symbol, start=start, end=end)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/gold_prices.csv")
    print("Gold price data saved to data/gold_prices.csv")


if __name__ == "__main__":
    fetch_gold_data()
