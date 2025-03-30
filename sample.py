import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Download historical data for a stock
ticker = "AAPL"
df = yf.download(ticker, start="2020-01-01", end="2024-03-01")

# Calculate Moving Averages
df["SMA_10"] = df["Close"].rolling(window=10).mean()
df["SMA_50"] = df["Close"].rolling(window=50).mean()

# Plot Stock Price and Moving Averages
plt.figure(figsize=(12, 6))
plt.plot(df["Close"], label="Stock Price", color="black")
plt.plot(df["SMA_10"], label="SMA 10", color="blue")
plt.plot(df["SMA_50"], label="SMA 50", color="red")
plt.title(f"{ticker} Moving Average Crossover")
plt.legend()
plt.show()

# Define Buy/Sell Signals
df["Signal"] = 0
df.loc[df["SMA_10"] > df["SMA_50"], "Signal"] = 1  # Buy Signal
df.loc[df["SMA_10"] < df["SMA_50"], "Signal"] = -1  # Sell Signal

# Plot Buy/Sell signals
plt.figure(figsize=(12, 6))
plt.plot(df["Close"], label="Stock Price", color="black")
plt.plot(df["SMA_10"], label="SMA 10", color="blue")
plt.plot(df["SMA_50"], label="SMA 50", color="red")

# Mark Buy and Sell Points
buy_signals = df[df["Signal"] == 1]
sell_signals = df[df["Signal"] == -1]

plt.scatter(
    buy_signals.index,
    buy_signals["Close"],
    marker="^",
    color="green",
    label="Buy",
    alpha=1,
)
plt.scatter(
    sell_signals.index,
    sell_signals["Close"],
    marker="v",
    color="red",
    label="Sell",
    alpha=1,
)

plt.title(f"{ticker} Buy/Sell Signals")
plt.legend()
plt.show()

capital = 1000  # Starting Capital
shares = 0  # Number of shares owned

df["Portfolio"] = capital  # Track Portfolio Value

for i in range(1, len(df)):
    if df.iloc[i]["Signal"] == 1:  # Buy
        shares = capital // df.iloc[i]["Close"]
        capital -= shares * df.iloc[i]["Close"]

    elif df.iloc[i]["Signal"] == -1 and shares > 0:  # Sell
        capital += shares * df.iloc[i]["Close"]
        shares = 0

    df.at[df.index[i], "Portfolio"] = capital + (shares * df.iloc[i]["Close"])

# Plot Portfolio Growth
plt.figure(figsize=(12, 6))
plt.plot(df["Portfolio"], label="Portfolio Value", color="green")
plt.title(f"{ticker} Portfolio Performance")
plt.legend()
plt.show()
