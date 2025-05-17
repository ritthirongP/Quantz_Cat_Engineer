import pandas as pd
import ta
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import alpaca_trade_api as tradeapi
from transformers import pipeline
from newsapi import NewsApiClient

NUMBER_OF_STOCK = 5
TICKET_LIST = []
TRADE_HISTORY = {}

# Alpaca API Credentials
API_KEY = "PKIGLY4TU8ZBSN72M8IM"
SECRET_KEY = "WeC5byh8x7nQIZ8f7qUqZ32BCaH2XFssjGvVg7h4"
BASE_URL = "https://paper-api.alpaca.markets"  # Paper Trading
NEWS_API_KEY = "79ec7534e10546869be256d32082bd3e"

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# Initialize sentiment analysis
sentiment_pipeline = pipeline("sentiment-analysis")

def get_news_for_stock(symbol):
    query = f"{symbol} stock"
    articles = newsapi.get_everything(q=query, language='en', sort_by='publishedAt', page_size=5)
    return [article['title'] + ". " + article['description'] for article in articles['articles']]

def detect_market_trend(df, atr_window=14, ma_window=50, threshold=0.005):
    """
    Determines if the market is trending or sideways and adds the trend classification to the DataFrame.

    Parameters:
    df (DataFrame): Stock data with 'Close' prices.
    atr_window (int): Window size for ATR (volatility measurement).
    ma_window (int): Window size for Moving Average.
    threshold (float): Threshold for slope to determine trend (default 0.5%).

    Returns:
    DataFrame: Updated DataFrame with 'ATR', 'MA', 'MA_Slope', and 'Market_Trend' columns.
    """
    # Calculate ATR (Average True Range) to measure volatility
    df["ATR"] = ta.volatility.average_true_range(
        df["High"].squeeze(),
        df["Low"].squeeze(),
        df["Close"].squeeze(),
        window=atr_window,
    )
    
    # Calculate Moving Average
    df["MA"] = df["Close"].squeeze().rolling(window=ma_window).mean()

    # Calculate Slope of Moving Average (Last 10 Periods)
    df["MA_Slope"] = df["MA"].squeeze().diff(ma_window)

    # Define threshold for trend detection
    avg_atr = (
        df["ATR"].squeeze().rolling(window=10).mean()
    )  # Average ATR over last 10 periods
    avg_slope = (
        df["MA_Slope"].squeeze().rolling(window=10).mean()
    )  # Average MA slope over last 10 periods

    # Initialize Market Trend Column
    df["Market_Trend"] = "Sideways"

    # Trend Detection Logic
    df.loc[avg_slope > threshold * df["Close"].squeeze(), "Market_Trend"] = "Uptrend"
    df.loc[avg_slope < -threshold * df["Close"].squeeze(), "Market_Trend"] = "Downtrend"

    return df


def get_stock_data(symbol, period="5y", interval="1d"):
    df = yf.download(symbol, period=period, interval=interval)
    if df.empty:
        raise ValueError(f"No data returned for symbol: {symbol}")
    return df


# def calculate_rsi(series, period=14):
#     """Compute the Relative Strength Index (RSI)"""
#     delta = series.diff()
#     gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

#     rs = gain / loss
#     rsi = 100 - (100 / (1 + rs))
#     return rsi


def calculate_indicators(df):
    # Bollinger Bands
    df["BB_High"] = ta.volatility.bollinger_hband(df["Close"].squeeze())
    df["BB_Low"] = ta.volatility.bollinger_lband(df["Close"].squeeze())
    df["BB_Middle"] = ta.volatility.bollinger_mavg(df["Close"].squeeze())

    # MACD
    df["MACD"] = ta.trend.macd(df["Close"].squeeze())
    df["MACD_Signal"] = ta.trend.macd_signal(df["Close"].squeeze())
    df["MACD_Hist"] = ta.trend.macd_diff(df["Close"].squeeze())

    # 100-period Moving Average
    df["MA100"] = ta.trend.ema_indicator(df["Close"].squeeze(), window=100)

    # Stochastic Oscillator
    df["Stoch_%K"] = ta.momentum.stoch(
        df["High"].squeeze(), df["Low"].squeeze(), df["Close"].squeeze()
    )
    df["Stoch_%D"] = ta.momentum.stoch_signal(
        df["High"].squeeze(), df["Low"].squeeze(), df["Close"].squeeze()
    )

    # On-Balance Volume (OBV)
    df["OBV"] = ta.volume.on_balance_volume(
        df["Close"].squeeze(), df["Volume"].squeeze()
    )

    # **RSI (Relative Strength Index) - 14 period**
    df["RSI"] = ta.momentum.rsi(df["Close"].squeeze(), window=14)
    # **MFI - 14 period**
    df["MFI"] = ta.volume.money_flow_index(
        df["High"].squeeze(),
        df["Low"].squeeze(),
        df["Close"].squeeze(),
        df["Volume"].squeeze(),
        window=14,
    )
    df["Daily_Return"] = df["Close"].squeeze().pct_change()
    df["Cumulative_Returns"] = (1 + df["Daily_Return"]).cumprod() - 1

    return df

def calculate_position_size(capital, risk_per_trade, df):
    """Calculate position size dynamically based on ATR."""
    return (capital * risk_per_trade) / (df["ATR"] * 2)

def get_sentiment_score(text_list):
    sentiments = sentiment_pipeline(text_list)
    score = sum(1 if s["label"] == "POSITIVE" else -1 for s in sentiments)
    return score / len(sentiments)


def compute_risk_metrics(df):
    """Calculate Sharpe Ratio and Max Drawdown."""
    df["Log_Return"] = np.log(df["Portfolio"] / df["Portfolio"].shift(1))
    sharpe_ratio = df["Log_Return"].mean() / df["Log_Return"].std() * np.sqrt(252)
    rolling_max = df["Cumulative_Returns"].cummax()
    drawdown = df["Cumulative_Returns"] / rolling_max - 1
    max_drawdown = drawdown.min()
    return sharpe_ratio, max_drawdown

def backtest_strategy(
    df, initial_capital=1000, stop_loss_pct=50, max_drawdown_limit=15,risk_per_trade=0.05,take_profit = 50
):
    """
    Simulates a trading strategy based on buy/sell signals and tracks portfolio value.

    Parameters:
    df (DataFrame): The DataFrame containing stock prices and trading signals.
    initial_capital (float): The starting capital for the strategy.

    Returns:
    DataFrame: Updated DataFrame with 'Portfolio' column tracking portfolio value.
    """
    capital = initial_capital  # Starting Capital
    shares = 0  # Number of shares owned
    df["Portfolio"] = capital  # Initialize Portfolio Value
    stop_loss_price = None
    take_profit_price = None
    

    for index, row in df.iterrows():
        signal = row["Signal"][0]
        current_price = row["Close"][0]
        strength = row["Signal_Strength"][0]
        print(f"[{index}] Signal: {signal}, Strength: {strength}, Capital: {capital:.2f}, Shares: {shares}")

        df.at[index, "Portfolio"] = capital + (shares * current_price)

        # Compute ATR for dynamic stop-loss
        if "ATR" in df.columns:
            atr = row["ATR"][0]
            # stop_loss_level = current_price - (2.5 * atr)  # ATR-based stop-loss
            # take_profit_level = current_price + (3 * atr)
        # else:
        #     stop_loss_level = current_price * (
        #         1 - stop_loss_pct / 100
        #     )  # Fixed stop-loss
        #     take_profit_level = current_price * (
        #         1 + stop_loss_pct / 100
        #     )

        if shares > 0:

            if stop_loss_price and current_price <= stop_loss_price:
                capital += shares * current_price
                print(f"[STOP LOSS] {index} sell at {current_price:.2f}")
                shares = 0
                stop_loss_price = None
                take_profit_price = None
                df.at[index, "Signal"] = -1
                continue
            elif take_profit_price and current_price >= take_profit_price:
                capital += shares * current_price
                print(f"[TAKE PROFIT] {index} sell at {current_price:.2f}")
                shares = 0
                stop_loss_price = None
                take_profit_price = None
                df.at[index, "Signal"] = -1
                continue


        # if shares > 0 and ((stop_loss_price and row["Close"][0] <= stop_loss_price) or (take_profit_price and row["Close"][0] >= take_profit_price)):
        #     capital += shares * current_price # Sell all shares
        #     shares = 0  # Reset shares
        #     stop_loss_price = None  # Reset stop-loss level
        #     take_profit_price = None
        #     df.at[index, "Signal"] = -1


        if signal == 1 and shares == 0 and abs(strength) >= 0.3:  # Buy
            if atr == 0 or np.isnan(atr):
                continue
            price = current_price
            stop_loss_distance = 2.5 * atr
            risk_amount = capital * risk_per_trade  # e.g., 2%
            position_size = int(risk_amount / stop_loss_distance)
            total_cost = position_size * price

            if total_cost > capital:
                continue  # skip if not enough capital
            if position_size > 0 and total_cost <= capital:  
                shares = position_size
                capital -= total_cost

                stop_loss_price = price - stop_loss_distance
                take_profit_price = price + (3 * atr)
                print(f"[BUY] {index} {shares} shares at {price:.2f}")

        if signal == -1 and shares > 0:  # Sell
            capital += shares * current_price  # Sell all shares
            print(f"[SELL] {index} {shares} shares at {price:.2f}")
            shares = 0  # Reset share count
            stop_loss_price = None
            take_profit_price = None
        
        if signal == 1:
            print(f"[BUY] {index} at {current_price:.2f}")
        if signal == -1:
            print(f"[SELL] {index} at {current_price:.2f}")


        df.at[index, "Portfolio"] = capital + (
            shares * current_price
        )  # Update portfolio value

    return df


def calculate_trend_indicators(df):
    """
    Computes trend indicators: ADX, EMA crossover, and SuperTrend
    """
    # ✅ Compute ADX (Average Directional Index)
    adx = ta.trend.ADXIndicator(
        high=df["High"].squeeze(),
        low=df["Low"].squeeze(),
        close=df["Close"].squeeze(),
        window=14,
    )
    df["ADX"] = adx.adx()
    df["+DI"] = adx.adx_pos()
    df["-DI"] = adx.adx_neg()

    # ✅ Compute EMA Crossovers
    df["EMA_50"] = df["Close"].squeeze().ewm(span=50, adjust=False).mean()
    df["EMA_200"] = df["Close"].squeeze().ewm(span=200, adjust=False).mean()

    # ✅ SuperTrend Indicator (Custom Implementation)
    df["SuperTrend"],df = calculate_supertrend(df)

    return df


def calculate_supertrend(df, multiplier=2):
    """
    Compute SuperTrend indicator (Helps confirm strong trends).
    """
    # df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)

    df["hl2"] = (df["High"] + df["Low"]) / 2

    df["ATR"] = (
        df["ATR"].iloc[:, 0] if isinstance(df["ATR"], pd.DataFrame) else df["ATR"]
    )
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df["ATR"] = df["ATR"].astype(float)
    df["Close"] = df["Close"].astype(float)


    # Flatten the multi-index column names
    for col in df.columns:
        if isinstance(col, tuple):  # Customize this if needed
            flat_name = f"{col[0]}_raw"
            df[flat_name] = df[col]

    df["Stop_Loss"] = df["Close_raw"] - (df["ATR"] * 2)  # Adjust ATR multiplier as needed
    
    df["Take_Profit"] = df["Close_raw"] + (df["ATR"] * 3)  # Risk-to-reward ratio: 1:1.5
    df["UpperBand"] = df["hl2"].astype(float) + (df["ATR"].astype(float) * multiplier)

    df["LowerBand"] = df["hl2"].astype(float) - (df["ATR"].astype(float) * multiplier)
    # Initialize SuperTrend direction
    df["SuperTrend"] = 1  # 1 for bullish, -1 for bearish
    for i in range(1, len(df)):
        close_price = df.iloc[i]["Close_raw"].values
        upper_band = df.iloc[i - 1]["UpperBand"].values
        lower_band = df.iloc[i - 1]["LowerBand"].values
        # print(f'{close_price.values}.{upper_band.values}.{lower_band.values}')

        if close_price > upper_band:
            df.at[i, "SuperTrend"] = 1  # Uptrend
        elif close_price < lower_band:
            df.at[i, "SuperTrend"] = -1  # Downtrend
        else:
            df.at[i, "SuperTrend"] = df.iloc[i - 1]["SuperTrend"]  # Carry forward trend

    return df["SuperTrend"],df


def signal_calculate(df, symbol):

    df["Signal"] = 0  # Initialize Signal column
    df["MA100"] = df["MA100"].fillna(0)

    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df = calculate_trend_indicators(df)
    # Check if we can add more stocks to the TICKET_LIST
    # Uptrend condition: Close price above MA100
    # Uptrend condition: Close price above MA100

    # MACD Buy/Sell Signal

    # df["Buy_MACD"] = (df["MACD"] > df["MACD_Signal"]) & (
    #     (df["RSI"] > 30)
    #     | ((df["RSI"] >= 40) & (df["RSI"] <= 60) & (df["Market_Trend"] == "Uptrend"))
    # )
    df["Buy_MACD"] = (
        (df["MACD"] > df["MACD_Signal"])
        & (df["RSI"] > 40)
        # & ((df["MFI"] > 30) )
        # & (df["OBV"] >= df["OBV"].rolling(10).mean())
        & (df["ADX"] > 20)
        & (df["SuperTrend"] == 1)
        & (df["EMA_50"] > df["EMA_200"])
    )
    df["Sell_MACD"] = (
        (df["MACD"] < df["MACD_Signal"])
        & ((df["MFI"] < 70) )
        & (df["OBV"] < df["OBV"].rolling(10).mean())
        & (df["ADX"] > 25)
        & (df["SuperTrend"] == -1)
        & (df["EMA_50"] < df["EMA_200"])
    )
    # df["Sell_MACD"] = (df["MACD"] < df["MACD_Signal"]) & (
    #     (df["RSI"] < 70)
    #     | ((df["RSI"] >= 40) & (df["RSI"] <= 60) & (df["Market_Trend"] == "Downtrend"))
    # )

    df["Buy_Stoch"] = (
        (
            (df["Stoch_%K"] > df["Stoch_%D"])
            & (df["Stoch_%K"] > 40)
            & (df["Stoch_%D"] > 40)
        )
        & (df["MFI"] > 40)
        & (df["OBV"] >= df["OBV"].rolling(10).mean())
        & (df["ADX"] > 25)
        & (df["SuperTrend"] == 1)
        & (df["EMA_50"] > df["EMA_200"])
    )

    df["Sell_Stoch"] = (
    (df["Stoch_%K"] < df["Stoch_%D"]) |
    (df["MFI"] < 30)
    )


    df.loc[df["Buy_MACD"], "Signal"] = 1
    df["Signal"] = df["Buy_MACD"].astype(int) + df["Buy_Stoch"].astype(int) - df["Sell_MACD"].astype(int) - df["Sell_Stoch"].astype(int)

    # df.loc[
    #     df["Buy_Stoch"],
    #     "Signal",
    # ] = 1

    # # Apply sell signals across all rows where conditions are met
    # df.loc[df["Sell_MACD"], "Signal"] = -1
    # df.loc[
    #     df["Sell_Stoch"],
    #     "Signal",
    # ] = -1

    return df


# def calculate_stop_loss(df, risk_percentage=120):
#     stop_losses = []
#     for i in range(len(df)):
#         stop_loss = df["BB_Low"].iloc[i] + (
#             (1 - risk_percentage / 100) * (df["Close"].iloc[i] - df["BB_Low"].iloc[i])
#         )
#         stop_losses.append(stop_loss)
#     df["Stop_Loss"] = stop_losses
#     return df


def plot_indicators(df, symbol):
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Mark Buy and Sell Points
    buy_signals = df[df["Signal"] > 0]
    sell_signals = df[df["Signal"] < 0]

    axs[0].scatter(
        buy_signals.index,
        buy_signals["Close"],
        marker="^",
        color="green",
        label="Buy",
        alpha=1,
    )
    axs[0].scatter(
        sell_signals.index,
        sell_signals["Close"],
        marker="v",
        color="red",
        label="Sell",
        alpha=1,
    )
    # Price Chart with Bollinger Bands and MA100
    axs[0].plot(df.index, df["Close"], label="Close Price", color="blue")
    axs[0].plot(
        df.index, df["BB_High"], label="BB High", linestyle="dashed", color="red"
    )
    axs[0].plot(
        df.index, df["BB_Low"], label="BB Low", linestyle="dashed", color="green"
    )
    # axs[0].plot(df.index, df['MA100'], label='100-MA', color='purple')
    axs[0].plot(
        df.index, df["Stop_Loss"], label="Stop Loss", linestyle="dotted", color="black"
    )

    axs[0].set_title(f"{symbol} Price with Bollinger Bands and 100-MA")
    axs[0].legend()

    # MACD
    axs[1].plot(df.index, df["MACD"], label="MACD", color="black")
    axs[1].plot(df.index, df["MACD_Signal"], label="MACD Signal", color="orange")
    axs[1].bar(df.index, df["MACD_Hist"], label="MACD Hist", color="gray", alpha=0.5)
    axs[1].set_title("MACD Indicator")
    axs[1].legend()

    # Stochastic Oscillator
    axs[2].plot(df.index, df["Stoch_%K"], label="Stochastic %K", color="blue")
    axs[2].plot(df.index, df["Stoch_%D"], label="Stochastic %D", color="red")
    axs[2].set_title("Stochastic Oscillator")
    axs[2].legend()

    axs[3].plot(df["Portfolio"], label=f"{symbol} Portfolio")
    axs[3].set_title(f"{symbol} Portfolio Over Time")
    axs[3].set_xlabel("Date")
    axs[3].set_ylabel("Value ($)")
    axs[3].grid()
    axs[3].legend()


    plt.show()


def place_order(api, symbol, qty, side, stop_loss_pct=0.02, take_profit_pct=0.04):
    try:
        # Get the current market price
        market_price = api.get_last_trade(symbol).price

        # Calculate stop loss and take profit prices
        if side == "buy":
            stop_price = market_price * (1 - stop_loss_pct)  # Stop Loss Price
            take_profit_price = market_price * (
                1 + take_profit_pct
            )  # Take Profit Price
        elif side == "sell":
            stop_price = market_price * (1 + stop_loss_pct)  # Stop Loss Price
            take_profit_price = market_price * (
                1 - take_profit_pct
            )  # Take Profit Price

        # Submit OCO (One Cancels Other) order
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="gtc",  # Good 'til canceled
            order_class="bracket",
            stop_loss={
                "stop_price": str(round(stop_price, 2)),  # Round to 2 decimal places
                "limit_price": str(
                    round(stop_price * 0.99, 2)
                ),  # A bit below to ensure execution
            },
            take_profit={"limit_price": str(round(take_profit_price, 2))},
        )
        print(
            f"Successfully placed {side} order for {qty} shares of {symbol} with Stop Loss at {stop_price} and Take Profit at {take_profit_price}"
        )

    except Exception as e:
        print(f"Order placement failed: {e}")


def signal_strength(df,symbol,mode="back_test"):
    # Calculate the difference between fast and slow moving averages
    diff = df["SMA_10"] - df["SMA_50"]

    # Define thresholds for strong and weak signals
    strong_threshold = 0.01  # 1% difference between SMAs
    weak_threshold = 0.005  # 0.5% difference between SMAs

    # Initialize signal strength
    df["Signal_Strength"] = 0

    for i in range(1, len(df)):
        tech_strength = 0
        # Strong Buy Signal
        if diff.iloc[i] > strong_threshold:
            tech_strength = 1  # Strong Buy

        # Weak Buy Signal
        elif weak_threshold < diff.iloc[i] <= strong_threshold:
            tech_strength = 0.5  # Weak Buy

        # Strong Sell Signal
        elif diff.iloc[i] < -strong_threshold:
            tech_strength = -1  # Strong Sell

        # Weak Sell Signal
        elif -strong_threshold <= diff.iloc[i] < -weak_threshold:
            tech_strength = -0.5  # Weak Sell


        # Get latest sentiment
        if(mode != "back_test"):
            try:
                headlines = get_news_for_stock(symbol)
                sentiment_score = get_sentiment_score(headlines)  # Range: -1 to 1
            except Exception as e:
                print(f"Sentiment failed for {symbol}: {e}")
                sentiment_score = 0  # Neutral fallback
            adjusted_strength = tech_strength * (1 + 0.5 * sentiment_score)  # boost/reduce by up to ±50%
            adjusted_strength = np.clip(adjusted_strength, -1.0, 1.0)
            df.at[df.index[i], "Signal_Strength"] = round(adjusted_strength, 2)
        else:
            sentiment_score = 0
            print(df)
            adjusted_strength = np.clip(sentiment_score, -1.0, 1.0)
            
            df.loc[df.index[i], "Signal_Strength"] = tech_strength
        # Integrate sentiment
    return df



def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calculate the Sharpe Ratio.
    :param returns: Pandas Series of daily portfolio returns.
    :param risk_free_rate: Annual risk-free rate (default 2% for US Treasury Bonds).
    :return: Sharpe Ratio
    """
    excess_returns = returns - (risk_free_rate / 252)  # Convert to daily rate
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
    sharpe_ratio *= np.sqrt(252)  # Annualize
    return sharpe_ratio


def calculate_max_drawdown(portfolio_values):
    """
    Calculate the Maximum Drawdown (largest peak-to-trough drop).
    :param portfolio_values: Pandas Series of portfolio values over time.
    :return: Max Drawdown percentage
    """
    peak = portfolio_values.cummax()  # Track historical peaks
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = drawdown.min()  # Worst drop
    return max_drawdown * 100  # Convert to percentage


def main():
    # symbol = "AMZN"
    # api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")  #
    balance = []
    # symbols = ["TSM","AMZN",'USEG','OKLO','COIN']
    # symbols = ["TSM", "AMZN", "MSFT", "COIN", "V"]
    symbols = ["TSM"]
    # Get total account balance
    account = api.get_account()
    total_balance = float(account.equity)
    num_stocks = len(symbols)
    allocation_per_stock = total_balance / num_stocks  # Equal allocation
    max_drawdown = []
    sharpe_ratio = []

    print(f"Total Portfolio Balance: ${total_balance}")
    print(f"Allocation per stock: ${allocation_per_stock}")
    for symbol in symbols:
        print(f"\nProcessing symbol: {symbol}")
        try:
            df = get_stock_data(symbol)
            df = calculate_indicators(df)
            df = detect_market_trend(df)
            # df = calculate_stop_loss(df)
            df = signal_calculate(df, symbol)
            df = signal_strength(df,symbol)
            # df = signal_strength(df,symbol,mode="real")
            df = backtest_strategy(df)
            # print(df[['Close', 'BB_High', 'BB_Low', 'BB_Middle', 'MACD', 'MACD_Signal', 'MACD_Hist', 'MA100', 'Stoch_%K', 'Stoch_%D', 'OBV','Signal','Portfolio']].tail())
            # Place Order Based on the Last Signal
            latest_signal = df.iloc[-1]["Signal"]
            qty = 10  # You can adjust quantity based on strategy or capital
            # Set your risk management
            stop_loss_pct = 0.5  # 50% Stop Loss
            take_profit_pct = 0.8  # 80% Take Profit

            # if latest_signal == 1 and score > 0.3:
            #     place_order(api, symbol, qty, "buy", stop_loss_pct, take_profit_pct)
            # elif latest_signal == -1:
            #     place_order(api, symbol, qty, "sell", stop_loss_pct, take_profit_pct)
            plot_indicators(df, symbol)
            # Calculate daily returns
            df["Daily_Return"] = df["Portfolio"].pct_change()

            sharpe_ratio, max_drawdown = compute_risk_metrics(df)
            
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}, Max Drawdown: {max_drawdown:.2%}")
            balance.append({"symbol": symbol, "balance": df[["Portfolio"]].tail()})
            print(df[["Signal", "Signal_Strength", "Portfolio"]].tail(10))
        except Exception as e:
            print(f"Error processing symbol {symbol}: {e}")
    print("--" * 30)
    print(balance)
    print("--" * 30)
    print(max_drawdown)
    print("--" * 30)
    print(sharpe_ratio)


if __name__ == "__main__":
    main()
