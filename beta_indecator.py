#  red zone 

import sys
import types
try:
    import six
    import urllib3
    urllib3_six = types.SimpleNamespace(moves=six.moves)
    sys.modules['urllib3.packages.six.moves'] = six.moves
    sys.modules['urllib3.packages.six'] = urllib3_six
except ImportError:
    pass
# # Fake the deprecated path for backward compatibility
# urllib3_six = types.SimpleNamespace(moves=six.moves)
# sys.modules['urllib3.packages.six.moves'] = six.moves
# sys.modules['urllib3.packages.six'] = urllib3_six

# Momentum Indicators
# Small/mid-cap stocks, growth stocks, news-driven stocks, high relative volume, new highs/lows—assets that move sharply and quickly.

import pandas as pd 
import ta
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import alpaca_trade_api as tradeapi
from transformers import pipeline
from newsapi import NewsApiClient
# import signal_strategy_backtest_bt as bt

NUMBER_OF_STOCK = 5
TICKET_LIST = []
TRADE_HISTORY = {}
TF_LOOKBACK = 20 # Timeframe for lookback in fibonacci levels
# Alpaca API Credentials
API_KEY = "PKIGLY4TU8ZBSN72M8IM"
SECRET_KEY = "WeC5byh8x7nQIZ8f7qUqZ32BCaH2XFssjGvVg7h4"
BASE_URL = "https://paper-api.alpaca.markets"  # Paper Trading
NEWS_API_KEY = "79ec7534e10546869be256d32082bd3e"

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# Initialize sentiment analysis
sentiment_pipeline = pipeline("sentiment-analysis")

def get_news_for_stock(symbol, date):
    from_dt = date.strftime("%Y-%m-%dT00:00:00")
    to_dt = date.strftime("%Y-%m-%dT23:59:59")
    articles = newsapi.get_everything(
        q=f"{symbol} stock",
        language='en',
        from_param=from_dt,
        to=to_dt,
        sort_by='relevancy',
        page_size=5,
    )
    return [a["title"] + ". " + (a["description"] or "") for a in articles["articles"]]

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

# def calculate_fibonacci_levels(df,period=60):
#     """Calculate Fibonacci retracement levels based on the high and low of the last 100 days."""
#     if len(df) < 100:
#         raise ValueError("DataFrame must contain at least 100 rows for Fibonacci calculation.")

#     high = df["High"].rolling(window=period).max()
#     low = df["Low"].rolling(window=period).min()
#     diff = high - low

#     # Fibonacci levels
#     levels = {
#         "fibo_0": 0.0,
#         "fibo_236": 0.236,
#         "fibo_382": 0.382,
#         "fibo_5": 0.5 ,
#         "fibo_618": 0.618 ,
#         "fibo_786": 0.786,
#         "fibo_1": 1.0
#     }
#     df_new = df.copy()  # Avoid modifying the original DataFrame
#     for col,level in levels.items():
#         df[col] = low + (diff * level)
#     return df_new

def adaptive_exit_levels(entry_price, bb_width, bb_mid, bb_high, bb_low, 
                        fibo_levels, width_threshold=0.03):
    """
    Selects stop-loss and take-profit dynamically using Fibonacci & BB.
    
    Args:
        entry_price: float, entry price of the trade
        bb_width: float, BB high - BB low, relative to entry price
        bb_mid, bb_high, bb_low: floats, Bollinger band levels
        fibo_levels: dict, e.g. {'fibo_382': val, 'fibo_500': val, 'fibo_618': val, ...}
        width_threshold: float, threshold (as a proportion of price) to classify BB width
    Returns:
        stop_loss: float
        take_profit: float
    """
    # Classify band width as "narrow" or "wide"
    rel_width = bb_width / entry_price
    
    if rel_width < width_threshold:  # Narrow BB
        take_profit = min(fibo_levels['fibo_382'], bb_mid, bb_high)
        stop_loss = max(fibo_levels['fibo_500'], bb_low)
    else:  # Wide BB
        take_profit = min(fibo_levels['fibo_618'], bb_high)
        stop_loss = max(fibo_levels['fibo_786'], bb_low)
    
    # Ensure stop < entry < profit (for long trades)
    take_profit = max(take_profit, entry_price * 1.001)
    stop_loss = min(stop_loss, entry_price * 0.999)
    return stop_loss, take_profit

def calculate_indicators(df):
    # Ensure we're working with Series, not DataFrames
    close = df["Close"] if isinstance(df["Close"], pd.Series) else df["Close"].iloc[:, 0]
    high = df["High"] if isinstance(df["High"], pd.Series) else df["High"].iloc[:, 0]
    low = df["Low"] if isinstance(df["Low"], pd.Series) else df["Low"].iloc[:, 0]
    volume = df["Volume"] if isinstance(df["Volume"], pd.Series) else df["Volume"].iloc[:, 0]

    # 100-period Moving Average
    df["MA100"] = ta.trend.ema_indicator(close, window=100)

    # Stochastic Oscillator
    df["Stoch_%K"] = ta.momentum.stoch(high, low, close)
    df["Stoch_%D"] = ta.momentum.stoch_signal(high, low, close)

    df['RSI'] = ta.momentum.RSIIndicator(close).rsi()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close)
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()
    df["BB_Middle"] = bb.bollinger_mavg()

    # MACD
    macd = ta.trend.MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    # On-Balance Volume (OBV)
    df["OBV"] = ta.volume.on_balance_volume(close, volume)
    df["MFI"] = ta.volume.money_flow_index(high, low, close, volume)
    df["MFI"] = pd.to_numeric(df["MFI"], errors="coerce")
    
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
    trades = []  # ✅ New: Store trade history for analysis
    

    for index, row in df.iterrows():
        # signal = row["Signal"][0]
        # current_price = row["Close"][0]
        # strength = row["Signal_Strength"][0]
        signal = row.get("Signal", 0)
        current_price = row.get("Close", 0)
        strength = row.get("Signal_Strength", 0)
        ema_50  = row.get("EMA_50_raw", 0)
        ema_200 = row.get("EMA_200_raw", 0)
        print(f"[{index}] Signal: {signal}, Strength: {strength}, Capital: {capital:.2f}, Shares: {shares}")

        bb_width = row.get("BB_Width", 0)
        if isinstance(bb_width, (pd.Series, np.ndarray)):
            bb_width = bb_width.item()
        bb_mid = row.get("BB_Mid", 0)
        if isinstance(bb_mid, (pd.Series, np.ndarray)):
            bb_mid = bb_mid.item()
        bb_high = row.get("BB_High", 0)
        if isinstance(bb_high, (pd.Series, np.ndarray)):
            bb_high = bb_high.item()
        bb_low = row.get("BB_Low", 0)
        if isinstance(bb_low, (pd.Series, np.ndarray)):
            bb_low = bb_low.item()
        # Calculate recent swing high/low as needed
        fibo_high = row.get("swing_high", 0)
        if isinstance(fibo_high, (pd.Series, np.ndarray)):
            fibo_high = fibo_high.item()
        fibo_low = row.get("swing_low", 0) 
        if isinstance(fibo_low, (pd.Series, np.ndarray)):
            fibo_low = fibo_low.item()

        # Calculate Fibonacci retracements
        fibo_levels = {
            'fibo_236': fibo_high - 0.236 * (fibo_high - fibo_low),
            'fibo_382': fibo_high - 0.382 * (fibo_high - fibo_low),
            'fibo_500': fibo_high - 0.5 * (fibo_high - fibo_low),
            'fibo_618': fibo_high - 0.618 * (fibo_high - fibo_low),
            'fibo_786': fibo_high - 0.786 * (fibo_high - fibo_low),
        }
        stop_loss, take_profit = adaptive_exit_levels(
            current_price, bb_width, bb_mid, bb_high, bb_low, fibo_levels
        )
        if isinstance(signal, (pd.Series, np.ndarray)):
            signal = signal.item()
        if isinstance(strength, (pd.Series, np.ndarray)):
            strength = strength.item()
        if isinstance(current_price, (pd.Series, np.ndarray)):
            current_price = current_price.item()
        if isinstance(ema_50, (pd.Series, np.ndarray)):
            ema_50 = ema_50.item()
        if isinstance(ema_200, (pd.Series, np.ndarray)):
            ema_200 = ema_200.item()
        
        df.at[index, "Portfolio"] = capital + (shares * current_price)
        
        # Compute ATR for dynamic stop-loss
        atr = None
        if "ATR" in df.columns:
            atr = row.get("ATR", 0)
            if isinstance(atr, (pd.Series, np.ndarray)):
                atr = atr.item()
        if pd.isna(atr) or atr == 0:
            continue 
        print('a')
        if shares > 0:
            if stop_loss_price and current_price <= stop_loss_price:
                capital += shares * current_price
                print(f"[STOP LOSS] {index} sell at {current_price:.2f}")
                shares = 0
                stop_loss_price = None
                take_profit_price = None
                df.at[index, "Signal"] = -1
                trades.append({"date": index, "type": "STOP_LOSS", "price": current_price})
                continue
            elif take_profit_price and current_price >= take_profit_price:
                capital += shares * current_price
                print(f"[TAKE PROFIT] {index} sell at {current_price:.2f}")
                shares = 0
                stop_loss_price = None
                take_profit_price = None
                df.at[index, "Signal"] = -1
                trades.append({"date": index, "type": "TAKE_PROFIT", "price": current_price})
                continue
        if signal == 1 and shares == 0 and abs(strength) >= 0.3:  # Buy
            if pd.isna(atr) or atr == 0:
                continue
            price = current_price
            stop_loss_distance = 2.5 * atr
            risk_amount = capital * risk_per_trade  # e.g., 2%
            scaled_risk = risk_amount * abs(strength)
            position_size = int(scaled_risk / stop_loss_distance)
            total_cost = position_size * price

            if total_cost > capital:
                continue  # skip if not enough capital
            if position_size > 0 and total_cost <= capital:  
                shares = position_size
                capital -= total_cost

                stop_loss_price = stop_loss
                take_profit_price = take_profit
                print(f"[BUY] {index} {shares} shares at {price:.2f}")
                trades.append({"date": index, "type": "BUY", "price": price, "shares": shares, "capital": capital})
                # ✅ Trend exit if long and trend flips
        elif shares > 0 and ema_50 < ema_200:
            capital += shares * current_price
            trades.append({"date": index, "type": "TREND_EXIT", "price": current_price, "shares": shares})
            shares = 0
            stop_loss_price = None
            take_profit_price = None
        if signal == -1 and shares > 0:  # Sell
            capital += shares * current_price  # Sell all shares
            print(f"[SELL] {index} {shares} shares at {price:.2f}")
            trades.append({"date": index, "type": "SELL", "price": current_price, "shares": shares})
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
    pd.DataFrame(trades).to_csv("trades_log.csv", index=False)
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
    try:
        for i in range(1, len(df)):
            close_price = df.iloc[i]["Close_raw"]
            upper_band = df.iloc[i - 1]["UpperBand"]
            lower_band = df.iloc[i - 1]["LowerBand"]
            # print(f'{close_price.values}.{upper_band.values}.{lower_band.values}')

            if close_price > upper_band:
                df.at[i, "SuperTrend"] = 1  # Uptrend
            elif close_price < lower_band:
                df.at[i, "SuperTrend"] = -1  # Downtrend
            else:
                df.at[i, "SuperTrend"] = df.iloc[i - 1]["SuperTrend"]  # Carry forward trend
    except Exception as e:
        error_msg = str(e)
        print(error_msg)

    return df["SuperTrend"],df


def signal_calculate(df, adx_thresh=25):
    """Enhanced signal generation with multiple confirmations"""
    df['Signal'] = 0
    df['Signal_Strength'] = 0

    # Calculate signal conditions
    print('Note: RSI is a momentum indicator, it can be used to identify overbought or oversold conditions.')
    df['RSI'] = pd.to_numeric(df['RSI'], errors='coerce')
    df['Prev_RSI'] = df['RSI'].shift(1).fillna(df['RSI'])
    momentum_buy = (df['RSI'] > 50) & (df['Prev_RSI']<= 50)
    momentum_sell = (df['RSI'] < 50) & (df['Prev_RSI'] >= 50)
    
    df['Prev_Close'] = df['Close'].shift(1)
    df['prev_BB_Low'] = df['BB_Low'].shift(1)
    print('<>'*30)
    reversion_buy = (
        (df['Close'] < df['BB_Low']) & 
        (df['Prev_Close'] >= df['prev_BB_Low']) &
        (df['MFI'] < 30)  # Oversold on MFI
    )
    print('<>'*30)
    df['Prev_BB_High'] = df['BB_High'].shift(1).fillna(df['BB_High'])
    reversion_sell = (
        (df['Close'] > df['BB_High']) & 
        (df['Prev_Close'] <= df['Prev_BB_High']) &
        (df['MFI'] > 70)  # Overbought on MFI
    )
    
    # Strong trend confirmation
    strong_trend = df['ADX'] > adx_thresh
    
    # Volume confirmation
    volume_confirm = df['OBV'] > df['OBV'].rolling(20).mean()
    
    # Generate signals with strength
    df.loc[momentum_buy & volume_confirm, 'Signal'] = 1
    df.loc[reversion_buy & volume_confirm, 'Signal'] = 1
    df.loc[momentum_sell | reversion_sell, 'Signal'] = -1
    
    # Calculate signal strength (0 to 1)
    df['Signal_Strength'] = df['Signal'].abs() * (
        (df['ADX'] / 100) * 0.4 +  # Trend strength
        (abs(50 - df['RSI']) / 50) * 0.3 +  # RSI extremity
        (df['BB_Width'] / df['BB_Width'].rolling(20).mean()) * 0.3  # Volatility regime
    )
    df['swing_high'] = np.nan
    df['swing_low'] = np.nan
    for i in range(TF_LOOKBACK, len(df)):
        swing_high = df['High'].iloc[i-TF_LOOKBACK:i].max()
        swing_low = df['Low'].iloc[i-TF_LOOKBACK:i].min()
        df.at[df.index[i], 'swing_high'] = swing_high
        df.at[df.index[i], 'swing_low'] = swing_low
        # fibo = get_fibo_levels(swing_low, swing_high)

        # # Example: Use 23.6% retracement for stop loss, 61.8% for take profit
        # if df['Signal'].iloc[i] == 1:  # Buy
        #     df.at[df.index[i], 'Fibo_SL'] = fibo['23.6']
        #     df.at[df.index[i], 'Fibo_TP'] = fibo['61.8']
        # elif df['Signal'].iloc[i] == -1:  # Sell/Short (reverse levels)
        #     df.at[df.index[i], 'Fibo_SL'] = fibo['61.8']
        #     df.at[df.index[i], 'Fibo_TP'] = fibo['23.6']

    
    return df


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
        if mode != "back_test":
            try:
                headlines = get_news_for_stock(symbol, df.index[i])  # Pass date for contextual news
                sentiment_score = get_sentiment_score(headlines)
            except Exception as e:
                print(f"Sentiment failed for {symbol}: {e}")
                sentiment_score = 0

            adjusted_strength = tech_strength * (1 + 0.5 * sentiment_score)
            adjusted_strength = np.clip(adjusted_strength, -1.0, 1.0)
            df.at[df.index[i], "Signal_Strength"] = round(adjusted_strength, 2)
        else:
            df.at[df.index[i], "Signal_Strength"] = tech_strength
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
    # symbols = ["SHOP","SMCI",'CELH','TOAST','QBTS','PLUG']
    symbols = ["PLUG"]
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
            
            df = df.fillna(0)  # Fill missing values
            # df = calculate_stop_loss(df)
            df = signal_calculate(df, symbol)
            print("<>" * 30)
            df = signal_strength(df,symbol)

            # df = signal_strength(df,symbol,mode="real")
            df = backtest_strategy(df)
            # print(df[['Close', 'BB_High', 'BB_Low', 'BB_Middle', 'MACD', 'MACD_Signal', 'MACD_Hist', 'MA100', 'Stoch_%K', 'Stoch_%D', 'OBV','Signal','Portfolio']].tail())
            # Place Order Based on the Last Signal
            # latest_signal = df.iloc[-1]["Signal"]
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
            df_export = df[["Open", "High", "Low", "Close", "Volume", "Signal_Strength", "SuperTrend", "Market_Trend","Signal",'Date']].copy()
            df_export.reset_index(inplace=True)  # bring Date back as a column
            df_export.to_csv(f"stock_{symbol}.csv", index=False)
        except Exception as e:
            print(f"Error processing symbol {symbol}: {e}")
    print("--" * 30)
    print(balance)
    print("--" * 30)
    print(max_drawdown)
    print("--")

if __name__ == "__main__":
    main()