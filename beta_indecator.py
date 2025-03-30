import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import ta
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import alpaca_trade_api as tradeapi


NUMBER_OF_STOCK = 5
TICKET_LIST = []
TRADE_HISTORY = {}

# Alpaca API Credentials
API_KEY = "PKIGLY4TU8ZBSN72M8IM"
SECRET_KEY = "WeC5byh8x7nQIZ8f7qUqZ32BCaH2XFssjGvVg7h4"
BASE_URL = "https://paper-api.alpaca.markets"  # Paper Trading

class MLSignalGenerator:
    def __init__(self, lookback_periods=30, future_periods=5):
        self.lookback_periods = lookback_periods
        self.future_periods = future_periods
        self.model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight='balanced'
        )
        self.scaler = StandardScaler()

    
    def prepare_features(self, df):
        """
        Prepare advanced technical features for ML model
        """
        features = pd.DataFrame()
        
        # Momentum Indicators
        features['rsi'] = df['RSI']
        features['macd'] = df['MACD']
        features['stoch_k'] = df['Stoch_%K']
        
        # Trend Indicators
        features['adx'] = df['ADX']
        features['ema_50'] = df['EMA_50']
        features['ema_200'] = df['EMA_200']
        
        # Volatility
        features['atr'] = df['ATR']
        features['bb_width'] = df['BB_High'] - df['BB_Low']
        
        # Volume
        features['obv'] = df['OBV']
        
        return features
    
    def create_labels(self, prices):
        """
        Create binary classification labels
        1: Strong Buy
        0: Hold
        -1: Strong Sell
        """
        future_returns = prices.pct_change(self.future_periods).shift(-self.future_periods)
        labels = np.where(
            future_returns > 0.02, 1,  # Strong Buy if >2% gain
            np.where(future_returns < -0.02, -1, 0)  # Strong Sell if <2% loss
        )
        return labels
    
    def train_model(self, df):
        features = self.prepare_features(df)
        labels = self.create_labels(df['Close'])
        
        # Remove NaN values
        valid_indices = ~np.isnan(features).any(axis=1) & ~np.isnan(labels)
        features = features[valid_indices]
        labels = labels[valid_indices]
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Print model performance
        print(f"Model Accuracy: {self.model.score(X_test, y_test):.2%}")
    
    def predict_signal(self, df):
        """
        Generate ML-enhanced trading signals
        """
        features = self.prepare_features(df)
        features_scaled = self.scaler.transform(features)
        
        # Predict signal probabilities
        probabilities = self.model.predict_proba(features_scaled)
        
        # Combine traditional and ML signals
        ml_signals = self.model.predict(features_scaled)
        
        return ml_signals, probabilities



# Example usage in main strategy
# ml_signal_generator = MLSignalGenerator()
# df = enhanced_signal_calculate(df, ml_signal_generator)
# Debug function to help identify shape issues
def debug_shape_mismatch(df, ml_signal_generator):
    print("DataFrame Shape:", df.shape)
    
    features = ml_signal_generator.prepare_features(df)
    print("Features Shape:", features.shape)
    
    labels = ml_signal_generator.create_labels(df['Close'])
    print("Labels Shape:", labels.shape)
    
    return features, labels

# Integration with existing strategy
def enhanced_signal_calculate(df, ml_signal_generator):
        """
        Enhanced signal calculation with adaptive stop-loss
        
        Parameters:
        - df: Input DataFrame
        - ml_signal_generator: Machine Learning Signal Generator
        - risk_percentage: Base risk percentage for stop-loss
        
        Returns:
        - DataFrame with enhanced signals and stop-loss
        """
        # Train ML model
        ml_signal_generator.train_model(df)
        
        # Get ML signals
        ml_signals, probabilities = ml_signal_generator.predict_signal(df)
        
        # Add ML signals to DataFrame
        df['ML_Signal'] = ml_signals
        df['Signal_Probability'] = probabilities[:, 1]
        
        # Enhanced signal generation logic
        df['Final_Signal'] = np.select(
            [
                (df['Signal'] == 1) & (df['ML_Signal'] == 1),  # Strong Buy
                (df['Signal'] == -1) & (df['ML_Signal'] == -1),  # Strong Sell
                True  # Default case
            ],
            [1, -1, 0]
        )
        
        # Calculate Adaptive Stop-Loss
        df = calculate_adaptive_stop_loss(df, risk_percentage=2.0)
        
        # Stop-Loss Trigger Logic
        df['Stop_Loss_Triggered'] = np.where(
            (df['Close'] <= df['Adaptive_Stop_Loss']) & (df['Final_Signal'] == 1),
            -1,  # Sell signal if stop-loss triggered on a long position
            np.where(
                (df['Close'] >= df['Adaptive_Stop_Loss']) & (df['Final_Signal'] == -1),
                1,  # Buy signal if stop-loss triggered on a short position
                0  # No action
            )
        )
        
        # Final Trading Signal (Incorporating Stop-Loss)
        df['Trading_Signal'] = np.select(
            [
                df['Final_Signal'] == 1,  # Buy Signal
                df['Final_Signal'] == -1,  # Sell Signal
                df['Stop_Loss_Triggered'] == -1,  # Stop-Loss Sell Trigger
                df['Stop_Loss_Triggered'] == 1,  # Stop-Loss Buy Trigger
                True  # Hold
            ],
            [1, -1, -1, 1, 0]
        )
        
        return df

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
    return df


# def calculate_rsi(series, period=14):
#     """Compute the Relative Strength Index (RSI)"""
#     delta = series.diff()
#     gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

#     rs = gain / loss
#     rsi = 100 - (100 / (1 + rs))
#     return rsi

def calculate_adaptive_stop_loss(df, risk_method='atr', risk_percentage=2.0):
        """
        Calculate adaptive stop-loss using multiple methods
        
        Parameters:
        - df: DataFrame with price and indicator data
        - risk_method: Method for stop-loss calculation
        - risk_percentage: Base risk percentage
        
        Returns:
        - DataFrame with stop-loss levels
        """
        if risk_method == 'atr':
            # ATR-based stop-loss (Dynamic Volatility-Adjusted)
            df['Stop_Loss_ATR'] = df['Close'] - (df['ATR'] * 2.5)
        
        elif risk_method == 'bollinger':
            # Bollinger Bands-based stop-loss
            df['Stop_Loss_Bollinger'] = df['BB_Low'] - (df['Close'] * risk_percentage / 100)
        
        elif risk_method == 'percentile':
            # Rolling percentile-based stop-loss
            df['Stop_Loss_Percentile'] = df['Close'].rolling(window=20).quantile(0.05)
        
        # Combine multiple stop-loss methods
        df['Adaptive_Stop_Loss'] = np.select(
            [
                df['ATR'] > 0,  # Prefer ATR when available
                df['BB_Low'] > 0,  # Fallback to Bollinger
                True  # Final fallback
            ],
            [
                df['Stop_Loss_ATR'],
                df['Stop_Loss_Bollinger'],
                df['Close'] * (1 - risk_percentage / 100)
            ]
        )
        
        return df

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

    return df


def backtest_strategy(
    df, initial_capital=1000, stop_loss_pct=50, max_drawdown_limit=15
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

        # Compute ATR for dynamic stop-loss
        if "ATR" in df.columns:
            atr = row["ATR"][0]
            stop_loss_level = current_price - (2 * atr)  # ATR-based stop-loss
            take_profit_level = current_price + (3 * atr)
        else:
            stop_loss_level = current_price * (
                1 - stop_loss_pct / 100
            )  # Fixed stop-loss
            take_profit_level = current_price * (
                1 + stop_loss_pct / 100
            )  # Fixed take_profit

        if shares > 0 and ((stop_loss_price and row["Close"][0] <= stop_loss_price) or (take_profit_price and row["Close"][0] >= take_profit_price)):
            capital = shares + (capital // row["Close"][0])  # Sell all shares
            shares = 0  # Reset shares
            stop_loss_price = None  # Reset stop-loss level
            take_profit_level = None 
            df.at[index, "Signal"] = -1

        if row["Signal_Strength"][0] != 0 and row["Market_Trend"][0] in [
            "Uptrend",
            "Sideways",
        ]:
            if signal == 1 and shares == 0:  # Buy
                if row["Signal_Strength"][0] == 1:
                    shares = shares + (
                        capital // row["Close"][0]
                    )  # Buy as many shares as possible
                    capital -= (capital // row["Close"][0]) * row["Close"][0]
                    stop_loss_price = stop_loss_level
                    take_profit_price = take_profit_level

        if signal == -1 and shares > 0:  # Sell
            if row["Signal_Strength"][0] < 1:
                capital += shares * row["Close"][0]  # Sell all shares
                shares = 0  # Reset share count
                stop_loss_price = None
                take_profit_price = None


        df.at[index, "Portfolio"] = capital + (
            shares * row["Close"][0]
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
    df["SuperTrend"] = calculate_supertrend(df)

    return df

def calculate_stop_loss(df, risk_percentage=2):
    stop_losses = []
    for i in range(len(df)):
        stop_loss = df["Close"] - (df["ATR"].iloc[i] *risk_percentage)
        stop_losses.append(stop_loss)
    df["Stop_Loss"] = stop_loss
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

    df["UpperBand"] = df["hl2"].astype(float) + (df["ATR"].astype(float) * multiplier)

    df["LowerBand"] = df["hl2"].astype(float) - (df["ATR"].astype(float) * multiplier)
    # Initialize SuperTrend direction
    df["SuperTrend"] = 1  # 1 for bullish, -1 for bearish
    for i in range(1, len(df)):
        close_price = df.iloc[i]["Close"].values
        upper_band = df.iloc[i - 1]["UpperBand"].values
        lower_band = df.iloc[i - 1]["LowerBand"].values
        # print(f'{close_price.values}.{upper_band.values}.{lower_band.values}')

        if close_price > upper_band:
            df.at[i, "SuperTrend"] = 1  # Uptrend
        elif close_price < lower_band:
            df.at[i, "SuperTrend"] = -1  # Downtrend
        else:
            df.at[i, "SuperTrend"] = df.iloc[i - 1]["SuperTrend"]  # Carry forward trend

    return df["SuperTrend"]


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
        & ((df["MFI"] > 30) )
        # & (df["OBV"] >= df["OBV"].rolling(10).mean())
        & (df["ADX"] > 25)
        & (df["SuperTrend"] == 1)
        & (df["EMA_50"] > df["EMA_200"])
    )
    df["Sell_MACD"] = (
        (df["MACD"] < df["MACD_Signal"])
        & ((df["MFI"] < 60))
        & (df["OBV"] < df["OBV"].rolling(10).mean())
        & (df["ADX"] > 25)
        & (df["SuperTrend"] == -1)
        # & (df["EMA_50"] < df["EMA_200"])
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
        # & (df["MFI"] > 40)
        & (df["OBV"] >= df["OBV"].rolling(10).mean())
        # & (df["ADX"] > 25)
        # & (df["SuperTrend"] == 1)
        # & (df["EMA_50"] > df["EMA_200"])
    )

    df["Sell_Stoch"] = (
        (df["Stoch_%K"] < df["Stoch_%D"])
        # & (df["MFI"] < 60)
        & (df["OBV"] < df["OBV"].rolling(10).mean())
        # & (df["ADX"] > 25)
        # & (df["SuperTrend"] == -1)
        # & (df["EMA_50"] < df["EMA_200"])
    )


    df.loc[df["Buy_MACD"], "Signal"] = 1
    df.loc[
        df["Buy_Stoch"],
        "Signal",
    ] = 1

    # Apply sell signals across all rows where conditions are met
    df.loc[df["Sell_MACD"], "Signal"] = -1
    df.loc[
        df["Sell_Stoch"],
        "Signal",
    ] = -1

    return df



def plot_indicators(df, symbol):
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Mark Buy and Sell Points
    buy_signals = df[df["Signal"] == 1]
    sell_signals = df[df["Signal"] == -1]

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

    axs[3].plot(df["Portfolio"], label="Portfolio Value", color="green")
    axs[3].set_title(f"{symbol} Portfolio Performance")
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


def signal_strength(df):
    # Calculate the difference between fast and slow moving averages
    diff = df["SMA_10"] - df["SMA_50"]

    # Define thresholds for strong and weak signals
    strong_threshold = 0.01  # 1% difference between SMAs
    weak_threshold = 0.005  # 0.5% difference between SMAs

    # Initialize signal strength
    df["Signal_Strength"] = 0

    for i in range(1, len(df)):
        # Strong Buy Signal
        if diff.iloc[i] > strong_threshold:
            df.at[df.index[i], "Signal_Strength"] = 1  # Strong Buy

        # Weak Buy Signal
        elif weak_threshold < diff.iloc[i] <= strong_threshold:
            df.at[df.index[i], "Signal_Strength"] = 0.5  # Weak Buy

        # Strong Sell Signal
        elif diff.iloc[i] < -strong_threshold:
            df.at[df.index[i], "Signal_Strength"] = -1  # Strong Sell

        # Weak Sell Signal
        elif -strong_threshold <= diff.iloc[i] < -weak_threshold:
            df.at[df.index[i], "Signal_Strength"] = -0.5  # Weak Sell

        # No Signal
        else:
            df.at[df.index[i], "Signal_Strength"] = 0

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
    api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")  #
    balance = []
    # symbols = ["TSM","AMZN",'USEG','OKLO','COIN']
    symbols = ["TSM", "AMZN", "MSFT", "COIN", "V"]
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
            df = calculate_stop_loss(df)

            df = signal_calculate(df, symbol)
            df = signal_strength(df)
            # ml_signal_generator = MLSignalGenerator()
            # features, labels = debug_shape_mismatch(df, ml_signal_generator)
            # print(f'features:{features},labelsl:{labels}')
            # df = enhanced_signal_calculate(df, ml_signal_generator)
            df = backtest_strategy(df)
            plot_indicators(df, symbol)
            # print(df[['Close', 'BB_High', 'BB_Low', 'BB_Middle', 'MACD', 'MACD_Signal', 'MACD_Hist', 'MA100', 'Stoch_%K', 'Stoch_%D', 'OBV','Signal','Portfolio']].tail())
            # Place Order Based on the Last Signal
            latest_signal = df.iloc[-1]["Signal"]
            qty = 10  # You can adjust quantity based on strategy or capital

            # Set your risk management
            stop_loss_pct = 0.5  # 50% Stop Loss
            take_profit_pct = 0.8  # 80% Take Profit

            # if latest_signal == 1:
            #     place_order(api, symbol, qty, "buy", stop_loss_pct, take_profit_pct)
            # elif latest_signal == -1:
            #     place_order(api, symbol, qty, "sell", stop_loss_pct, take_profit_pct)
            
            # Calculate daily returns
            df["Daily_Return"] = df["Portfolio"].pct_change()

            # Compute Sharpe Ratio
            sharpe_rt = calculate_sharpe_ratio(df["Daily_Return"].dropna())

            # Compute Max Drawdown
            max_dd = calculate_max_drawdown(df["Portfolio"])

            max_drawdown.append({"symbol": symbol, "max_dd": max_dd})
            sharpe_ratio.append({"symbol": symbol, "sharpe_rt": sharpe_rt})
            balance.append({"symbol": symbol, "balance": df[["Portfolio"]].tail()})
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
