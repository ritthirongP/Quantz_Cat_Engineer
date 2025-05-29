import backtrader as bt
import pandas as pd
import glob


class CustomPandasData(bt.feeds.PandasData):
    lines = ('signal_strength', 'supertrend', 'market_trend','signal')

    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', -1),
        ('signal_strength', 'Signal_Strength'),
        ('supertrend', 'SuperTrend'),
        ('market_trend', 'Market_Trend_Encoded'),
        ('signal', 'Signal'),
    )

class SignalStrengthStrategy(bt.Strategy):
        # Add custom columns to lines
    params = dict(
        atr_period=14,
        risk_per_trade=0.02,
        stop_loss_atr=2.5,
        take_profit_atr=3.0,
    )

    def __init__(self):
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)
        self.order = None

    def next(self):
        if self.order:
            return  # waiting for pending order

        # lastest in ts
        strength = self.data.signal_strength[0]
        trend = self.data.market_trend[0]
        signal = self.data.signal[0]
        atr = self.atr[0]

        if not self.position:
            if signal == 1 and strength >= 0.5 and trend == 1 and atr > 0:
                cash = self.broker.get_cash()
                size = self._calculate_position_size(cash, atr)
                stop_loss_dist = self.params.stop_loss_atr * atr
                take_profit_dist = self.params.take_profit_atr * atr

                if size > 0:
                    price = self.data.close[0]
                    stop_price = price - stop_loss_dist
                    take_price = price + take_profit_dist

                    self.order = self.buy_bracket(
                        size=size,
                        price=price,
                        stopprice=stop_price,
                        limitprice=take_price
                    )
                    print(f"BUY  @ {price:.2f} x {size} on {self.datas[0].datetime.date(0)}")
        else:
            if signal == -1 and trend == -1:
                self.close()
                print(f"SELL @ {self.data.close[0]:.2f} on {self.datas[0].datetime.date(0)}")

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            self.order = None


    def _calculate_position_size(self, cash, atr):
        """
        Calculate position size dynamically based on available cash and ATR-based risk
        """
        risk_amount = cash * self.params.risk_per_trade
        stop_loss_dist = self.params.take_profit_atr * atr
        size = int(risk_amount / stop_loss_dist) if stop_loss_dist > 0 else 0
        return size


# === Running Backtest ===
if __name__ == '__main__':
    cerebro = bt.Cerebro()


    # Specify the directory and prefix
    directory = "./"
    prefix = "stock_"

    # Use glob to find matching files
    files = glob.glob(f"{directory}/{prefix}*")
    for f in files:
        print(f'filename:{f}')
        df = pd.read_csv(f, index_col='Date', parse_dates=True)
        print(df.dtypes)
        numeric_cols = ["Open", "High", "Low", "Close", "Volume", "Signal_Strength", "SuperTrend", "Signal"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        df.dropna(subset=numeric_cols, inplace=True)
        # Convert 'Uptrend', 'Downtrend', 'Sideways' to numeric codes
        trend_map = {"Uptrend": 1, "Downtrend": -1, "Sideways": 0}
        df["Market_Trend_Encoded"] = df["Market_Trend"].map(trend_map)
        print(df.dtypes)
        data = CustomPandasData(dataname=df)
        cerebro = bt.Cerebro()
        cerebro.adddata(data)
        cerebro.addstrategy(SignalStrengthStrategy)
        cerebro.broker.set_cash(10000)
        cerebro.run()
        cerebro.plot()

