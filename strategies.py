# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ccxt  # For connecting to exchange APIs
from backtesting import Backtest, Strategy  # For backtesting
import statsmodels.api as sm

# Fetch historical data from an exchange (e.g., Binance)
exchange = ccxt.binance()
symbol = 'BTC/USDT'
timeframe = '1h'
limit = 1000  # Number of data points to fetch

# Fetch historical OHLCV data
data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# Feature Engineering
# Calculate moving averages
df['MA20'] = df['close'].rolling(window=20).mean()
df['MA50'] = df['close'].rolling(window=50).mean()

# Calculate Bollinger Bands
df['std20'] = df['close'].rolling(window=20).std()
df['upper_band'] = df['MA20'] + (df['std20'] * 2)
df['lower_band'] = df['MA20'] - (df['std20'] * 2)

# Calculate RSI
delta = df['close'].diff()
up, down = delta.copy(), delta.copy()
up[up < 0] = 0
down[down > 0] = 0
roll_up = up.rolling(window=14).mean()
roll_down = down.abs().rolling(window=14).mean()
RS = roll_up / roll_down
df['RSI'] = 100.0 - (100.0 / (1.0 + RS))

# Statistical Strategy using mean reversion
class MeanReversionStrategy(Strategy):
    def init(self):
        pass

    def next(self):
        price = self.data.Close[-1]
        upper_band = self.data.upper_band[-1]
        lower_band = self.data.lower_band[-1]
        
        # Entry signals
        if price < lower_band:
            self.buy()
        elif price > upper_band:
            self.sell()
        # Exit positions
        else:
            if self.position.is_long and price > self.data.MA20[-1]:
                self.position.close()
            elif self.position.is_short and price < self.data.MA20[-1]:
                self.position.close()

# Prepare data for backtesting
df.dropna(inplace=True)

# Backtesting the strategy
bt = Backtest(df, MeanReversionStrategy, cash=10000, commission=.002)
stats = bt.run()
print(stats)
bt.plot()

# Connecting to Broker API for live trading
# Set your API keys (replace with your actual keys)
exchange.apiKey = 'YOUR_API_KEY'
exchange.secret = 'YOUR_SECRET_KEY'

# Function to place orders
def place_order(symbol, side, amount, price=None):
    if price:
        # Limit order
        order = exchange.create_limit_order(symbol, side, amount, price)
    else:
        # Market order
        order = exchange.create_market_order(symbol, side, amount)
    return order

# Example usage:
# order = place_order('BTC/USDT', 'buy', 0.001)