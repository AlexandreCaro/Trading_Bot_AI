# trading_bot_app.py

import streamlit as st
import pandas as pd
import numpy as np
import ccxt
from backtesting import Backtest, Strategy
import matplotlib.pyplot as plt
from io import BytesIO
import datetime

# Define the indicator functions
def SMA(values, n):
    return pd.Series(values).rolling(n).mean().values

def STD(values, n):
    return pd.Series(values).rolling(n).std().values

# Define the trading strategy with parameters
class MeanReversionStrategy(Strategy):
    # Define the parameters with default values
    ma_short = 20
    ma_long = 50
    std_multiplier = 2.0

    def init(self):
        # Precompute the indicators
        close = self.data.Close
        self.ma_short_series = self.I(SMA, close, self.ma_short)
        self.ma_long_series = self.I(SMA, close, self.ma_long)
        self.std_series = self.I(STD, close, self.ma_short)
        self.upper_band = self.ma_short_series + self.std_multiplier * self.std_series
        self.lower_band = self.ma_short_series - self.std_multiplier * self.std_series

    def next(self):
        price = self.data.Close[-1]
        upper_band = self.upper_band[-1]
        lower_band = self.lower_band[-1]
        ma_short = self.ma_short_series[-1]

        # Entry signals
        if price < lower_band:
            self.buy()
        elif price > upper_band:
            self.sell()
        # Exit positions
        else:
            if self.position.is_long and price > ma_short:
                self.position.close()
            elif self.position.is_short and price < ma_short:
                self.position.close()

def fetch_data(symbol, timeframe, since, until):
    exchange = ccxt.binance()
    since_timestamp = exchange.parse8601(since)
    until_timestamp = exchange.parse8601(until)
    all_data = []
    while since_timestamp < until_timestamp:
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_timestamp, limit=1000)
        if not data:
            break
        last_timestamp = data[-1][0]
        since_timestamp = last_timestamp + 1
        all_data.extend(data)
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    # Rename columns to match Backtesting library expectations
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    return df

# Streamlit App
def main():
    st.title("Trading Bot Web Application")

    # Sidebar for input parameters
    st.sidebar.header("Input Parameters")
    symbol = st.sidebar.text_input("Symbol", value="BTC/USDT")
    timeframe = st.sidebar.selectbox(
        "Timeframe", options=['1m', '5m', '15m', '1h', '4h', '1d'], index=3)
    start_date = st.sidebar.date_input("Start Date", datetime.date(2022, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.date(2022, 12, 31))
    ma_short = st.sidebar.number_input(
        "Short Moving Average Window", min_value=5, max_value=50, value=20)
    ma_long = st.sidebar.number_input(
        "Long Moving Average Window", min_value=20, max_value=200, value=50)
    std_multiplier = st.sidebar.number_input(
        "Standard Deviation Multiplier", min_value=1.0, max_value=3.0, value=2.0)
    cash = st.sidebar.number_input(
        "Initial Cash", min_value=1000, max_value=100000, value=10000, step=1000)
    commission = st.sidebar.number_input(
        "Commission (%)", min_value=0.0, max_value=1.0, value=0.2) / 100

    optimize = st.sidebar.checkbox("Optimize Parameters")

    if st.sidebar.button("Run Backtest"):
        with st.spinner("Fetching data and running backtest..."):
            since = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
            until = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
            df = fetch_data(symbol, timeframe, since, until)
            if df.empty:
                st.error("No data fetched. Please adjust the date range or check the symbol.")
                return

            df.dropna(inplace=True)

            if optimize:
                # Define the parameter ranges for optimization
                ma_short_range = range(10, 31, 5)
                ma_long_range = range(40, 101, 10)
                std_multiplier_range = [1.5, 2.0, 2.5]

                bt = Backtest(df, MeanReversionStrategy, cash=cash, commission=commission)
                stats, heatmap = bt.optimize(
                    ma_short=ma_short_range,
                    ma_long=ma_long_range,
                    std_multiplier=std_multiplier_range,
                    maximize='Equity Final [$]',
                    constraint=lambda param: param.ma_short < param.ma_long,
                    return_heatmap=True
                )
                st.success("Optimization Completed!")
                st.subheader("Best Parameters")
                st.write(stats._strategy)
                st.subheader("Backtest Statistics")
                st.write(stats)
                st.subheader("Optimization Heatmap")
                ax = heatmap.plot()
                fig = ax.get_figure()
                st.pyplot(fig)
                plt.close(fig)
            else:
                bt = Backtest(df, MeanReversionStrategy, cash=cash, commission=commission)
                stats = bt.run(
                    ma_short=ma_short,
                    ma_long=ma_long,
                    std_multiplier=std_multiplier
                )
                st.success("Backtest Completed!")
                st.subheader("Backtest Statistics")
                st.write(stats)

            st.subheader("Equity Curve")
            bt.plot(open_browser=False)
            img = BytesIO()
            plt.savefig(img, format='png')
            plt.close()
            st.image(img)

            # Paper Trading Simulation
            st.subheader("Paper Trading Simulation")
            if 'account_balance' not in st.session_state:
                st.session_state.account_balance = cash
                st.session_state.positions = []

            st.write(f"Account Balance: ${st.session_state.account_balance:.2f}")

            # Optionally, display the data
            if st.checkbox("Show Data"):
                st.subheader("Data")
                st.write(df)

    st.sidebar.header("Live Trading (Use with Caution)")
    api_key = st.sidebar.text_input("API Key")
    secret_key = st.sidebar.text_input("Secret Key")
    order_symbol = st.sidebar.text_input("Order Symbol", value=symbol)
    order_side = st.sidebar.selectbox("Order Side", options=['buy', 'sell'])
    order_amount = st.sidebar.number_input(
        "Order Amount", min_value=0.0001, value=0.001, step=0.0001)
    order_price = st.sidebar.number_input(
        "Order Price (Leave 0 for Market Order)", min_value=0.0, value=0.0)

    if st.sidebar.button("Place Order"):
        if not api_key or not secret_key:
            st.error("Please enter API Key and Secret Key")
        else:
            exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': secret_key,
                'enableRateLimit': True,
            })
            try:
                if order_price > 0:
                    # Limit Order
                    order = exchange.create_limit_order(
                        order_symbol, order_side, order_amount, order_price)
                else:
                    # Market Order
                    order = exchange.create_order(
                        order_symbol, 'market', order_side, order_amount)
                st.success(f"Order placed: {order}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
