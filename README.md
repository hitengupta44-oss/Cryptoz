# Cryptoz
Real-time BTCUSD analysis using MT5, Python, and LSTM. Fetches live 1-min candles, computes indicators (EMA, SMA, RSI, MACD, VWAP), predicts future prices, generates buy/sell signals, and visualizes actual vs predicted candlesticks using Plotly for advanced crypto forecasting.

Technical Indicators Used

1. EMA20 (Exponential Moving Average – 20 periods)

* Gives more weight to recent prices.
* Reacts faster to trend changes than SMA.
* Used to detect **short-term trend direction.

* When **EMA20 crosses above SMA50 → BUY
* When **EMA20 crosses below SMA50 → SELL

2. SMA50 (Simple Moving Average – 50 periods)

* Calculates the average price of the last 50 candles.
* Smoother, slower trend indicator.
* Used as the **baseline trend filter.

EMA20 vs SMA50 crossovers = trend reversal signals.

3. RSI (Relative Strength Index – 14 periods)

* Measures momentum between 0–100.
* Shows if the market is overbought (>70) or oversold (<30).
* Helps model understand price pressure direction.

4. MACD (Moving Average Convergence Divergence)

* MACD Line** = EMA12 – EMA26
* Signal Line** = 9-period EMA of MACD

MACD tells:

* Momentum strength
* Trend continuation or reversal
* Zero-line crossovers

5. Bollinger Bands (Upper & Lower Bands)

* Middle band: SMA20
* Upper band: +2 standard deviations
* Lower band: –2 standard deviations

Helps detect:

* Volatility expansion
* Price breakouts
* Mean reversion behavior

6. VWAP (Volume Weighted Average Price)

* Shows the **fair price relative to volume.
* Higher weight to candles with big volume.
* Used by traders to detect institutional entry/exit zones.

2. LSTM Model Explanation

LSTM(64, return_sequences=True)
Dropout(0.2)
LSTM(32)
Dense(16, activation='relu')
Dense(1)   

* Takes the last 60 minutes of data (lookback)
* Each minute includes 7 features
  (close, EMA20, SMA50, RSI, MACD, MACD_Signal, VWAP)
* Learns:

  * Trends
  * Momentum shifts
  * Indicator patterns
  * Price cycles

*Training:

* 80% train
* 20% test
* Loss function: MSE
* Optimizer: Adam

The model outputs the next candle’s close price (scaled).

3. Future Prediction Loop (Next 10 Minutes)

For each future step:

1. Predict next close price
2. Convert prediction from scaled → actual price
3. Build a new synthetic candle (open, high, low, close)
4. Append it to temp dataframe
5. Recalculate all indicators with new data
6. Feed the updated series back to LSTM
7. Repeat for 10 steps


4. Final Output

1. Live Candlestick Chart (Actual Data)

Fetched from MT5 for the last 24 hours (M1 timeframe).

2. Predicted Future Candles (Next 10 Minutes)

Shows:

* Predicted (open)
* Predicted (close)
* Synthetic (high) and (low)
* Time progression (1 min each)

3. Trading Signals

Generated using EMA20–SMA50 crossover logic:

* BUY (Triangle Up) bullish crossover
* SELL (Triangle Down)  bearish crossover



5. Visualization Summary

The final Plotly chart includes:

Actual Candlesticks

(green / red)

Predicted Candlesticks

(orange / yellow)

Technical Indicators

EMA20 (blue)
SMA50 (orange)
VWAP (purple)

Trading Signals

Blue ▲ for BUY
Red ▼ for SELL




