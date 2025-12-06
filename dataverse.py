# === IMPORTS ===
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import ta  # for RSI, MACD, Bollinger Bands, etc.

# === CONNECT TO METATRADER 5 ===
if not mt5.initialize():
    print("❌ MT5 initialization failed")
    quit()

# === SETTINGS ===
symbol = "BTCUSD"
timeframe = mt5.TIMEFRAME_M1  # 1-min candles
end_time = datetime.now()
start_time = end_time - timedelta(days=1)  # last 24h

# === FETCH DATA ===
rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
mt5.shutdown()

if rates is None or len(rates) == 0:
    print("⚠️ No data returned. Check symbol or MT5 feed.")
    quit()

# === PREPARE DATAFRAME ===
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')

# === TECHNICAL INDICATORS ===
df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
df['SMA50'] = df['close'].rolling(window=50).mean()
df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

macd = ta.trend.MACD(df['close'])
df['MACD'] = macd.macd()
df['MACD_Signal'] = macd.macd_signal()

bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
df['BB_Upper'] = bb.bollinger_hband()
df['BB_Lower'] = bb.bollinger_lband()

df['VWAP'] = (df['close'] * df['tick_volume']).cumsum() / df['tick_volume'].cumsum()

df = df.dropna().reset_index(drop=True)

# === FEATURE SELECTION ===
features = ['close', 'EMA20', 'SMA50', 'RSI', 'MACD', 'MACD_Signal', 'VWAP']
data = df[features].values

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# === CREATE SEQUENCES ===
def create_sequences(data, lookback=60):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, 0])  # predict close
    return np.array(X), np.array(y)

lookback = 60
X, y = create_sequences(scaled_data, lookback)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# === LSTM MODEL ===
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(lookback, len(features))),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dense(16, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# === IMPROVED PREDICTION LOOP ===
n_future = 10
last_seq = scaled_data[-lookback:].copy()
preds_scaled = []

temp_df = df.copy()  # temporary DF for indicator recalculation

for step in range(n_future):
    # Predict next close (scaled)
    pred_scaled = model.predict(last_seq.reshape(1, lookback, len(features)), verbose=0)[0][0]
    preds_scaled.append(pred_scaled)

    # Convert scaled prediction to actual close
    pred_close = scaler.inverse_transform(
        np.hstack([[pred_scaled], np.zeros(len(features) - 1)]).reshape(1, -1)
    )[0][0]

    # Create new candle based on prediction
    new_row = {
        'time': temp_df['time'].iloc[-1] + timedelta(minutes=1),
        'open': temp_df['close'].iloc[-1],
        'high': max(pred_close, temp_df['close'].iloc[-1]) * (1 + np.random.uniform(0.0003, 0.001)),
        'low': min(pred_close, temp_df['close'].iloc[-1]) * (1 - np.random.uniform(0.0003, 0.001)),
        'close': pred_close,
        'tick_volume': temp_df['tick_volume'].iloc[-1]
    }

    temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)

    # Recalculate indicators with the new predicted close
    temp_df['EMA20'] = temp_df['close'].ewm(span=20, adjust=False).mean()
    temp_df['SMA50'] = temp_df['close'].rolling(window=50).mean()
    temp_df['RSI'] = ta.momentum.RSIIndicator(temp_df['close'], window=14).rsi()
    macd = ta.trend.MACD(temp_df['close'])
    temp_df['MACD'] = macd.macd()
    temp_df['MACD_Signal'] = macd.macd_signal()
    temp_df['VWAP'] = (temp_df['close'] * temp_df['tick_volume']).cumsum() / temp_df['tick_volume'].cumsum()

    # Prepare new sequence (last 60 rows, scaled)
    latest_features = temp_df[features].tail(lookback).values
    last_seq = scaler.transform(latest_features)

# === CONVERT FINAL PREDICTIONS BACK ===
predicted_close = scaler.inverse_transform(
    np.hstack([np.array(preds_scaled).reshape(-1, 1),
                np.zeros((n_future, len(features) - 1))])
)[:, 0]

# === PREDICTED DATAFRAME ===
pred_times = [df['time'].iloc[-1] + timedelta(minutes=i + 1) for i in range(n_future)]
pred_df = temp_df.tail(n_future).copy()
pred_df['time'] = pred_times
pred_df['close'] = predicted_close
pred_df['open'] = pred_df['close'].shift(1).fillna(df['close'].iloc[-1])
pred_df['high'] = pred_df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0.0003, 0.001, n_future))
pred_df['low'] = pred_df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0.0003, 0.001, n_future))
pred_df = pred_df.reset_index(drop=True)

# === COMBINE REAL + PREDICTED ===
full_df = pd.concat([df, pred_df]).reset_index(drop=True)

# === BUY/SELL SIGNALS ===
full_df['signal'] = None
full_df.loc[(full_df['EMA20'] > full_df['SMA50']) & (full_df['EMA20'].shift(1) <= full_df['SMA50'].shift(1)), 'signal'] = 'BUY'
full_df.loc[(full_df['EMA20'] < full_df['SMA50']) & (full_df['EMA20'].shift(1) >= full_df['SMA50'].shift(1)), 'signal'] = 'SELL'

buy_df = full_df[full_df['signal'] == 'BUY']
sell_df = full_df[full_df['signal'] == 'SELL']

# === PLOT ===
fig = go.Figure()

# Actual Candles
fig.add_trace(go.Candlestick(
    x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
    name='Actual', increasing_line_color='green', decreasing_line_color='red'
))

# Predicted Candles
fig.add_trace(go.Candlestick(
    x=pred_df['time'], open=pred_df['open'], high=pred_df['high'], low=pred_df['low'], close=pred_df['close'],
    name='Predicted', increasing_line_color='orange', decreasing_line_color='yellow', opacity=0.8
))

# Indicators
fig.add_trace(go.Scatter(x=full_df['time'], y=full_df['EMA20'], mode='lines', name='EMA20', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=full_df['time'], y=full_df['SMA50'], mode='lines', name='SMA50', line=dict(color='orange')))
fig.add_trace(go.Scatter(x=full_df['time'], y=full_df['VWAP'], mode='lines', name='VWAP', line=dict(color='purple')))

# Buy/Sell markers
fig.add_trace(go.Scatter(
    x=buy_df['time'], y=buy_df['close'], mode='markers', name='Buy',
    marker=dict(symbol='triangle-up', size=10, color='blue', line=dict(width=1, color='white'))
))
fig.add_trace(go.Scatter(
    x=sell_df['time'], y=sell_df['close'], mode='markers', name='Sell',
    marker=dict(symbol='triangle-down', size=10, color='red', line=dict(width=1, color='white'))
))

# Layout
fig.update_layout(
    title=f"{symbol} — Live + Dynamic Predicted (Next 10 min) | EMA/SMA/RSI/MACD/VWAP",
    xaxis_title="Time",
    yaxis_title="Price",
    template="plotly_white",
    hovermode='x unified',
    xaxis_rangeslider_visible=True,
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=60, label="1h", step="minute", stepmode="backward"),
                dict(count=360, label="6h", step="minute", stepmode="backward"),
                dict(count=720, label="12h", step="minute", stepmode="backward"),
                dict(step="all")
            ])
        )
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

fig.show()
