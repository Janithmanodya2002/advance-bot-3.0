import os
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
from binance.client import Client  # Keep for constants
import glob

# --- Configuration ---
DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
SYMBOLS_FILE = "symbols.csv"
SWING_WINDOW = 5
LOOKBACK_CANDLES = 100 # From main.py config
TRADE_EXPIRY_BARS = 96 # How many 15-min bars to wait for a result (1 day)

# Timezone definitions for trading sessions (in UTC)
SESSIONS = {
    "Asia": (0, 8),
    "London": (7, 15),
    "New_York": (12, 20)
}

# --- Data Loading and Initial Processing ---

def get_session(timestamp):
    """Determines the trading session(s) for a given timestamp."""
    if isinstance(timestamp, pd.Series):
        return timestamp.apply(lambda ts: _get_single_session(ts))
    else:
        return _get_single_session(timestamp)

def _get_single_session(timestamp):
    """Helper function to process a single timestamp."""
    utc_time = pd.to_datetime(timestamp, unit='ms').tz_localize('UTC')
    hour = utc_time.hour
    active_sessions = [name for name, (start, end) in SESSIONS.items() if start <= hour <= end]
    return ",".join(active_sessions) if active_sessions else "Inactive"

def process_and_save_kline_data(klines, symbol):
    """Processes raw kline data and saves it to a Parquet file."""
    if not klines:
        print(f"No kline data to process for {symbol}.")
        return
    print(f"Processing and saving {len(klines)} candles for {symbol}...")
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['session'] = get_session(df['timestamp'])
    save_data_to_parquet(df, symbol)

def save_data_to_parquet(df, symbol):
    """Saves the DataFrame to a Parquet file."""
    if df is None or df.empty:
        return
    symbol_dir = os.path.join(DATA_DIR, symbol)
    os.makedirs(symbol_dir, exist_ok=True)
    file_date = pd.to_datetime(df['timestamp'].iloc[-1], unit='ms').strftime('%Y-%m-%d')
    file_path = os.path.join(symbol_dir, f"{file_date}.parquet")
    print(f"  - Saving data for {symbol} to {file_path}")
    table = pa.Table.from_pandas(df)
    pq.write_table(table, file_path)

# --- Preprocessing and Labeling (Step 3) ---

def load_raw_data_for_symbol(symbol):
    """Loads all parquet files for a symbol and concatenates them."""
    symbol_dir = os.path.join(DATA_DIR, symbol)
    files = glob.glob(os.path.join(symbol_dir, "*.parquet"))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files])
    df.sort_values('timestamp', inplace=True)
    df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def get_swing_points_df(df, window=5):
    """Identify swing points from a DataFrame."""
    highs = df['high']
    lows = df['low']

    swing_highs = []
    swing_lows = []

    for i in range(window, len(df) - window):
        # Swing High
        is_swing_high = highs.iloc[i] > highs.iloc[i-window:i].max() and highs.iloc[i] > highs.iloc[i+1:i+window+1].max()
        if is_swing_high:
            swing_highs.append({'timestamp': df['timestamp'].iloc[i], 'price': highs.iloc[i]})

        # Swing Low
        is_swing_low = lows.iloc[i] < lows.iloc[i-window:i].min() and lows.iloc[i] < lows.iloc[i+1:i+window+1].min()
        if is_swing_low:
            swing_lows.append({'timestamp': df['timestamp'].iloc[i], 'price': lows.iloc[i]})

    return pd.DataFrame(swing_highs), pd.DataFrame(swing_lows)

def get_trend_df(swing_highs, swing_lows):
    """Determine the trend based on swing points DataFrames."""
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "undetermined"
    if swing_highs['price'].iloc[-1] > swing_highs['price'].iloc[-2] and swing_lows['price'].iloc[-1] > swing_lows['price'].iloc[-2]:
        return "uptrend"
    if swing_highs['price'].iloc[-1] < swing_highs['price'].iloc[-2] and swing_lows['price'].iloc[-1] < swing_lows['price'].iloc[-2]:
        return "downtrend"
    return "undetermined"

def get_fib_retracement(p1, p2, trend):
    """Calculate Fibonacci retracement levels."""
    price_range = abs(p1 - p2)
    if trend == "downtrend":
        golden_zone_start = p1 - (price_range * 0.5)
        golden_zone_end = p1 - (price_range * 0.618)
    else: # Uptrend
        golden_zone_start = p1 + (price_range * 0.5)
        golden_zone_end = p1 + (price_range * 0.618)
    return (golden_zone_start + golden_zone_end) / 2

def label_trade(df, entry_idx, sl_price, tp1_price, tp2_price, side):
    """Labels a single trade based on future price action."""
    future_candles = df.iloc[entry_idx + 1 : entry_idx + 1 + TRADE_EXPIRY_BARS]

    for _, candle in future_candles.iterrows():
        if side == 'long':
            if candle['low'] <= sl_price: return 0 # Loss
            if candle['high'] >= tp2_price: return 2 # Win to TP2
            if candle['high'] >= tp1_price: return 1 # Win to TP1
        elif side == 'short':
            if candle['high'] >= sl_price: return 0 # Loss
            if candle['low'] <= tp2_price: return 2 # Win to TP2
            if candle['low'] <= tp1_price: return 1 # Win to TP1

    return -1 # Trade expired without result

def find_and_label_setups(df):
    """Finds all potential trade setups and labels them."""
    labeled_setups = []

    for i in range(LOOKBACK_CANDLES, len(df) - TRADE_EXPIRY_BARS):
        # Only consider setups in active sessions
        if df['session'].iloc[i] == 'Inactive':
            continue

        current_klines_df = df.iloc[i - LOOKBACK_CANDLES : i]
        swing_highs, swing_lows = get_swing_points_df(current_klines_df, SWING_WINDOW)

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            continue

        trend = get_trend_df(swing_highs, swing_lows)

        setup = None
        if trend == 'downtrend':
            last_swing_high = swing_highs.iloc[-1]['price']
            last_swing_low = swing_lows.iloc[-1]['price']
            entry_price = get_fib_retracement(last_swing_high, last_swing_low, trend)

            if df['close'].iloc[i] > entry_price: # Check if price is in position to trigger
                sl = last_swing_high
                tp1 = entry_price - (sl - entry_price)
                tp2 = entry_price - (sl - entry_price) * 2
                label = label_trade(df, i, sl, tp1, tp2, 'short')
                if label != -1:
                    setup = {'timestamp': df['timestamp'].iloc[i], 'side': 'short', 'entry_price': entry_price, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'label': label}

        elif trend == 'uptrend':
            last_swing_high = swing_highs.iloc[-1]['price']
            last_swing_low = swing_lows.iloc[-1]['price']
            entry_price = get_fib_retracement(last_swing_low, last_swing_high, trend)

            if df['close'].iloc[i] < entry_price: # Check if price is in position to trigger
                sl = last_swing_low
                tp1 = entry_price + (entry_price - sl)
                tp2 = entry_price + (entry_price - sl) * 2
                label = label_trade(df, i, sl, tp1, tp2, 'long')
                if label != -1:
                    setup = {'timestamp': df['timestamp'].iloc[i], 'side': 'long', 'entry_price': entry_price, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'label': label}

        if setup:
            labeled_setups.append(setup)

    return pd.DataFrame(labeled_setups)

def main_preprocess():
    """Main function for preprocessing and labeling."""
    print("Starting preprocessing and labeling...")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    try:
        symbols = pd.read_csv(SYMBOLS_FILE, header=None)[0].tolist()
    except FileNotFoundError:
        print(f"Error: {SYMBOLS_FILE} not found.")
        return

    all_labeled_data = []
    for symbol in symbols:
        print(f"Processing {symbol}...")
        df = load_raw_data_for_symbol(symbol)
        if df is None or df.empty:
            print(f"  - No raw data found for {symbol}. Skipping.")
            continue

        labeled_setups = find_and_label_setups(df)
        if not labeled_setups.empty:
            labeled_setups['symbol'] = symbol
            all_labeled_data.append(labeled_setups)
            print(f"  - Found and labeled {len(labeled_setups)} setups for {symbol}.")
        else:
            print(f"  - No valid setups found for {symbol}.")

    if all_labeled_data:
        final_df = pd.concat(all_labeled_data, ignore_index=True)
        save_path = os.path.join(PROCESSED_DATA_DIR, "labeled_trades.parquet")
        final_df.to_parquet(save_path)
        print(f"Preprocessing complete. Labeled data saved to {save_path}")
    else:
        print("No labeled data was generated.")

# --- Placeholder for Feature Engineering (Step 4) ---
def engineer_features():
    print("Function 'engineer_features' is not yet implemented.")
    pass

# --- Placeholder for Model Training (Step 5) ---
def train_model():
    print("Function 'train_model' is not yet implemented.")
    pass

if __name__ == '__main__':
    # This allows running the preprocessing and labeling step directly
    main_preprocess()
