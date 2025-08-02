import os
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
from binance.client import Client as BinanceClient
import glob
import keys

# --- Configuration ---
DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
MODELS_DIR = "models"
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

# --- Data Acquisition (Step 1) ---

def fetch_historical(client, symbol, interval=BinanceClient.KLINE_INTERVAL_15MINUTE, total_limit=20000):
    """
    Fetches historical klines from Binance, handling pagination to get more data.
    """
    all_klines = []
    limit = 1000  # Max limit per request
    
    while len(all_klines) < total_limit:
        try:
            # If we have klines, fetch from before the oldest one
            end_time = all_klines[0][0] if all_klines else None
            
            klines = client.get_historical_klines(
                symbol=symbol, 
                interval=interval, 
                limit=min(limit, total_limit - len(all_klines)),
                end_str=end_time
            )

            # If no more klines are returned, break the loop
            if not klines:
                break

            # Prepend new klines to maintain chronological order
            all_klines = klines + all_klines
            
            print(f"  - Fetched {len(klines)} more candles for {symbol}. Total: {len(all_klines)}/{total_limit}")

        except Exception as e:
            print(f"Error fetching klines for {symbol}: {e}")
            break # Exit on error

    return all_klines


def fetch_initial_data():
    """
    Fetches a large number of historical candles for each symbol
    in symbols.csv and saves them.
    """
    print("Starting initial data acquisition...")
    try:
        symbols = pd.read_csv(SYMBOLS_FILE, header=None)[0].tolist()
    except FileNotFoundError:
        print(f"Error: {SYMBOLS_FILE} not found.")
        return

    client = BinanceClient(keys.api_mainnet, keys.secret_mainnet)
    
    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        klines = fetch_historical(client, symbol, total_limit=20000) # Fetch 20,000 candles

        if klines:
            # Save to a file named after the total number of candles
            filename = f"initial_{len(klines)}.parquet"
            process_and_save_kline_data(klines, symbol, filename)

    print("\nInitial data acquisition complete.")


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

def process_and_save_kline_data(klines, symbol, filename=None):
    """Processes raw kline data and saves it to a Parquet file."""
    if not klines:
        print(f"No kline data to process for {symbol}.")
        return
    print(f"Processing and saving {len(klines)} candles for {symbol}...")
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    
    # Keep the new taker volume column
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume']]
    
    # Coerce all numeric columns to numbers, handling potential errors
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['session'] = get_session(df['timestamp'])
    save_data_to_parquet(df, symbol, filename)

def save_data_to_parquet(df, symbol, filename=None):
    """Saves the DataFrame to a Parquet file."""
    if df is None or df.empty:
        return
    symbol_dir = os.path.join(DATA_DIR, symbol)
    os.makedirs(symbol_dir, exist_ok=True)
    if filename:
        file_path = os.path.join(symbol_dir, filename)
    else:
        file_date = pd.to_datetime(df['timestamp'].iloc[-1], unit='ms').strftime('%Y-%m-%d')
        file_path = os.path.join(symbol_dir, f"{file_date}.parquet")
    print(f"  - Saving data for {symbol} to {file_path}")
    table = pa.Table.from_pandas(df)
    pq.write_table(table, file_path)

# --- Preprocessing and Labeling (Step 3) ---

def load_raw_data_for_symbol(symbol):
    """Loads all parquet files for a symbol and concatenates them."""
    symbol_dir = os.path.join(DATA_DIR, symbol)
    
    # Find any file starting with 'initial_'
    initial_files = glob.glob(os.path.join(symbol_dir, "initial_*.parquet"))
    
    if initial_files:
        # If there are multiple, prefer the one with the most candles (largest file)
        files = [max(initial_files, key=os.path.getsize)]
    else:
        # Fallback to any other parquet files if no 'initial_' file is found
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

    tp1_hit = False
    for _, candle in future_candles.iterrows():
        if side == 'long':
            if candle['low'] <= sl_price:
                # If TP1 was hit before SL, it's a TP1 win. Otherwise, a loss.
                return 1 if tp1_hit else 0
            if candle['high'] >= tp2_price: return 2 # Win to TP2
            if candle['high'] >= tp1_price:
                tp1_hit = True # TP1 is hit, continue checking for TP2 or SL
        elif side == 'short':
            if candle['high'] >= sl_price:
                return 1 if tp1_hit else 0 # Loss
            if candle['low'] <= tp2_price: return 2 # Win to TP2
            if candle['low'] <= tp1_price:
                tp1_hit = True # TP1 is hit

    # If loop finishes
    if tp1_hit:
        return 1 # Exited at TP1 (or expired after hitting TP1)
    return -1 # Trade expired without result

def find_and_label_setups(df):
    """Finds all potential trade setups and labels them."""
    labeled_setups = []

    # Adding tqdm for a progress bar
    for i in tqdm(range(LOOKBACK_CANDLES, len(df) - TRADE_EXPIRY_BARS), desc="Finding Setups"):
        if df['session'].iloc[i] == 'Inactive':
            continue

        current_klines_df = df.iloc[i - LOOKBACK_CANDLES : i]
        swing_highs, swing_lows = get_swing_points_df(current_klines_df, SWING_WINDOW)

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            continue

        trend = get_trend_df(swing_highs, swing_lows)
        setup = None
        if trend == 'downtrend':
            last_swing_high_price = swing_highs.iloc[-1]['price']
            last_swing_low_price = swing_lows.iloc[-1]['price']
            # Ensure the last swing low is below the previous one for a confirmed downtrend setup
            if last_swing_low_price >= swing_lows.iloc[-2]['price']: continue

            entry_price = get_fib_retracement(last_swing_high_price, last_swing_low_price, trend)
            sl = last_swing_high_price
            tp1 = entry_price - (sl - entry_price)
            tp2 = entry_price - (sl - entry_price) * 2

            # Find the actual candle index where the entry was triggered
            entry_candle_idx = -1
            for k in range(i, min(i + TRADE_EXPIRY_BARS, len(df))):
                 if df['high'].iloc[k] >= entry_price:
                     entry_candle_idx = k
                     break
            if entry_candle_idx == -1: continue

            label = label_trade(df, entry_candle_idx, sl, tp1, tp2, 'short')
            if label != -1:
                setup = {
                    'timestamp': df['timestamp'].iloc[i], 'side': 'short', 
                    'entry_price': entry_price, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'label': label,
                    'swing_high_price': last_swing_high_price, 'swing_low_price': last_swing_low_price
                }

        elif trend == 'uptrend':
            last_swing_high_price = swing_highs.iloc[-1]['price']
            last_swing_low_price = swing_lows.iloc[-1]['price']
            if last_swing_high_price <= swing_highs.iloc[-2]['price']: continue

            entry_price = get_fib_retracement(last_swing_low_price, last_swing_high_price, trend)
            sl = last_swing_low_price
            tp1 = entry_price + (entry_price - sl)
            tp2 = entry_price + (entry_price - sl) * 2

            entry_candle_idx = -1
            for k in range(i, min(i + TRADE_EXPIRY_BARS, len(df))):
                 if df['low'].iloc[k] <= entry_price:
                     entry_candle_idx = k
                     break
            if entry_candle_idx == -1: continue

            label = label_trade(df, entry_candle_idx, sl, tp1, tp2, 'long')
            if label != -1:
                setup = {
                    'timestamp': df['timestamp'].iloc[i], 'side': 'long',
                    'entry_price': entry_price, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'label': label,
                    'swing_high_price': last_swing_high_price, 'swing_low_price': last_swing_low_price
                }

        if setup:
            labeled_setups.append(setup)

    return pd.DataFrame(labeled_setups)

from multiprocessing import Pool

def process_symbol_for_labeling(symbol):
    """Helper function to process and label setups for a single symbol."""
    print(f"Processing {symbol}...")
    df = load_raw_data_for_symbol(symbol)
    if df is None or df.empty:
        print(f"  - No raw data found for {symbol}. Skipping.")
        return None

    labeled_setups = find_and_label_setups(df)
    if not labeled_setups.empty:
        labeled_setups['symbol'] = symbol
        print(f"  - Found and labeled {len(labeled_setups)} setups for {symbol}.")
        return labeled_setups
    else:
        print(f"  - No valid setups found for {symbol}.")
        return None

def main_preprocess():
    """Main function for preprocessing and labeling, using multiprocessing."""
    print("Starting preprocessing and labeling...")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    try:
        symbols = pd.read_csv(SYMBOLS_FILE, header=None)[0].tolist()
    except FileNotFoundError:
        print(f"Error: {SYMBOLS_FILE} not found.")
        return

    # Use multiprocessing.Pool to process symbols in parallel
    with Pool() as pool:
        results = pool.map(process_symbol_for_labeling, symbols)

    # Filter out None results
    all_labeled_data = [res for res in results if res is not None]

    if all_labeled_data:
        final_df = pd.concat(all_labeled_data, ignore_index=True)
        save_path = os.path.join(PROCESSED_DATA_DIR, "labeled_trades.parquet")
        final_df.to_parquet(save_path)
        print(f"Preprocessing complete. Labeled data saved to {save_path}")
    else:
        print("No labeled data was generated.")

# --- Feature Engineering (Step 3) ---

def calculate_atr(df, period=14):
    """Calculates the Average True Range (ATR)."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def calculate_rsi(df, period=14):
    """Calculates the Relative Strength Index (RSI)."""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    
    # Avoid division by zero
    rs = gain / loss
    rs = rs.replace([np.inf, -np.inf], np.nan) # Handle infinities if loss is 0

    rsi = 100 - (100 / (1 + rs))
    
    # When gain and loss are both 0, RSI is NaN. A neutral 50 is a common treatment.
    rsi.fillna(50, inplace=True)
    
    return rsi

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    """Calculates the Moving Average Convergence Divergence (MACD)."""
    fast_ema = df['close'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = df['close'].ewm(span=slow_period, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

def calculate_adx(df, period=14):
    """Calculates the Average Directional Index (ADX)."""
    df['high_diff'] = df['high'].diff()
    df['low_diff'] = df['low'].diff()
    df['plus_dm'] = np.where((df['high_diff'] > df['low_diff']) & (df['high_diff'] > 0), df['high_diff'], 0)
    df['minus_dm'] = np.where((df['low_diff'] > df['high_diff']) & (df['low_diff'] > 0), df['low_diff'], 0)

    # ATR is needed for ADX calculation
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()

    plus_di = 100 * (df['plus_dm'].ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (df['minus_dm'].ewm(alpha=1/period, adjust=False).mean() / atr)

    dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    # Clean up temporary columns
    df.drop(['high_diff', 'low_diff', 'plus_dm', 'minus_dm'], axis=1, inplace=True)
    
    return adx.fillna(0) # Return ADX, fill initial NaNs with 0

def calculate_bollinger_bands(df, window=20, num_std_dev=2):
    """Calculates Bollinger Bands."""
    rolling_mean = df['close'].rolling(window=window).mean()
    rolling_std = df['close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band

def calculate_ichimoku_cloud(df):
    """Calculates Ichimoku Cloud components."""
    # Tenkan-sen (Conversion Line)
    tenkan_sen_high = df['high'].rolling(window=9).max()
    tenkan_sen_low = df['low'].rolling(window=9).min()
    tenkan_sen = (tenkan_sen_high + tenkan_sen_low) / 2

    # Kijun-sen (Base Line)
    kijun_sen_high = df['high'].rolling(window=26).max()
    kijun_sen_low = df['low'].rolling(window=26).min()
    kijun_sen = (kijun_sen_high + kijun_sen_low) / 2

    # Senkou Span A (Leading Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

    # Senkou Span B (Leading Span B)
    senkou_span_b_high = df['high'].rolling(window=52).max()
    senkou_span_b_low = df['low'].rolling(window=52).min()
    senkou_span_b = ((senkou_span_b_high + senkou_span_b_low) / 2).shift(26)

    # Chikou Span (Lagging Span)
    chikou_span = df['close'].shift(-26)

    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def engineer_features():
    """Loads labeled trades and engineers features for the ML model."""
    print("Starting feature engineering...")
    try:
        labeled_df = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "labeled_trades.parquet"))
    except FileNotFoundError:
        print("Error: labeled_trades.parquet not found. Please run main_preprocess() first.")
        return

    # Load all raw data into a dictionary for quick access
    all_raw_data = {}
    symbols = labeled_df['symbol'].unique()
    for symbol in symbols:
        df = load_raw_data_for_symbol(symbol)
        if df is not None:
            # Pre-calculate indicators for the whole series to speed up lookups
            df['atr'] = calculate_atr(df)
            df['rsi'] = calculate_rsi(df)
            df['macd'], df['macd_signal'] = calculate_macd(df)
            df['adx'] = calculate_adx(df)
            df['bb_upper'], df['bb_lower'] = calculate_bollinger_bands(df)
            df['tenkan_sen'], df['kijun_sen'], df['senkou_span_a'], df['senkou_span_b'], df['chikou_span'] = calculate_ichimoku_cloud(df)
            
            for p in [20, 50, 100]:
                df[f'volatility_{p}'] = df['close'].pct_change().rolling(window=p).std()
            
            df['liquidity_proxy'] = df['volume'].rolling(window=96).mean()
            
            df.set_index(pd.to_datetime(df['timestamp'], unit='ms'), inplace=True)

            df_4h = df['close'].resample('4h').ohlc()
            df_4h['rsi'] = calculate_rsi(df_4h)
            df_4h['macd'], df_4h['macd_signal'] = calculate_macd(df_4h)
            df_4h['macd_diff'] = df_4h['macd'] - df_4h['macd_signal']
            
            all_raw_data[symbol] = {'15m': df, '4h': df_4h}

    # --- Volatility Regime Calculation ---
    all_volatility = pd.concat([data['15m']['volatility_50'] for symbol, data in all_raw_data.items() if 'volatility_50' in data['15m'] and not data['15m'].empty]).dropna()
    if not all_volatility.empty:
        vol_low_threshold, vol_high_threshold = all_volatility.quantile([0.33, 0.67])
    else:
        vol_low_threshold, vol_high_threshold = 0, 0

    feature_list = []
    for _, row in tqdm(labeled_df.iterrows(), total=labeled_df.shape[0], desc="Engineering Features"):
        symbol = row['symbol']
        timestamp = row['timestamp']

        if symbol not in all_raw_data:
            continue
        
        data_dict = all_raw_data[symbol]
        raw_df_15m = data_dict['15m']
        raw_df_4h = data_dict['4h']

        setup_timestamp_dt = pd.to_datetime(timestamp, unit='ms')

        try:
            setup_candle = raw_df_15m.loc[setup_timestamp_dt]
            htf_candle = raw_df_4h.asof(setup_timestamp_dt)
        except KeyError:
            continue

        features = {}
        features['symbol'], features['timestamp'], features['side'] = symbol, timestamp, row['side']
        features['entry_price'], features['sl'], features['tp1'], features['tp2'] = row['entry_price'], row['sl'], row['tp1'], row['tp2']
        features['label'] = row['label']

        swing_height = row['swing_high_price'] - row['swing_low_price']
        if swing_height > 0:
            if row['side'] == 'long':
                features['golden_zone_ratio'] = (row['entry_price'] - row['swing_low_price']) / swing_height
            else:
                features['golden_zone_ratio'] = (row['swing_high_price'] - row['entry_price']) / swing_height
            features['sl_pct'] = abs(row['entry_price'] - row['sl']) / swing_height
            features['tp1_pct'] = abs(row['tp1'] - row['entry_price']) / swing_height
            features['tp2_pct'] = abs(row['tp2'] - row['entry_price']) / swing_height
        else:
            features['golden_zone_ratio'], features['sl_pct'], features['tp1_pct'], features['tp2_pct'] = np.nan, np.nan, np.nan, np.nan

        candle_range = setup_candle['high'] - setup_candle['low']
        if candle_range > 0:
            features['body_to_range_ratio'] = abs(setup_candle['open'] - setup_candle['close']) / candle_range
            features['upper_wick_ratio'] = (setup_candle['high'] - max(setup_candle['open'], setup_candle['close'])) / candle_range
            features['lower_wick_ratio'] = (min(setup_candle['open'], setup_candle['close']) - setup_candle['low']) / candle_range
        else:
            features['body_to_range_ratio'], features['upper_wick_ratio'], features['lower_wick_ratio'] = 0, 0, 0

        try:
            setup_idx_loc = raw_df_15m.index.get_loc(setup_timestamp_dt)
            for n in range(1, 4):
                prev_candle = raw_df_15m.iloc[setup_idx_loc - n]
                prev_range = prev_candle['high'] - prev_candle['low']
                if prev_range > 0:
                    features[f'body_to_range_ratio_t-{n}'] = abs(prev_candle['open'] - prev_candle['close']) / prev_range
                    features[f'upper_wick_ratio_t-{n}'] = (prev_candle['high'] - max(prev_candle['open'], prev_candle['close'])) / prev_range
                    features[f'lower_wick_ratio_t-{n}'] = (min(prev_candle['open'], prev_candle['close']) - prev_candle['low']) / prev_range
                else:
                    features[f'body_to_range_ratio_t-{n}'] = 0
                    features[f'upper_wick_ratio_t-{n}'] = 0
                    features[f'lower_wick_ratio_t-{n}'] = 0
        except (KeyError, IndexError):
            # If we can't find the candle or there aren't enough previous ones, fill with 0
            for n in range(1, 4):
                features[f'body_to_range_ratio_t-{n}'] = 0
                features[f'upper_wick_ratio_t-{n}'] = 0
                features[f'lower_wick_ratio_t-{n}'] = 0
        
        # Session One-Hot Encoding
        session_str = setup_candle['session']
        features['is_asia'] = 1 if 'Asia' in session_str else 0
        features['is_london'] = 1 if 'London' in session_str else 0
        features['is_new_york'] = 1 if 'New_York' in session_str else 0
        
        # Market Context Features (from pre-calculated values)
        features['atr'] = setup_candle['atr']
        features['rsi'] = setup_candle['rsi']
        features['macd_diff'] = setup_candle['macd'] - setup_candle['macd_signal']
        features['adx'] = setup_candle['adx']
        for p in [20, 50, 100]:
            features[f'volatility_{p}'] = setup_candle[f'volatility_{p}']
        
        # Bollinger Bands
        features['bb_upper'] = setup_candle['bb_upper']
        features['bb_lower'] = setup_candle['bb_lower']

        # Ichimoku Cloud
        features['tenkan_sen'] = setup_candle['tenkan_sen']
        features['kijun_sen'] = setup_candle['kijun_sen']
        features['senkou_span_a'] = setup_candle['senkou_span_a']
        features['senkou_span_b'] = setup_candle['senkou_span_b']
        features['chikou_span'] = setup_candle['chikou_span']

        # Higher-Timeframe Context
        features['rsi_4h'] = htf_candle['rsi']
        features['macd_diff_4h'] = htf_candle['macd_diff']

        # Symbol-Level Features
        features['liquidity_proxy'] = setup_candle['liquidity_proxy']

        # New Taker Volume Ratio
        if 'taker_buy_base_asset_volume' in setup_candle and setup_candle['volume'] > 0:
            features['taker_volume_ratio'] = setup_candle['taker_buy_base_asset_volume'] / setup_candle['volume']
        else:
            features['taker_volume_ratio'] = 0.5

        # New Volatility Regime
        vol = setup_candle['volatility_50']
        features['vol_regime_low'] = 1 if vol < vol_low_threshold else 0
        features['vol_regime_medium'] = 1 if vol_low_threshold <= vol < vol_high_threshold else 0
        features['vol_regime_high'] = 1 if vol >= vol_high_threshold else 0

        feature_list.append(features)

    if not feature_list:
        print("No features were generated.")
        return

    # --- Volatility Regime Calculation ---
    # First, collect all volatility values to determine global regimes
    all_volatility = pd.concat([data['15m']['volatility_50'] for data in all_raw_data.values()]).dropna()
    vol_low_threshold, vol_high_threshold = all_volatility.quantile([0.33, 0.67])
    
    # --- Feature List Generation ---
    feature_list = []
    for _, row in tqdm(labeled_df.iterrows(), total=labeled_df.shape[0], desc="Engineering Features"):
        # ... (existing setup candle lookup)
        
        # --- Calculate Features ---
        # ... (existing features)
        
        # New Taker Volume Ratio
        if setup_candle['volume'] > 0:
            features['taker_volume_ratio'] = setup_candle.get('taker_buy_base_asset_volume', 0) / setup_candle['volume']
        else:
            features['taker_volume_ratio'] = 0.5 # Neutral value

        # New Volatility Regime
        vol = setup_candle['volatility_50']
        if vol < vol_low_threshold:
            features['vol_regime'] = 'Low'
        elif vol < vol_high_threshold:
            features['vol_regime'] = 'Medium'
        else:
            features['vol_regime'] = 'High'

        feature_list.append(features)

    # Create the final feature DataFrame
    features_df = pd.DataFrame(feature_list)
    
    # --- Post-computation Feature Creation ---
    # One-hot encode volatility regime
    features_df = pd.get_dummies(features_df, columns=['vol_regime'], prefix='vol')

    # Interaction Feature: ATR x Body Ratio
    # This must be done after the main loop and after body_to_range_ratio is computed.
    if 'atr' in features_df.columns and 'body_to_range_ratio' in features_df.columns:
        features_df['atr_x_body_ratio'] = features_df['atr'] * features_df['body_to_range_ratio']

    print(f"Generated {len(features_df)} total feature vectors before cleaning.")
    
    # Drop any rows that still have NaNs for any reason (e.g. from lookback periods)
    features_df.dropna(inplace=True) 
    print(f"Feature vectors remaining after final dropna: {len(features_df)}")

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # --- Chunking Logic ---
    chunk_size = 25000
    num_chunks = int(np.ceil(len(features_df) / chunk_size))
    
    # Clean up old single file if it exists
    old_file_path = os.path.join(PROCESSED_DATA_DIR, "features.parquet")
    if os.path.exists(old_file_path):
        os.remove(old_file_path)

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk_df = features_df.iloc[start_idx:end_idx]
        
        save_path = os.path.join(PROCESSED_DATA_DIR, f"features_part_{i+1}.parquet")
        chunk_df.to_parquet(save_path)
        print(f"Saved chunk {i+1}/{num_chunks} to {save_path}")

    print(f"Feature engineering complete. Generated {len(features_df)} feature vectors in {num_chunks} chunks.")


def generate_features_for_live_setup(klines_df, setup_info, feature_columns, vol_thresholds=None):
    """
    Generates a feature vector for a single, live trade setup.
    `klines_df` should be a DataFrame of recent klines.
    `setup_info` is a dict with trade parameters.
    `feature_columns` is the list of columns the model was trained on.
    `vol_thresholds` is a tuple (low_thresh, high_thresh) loaded from the model artifact.
    """
    df = klines_df.copy()

    # --- Pre-calculate all indicators ---
    df['atr'] = calculate_atr(df)
    df['rsi'] = calculate_rsi(df)
    df['macd'], df['macd_signal'] = calculate_macd(df)
    df['adx'] = calculate_adx(df)
    df['bb_upper'], df['bb_lower'] = calculate_bollinger_bands(df)
    df['tenkan_sen'], df['kijun_sen'], df['senkou_span_a'], df['senkou_span_b'], df['chikou_span'] = calculate_ichimoku_cloud(df)
    for p in [20, 50, 100]:
        df[f'volatility_{p}'] = df['close'].pct_change().rolling(window=p).std()
    df['liquidity_proxy'] = df['volume'].rolling(window=96).mean()

    df.set_index(pd.to_datetime(df['timestamp'], unit='ms'), inplace=True)

    df_4h = df['close'].resample('4h').ohlc()
    df_4h['rsi'] = calculate_rsi(df_4h)
    df_4h['macd'], df_4h['macd_signal'] = calculate_macd(df_4h)
    df_4h['macd_diff'] = df_4h['macd'] - df_4h['macd_signal']
    
    setup_candle = df.iloc[-1]
    htf_candle = df_4h.asof(df.index[-1])

    features = {}

    # Price & Strategy Features
    swing_height = setup_info['swing_high_price'] - setup_info['swing_low_price']
    if swing_height > 0:
        features['golden_zone_ratio'] = (setup_info['entry_price'] - (setup_info['swing_low_price'] if setup_info['side'] == 'long' else setup_info['swing_high_price'])) / swing_height
        features['sl_pct'] = abs(setup_info['entry_price'] - setup_info['sl']) / swing_height
        features['tp1_pct'] = abs(setup_info['tp1'] - setup_info['entry_price']) / swing_height
        features['tp2_pct'] = abs(setup_info['tp2'] - setup_info['entry_price']) / swing_height
    else:
        features['golden_zone_ratio'], features['sl_pct'], features['tp1_pct'], features['tp2_pct'] = np.nan, np.nan, np.nan, np.nan

    # Price Action & Pattern Sequence
    candle_range = setup_candle['high'] - setup_candle['low']
    features['body_to_range_ratio'] = abs(setup_candle['open'] - setup_candle['close']) / candle_range if candle_range > 0 else 0
    features['upper_wick_ratio'] = (setup_candle['high'] - max(setup_candle['open'], setup_candle['close'])) / candle_range if candle_range > 0 else 0
    features['lower_wick_ratio'] = (min(setup_candle['open'], setup_candle['close']) - setup_candle['low']) / candle_range if candle_range > 0 else 0
    
    for n in range(1, 4):
        if len(df) > n:
            prev_candle = df.iloc[-1-n]
            prev_range = prev_candle['high'] - prev_candle['low']
            features[f'body_to_range_ratio_t-{n}'] = abs(prev_candle['open'] - prev_candle['close']) / prev_range if prev_range > 0 else 0
            features[f'upper_wick_ratio_t-{n}'] = (prev_candle['high'] - max(prev_candle['open'], prev_candle['close'])) / prev_range if prev_range > 0 else 0
            features[f'lower_wick_ratio_t-{n}'] = (min(prev_candle['open'], prev_candle['close']) - prev_candle['low']) / prev_range if prev_range > 0 else 0
        else:
            features[f'body_to_range_ratio_t-{n}'], features[f'upper_wick_ratio_t-{n}'], features[f'lower_wick_ratio_t-{n}'] = 0, 0, 0

    # Session
    session_str = get_session(setup_candle['timestamp'])
    features['is_asia'], features['is_london'], features['is_new_york'] = (1 if s in session_str else 0 for s in ['Asia', 'London', 'New_York'])

    # Context Features
    features['atr'], features['rsi'], features['macd_diff'], features['adx'] = setup_candle['atr'], setup_candle['rsi'], setup_candle['macd'] - setup_candle['macd_signal'], setup_candle['adx']
    for p in [20, 50, 100]:
        features[f'volatility_{p}'] = setup_candle[f'volatility_{p}']
    features['bb_upper'], features['bb_lower'] = setup_candle['bb_upper'], setup_candle['bb_lower']
    features['tenkan_sen'], features['kijun_sen'], features['senkou_span_a'], features['senkou_span_b'], features['chikou_span'] = setup_candle['tenkan_sen'], setup_candle['kijun_sen'], setup_candle['senkou_span_a'], setup_candle['senkou_span_b'], setup_candle['chikou_span']
    features['rsi_4h'], features['macd_diff_4h'] = htf_candle['rsi'], htf_candle['macd_diff']
    features['liquidity_proxy'] = setup_candle['liquidity_proxy']
    
    # New Features
    features['taker_volume_ratio'] = (setup_candle.get('taker_buy_base_asset_volume', 0) / setup_candle['volume']) if setup_candle['volume'] > 0 else 0.5
    
    if vol_thresholds:
        low_thresh, high_thresh = vol_thresholds
    else: # Fallback for testing or if not provided
        low_thresh, high_thresh = df['volatility_50'].quantile([0.33, 0.67])
        
    vol = setup_candle['volatility_50']
    features['vol_regime_low'] = 1 if vol < low_thresh else 0
    features['vol_regime_medium'] = 1 if low_thresh <= vol < high_thresh else 0
    features['vol_regime_high'] = 1 if vol >= high_thresh else 0

    features['atr_x_body_ratio'] = features['atr'] * features['body_to_range_ratio']
    features['side_long'] = 1 if setup_info['side'] == 'long' else 0

    feature_vector = pd.DataFrame([features])
    feature_vector = feature_vector[feature_columns]
    feature_vector.fillna(0, inplace=True)
    
    return feature_vector


import joblib
from tqdm.auto import tqdm
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import lightgbm as lgb
import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay


# --- Model Training (Step 4) ---

def _run_hyperparameter_search(X_train, y_train, X_val, y_val, model_type='lgbm'):
    """Helper to run Optuna study for a given model type."""
    
    def objective(trial):
        if model_type == 'lgbm':
            params = {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                'random_state': 42,
                'n_jobs': -1,
            }
            model = lgb.LGBMClassifier(**params)
        else: # xgb
            params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'random_state': 42,
                'n_jobs': -1,
            }
            model = xgb.XGBClassifier(**params)

        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        sample_weights = y_train.map(dict(zip(np.unique(y_train), class_weights)))
        
        model.fit(X_train, y_train, sample_weight=sample_weights)
        y_pred = model.predict(X_val)
        # We optimize for F1 score of the minority class (TP2)
        return f1_score(y_val, y_pred, labels=[2], average='macro')

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50) # Number of trials can be adjusted
    print(f"Best {model_type.upper()} params: {study.best_params}")
    return study.best_params

def train_model():
    """Loads features, runs hyperparameter search, trains an ensemble, calibrates, and saves."""
    print("Starting model training...")
    try:
        feature_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, "features_part_*.parquet"))
        if not feature_files: raise FileNotFoundError("No feature chunk files found.")
        
        df_list = [pd.read_parquet(f) for f in feature_files]
        features_df = pd.concat(df_list, ignore_index=True)
        features_df.sort_values('timestamp', inplace=True)
        features_df.reset_index(drop=True, inplace=True)
        print(f"Loaded {len(features_df)} features from {len(feature_files)} chunks.")
        
    except FileNotFoundError:
        print("Error: No feature chunk files found. Please run engineer_features() first.")
        return

    if features_df.empty:
        print("Feature set is empty. Cannot train model.")
        return

    # --- Data Preparation ---
    feature_columns = [col for col in features_df.columns if col not in ['symbol', 'timestamp', 'side', 'entry_price', 'sl', 'tp1', 'tp2', 'label']]
    X = features_df[feature_columns]
    y = features_df['label']

    # Time-based Split: 60% train, 20% validation (for Optuna), 20% test
    n_samples = len(X)
    train_end = int(n_samples * 0.6)
    val_end = int(n_samples * 0.8)
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    # Combine train and validation for final model training
    X_train_full, y_train_full = X.iloc[:val_end], y.iloc[:val_end]
    
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    # --- Hyperparameter Search ---
    print("\n--- Running Hyperparameter Search for LightGBM ---")
    best_lgbm_params = _run_hyperparameter_search(X_train, y_train, X_val, y_val, 'lgbm')
    
    print("\n--- Running Hyperparameter Search for XGBoost ---")
    best_xgb_params = _run_hyperparameter_search(X_train, y_train, X_val, y_val, 'xgb')

    # --- Final Model Training ---
    print("\n--- Training Final Models with Best Parameters ---")
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_full), y=y_train_full)
    sample_weights = y_train_full.map(dict(zip(np.unique(y_train_full), class_weights)))

    lgbm_final = lgb.LGBMClassifier(**best_lgbm_params, objective='multiclass', num_class=3, random_state=42, n_jobs=-1)
    lgbm_final.fit(X_train_full, y_train_full, sample_weight=sample_weights)

    xgb_final = xgb.XGBClassifier(**best_xgb_params, objective='multi:softprob', num_class=3, random_state=42, n_jobs=-1)
    xgb_final.fit(X_train_full, y_train_full, sample_weight=sample_weights)

    # --- Probability Calibration (on the test set for simplicity here, could use another split) ---
    print("\n--- Calibrating Models ---")
    calibrated_lgbm = CalibratedClassifierCV(lgbm_final, method='isotonic', cv='prefit').fit(X_test, y_test)
    calibrated_xgb = CalibratedClassifierCV(xgb_final, method='isotonic', cv='prefit').fit(X_test, y_test)

    # --- Evaluation ---
    print("\n--- Evaluating Models on Test Set ---")
    evaluate_ensemble(calibrated_lgbm, calibrated_xgb, X_test, y_test)

    # --- Persistence ---
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "trading_model_v2.joblib")
    
    model_artifact = {
        'model_version': '2.0.0',
        'training_timestamp_utc': datetime.utcnow().isoformat(),
        'lgbm_model': calibrated_lgbm,
        'xgb_model': calibrated_xgb,
        'feature_columns': feature_columns,
        'vol_thresholds': (features_df['volatility_50'].quantile(0.33), features_df['volatility_50'].quantile(0.67))
    }
    joblib.dump(model_artifact, model_path)
    print(f"\nEnsemble model artifact saved to {model_path}")

    # --- Post-Training Validation ---
    plot_and_save_calibration_curve(model_artifact, X_test, y_test)
    smoke_test_model_artifact(model_path, X_test.head(5))

def evaluate_ensemble(lgbm_model, xgb_model, X_test, y_test):
    """Evaluates the performance of each model and the ensemble."""
    lgbm_proba = lgbm_model.predict_proba(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)
    ensemble_proba = (lgbm_proba + xgb_proba) / 2
    ensemble_pred = np.argmax(ensemble_proba, axis=1)

    print("\n--- LGBM Classification Report ---")
    print(classification_report(y_test, lgbm_model.predict(X_test), target_names=['Loss', 'TP1', 'TP2'], zero_division=0))
    
    print("\n--- XGBoost Classification Report ---")
    print(classification_report(y_test, xgb_model.predict(X_test), target_names=['Loss', 'TP1', 'TP2'], zero_division=0))

    print("\n--- Ensemble Classification Report ---")
    print(classification_report(y_test, ensemble_pred, target_names=['Loss', 'TP1', 'TP2'], zero_division=0))
    print("\nEnsemble Confusion Matrix:")
    print(confusion_matrix(y_test, ensemble_pred))

def smoke_test_model_artifact(model_path, X_sample):
    """Loads a saved model artifact and runs a quick prediction."""
    print("\n--- Running Model Artifact Smoke Test ---")
    try:
        model_data = joblib.load(model_path)
        lgbm = model_data['lgbm_model']
        xgb = model_data['xgb_model']
        
        lgbm_preds = lgbm.predict_proba(X_sample)
        xgb_preds = xgb.predict_proba(X_sample)
        
        assert lgbm_preds.shape == (5, 3)
        assert xgb_preds.shape == (5, 3)
        print("Smoke test PASSED. Models loaded and predicted successfully.")
    except Exception as e:
        print(f"Smoke test FAILED: {e}")
        raise

def plot_and_save_calibration_curve(model_artifact, X_test, y_test):
    """Generates and saves calibration curves for the ensemble."""
    print("\nGenerating calibration curves...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    models = {
        "LGBM": model_artifact['lgbm_model'],
        "XGBoost": model_artifact['xgb_model']
    }
    
    for model_name, model in models.items():
        for i, class_name in enumerate(['Loss', 'TP1', 'TP2']):
            CalibrationDisplay.from_estimator(
                model, X_test, y_test, n_bins=10, class_to_plot=i,
                name=f'{model_name} - {class_name}', ax=ax, strategy='uniform'
            )
    
    ax.set_title("Calibration Curve for Each Model and Class")
    plt.grid(True)
    plt.legend(loc="upper left")
    
    timestamp = model_artifact['training_timestamp_utc'].replace(':', '-')
    plot_path = os.path.join(MODELS_DIR, f"calibration_curve_ensemble_{timestamp}.png")
    plt.savefig(plot_path)
    print(f"Calibration curves saved to {plot_path}")
    plt.close()


def run_pipeline():
    """
    Runs the full data pipeline from acquisition to model training,
    skipping steps if the data already exists.
    """
    features_path = os.path.join(PROCESSED_DATA_DIR, "features.parquet")
    labeled_path = os.path.join(PROCESSED_DATA_DIR, "labeled_trades.parquet")

    if os.path.exists(features_path):
        print("--- Found existing features.parquet. Skipping to Step 4: Model Training ---")
        train_model()
    elif os.path.exists(labeled_path):
        print("--- Found existing labeled_trades.parquet. Skipping to Step 3: Feature Engineering ---")
        engineer_features()
        print("\n--- Running Step 4: Model Training ---")
        train_model()
    else:
        print("--- Running full pipeline from scratch ---")
        print("--- Running Step 1: Data Acquisition ---")
        fetch_initial_data()
        
        print("\n--- Running Step 2: Preprocessing and Labeling ---")
        main_preprocess()

        print("\n--- Running Step 3: Feature Engineering ---")
        engineer_features()

        print("\n--- Running Step 4: Model Training ---")
        train_model()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run the ML pipeline for the trading bot.")
    parser.add_argument(
        'step', 
        nargs='?', 
        default='all', 
        choices=['fetch', 'label', 'features', 'train', 'all'],
        help="The pipeline step to run: 'fetch', 'label', 'features', 'train', or 'all' (default)."
    )
    args = parser.parse_args()

    if args.step == 'all':
        run_pipeline()
    elif args.step == 'fetch':
        fetch_initial_data()
    elif args.step == 'label':
        main_preprocess()
    elif args.step == 'features':
        engineer_features()
    elif args.step == 'train':
        train_model()
