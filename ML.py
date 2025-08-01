import os
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
from binance.client import Client
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

def fetch_historical(client, symbol, interval=Client.KLINE_INTERVAL_15MINUTE, total_limit=20000):
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

    client = Client(keys.api_mainnet, keys.secret_mainnet)
    
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
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    for col in ['open', 'high', 'low', 'close', 'volume']:
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

    for i in range(LOOKBACK_CANDLES, len(df) - TRADE_EXPIRY_BARS):
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
            df['rolling_volatility'] = df['close'].pct_change().rolling(window=20).std()
            df['liquidity_proxy'] = df['volume'].rolling(window=96).mean()
            df.set_index('timestamp', inplace=True)
            all_raw_data[symbol] = df

    feature_list = []
    for _, row in labeled_df.iterrows():
        symbol = row['symbol']
        timestamp = row['timestamp']

        if symbol not in all_raw_data:
            continue

        raw_df = all_raw_data[symbol]
        try:
            # Find the exact setup candle in the raw data
            setup_candle = raw_df.loc[timestamp]
        except KeyError:
            # This can happen if the labeled setup is too close to the edge of the raw data
            continue

        # --- Calculate Features ---
        features = {}
        
        # Original data to keep for model analysis and execution
        features['symbol'] = symbol
        features['timestamp'] = timestamp
        features['side'] = row['side']
        features['entry_price'] = row['entry_price']
        features['sl'] = row['sl']
        features['tp1'] = row['tp1']
        features['tp2'] = row['tp2']
        features['label'] = row['label']

        # Price & Strategy Features
        swing_height = row['swing_high_price'] - row['swing_low_price']
        if swing_height > 0:
            if row['side'] == 'long':
                features['golden_zone_ratio'] = (row['entry_price'] - row['swing_low_price']) / swing_height
            else: # short
                features['golden_zone_ratio'] = (row['swing_high_price'] - row['entry_price']) / swing_height
            
            features['sl_pct'] = abs(row['entry_price'] - row['sl']) / swing_height
            features['tp1_pct'] = abs(row['tp1'] - row['entry_price']) / swing_height
            features['tp2_pct'] = abs(row['tp2'] - row['entry_price']) / swing_height
        else:
            features['golden_zone_ratio'] = np.nan
            features['sl_pct'] = np.nan
            features['tp1_pct'] = np.nan
            features['tp2_pct'] = np.nan

        # Session One-Hot Encoding
        session_str = setup_candle['session']
        features['is_asia'] = 1 if 'Asia' in session_str else 0
        features['is_london'] = 1 if 'London' in session_str else 0
        features['is_new_york'] = 1 if 'New_York' in session_str else 0
        
        # Market Context Features (from pre-calculated values)
        features['atr'] = setup_candle['atr']
        features['rsi'] = setup_candle['rsi']
        features['macd_diff'] = setup_candle['macd'] - setup_candle['macd_signal']
        features['rolling_volatility'] = setup_candle['rolling_volatility']
        
        # Symbol-Level Features
        features['liquidity_proxy'] = setup_candle['liquidity_proxy']

        feature_list.append(features)

    if not feature_list:
        print("No features were generated.")
        return

    # Create and save the final feature DataFrame
    features_df = pd.DataFrame(feature_list)
    print(f"Generated {len(features_df)} total feature vectors before cleaning.")
    
    # Drop any rows that still have NaNs for any reason (e.g. from lookback periods)
    features_df.dropna(inplace=True) 
    print(f"Feature vectors remaining after final dropna: {len(features_df)}")

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    save_path = os.path.join(PROCESSED_DATA_DIR, "features.parquet")
    features_df.to_parquet(save_path)
    print(f"Feature engineering complete. Features saved to {save_path}")
    print(f"Generated {len(features_df)} feature vectors.")


def generate_features_for_live_setup(klines_df, setup_info, feature_columns):
    """
    Generates a feature vector for a single, live trade setup.
    `klines_df` should be a DataFrame of recent klines.
    `setup_info` is a dict with trade parameters.
    `feature_columns` is the list of columns the model was trained on.
    """
    # Use a copy to avoid modifying the original DataFrame
    df = klines_df.copy()

    # --- Pre-calculate indicators on the DataFrame ---
    df['atr'] = calculate_atr(df)
    df['rsi'] = calculate_rsi(df)
    df['macd'], df['macd_signal'] = calculate_macd(df)
    df['rolling_volatility'] = df['close'].pct_change().rolling(window=20).std()
    df['liquidity_proxy'] = df['volume'].rolling(window=96).mean()

    # Get the most recent candle for context features
    # The timestamp of the setup is the last candle in the klines_df
    setup_candle_timestamp = df.iloc[-1]['timestamp']
    setup_candle = df.iloc[-1]

    # --- Calculate Features ---
    features = {}

    # Price & Strategy Features
    swing_height = setup_info['swing_high_price'] - setup_info['swing_low_price']
    if swing_height > 0:
        if setup_info['side'] == 'long':
            features['golden_zone_ratio'] = (setup_info['entry_price'] - setup_info['swing_low_price']) / swing_height
        else: # short
            features['golden_zone_ratio'] = (setup_info['swing_high_price'] - setup_info['entry_price']) / swing_height
        
        features['sl_pct'] = abs(setup_info['entry_price'] - setup_info['sl']) / swing_height
        features['tp1_pct'] = abs(setup_info['tp1'] - setup_info['entry_price']) / swing_height
        features['tp2_pct'] = abs(setup_info['tp2'] - setup_info['entry_price']) / swing_height
    else:
        features['golden_zone_ratio'] = np.nan
        features['sl_pct'] = np.nan
        features['tp1_pct'] = np.nan
        features['tp2_pct'] = np.nan

    # Session One-Hot Encoding
    session_str = get_session(setup_candle_timestamp)
    features['is_asia'] = 1 if 'Asia' in session_str else 0
    features['is_london'] = 1 if 'London' in session_str else 0
    features['is_new_york'] = 1 if 'New_York' in session_str else 0

    # Market Context Features
    features['atr'] = setup_candle['atr']
    features['rsi'] = setup_candle['rsi']
    features['macd_diff'] = setup_candle['macd'] - setup_candle['macd_signal']
    features['rolling_volatility'] = setup_candle['rolling_volatility']
        
    # Symbol-Level Features
    features['liquidity_proxy'] = setup_candle['liquidity_proxy']

    # Side feature
    features['side_long'] = 1 if setup_info['side'] == 'long' else 0

    # Create a DataFrame with the correct column order
    feature_vector = pd.DataFrame([features], columns=feature_columns)
    
    return feature_vector


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib

from sklearn.model_selection import GridSearchCV

# --- Model Training (Step 4) ---
def train_model():
    """Loads features, trains a model with hyperparameter tuning, and saves it."""
    print("Starting model training...")
    try:
        features_df = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "features.parquet"))
    except FileNotFoundError:
        print("Error: features.parquet not found. Please run engineer_features() first.")
        return

    if features_df.empty:
        print("Feature set is empty. Cannot train model.")
        return

    # --- Data Preparation ---
    feature_columns = [
        'golden_zone_ratio', 'sl_pct', 'tp1_pct', 'tp2_pct',
        'is_asia', 'is_london', 'is_new_york',
        'atr', 'rsi', 'macd_diff', 'rolling_volatility',
        'liquidity_proxy'
    ]
    features_df['side_long'] = (features_df['side'] == 'long').astype(int)
    feature_columns.append('side_long')

    X = features_df[feature_columns]
    y = features_df['label']

    # --- Time-based Split ---
    sorted_indices = features_df['timestamp'].argsort()
    X = X.iloc[sorted_indices]
    y = y.iloc[sorted_indices]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # --- Handle Class Imbalance ---
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weights_dict = {i : class_weights[i] for i, w in enumerate(class_weights)}
    print(f"Calculated class weights: {weights_dict}")
    sample_weights = y_train.map(weights_dict)

    # --- Hyperparameter Tuning with GridSearchCV ---
    # Define the parameter grid to search
    # This is a small grid for demonstration; a real search would be larger.
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.7, 0.8],
    }

    # Initialize XGBoost Classifier
    # n_jobs=-1 uses all available CPU cores for training
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_jobs=-1  # Enable multithreading
    )

    # Initialize GridSearchCV
    # cv=3 uses 3-fold cross-validation. TimeSeriesSplit is better for time-series data.
    # We use n_jobs=-1 here as well to parallelize the grid search process.
    grid_search = GridSearchCV(
        estimator=model, 
        param_grid=param_grid, 
        scoring='f1_weighted', 
        cv=3, 
        verbose=1,
        n_jobs=-1 # Use all available cores for grid search
    )

    print("Starting hyperparameter search with GridSearchCV...")
    grid_search.fit(X_train, y_train, sample_weight=sample_weights)

    print(f"\nBest parameters found: {grid_search.best_params_}")
    print(f"Best f1_weighted score: {grid_search.best_score_}")

    # Get the best model
    best_model = grid_search.best_estimator_

    # --- Evaluation ---
    print("\nModel Evaluation on Test Set with Best Estimator:")
    y_pred = best_model.predict(X_test)
    
    print(classification_report(y_test, y_pred, target_names=['Loss (0)', 'TP1 Win (1)', 'TP2 Win (2)']))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # --- Persistence ---
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "trading_model.joblib")
    
    # Save the best model and the feature columns together
    joblib.dump({
        'model': best_model,
        'feature_columns': feature_columns
    }, model_path)

    print(f"\nModel and metadata saved to {model_path}")


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
