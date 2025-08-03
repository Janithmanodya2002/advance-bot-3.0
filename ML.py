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
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import concurrent.futures
from multiprocessing import Pool
from functools import partial
import argparse
from sklearn.utils.class_weight import compute_class_weight

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- Configuration ---
DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
MODELS_DIR = "models"
SYMBOLS_FILE = "symbols.csv"
SWING_WINDOW = 5
LOOKBACK_CANDLES = 100
TRADE_EXPIRY_BARS = 96
SEQUENCE_LENGTH = 60
PARQUET_ROW_GROUP_SIZE = 100_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SESSIONS = {
    "Asia": (0, 8),
    "London": (7, 15),
    "New_York": (12, 20)
}

# --- Data Pipeline Functions ---

def fetch_historical_for_symbol(symbol, total_limit=20000):
    client = BinanceClient(keys.api_mainnet, keys.secret_mainnet)
    print(f"Fetching data for {symbol}...")
    all_klines = []
    limit = 1000
    # To prevent infinite loops for symbols with no data
    max_attempts = 5
    attempts = 0
    while len(all_klines) < total_limit and attempts < max_attempts:
        try:
            end_time = all_klines[0][0] if all_klines else None
            klines = client.get_historical_klines(symbol=symbol, interval=BinanceClient.KLINE_INTERVAL_15MINUTE, limit=min(limit, total_limit - len(all_klines)), end_str=end_time)
            if not klines:
                break
            all_klines = klines + all_klines
        except Exception as e:
            print(f"Error fetching klines for {symbol}: {e}")
            return symbol, None # Return symbol to identify failure
        attempts += 1
    return symbol, all_klines

def fetch_initial_data(sessions_dict, quick_test=False):
    print("Starting parallel initial data acquisition...")
    try:
        symbols_df = pd.read_csv(SYMBOLS_FILE, header=None)
        symbols = symbols_df[0].tolist()
    except FileNotFoundError:
        print(f"Error: {SYMBOLS_FILE} not found.")
        return

    if quick_test:
        symbols = symbols[:3] # Use a small subset of symbols for a quick test
        print(f"--- QUICK TEST MODE: Using symbols {symbols} ---")
        total_limit = 1000 # Fetch fewer candles
    else:
        total_limit = 20000

    # Using ProcessPoolExecutor to avoid GIL issues with CPU-bound processing later
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Create a future for each symbol download
        future_to_symbol = {executor.submit(fetch_historical_for_symbol, symbol, total_limit): symbol for symbol in symbols}
        
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol, klines = future.result()
            if klines:
                print(f"  - Successfully fetched {len(klines)} candles for {symbol}")
                filename = f"initial_{len(klines)}.parquet"
                # Note: process_and_save_kline_data is called in the main process
                process_and_save_kline_data(klines, symbol, filename, sessions_dict)
            else:
                print(f"  - Failed to fetch data for {symbol}")

    print("\nInitial data acquisition complete.")

def get_session(timestamps_ms, sessions_dict):
    """Vectorized function to determine the trading session for a series of timestamps."""
    timestamps = pd.to_datetime(timestamps_ms, unit='ms', utc=True)
    hours = timestamps.dt.hour
    
    # Initialize a session column
    session_series = pd.Series("Inactive", index=timestamps.index, dtype='object')
    
    # Build session strings based on vectorized boolean conditions
    session_parts = {name: pd.Series("", index=timestamps.index) for name in sessions_dict.keys()}
    
    for name, (start, end) in sessions_dict.items():
        if start <= end: # Normal case (e.g., 7-15)
            mask = (hours >= start) & (hours <= end)
        else: # Overnight case (e.g., London close to Asia open)
            mask = (hours >= start) | (hours <= end)
        session_series[mask] = name

    # For overlapping sessions, we need a more careful construction
    is_asia = (hours >= SESSIONS['Asia'][0]) & (hours < SESSIONS['Asia'][1])
    is_london = (hours >= SESSIONS['London'][0]) & (hours < SESSIONS['London'][1])
    is_ny = (hours >= SESSIONS['New_York'][0]) & (hours < SESSIONS['New_York'][1])

    # Create session strings
    sessions = pd.Series("", index=timestamps.index)
    sessions[is_asia] += "Asia,"
    sessions[is_london] += "London,"
    sessions[is_ny] += "New_York,"
    sessions = sessions.str.rstrip(',')
    sessions[sessions == ""] = "Inactive"
    
    return sessions

def process_and_save_kline_data(klines, symbol, filename, sessions_dict):
    if not klines: return
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume']]
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume']
    for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
    df['session'] = get_session(df['timestamp'], sessions_dict)
    save_data_to_parquet(df, symbol, filename)

def save_data_to_parquet(df, symbol, filename):
    if df is None or df.empty: return
    symbol_dir = os.path.join(DATA_DIR, symbol)
    os.makedirs(symbol_dir, exist_ok=True)
    file_path = os.path.join(symbol_dir, filename)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, file_path, row_group_size=PARQUET_ROW_GROUP_SIZE)

def load_raw_data_for_symbol(symbol):
    symbol_dir = os.path.join(DATA_DIR, symbol)
    files = glob.glob(os.path.join(symbol_dir, "*.parquet"))
    if not files: return None
    df = pd.concat([pd.read_parquet(f) for f in files])
    df.sort_values('timestamp', inplace=True)
    df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def get_swing_points_df(df, window=5):
    """Vectorized calculation of swing points."""
    # Ensure window is odd for centering
    rolling_window_size = 2 * window + 1
    
    # Find local max/min using a rolling window
    local_max = df['high'].rolling(window=rolling_window_size, center=True).max()
    local_min = df['low'].rolling(window=rolling_window_size, center=True).min()
    
    # A point is a swing high if it's the absolute highest point in its window
    is_swing_high = (df['high'] == local_max)
    # A point is a swing low if it's the absolute lowest point in its window
    is_swing_low = (df['low'] == local_min)
    
    # Remove consecutive swing points by keeping only the first occurrence
    is_swing_high = is_swing_high & (is_swing_high.shift(1) == False)
    is_swing_low = is_swing_low & (is_swing_low.shift(1) == False)

    # Extract the swing points
    swing_highs = df[is_swing_high][['timestamp', 'high']].rename(columns={'high': 'price'})
    swing_lows = df[is_swing_low][['timestamp', 'low']].rename(columns={'low': 'price'})
    
    return swing_highs, swing_lows

def get_trend_df(swing_highs, swing_lows):
    if len(swing_highs) < 2 or len(swing_lows) < 2: return "undetermined"
    if swing_highs['price'].iloc[-1] > swing_highs['price'].iloc[-2] and swing_lows['price'].iloc[-1] > swing_lows['price'].iloc[-2]: return "uptrend"
    if swing_highs['price'].iloc[-1] < swing_highs['price'].iloc[-2] and swing_lows['price'].iloc[-1] < swing_lows['price'].iloc[-2]: return "downtrend"
    return "undetermined"

def get_fib_retracement(p1, p2, trend):
    price_range = abs(p1 - p2)
    if trend == "downtrend":
        golden_zone_start, golden_zone_end = p1 - (price_range * 0.5), p1 - (price_range * 0.618)
    else:
        golden_zone_start, golden_zone_end = p1 + (price_range * 0.5), p1 + (price_range * 0.618)
    return (golden_zone_start + golden_zone_end) / 2

def label_trade(df, entry_idx, sl_price, tp1_price, tp2_price, side):
    future_candles = df.iloc[entry_idx + 1 : entry_idx + 1 + TRADE_EXPIRY_BARS]
    tp1_hit = False
    for _, candle in future_candles.iterrows():
        if side == 'long':
            if candle['low'] <= sl_price: return 1 if tp1_hit else 0
            if candle['high'] >= tp2_price: return 2
            if candle['high'] >= tp1_price: tp1_hit = True
        elif side == 'short':
            if candle['high'] >= sl_price: return 1 if tp1_hit else 0
            if candle['low'] <= tp2_price: return 2
            if candle['low'] <= tp1_price: tp1_hit = True
    if tp1_hit: return 1
    return -1

def find_and_label_setups(df):
    labeled_setups = []
    
    # 1. Pre-calculate all swing points for the entire DataFrame at once
    all_swing_highs, all_swing_lows = get_swing_points_df(df, SWING_WINDOW)

    for i in tqdm(range(LOOKBACK_CANDLES, len(df) - TRADE_EXPIRY_BARS), desc="Finding Setups"):
        if df['session'].iloc[i] == 'Inactive':
            continue
            
        # Define the lookback window
        window_start_time = df['timestamp'].iloc[i - LOOKBACK_CANDLES]
        window_end_time = df['timestamp'].iloc[i]
        
        # 2. Filter pre-calculated swing points within the current lookback window
        swing_highs = all_swing_highs[all_swing_highs['timestamp'].between(window_start_time, window_end_time)]
        swing_lows = all_swing_lows[all_swing_lows['timestamp'].between(window_start_time, window_end_time)]

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            continue
            
        trend = get_trend_df(swing_highs, swing_lows)
        setup = None
        
        if trend == 'downtrend':
            last_swing_high_price, last_swing_low_price = swing_highs.iloc[-1]['price'], swing_lows.iloc[-1]['price']
            if last_swing_low_price >= swing_lows.iloc[-2]['price']: continue
            entry_price = get_fib_retracement(last_swing_high_price, last_swing_low_price, trend)
            sl = last_swing_high_price
            tp1 = entry_price - (sl - entry_price)
            tp2 = entry_price - (sl - entry_price) * 2
            
            # Find entry candle
            future_candles = df.iloc[i : i + TRADE_EXPIRY_BARS]
            entry_mask = future_candles['high'] >= entry_price
            if not entry_mask.any(): continue
            entry_candle_idx = future_candles.index[entry_mask][0]

            label = label_trade(df, entry_candle_idx, sl, tp1, tp2, 'short')
            if label != -1:
                setup = {'timestamp': df['timestamp'].iloc[i], 'side': 'short', 'entry_price': entry_price, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'label': label, 'swing_high_price': last_swing_high_price, 'swing_low_price': last_swing_low_price}

        elif trend == 'uptrend':
            last_swing_high_price, last_swing_low_price = swing_highs.iloc[-1]['price'], swing_lows.iloc[-1]['price']
            if last_swing_high_price <= swing_highs.iloc[-2]['price']: continue
            entry_price = get_fib_retracement(last_swing_low_price, last_swing_high_price, trend)
            sl = last_swing_low_price
            tp1 = entry_price + (entry_price - sl)
            tp2 = entry_price + (entry_price - sl) * 2
            
            # Find entry candle
            future_candles = df.iloc[i : i + TRADE_EXPIRY_BARS]
            entry_mask = future_candles['low'] <= entry_price
            if not entry_mask.any(): continue
            entry_candle_idx = future_candles.index[entry_mask][0]

            label = label_trade(df, entry_candle_idx, sl, tp1, tp2, 'long')
            if label != -1:
                setup = {'timestamp': df['timestamp'].iloc[i], 'side': 'long', 'entry_price': entry_price, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'label': label, 'swing_high_price': last_swing_high_price, 'swing_low_price': last_swing_low_price}
        
        if setup:
            labeled_setups.append(setup)
            
    return pd.DataFrame(labeled_setups)

def process_symbol_for_labeling(symbol, sessions_dict):
    """Loads, processes, and labels data for a single symbol."""
    df = load_raw_data_for_symbol(symbol)
    if df is None or df.empty:
        return None
    
    # Vectorized session calculation
    df['session'] = get_session(df['timestamp'], sessions_dict)
    
    # Labeling
    labeled_setups = find_and_label_setups(df)
    
    if not labeled_setups.empty:
        labeled_setups['symbol'] = symbol
        return labeled_setups
    return None

def main_preprocess(sessions_dict, quick_test=False):
    print("Starting robust preprocessing and labeling with chunking...")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    try:
        symbols = pd.read_csv(SYMBOLS_FILE, header=None)[0].tolist()
    except FileNotFoundError:
        print(f"Error: {SYMBOLS_FILE} not found."); return

    if quick_test:
        symbols = symbols[:3]
        print(f"--- QUICK TEST MODE: Processing symbols {symbols} ---")

    # Better chunking and pool reuse
    chunk_size = os.cpu_count() or 4
    worker_func = partial(process_symbol_for_labeling, sessions_dict=sessions_dict)
    symbol_chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]
    
    # Create a single pool to be reused
    with Pool(processes=chunk_size) as pool:
        for i, chunk in enumerate(symbol_chunks):
            chunk_file_name = f"labeled_chunk_{i+1}_quick.parquet" if quick_test else f"labeled_chunk_{i+1}.parquet"
            chunk_file_path = os.path.join(PROCESSED_DATA_DIR, chunk_file_name)

            if os.path.exists(chunk_file_path):
                print(f"--- Chunk {i+1}/{len(symbol_chunks)} already processed. Skipping. ---")
                continue
                
            print(f"--- Processing Chunk {i+1}/{len(symbol_chunks)}: {chunk} ---")
            results = pool.map(worker_func, chunk)
            
            all_labeled_data = [res for res in results if res is not None]
            
            if all_labeled_data:
                chunk_df = pd.concat(all_labeled_data, ignore_index=True)
                chunk_df.to_parquet(chunk_file_path)
                print(f"Saved chunk {i+1} to {chunk_file_path}")
            else:
                print(f"No labeled data generated for chunk {i+1}.")

    print("\n--- Combining all processed chunks ---")
    file_pattern = "labeled_chunk_*_quick.parquet" if quick_test else "labeled_chunk_*.parquet"
    all_chunk_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, file_pattern))
    
    if not all_chunk_files:
        print("No chunk files found to combine. Preprocessing did not generate any labeled data.")
        final_save_path = os.path.join(PROCESSED_DATA_DIR, "labeled_trades_quick.parquet" if quick_test else "labeled_trades.parquet")
        pd.DataFrame().to_parquet(final_save_path) # Create empty placeholder
        return
        
    combined_df = pd.concat([pd.read_parquet(f) for f in all_chunk_files], ignore_index=True)
    final_save_path = os.path.join(PROCESSED_DATA_DIR, "labeled_trades_quick.parquet" if quick_test else "labeled_trades.parquet")
    combined_df.to_parquet(final_save_path)
    print(f"All chunks combined. Final labeled data saved to {final_save_path}")

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return (100 - (100 / (1 + rs))).fillna(50)

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = df['close'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = df['close'].ewm(span=slow_period, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal, macd - signal

def calculate_bollinger_bands(df, window=20, num_std_dev=2):
    rolling_mean = df['close'].rolling(window=window).mean()
    rolling_std = df['close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band

from sklearn.ensemble import RandomForestClassifier

def create_sequences_and_features(labeled_setups_df):
    print("Starting sequence and feature generation...")
    all_raw_data = {}
    
    # Define feature names for later use
    sequence_feature_names = [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'atr', 'macd_hist', 'bb_width', 'bb_pos'
    ]

    for symbol in tqdm(labeled_setups_df['symbol'].unique(), desc="Loading and Pre-calculating Features"):
        df = load_raw_data_for_symbol(symbol)
        if df is not None:
            # 1. Early Column Pruning: Keep only what's necessary
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df = df[required_cols]

            # 2. Bulk Indicator Computation using .assign()
            df = df.assign(
                atr = calculate_atr(df),
                rsi = calculate_rsi(df),
                macd = calculate_macd(df)[0],
                macd_signal = calculate_macd(df)[1],
                macd_hist = calculate_macd(df)[2],
                bb_upper = calculate_bollinger_bands(df)[0],
                bb_lower = calculate_bollinger_bands(df)[1]
            )
            df.set_index('timestamp', inplace=True)
            all_raw_data[symbol] = df
            
    sequence_list, static_list, label_list = [], [], []
    for _, row in tqdm(labeled_setups_df.iterrows(), total=labeled_setups_df.shape[0], desc="Generating Sequences"):
        raw_df = all_raw_data.get(row['symbol'])
        if raw_df is None: continue
        try:
            setup_idx_loc = raw_df.index.get_loc(row['timestamp'])
        except KeyError: continue
        if setup_idx_loc < SEQUENCE_LENGTH: continue
        
        sequence_df = raw_df.iloc[setup_idx_loc - SEQUENCE_LENGTH : setup_idx_loc].copy()
        last_close = sequence_df['close'].iloc[-1]
        
        # --- Feature Engineering and Normalization ---
        seq_features = pd.DataFrame(index=sequence_df.index)
        
        # Price and Volume Features (Normalized)
        seq_features['open'] = sequence_df['open'] / last_close - 1
        seq_features['high'] = sequence_df['high'] / last_close - 1
        seq_features['low'] = sequence_df['low'] / last_close - 1
        seq_features['close'] = sequence_df['close'] / last_close - 1
        seq_features['volume'] = (sequence_df['volume'] - sequence_df['volume'].mean()) / (sequence_df['volume'].std() + 1e-8)
        
        # Indicator Features (Normalized)
        seq_features['rsi'] = (sequence_df['rsi'] - 50) / 50
        seq_features['atr'] = sequence_df['atr'] / last_close
        seq_features['macd_hist'] = sequence_df['macd_hist'] / last_close # Normalize by price
        seq_features['bb_width'] = (sequence_df['bb_upper'] - sequence_df['bb_lower']) / last_close # Normalize by price
        # Calculate bb_pos safely
        bb_range = sequence_df['bb_upper'] - sequence_df['bb_lower']
        seq_features['bb_pos'] = (sequence_df['close'] - sequence_df['bb_lower']) / (bb_range + 1e-8)
        
        # Ensure order is consistent
        sequence_list.append(seq_features[sequence_feature_names].values)
        
        static_features = {}
        static_features['side_long'] = 1 if row['side'] == 'long' else 0
        static_features['side_short'] = 1 if row['side'] == 'short' else 0
        swing_height = row['swing_high_price'] - row['swing_low_price']
        if swing_height > 0:
            if row['side'] == 'long':
                static_features['golden_zone_ratio'] = (row['entry_price'] - row['swing_low_price']) / swing_height
            else:
                static_features['golden_zone_ratio'] = (row['swing_high_price'] - row['entry_price']) / swing_height
            static_features['sl_pct_of_swing'] = abs(row['entry_price'] - row['sl']) / swing_height
        else:
            static_features['golden_zone_ratio'], static_features['sl_pct_of_swing'] = 0, 0
        
        static_list.append(list(static_features.values()))
        label_list.append(row['label'])
        
    sequences_np, static_np, labels_np = np.array(sequence_list), np.array(static_list), np.array(label_list)
    return sequences_np, static_np, labels_np, sequence_feature_names

def run_feature_importance_check(sequences, labels, feature_names):
    """Trains a RandomForest to check the importance of different sequence features."""
    print("\n--- Running Feature Importance Check ---")
    if len(sequences) == 0:
        print("Cannot run feature importance check with no sequences.")
        return

    # The model expects 2D data, so we can average features across the sequence length
    # This gives a good approximation of feature importance.
    num_samples, _, num_features = sequences.shape
    X_2d = sequences.mean(axis=1)
    
    # Subsample if the dataset is very large to speed up the check
    sample_size = min(len(X_2d), 25000)
    indices = np.random.choice(np.arange(len(X_2d)), sample_size, replace=False)
    X_sample, y_sample = X_2d[indices], labels[indices]

    print(f"Training RandomForest on a sample of {sample_size} setups...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_sample, y_sample)
    
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("Top Sequence Feature Importances:")
    print(importance_df)
    print("----------------------------------------")


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class FocalLoss(nn.Module):
    """Implementation of Focal Loss."""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LSTMModel(nn.Module):
    def __init__(self, sequence_input_size, static_input_size, lstm_hidden_size=50, num_classes=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(sequence_input_size, lstm_hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(lstm_hidden_size + static_input_size, 64)
        self.bn1 = nn.BatchNorm1d(64) # Batch Normalization
        self.dropout = nn.Dropout(0.5) # Increased Dropout
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, sequence_data, static_data):
        lstm_out, _ = self.lstm(sequence_data)
        lstm_out = lstm_out[:, -1, :]
        combined = torch.cat((lstm_out, static_data), dim=1)
        
        x = self.fc1(combined)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model(sequences, static_features, labels, labeled_setups_df, loss_type='cross-entropy', epochs=100):
    print("\n--- Starting PyTorch Model Training ---")
    
    indices = np.arange(len(sequences))
    
    X_seq_train_val, X_seq_test, X_static_train_val, X_static_test, y_train_val, y_test, _, test_indices = train_test_split(
        sequences, static_features, labels, indices, test_size=0.20, shuffle=False
    )
    
    val_size_relative = 0.10 / 0.80
    X_seq_train, X_seq_val, X_static_train, X_static_val, y_train, y_val = train_test_split(
        X_seq_train_val, X_static_train_val, y_train_val, test_size=val_size_relative, shuffle=False
    )
    
    test_setups_df = labeled_setups_df.iloc[test_indices]
    
    X_seq_train_t = torch.tensor(X_seq_train, dtype=torch.float32)
    X_static_train_t = torch.tensor(X_static_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    
    X_seq_val_t = torch.tensor(X_seq_val, dtype=torch.float32).to(DEVICE)
    X_static_val_t = torch.tensor(X_static_val, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(DEVICE)

    X_seq_test_t = torch.tensor(X_seq_test, dtype=torch.float32).to(DEVICE)
    X_static_test_t = torch.tensor(X_static_test, dtype=torch.float32).to(DEVICE)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(DEVICE)

    train_dataset = TensorDataset(X_seq_train_t, X_static_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) # Increased batch size

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    print(f"Using class weights: {class_weights}")

    model = LSTMModel(X_seq_train.shape[2], X_static_train.shape[1]).to(DEVICE)
    
    if loss_type == 'focal':
        print("Using Focal Loss")
        criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
    else:
        print("Using Cross-Entropy Loss")
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    
    model_path = os.path.join(MODELS_DIR, 'pytorch_lstm_model.pth')
    os.makedirs(MODELS_DIR, exist_ok=True)
    early_stopping = EarlyStopping(patience=10, verbose=True, path=model_path)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for seq_batch, static_batch, labels_batch in train_loader:
            seq_batch, static_batch, labels_batch = seq_batch.to(DEVICE), static_batch.to(DEVICE), labels_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(seq_batch, static_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_seq_val_t, X_static_val_t)
            val_loss = criterion(val_outputs, y_val_t)
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_accuracy = (val_predicted == y_val_t).sum().item() / y_val_t.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    print("\n--- Evaluating Best Model on Final Test Set ---")
    best_model = LSTMModel(X_seq_train.shape[2], X_static_train.shape[1]).to(DEVICE)
    best_model.load_state_dict(torch.load(model_path)) # Load the best model saved by early stopping
    best_model.eval()
    with torch.no_grad():
        test_outputs = best_model(X_seq_test_t, X_static_test_t)
        y_pred = torch.max(test_outputs, 1)[1].cpu().numpy()
        
    print("\nFinal Classification Report (on test set):")
    print(classification_report(y_test, y_pred, target_names=['Loss', 'TP1', 'TP2'], zero_division=0))
    
    return best_model, X_seq_test_t, X_static_test_t, test_setups_df

def calculate_backtest_metrics(trades_log, equity_curve):
    print("\n--- Detailed Backtest Metrics ---")
    if trades_log.empty:
        print("No trades were executed.")
        return

    total_trades = len(trades_log)
    wins = trades_log[trades_log['pnl'] > 0]
    losses = trades_log[trades_log['pnl'] < 0]
    
    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    gross_profit = wins['pnl'].sum()
    gross_loss = abs(losses['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    avg_win = wins['pnl'].mean() if not wins.empty else 0
    avg_loss = abs(losses['pnl'].mean()) if not losses.empty else 0
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    # Max Drawdown
    equity_series = pd.Series(equity_curve)
    peak = equity_series.expanding(min_periods=1).max()
    drawdown = (equity_series - peak) / peak
    max_drawdown = abs(drawdown.min())
    
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Expectancy per Trade: ${expectancy:,.2f}")
    print("---------------------------------")

def run_backtest(model, test_sequences, test_static_features, test_setups_df, initial_balance=10000, risk_percent=2.0, confidence_threshold=0.4):
    print("\n--- Running PyTorch Backtest ---")
    model.eval()
    with torch.no_grad():
        outputs = model(test_sequences, test_static_features)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        predicted_classes = np.argmax(probabilities, axis=1)
    
    equity_curve, trades_log = [initial_balance], []
    balance = float(initial_balance)
    test_setups_df = test_setups_df.reset_index(drop=True)

    for i in range(len(test_setups_df)):
        confidence = probabilities[i].max()
        if predicted_classes[i] in [1, 2] and confidence >= confidence_threshold:
            true_label = test_setups_df.loc[i, 'label']
            
            # Dynamic Position Sizing: Risk is based on current balance
            risk_amount = balance * (risk_percent / 100.0)
            
            # Sanity Guards
            if risk_amount <= 0: continue # Stop trading if balance is zero or negative
            
            pnl = 0.0
            if true_label == 0: # Loss
                pnl = -risk_amount
            elif true_label == 1: # TP1
                pnl = risk_amount 
            elif true_label == 2: # TP2
                pnl = risk_amount * 2
            
            # Clip extreme P&L to avoid unrealistic scenarios (e.g., max 3R win)
            pnl = np.clip(pnl, -risk_amount, risk_amount * 3)
            
            balance += pnl
            equity_curve.append(balance)
            trades_log.append({'timestamp': test_setups_df.loc[i, 'timestamp'], 'pnl': pnl, 'balance': balance})
            
    print(f"Backtest complete. Executed {len(trades_log)} trades.")
    if trades_log:
        print(f"Final Balance: ${equity_curve[-1]:,.2f}")
        
    trades_df = pd.DataFrame(trades_log)
    calculate_backtest_metrics(trades_df, equity_curve)
    
    return equity_curve, trades_df

def plot_backtest_results(equity_curve):
    if len(equity_curve) <= 1:
        print("Not enough data to plot equity curve.")
        return
    
    plt.figure(figsize=(12, 7))
    plt.plot(equity_curve)
    plt.title('Backtest Equity Curve', fontsize=16)
    plt.xlabel('Trade Number'); plt.ylabel('Portfolio Balance ($)')
    plt.grid(True)
    chart_path = os.path.join(MODELS_DIR, 'pytorch_lstm_backtest_chart.png')
    plt.savefig(chart_path)
    print(f"Backtest chart saved to: {chart_path}")
    plt.close()

def run_pipeline(quick_test=False):
    print("--- Starting Full PyTorch Model Pipeline ---")
    if quick_test:
        print("--- RUNNING IN QUICK TEST MODE ---")
    print(f"Using device: {DEVICE}")

    # Check for raw data, fetch if needed
    raw_data_exists = os.path.exists(DATA_DIR) and len(glob.glob(os.path.join(DATA_DIR, '*', '*.parquet'))) > 0
    if not raw_data_exists:
        print("Raw data not found. Fetching initial historical data...")
        fetch_initial_data(SESSIONS, quick_test=quick_test)
    else:
        print("Raw data directory found. Skipping initial fetch.")

    # Check for labeled data, preprocess if needed
    labeled_data_filename = "labeled_trades_quick.parquet" if quick_test else "labeled_trades.parquet"
    labeled_data_path = os.path.join(PROCESSED_DATA_DIR, labeled_data_filename)
    if not os.path.exists(labeled_data_path):
        print("Labeled data not found. Running preprocessing...")
        main_preprocess(SESSIONS, quick_test=quick_test)
    else:
        print("Found existing labeled data, skipping preprocessing.")
        
    try:
        labeled_df = pd.read_parquet(labeled_data_path)
    except Exception as e:
        print(f"Could not read labeled data file: {e}. It might be empty if no setups were found."); return

    if labeled_df.empty:
        print("No labeled setups were generated during preprocessing. Exiting.")
        return

    labeled_df = labeled_df[labeled_df['label'] != -1].copy()
    if labeled_df.empty:
        print("No valid labeled setups found after filtering for label != -1. Exiting."); return
    
    sequences, static_features, labels, _ = create_sequences_and_features(labeled_df)
    if len(sequences) == 0:
        print("No sequences were generated from the labeled data. Exiting."); return
    
    model, test_seq, test_static, test_setups = train_model(sequences, static_features, labels, labeled_df)
    
    equity_curve, trades_log = run_backtest(model, test_seq, test_static, test_setups)
    
    plot_backtest_results(equity_curve)
    print("\n--- PyTorch Pipeline Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the full ML pipeline.")
    parser.add_argument('all', nargs='?', help='Dummy argument to accept "all"')
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run the pipeline with a small subset of data for testing purposes.'
    )
    parser.add_argument(
        '--skip-pre-check',
        action='store_true',
        help='(For developers) Skip the mandatory quick test pre-check before a full run.'
    )
    args = parser.parse_args()

    if args.quick_test:
        print("--- Running in Standalone Quick Test Mode ---")
        run_pipeline(quick_test=True)
    else:
        # This is a full run, which requires a pre-check first
        pre_check_passed = False
        if not args.skip_pre_check:
            print("--- Running Mandatory Quick Test Pre-check ---")
            try:
                run_pipeline(quick_test=True)
                print("\n--- Quick Test Pre-check PASSED ---\n")
                pre_check_passed = True
            except Exception as e:
                print(f"\n--- Quick Test Pre-check FAILED: {e} ---\n")
                print("Aborting full run due to pre-check failure.")
        
        if args.skip_pre_check or pre_check_passed:
            if args.skip_pre_check:
                print("--- Skipping Pre-check. Starting Full Run Directly ---")
            else:
                print("\n--- Starting Full Run ---")
            run_pipeline(quick_test=False)
