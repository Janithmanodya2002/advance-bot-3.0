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
from sklearn.utils.class_weight import compute_class_weight

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- Configuration ---
DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
MODELS_DIR = "models"
SYMBOLS_FILE = "symbols.csv"
SWING_WINDOW = 5
LOOKBACK_CANDLES = 100
TRADE_EXPIRY_BARS = 96
SEQUENCE_LENGTH = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SESSIONS = {
    "Asia": (0, 8),
    "London": (7, 15),
    "New_York": (12, 20)
}

# --- Data Pipeline Functions ---

def fetch_historical_for_symbol(symbol, client, total_limit=20000):
    print(f"Fetching data for {symbol}...")
    all_klines = []
    limit = 1000
    while len(all_klines) < total_limit:
        try:
            end_time = all_klines[0][0] if all_klines else None
            klines = client.get_historical_klines(symbol=symbol, interval=BinanceClient.KLINE_INTERVAL_15MINUTE, limit=min(limit, total_limit - len(all_klines)), end_str=end_time)
            if not klines: break
            all_klines = klines + all_klines
        except Exception as e:
            print(f"Error fetching klines for {symbol}: {e}")
            return symbol, None # Return symbol to identify failure
    return symbol, all_klines

def fetch_initial_data(sessions_dict):
    print("Starting parallel initial data acquisition...")
    try:
        symbols = pd.read_csv(SYMBOLS_FILE, header=None)[0].tolist()
    except FileNotFoundError:
        print(f"Error: {SYMBOLS_FILE} not found.")
        return
    
    client = BinanceClient(keys.api_mainnet, keys.secret_mainnet)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Create a future for each symbol download
        future_to_symbol = {executor.submit(fetch_historical_for_symbol, symbol, client): symbol for symbol in symbols}
        
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol, klines = future.result()
            if klines:
                print(f"  - Successfully fetched {len(klines)} candles for {symbol}")
                filename = f"initial_{len(klines)}.parquet"
                process_and_save_kline_data(klines, symbol, filename, sessions_dict)
            else:
                print(f"  - Failed to fetch data for {symbol}")

    print("\nInitial data acquisition complete.")

def get_session(timestamp, sessions_dict):
    if isinstance(timestamp, pd.Series): return timestamp.apply(lambda ts: _get_single_session(ts, sessions_dict))
    else: return _get_single_session(timestamp, sessions_dict)

def _get_single_session(timestamp, sessions_dict):
    utc_time = pd.to_datetime(timestamp, unit='ms').tz_localize('UTC')
    hour = utc_time.hour
    active_sessions = [name for name, (start, end) in sessions_dict.items() if start <= hour <= end]
    return ",".join(active_sessions) if active_sessions else "Inactive"

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
    table = pa.Table.from_pandas(df)
    pq.write_table(table, file_path)

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
    highs, lows = df['high'], df['low']
    swing_highs, swing_lows = [], []
    for i in range(window, len(df) - window):
        is_swing_high = highs.iloc[i] > highs.iloc[i-window:i].max() and highs.iloc[i] > highs.iloc[i+1:i+window+1].max()
        if is_swing_high: swing_highs.append({'timestamp': df['timestamp'].iloc[i], 'price': highs.iloc[i]})
        is_swing_low = lows.iloc[i] < lows.iloc[i-window:i].min() and lows.iloc[i] < lows.iloc[i+1:i+window+1].min()
        if is_swing_low: swing_lows.append({'timestamp': df['timestamp'].iloc[i], 'price': lows.iloc[i]})
    return pd.DataFrame(swing_highs), pd.DataFrame(swing_lows)

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
    for i in tqdm(range(LOOKBACK_CANDLES, len(df) - TRADE_EXPIRY_BARS), desc="Finding Setups"):
        if df['session'].iloc[i] == 'Inactive': continue
        current_klines_df = df.iloc[i - LOOKBACK_CANDLES : i]
        swing_highs, swing_lows = get_swing_points_df(current_klines_df, SWING_WINDOW)
        if len(swing_highs) < 2 or len(swing_lows) < 2: continue
        trend = get_trend_df(swing_highs, swing_lows)
        setup = None
        if trend == 'downtrend':
            last_swing_high_price, last_swing_low_price = swing_highs.iloc[-1]['price'], swing_lows.iloc[-1]['price']
            if last_swing_low_price >= swing_lows.iloc[-2]['price']: continue
            entry_price = get_fib_retracement(last_swing_high_price, last_swing_low_price, trend)
            sl = last_swing_high_price
            tp1 = entry_price - (sl - entry_price)
            tp2 = entry_price - (sl - entry_price) * 2
            entry_candle_idx = -1
            for k in range(i, min(i + TRADE_EXPIRY_BARS, len(df))):
                if df['high'].iloc[k] >= entry_price: entry_candle_idx = k; break
            if entry_candle_idx == -1: continue
            label = label_trade(df, entry_candle_idx, sl, tp1, tp2, 'short')
            if label != -1: setup = {'timestamp': df['timestamp'].iloc[i], 'side': 'short', 'entry_price': entry_price, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'label': label, 'swing_high_price': last_swing_high_price, 'swing_low_price': last_swing_low_price}
        elif trend == 'uptrend':
            last_swing_high_price, last_swing_low_price = swing_highs.iloc[-1]['price'], swing_lows.iloc[-1]['price']
            if last_swing_high_price <= swing_highs.iloc[-2]['price']: continue
            entry_price = get_fib_retracement(last_swing_low_price, last_swing_high_price, trend)
            sl = last_swing_low_price
            tp1 = entry_price + (entry_price - sl)
            tp2 = entry_price + (entry_price - sl) * 2
            entry_candle_idx = -1
            for k in range(i, min(i + TRADE_EXPIRY_BARS, len(df))):
                if df['low'].iloc[k] <= entry_price: entry_candle_idx = k; break
            if entry_candle_idx == -1: continue
            label = label_trade(df, entry_candle_idx, sl, tp1, tp2, 'long')
            if label != -1: setup = {'timestamp': df['timestamp'].iloc[i], 'side': 'long', 'entry_price': entry_price, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'label': label, 'swing_high_price': last_swing_high_price, 'swing_low_price': last_swing_low_price}
        if setup: labeled_setups.append(setup)
    return pd.DataFrame(labeled_setups)

def process_symbol_for_labeling(symbol, sessions_dict):
    df = load_raw_data_for_symbol(symbol)
    if df is None or df.empty: return None
    df['session'] = get_session(df['timestamp'], sessions_dict)
    labeled_setups = find_and_label_setups(df)
    if not labeled_setups.empty:
        labeled_setups['symbol'] = symbol
        return labeled_setups
    return None

def main_preprocess(sessions_dict, chunk_size=4):
    print("Starting robust preprocessing and labeling with chunking...")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    try:
        symbols = pd.read_csv(SYMBOLS_FILE, header=None)[0].tolist()
    except FileNotFoundError:
        print(f"Error: {SYMBOLS_FILE} not found."); return

    worker_func = partial(process_symbol_for_labeling, sessions_dict=sessions_dict)
    symbol_chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]
    
    for i, chunk in enumerate(symbol_chunks):
        chunk_file_path = os.path.join(PROCESSED_DATA_DIR, f"labeled_chunk_{i+1}.parquet")
        if os.path.exists(chunk_file_path):
            print(f"--- Chunk {i+1}/{len(symbol_chunks)} already processed. Skipping. ---")
            continue
            
        print(f"--- Processing Chunk {i+1}/{len(symbol_chunks)}: {chunk} ---")
        with Pool() as pool:
            results = pool.map(worker_func, chunk)
        
        all_labeled_data = [res for res in results if res is not None]
        
        if all_labeled_data:
            chunk_df = pd.concat(all_labeled_data, ignore_index=True)
            chunk_df.to_parquet(chunk_file_path)
            print(f"Saved chunk {i+1} to {chunk_file_path}")
        else:
            print(f"No labeled data generated for chunk {i+1}.")

    print("\n--- Combining all processed chunks ---")
    all_chunk_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, "labeled_chunk_*.parquet"))
    if not all_chunk_files:
        print("No chunk files found to combine. Preprocessing did not generate any labeled data.")
        # Create an empty placeholder file so the pipeline doesn't fail on read
        pd.DataFrame().to_parquet(os.path.join(PROCESSED_DATA_DIR, "labeled_trades.parquet"))
        return
        
    combined_df = pd.concat([pd.read_parquet(f) for f in all_chunk_files], ignore_index=True)
    final_save_path = os.path.join(PROCESSED_DATA_DIR, "labeled_trades.parquet")
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

def create_sequences_and_features(labeled_setups_df):
    print("Starting sequence and feature generation...")
    all_raw_data = {}
    for symbol in tqdm(labeled_setups_df['symbol'].unique(), desc="Loading Raw Data"):
        df = load_raw_data_for_symbol(symbol)
        if df is not None:
            df['atr'] = calculate_atr(df)
            df['rsi'] = calculate_rsi(df)
            df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df)
            df['bb_upper'], df['bb_lower'] = calculate_bollinger_bands(df)
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
        
        sequence_df = raw_df.iloc[setup_idx_loc - SEQUENCE_LENGTH:setup_idx_loc]
        last_close = sequence_df['close'].iloc[-1]
        
        seq_features = pd.DataFrame()
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
        seq_features['bb_pos'] = (sequence_df['close'] - sequence_df['bb_lower']) / (sequence_df['bb_upper'] - sequence_df['bb_lower'] + 1e-8)

        sequence_list.append(seq_features.values)
        
        static_features = {}
        static_features['side_long'] = 1 if row['side'] == 'long' else 0
        static_features['side_short'] = 1 if row['side'] == 'short' else 0
        swing_height = row['swing_high_price'] - row['swing_low_price']
        if swing_height > 0:
            if row['side'] == 'long': static_features['golden_zone_ratio'] = (row['entry_price'] - row['swing_low_price']) / swing_height
            else: static_features['golden_zone_ratio'] = (row['swing_high_price'] - row['entry_price']) / swing_height
            static_features['sl_pct_of_swing'] = abs(row['entry_price'] - row['sl']) / swing_height
        else:
            static_features['golden_zone_ratio'], static_features['sl_pct_of_swing'] = 0, 0
        
        static_list.append(list(static_features.values()))
        label_list.append(row['label'])
        
    sequences_np, static_np, labels_np = np.array(sequence_list), np.array(static_list), np.array(label_list)
    return sequences_np, static_np, labels_np

class LSTMModel(nn.Module):
    def __init__(self, sequence_input_size, static_input_size, lstm_hidden_size=50, num_classes=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(sequence_input_size, lstm_hidden_size, batch_first=True)
        self.fc1 = nn.Linear(lstm_hidden_size + static_input_size, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, sequence_data, static_data):
        lstm_out, _ = self.lstm(sequence_data)
        lstm_out = lstm_out[:, -1, :]
        combined = torch.cat((lstm_out, static_data), dim=1)
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model(sequences, static_features, labels, labeled_setups_df):
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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    print(f"Using class weights: {class_weights}")

    model = LSTMModel(X_seq_train.shape[2], X_static_train.shape[1]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_accuracy = 0.0
    model_path = os.path.join(MODELS_DIR, 'pytorch_lstm_model.pth')
    os.makedirs(MODELS_DIR, exist_ok=True)

    for epoch in range(50):
        model.train()
        for seq_batch, static_batch, labels_batch in train_loader:
            seq_batch, static_batch, labels_batch = seq_batch.to(DEVICE), static_batch.to(DEVICE), labels_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(seq_batch, static_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_seq_val_t, X_static_val_t)
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_accuracy = (val_predicted == y_val_t).sum().item() / y_val_t.size(0)
            print(f'Epoch [{epoch+1}/50], Loss: {loss.item():.4f}, Validation Accuracy: {val_accuracy:.4f}')
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), model_path)
                print(f"New best model saved to {model_path} with validation accuracy: {best_val_accuracy:.4f}")

    print("\n--- Evaluating Best Model on Final Test Set ---")
    best_model = LSTMModel(X_seq_train.shape[2], X_static_train.shape[1]).to(DEVICE)
    best_model.load_state_dict(torch.load(model_path))
    best_model.eval()
    with torch.no_grad():
        test_outputs = best_model(X_seq_test_t, X_static_test_t)
        y_pred = torch.max(test_outputs, 1)[1].cpu().numpy()
        
    print("\nFinal Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Loss', 'TP1', 'TP2'], zero_division=0))
    
    return best_model, X_seq_test_t, X_static_test_t, test_setups_df

def run_backtest(model, test_sequences, test_static_features, test_setups_df, initial_balance=10000, risk_percent=2.0, confidence_threshold=0.4):
    print("\n--- Running PyTorch Backtest ---")
    model.eval()
    with torch.no_grad():
        outputs = model(test_sequences, test_static_features)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        predicted_classes = np.argmax(probabilities, axis=1)
    
    equity_curve, trades_log = [initial_balance], []
    balance = initial_balance
    test_setups_df = test_setups_df.reset_index(drop=True)

    for i in range(len(test_setups_df)):
        confidence = probabilities[i].max()
        if predicted_classes[i] in [1, 2] and confidence >= confidence_threshold:
            true_label = test_setups_df.loc[i, 'label']
            risk_amount = initial_balance * (risk_percent / 100.0) # Use initial_balance for stable risk
            
            pnl = 0
            if true_label == 0: pnl = -risk_amount
            elif true_label == 1: pnl = risk_amount
            elif true_label == 2: pnl = risk_amount * 2
            
            balance += pnl
            equity_curve.append(balance)
            trades_log.append({'timestamp': test_setups_df.loc[i, 'timestamp'], 'pnl': pnl, 'balance': balance})
            
    print(f"Backtest complete. Executed {len(trades_log)} trades.")
    if trades_log: print(f"Final Balance: ${equity_curve[-1]:,.2f}")
    return equity_curve, pd.DataFrame(trades_log)

def plot_backtest_results(equity_curve):
    if len(equity_curve) <= 1: print("Not enough data to plot equity curve."); return
    plt.figure(figsize=(12, 7))
    plt.plot(equity_curve)
    plt.title('PyTorch LSTM Model Backtest Equity Curve', fontsize=16)
    plt.xlabel('Trade Number'); plt.ylabel('Portfolio Balance ($)')
    plt.grid(True)
    chart_path = os.path.join(MODELS_DIR, 'pytorch_lstm_backtest_chart.png')
    plt.savefig(chart_path)
    print(f"Backtest chart saved to: {chart_path}")
    plt.close()

def run_pipeline():
    print("--- Starting Full PyTorch Model Pipeline ---")
    print(f"Using device: {DEVICE}")

    raw_data_exists = os.path.exists(DATA_DIR) and len(glob.glob(os.path.join(DATA_DIR, '*', '*.parquet'))) > 0
    if not raw_data_exists:
        print("Raw data not found. Fetching initial historical data...")
        fetch_initial_data(SESSIONS)
    else:
        print("Raw data directory found. Skipping initial fetch.")

    labeled_data_path = os.path.join(PROCESSED_DATA_DIR, "labeled_trades.parquet")
    if not os.path.exists(labeled_data_path):
        print("Labeled data not found. Running preprocessing...")
        main_preprocess(SESSIONS)
    else:
        print("Found existing labeled data, skipping preprocessing.")
        
    try:
        labeled_df = pd.read_parquet(labeled_data_path)
    except Exception as e:
        print(f"Could not read labeled data file: {e}."); return
        
    labeled_df = labeled_df[labeled_df['label'] != -1].copy()
    if labeled_df.empty: print("No valid labeled data found."); return
    
    sequences, static_features, labels = create_sequences_and_features(labeled_df)
    if len(sequences) == 0: print("No sequences were generated."); return
    
    model, test_seq, test_static, test_setups = train_model(sequences, static_features, labels, labeled_df)
    
    equity_curve, trades_log = run_backtest(model, test_seq, test_static, test_setups)
    
    plot_backtest_results(equity_curve)
    print("\n--- PyTorch Pipeline Finished ---")

if __name__ == '__main__':
    run_pipeline()
