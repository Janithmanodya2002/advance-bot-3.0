import pandas as pd
import numpy as np
import os
import pyarrow as pa
import pyarrow.parquet as pq

# --- Create a more realistic dummy dataset ---
def generate_dummy_data(num_candles=200):
    base_price = 10000
    timestamps = pd.to_datetime(pd.date_range(start='2025-01-01', periods=num_candles, freq='15min'))

    # Create a wave-like pattern for price
    price_movement = np.sin(np.linspace(0, 20, num_candles)) * 50  # Swing of $50
    price = base_price + price_movement

    # Add some noise
    noise = np.random.normal(0, 5, num_candles)
    price += noise

    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': price - np.random.uniform(0, 5, num_candles),
        'high': price + np.random.uniform(0, 5, num_candles),
        'low': price - np.random.uniform(5, 10, num_candles),
        'close': price,
        'volume': np.random.randint(100, 1000, num_candles),
    })

    # Ensure OHLC consistency
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    # Add session data
    def get_session_from_dt(dt):
        hour = dt.hour
        if 0 <= hour < 9: return "Asia"
        if 7 <= hour < 16: return "London"
        if 12 <= hour < 21: return "New_York"
        return "Inactive"

    df['session'] = df['timestamp'].apply(get_session_from_dt)

    # Convert timestamp to milliseconds
    df['timestamp'] = df['timestamp'].apply(lambda x: int(x.timestamp() * 1000))

    return df

# --- Save the data ---
symbol = 'BTCUSDT'
df = generate_dummy_data()

symbol_dir = f'data/raw/{symbol}'
os.makedirs(symbol_dir, exist_ok=True)
file_path = os.path.join(symbol_dir, '2025-01-01.parquet')

table = pa.Table.from_pandas(df)
pq.write_table(table, file_path)

print(f"Dummy data with {len(df)} candles created at {file_path}")
