import time
import json
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.enums import *
from binance.enums import ORDER_TYPE_STOP_MARKET
from telegram import Bot
import argparse
from datetime import datetime, timezone
import logging
from logging.handlers import RotatingFileHandler
import os
from keys import BINANCE_API_KEY, BINANCE_API_SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

# Configure logging
log_handler = RotatingFileHandler('trading.log', maxBytes=10*1024*1024, backupCount=5)
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(log_handler)
logging.getLogger().setLevel(logging.INFO)

# Load configuration
symbols_df = pd.read_csv('symbol.csv')
symbols = symbols_df['symbol'].tolist()
with open('config.json', 'r') as f:
    config = json.load(f)

def time_to_timestamp(time_str, date=None):
    if date is None:
        date = datetime.now(timezone.utc).date()
    dt = datetime.strptime(f"{date} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def fetch_with_retry(func, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise

class BinanceClientWrapper:
    def _init_(self, api_key, api_secret):
        self.client = Client(api_key, api_secret)
        self.client.API_URL = 'https://fapi.binance.com'

    def get_klines(self, symbol: str, interval: str, start_time=None, end_time=None, limit: int = 1000) -> list:
        """Fetch klines from Binance."""
        try:
            klines = fetch_with_retry(
                lambda: self.client.futures_klines(symbol=symbol, interval=interval, startTime=start_time, endTime=end_time, limit=limit)
            )
            logging.info(f"Fetching klines for {symbol}, interval: {interval}")
            return klines or []
        except BinanceAPIException as e:
            logging.error(f"Binance API error fetching klines for {symbol}: {e}")
            return []
        except Exception as e:
            logging.error(f"Unexpected error fetching klines for {symbol}: {e}")
            return []

    def place_order(self, symbol: str, side: str, order_type: str, quantity: float, **kwargs) -> dict:
        """Place an order on Binance."""
        try:
            order = fetch_with_retry(
                lambda: self.client.futures_create_order(symbol=symbol, side=side, type=order_type, quantity=quantity, **kwargs)
            )
            return order
        except BinanceAPIException as e:
            logging.error(f"Binance API error placing order for {symbol}: {e}")
            return None

    def get_account_info(self) -> dict:
        """Get futures account information."""
        try:
            return fetch_with_retry(self.client.futures_account)
        except BinanceAPIException as e:
            logging.error(f"Binance API error fetching account info: {e}")
            return None

class Strategy:
    def _init_(self, config: dict):
        self.config = config

    def calculate_key_levels(self, symbol: str, klines_dict: dict) -> dict:
        """Calculate key price levels."""
        key_levels = {'lows': [], 'highs': []}
        lookback_periods = self.config.get('lookback_periods', 10)
        for tf in self.config['high_time_frames']:
            klines = klines_dict.get(tf, [])
            if klines:
                recent_candles = klines[-lookback_periods-1:-1]
                if recent_candles:
                    highs = [float(k[2]) for k in recent_candles]
                    lows = [float(k[3]) for k in recent_candles]
                    key_levels['highs'].append(max(highs))
                    key_levels['lows'].append(min(lows))
                    key_levels['highs'].append(sorted(highs)[len(highs)//2])
                    key_levels['lows'].append(sorted(lows)[len(lows)//2])
        for session_name in self.config['sessions']:
            klines = klines_dict.get(session_name, [])
            if klines:
                highs = [float(k[2]) for k in klines]
                lows = [float(k[3]) for k in klines]
                volumes = [float(k[5]) for k in klines]
                if highs and volumes:
                    high_idx = highs.index(max(highs))
                    low_idx = lows.index(min(lows))
                    avg_volume = sum(volumes) / len(volumes)
                    if volumes[high_idx] > avg_volume:
                        key_levels['highs'].append(max(highs))
                    if volumes[low_idx] > avg_volume:
                        key_levels['lows'].append(min(lows))
        key_levels['highs'] = sorted(list(set(key_levels['highs'])))
        key_levels['lows'] = sorted(list(set(key_levels['lows'])))
        logging.info(f"Calculated key levels for {symbol}: highs={key_levels['highs']}, lows={key_levels['lows']}")
        return key_levels

    def detect_reversal(self, symbol: str, current_price: float, key_levels: dict, klines_low_tf: list) -> str:
        """Detect potential reversals."""
        if not klines_low_tf or len(klines_low_tf) < 3:
            return None
        last_candles = klines_low_tf[-3:]
        if current_price <= min(key_levels['lows']):
            if (float(last_candles[0][4]) > float(last_candles[1][4]) and 
                float(last_candles[1][4]) < float(last_candles[0][3]) and 
                current_price > float(last_candles[1][2])):
                return 'bullish'
        elif current_price >= max(key_levels['highs']):
            if (float(last_candles[0][4]) < float(last_candles[1][4]) and 
                float(last_candles[1][4]) > float(last_candles[0][2]) and 
                current_price < float(last_candles[1][3])):
                return 'bearish'
        return None

    def confirm_continuation(self, symbol: str, reversal_type: str, klines_low_tf: list) -> bool:
        """Confirm continuation after a reversal."""
        if not klines_low_tf or len(klines_low_tf) < 2:
            return False
        prev_candle = klines_low_tf[-2]
        curr_candle = klines_low_tf[-1]
        if reversal_type == 'bullish' and float(curr_candle[4]) > float(prev_candle[4]):
            return True
        elif reversal_type == 'bearish' and float(curr_candle[4]) < float(prev_candle[4]):
            return True
        return False

class TelegramNotifier:
    def _init_(self, bot_token: str, chat_id: str):
        self.bot = Bot(token=bot_token)
        self.chat_id = chat_id

    def send_message(self, message: str):
        """Send a message via Telegram."""
        try:
            self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode='Markdown')
        except Exception as e:
            logging.error(f"Error sending Telegram message: {e}")

def update_order_status(client: BinanceClientWrapper, open_orders: dict):
    """Update the status of open orders."""
    for symbol in open_orders:
        for order in open_orders[symbol]:
            if order['status'] != 'FILLED':
                try:
                    order_status = client.client.futures_get_order(symbol=symbol, orderId=order['order_id'])
                    order['status'] = order_status['status']
                    if order['status'] == 'FILLED':
                        logging.info(f"Order {order['order_id']} for {symbol} filled")
                except Exception as e:
                    logging.error(f"Error checking order status for {symbol}: {e}")

def check_positions(client: BinanceClientWrapper):
    """Check open positions in the account."""
    account_info = client.get_account_info()
    if account_info:
        for position in account_info['positions']:
            if float(position['positionAmt']) != 0:
                logging.info(f"Open position for {position['symbol']}: {position['positionAmt']}")

def main(mode: str):
    client = BinanceClientWrapper(BINANCE_API_KEY, BINANCE_API_SECRET)
    strategy = Strategy(config)
    notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    open_orders = {symbol: [] for symbol in symbols}
    cycle_count = 0

    while True:
        if os.path.exists('stop.txt'):
            logging.info("Stop signal received. Exiting.")
            notifier.send_message("Script stopped gracefully")
            os.remove('stop.txt')
            break
        cycle_start = time.time()
        logging.info(f"Starting cycle {cycle_count + 1}")

        try:
            prices = client.client.futures_symbol_ticker()
            price_dict = {price['symbol']: float(price['price']) for price in prices}
        except Exception as e:
            logging.error(f"Error fetching prices: {e}")
            time.sleep(60)
            continue

        for symbol in symbols:
            current_price = price_dict.get(symbol)
            if not current_price:
                continue

            klines_dict = {}
            for tf in config['high_time_frames']:
                klines_dict[tf] = client.get_klines(symbol, tf)
            for session_name, times in config['sessions'].items():
                start_time = time_to_timestamp(times['start'])
                end_time = time_to_timestamp(times['end'])
                klines_dict[session_name] = client.get_klines(symbol, '1m', start_time, end_time)
            klines_low_tf = client.get_klines(symbol, config['low_time_frame'])

            key_levels = strategy.calculate_key_levels(symbol, klines_dict)
            reversal = strategy.detect_reversal(symbol, current_price, key_levels, klines_low_tf)
            if reversal and strategy.confirm_continuation(symbol, reversal, klines_low_tf):
                direction = 'Long' if reversal == 'bullish' else 'Short'
                entry = current_price
                sl = min(key_levels['lows']) - 100 if direction == 'Long' else max(key_levels['highs']) + 100
                tp = max(key_levels['highs']) if direction == 'Long' else min(key_levels['lows'])

                if mode == 'live':
                    market_order = client.place_order(symbol, SIDE_BUY if direction == 'Long' else SIDE_SELL, ORDER_TYPE_MARKET, 0.001)
                    if market_order:
                        open_orders[symbol].append({'type': 'market', 'order_id': market_order['orderId'], 'status': 'NEW'})
                        sl_order = client.place_order(symbol, SIDE_SELL if direction == 'Long' else SIDE_BUY, ORDER_TYPE_STOP_MARKET, 0.001, stopPrice=sl)
                        tp_order = client.place_order(symbol, SIDE_SELL if direction == 'Long' else SIDE_BUY, ORDER_TYPE_LIMIT, 0.001, price=tp)
                        open_orders[symbol].append({'type': 'sl', 'order_id': sl_order['orderId'], 'status': 'NEW'})
                        open_orders[symbol].append({'type': 'tp', 'order_id': tp_order['orderId'], 'status': 'NEW'})
                        message = f"Trade Executed: {symbol} {direction}\nEntry: {entry}\nSL: {sl}\nTP: {tp}"
                        notifier.send_message(message)
                elif mode == 'signal':
                    message = f"Signal: {symbol} {direction}\nEntry: {entry}\nSL: {sl}\nTP: {tp}"
                    notifier.send_message(message)

            time.sleep(1)

        update_order_status(client, open_orders)
        check_positions(client)

        cycle_end = time.time()
        cycle_duration = cycle_end - cycle_start
        logging.info(f"Cycle completed in {cycle_duration:.2f} seconds")
        time.sleep(max(0, 120 - cycle_duration))
        cycle_count += 1
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trading Script')
    parser.add_argument('--mode', choices=['live', 'signal'], required=True, help='Mode: live or signal')
    args = parser.parse_args()
    main(args.mode)