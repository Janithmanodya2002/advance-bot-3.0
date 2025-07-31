import pandas as pd
import time
from binance.client import Client
import keys
import asyncio
import telegram
from telegram.ext import ApplicationBuilder, CommandHandler
import numpy as np
import json
import pytz
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import matplotlib.patches as mpatches
import io
import threading
import requests
try:
    import mplfinance as mpf
except ImportError:
    print("mplfinance not found. Please install it by running: pip install mplfinance")
    exit()
from tabulate import tabulate

class TradeResult:
    def __init__(self, symbol, side, entry_price, exit_price, entry_timestamp, exit_timestamp, status, pnl_usd, pnl_pct, drawdown, reason_for_entry, reason_for_exit, fib_levels):
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.entry_timestamp = entry_timestamp
        self.exit_timestamp = exit_timestamp
        self.status = status
        self.pnl_usd = pnl_usd
        self.pnl_pct = pnl_pct
        self.drawdown = drawdown
        self.reason_for_entry = reason_for_entry
        self.reason_for_exit = reason_for_exit
        self.fib_levels = fib_levels

def generate_fib_chart(symbol, klines, trend, swing_high, swing_low, entry_price, sl, tp1, tp2):
    """
    Generate a detailed candlestick chart with Fibonacci levels, entry, SL, and TP.
    """
    df = pd.DataFrame(klines, columns=['dt', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['dt'] = pd.to_datetime(df['dt'], unit='ms')
    df.set_index('dt', inplace=True)
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
    
    # Chart Styling
    mc = mpf.make_marketcolors(up='#26A69A', down='#EF5350', wick={'up':'#26A69A', 'down':'#EF5350'}, volume='in', ohlc='i')
    s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc, gridcolor='lightgrey', facecolor='white')

    # Plot
    fig, axlist = mpf.plot(df, type='candle', style=s,
                          figsize=(8, 4.5),
                          returnfig=True,
                          volume=False) # We will plot volume ourselves if needed
    
    ax = axlist[0]
    ax.set_title(f'{symbol} 15m - Fib Entry/SL/TP', fontsize=16, weight='bold')
    ax.set_ylabel('Price (USDT)', fontsize=10)
    ax.tick_params(axis='x', labelsize=10, labelrotation=45)
    ax.tick_params(axis='y', labelsize=10)
    
    # Fibonacci Levels
    fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 1.0]
    price_range = swing_high - swing_low
    if trend == 'downtrend':
        fib_prices = [swing_high - level * price_range for level in fib_levels]
        golden_zone_start = swing_high - (price_range * 0.5)
        golden_zone_end = swing_high - (price_range * 0.618)
    else: # uptrend
        fib_prices = [swing_low + level * price_range for level in fib_levels]
        golden_zone_start = swing_low + (price_range * 0.5)
        golden_zone_end = swing_low + (price_range * 0.618)

    ax.axhspan(golden_zone_start, golden_zone_end, color='gold', alpha=0.2)

    for level, price in zip(fib_levels, fib_prices):
        ax.axhline(y=price, color='#455A64', linestyle='--', linewidth=1.2)
        ax.text(df.index[-1], price, f' {level*100:.1f}% - {price:.2f}', color='#455A64', va='center', ha='left', fontsize=9)

    # Current Price
    current_price = df['close'].iloc[-1]
    ax.axhline(y=current_price, color='#000000', linestyle='-', linewidth=1)
    ax.text(df.index[-1], current_price, f' PRICE {current_price:.2f}', color='#000000', va='center', ha='left', fontsize=9, weight='bold')

    # Entry/SL/TP Zones
    entry_high = entry_price * 1.005
    entry_low = entry_price * 0.995
    ax.axhspan(entry_low, entry_high, color='green', alpha=0.2)
    x_mid = df.index[0] + (df.index[-1] - df.index[0]) / 2
    ax.text(x_mid, (entry_high+entry_low)/2, f'ENTRY {entry_price:.2f}', color='white', va='center', ha='center', fontsize=10)

    sl_high = sl * 1.005
    sl_low = sl * 0.995
    ax.axhspan(sl_low, sl_high, color='red', alpha=0.2)
    ax.text(x_mid, (sl_high+sl_low)/2, f'SL {sl:.2f}', color='white', va='center', ha='center', fontsize=10)

    tp_high = tp1 * 1.005
    tp_low = tp1 * 0.995
    ax.axhspan(tp_low, tp_high, color='blue', alpha=0.2)
    ax.text(x_mid, (tp_high+tp_low)/2, f'TP {tp1:.2f}', color='white', va='center', ha='center', fontsize=10)

    # Entry/SL/TP Lines
    ax.axhline(y=entry_price, color='green', linestyle='-', linewidth=1.2)
    ax.axhline(y=sl, color='red', linestyle='-', linewidth=1.2)
    ax.axhline(y=tp1, color='blue', linestyle='-', linewidth=1.2)

    # Legend
    entry_patch = mpatches.Patch(color='green', label='ENTRY')
    sl_patch = mpatches.Patch(color='red', label='SL')
    tp_patch = mpatches.Patch(color='blue', label='TP')
    ax.legend(handles=[entry_patch, sl_patch, tp_patch], loc='lower left')

    # Save to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100) # dpi=100 and figsize=(8,4.5) gives 800x450
    buf.seek(0)
    
    return buf

def get_public_ip():
    """
    Get the public IP address.
    """
    try:
        response = requests.get('https://api.ipify.org?format=json')
        response.raise_for_status()
        ip_data = response.json()
        return ip_data['ip']
    except requests.exceptions.RequestException as e:
        print(f"Error getting public IP address: {e}")
        return None

def get_swing_points(klines, window=5):
    """
    Identify swing points from kline data.
    """
    highs = np.array([float(k[2]) for k in klines])
    lows = np.array([float(k[3]) for k in klines])
    
    swing_highs = []
    swing_lows = []

    for i in range(window, len(highs) - window):
        is_swing_high = True
        for j in range(1, window + 1):
            if highs[i] < highs[i-j] or highs[i] < highs[i+j]:
                is_swing_high = False
                break
        if is_swing_high:
            swing_highs.append((klines[i][0], highs[i]))

        is_swing_low = True
        for j in range(1, window + 1):
            if lows[i] > lows[i-j] or lows[i] > lows[i+j]:
                is_swing_low = False
                break
        if is_swing_low:
            swing_lows.append((klines[i][0], lows[i]))
            
    return swing_highs, swing_lows

def get_trend(swing_highs, swing_lows):
    """
    Determine the trend based on swing points.
    """
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "undetermined"

    last_high = swing_highs[-1][1]
    prev_high = swing_highs[-2][1]
    last_low = swing_lows[-1][1]
    prev_low = swing_lows[-2][1]

    if last_high > prev_high and last_low > prev_low:
        return "uptrend"
    elif last_high < prev_high and last_low < prev_low:
        return "downtrend"
    else:
        return "undetermined"

def get_fib_retracement(p1, p2, trend):
    """
    Calculate Fibonacci retracement levels.
    """
    price_range = abs(p1 - p2)
    
    if trend == "downtrend":
        golden_zone_start = p1 - (price_range * 0.5)
        golden_zone_end = p1 - (price_range * 0.618)
    else: # Uptrend
        golden_zone_start = p1 + (price_range * 0.5)
        golden_zone_end = p1 + (price_range * 0.618)

    entry_price = (golden_zone_start + golden_zone_end) / 2
    
    return entry_price

def calculate_quantity(client, symbol, risk_per_trade, sl_price, entry_price, leverage):
    """
    Calculate the order quantity based on risk.
    """
    try:
        # Get account balance
        account_info = client.futures_account()
        balance = float(account_info['totalWalletBalance'])
        
        # Calculate position size
        risk_amount = balance * (risk_per_trade / 100)
        sl_percentage = abs(entry_price - sl_price) / entry_price
        position_size = risk_amount / sl_percentage
        
        # Calculate quantity
        quantity = (position_size * leverage) / entry_price
        
        # Adjust for symbol's precision
        info = client.get_symbol_info(symbol)
        step_size = float(info['filters'][1]['stepSize'])
        quantity = (quantity // step_size) * step_size
        
        return quantity
    except Exception as e:
        print(f"Error calculating quantity for {symbol}: {e}")
        return None

def get_atr(klines, period=14):
    """
    Calculate the Average True Range (ATR).
    """
    highs = np.array([float(k[2]) for k in klines])
    lows = np.array([float(k[3]) for k in klines])
    closes = np.array([float(k[4]) for k in klines])
    
    tr1 = highs - lows
    tr2 = np.abs(highs - np.roll(closes, 1))
    tr3 = np.abs(lows - np.roll(closes, 1))
    
    tr = np.amax([tr1, tr2, tr3], axis=0)
    
    atr = np.zeros(len(tr))
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
    return atr


from binance.exceptions import BinanceAPIException

def get_klines(client, symbol, interval='15m', limit=100, start_str=None, end_str=None):
    """
    Get historical kline data from Binance.
    """
    try:
        klines = client.get_historical_klines(symbol=symbol, interval=interval, start_str=start_str, end_str=end_str)
        return klines
    except BinanceAPIException as e:
        print(f"Error fetching klines for {symbol}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred fetching klines for {symbol}: {e}")
        return None

def calculate_performance_metrics(backtest_trades, starting_balance):
    """
    Calculate performance metrics from a list of trades.
    """
    num_trades = len(backtest_trades)
    wins = sum(1 for trade in backtest_trades if trade.status == 'win')
    losses = num_trades - wins
    win_rate = (wins / num_trades) * 100 if num_trades > 0 else 0
    
    total_win_amount = sum(trade.pnl_usd for trade in backtest_trades if trade.status == 'win')
    total_loss_amount = sum(trade.pnl_usd for trade in backtest_trades if trade.status == 'loss')
    
    avg_win = total_win_amount / wins if wins > 0 else 0
    avg_loss = total_loss_amount / losses if losses > 0 else 0
    
    profit_factor = total_win_amount / abs(total_loss_amount) if total_loss_amount != 0 else float('inf')
    
    net_pnl_usd = total_win_amount + total_loss_amount
    net_pnl_pct = (net_pnl_usd / starting_balance) * 100
    
    expectancy = (win_rate/100 * avg_win) - ( (losses/num_trades) * abs(avg_loss)) if num_trades > 0 else 0

    # Drawdown calculation
    balance_over_time = [starting_balance] + [trade.balance for trade in backtest_trades]
    peak = balance_over_time[0]
    max_drawdown = 0
    for balance in balance_over_time:
        if balance > peak:
            peak = balance
        drawdown = (peak - balance) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return {
        'total_trades': num_trades,
        'winning_trades': wins,
        'losing_trades': losses,
        'win_rate': win_rate,
        'average_win': avg_win,
        'average_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown * 100,
        'net_pnl_usd': net_pnl_usd,
        'net_pnl_pct': net_pnl_pct,
        'expectancy': expectancy
    }

def analyze_strategy_behavior(backtest_trades):
    """
    Analyze the performance of the strategy based on different conditions.
    """
    # Performance by hour
    hourly_performance = {}
    for trade in backtest_trades:
        hour = datetime.datetime.fromtimestamp(trade.entry_timestamp/1000).hour
        if hour not in hourly_performance:
            hourly_performance[hour] = {'wins': 0, 'losses': 0, 'total': 0}
        hourly_performance[hour]['total'] += 1
        if trade.status == 'win':
            hourly_performance[hour]['wins'] += 1
        else:
            hourly_performance[hour]['losses'] += 1
            
    # Performance by trend
    trend_performance = {'uptrend': {'wins': 0, 'losses': 0, 'total': 0}, 'downtrend': {'wins': 0, 'losses': 0, 'total': 0}}
    for trade in backtest_trades:
        if "uptrend" in trade.reason_for_entry:
            trend_performance['uptrend']['total'] += 1
            if trade.status == 'win':
                trend_performance['uptrend']['wins'] += 1
            else:
                trend_performance['uptrend']['losses'] += 1
        elif "downtrend" in trade.reason_for_entry:
            trend_performance['downtrend']['total'] += 1
            if trade.status == 'win':
                trend_performance['downtrend']['wins'] += 1
            else:
                trend_performance['downtrend']['losses'] += 1

    return {
        'hourly_performance': hourly_performance,
        'trend_performance': trend_performance
    }

def generate_drawdown_curve(backtest_trades, starting_balance):
    """
    Generate and save a plot of the drawdown curve.
    """
    balance_over_time = [starting_balance] + [trade.balance for trade in backtest_trades]
    peak = balance_over_time[0]
    drawdowns = []
    for balance in balance_over_time:
        if balance > peak:
            peak = balance
        drawdown = (peak - balance) / peak
        drawdowns.append(drawdown * 100)
        
    plt.figure(figsize=(10, 6))
    plt.plot(drawdowns, color='red')
    plt.title('Drawdown Curve')
    plt.xlabel('Trade Number')
    plt.ylabel('Drawdown (%)')
    plt.grid(True)
    plt.savefig('backtest/drawdown_curve.png')
    plt.close()

def generate_win_loss_distribution(backtest_trades):
    """
    Generate and save a plot of the win/loss distribution.
    """
    wins = sum(1 for trade in backtest_trades if trade.status == 'win')
    losses = len(backtest_trades) - wins
    labels = 'Wins', 'Losses'
    sizes = [wins, losses]
    colors = ['#26A69A', '#EF5350']
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Win/Loss Distribution')
    plt.axis('equal')
    plt.savefig('backtest/win_loss_distribution.png')
    plt.close()

def generate_returns_histogram(backtest_trades):
    """
    Generate and save a histogram of trade returns.
    """
    returns = [trade.pnl_pct for trade in backtest_trades]
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=50, color='blue', alpha=0.7)
    plt.title('Trade Returns Histogram')
    plt.xlabel('Return (%)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('backtest/returns_histogram.png')
    plt.close()

def generate_csv_report(backtest_trades):
    """
    Generate a CSV report from the backtest results.
    """
    df = pd.DataFrame([vars(t) for t in backtest_trades])
    df.to_csv('backtest/backtest_trades.csv', index=False)
    print("Backtest trades saved to backtest/backtest_trades.csv")

def generate_json_report(backtest_trades, metrics, strategy_analysis):
    """
    Generate a JSON report from the backtest results.
    """
    report = {
        'metrics': metrics,
        'strategy_analysis': strategy_analysis,
        'trades': [vars(t) for t in backtest_trades]
    }
    with open('backtest/backtest_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    print("Backtest report saved to backtest/backtest_report.json")

def generate_summary_report(backtest_trades, metrics, strategy_analysis, config, starting_balance):
    """
    Generate a human-readable summary of the backtest results.
    """
    headers = ["Metric", "Value"]
    table = [
        ["Starting Balance", f"${starting_balance:,.2f}"],
        ["Ending Balance", f"${metrics['net_pnl_usd'] + starting_balance:,.2f}"],
        ["Total Profit", f"${metrics['net_pnl_usd']:,.2f}"],
        ["Total Trades", metrics['total_trades']],
        ["Winning Trades", metrics['winning_trades']],
        ["Losing Trades", metrics['losing_trades']],
        ["Win Rate", f"{metrics['win_rate']:.2f}%"],
        ["Average Win", f"${metrics['average_win']:,.2f}"],
        ["Average Loss", f"${metrics['average_loss']:,.2f}"],
        ["Profit Factor", f"{metrics['profit_factor']:.2f}"],
        ["Max Drawdown", f"{metrics['max_drawdown']:.2f}%"],
        ["Expectancy", f"${metrics['expectancy']:,.2f}"]
    ]
    
    report = "Backtesting Summary\n"
    report += "===================\n\n"
    report += "Configuration:\n"
    report += "--------------\n"
    report += f"Risk per trade: {config['risk_per_trade']}%\n"
    report += f"Leverage: {config['leverage']}x\n"
    report += f"ATR Value: {config['atr_value']}\n"
    report += f"Lookback Candles: {config['lookback_candles']}\n"
    report += f"Swing Window: {config['swing_window']}\n\n"
    
    report += "Overall Performance:\n"
    report += "--------------------\n"
    report += tabulate(table, headers=headers, tablefmt="grid")
    report += "\n\n"
    
    report += "Strategy Behavior Insights:\n"
    report += "-------------------------\n"
    report += "\nHourly Performance:\n"
    hourly_table = [["Hour", "Wins", "Losses", "Win Rate"]]
    for hour, data in sorted(strategy_analysis['hourly_performance'].items()):
        win_rate = (data['wins'] / data['total']) * 100 if data['total'] > 0 else 0
        hourly_table.append([f"{hour:02d}", data['wins'], data['losses'], f"{win_rate:.2f}%"])
    report += tabulate(hourly_table, headers="firstrow", tablefmt="grid")
    report += "\n\n"
    
    report += "Trend Performance:\n"
    trend_table = [["Trend", "Wins", "Losses", "Win Rate"]]
    for trend, data in strategy_analysis['trend_performance'].items():
        win_rate = (data['wins'] / data['total']) * 100 if data['total'] > 0 else 0
        trend_table.append([trend.capitalize(), data['wins'], data['losses'], f"{win_rate:.2f}%"])
    report += tabulate(trend_table, headers="firstrow", tablefmt="grid")
    
    with open("backtest/backtest_summary.txt", "w") as f:
        f.write(report)
        
    print("Human-readable summary saved to backtest/backtest_summary.txt")

def generate_equity_curve(backtest_trades, starting_balance):
    """
    Generate and save a plot of the equity curve.
    """
    balance_over_time = [starting_balance] + [trade.balance for trade in backtest_trades]
    plt.figure(figsize=(10, 6))
    plt.plot(balance_over_time)
    plt.title('Equity Curve')
    plt.xlabel('Trade Number')
    plt.ylabel('Balance (USD)')
    plt.grid(True)
    plt.savefig('backtest/equity_curve.png')
    plt.close()

def generate_backtest_report(backtest_trades, config, starting_balance):
    """
    Generate a detailed report from the backtest results.
    """
    if not os.path.exists('backtest'):
        os.makedirs('backtest')
    
    if not backtest_trades:
        print("No trades to generate a report for.")
        return

    metrics = calculate_performance_metrics(backtest_trades, starting_balance)
    strategy_analysis = analyze_strategy_behavior(backtest_trades)
    
    report = f"""
Backtesting Report
==================

Configuration:
--------------
Risk per trade: {config['risk_per_trade']}%
Leverage: {config['leverage']}x
ATR Value: {config['atr_value']}
Lookback Candles: {config['lookback_candles']}
Swing Window: {config['swing_window']}

Results:
--------
Starting Balance: ${starting_balance:,.2f}
Ending Balance: ${metrics['net_pnl_usd'] + starting_balance:,.2f}
Total Profit: ${metrics['net_pnl_usd']:,.2f}
Total Trades: {metrics['total_trades']}
Winning Trades: {metrics['winning_trades']}
Losing Trades: {metrics['losing_trades']}
Win Rate: {metrics['win_rate']:.2f}%
Average Win: ${metrics['average_win']:,.2f}
Average Loss: ${metrics['average_loss']:,.2f}
Profit Factor: {metrics['profit_factor']:.2f}
Max Drawdown: {metrics['max_drawdown']:.2f}%
Expectancy: ${metrics['expectancy']:,.2f}

Strategy Behavior Insights:
-------------------------
"""
    report += "\nHourly Performance:\n"
    for hour, data in sorted(strategy_analysis['hourly_performance'].items()):
        win_rate = (data['wins'] / data['total']) * 100 if data['total'] > 0 else 0
        report += f"  Hour {hour:02d}: {data['wins']} wins, {data['losses']} losses, {win_rate:.2f}% win rate\n"
        
    report += "\nTrend Performance:\n"
    for trend, data in strategy_analysis['trend_performance'].items():
        win_rate = (data['wins'] / data['total']) * 100 if data['total'] > 0 else 0
        report += f"  {trend.capitalize()}: {data['wins']} wins, {data['losses']} losses, {win_rate:.2f}% win rate\n"

    report += """
Trade Log:
----------
"""
    for trade in backtest_trades:
        report += f"Timestamp: {datetime.datetime.fromtimestamp(trade.entry_timestamp/1000).strftime('%Y-%m-%d %H:%M:%S')}, Symbol: {trade.symbol}, Side: {trade.side}, Entry: {trade.entry_price:.8f}, Exit: {trade.exit_price:.8f}, Status: {trade.status}, PnL: ${trade.pnl_usd:,.2f} ({trade.pnl_pct:.2f}%), Drawdown: {trade.drawdown:.2f}%\n"
        
    with open("backtest/backtest_report.txt", "w") as f:
        f.write(report)
    
    print("Backtest report generated: backtest/backtest_report.txt")
    generate_equity_curve(backtest_trades, starting_balance)
    generate_drawdown_curve(backtest_trades, starting_balance)
    generate_win_loss_distribution(backtest_trades)
    generate_returns_histogram(backtest_trades)
    generate_csv_report(backtest_trades)
    generate_json_report(backtest_trades, metrics, strategy_analysis)
    generate_summary_report(backtest_trades, metrics, strategy_analysis, config, starting_balance)

def run_backtest(client, symbols, days_to_backtest, config):
    """
    Run the backtesting simulation.
    """
    print("Starting backtest...")
    end_date = datetime.datetime.now(pytz.utc)
    start_date = end_date - datetime.timedelta(days=days_to_backtest)
    
    all_klines = {}
    for symbol in symbols:
        print(f"Fetching historical data for {symbol}...")
        klines = get_klines(client, symbol, interval=Client.KLINE_INTERVAL_15MINUTE, 
                              start_str=start_date.strftime("%Y-%m-%d %H:%M:%S"),
                              end_str=end_date.strftime("%Y-%m-%d %H:%M:%S"))
        if klines:
            all_klines[symbol] = klines
    
    print("Backtest data fetched.")
    
    backtest_trades = []
    balance = config['starting_balance']
    
    for symbol in all_klines:
        print(f"Backtesting {symbol}...")
        klines = all_klines[symbol]
        for i in range(config['lookback_candles'], len(klines)):
            current_klines = klines[i-config['lookback_candles']:i]
            swing_highs, swing_lows = get_swing_points(current_klines, config['swing_window'])
            trend = get_trend(swing_highs, swing_lows)
            
            if trend == "downtrend" and len(swing_highs) > 1 and len(swing_lows) > 1:
                last_swing_high = swing_highs[-1][1]
                last_swing_low = swing_lows[-1][1]
                entry_price = get_fib_retracement(last_swing_high, last_swing_low, trend)
                sl = last_swing_high
                tp1 = entry_price - (sl - entry_price)
                
                # Simulate trade entry
                if float(current_klines[-1][4]) > entry_price:
                    entry_timestamp = current_klines[-1][0]
                    quantity = calculate_quantity(client, symbol, config['risk_per_trade'], sl, entry_price, config['leverage'])
                    if quantity is None or quantity == 0:
                        continue

                    # Simulate trade exit
                    exit_timestamp = 0
                    exit_price = 0
                    status = ''
                    reason_for_exit = ''

                    for j in range(i, len(klines)):
                        future_kline = klines[j]
                        high_price = float(future_kline[2])
                        low_price = float(future_kline[3])

                        if low_price <= tp1:
                            exit_price = tp1
                            status = 'win'
                            reason_for_exit = 'TP1 Hit'
                            exit_timestamp = future_kline[0]
                            break
                        elif high_price >= sl:
                            exit_price = sl
                            status = 'loss'
                            reason_for_exit = 'SL Hit'
                            exit_timestamp = future_kline[0]
                            break
                    
                    if status == '':
                        continue

                    pnl_usd = (entry_price - exit_price) * quantity if status == 'win' else (entry_price - exit_price) * quantity
                    pnl_pct = (pnl_usd / (entry_price * quantity)) * 100
                    balance += pnl_usd
                    
                    trade = TradeResult(
                        symbol=symbol,
                        side='short',
                        entry_price=entry_price,
                        exit_price=exit_price,
                        entry_timestamp=entry_timestamp,
                        exit_timestamp=exit_timestamp,
                        status=status,
                        pnl_usd=pnl_usd,
                        pnl_pct=pnl_pct,
                        drawdown=0, # Simplified for now
                        reason_for_entry=f"Fib retracement in downtrend",
                        reason_for_exit=reason_for_exit,
                        fib_levels=[0, 0.236, 0.382, 0.5, 0.618, 1.0] # Simplified for now
                    )
                    trade.balance = balance
                    backtest_trades.append(trade)
                    i += (j - i) # Move to the next candle after the trade is closed

            elif trend == "uptrend" and len(swing_highs) > 1 and len(swing_lows) > 1:
                last_swing_high = swing_highs[-1][1]
                last_swing_low = swing_lows[-1][1]
                entry_price = get_fib_retracement(last_swing_low, last_swing_high, trend)
                sl = last_swing_low
                tp1 = entry_price + (entry_price - last_swing_low)

                # Simulate trade entry
                if float(current_klines[-1][4]) < entry_price:
                    entry_timestamp = current_klines[-1][0]
                    quantity = calculate_quantity(client, symbol, config['risk_per_trade'], sl, entry_price, config['leverage'])
                    if quantity is None or quantity == 0:
                        continue

                    # Simulate trade exit
                    exit_timestamp = 0
                    exit_price = 0
                    status = ''
                    reason_for_exit = ''

                    for j in range(i, len(klines)):
                        future_kline = klines[j]
                        high_price = float(future_kline[2])
                        low_price = float(future_kline[3])

                        if high_price >= tp1:
                            exit_price = tp1
                            status = 'win'
                            reason_for_exit = 'TP1 Hit'
                            exit_timestamp = future_kline[0]
                            break
                        elif low_price <= sl:
                            exit_price = sl
                            status = 'loss'
                            reason_for_exit = 'SL Hit'
                            exit_timestamp = future_kline[0]
                            break
                    
                    if status == '':
                        continue

                    pnl_usd = (exit_price - entry_price) * quantity if status == 'win' else (exit_price - entry_price) * quantity
                    pnl_pct = (pnl_usd / (entry_price * quantity)) * 100
                    balance += pnl_usd
                    
                    trade = TradeResult(
                        symbol=symbol,
                        side='long',
                        entry_price=entry_price,
                        exit_price=exit_price,
                        entry_timestamp=entry_timestamp,
                        exit_timestamp=exit_timestamp,
                        status=status,
                        pnl_usd=pnl_usd,
                        pnl_pct=pnl_pct,
                        drawdown=0, # Simplified for now
                        reason_for_entry=f"Fib retracement in uptrend",
                        reason_for_exit=reason_for_exit,
                        fib_levels=[0, 0.236, 0.382, 0.5, 0.618, 1.0] # Simplified for now
                    )
                    trade.balance = balance
                    backtest_trades.append(trade)
                    i += (j - i) # Move to the next candle after the trade is closed

    print(f"Backtest complete. Found {len(backtest_trades)} potential trades.")
    return backtest_trades

def update_trade_report(trades, backtest_mode=False):
    """
    Update the trade report JSON file.
    """
    if not backtest_mode:
        with open('trades.json', 'w') as f:
            json.dump(trades, f, indent=4, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else o)

async def place_real_order(client, symbol, side, quantity, bot, live_mode=False):
    """
    Place a real order on Binance.
    """
    if not live_mode:
        print(f"VIRTUAL ORDER: Placing {side} order for {quantity} {symbol}")
        return {"status": "success"}
    try:
        if side.lower() == 'long':
            order_side = Client.SIDE_BUY
        elif side.lower() == 'short':
            order_side = Client.SIDE_SELL
        else:
            raise ValueError("Side must be 'long' or 'short'")
            
        order = client.create_order(
            symbol=symbol,
            side=order_side,
            type=Client.ORDER_TYPE_MARKET,
            quantity=quantity
        )
        print(f"Placed {side} order for {quantity} {symbol}: {order}")
        return {"status": "success", "order": order}
    except Exception as e:
        print(f"Error placing real order for {symbol}: {e}")
        try:
            await bot.send_message(chat_id=keys.telegram_chat_id, text=f"⚠️ ORDER FAILED ⚠️\nSymbol: {symbol}\nSide: {side}\nQuantity: {quantity}\nError: {e}")
        except Exception as telegram_e:
            print(f"Error sending Telegram message: {telegram_e}")
        return {"status": "error", "message": str(e)}

async def send_start_message(bot, backtest_mode=False):
    if backtest_mode:
        return
    try:
        await bot.send_message(chat_id=keys.telegram_chat_id, text="🤖 Bot started!")
    except Exception as e:
        print(f"Error sending start message: {e}")

async def send_backtest_summary(bot, metrics, backtest_trades, starting_balance):
    """
    Send a summary of the backtest results to Telegram.
    """
    summary_text = f"""
*Backtest Summary*
-------------------
*Total Trades:* {metrics['total_trades']}
*Win Rate:* {metrics['win_rate']:.2f}%
*Net PnL:* ${metrics['net_pnl_usd']:,.2f} ({metrics['net_pnl_pct']:.2f}%)
*Profit Factor:* {metrics['profit_factor']:.2f}
*Max Drawdown:* {metrics['max_drawdown']:.2f}%
"""
    try:
        await bot.send_message(chat_id=keys.telegram_chat_id, text=summary_text, parse_mode='Markdown')
        with open('backtest/equity_curve.png', 'rb') as photo:
            await bot.send_photo(chat_id=keys.telegram_chat_id, photo=photo, caption="Equity Curve")
        with open('backtest/backtest_trades.csv', 'rb') as document:
            await bot.send_document(chat_id=keys.telegram_chat_id, document=document, filename='backtest_trades.csv')
    except Exception as e:
        print(f"Error sending backtest summary to Telegram: {e}")

async def send_backtest_complete_message(bot):
    """
    Send a message to Telegram to indicate that the backtest is complete.
    """
    try:
        await bot.send_message(chat_id=keys.telegram_chat_id, text="✅ Backtest is done.")
    except Exception as e:
        print(f"Error sending backtest complete message: {e}")

async def send_market_analysis_image(bot, chat_id, image_buffer, caption, backtest_mode=False):
    """
    Send the market analysis image to Telegram.
    """
    if backtest_mode:
        return
    try:
        image_buffer.seek(0)
        await bot.send_photo(chat_id=chat_id, photo=image_buffer, caption=caption)
    except Exception as e:
        print(f"Error sending market analysis image: {e}")
        # Fallback to text message
        await bot.send_message(chat_id=chat_id, text=f"Error generating chart. {caption}")

async def op_command(update, context):
    """
    Send the latest chart for all open trades.
    """
    with trades_lock:
        open_trades = [trade for trade in trades if trade['status'] in ['running', 'tp1_hit', 'tp2_hit']]
    
    if not open_trades:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="No open trades.")
        return

    client = Client(keys.api_mainnet, keys.secret_mainnet)

    for trade in open_trades:
        symbol = trade['symbol']
        klines = get_klines(client, symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=100)
        if klines:
            swing_highs, swing_lows = get_swing_points(klines, 5)
            if not swing_highs or not swing_lows:
                continue

            last_swing_high = swing_highs[-1][1]
            last_swing_low = swing_lows[-1][1]
            
            image_buffer = generate_fib_chart(symbol, klines, trade['side'], last_swing_high, last_swing_low, trade['entry_price'], trade['sl'], trade['tp1'], trade['tp2'])
            current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
            caption = f"Open Trade: {symbol}\nSide: {trade['side']}\nEntry: {trade['entry_price']:.8f}\nCurrent Price: {current_price:.8f}\nSL: {trade['sl']:.8f}\nTP1: {trade['tp1']:.8f}"
            await context.bot.send_photo(chat_id=update.effective_chat.id, photo=image_buffer, caption=caption)

async def order_status_monitor(client, application, backtest_mode=False, live_mode=False):
    """
    Continuously monitor the status of open and pending trades.
    """
    if backtest_mode:
        return
    print("Order status monitor started.")
    bot = application.bot
    while True:
        try:
            with trades_lock:
                if not trades:
                    await asyncio.sleep(1)
                    continue

                print(f"Monitor: Checking {len(trades)} trades.")
                for trade in trades:
                    if trade['status'] not in ['pending', 'running', 'tp1_hit', 'tp2_hit']:
                        continue

                    symbol = trade['symbol']
                    print(f"Monitor: Checking {symbol}.")
                    # Fetch the last 5 seconds of k-line data
                    klines = get_klines(client, symbol, interval=Client.KLINE_INTERVAL_1SECOND, limit=5)
                    if not klines:
                        continue

                    # Check for expired orders
                    if time.time() * 1000 - trade['timestamp'] > 4 * 60 * 60 * 1000:
                        if trade['status'] == 'pending':
                            try:
                                await bot.send_message(chat_id=keys.telegram_chat_id, text=f"⚠️ TRADE INVALIDATED ⚠️\nSymbol: {symbol}\nSide: {trade['side']}\nReason: Order expired (4 hours)")
                            except Exception as e:
                                print(f"Error sending Telegram message: {e}")
                            trade['status'] = 'rejected'
                            update_trade_report(trades)
                            if symbol in virtual_orders:
                                del virtual_orders[symbol]
                        continue

                    prices = [float(k[4]) for k in klines]

                    for price in prices:
                        # Check for SL/TP hits
                        if trade['status'] in ['running', 'tp1_hit', 'tp2_hit']:
                            if (trade['side'] == 'long' and price <= trade['sl']) or \
                               (trade['side'] == 'short' and price >= trade['sl']):
                                try:
                                    klines_for_chart = get_klines(client, symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=100)
                                    if klines_for_chart:
                                        image_buffer = generate_fib_chart(symbol, klines_for_chart, trade['side'], trade['entry_price'] + (trade['entry_price'] - trade['sl']), trade['sl'], trade['entry_price'], trade['sl'], trade['tp1'], trade['tp2'])
                                        caption = f"🛑 STOP LOSS HIT 🛑\nSymbol: {symbol}\nSide: {trade['side']}\nPrice: {price:.8f}"
                                        await send_market_analysis_image(bot, keys.telegram_chat_id, image_buffer, caption)
                                    else:
                                        await bot.send_message(chat_id=keys.telegram_chat_id, text=f"🛑 STOP LOSS HIT 🛑\nSymbol: {symbol}\nSide: {trade['side']}\nPrice: {price:.8f}")
                                except Exception as e:
                                    print(f"Error sending Telegram message: {e}")
                                trade['status'] = 'sl_hit'
                                rejected_symbols[symbol] = time.time()
                                update_trade_report(trades)
                                if symbol in virtual_orders:
                                    del virtual_orders[symbol]
                                break

                            if trade['status'] == 'running':
                                if (trade['side'] == 'long' and price >= trade['entry_price'] * 1.005) or \
                                   (trade['side'] == 'short' and price <= trade['entry_price'] * 0.995):
                                    new_sl = trade['entry_price'] * 1.001 if trade['side'] == 'long' else trade['entry_price'] * 0.999
                                    if (trade['side'] == 'long' and new_sl > trade['sl']) or \
                                       (trade['side'] == 'short' and new_sl < trade['sl']):
                                        trade['sl'] = new_sl
                                        update_trade_report(trades)
                                        try:
                                            await bot.send_message(chat_id=keys.telegram_chat_id, text=f"🔒 STOP LOSS UPDATED 🔒\nSymbol: {symbol}\nSide: {trade['side']}\nNew SL: {trade['sl']:.8f}")
                                        except Exception as e:
                                            print(f"Error sending Telegram message: {e}")

                            if trade['status'] == 'running' and ((trade['side'] == 'long' and price >= trade['tp1']) or \
                               (trade['side'] == 'short' and price <= trade['tp1'])):
                                trade['status'] = 'tp1_hit'
                                trade['sl'] = trade['entry_price']
                                trade['quantity'] *= 0.5
                                update_trade_report(trades)
                                try:
                                    klines_for_chart = get_klines(client, symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=100)
                                    if klines_for_chart:
                                        image_buffer = generate_fib_chart(symbol, klines_for_chart, trade['side'], trade['entry_price'] + (trade['entry_price'] - trade['sl']), trade['sl'], trade['entry_price'], trade['sl'], trade['tp1'], trade['tp2'])
                                        caption = f"🎉 TAKE PROFIT 1 HIT 🎉\nSymbol: {symbol}\nSide: {trade['side']}\nPrice: {price:.8f}\nClosing 50% of the position.\nNew SL: {trade['sl']:.8f}"
                                        await send_market_analysis_image(bot, keys.telegram_chat_id, image_buffer, caption)
                                    else:
                                        await bot.send_message(chat_id=keys.telegram_chat_id, text=f"🎉 TAKE PROFIT 1 HIT 🎉\nSymbol: {symbol}\nSide: {trade['side']}\nPrice: {price:.8f}\nClosing 50% of the position.\nNew SL: {trade['sl']:.8f}")
                                except Exception as e:
                                    print(f"Error sending Telegram message: {e}")

                            if trade['status'] == 'tp1_hit' and ((trade['side'] == 'long' and price >= trade['tp2']) or \
                               (trade['side'] == 'short' and price <= trade['tp2'])):
                                trade['status'] = 'tp2_hit'
                                trade['sl'] = trade['tp1']
                                trade['quantity'] *= 0.6
                                update_trade_report(trades)
                                try:
                                    klines_for_chart = get_klines(client, symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=100)
                                    if klines_for_chart:
                                        image_buffer = generate_fib_chart(symbol, klines_for_chart, trade['side'], trade['entry_price'] + (trade['entry_price'] - trade['sl']), trade['sl'], trade['entry_price'], trade['sl'], trade['tp1'], trade['tp2'])
                                        caption = f"🎉 TAKE PROFIT 2 HIT 🎉\nSymbol: {symbol}\nSide: {trade['side']}\nPrice: {price:.8f}\nClosing 40% of the position.\nNew SL: {trade['sl']:.8f}"
                                        await send_market_analysis_image(bot, keys.telegram_chat_id, image_buffer, caption)
                                    else:
                                        await bot.send_message(chat_id=keys.telegram_chat_id, text=f"🎉 TAKE PROFIT 2 HIT 🎉\nSymbol: {symbol}\nSide: {trade['side']}\nPrice: {price:.8f}\nClosing 40% of the position.\nNew SL: {trade['sl']:.8f}")
                                except Exception as e:
                                    print(f"Error sending Telegram message: {e}")

                        # Check for triggered orders
                        elif trade['status'] == 'pending':
                            if (trade['side'] == 'long' and price >= trade['entry_price']) or \
                               (trade['side'] == 'short' and price <= trade['entry_price']):
                                
                                # In a real scenario, you would place a limit order here
                                if live_mode:
                                    await place_real_order(client, symbol, trade['side'], trade['quantity'], bot, live_mode=live_mode)
                                trade['status'] = 'running'
                                update_trade_report(trades)
                                try:
                                    await bot.send_message(chat_id=keys.telegram_chat_id, text=f"✅ TRADE TRIGGERED ✅\nSymbol: {symbol}\nEntry: {trade['entry_price']:.8f}\nSide: {trade['side']}\nTP1: {trade['tp1']:.8f}\nTP2: {trade['tp2']:.8f}\nTP3: {trade['tp3']:.8f}\nSL: {trade['sl']:.8f}\nLeverage: {leverage}x")
                                except Exception as e:
                                    print(f"Error sending Telegram message: {e}")
                                break
        except Exception as e:
            print(f"Error in order status monitor: {e}")

        await asyncio.sleep(1)


# Global variables for trades
virtual_orders = {}
trades = []
leverage = 0
trades_lock = threading.Lock()
rejected_symbols = {}

async def main():
    """
    Main function to run the Binance trading bot.
    """
    print("Starting bot...")
    backtest_mode = False
    
    # Get and print public IP address
    public_ip = get_public_ip()
    if public_ip:
        print(f"Public IP Address: {public_ip}")
    else:
        print("Could not determine public IP address.")
    
    bot = telegram.Bot(token=keys.telegram_bot_token)
    # Send start message
    await send_start_message(bot, backtest_mode)

    # Load configuration
    global leverage
    try:
        config = pd.read_csv('configuration.csv').iloc[0]
        risk_per_trade = config['risk_per_trade']
        leverage = config['leverage']
        atr_value = int(config['atr_value'])
        lookback_candles = int(config['lookback_candles'])
        swing_window = int(config['swing_window'])
        starting_balance = int(config['starting_balance'])
        print("Configuration loaded.")
    except FileNotFoundError:
        print("Error: configuration.csv not found.")
        return
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # Load symbols
    try:
        symbols = pd.read_csv('symbols.csv', header=None)[0].tolist()
        print("Symbols loaded.")
    except FileNotFoundError:
        print("Error: symbols.csv not found.")
        return
    except Exception as e:
        print(f"Error loading symbols: {e}")
        return

    # Initialize Binance client
    try:
        client = Client(keys.api_mainnet, keys.secret_mainnet)
        print("Binance client initialized.")
    except Exception as e:
        print(f"Error initializing Binance client: {e}")
        return

    # Check API key permissions
    try:
        account_info = client.futures_account()
        if not account_info['canTrade']:
            print("Error: API key does not have permission to trade futures.")
            return
    except BinanceAPIException as e:
        print(f"Error checking API key permissions: {e}")
        return

    # Set up the Telegram bot
    application = ApplicationBuilder().token(keys.telegram_bot_token).build()
    
    # Start the order status monitor in a separate thread
    def run_monitor(backtest_mode, live_mode):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(order_status_monitor(client, application, backtest_mode, live_mode))

    op_handler = CommandHandler('op', op_command)
    application.add_handler(op_handler)

    # Get user input for mode
    while True:
        mode = input("Select (1)Live / (2)Signal / (3)Backtest: ")
        if mode in ['1', '2', '3']:
            break
        else:
            print("Invalid input. Please select 1, 2, or 3.")

    if mode == '1':
        print("Running in Live mode.")
        live_mode = True
        backtest_mode = False
        try:
            account_info = client.futures_account()
            balance = float(account_info['totalWalletBalance'])
            await bot.send_message(chat_id=keys.telegram_chat_id, text=f"Futures Account Balance: {balance:.2f} USDT")
        except BinanceAPIException as e:
            print(f"Error fetching account balance: {e}")
            await bot.send_message(chat_id=keys.telegram_chat_id, text=f"Error fetching account balance: {e}")
            return
    elif mode == '2':
        print("Running in Signal mode.")
        live_mode = False
        backtest_mode = False
    else:
        print("Running in Backtest mode.")
        live_mode = False
        backtest_mode = True
        while True:
            try:
                days_to_backtest = int(input("Enter the number of days to backtest: "))
                if days_to_backtest > 0:
                    break
                else:
                    print("Please enter a positive number of days.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    # Start the Telegram bot
    def run_bot():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(application.run_polling())

    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()

    # Load trades from JSON
    if not backtest_mode and os.path.exists('trades.json'):
        with open('trades.json', 'r') as f:
            try:
                loaded_trades = json.load(f)
                with trades_lock:
                    trades.extend(loaded_trades)
                for trade in loaded_trades:
                    if trade['status'] in ['running', 'tp1_hit', 'tp2_hit', 'pending']:
                        virtual_orders[trade['symbol']] = trade
            except json.JSONDecodeError:
                pass
    
    monitor_thread = threading.Thread(target=run_monitor, args=(backtest_mode, live_mode), daemon=True)
    monitor_thread.start()


    if backtest_mode:
        config_dict = {
            'risk_per_trade': risk_per_trade,
            'leverage': leverage,
            'atr_value': atr_value,
            'lookback_candles': lookback_candles,
            'swing_window': swing_window,
            'starting_balance': starting_balance
        }
        backtest_trades = run_backtest(client, symbols, days_to_backtest, config_dict)
        metrics = calculate_performance_metrics(backtest_trades, starting_balance)
        strategy_analysis = analyze_strategy_behavior(backtest_trades)
        generate_backtest_report(backtest_trades, config_dict, starting_balance)
        await send_backtest_summary(bot, metrics, backtest_trades, starting_balance)
        await send_backtest_complete_message(bot)
        # The script will exit after the backtest is complete.
    else:
        # Main scanning loop
        print("Entering main loop...")
        while True:
            print("Starting new scan cycle...")
            for symbol in symbols:
                try:
                    # Check for rejected symbols cooldown
                    if symbol in rejected_symbols and time.time() - rejected_symbols[symbol] < 4 * 60 * 60:
                        continue

                    # Check if there is already an open or pending trade for this symbol
                    if symbol in virtual_orders:
                        continue

                    print(f"Scanning {symbol}...")
                    klines = get_klines(client, symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=lookback_candles)
                    if not klines:
                        continue

                    swing_highs, swing_lows = get_swing_points(klines, swing_window)
                    trend = get_trend(swing_highs, swing_lows)

                    # Check for new signals
                    current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                    if trend == "downtrend" and len(swing_highs) > 1 and len(swing_lows) > 1:
                        if time.time() * 1000 - swing_highs[-1][0] > 4 * 60 * 60 * 1000:
                            continue
                        last_swing_high = swing_highs[-1][1]
                        last_swing_low = swing_lows[-1][1]
                        entry_price = get_fib_retracement(last_swing_high, last_swing_low, trend)
                        if current_price < entry_price:
                            continue
                        
                        sl = last_swing_high
                        tp1 = entry_price - (sl - entry_price)
                        tp2 = entry_price - (sl - entry_price) * 2
                        tp3 = 0 # Floating TP
                        
                        quantity = calculate_quantity(client, symbol, risk_per_trade, sl, entry_price, leverage)
                        if quantity is None or quantity == 0:
                            continue
                        
                        image_buffer = generate_fib_chart(symbol, klines, trend, last_swing_high, last_swing_low, entry_price, sl, tp1, tp2)
                        caption = f"🚀 NEW TRADE SIGNAL 🚀\nSymbol: {symbol}\nSide: Short\nLeverage: {leverage}x\nRisk : {risk_per_trade}%\nProposed Entry: {entry_price:.8f}\nStop Loss: {sl:.8f}\nTake Profit 1: {tp1:.8f}\nTake Profit 2: {tp2:.8f}\nTake Profit 3: Floating"
                        await send_market_analysis_image(bot, keys.telegram_chat_id, image_buffer, caption, backtest_mode)

                        new_trade = {'symbol': symbol, 'side': 'short', 'entry_price': entry_price, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3, 'status': 'pending', 'quantity': quantity, 'timestamp': klines[-1][0]}
                        with trades_lock:
                            trades.append(new_trade)
                        virtual_orders[symbol] = new_trade
                        update_trade_report(trades)

                    elif trend == "uptrend" and len(swing_highs) > 1 and len(swing_lows) > 1:
                        if time.time() * 1000 - swing_lows[-1][0] > 4 * 60 * 60 * 1000:
                            continue
                        last_swing_high = swing_highs[-1][1]
                        last_swing_low = swing_lows[-1][1]
                        entry_price = get_fib_retracement(last_swing_low, last_swing_high, trend)
                        if current_price > entry_price:
                            continue

                        sl = last_swing_low
                        tp1 = entry_price + (entry_price - last_swing_low)
                        tp2 = entry_price + (entry_price - last_swing_low) * 2
                        tp3 = 0 # Floating TP

                        quantity = calculate_quantity(client, symbol, risk_per_trade, sl, entry_price, leverage)
                        if quantity is None or quantity == 0:
                            continue

                        image_buffer = generate_fib_chart(symbol, klines, trend, last_swing_high, last_swing_low, entry_price, sl, tp1, tp2)
                        caption = f"🚀 NEW TRADE SIGNAL 🚀\nSymbol: {symbol}\nSide: Long\nLeverage: {leverage}x\nRisk : {risk_per_trade}%\nProposed Entry: {entry_price:.8f}\nStop Loss: {sl:.8f}\nTake Profit 1: {tp1:.8f}\nTake Profit 2: {tp2:.8f}\nTake Profit 3: Floating"
                        await send_market_analysis_image(bot, keys.telegram_chat_id, image_buffer, caption, backtest_mode)

                        new_trade = {'symbol': symbol, 'side': 'long', 'entry_price': entry_price, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3, 'status': 'pending', 'quantity': quantity, 'timestamp': klines[-1][0]}
                        with trades_lock:
                            trades.append(new_trade)
                        virtual_orders[symbol] = new_trade
                        update_trade_report(trades)
                except Exception as e:
                    print(f"Error scanning {symbol}: {e}")
                    rejected_symbols[symbol] = time.time() # Add to rejected list to avoid spamming errors

            print("Scan cycle complete. Cooling down for 2 minutes...")
            await asyncio.sleep(120)

if __name__ == "__main__":
    asyncio.run(main())
