# ======================================================
# run_backtest.py
# V8H CORE — Professional Backtest Runner & Analyzer
# ======================================================

import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt

from backtest_engine import BacktestEngine

# ------------------------------------------------------
# CLEAN ENVIRONMENT
# ------------------------------------------------------
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------
DATA_PATH = r"C:\Users\CIT\Documents\MT5_AI\V8H-CORE\data\clean"
SYMBOL = "GOLD#"
INITIAL_BALANCE = 1000.0
TIMEFRAME = "M15"   # M1 | M5 | M15

# ------------------------------------------------------
# DATA LOADER
# ------------------------------------------------------
def load_data(tf: str) -> pd.DataFrame:
    candidates = [
        f"{tf}.csv",
        f"GOLD_{tf}.csv",
        f"{tf}.parquet",
        f"GOLD_{tf}.parquet",
    ]

    for file in candidates:
        path = os.path.join(DATA_PATH, file)
        if os.path.exists(path):
            print(f"[OK] Loading data: {file}")
            if file.endswith(".csv"):
                return pd.read_csv(path)
            return pd.read_parquet(path)

    raise FileNotFoundError(f"No data found for timeframe {tf}")

# ------------------------------------------------------
# ADVANCED REPORT
# ------------------------------------------------------
def generate_advanced_report(history: pd.DataFrame, initial_balance: float):
    closes = history[history.event == "close"].copy()
    if closes.empty:
        print("[WARN] No closed trades")
        return

    pnl = closes.pnl

    gross_profit = pnl[pnl > 0].sum()
    gross_loss = pnl[pnl < 0].sum()
    net_profit = gross_profit + gross_loss

    win_trades = closes[pnl > 0]
    loss_trades = closes[pnl < 0]

    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")
    expectancy = pnl.mean()

    avg_win = win_trades.pnl.mean() if not win_trades.empty else 0
    avg_loss = loss_trades.pnl.mean() if not loss_trades.empty else 0

    max_win = pnl.max()
    max_loss = pnl.min()

    max_dd = closes.dd.max()
    recovery_factor = (
        net_profit / (initial_balance * max_dd / 100)
        if max_dd > 0 else float("inf")
    )

    # Consecutive Loss
    loss_streak = 0
    max_loss_streak = 0
    for p in pnl:
        if p < 0:
            loss_streak += 1
            max_loss_streak = max(max_loss_streak, loss_streak)
        else:
            loss_streak = 0

    print("\n============== ADVANCED PERFORMANCE REPORT ==============\n")
    print(f"Net Profit          : {net_profit:,.2f}")
    print(f"Gross Profit        : {gross_profit:,.2f}")
    print(f"Gross Loss          : {gross_loss:,.2f}")
    print(f"Profit Factor       : {profit_factor:.2f}")
    print(f"Expectancy          : {expectancy:.2f} / trade")
    print(f"Average Win         : {avg_win:.2f}")
    print(f"Average Loss        : {avg_loss:.2f}")
    print(f"Max Win             : {max_win:.2f}")
    print(f"Max Loss            : {max_loss:.2f}")
    print(f"Max Drawdown        : {max_dd:.2f}%")
    print(f"Recovery Factor     : {recovery_factor:.2f}")
    print(f"Max Loss Streak     : {max_loss_streak}")

# ------------------------------------------------------
# PLOT RESULT
# ------------------------------------------------------
def plot_result(history: pd.DataFrame):
    closes = history[history.event == "close"].copy()
    if closes.empty:
        print("[WARN] No trades to plot")
        return

    closes["trade_id"] = range(1, len(closes) + 1)

    fig, axs = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

    # Equity Curve
    axs[0].plot(closes.trade_id, closes.equity, linewidth=2)
    axs[0].set_title("Equity Curve")
    axs[0].grid(True)

    # PnL per Trade
    axs[1].bar(closes.trade_id, closes.pnl)
    axs[1].axhline(0)
    axs[1].set_title("PnL per Trade")
    axs[1].grid(True)

    # Drawdown
    axs[2].plot(closes.trade_id, closes.dd)
    axs[2].set_title("Drawdown (%)")
    axs[2].grid(True)

    plt.xlabel("Trade #")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
def main():
    print("=" * 60)
    print("V8H CORE — BACKTEST & ANALYSIS ENGINE")
    print("=" * 60)

    # Load Data
    df = load_data(TIMEFRAME)

    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(df.columns):
        raise ValueError("Data missing OHLC columns")

    print(f"[INFO] Rows loaded: {len(df):,}")

    # Run Backtest
    engine = BacktestEngine(
        symbol=SYMBOL,
        initial_balance=INITIAL_BALANCE,
    )

    result = engine.run(df)

    # Basic Report
    print("\n================ BASIC BACKTEST REPORT ================\n")
    print(f"Symbol            : {SYMBOL}")
    print(f"Timeframe         : {TIMEFRAME}")
    print(f"Initial Balance   : ${result['initial_balance']:.2f}")
    print(f"Final Balance     : ${result['final_balance']:.2f}")
    print(f"Max Equity        : ${result['max_equity']:.2f}")
    print(f"Max Drawdown      : {result['max_dd']:.2f}%")
    print(f"Total Trades      : {result['trade_count']}")
    print(f"Win Rate          : {result['winrate']:.2f}%")

    # Save History
    history = result["history"]
    out_file = f"backtest_{SYMBOL}_{TIMEFRAME}.csv"
    history.to_csv(out_file, index=False)
    print(f"\n[OK] Trade history saved to {out_file}")

    # Advanced Report
    generate_advanced_report(history, INITIAL_BALANCE)

    # Plot
    plot_result(history)

# ------------------------------------------------------
if __name__ == "__main__":
    main()