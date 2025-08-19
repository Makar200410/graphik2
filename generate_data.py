from pathlib import Path
from datetime import datetime, timezone, timedelta
import time
import signal
import subprocess
import sys
import json

import pandas as pd
import yfinance as yf

# ---------- Configuration ----------
TICKERS = ["AAPL", "MSFT", "GOOG", "NVDA", "AMZN"]  # List of tickers to track
INTERVAL = "1m"  # '1m' for intraday minute bars, or '1d' for daily bars
DATA_DIR = Path("data")  # Directory to save the JSON files
LOOP_SECONDS = 60  # Run the update cycle every 60 seconds
JSON_COMPACT = True  # True = save compact JSON; False = save human-readable JSON
# ------------------------------------
#nef comment
# --- Graceful shutdown handling ---
stop_event = False


def _handle_stop_signal(signum, frame):
    """Signal handler to initiate a graceful shutdown."""
    global stop_event
    stop_event = True
    print("\nStop signal received, finishing current cycle...")


signal.signal(signal.SIGINT, _handle_stop_signal)  # Handle Ctrl+C
try:
    signal.signal(signal.SIGTERM, _handle_stop_signal)  # Handle termination signals
except AttributeError:
    pass  # SIGTERM is not available on all platforms (e.g., Windows)


def to_epoch_ms(ts: pd.Timestamp) -> int:
    """Converts a pandas Timestamp to a UTC epoch millisecond integer."""
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def load_existing_json(path: Path, ticker: str) -> list:
    """Loads and returns the list of data points from an existing JSON file."""
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Support both {"TICKER": [...]} and raw [...] formats
        if isinstance(data, dict) and ticker in data and isinstance(data[ticker], list):
            return data[ticker]
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, IOError):
        pass
    return []


def save_json_atomic(path: Path, ticker: str, items: list):
    """Saves data to a JSON file atomically to prevent corrupted files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    obj_to_save = {ticker: items}
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as f:
        if JSON_COMPACT:
            json.dump(obj_to_save, f, ensure_ascii=False, separators=(",", ":"))
        else:
            json.dump(obj_to_save, f, ensure_ascii=False, indent=2)
            f.write("\n")
    tmp_path.replace(path)


def merge_ohlc_data(existing_data: list, new_data: list) -> list:
    """Merges new OHLC data into existing data, overwriting duplicates by timestamp."""
    data_by_timestamp = {item['t']: item for item in existing_data if isinstance(item, dict) and 't' in item}
    for item in new_data:
        if isinstance(item, dict) and 't' in item:
            data_by_timestamp[item['t']] = item

    merged_list = list(data_by_timestamp.values())
    merged_list.sort(key=lambda x: x['t'])
    return merged_list


def fetch_and_save_json_data(ticker: str, is_intraday: bool) -> bool:
    """
    Fetches stock data, processes it into the required JSON format, and saves it.
    Returns True if the file content changed, False otherwise.
    """
    if is_intraday:
        path = DATA_DIR / f"{ticker}_1m.json"
        # Fetch last 2 days of 1-minute data to catch up on recent activity
        df = yf.download(ticker, period="2d", interval="1m", progress=False, auto_adjust=False)
    else:  # Daily
        path = DATA_DIR / f"{ticker}.json"
        existing_data = load_existing_json(path, ticker)
        start_date = "2000-01-01"
        if existing_data:
            last_ms = existing_data[-1]['t']
            last_date_utc = pd.to_datetime(last_ms, unit="ms", utc=True).date()
            start_date = (last_date_utc + timedelta(days=1)).isoformat()
        df = yf.download(ticker, start=start_date, interval="1d", progress=False, auto_adjust=False)

    if df.empty:
        print(f"[{ticker}] No new data returned (market might be closed).")
        return False

    # --- Efficiently convert DataFrame to list of dicts ---
    df.reset_index(inplace=True)
    # yfinance uses 'Datetime' for intraday and 'Date' for daily
    timestamp_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
    df.rename(columns={timestamp_col: 'Timestamp'}, inplace=True)

    df = df[["Timestamp", "Open", "High", "Low", "Close"]].dropna(how="any")
    df['t'] = df['Timestamp'].apply(to_epoch_ms)
    df.rename(columns={'Open': 'o', 'High': 'h', 'Low': 'l', 'Close': 'c'}, inplace=True)

    new_items = df[['t', 'o', 'h', 'l', 'c']].to_dict('records')
    # --- End of conversion block ---

    existing_items = load_existing_json(path, ticker)
    combined_items = merge_ohlc_data(existing_items, new_items)

    if combined_items != existing_items:
        save_json_atomic(path, ticker, combined_items)
        new_rows = len(combined_items) - len(existing_items)
        print(f"[{ticker}] Updated file with {new_rows} new/changed bars.")
        return True
    else:
        print(f"[{ticker}] No changes detected.")
        return False


def git_has_changes(paths=(str(DATA_DIR),)) -> bool:
    """Checks if there are any uncommitted changes in the specified paths."""
    cmd = ["git", "status", "--porcelain"] + list(paths)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return bool(result.stdout.strip())


def git_commit_and_push(message: str, paths=(str(DATA_DIR),)):
    """Adds, commits, and pushes changes to the remote repository."""
    try:
        subprocess.run(["git", "add"] + list(paths), check=True)
        # Only commit if there are staged changes
        if git_has_changes(paths):
            subprocess.run(["git", "commit", "-m", message], check=True)
            print("Committing changes...")
            subprocess.run(["git", "push"], check=True)
            print("Successfully pushed changes to remote repository.")
        else:
            print("No changes to commit.")
    except subprocess.CalledProcessError as e:
        print(f"A Git command failed: {e}")
        print("Please check your Git configuration and credentials.")


def sleep_until_next_minute():
    """Waits until the beginning of the next minute to synchronize the loop."""
    now = time.time()
    next_run = (int(now // 60) + 1) * 60
    time.sleep(max(0, next_run - now))


def main():
    """Main loop to fetch data and push to Git."""
    print(f"Starting data fetch loop: INTERVAL={INTERVAL} | Tickers={TICKERS}")
    print(f"Will push to Git every {LOOP_SECONDS} seconds if changes are detected.")
    print("Press Ctrl+C to stop gracefully.")
    DATA_DIR.mkdir(exist_ok=True)

    while not stop_event:
        cycle_start_time = time.time()
        any_file_changed = False
        is_intraday = INTERVAL == "1m"

        for ticker in TICKERS:
            try:
                changed = fetch_and_save_json_data(ticker, is_intraday)
                if changed:
                    any_file_changed = True
            except Exception as e:
                print(f"[{ticker}] An error occurred: {e}")

        if any_file_changed:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            commit_message = f"Automated data update ({INTERVAL}) at {ts}"
            git_commit_and_push(commit_message)

        if stop_event:
            break

        # Wait for the next cycle
        elapsed = time.time() - cycle_start_time
        sleep_duration = max(0, LOOP_SECONDS - elapsed)
        time.sleep(sleep_duration)

    print("Shutdown initiated. Performing one final check for unsaved changes...")
    if git_has_changes():
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        commit_message = f"Final data update before shutdown at {ts}"
        git_commit_and_push(commit_message)

    print("Exited cleanly.")


if __name__ == "__main__":
    # Pre-flight check: Ensure we are inside a Git repository
    try:
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            check=True, capture_output=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: This script must be run from within a Git repository.")
        sys.exit(1)

    main()
