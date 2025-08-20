import json
import time
import math
import random
import os
import subprocess
from datetime import datetime, timezone, timedelta

# ===================== CONFIG =====================
STATE_FILE = "engine_state.json"
DATA_DIR = "data"  # Directory to store individual stock history files
SYMBOLS = ["AAPL", "GOOG", "AMZN", "MSFT", "NVDA"]
DAYS_TO_KEEP = 30
ROLLING_WINDOW_MINUTES = DAYS_TO_KEEP * 24 * 60  # 30 days of 1-min bars
TICK_SIZE = 0.00001  # minimum price step (CHANGED)
PUSH_TO_GIT = True  # set False if you don't want auto-push

# Timeframe bounds (max movement from timeframe open)
TF_BOUNDS = {
    # M1 bounds changed to reflect smaller price fluctuations.
    # A range of 0.01% to 0.5% from the open. For a $150 stock,
    # this allows for movement between ~$0.015 and ~$0.75,
    # aligning with the spirit of the request for smaller M1 candles.
    "m1": (0.0001, 0.005),
    "m5": (0.005, 0.07),  # 0.5% - 7%
    "h1": (0.007, 0.15),  # 0.7% - 15%
    "d1": (0.01, 0.40),  # 1% - 40%
}

# GARCH-like volatility parameters (stochastic vol)
GARCH_OMEGA = 1e-7
GARCH_ALPHA = 0.06
GARCH_BETA = 0.88
GARCH_GAMMA = 0.06  # leverage


# ===================== UTIL =====================

def floor_minute(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)


def round_to_tick(x: float, tick=TICK_SIZE) -> float:
    # Quantize to nearest tick (avoid float error), updated precision
    return round(round(x / tick) * tick, 5)


def fmt_ts(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def seasonality_scale(minute_of_day: int) -> float:
    # Intraday U-shaped seasonality (open/close > midday)
    x = 2.0 * math.pi * (minute_of_day / 1440.0)
    return 0.75 + 0.5 * (math.cos(x) ** 2)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ===================== MODEL =====================

class SyntheticStream:
    """
    Modern-ish synthetic generator:
      - Stochastic volatility (GARCH-like)
      - Intraday seasonality
      - Slow drift/regime via OU process
      - Rare, small jumps
      - Price bounded by intersection of timeframe channels
    """

    def __init__(self, initial_price=100.0):
        self.p = float(initial_price)
        # Anchors for channel bounds
        self.d1_open = self.p
        self.h1_open = self.p
        self.m5_open = self.p
        self.m1_open = self.p

        # Channel widths (dynamic within bounds)
        self.band = {
            "m1": random.uniform(*TF_BOUNDS["m1"]),
            "m5": random.uniform(*TF_BOUNDS["m5"]),
            "h1": random.uniform(*TF_BOUNDS["h1"]),
            "d1": random.uniform(*TF_BOUNDS["d1"]),
        }

        # Volatility + drift state
        self.garch_variance = 1e-5
        self.prev_return = 0.0
        self.boundary_trend = 0  # -1,0,1 when hugging upper/lower bound
        self.trend_ou = 0.0  # slow trend (OU)

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, data):
        inst = cls()
        inst.__dict__.update(data)
        return inst

    def _update_vol(self):
        # GARCH with leverage
        leverage_effect = GARCH_GAMMA * (self.prev_return ** 2 if self.prev_return < 0 else 0.0)
        new_var = GARCH_OMEGA + GARCH_ALPHA * (
                    self.prev_return ** 2) + leverage_effect + GARCH_BETA * self.garch_variance
        self.garch_variance = max(1e-10, new_var)
        return math.sqrt(self.garch_variance)

    def _update_trend_ou(self, dt_minutes=1.0):
        # Ornstein-Uhlenbeck mean-reverting drift (slow regimes)
        theta = 0.0
        kappa = 0.02  # reversion speed
        sigma_ou = 0.02
        z = random.gauss(0, 1)
        self.trend_ou += kappa * (theta - self.trend_ou) * (dt_minutes / 60.0) + sigma_ou * math.sqrt(
            dt_minutes / 60.0) * z

    def _update_band_targets(self, minute_of_day: int):
        # Map volatility to channel widths (smooth, within TF_BOUNDS)
        base_sigma = self._update_vol()  # "per-minute-scale" proxy
        seas = seasonality_scale(minute_of_day)
        sigma_minute = base_sigma * seas

        # Target widths proportional to sigma * sqrt(window) but clamped to bounds
        targets = {
            "m1": clamp(0.5 * sigma_minute + 0.0001, *TF_BOUNDS["m1"]),
            "m5": clamp(0.45 * sigma_minute * math.sqrt(5) + 0.005, *TF_BOUNDS["m5"]),
            "h1": clamp(0.40 * sigma_minute * math.sqrt(60) + 0.007, *TF_BOUNDS["h1"]),
            "d1": clamp(0.35 * sigma_minute * math.sqrt(1440) + 0.01, *TF_BOUNDS["d1"]),
        }
        # Smoothly approach targets
        for tf in self.band:
            self.band[tf] += 0.2 * (targets[tf] - self.band[tf])

    def _internal_tick(self, minute_of_day: int):
        """
        One sub-minute tick. Keeps price inside intersection of TF channels.
        """
        # Update OU drift slowly (once per minute would be fine too)
        # but keeping here minor impact per second
        self._update_trend_ou(dt_minutes=1 / 60)

        # Update vol + intraday seasonality
        sigma_minute = math.sqrt(self.garch_variance) * seasonality_scale(minute_of_day)
        # Approx per-second step:
        dt = 1.0 / 60.0
        z = random.gauss(0, 1)
        micro = 0.0002 * random.gauss(0, 1)  # microstructure noise
        jump = 0.0
        # Rare small jumps
        if random.random() < 0.001:
            jump = random.choice([-1, 1]) * random.uniform(0.0005, 0.003)

        # Drift with OU + boundary trend bias
        boundary_bias = self.boundary_trend * sigma_minute * 0.1
        mu = self.trend_ou * 0.001 + boundary_bias

        # Log-return step
        r = mu * dt + math.sqrt(max(1e-12, sigma_minute)) * math.sqrt(dt) * z + micro + jump

        o = self.p
        c = o * math.exp(r)

        # Bounds from intersection of channels relative to each open
        bounds = {
            "m1": (self.m1_open * (1 - self.band["m1"]), self.m1_open * (1 + self.band["m1"])),
            "m5": (self.m5_open * (1 - self.band["m5"]), self.m5_open * (1 + self.band["m5"])),
            "h1": (self.h1_open * (1 - self.band["h1"]), self.h1_open * (1 + self.band["h1"])),
            "d1": (self.d1_open * (1 - self.band["d1"]), self.d1_open * (1 + self.band["d1"])),
        }
        lower_bound = max(b[0] for b in bounds.values())
        upper_bound = min(b[1] for b in bounds.values())

        if c >= upper_bound:
            c = upper_bound
            self.boundary_trend = -1
        elif c <= lower_bound:
            c = lower_bound
            self.boundary_trend = 1
        else:
            # reset boundary bias if we crossed back past mid
            if self.boundary_trend != 0:
                mid = (upper_bound + lower_bound) / 2
                if (self.boundary_trend == -1 and c < mid) or (self.boundary_trend == 1 and c > mid):
                    self.boundary_trend = 0

        self.prev_return = math.log(c / o) if o > 0 else 0.0
        self.p = c

    def generate_one_minute_bar(self, dt_utc: datetime):
        """
        Simulate 60 internal ticks to produce 1-minute OHLC,
        then quantize to tick size.
        """
        minute_of_day = dt_utc.hour * 60 + dt_utc.minute

        # On minute boundary, reset m1_open and update channel targets
        self.m1_open = self.p
        self._update_band_targets(minute_of_day)

        # Reset other timeframe anchors on their boundaries
        if dt_utc.minute % 5 == 0:
            self.m5_open = self.p
        if dt_utc.minute == 0:
            self.h1_open = self.p
        if dt_utc.hour == 0 and dt_utc.minute == 0:
            self.d1_open = self.p

        o = self.p
        hi = o
        lo = o
        # 60 internal ticks (1 per second)
        for _ in range(60):
            self._internal_tick(minute_of_day)
            hi = max(hi, self.p)
            lo = min(lo, self.p)
        c = self.p

        # Quantize to tick size
        o_q = round_to_tick(o)
        h_q = round_to_tick(hi)
        l_q = round_to_tick(lo)
        c_q = round_to_tick(c)

        # Ensure OHLC consistency after quantization
        h_q = max(h_q, o_q, c_q)
        l_q = min(l_q, o_q, c_q)

        bar_ts = floor_minute(dt_utc)
        return {
            "t": fmt_ts(bar_ts),
            "o": float(o_q),
            "h": float(h_q),
            "l": float(l_q),
            "c": float(c_q),
        }


# ===================== IO =====================

def atomic_write_json(path: str, obj):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            raw = json.load(f)
        return {s: SyntheticStream.from_dict(d) for s, d in raw.items()}
    return {s: SyntheticStream(random.uniform(50, 200)) for s in SYMBOLS}


def load_history():
    """Loads history for all symbols from their individual files."""
    os.makedirs(DATA_DIR, exist_ok=True)
    history = {}
    for sym in SYMBOLS:
        filepath = os.path.join(DATA_DIR, f"{sym}.json")
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                history[sym] = json.load(f)
        else:
            history[sym] = []
    return history


def save_state(states):
    atomic_write_json(STATE_FILE, {s: st.to_dict() for s, st in states.items()})


def save_symbol_history(symbol: str, series: list):
    """Saves the history for a single symbol to its dedicated file."""
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, f"{symbol}.json")
    atomic_write_json(filepath, series)


def git_push(now):
    if not PUSH_TO_GIT:
        return
    try:
        # Add the state file and the entire data directory
        subprocess.run(["git", "add", STATE_FILE, DATA_DIR], check=True, capture_output=True)
        commit_msg = f"Data update {now.strftime('%Y-%m-%d %H:%M UTC')}"
        # Use --allow-empty in case state hasn't changed but history has
        subprocess.run(["git", "commit", "--allow-empty", "-m", commit_msg], check=True, capture_output=True)
        subprocess.run(["git", "push"], check=True, capture_output=True)
        print("Successfully pushed to GitHub.")
    except Exception as e:
        print(f"Error pushing to git: {e}")
        # Check if it's a "nothing to commit" error, which is benign
        if hasattr(e, 'stderr') and "nothing to commit" in e.stderr.decode():
            print("Benign error: Nothing new to commit.")
        else:
            print(f"A significant git error occurred: {e}")


# ===================== BACKFILL (30 DAYS) =====================

def backfill_30_days(states, history):
    """
    Ensure we have a full rolling window of 30 days of minute bars up to now.
    """
    now = floor_minute(datetime.now(timezone.utc))
    start = now - timedelta(minutes=ROLLING_WINDOW_MINUTES)

    # Process and save each symbol individually
    for sym, st in states.items():
        print(f"Backfilling {sym}...")
        series = history.get(sym, [])
        if not series:
            current = start
        else:
            last_ts = series[-1]["t"]
            last_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
            current = floor_minute(last_dt + timedelta(minutes=1))

        # Generate bars until 'now'
        new_bars = []
        while current <= now:
            bar = st.generate_one_minute_bar(current)
            new_bars.append(bar)
            current += timedelta(minutes=1)

        # Append new bars and truncate to rolling window
        series.extend(new_bars)
        if len(series) > ROLLING_WINDOW_MINUTES:
            series = series[-ROLLING_WINDOW_MINUTES:]

        history[sym] = series
        save_symbol_history(sym, series)
        print(f"Backfill for {sym} complete. Total bars: {len(series)}")


# ===================== MAIN LOOP =====================

def main_loop():
    print("Starting minute-bar generator (30-day rolling window)...")

    # Ensure data directory exists before any IO
    os.makedirs(DATA_DIR, exist_ok=True)

    states = load_state()
    history = load_history()

    # Initial backfill
    backfill_30_days(states, history)
    save_state(states)
    git_push(datetime.now(timezone.utc))
    print("Backfill completed. Entering live mode...")

    last_min = -1
    while True:
        now = datetime.now(timezone.utc)
        if now.minute != last_min:
            now_min = floor_minute(now)
            print(f"[{now_min.isoformat()}] Generating live bar for all symbols...")

            for sym, st in states.items():
                bar = st.generate_one_minute_bar(now_min)
                series = history.get(sym, [])
                series.append(bar)
                # rolling window
                if len(series) > ROLLING_WINDOW_MINUTES:
                    series.pop(0)

                history[sym] = series
                save_symbol_history(sym, series)  # Save each symbol's file immediately

            # Save the engine state and push all changes to git
            save_state(states)
            git_push(now_min)
            last_min = now.minute

        # Sleep efficiently to the next minute boundary
        time_to_sleep = 60.0 - (
                    datetime.now(timezone.utc).second + datetime.now(timezone.utc).microsecond / 1_000_000.0)
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)


if __name__ == "__main__":
    main_loop()
