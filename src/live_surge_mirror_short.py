from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class MirrorShortParams:
    overheat_cum_rise_pct: float = 8.0
    overheat_upper_wick_pct: float = 40.0
    overheat_volume_ratio: float = 5.0
    volume_lookback: int = 20
    stop_loss_pct: float = 1.0
    trail_start_pct: float = 3.0
    trail_rebound_pct: float = 1.2


def _volume_ratio(df: pd.DataFrame, idx: int, lookback: int) -> float:
    if idx <= 0:
        return 0.0
    start = max(0, idx - lookback)
    hist = df.iloc[start:idx]
    if hist.empty:
        return 0.0
    avg = float(hist["volume"].mean())
    if avg <= 0:
        return 0.0
    return float(df.iloc[idx]["volume"]) / avg


def overheat_confirmed(df: pd.DataFrame, signal_idx: int, params: MirrorShortParams) -> bool:
    if df is None or signal_idx < 0 or signal_idx >= len(df):
        return False

    row = df.iloc[signal_idx]

    cond_cum_rise = False
    if signal_idx >= 3:
        start_close = float(df.iloc[signal_idx - 3]["close"])
        end_close = float(row["close"])
        if start_close > 0:
            cum_rise = (end_close / start_close - 1.0) * 100.0
            cond_cum_rise = cum_rise >= params.overheat_cum_rise_pct

    high = float(row["high"])
    low = float(row["low"])
    open_ = float(row["open"])
    close = float(row["close"])
    candle_range = max(high - low, 1e-12)
    upper_wick = max(0.0, high - max(open_, close))
    upper_wick_pct = upper_wick / candle_range * 100.0
    cond_upper_wick = upper_wick_pct >= params.overheat_upper_wick_pct

    vol_ratio = _volume_ratio(df, signal_idx, params.volume_lookback)
    cond_volume = vol_ratio >= params.overheat_volume_ratio

    return bool(cond_cum_rise or cond_upper_wick or cond_volume)


def schedule_next_candle_entries(
    surge_signal: pd.Series, overheat_mask: pd.Series, delay_candles: int = 1
) -> pd.Series:
    eligible = surge_signal.astype(bool) & overheat_mask.astype(bool)
    return eligible.shift(delay_candles, fill_value=False).astype(bool)


def simulate_short_exit_ohlc(
    position: Dict, candle: Dict, params: MirrorShortParams
) -> Tuple[Optional[Dict], Dict]:
    updated = dict(position)
    entry = float(updated["entry_price"])
    stop_loss = float(updated["stop_loss"])
    trailing_active = bool(updated.get("trailing_active", False))
    lowest = float(updated.get("lowest_since_entry", entry))
    trail_stop = updated.get("trail_stop")
    trail_stop = float(trail_stop) if trail_stop is not None else None

    trail_activate_price = entry * (1.0 - params.trail_start_pct / 100.0)
    prices = [
        float(candle["open"]),
        float(candle["high"]),
        float(candle["low"]),
        float(candle["close"]),
    ]

    for price in prices:
        if price >= stop_loss:
            updated["trailing_active"] = trailing_active
            updated["lowest_since_entry"] = lowest
            updated["trail_stop"] = trail_stop
            return {"reason": "SL", "price": stop_loss}, updated

        if price <= trail_activate_price:
            trailing_active = True

        if trailing_active:
            if price < lowest:
                lowest = price
                trail_stop = lowest * (1.0 + params.trail_rebound_pct / 100.0)
            elif trail_stop is not None and price >= trail_stop:
                updated["trailing_active"] = trailing_active
                updated["lowest_since_entry"] = lowest
                updated["trail_stop"] = trail_stop
                return {"reason": "TRAIL", "price": round(trail_stop, 10)}, updated

    updated["trailing_active"] = trailing_active
    updated["lowest_since_entry"] = lowest
    updated["trail_stop"] = trail_stop
    return None, updated
