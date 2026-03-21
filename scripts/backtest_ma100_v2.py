"""
MA100 V2: TouchBounce Only - SlopeRev 시그널 제거

100일 이동평균선의 터치 반등만을 이용한 롱/숏 전략.

시그널: MA100 터치 반등/저항 (Touch Bounce)
  - 기울기 상향 + 저가가 MA100 터치 + 종가 MA100 위 → 롱 (지지 반등)
  - 기울기 하향 + 고가가 MA100 터치 + 종가 MA100 아래 → 숏 (저항 하락)

사용법:
    python scripts/backtest_ma100_v2.py
    python scripts/backtest_ma100_v2.py --initial 1000 --start 2024-06-01 --end 2026-01-31
"""

import argparse
import json as _json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── 파라미터 ─────────────────────────────────────────────────

MA100_PARAMS = {
    "ma_period": 100,
    "slope_lookback": 3,         # 기울기 계산 기간 (3일 평활화)
    "touch_buffer_pct": 1.0,     # MA100 터치 판정 범위 ±1%
    "sl_pct": 10.0,              # 손절 10% (5x → -50% 마진)
    "tp_pct": 0,                 # 고정 TP 없음 (트레일링으로 청산)
    "trail_start_pct": 5.0,      # +5%에서 트레일링 시작
    "trail_pct": 3.0,            # 3% 트레일링 (고점 대비 3% 하락 시 청산)
    "cooldown_days": 3,          # 같은 코인 재진입 대기 (일)
    "leverage": 5,
    "position_pct": 0.05,        # 자본금 대비 포지션 비율 5%
    "max_positions": 5,
    "fee_rate": 0.00055,         # 테이커 0.055%
}


# ─── 데이터 로드 ──────────────────────────────────────────────

def load_all_daily_data(
    loader: DataLoader,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """1d 데이터 로드 + 4h→1d 리샘플링으로 전체 심볼 데이터 준비."""
    symbols = loader.get_available_symbols()
    all_data: Dict[str, pd.DataFrame] = {}
    loaded_1d = 0
    resampled_4h = 0
    skipped = 0

    for i, symbol in enumerate(symbols):
        tfs = loader.get_available_timeframes(symbol)

        df = None

        # 1d 데이터 우선
        if "1d" in tfs:
            df = loader.load(symbol, "1d", start=start, end=end)
            if df is not None and len(df) >= 100:
                loaded_1d += 1
            else:
                df = None

        # 1d 없으면 4h → 1d 리샘플링
        if df is None and "4h" in tfs:
            raw = loader.load(symbol, "4h", start=start, end=end)
            if raw is not None and len(raw) >= 600:  # 최소 100일 × 6캔들
                df = _resample_4h_to_1d(raw)
                if df is not None and len(df) >= 100:
                    resampled_4h += 1
                else:
                    df = None

        if df is not None:
            all_data[symbol] = df
        else:
            skipped += 1

        if (i + 1) % 100 == 0 or (i + 1) == len(symbols):
            logger.info(f"데이터 로드: {i+1}/{len(symbols)} (1d: {loaded_1d}, 4h→1d: {resampled_4h}, skip: {skipped})")

    return all_data


def _resample_4h_to_1d(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """4h 캔들을 1d 캔들로 리샘플링."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    daily = df.resample("1D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    if len(daily) < 100:
        return None

    daily = daily.reset_index()
    return daily


# ─── 백테스트 엔진 ─────────────────────────────────────────────

class MA100Backtester:
    """MA100 기울기 전략 백테스트 엔진."""

    def __init__(
        self,
        initial_balance: float = 1000.0,
        max_positions: int = 5,
        params: dict = None,
        fixed_size: bool = False,
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_positions = max_positions
        self.params = params or MA100_PARAMS
        self.fixed_size = fixed_size  # True면 단리 (초기 잔고 기준 고정)

        self.positions: Dict[str, dict] = {}
        self.last_exit_times: Dict[str, pd.Timestamp] = {}

        self.trades: List[dict] = []
        self.equity_curve: List[dict] = []

        self.peak_equity = initial_balance
        self.max_drawdown = 0.0
        self.max_drawdown_pct = 0.0

    def _calc_qty(self, price: float) -> float:
        base = self.initial_balance if self.fixed_size else self.balance
        margin = base * self.params["position_pct"]
        position_value = margin * self.params["leverage"]
        return position_value / price

    def _update_equity(self, dt: pd.Timestamp, all_data: Dict[str, pd.DataFrame]):
        unrealized = 0.0
        for sym, pos in self.positions.items():
            df = all_data.get(sym)
            if df is None:
                continue
            mask = df["timestamp"] <= dt
            if mask.sum() == 0:
                continue
            cp = float(df.loc[mask, "close"].iloc[-1])
            entry = pos["entry_price"]
            qty = pos["size"]
            lev = self.params["leverage"]
            if pos["side"] == "long":
                pnl_pct = (cp - entry) / entry
            else:
                pnl_pct = (entry - cp) / entry
            unrealized += pnl_pct * (entry * qty) / lev

        equity = self.balance + unrealized

        if equity > self.peak_equity:
            self.peak_equity = equity
        dd = self.peak_equity - equity
        if dd > self.max_drawdown:
            self.max_drawdown = dd
            self.max_drawdown_pct = dd / self.peak_equity * 100

        self.equity_curve.append({"timestamp": dt, "equity": equity})

    def _close_position(self, symbol: str, exit_price: float, reason: str, exit_time: pd.Timestamp):
        pos = self.positions.pop(symbol)
        entry = pos["entry_price"]
        qty = pos["size"]
        lev = self.params["leverage"]

        if pos["side"] == "long":
            pnl_pct = (exit_price - entry) / entry * 100 * lev
        else:
            pnl_pct = (entry - exit_price) / entry * 100 * lev

        pnl_usd = pnl_pct / 100 * (entry * qty) / lev

        fee_rate = self.params.get("fee_rate", 0)
        fee_usd = 0.0
        if fee_rate > 0:
            entry_fee = qty * entry * fee_rate
            exit_fee = qty * exit_price * fee_rate
            fee_usd = entry_fee + exit_fee
            pnl_usd -= fee_usd

        self.balance += pnl_usd
        self.last_exit_times[symbol] = exit_time

        self.trades.append({
            "symbol": symbol,
            "side": pos["side"],
            "entry_price": entry,
            "exit_price": exit_price,
            "entry_time": pos["entry_time"],
            "exit_time": exit_time,
            "pnl_pct": pnl_pct,
            "pnl_usd": pnl_usd,
            "fee_usd": fee_usd,
            "reason": reason,
            "signal_type": pos.get("signal_type", ""),
        })

    def _check_exit_candle(self, symbol: str, candle: pd.Series, dt: pd.Timestamp):
        """
        캔들 하나로 SL/TP/트레일링/시그널 반전 체크.

        캔들 내부 가격 경로 추정:
          양봉 (close >= open): O → L → H → C
          음봉 (close <  open): O → H → L → C
        """
        pos = self.positions.get(symbol)
        if pos is None:
            return

        entry = pos["entry_price"]
        sl = pos["stop_loss"]
        tp = pos["take_profit"]  # 0이면 고정 TP 비활성
        side = pos["side"]
        high = float(candle["high"])
        low = float(candle["low"])
        close = float(candle["close"])
        open_p = float(candle["open"])

        is_green = close >= open_p
        use_fixed_tp = tp > 0

        # ── SL/TP 먼저 체크 (캔들 내부 가격 경로 기준) ──
        # ── LONG ──
        if side == "long":
            if is_green:
                # 양봉: O → L → H → C
                if low <= sl:
                    self._close_position(symbol, sl, "SL", dt)
                    return
                if use_fixed_tp and high >= tp:
                    self._close_position(symbol, tp, "TP", dt)
                    return
            else:
                # 음봉: O → H → L → C
                if use_fixed_tp and high >= tp:
                    self._close_position(symbol, tp, "TP", dt)
                    return
                if low <= sl:
                    self._close_position(symbol, sl, "SL", dt)
                    return

        # ── SHORT ──
        else:
            if not is_green:
                # 음봉: O → H → L → C
                if high >= sl:
                    self._close_position(symbol, sl, "SL", dt)
                    return
                if use_fixed_tp and low <= tp:
                    self._close_position(symbol, tp, "TP", dt)
                    return
            else:
                # 양봉: O → L → H → C
                if use_fixed_tp and low <= tp:
                    self._close_position(symbol, tp, "TP", dt)
                    return
                if high >= sl:
                    self._close_position(symbol, sl, "SL", dt)
                    return

        # ── SL/TP 안 걸렸으면 시그널 반전 청산 ──
        if side == "long" and candle.get("short_signal", False):
            self._close_position(symbol, close, "Reversal", dt)
            return
        if side == "short" and candle.get("long_signal", False):
            self._close_position(symbol, close, "Reversal", dt)
            return

        # ── Trailing Stop (close 기준) ──
        if side == "long":
            cur_pnl_pct = (close - entry) / entry * 100
        else:
            cur_pnl_pct = (entry - close) / entry * 100

        trail_start = self.params["trail_start_pct"]
        trail_pct = self.params["trail_pct"]

        if cur_pnl_pct >= trail_start:
            pos["trailing"] = True

            if side == "long":
                if high > pos["highest"]:
                    pos["highest"] = high
                    pos["trail_stop"] = high * (1 - trail_pct / 100)
                if close <= pos["trail_stop"]:
                    self._close_position(symbol, pos["trail_stop"], "Trail", dt)
                    return
            else:
                if low < pos["lowest"]:
                    pos["lowest"] = low
                    pos["trail_stop"] = low * (1 + trail_pct / 100)
                if close >= pos["trail_stop"]:
                    self._close_position(symbol, pos["trail_stop"], "Trail", dt)
                    return
        elif pos.get("trailing"):
            if side == "long" and close <= pos["trail_stop"]:
                self._close_position(symbol, pos["trail_stop"], "Trail", dt)
                return
            if side == "short" and close >= pos["trail_stop"]:
                self._close_position(symbol, pos["trail_stop"], "Trail", dt)
                return

    def _precompute_signals(self, all_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """모든 심볼의 MA100 시그널을 벡터화로 사전 계산."""
        params = self.params
        ma_period = params["ma_period"]
        slope_lookback = params["slope_lookback"]
        touch_buf = params["touch_buffer_pct"] / 100

        result = {}
        total = len(all_data)

        for idx, (symbol, df) in enumerate(all_data.items()):
            df = df.copy()

            # MA100 계산
            df["ma100"] = df["close"].rolling(ma_period).mean()

            # 기울기 계산 (% 변화율)
            df["slope"] = (
                (df["ma100"] - df["ma100"].shift(slope_lookback))
                / df["ma100"].shift(slope_lookback)
                * 100
            )
            df["slope_prev"] = df["slope"].shift(1)

            # Signal: MA100 터치 반등/저항 (TouchBounce only)
            df["touch_long"] = (
                (df["slope"] > 0)
                & (df["low"] <= df["ma100"] * (1 + touch_buf))
                & (df["close"] > df["ma100"])
            )
            df["touch_short"] = (
                (df["slope"] < 0)
                & (df["high"] >= df["ma100"] * (1 - touch_buf))
                & (df["close"] < df["ma100"])
            )

            # 시그널 (TouchBounce only)
            df["long_signal"] = df["touch_long"]
            df["short_signal"] = df["touch_short"]

            # 시그널 타입 (리포트용)
            df["signal_type"] = ""
            df.loc[df["touch_long"], "signal_type"] = "TouchBounce"
            df.loc[df["touch_short"], "signal_type"] = "TouchBounce"

            # NaN 처리
            for col in ["long_signal", "short_signal", "touch_long", "touch_short"]:
                df[col] = df[col].fillna(False)

            result[symbol] = df

            if (idx + 1) % 100 == 0:
                logger.info(f"  신호 사전 계산: {idx+1}/{total}")

        return result

    def run(self, all_data: Dict[str, pd.DataFrame], start_dt: datetime, end_dt: datetime):
        """전체 백테스트 실행."""
        # 1) 시그널 사전 계산
        logger.info("MA100 시그널 사전 계산 중...")
        precomputed = self._precompute_signals(all_data)
        self._precomputed = precomputed  # HTML 리포트용 캐시

        # 2) 타임스탬프→인덱스 매핑
        sym_ts_idx: Dict[str, Dict] = {}
        for symbol, df in precomputed.items():
            sym_ts_idx[symbol] = dict(zip(df["timestamp"], df.index))

        # 3) 백테스트 구간 타임스탬프 수집
        all_ts = set()
        for df in precomputed.values():
            mask = (df["timestamp"] >= pd.Timestamp(start_dt)) & (df["timestamp"] <= pd.Timestamp(end_dt))
            all_ts.update(df.loc[mask, "timestamp"].tolist())
        all_ts = sorted(all_ts)

        logger.info(f"백테스트 구간: {len(all_ts)} 일봉 ({start_dt} ~ {end_dt})")

        # 4) 시그널 발생 시점 미리 추출
        signals_at: Dict[pd.Timestamp, List[dict]] = {}
        for symbol, df in precomputed.items():
            long_mask = df["long_signal"] & (df["timestamp"] >= pd.Timestamp(start_dt)) & (df["timestamp"] <= pd.Timestamp(end_dt))
            short_mask = df["short_signal"] & (df["timestamp"] >= pd.Timestamp(start_dt)) & (df["timestamp"] <= pd.Timestamp(end_dt))

            for _, row in df.loc[long_mask].iterrows():
                ts = row["timestamp"]
                if ts not in signals_at:
                    signals_at[ts] = []
                signals_at[ts].append({
                    "symbol": symbol,
                    "side": "long",
                    "price": float(row["close"]),
                    "slope": float(row["slope"]) if pd.notna(row["slope"]) else 0,
                    "signal_type": row["signal_type"],
                })

            for _, row in df.loc[short_mask].iterrows():
                ts = row["timestamp"]
                if ts not in signals_at:
                    signals_at[ts] = []
                signals_at[ts].append({
                    "symbol": symbol,
                    "side": "short",
                    "price": float(row["close"]),
                    "slope": float(row["slope"]) if pd.notna(row["slope"]) else 0,
                    "signal_type": row["signal_type"],
                })

        total_signals = sum(len(v) for v in signals_at.values())
        logger.info(f"사전 감지된 시그널: {total_signals}건 (across {len(signals_at)} timestamps)")

        cooldown_td = timedelta(days=self.params["cooldown_days"])

        # 5) 캔들 순회
        for i, ts in enumerate(all_ts):
            # 기존 포지션 청산 체크
            for symbol in list(self.positions.keys()):
                df = precomputed.get(symbol)
                if df is None:
                    continue
                row_idx = sym_ts_idx[symbol].get(ts)
                if row_idx is None:
                    continue
                candle = df.loc[row_idx]
                self._check_exit_candle(symbol, candle, ts)

            # 신규 진입
            if len(self.positions) >= self.max_positions:
                if i % 7 == 0:
                    self._update_equity(ts, precomputed)
                continue

            candidates = signals_at.get(ts)
            if not candidates:
                if i % 7 == 0:
                    self._update_equity(ts, precomputed)
                continue

            # 기울기 절대값이 큰 순으로 정렬 (강한 시그널 우선)
            candidates_sorted = sorted(candidates, key=lambda s: abs(s["slope"]), reverse=True)

            for sig in candidates_sorted:
                symbol = sig["symbol"]
                side = sig["side"]

                if symbol in self.positions:
                    continue
                if len(self.positions) >= self.max_positions:
                    break
                if self.balance <= 0:
                    break

                # 쿨다운
                last_exit = self.last_exit_times.get(symbol)
                if last_exit and (ts - last_exit) < cooldown_td:
                    continue

                entry_price = sig["price"]
                tp_pct = self.params["tp_pct"]

                if side == "long":
                    sl_price = entry_price * (1 - self.params["sl_pct"] / 100)
                    tp_price = entry_price * (1 + tp_pct / 100) if tp_pct > 0 else 0
                else:
                    sl_price = entry_price * (1 + self.params["sl_pct"] / 100)
                    tp_price = entry_price * (1 - tp_pct / 100) if tp_pct > 0 else 0

                qty = self._calc_qty(entry_price)
                if qty <= 0:
                    break

                self.positions[symbol] = {
                    "symbol": symbol,
                    "side": side,
                    "entry_price": entry_price,
                    "entry_time": ts,
                    "stop_loss": sl_price,
                    "take_profit": tp_price,
                    "highest": entry_price,
                    "lowest": entry_price,
                    "trail_stop": sl_price if side == "long" else sl_price,
                    "trailing": False,
                    "size": qty,
                    "signal_type": sig["signal_type"],
                }

            # equity 기록 (주 1회)
            if i % 7 == 0:
                self._update_equity(ts, precomputed)

            if (i + 1) % 100 == 0:
                logger.info(f"  백테스트 진행: {i+1}/{len(all_ts)} ({(i+1)/len(all_ts)*100:.0f}%)")

        # 잔여 포지션 청산
        last_ts = all_ts[-1] if all_ts else pd.Timestamp(end_dt)
        for symbol in list(self.positions.keys()):
            df = precomputed.get(symbol)
            if df is None:
                continue
            last_close = float(df.iloc[-1]["close"])
            self._close_position(symbol, last_close, "BacktestEnd", last_ts)

        self._update_equity(last_ts, precomputed)


# ─── 리포트 ────────────────────────────────────────────────────

def print_report(bt: MA100Backtester, start_dt: datetime, end_dt: datetime):
    """결과 리포트 출력."""
    days = (end_dt - start_dt).days
    trades = bt.trades
    n = len(trades)

    print()
    print("=" * 55)
    print("  MA100 V2: TouchBounce Only Backtest")
    print("=" * 55)
    print(f"  Period : {start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')} ({days} days)")
    print(f"  Initial: ${bt.initial_balance:,.2f}")
    print()

    if n == 0:
        print("  No trades executed.")
        print("=" * 55)
        return

    wins = [t for t in trades if t["pnl_usd"] > 0]
    losses = [t for t in trades if t["pnl_usd"] <= 0]
    total_pnl = sum(t["pnl_usd"] for t in trades)
    total_pnl_pct = total_pnl / bt.initial_balance * 100
    win_rate = len(wins) / n * 100

    gross_profit = sum(t["pnl_usd"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl_usd"] for t in losses)) if losses else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    total_fees = sum(t["fee_usd"] for t in trades)

    print("--- Results ---")
    print(f"  Total Trades : {n}")
    print(f"  Win Rate     : {win_rate:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"  Total PnL    : ${total_pnl:+,.2f} ({total_pnl_pct:+.2f}%)")
    print(f"  Final Balance: ${bt.balance:,.2f}")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Max Drawdown : -${bt.max_drawdown:,.2f} (-{bt.max_drawdown_pct:.1f}%)")
    print(f"  Total Fees   : ${total_fees:,.2f}")
    print()

    # Side breakdown
    longs = [t for t in trades if t["side"] == "long"]
    shorts = [t for t in trades if t["side"] == "short"]
    long_wins = [t for t in longs if t["pnl_usd"] > 0]
    short_wins = [t for t in shorts if t["pnl_usd"] > 0]
    print("--- Side Breakdown ---")
    if longs:
        print(f"  LONG : {len(longs)} trades, {len(long_wins)}W ({len(long_wins)/len(longs)*100:.0f}%), PnL=${sum(t['pnl_usd'] for t in longs):+,.2f}")
    if shorts:
        print(f"  SHORT: {len(shorts)} trades, {len(short_wins)}W ({len(short_wins)/len(shorts)*100:.0f}%), PnL=${sum(t['pnl_usd'] for t in shorts):+,.2f}")
    print()

    # Signal type breakdown
    sig_types = {}
    for t in trades:
        st = t.get("signal_type", "Unknown")
        if st not in sig_types:
            sig_types[st] = {"count": 0, "wins": 0, "pnl": 0.0}
        sig_types[st]["count"] += 1
        if t["pnl_usd"] > 0:
            sig_types[st]["wins"] += 1
        sig_types[st]["pnl"] += t["pnl_usd"]
    print("--- Signal Type Breakdown ---")
    for st, info in sorted(sig_types.items(), key=lambda x: -x[1]["count"]):
        wr = info["wins"] / info["count"] * 100 if info["count"] > 0 else 0
        print(f"  {st:15s}: {info['count']} trades, {info['wins']}W ({wr:.0f}%), PnL=${info['pnl']:+,.2f}")
    print()

    # Exit reasons
    reasons = {}
    for t in trades:
        r = t["reason"]
        reasons[r] = reasons.get(r, 0) + 1
    print("--- Exit Reasons ---")
    for r, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {r:15s}: {cnt}")
    print()

    # Top trades
    sorted_trades = sorted(trades, key=lambda t: t["pnl_usd"], reverse=True)
    print("--- Top Wins ---")
    for t in sorted_trades[:5]:
        sym = t["symbol"].split("/")[0]
        print(f"  {sym:12s} {t['side']:5s} {t['pnl_pct']:+7.1f}%  ${t['pnl_usd']:+8.2f}  ({t['reason']}, {t.get('signal_type','')})")

    print()
    print("--- Top Losses ---")
    for t in sorted_trades[-5:]:
        sym = t["symbol"].split("/")[0]
        print(f"  {sym:12s} {t['side']:5s} {t['pnl_pct']:+7.1f}%  ${t['pnl_usd']:+8.2f}  ({t['reason']}, {t.get('signal_type','')})")

    print()

    # Monthly breakdown
    if trades:
        trade_df = pd.DataFrame(trades)
        trade_df["exit_time"] = pd.to_datetime(trade_df["exit_time"])
        trade_df["month"] = trade_df["exit_time"].dt.to_period("M")

        print("--- Monthly Breakdown ---")
        for m in sorted(trade_df["month"].unique()):
            mt = trade_df[trade_df["month"] == m]
            m_pnl = mt["pnl_usd"].sum()
            m_n = len(mt)
            m_wins = int((mt["pnl_usd"] > 0).sum())
            m_losses = m_n - m_wins
            m_wr = m_wins / m_n * 100 if m_n > 0 else 0
            print(f"  {str(m):8s}: {m_n:3d} trades  {m_wins}W/{m_losses}L  WR={m_wr:.0f}%  PnL=${m_pnl:+,.2f}")

    print()
    print("=" * 55)


# ─── HTML 리포트 ───────────────────────────────────────────────

def generate_html_report(
    bt: MA100Backtester,
    start_dt: datetime,
    end_dt: datetime,
    all_data: Dict[str, pd.DataFrame] = None,
) -> str:
    """HTML 리포트 생성."""
    trades = bt.trades
    n = len(trades)
    days = (end_dt - start_dt).days

    wins = [t for t in trades if t["pnl_usd"] > 0]
    losses = [t for t in trades if t["pnl_usd"] <= 0]
    total_pnl = sum(t["pnl_usd"] for t in trades) if trades else 0
    total_pnl_pct = total_pnl / bt.initial_balance * 100
    win_rate = len(wins) / n * 100 if n > 0 else 0
    gross_profit = sum(t["pnl_usd"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl_usd"] for t in losses)) if losses else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    avg_win = (gross_profit / len(wins)) if wins else 0
    avg_loss = (gross_loss / len(losses)) if losses else 0

    longs = [t for t in trades if t["side"] == "long"]
    shorts = [t for t in trades if t["side"] == "short"]
    long_wins = len([t for t in longs if t["pnl_usd"] > 0])
    short_wins = len([t for t in shorts if t["pnl_usd"] > 0])
    long_pnl = sum(t["pnl_usd"] for t in longs)
    short_pnl = sum(t["pnl_usd"] for t in shorts)

    # Exit reasons
    reasons = {}
    for t in trades:
        r = t["reason"]
        reasons[r] = reasons.get(r, 0) + 1

    # Equity curve data
    eq_labels = [e["timestamp"].strftime("%Y-%m-%d") for e in bt.equity_curve]
    eq_values = [round(e["equity"], 2) for e in bt.equity_curve]

    # Cumulative PnL
    cum_pnl = []
    running = 0
    for t in trades:
        running += t["pnl_usd"]
        cum_pnl.append(round(running, 2))
    trade_labels = [f"#{i+1}" for i in range(n)]

    # Per-trade PnL
    trade_pnls = [round(t["pnl_usd"], 2) for t in trades]
    trade_colors = ["rgba(34,197,94,0.8)" if p > 0 else "rgba(239,68,68,0.8)" for p in trade_pnls]
    trade_syms = [t["symbol"].split("/")[0][:6] for t in trades]

    # Monthly PnL
    monthly_pnl_map = {}
    for t in trades:
        et = t["exit_time"]
        m = et.strftime("%Y-%m") if hasattr(et, "strftime") else str(et)[:7]
        monthly_pnl_map[m] = monthly_pnl_map.get(m, 0) + t["pnl_usd"]
    monthly_labels = list(monthly_pnl_map.keys())
    monthly_values = [round(v, 2) for v in monthly_pnl_map.values()]
    monthly_colors = ["rgba(34,197,94,0.8)" if v > 0 else "rgba(239,68,68,0.8)" for v in monthly_values]

    # Monthly breakdown table
    monthly_rows = ""
    if trades:
        trade_df = pd.DataFrame(trades)
        trade_df["exit_time"] = pd.to_datetime(trade_df["exit_time"])
        trade_df["month"] = trade_df["exit_time"].dt.to_period("M")
        for m in sorted(trade_df["month"].unique()):
            mt = trade_df[trade_df["month"] == m]
            m_pnl = mt["pnl_usd"].sum()
            m_n = len(mt)
            m_wins = int((mt["pnl_usd"] > 0).sum())
            m_losses = m_n - m_wins
            m_wr = m_wins / m_n * 100 if m_n > 0 else 0
            pnl_cls = "positive" if m_pnl >= 0 else "negative"
            monthly_rows += f"""
            <tr>
                <td>{str(m)}</td>
                <td>{m_n}</td>
                <td>{m_wins}W / {m_losses}L</td>
                <td>{m_wr:.0f}%</td>
                <td class="{pnl_cls}">${m_pnl:+,.2f}</td>
            </tr>"""

    # Trade chart data
    trade_chart_data = {}
    if all_data:
        for i, t in enumerate(trades):
            symbol = t["symbol"]
            df = all_data.get(symbol)
            if df is None:
                continue
            entry_time = pd.Timestamp(t["entry_time"])
            exit_time = pd.Timestamp(t["exit_time"])
            margin_before = timedelta(days=30)
            margin_after = timedelta(days=15)
            mask = (df["timestamp"] >= entry_time - margin_before) & (df["timestamp"] <= exit_time + margin_after)
            chart_df = df.loc[mask]
            if len(chart_df) < 5:
                continue
            candles = []
            for _, row in chart_df.iterrows():
                candles.append({
                    "time": int(row["timestamp"].timestamp()),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                })
            # MA100 라인 데이터
            ma_data = []
            if "ma100" in chart_df.columns:
                for _, row in chart_df.iterrows():
                    if pd.notna(row["ma100"]):
                        ma_data.append({
                            "time": int(row["timestamp"].timestamp()),
                            "value": float(row["ma100"]),
                        })

            price_str = f"{t['entry_price']:.15g}"
            precision = len(price_str.split('.')[1]) if '.' in price_str else 2
            precision = max(precision, 2)

            trade_chart_data[i] = {
                "symbol": symbol.split("/")[0],
                "candles": candles,
                "ma100": ma_data,
                "entry_time": int(entry_time.timestamp()),
                "exit_time": int(exit_time.timestamp()),
                "entry_price": t["entry_price"],
                "exit_price": t["exit_price"],
                "sl": t["entry_price"] * (1 - bt.params["sl_pct"] / 100) if t["side"] == "long" else t["entry_price"] * (1 + bt.params["sl_pct"] / 100),
                "tp": (t["entry_price"] * (1 + bt.params["tp_pct"] / 100) if t["side"] == "long" else t["entry_price"] * (1 - bt.params["tp_pct"] / 100)) if bt.params["tp_pct"] > 0 else 0,
                "pnl_pct": t["pnl_pct"],
                "pnl_usd": t["pnl_usd"],
                "reason": t["reason"],
                "side": t["side"],
                "signal_type": t.get("signal_type", ""),
                "precision": precision,
            }
    trade_chart_json = _json.dumps(trade_chart_data)

    # Trade table rows
    trade_rows = ""
    for i, t in enumerate(trades):
        sym = t["symbol"].split("/")[0]
        pnl_cls = "positive" if t["pnl_usd"] > 0 else "negative"
        entry_t = t["entry_time"].strftime("%Y-%m-%d") if hasattr(t["entry_time"], "strftime") else str(t["entry_time"])
        exit_t = t["exit_time"].strftime("%Y-%m-%d") if hasattr(t["exit_time"], "strftime") else str(t["exit_time"])
        if hasattr(t["entry_time"], "strftime"):
            dur = t["exit_time"] - t["entry_time"]
            dur_d = dur.days
            dur_str = f"{dur_d}d"
        else:
            dur_str = "-"
        side_cls = "positive" if t["side"] == "long" else "negative"
        sig_type = t.get("signal_type", "")
        trade_rows += f"""
        <tr>
            <td>{i+1}</td>
            <td><a href="#" class="symbol-link" onclick="openTradeChart({i}); return false;"><strong>{sym}</strong></a></td>
            <td class="{side_cls}">{t['side'].upper()}</td>
            <td>{entry_t}</td>
            <td>{exit_t}</td>
            <td>{dur_str}</td>
            <td>${t['entry_price']:.6g}</td>
            <td>${t['exit_price']:.6g}</td>
            <td class="{pnl_cls}">{t['pnl_pct']:+.1f}%</td>
            <td class="{pnl_cls}">${t['pnl_usd']:+.2f}</td>
            <td><span class="badge badge-{t['reason'].lower()}">{t['reason']}</span></td>
            <td>{sig_type}</td>
        </tr>"""

    pnl_cls = "positive" if total_pnl >= 0 else "negative"
    pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"
    rr_str = f"{avg_win/avg_loss:.1f}x" if avg_loss > 0 else "N/A"

    num_symbols = len(all_data) if all_data else 0

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MA100 V2: TouchBounce Only Backtest Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root {{
    --bg: #0f1117;
    --card: #1a1d28;
    --border: #2a2d3a;
    --text: #e1e4ea;
    --muted: #8b8fa3;
    --green: #22c55e;
    --red: #ef4444;
    --blue: #3b82f6;
    --yellow: #eab308;
    --purple: #a855f7;
    --orange: #f97316;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 24px;
    line-height: 1.6;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 4px; }}
  .subtitle {{ color: var(--muted); margin-bottom: 24px; font-size: 14px; }}
  .grid {{ display: grid; gap: 16px; margin-bottom: 24px; }}
  .grid-4 {{ grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }}
  .grid-2 {{ grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); }}
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
  }}
  .card-title {{
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--muted);
    margin-bottom: 8px;
  }}
  .card-value {{ font-size: 28px; font-weight: 700; }}
  .card-sub {{ font-size: 13px; color: var(--muted); margin-top: 4px; }}
  .positive {{ color: var(--green); }}
  .negative {{ color: var(--red); }}
  .chart-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 24px;
  }}
  .chart-card h3 {{ font-size: 16px; margin-bottom: 16px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{
    text-align: left; padding: 10px 12px;
    border-bottom: 2px solid var(--border);
    color: var(--muted); font-weight: 600;
    font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); white-space: nowrap; }}
  tr:hover {{ background: rgba(255,255,255,0.02); }}
  .badge {{
    display: inline-block; padding: 2px 8px;
    border-radius: 6px; font-size: 11px; font-weight: 600;
  }}
  .badge-sl {{ background: rgba(239,68,68,0.15); color: var(--red); }}
  .badge-tp {{ background: rgba(34,197,94,0.15); color: var(--green); }}
  .badge-trail {{ background: rgba(168,85,247,0.15); color: var(--purple); }}
  .badge-reversal {{ background: rgba(59,130,246,0.15); color: var(--blue); }}
  .badge-backtestend {{ background: rgba(139,143,163,0.15); color: var(--muted); }}
  .section-title {{
    font-size: 18px; font-weight: 600;
    margin: 32px 0 16px; padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }}
  .param-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 8px; font-size: 13px;
  }}
  .param-item {{
    display: flex; justify-content: space-between;
    padding: 6px 10px;
    background: rgba(255,255,255,0.03);
    border-radius: 6px;
  }}
  .param-label {{ color: var(--muted); }}
  .param-value {{ font-weight: 600; }}
  canvas {{ max-height: 320px; }}
  .symbol-link {{
    color: var(--blue); text-decoration: none; cursor: pointer;
    transition: color 0.15s;
  }}
  .symbol-link:hover {{ color: #60a5fa; text-decoration: underline; }}
  /* Modal */
  .modal-overlay {{
    display: none; position: fixed; inset: 0;
    background: rgba(0,0,0,0.75); z-index: 1000;
    justify-content: center; align-items: center; padding: 24px;
  }}
  .modal-overlay.active {{ display: flex; }}
  .modal {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 16px; width: 100%; max-width: 1200px;
    max-height: 92vh; overflow: hidden;
    display: flex; flex-direction: column;
  }}
  .modal-header {{
    display: flex; justify-content: space-between;
    align-items: center; padding: 16px 20px;
    border-bottom: 1px solid var(--border);
  }}
  .modal-header h3 {{ font-size: 18px; font-weight: 600; }}
  .modal-close {{
    background: none; border: none; color: var(--muted);
    font-size: 24px; cursor: pointer; padding: 4px 8px;
    border-radius: 6px; transition: background 0.15s;
  }}
  .modal-close:hover {{ background: rgba(255,255,255,0.1); color: var(--text); }}
  .modal-body {{ padding: 20px; overflow-y: auto; }}
  .trade-info-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px; margin-bottom: 16px;
  }}
  .trade-info-item {{
    background: rgba(255,255,255,0.03);
    padding: 10px 12px; border-radius: 8px;
  }}
  .trade-info-label {{
    font-size: 11px; text-transform: uppercase;
    color: var(--muted); letter-spacing: 0.5px; margin-bottom: 4px;
  }}
  .trade-info-value {{ font-size: 16px; font-weight: 600; }}
  #tradeChartContainer {{ width: 100%; height: 520px; border-radius: 8px; overflow: hidden; }}
  .chart-legend {{
    display: flex; gap: 16px; margin-top: 12px;
    font-size: 12px; color: var(--muted);
  }}
  .chart-legend span {{ display: flex; align-items: center; gap: 4px; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
  .legend-line {{ width: 16px; height: 2px; display: inline-block; }}
</style>
</head>
<body>
<div class="container">

<h1>MA100 V2: TouchBounce Only Backtest</h1>
<p class="subtitle">{start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')} ({days} days) &nbsp;|&nbsp; {num_symbols} symbols &nbsp;|&nbsp; 1d candles</p>

<!-- Summary Cards -->
<div class="grid grid-4">
  <div class="card">
    <div class="card-title">Total PnL</div>
    <div class="card-value {pnl_cls}">${total_pnl:+,.2f}</div>
    <div class="card-sub">{total_pnl_pct:+.2f}% return</div>
  </div>
  <div class="card">
    <div class="card-title">Win Rate</div>
    <div class="card-value">{win_rate:.1f}%</div>
    <div class="card-sub">{len(wins)}W / {len(losses)}L of {n} trades</div>
  </div>
  <div class="card">
    <div class="card-title">Profit Factor</div>
    <div class="card-value">{pf_str}</div>
    <div class="card-sub">Gross +${gross_profit:,.2f} / -${gross_loss:,.2f}</div>
  </div>
  <div class="card">
    <div class="card-title">Max Drawdown</div>
    <div class="card-value negative">-${bt.max_drawdown:,.2f}</div>
    <div class="card-sub">-{bt.max_drawdown_pct:.1f}% from peak</div>
  </div>
</div>

<div class="grid grid-4">
  <div class="card">
    <div class="card-title">Final Balance</div>
    <div class="card-value">${bt.balance:,.2f}</div>
    <div class="card-sub">Initial: ${bt.initial_balance:,.2f}</div>
  </div>
  <div class="card">
    <div class="card-title">Avg Win / Loss</div>
    <div class="card-value">{rr_str}</div>
    <div class="card-sub">+${avg_win:,.2f} / -${avg_loss:,.2f}</div>
  </div>
  <div class="card">
    <div class="card-title">Long</div>
    <div class="card-value {"positive" if long_pnl >= 0 else "negative"}">${long_pnl:+,.2f}</div>
    <div class="card-sub">{len(longs)} trades, {long_wins}W ({f"{long_wins/len(longs)*100:.0f}" if longs else "0"}% WR)</div>
  </div>
  <div class="card">
    <div class="card-title">Short</div>
    <div class="card-value {"positive" if short_pnl >= 0 else "negative"}">${short_pnl:+,.2f}</div>
    <div class="card-sub">{len(shorts)} trades, {short_wins}W ({f"{short_wins/len(shorts)*100:.0f}" if shorts else "0"}% WR)</div>
  </div>
</div>

<!-- Equity Curve -->
<div class="chart-card">
  <h3>Equity Curve</h3>
  <canvas id="equityChart"></canvas>
</div>

<!-- PnL Charts -->
<div class="grid grid-2">
  <div class="chart-card">
    <h3>Per-Trade PnL ($)</h3>
    <canvas id="tradePnlChart"></canvas>
  </div>
  <div class="chart-card">
    <h3>Cumulative PnL ($)</h3>
    <canvas id="cumPnlChart"></canvas>
  </div>
</div>

<div class="grid grid-2">
  <div class="chart-card">
    <h3>Monthly PnL ($)</h3>
    <canvas id="monthlyPnlChart"></canvas>
  </div>
  <div class="chart-card">
    <h3>Exit Reasons</h3>
    <canvas id="reasonChart"></canvas>
  </div>
</div>

<!-- Monthly Breakdown -->
<h2 class="section-title">Monthly Breakdown</h2>
<div class="card" style="overflow-x:auto;">
  <table>
    <thead>
      <tr><th>Month</th><th>Trades</th><th>W/L</th><th>Win Rate</th><th>PnL</th></tr>
    </thead>
    <tbody>{monthly_rows}</tbody>
  </table>
</div>

<!-- Trade Log -->
<h2 class="section-title">Trade Log ({n} trades)</h2>
<div class="card" style="overflow-x:auto;">
  <table>
    <thead>
      <tr><th>#</th><th>Symbol</th><th>Side</th><th>Entry</th><th>Exit</th><th>Duration</th><th>Entry $</th><th>Exit $</th><th>PnL %</th><th>PnL $</th><th>Reason</th><th>Signal</th></tr>
    </thead>
    <tbody>{trade_rows}</tbody>
  </table>
</div>

<!-- Strategy Params -->
<h2 class="section-title">Strategy Parameters</h2>
<div class="card">
  <div class="param-grid">
    <div class="param-item"><span class="param-label">MA Period</span><span class="param-value">{bt.params['ma_period']}</span></div>
    <div class="param-item"><span class="param-label">Slope Lookback</span><span class="param-value">{bt.params['slope_lookback']} days</span></div>
    <div class="param-item"><span class="param-label">Touch Buffer</span><span class="param-value">&plusmn;{bt.params['touch_buffer_pct']}%</span></div>
    <div class="param-item"><span class="param-label">Stop Loss</span><span class="param-value">{bt.params['sl_pct']}%</span></div>
    <div class="param-item"><span class="param-label">Take Profit</span><span class="param-value">{bt.params['tp_pct']}%</span></div>
    <div class="param-item"><span class="param-label">Trail Start</span><span class="param-value">{bt.params['trail_start_pct']}%</span></div>
    <div class="param-item"><span class="param-label">Trail Stop</span><span class="param-value">{bt.params['trail_pct']}%</span></div>
    <div class="param-item"><span class="param-label">Cooldown</span><span class="param-value">{bt.params['cooldown_days']} days</span></div>
    <div class="param-item"><span class="param-label">Leverage</span><span class="param-value">{bt.params['leverage']}x</span></div>
    <div class="param-item"><span class="param-label">Position Size</span><span class="param-value">{bt.params['position_pct']*100:.0f}% of balance</span></div>
    <div class="param-item"><span class="param-label">Max Positions</span><span class="param-value">{bt.max_positions}</span></div>
    <div class="param-item"><span class="param-label">Fee Rate</span><span class="param-value">{bt.params['fee_rate']*100:.3f}%</span></div>
  </div>
</div>

<p style="text-align:center; color:var(--muted); font-size:12px; margin-top:32px;">
  Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}
</p>

</div>

<!-- Trade Chart Modal -->
<div class="modal-overlay" id="tradeModal">
  <div class="modal">
    <div class="modal-header">
      <h3 id="modalTitle">-</h3>
      <button class="modal-close" onclick="closeTradeChart()">&times;</button>
    </div>
    <div class="modal-body">
      <div class="trade-info-grid" id="modalInfo"></div>
      <div id="tradeChartContainer"></div>
      <div class="chart-legend">
        <span><span class="legend-line" style="background:#f97316;"></span> MA100</span>
        <span><span class="legend-line" style="background:#3b82f6;"></span> Entry</span>
        <span><span class="legend-line" style="background:#ef4444;"></span> Stop Loss</span>
        <span><span class="legend-line" style="background:#a3e635;"></span> Take Profit</span>
        <span><span class="legend-dot" style="background:#3b82f6;"></span> Buy</span>
        <span><span class="legend-dot" style="background:#ef4444;"></span> Sell</span>
      </div>
    </div>
  </div>
</div>

<script>
Chart.defaults.color = '#8b8fa3';
Chart.defaults.borderColor = '#2a2d3a';

// Equity Curve
new Chart(document.getElementById('equityChart'), {{
  type: 'line',
  data: {{
    labels: {_json.dumps(eq_labels)},
    datasets: [{{
      label: 'Equity ($)',
      data: {_json.dumps(eq_values)},
      borderColor: '#3b82f6',
      backgroundColor: 'rgba(59,130,246,0.1)',
      fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ display: true, ticks: {{ maxTicksLimit: 12 }} }},
      y: {{ display: true }}
    }}
  }}
}});

// Per-Trade PnL
new Chart(document.getElementById('tradePnlChart'), {{
  type: 'bar',
  data: {{
    labels: {_json.dumps(trade_syms)},
    datasets: [{{
      label: 'PnL ($)',
      data: {_json.dumps(trade_pnls)},
      backgroundColor: {_json.dumps(trade_colors)},
      borderRadius: 4,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{ x: {{ ticks: {{ font: {{ size: 9 }}, maxRotation: 90 }} }} }}
  }}
}});

// Cumulative PnL
new Chart(document.getElementById('cumPnlChart'), {{
  type: 'line',
  data: {{
    labels: {_json.dumps(trade_labels)},
    datasets: [{{
      label: 'Cumulative PnL ($)',
      data: {_json.dumps(cum_pnl)},
      borderColor: '{"#22c55e" if (cum_pnl and cum_pnl[-1] >= 0) else "#ef4444"}',
      backgroundColor: '{"rgba(34,197,94,0.1)" if (cum_pnl and cum_pnl[-1] >= 0) else "rgba(239,68,68,0.1)"}',
      fill: true, tension: 0.3, pointRadius: 3,
      pointBackgroundColor: {_json.dumps(trade_colors)},
      borderWidth: 2,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ display: false }} }},
  }}
}});

// Monthly PnL
new Chart(document.getElementById('monthlyPnlChart'), {{
  type: 'bar',
  data: {{
    labels: {_json.dumps(monthly_labels)},
    datasets: [{{
      label: 'Monthly PnL ($)',
      data: {_json.dumps(monthly_values)},
      backgroundColor: {_json.dumps(monthly_colors)},
      borderRadius: 4,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ display: false }} }},
  }}
}});

// Exit Reasons Donut
new Chart(document.getElementById('reasonChart'), {{
  type: 'doughnut',
  data: {{
    labels: {_json.dumps(list(reasons.keys()))},
    datasets: [{{
      data: {_json.dumps(list(reasons.values()))},
      backgroundColor: ['#ef4444', '#22c55e', '#a855f7', '#3b82f6', '#eab308', '#f97316'].slice(0, {len(reasons)}),
      borderWidth: 0,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ position: 'bottom', labels: {{ padding: 16 }} }} }}
  }}
}});

// ─── Trade Chart Modal ─────────────────────────────
const tradeChartData = {trade_chart_json};
let currentChart = null;

function openTradeChart(idx) {{
  const data = tradeChartData[idx];
  if (!data) return;

  const modal = document.getElementById('tradeModal');
  const title = document.getElementById('modalTitle');
  const info = document.getElementById('modalInfo');
  const container = document.getElementById('tradeChartContainer');

  const pnlCls = data.pnl_usd >= 0 ? 'positive' : 'negative';
  title.innerHTML = `${{data.symbol}} (${{data.side.toUpperCase()}}) <span class="${{pnlCls}}" style="font-size:14px;margin-left:8px;">${{data.pnl_pct >= 0 ? '+' : ''}}${{data.pnl_pct.toFixed(1)}}% ($$${{data.pnl_usd.toFixed(2)}})</span>`;

  const entryDate = new Date(data.entry_time * 1000);
  const exitDate = new Date(data.exit_time * 1000);
  const durDays = Math.round((data.exit_time - data.entry_time) / 86400);
  const fmt = d => d.toLocaleDateString('ko-KR', {{year:'numeric', month:'2-digit', day:'2-digit'}});

  info.innerHTML = `
    <div class="trade-info-item"><div class="trade-info-label">Side</div><div class="trade-info-value">${{data.side.toUpperCase()}}</div></div>
    <div class="trade-info-item"><div class="trade-info-label">Signal</div><div class="trade-info-value">${{data.signal_type}}</div></div>
    <div class="trade-info-item"><div class="trade-info-label">Entry</div><div class="trade-info-value">$${{data.entry_price.toPrecision(6)}}</div></div>
    <div class="trade-info-item"><div class="trade-info-label">Exit</div><div class="trade-info-value">$${{data.exit_price.toPrecision(6)}}</div></div>
    <div class="trade-info-item"><div class="trade-info-label">Duration</div><div class="trade-info-value">${{durDays}}d</div></div>
    <div class="trade-info-item"><div class="trade-info-label">Reason</div><div class="trade-info-value">${{data.reason}}</div></div>
  `;

  container.innerHTML = '';
  if (currentChart) {{ currentChart.remove(); currentChart = null; }}

  const precision = data.precision || 6;
  const chart = LightweightCharts.createChart(container, {{
    width: container.clientWidth,
    height: 520,
    layout: {{ background: {{ type: 'solid', color: '#1a1d28' }}, textColor: '#8b8fa3' }},
    grid: {{ vertLines: {{ color: '#2a2d3a' }}, horzLines: {{ color: '#2a2d3a' }} }},
    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
    timeScale: {{ timeVisible: false, borderColor: '#2a2d3a' }},
    rightPriceScale: {{ borderColor: '#2a2d3a' }},
    localization: {{ priceFormatter: (price) => price.toFixed(precision) }},
  }});
  currentChart = chart;

  const candleSeries = chart.addCandlestickSeries({{
    upColor: '#22c55e', downColor: '#ef4444',
    borderUpColor: '#22c55e', borderDownColor: '#ef4444',
    wickUpColor: '#22c55e', wickDownColor: '#ef4444',
    priceFormat: {{ type: 'price', precision: precision, minMove: 1 / Math.pow(10, precision) }},
  }});
  candleSeries.setData(data.candles);

  // MA100 line
  if (data.ma100 && data.ma100.length > 0) {{
    const maSeries = chart.addLineSeries({{
      color: '#f97316', lineWidth: 2, lineStyle: 0,
      priceFormat: {{ type: 'price', precision: precision, minMove: 1 / Math.pow(10, precision) }},
    }});
    maSeries.setData(data.ma100);
  }}

  // Price lines
  candleSeries.createPriceLine({{ price: data.entry_price, color: '#3b82f6', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true, title: 'Entry' }});
  candleSeries.createPriceLine({{ price: data.sl, color: '#ef4444', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true, title: 'SL' }});
  if (data.tp > 0) {{
    candleSeries.createPriceLine({{ price: data.tp, color: '#a3e635', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true, title: 'TP' }});
  }}

  // Markers
  const markers = [
    {{
      time: data.entry_time,
      position: data.side === 'long' ? 'belowBar' : 'aboveBar',
      color: '#3b82f6',
      shape: data.side === 'long' ? 'arrowUp' : 'arrowDown',
      text: data.side.toUpperCase() + ' $' + data.entry_price.toPrecision(5),
    }},
    {{
      time: data.exit_time,
      position: data.side === 'long' ? 'aboveBar' : 'belowBar',
      color: data.pnl_usd >= 0 ? '#a3e635' : '#ef4444',
      shape: data.side === 'long' ? 'arrowDown' : 'arrowUp',
      text: data.reason + ' $' + data.exit_price.toPrecision(5),
    }},
  ].sort((a, b) => a.time - b.time);
  candleSeries.setMarkers(markers);

  chart.timeScale().fitContent();

  const resizeObserver = new ResizeObserver(() => {{
    chart.applyOptions({{ width: container.clientWidth }});
  }});
  resizeObserver.observe(container);

  modal.classList.add('active');
  modal.onclick = function(e) {{ if (e.target === modal) closeTradeChart(); }};
}}

function closeTradeChart() {{
  document.getElementById('tradeModal').classList.remove('active');
  if (currentChart) {{ currentChart.remove(); currentChart = null; }}
}}

document.addEventListener('keydown', function(e) {{ if (e.key === 'Escape') closeTradeChart(); }});
</script>
</body>
</html>"""

    out_path = Path("data/ma100_v2_report.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return str(out_path)


# ─── 메인 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MA100 V2: TouchBounce Only Backtest")
    parser.add_argument("--initial", type=float, default=1000.0, help="Initial balance ($)")
    parser.add_argument("--start", default="2024-06-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-01-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--max-positions", type=int, default=None, help="Max concurrent positions")
    parser.add_argument("--fixed", action="store_true", help="Fixed position size (단리, 초기 잔고 기준)")
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(hour=23, minute=59)

    params = MA100_PARAMS.copy()
    max_pos = args.max_positions or params["max_positions"]

    logger.info(f"=== MA100 Slope Backtest: {args.start} ~ {args.end} ===")

    # ── 데이터 로드 ──
    loader = DataLoader()

    # start보다 MA100 계산을 위해 100일(+여유) 앞의 데이터부터 로드
    data_start = start_dt - timedelta(days=150)
    data_start_str = data_start.strftime("%Y-%m-%d")

    logger.info(f"데이터 로드 범위: {data_start_str} ~ {args.end} (MA100 계산 포함)")

    all_data = load_all_daily_data(loader, start=data_start_str, end=args.end)

    logger.info(f"최종 데이터: {len(all_data)}개 심볼")

    if not all_data:
        logger.error("데이터가 없습니다.")
        return

    # ── 백테스트 실행 ──
    bt = MA100Backtester(
        initial_balance=args.initial,
        max_positions=max_pos,
        params=params,
        fixed_size=args.fixed,
    )

    logger.info("백테스트 시작...")
    t0 = time.time()
    bt.run(all_data, start_dt, end_dt)
    elapsed = time.time() - t0
    logger.info(f"백테스트 완료 ({elapsed:.1f}s)")

    # ── 리포트 ──
    print_report(bt, start_dt, end_dt)

    # ── HTML 리포트 ──
    # 시그널 계산된 데이터 전달 (차트에 MA100 표시용)
    precomputed = getattr(bt, "_precomputed", all_data)
    html_path = generate_html_report(bt, start_dt, end_dt, precomputed)
    logger.info(f"HTML 리포트 생성: {html_path}")

    # 브라우저로 열기
    import webbrowser
    abs_path = Path(html_path).resolve()
    webbrowser.open(f"file:///{abs_path}")


if __name__ == "__main__":
    main()
