"""
MA100 분할매도 (Partial Take-Profit) 그리드 서치

현재 MA100 전략은 100% 트레일링 스톱으로 청산.
분할매도: 일부를 고정 TP에서 먼저 익절, 나머지를 트레일링으로 수익 극대화.

exit_ratios의 마지막 원소 = 트레일링 비중, 나머지 = 고정 TP 비중.
예: [1, 1, 2] → TP1에서 25%, TP2에서 25%, 50%는 트레일링.

DCA 진입은 현재 최적값(1:1:2, 4.0%) 고정.
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_ma100 import (
    MA100_PARAMS, load_all_data
)
from src.data_loader import DataLoader

import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class MA100PartialTPBacktester:
    """MA100 백테스터 - 분할매도(Partial TP) 지원 버전."""

    def __init__(
        self,
        initial_balance: float = 1000.0,
        max_positions: int = 5,
        params: dict = None,
        exit_ratios: list = None,
        tp_interval_pct: float = 3.0,
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_positions = max_positions
        self.params = params or MA100_PARAMS.copy()
        self.positions: Dict[str, dict] = {}
        self.trades: List[dict] = []
        self.last_exit_times: Dict[str, pd.Timestamp] = {}
        self.peak_equity = initial_balance
        self.max_drawdown = 0.0
        self.max_drawdown_pct = 0.0
        self.equity_curve = []

        # 분할매도 파라미터
        # exit_ratios: 마지막 = 트레일링, 나머지 = 고정 TP
        self.exit_ratios = exit_ratios or [1]
        self.tp_interval_pct = tp_interval_pct

        # 검증용 카운터
        self.partial_tp_fills = 0
        self.full_trail_exits = 0

    def _calc_qty(self, price: float) -> float:
        pct = self.params["position_pct"]
        lev = self.params["leverage"]
        margin = self.balance * pct
        if margin <= 0:
            return 0.0
        return margin * lev / price

    def _update_equity(self, dt, precomputed):
        equity = self.balance
        for sym, pos in self.positions.items():
            df = precomputed.get(sym)
            if df is None:
                continue
            last_row = df[df["timestamp"] <= dt]
            if last_row.empty:
                continue
            cur_price = float(last_row.iloc[-1]["close"])
            remaining_size = pos["remaining_size"]

            if pos["side"] == "long":
                pnl = (cur_price - pos["avg_entry"]) / pos["avg_entry"] * remaining_size * cur_price / self.params["leverage"]
            else:
                pnl = (pos["avg_entry"] - cur_price) / pos["avg_entry"] * remaining_size * cur_price / self.params["leverage"]
            equity += pnl

        if equity > self.peak_equity:
            self.peak_equity = equity
        dd = self.peak_equity - equity
        if dd > self.max_drawdown:
            self.max_drawdown = dd
            self.max_drawdown_pct = dd / self.peak_equity * 100

        self.equity_curve.append({"timestamp": dt, "equity": equity})

    def _close_partial(self, symbol: str, exit_price: float, reason: str, exit_time: pd.Timestamp, close_size: float):
        """포지션 일부 청산."""
        pos = self.positions[symbol]
        entry = pos["avg_entry"]
        lev = self.params["leverage"]

        if pos["side"] == "long":
            pnl_pct = (exit_price - entry) / entry * 100 * lev
        else:
            pnl_pct = (entry - exit_price) / entry * 100 * lev

        pnl_usd = pnl_pct / 100 * (entry * close_size) / lev

        fee_rate = self.params.get("fee_rate", 0)
        fee_usd = 0.0
        if fee_rate > 0:
            entry_fee = close_size * entry * fee_rate
            exit_fee = close_size * exit_price * fee_rate
            fee_usd = entry_fee + exit_fee
            pnl_usd -= fee_usd

        self.balance += pnl_usd
        pos["remaining_size"] -= close_size

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
            "partial_size": close_size,
            "original_size": pos["total_size"],
        })

    def _close_all_remaining(self, symbol: str, exit_price: float, reason: str, exit_time: pd.Timestamp):
        """잔여 포지션 전체 청산."""
        pos = self.positions.get(symbol)
        if pos is None:
            return
        remaining = pos["remaining_size"]
        if remaining > 0:
            self._close_partial(symbol, exit_price, reason, exit_time, remaining)
        self.last_exit_times[symbol] = exit_time
        self.positions.pop(symbol, None)

    def _process_dca_fills(self, symbol: str, candle: pd.Series):
        """DCA 주문 체결 처리."""
        pos = self.positions.get(symbol)
        if pos is None:
            return

        pending = pos.get("pending_dca")
        if not pending:
            return

        side = pos["side"]
        high = float(candle["high"])
        low = float(candle["low"])

        filled_new = []
        remaining = []

        for dca in pending:
            if side == "short" and high >= dca["price"]:
                filled_new.append(dca)
            elif side == "long" and low <= dca["price"]:
                filled_new.append(dca)
            else:
                remaining.append(dca)

        if not filled_new:
            return

        entries = pos["filled_entries"]
        for dca in filled_new:
            entries.append({"price": dca["price"], "size": dca["size"]})

        total_size = sum(e["size"] for e in entries)
        avg_price = sum(e["price"] * e["size"] for e in entries) / total_size

        added_size = sum(d["size"] for d in filled_new)
        pos["avg_entry"] = avg_price
        pos["remaining_size"] += added_size
        pos["total_size"] = total_size
        pos["filled_entries"] = entries
        pos["pending_dca"] = remaining

        # SL 재계산
        if side == "long":
            pos["stop_loss"] = avg_price * (1 - self.params["sl_pct"] / 100)
        else:
            pos["stop_loss"] = avg_price * (1 + self.params["sl_pct"] / 100)

        # TP 레벨 재계산 (avg_entry 기준)
        self._recalculate_tp_levels(pos)

    def _recalculate_tp_levels(self, pos):
        """평균단가 변경 시 TP 레벨 재계산."""
        avg = pos["avg_entry"]
        side = pos["side"]
        exit_ratios = self.exit_ratios
        tp_interval = self.tp_interval_pct
        total_ratio = sum(exit_ratios)

        # TP 트렌치 재생성 (아직 체결 안 된 것만)
        new_tp_tranches = []
        for i in range(len(exit_ratios) - 1):  # 마지막은 trailing
            if i < len(pos.get("tp_tranches_filled", [])):
                continue  # 이미 체결된 것은 스킵
            tranche_pct = exit_ratios[i] / total_ratio
            if side == "short":
                tp_price = avg * (1 - (i + 1) * tp_interval / 100)
            else:
                tp_price = avg * (1 + (i + 1) * tp_interval / 100)
            new_tp_tranches.append({
                "idx": i,
                "tp_price": tp_price,
                "ratio_pct": tranche_pct,
            })

        pos["tp_tranches"] = new_tp_tranches

    def _check_exit_candle(self, symbol: str, candle: pd.Series, dt: pd.Timestamp):
        """캔들로 SL/TP분할/트레일링 체크."""
        # DCA 체결 먼저
        self._process_dca_fills(symbol, candle)

        pos = self.positions.get(symbol)
        if pos is None:
            return
        if pos["remaining_size"] <= 1e-12:
            self._close_all_remaining(symbol, float(candle["close"]), "Empty", dt)
            return

        avg_entry = pos["avg_entry"]
        sl = pos["stop_loss"]
        side = pos["side"]
        high = float(candle["high"])
        low = float(candle["low"])
        close = float(candle["close"])
        open_p = float(candle["open"])

        is_green = close >= open_p

        # ── 1) SL 체크 (전체 잔여 포지션) ──
        if side == "short":
            sl_hit = (high >= sl) if not is_green else (high >= sl)
            # 캔들 경로: 음봉 O→H→L→C, 양봉 O→L→H→C
            if not is_green:
                if high >= sl:
                    self._close_all_remaining(symbol, sl, "SL", dt)
                    return
            else:
                # 양봉: L먼저 → 먼저 TP 체크 가능하지만 SL이 위쪽이므로 H에서 체크
                if high >= sl:
                    self._close_all_remaining(symbol, sl, "SL", dt)
                    return
        else:  # long
            if is_green:
                if low <= sl:
                    self._close_all_remaining(symbol, sl, "SL", dt)
                    return
            else:
                if low <= sl:
                    self._close_all_remaining(symbol, sl, "SL", dt)
                    return

        # ── 2) 고정 TP 분할 체결 체크 ──
        tp_tranches = pos.get("tp_tranches", [])
        filled_indices = []
        for j, tp_info in enumerate(tp_tranches):
            tp_price = tp_info["tp_price"]
            tp_ratio = tp_info["ratio_pct"]
            tp_size = pos["total_size"] * tp_ratio

            # 잔여 수량보다 크면 잔여 전체
            tp_size = min(tp_size, pos["remaining_size"])
            if tp_size <= 1e-12:
                continue

            hit = False
            if side == "short" and low <= tp_price:
                hit = True
            elif side == "long" and high >= tp_price:
                hit = True

            if hit:
                self._close_partial(symbol, tp_price, f"TP{tp_info['idx']+1}", dt, tp_size)
                filled_indices.append(j)
                self.partial_tp_fills += 1

                if not pos.get("tp_tranches_filled"):
                    pos["tp_tranches_filled"] = []
                pos["tp_tranches_filled"].append(tp_info["idx"])

        # 체결된 트렌치 제거
        for j in sorted(filled_indices, reverse=True):
            tp_tranches.pop(j)
        pos["tp_tranches"] = tp_tranches

        # 포지션 완전 청산 체크
        if symbol not in self.positions:
            return
        if pos["remaining_size"] <= 1e-12:
            self.positions.pop(symbol)
            self.last_exit_times[symbol] = dt
            return

        # ── 3) 시그널 반전 체크 ──
        if side == "long" and candle.get("short_signal", False):
            self._close_all_remaining(symbol, close, "Reversal", dt)
            return
        if side == "short" and candle.get("long_signal", False):
            self._close_all_remaining(symbol, close, "Reversal", dt)
            return

        # ── 4) Trailing Stop (잔여 포지션 전체) ──
        if side == "long":
            cur_pnl_pct = (close - avg_entry) / avg_entry * 100
        else:
            cur_pnl_pct = (avg_entry - close) / avg_entry * 100

        trail_start = self.params["trail_start_pct"]
        trail_pct = self.params["trail_pct"]

        if cur_pnl_pct >= trail_start:
            pos["trailing"] = True
            if side == "long":
                if high > pos["highest"]:
                    pos["highest"] = high
                    pos["trail_stop"] = high * (1 - trail_pct / 100)
                if close <= pos["trail_stop"]:
                    self.full_trail_exits += 1
                    self._close_all_remaining(symbol, pos["trail_stop"], "Trail", dt)
                    return
            else:
                if low < pos["lowest"]:
                    pos["lowest"] = low
                    pos["trail_stop"] = low * (1 + trail_pct / 100)
                if close >= pos["trail_stop"]:
                    self.full_trail_exits += 1
                    self._close_all_remaining(symbol, pos["trail_stop"], "Trail", dt)
                    return
        elif pos.get("trailing"):
            if side == "long" and close <= pos["trail_stop"]:
                self.full_trail_exits += 1
                self._close_all_remaining(symbol, pos["trail_stop"], "Trail", dt)
                return
            if side == "short" and close >= pos["trail_stop"]:
                self.full_trail_exits += 1
                self._close_all_remaining(symbol, pos["trail_stop"], "Trail", dt)
                return

    def _precompute_signals(self, all_data):
        """MA100 시그널 사전 계산 (backtest_ma100과 동일)."""
        params = self.params
        ma_period = params["ma_period"]
        slope_lookback = params["slope_lookback"]
        touch_buf = params["touch_buffer_pct"] / 100

        result = {}
        for symbol, df in all_data.items():
            df = df.copy()
            df["ma100"] = df["close"].rolling(ma_period).mean()
            df["slope"] = (
                (df["ma100"] - df["ma100"].shift(slope_lookback))
                / df["ma100"].shift(slope_lookback) * 100
            )
            df["slope_prev"] = df["slope"].shift(1)

            df["reversal_long"] = (df["slope_prev"] < 0) & (df["slope"] >= 0)
            df["reversal_short"] = (df["slope_prev"] > 0) & (df["slope"] <= 0)
            df["touch_long"] = (
                (df["slope"] > 0) & (df["low"] <= df["ma100"] * (1 + touch_buf)) & (df["close"] > df["ma100"])
            )
            df["touch_short"] = (
                (df["slope"] < 0) & (df["high"] >= df["ma100"] * (1 - touch_buf)) & (df["close"] < df["ma100"])
            )
            df["long_signal"] = df["reversal_long"] | df["touch_long"]
            df["short_signal"] = df["reversal_short"] | df["touch_short"]
            df["signal_type"] = ""
            df.loc[df["reversal_short"], "signal_type"] = "SlopeRev"
            df.loc[df["touch_short"] & ~df["reversal_short"], "signal_type"] = "TouchBounce"
            df.loc[df["reversal_long"], "signal_type"] = "SlopeRev"
            df.loc[df["touch_long"] & ~df["reversal_long"], "signal_type"] = "TouchBounce"

            for col in ["long_signal", "short_signal"]:
                df[col] = df[col].fillna(False)

            result[symbol] = df
        return result

    def run(self, all_data, start_dt, end_dt):
        """백테스트 실행."""
        precomputed = self._precompute_signals(all_data)
        self._precomputed = precomputed

        sym_ts_idx = {}
        for symbol, df in precomputed.items():
            sym_ts_idx[symbol] = dict(zip(df["timestamp"], df.index))

        all_ts = set()
        for df in precomputed.values():
            mask = (df["timestamp"] >= pd.Timestamp(start_dt)) & (df["timestamp"] <= pd.Timestamp(end_dt))
            all_ts.update(df.loc[mask, "timestamp"].tolist())
        all_ts = sorted(all_ts)

        # 시그널 추출
        signals_at = {}
        for symbol, df in precomputed.items():
            short_mask = df["short_signal"] & (df["timestamp"] >= pd.Timestamp(start_dt)) & (df["timestamp"] <= pd.Timestamp(end_dt))
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

        cooldown_td = timedelta(days=self.params["cooldown_days"])

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

                last_exit = self.last_exit_times.get(symbol)
                if last_exit and (ts - last_exit) < cooldown_td:
                    continue

                entry_price = sig["price"]

                # SL 설정
                sl_price = entry_price * (1 + self.params["sl_pct"] / 100)

                # DCA 설정 (고정: 1:1:2, 4.0%)
                total_qty = self._calc_qty(entry_price)
                if total_qty <= 0:
                    break

                dca_ratios = self.params.get("dca_ratios", [1])
                dca_interval = self.params.get("dca_interval_pct", 2.0)
                total_ratio = sum(dca_ratios)
                tranche_sizes = [total_qty * r / total_ratio for r in dca_ratios]

                first_size = tranche_sizes[0]

                pending_dca = []
                for k in range(1, len(dca_ratios)):
                    dca_price = entry_price * (1 + k * dca_interval / 100)
                    pending_dca.append({"price": dca_price, "size": tranche_sizes[k]})

                # TP 트렌치 설정
                exit_ratios = self.exit_ratios
                tp_total_ratio = sum(exit_ratios)
                tp_tranches = []
                for j in range(len(exit_ratios) - 1):  # 마지막은 trailing
                    tp_pct = (j + 1) * self.tp_interval_pct
                    if side == "short":
                        tp_price = entry_price * (1 - tp_pct / 100)
                    else:
                        tp_price = entry_price * (1 + tp_pct / 100)
                    tp_tranches.append({
                        "idx": j,
                        "tp_price": tp_price,
                        "ratio_pct": exit_ratios[j] / tp_total_ratio,
                    })

                self.positions[symbol] = {
                    "symbol": symbol,
                    "side": side,
                    "avg_entry": entry_price,
                    "entry_time": ts,
                    "stop_loss": sl_price,
                    "highest": entry_price,
                    "lowest": entry_price,
                    "trail_stop": sl_price,
                    "trailing": False,
                    "remaining_size": first_size,
                    "total_size": first_size,
                    "signal_type": sig["signal_type"],
                    "pending_dca": pending_dca,
                    "filled_entries": [{"price": entry_price, "size": first_size}],
                    "tp_tranches": tp_tranches,
                    "tp_tranches_filled": [],
                }

            if i % 7 == 0:
                self._update_equity(ts, precomputed)

        # 잔여 청산
        last_ts = all_ts[-1] if all_ts else pd.Timestamp(end_dt)
        for symbol in list(self.positions.keys()):
            df = precomputed.get(symbol)
            if df is None:
                continue
            last_close = float(df.iloc[-1]["close"])
            self._close_all_remaining(symbol, last_close, "BacktestEnd", last_ts)

        self._update_equity(last_ts, precomputed)


def run_single(all_data, start_dt, end_dt, exit_ratios, tp_interval, params_base):
    """단일 조합 백테스트."""
    params = deepcopy(params_base)

    bt = MA100PartialTPBacktester(
        initial_balance=1000.0,
        max_positions=params["max_positions"],
        params=params,
        exit_ratios=exit_ratios,
        tp_interval_pct=tp_interval,
    )
    bt.run(all_data, start_dt, end_dt)

    trades = bt.trades
    n = len(trades)
    if n == 0:
        return None

    wins = [t for t in trades if t["pnl_usd"] > 0]
    losses = [t for t in trades if t["pnl_usd"] <= 0]
    total_pnl = sum(t["pnl_usd"] for t in trades)
    win_rate = len(wins) / n * 100
    gross_profit = sum(t["pnl_usd"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl_usd"] for t in losses)) if losses else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    sl_count = sum(1 for t in trades if t["reason"] == "SL")
    trail_count = sum(1 for t in trades if t["reason"] == "Trail")
    tp_count = sum(1 for t in trades if t["reason"].startswith("TP"))

    return {
        "exit_ratios": ":".join(str(r) for r in exit_ratios),
        "tp_interval": tp_interval,
        "trades": n,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "pnl_pct": total_pnl / 1000 * 100,
        "pf": pf,
        "mdd_pct": bt.max_drawdown_pct,
        "sl": sl_count,
        "trail": trail_count,
        "tp_partial": tp_count,
        "dca_fills": bt.partial_tp_fills,
        "final_bal": bt.balance,
    }


def main():
    start_dt = datetime(2024, 6, 1)
    end_dt = datetime(2026, 3, 18, 23, 59)

    print("데이터 로드 중...")
    loader = DataLoader()
    buffer_days = 150
    data_start = (start_dt - timedelta(days=buffer_days)).strftime("%Y-%m-%d")
    all_data = load_all_data(loader, timeframe="1d", start=data_start, end="2026-03-18")
    print(f"로드 완료: {len(all_data)}개 심볼\n")

    # ── 분할매도 비율 (마지막=트레일링) ──
    exit_ratio_configs = [
        # 기준선: 100% 트레일링
        [1],
        # 2분할 (TP1 + Trailing)
        [1, 3],       # 25% TP, 75% Trail
        [1, 2],       # 33% TP, 67% Trail
        [1, 1],       # 50% TP, 50% Trail
        [2, 1],       # 67% TP, 33% Trail
        [3, 1],       # 75% TP, 25% Trail
        # 3분할 (TP1 + TP2 + Trailing)
        [1, 1, 2],    # 25% + 25% TP, 50% Trail
        [1, 1, 1],    # 33% + 33% TP, 33% Trail
        [1, 2, 1],    # 25% + 50% TP, 25% Trail  (중간에 많이 정리)
        [2, 1, 1],    # 50% + 25% TP, 25% Trail  (초반 많이 정리)
        [1, 1, 3],    # 20% + 20% TP, 60% Trail
        [1, 2, 3],    # 17% + 33% TP, 50% Trail
        # 4분할 (TP1 + TP2 + TP3 + Trailing)
        [1, 1, 1, 1], # 25%×3 TP, 25% Trail
        [1, 1, 1, 3], # 17%×3 TP, 50% Trail
        [1, 1, 2, 4], # 12.5%+12.5%+25% TP, 50% Trail
        [1, 2, 3, 4], # 10%+20%+30% TP, 40% Trail
    ]

    # TP 간격 (%, 숏 기준 진입가 아래로)
    tp_intervals = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    # DCA 진입은 최적값 고정
    params_base = MA100_PARAMS.copy()
    params_base["dca_ratios"] = [1, 1, 2]
    params_base["dca_interval_pct"] = 4.0

    results = []
    total = len(exit_ratio_configs) * len(tp_intervals) - (len(tp_intervals) - 1)  # [1]은 interval 무관
    done = 0

    print(f"그리드 서치: {len(exit_ratio_configs)}개 출구비율 × {len(tp_intervals)}개 TP간격 = {total}개 조합")
    print(f"DCA 진입 고정: 1:1:2, 간격 4.0%\n")

    hdr = f"{'출구비율':<12} {'TP간격':>6} {'거래':>5} {'승률':>6} {'수익률':>9} {'PF':>5} {'MDD':>6} {'SL':>4} {'Trail':>5} {'TP청산':>6} {'최종잔고':>10}"
    print(hdr)
    print("-" * len(hdr))

    t0 = time.time()

    for exit_ratios in exit_ratio_configs:
        if exit_ratios == [1]:
            # 기준선: 100% trailing, interval 무관
            r = run_single(all_data, start_dt, end_dt, exit_ratios, 0, params_base)
            done += 1
            if r:
                results.append(r)
                print(f"{r['exit_ratios']:<12} {'N/A':>6} {r['trades']:>5} {r['win_rate']:>5.1f}% {r['pnl_pct']:>+8.1f}% {r['pf']:>5.2f} {r['mdd_pct']:>5.1f}% {r['sl']:>4} {r['trail']:>5} {r['tp_partial']:>6} ${r['final_bal']:>9,.2f}")
            continue

        for tp_interval in tp_intervals:
            r = run_single(all_data, start_dt, end_dt, exit_ratios, tp_interval, params_base)
            done += 1
            if r:
                results.append(r)
                print(f"{r['exit_ratios']:<12} {r['tp_interval']:>5.1f}% {r['trades']:>5} {r['win_rate']:>5.1f}% {r['pnl_pct']:>+8.1f}% {r['pf']:>5.2f} {r['mdd_pct']:>5.1f}% {r['sl']:>4} {r['trail']:>5} {r['tp_partial']:>6} ${r['final_bal']:>9,.2f}")

            if done % 10 == 0:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(f"  ... {done}/{total} 완료 ({elapsed:.0f}s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"\n총 {elapsed:.1f}s 소요\n")

    # ── TOP 20 수익률 기준 ──
    results.sort(key=lambda x: x["pnl_pct"], reverse=True)
    print("=" * 100)
    print("  TOP 20 (수익률 기준)")
    print("=" * 100)
    print(f"{'#':>3} {'출구비율':<12} {'TP간격':>6} {'거래':>5} {'승률':>6} {'수익률':>9} {'PF':>5} {'MDD':>6} {'SL':>4} {'Trail':>5} {'TP청산':>6} {'최종잔고':>10}")
    print("-" * 100)
    for i, r in enumerate(results[:20]):
        intv = f"{r['tp_interval']:.1f}%" if r['tp_interval'] > 0 else "N/A"
        print(f"{i+1:>3} {r['exit_ratios']:<12} {intv:>6} {r['trades']:>5} {r['win_rate']:>5.1f}% {r['pnl_pct']:>+8.1f}% {r['pf']:>5.2f} {r['mdd_pct']:>5.1f}% {r['sl']:>4} {r['trail']:>5} {r['tp_partial']:>6} ${r['final_bal']:>9,.2f}")

    # ── TOP 10 PF 기준 ──
    results_pf = sorted(results, key=lambda x: x["pf"], reverse=True)
    print(f"\n{'=' * 100}")
    print("  TOP 10 (Profit Factor 기준)")
    print("=" * 100)
    print(f"{'#':>3} {'출구비율':<12} {'TP간격':>6} {'거래':>5} {'승률':>6} {'수익률':>9} {'PF':>5} {'MDD':>6} {'SL':>4} {'Trail':>5} {'TP청산':>6} {'최종잔고':>10}")
    print("-" * 100)
    for i, r in enumerate(results_pf[:10]):
        intv = f"{r['tp_interval']:.1f}%" if r['tp_interval'] > 0 else "N/A"
        print(f"{i+1:>3} {r['exit_ratios']:<12} {intv:>6} {r['trades']:>5} {r['win_rate']:>5.1f}% {r['pnl_pct']:>+8.1f}% {r['pf']:>5.2f} {r['mdd_pct']:>5.1f}% {r['sl']:>4} {r['trail']:>5} {r['tp_partial']:>6} ${r['final_bal']:>9,.2f}")

    # ── TOP 10 MDD 낮은 순 (수익률 1000%+) ──
    results_mdd = [r for r in results if r["pnl_pct"] >= 1000]
    results_mdd.sort(key=lambda x: x["mdd_pct"])
    print(f"\n{'=' * 100}")
    print("  TOP 10 (MDD 낮은 순, 수익률 1000%+ 필터)")
    print("=" * 100)
    print(f"{'#':>3} {'출구비율':<12} {'TP간격':>6} {'거래':>5} {'승률':>6} {'수익률':>9} {'PF':>5} {'MDD':>6} {'SL':>4} {'Trail':>5} {'TP청산':>6} {'최종잔고':>10}")
    print("-" * 100)
    for i, r in enumerate(results_mdd[:10]):
        intv = f"{r['tp_interval']:.1f}%" if r['tp_interval'] > 0 else "N/A"
        print(f"{i+1:>3} {r['exit_ratios']:<12} {intv:>6} {r['trades']:>5} {r['win_rate']:>5.1f}% {r['pnl_pct']:>+8.1f}% {r['pf']:>5.2f} {r['mdd_pct']:>5.1f}% {r['sl']:>4} {r['trail']:>5} {r['tp_partial']:>6} ${r['final_bal']:>9,.2f}")

    # ── 기준선 vs 최고 비교 ──
    baseline = next((r for r in results if r["exit_ratios"] == "1"), None)
    best = results[0] if results else None
    if baseline and best and baseline != best:
        print(f"\n{'=' * 100}")
        print("  기준선 vs 최고 수익률")
        print("=" * 100)
        print(f"  기준선 (100% Trail): 수익률 {baseline['pnl_pct']:+.1f}%, PF {baseline['pf']:.2f}, MDD {baseline['mdd_pct']:.1f}%, SL {baseline['sl']}")
        print(f"  최고 ({best['exit_ratios']}, TP {best['tp_interval']:.1f}%): 수익률 {best['pnl_pct']:+.1f}%, PF {best['pf']:.2f}, MDD {best['mdd_pct']:.1f}%, SL {best['sl']}")
        diff_pnl = best['pnl_pct'] - baseline['pnl_pct']
        diff_pf = best['pf'] - baseline['pf']
        print(f"  차이: 수익률 {diff_pnl:+.1f}%p, PF {diff_pf:+.2f}")


if __name__ == "__main__":
    main()
