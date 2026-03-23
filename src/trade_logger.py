"""
거래 로그 전건 누적 저장

모든 전략의 진입/청산을 CSV로 영구 저장합니다.
향후 분석용 데이터:
  - 청산 후 가격 추적 (트레일링 개선용)
  - 캔들 내 최고/최저 수익 (놓친 수익 측정)
  - ATR/변동성 (동적 트레일링용)
  - 보유 시간 (시간 기반 트레일링용)

사용법:
    from src.trade_logger import TradeLogger
    logger = TradeLogger()
    logger.log_entry(symbol, side, price, sl, tp, strategy, extra={})
    logger.log_exit(symbol, side, entry, exit_price, pnl_pct, pnl_usd, reason, strategy, extra={})
    logger.log_post_exit(symbol, exit_time, prices_after)  # 청산 후 가격 추적
"""

import csv
import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

LOG_DIR = Path("data/trade_logs")
ENTRY_LOG = LOG_DIR / "entries.csv"
EXIT_LOG = LOG_DIR / "exits.csv"
POST_EXIT_LOG = LOG_DIR / "post_exit_tracking.csv"
CANDLE_LOG = LOG_DIR / "candle_extremes.csv"

_lock = threading.Lock()


def _ensure_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _append_csv(filepath: Path, row: dict, fieldnames: list):
    """CSV에 한 줄 추가 (파일 없으면 헤더 포함 생성)."""
    with _lock:
        _ensure_dir()
        file_exists = filepath.exists() and filepath.stat().st_size > 0
        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


# ─── Entry/Exit Field Definitions ─────────────────────────────

ENTRY_FIELDS = [
    "timestamp",        # 진입 시각 (UTC)
    "symbol",           # 심볼
    "strategy",         # fractals, mirror_short, ma100
    "side",             # long / short
    "entry_price",      # 진입가
    "stop_loss",        # SL 가격
    "take_profit",      # TP 가격 (0=없음)
    "leverage",         # 레버리지
    "position_pct",     # 포지션 비율
    "margin_usdt",      # 투입 마진 (USDT)
    "qty",              # 수량
    # 진입 시점 지표
    "atr_14",           # ATR(14) — 변동성
    "adx",              # ADX — 추세 강도
    "rsi",              # RSI
    "volume_ratio",     # 거래량/SMA20 비율
    "ema_fast",         # EMA fast
    "ema_slow",         # EMA slow
    "fractal_level",    # 돌파/이탈한 프랙탈 레벨
]

EXIT_FIELDS = [
    "timestamp",        # 청산 시각 (UTC)
    "symbol",
    "strategy",
    "side",
    "entry_price",
    "exit_price",
    "pnl_pct",          # 레버리지 포함 수익률
    "pnl_usd",          # 실현 손익 (USDT)
    "reason",           # SL, TP, Trail, Cloud, Manual 등
    "leverage",
    "qty",
    "margin_usdt",
    # 보유 기간
    "entry_time",       # 진입 시각
    "hold_seconds",     # 보유 시간 (초)
    "hold_candles",     # 보유 캔들 수 (4h 기준)
    # 포지션 중 최고/최저
    "best_pnl_pct",     # 보유 중 최고 수익%
    "worst_pnl_pct",    # 보유 중 최저 수익% (최대 역행)
    "best_price",       # 최고 유리 가격
    "worst_price",      # 최저 유리 가격
    # 트레일링 상태
    "trailing_activated",  # 트레일링 활성화 여부
    "trail_stop_price",    # 트레일링 스탑 가격
    # 청산 시점 지표
    "atr_14",
    "adx",
    "rsi",
    "volume_ratio",
]

POST_EXIT_FIELDS = [
    "timestamp",        # 추적 기록 시각
    "symbol",
    "strategy",
    "side",
    "exit_price",       # 청산 가격
    "exit_time",        # 청산 시각
    "exit_reason",      # 청산 사유
    # 청산 후 가격
    "hours_after",      # 청산 후 경과 시간
    "price_now",        # 현재 가격
    "missed_pnl_pct",   # 놓친 수익% (청산 안 했으면 얼마나 더 벌었나)
    "max_favorable",    # 청산 후 최대 유리 이동%
    "max_adverse",      # 청산 후 최대 불리 이동%
]

CANDLE_FIELDS = [
    "timestamp",        # 캔들 시각
    "symbol",
    "strategy",
    "side",
    "entry_price",
    "candle_high",      # 캔들 고가
    "candle_low",       # 캔들 저가
    "candle_close",     # 캔들 종가
    "unrealized_best",  # 캔들 내 최고 수익%
    "unrealized_worst", # 캔들 내 최저 수익%
    "unrealized_close", # 캔들 종가 기준 수익%
]


class TradeLogger:
    """전략 공용 거래 로그."""

    def log_entry(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        strategy: str,
        leverage: float = 0,
        position_pct: float = 0,
        margin_usdt: float = 0,
        qty: float = 0,
        indicators: Optional[dict] = None,
    ):
        """진입 기록."""
        ind = indicators or {}
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "strategy": strategy,
            "side": side,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "leverage": leverage,
            "position_pct": position_pct,
            "margin_usdt": round(margin_usdt, 2),
            "qty": qty,
            "atr_14": ind.get("atr_14", ""),
            "adx": ind.get("adx", ""),
            "rsi": ind.get("rsi", ""),
            "volume_ratio": ind.get("volume_ratio", ""),
            "ema_fast": ind.get("ema_fast", ""),
            "ema_slow": ind.get("ema_slow", ""),
            "fractal_level": ind.get("fractal_level", ""),
        }
        try:
            _append_csv(ENTRY_LOG, row, ENTRY_FIELDS)
        except Exception as e:
            logger.error(f"Entry log failed: {e}")

    def log_exit(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl_pct: float,
        pnl_usd: float,
        reason: str,
        strategy: str,
        leverage: float = 0,
        qty: float = 0,
        margin_usdt: float = 0,
        entry_time: Optional[datetime] = None,
        best_pnl_pct: float = 0,
        worst_pnl_pct: float = 0,
        best_price: float = 0,
        worst_price: float = 0,
        trailing_activated: bool = False,
        trail_stop_price: float = 0,
        indicators: Optional[dict] = None,
    ):
        """청산 기록."""
        now = datetime.utcnow()
        hold_sec = (now - entry_time).total_seconds() if entry_time else 0
        hold_candles = hold_sec / (4 * 3600)  # 4h 캔들 기준

        ind = indicators or {}
        row = {
            "timestamp": now.isoformat(),
            "symbol": symbol,
            "strategy": strategy,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_pct": round(pnl_pct, 4),
            "pnl_usd": round(pnl_usd, 4),
            "reason": reason,
            "leverage": leverage,
            "qty": qty,
            "margin_usdt": round(margin_usdt, 2),
            "entry_time": entry_time.isoformat() if entry_time else "",
            "hold_seconds": round(hold_sec),
            "hold_candles": round(hold_candles, 1),
            "best_pnl_pct": round(best_pnl_pct, 4),
            "worst_pnl_pct": round(worst_pnl_pct, 4),
            "best_price": best_price,
            "worst_price": worst_price,
            "trailing_activated": trailing_activated,
            "trail_stop_price": trail_stop_price,
            "atr_14": ind.get("atr_14", ""),
            "adx": ind.get("adx", ""),
            "rsi": ind.get("rsi", ""),
            "volume_ratio": ind.get("volume_ratio", ""),
        }
        try:
            _append_csv(EXIT_LOG, row, EXIT_FIELDS)
        except Exception as e:
            logger.error(f"Exit log failed: {e}")

    def log_post_exit(
        self,
        symbol: str,
        strategy: str,
        side: str,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
        hours_after: float,
        price_now: float,
        max_favorable: float = 0,
        max_adverse: float = 0,
    ):
        """청산 후 가격 추적 기록."""
        if side == "long":
            missed = (price_now / exit_price - 1) * 100
        else:
            missed = (1 - price_now / exit_price) * 100

        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "strategy": strategy,
            "side": side,
            "exit_price": exit_price,
            "exit_time": exit_time.isoformat(),
            "exit_reason": exit_reason,
            "hours_after": round(hours_after, 1),
            "price_now": price_now,
            "missed_pnl_pct": round(missed, 4),
            "max_favorable": round(max_favorable, 4),
            "max_adverse": round(max_adverse, 4),
        }
        try:
            _append_csv(POST_EXIT_LOG, row, POST_EXIT_FIELDS)
        except Exception as e:
            logger.error(f"Post-exit log failed: {e}")

    def log_candle_extreme(
        self,
        symbol: str,
        strategy: str,
        side: str,
        entry_price: float,
        candle_high: float,
        candle_low: float,
        candle_close: float,
        candle_time: datetime,
    ):
        """캔들별 미실현 손익 극값 기록."""
        if side == "long":
            best = (candle_high / entry_price - 1) * 100
            worst = (candle_low / entry_price - 1) * 100
            close_pnl = (candle_close / entry_price - 1) * 100
        else:
            best = (1 - candle_low / entry_price) * 100
            worst = (1 - candle_high / entry_price) * 100
            close_pnl = (1 - candle_close / entry_price) * 100

        row = {
            "timestamp": candle_time.isoformat() if isinstance(candle_time, datetime) else str(candle_time),
            "symbol": symbol,
            "strategy": strategy,
            "side": side,
            "entry_price": entry_price,
            "candle_high": candle_high,
            "candle_low": candle_low,
            "candle_close": candle_close,
            "unrealized_best": round(best, 4),
            "unrealized_worst": round(worst, 4),
            "unrealized_close": round(close_pnl, 4),
        }
        try:
            _append_csv(CANDLE_LOG, row, CANDLE_FIELDS)
        except Exception as e:
            logger.error(f"Candle log failed: {e}")
