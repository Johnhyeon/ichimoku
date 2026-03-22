"""
Williams Fractals Backtest

윌리엄즈 프랙탈(period=5) 기반 롱/숏 진입 백테스트.
- Bearish Fractal (고점 프랙탈) 돌파 → 롱 진입
- Bullish Fractal (저점 프랙탈) 이탈 → 숏 진입

20개 주요 코인, 4h / 1d 타임프레임.

사용법:
    python scripts/backtest_williams_fractals.py
    python scripts/backtest_williams_fractals.py --timeframe 4h
    python scripts/backtest_williams_fractals.py --timeframe 1d
    python scripts/backtest_williams_fractals.py --balance 6500 --start 2025-01-02 --end 2026-03-22
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.strategy import MAJOR_COINS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Default Config ───────────────────────────────────────────

DEFAULT_CONFIG = {
    "fractal_period": 5,       # Williams Fractals N (양쪽 N개 봉 비교)
    "leverage": 5,
    "position_pct": 0.05,      # 잔고 대비 포지션 비율
    "max_positions": 5,
    "fee_rate": 0.00055,       # 테이커 수수료 (편도)
    "sl_pct": 3.0,             # 손절 %
    "tp_pct": 6.0,             # 익절 %
    "trail_start_pct": 3.0,    # 트레일링 시작 수익%
    "trail_pct": 1.5,          # 트레일링 되돌림%
    "cooldown_candles": 3,     # 청산 후 재진입 대기 캔들 수
}


# ─── Williams Fractals Calculation ────────────────────────────

def compute_fractals(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Williams Fractals 계산.

    Bearish Fractal (Up fractal): high[i]가 양쪽 n개 봉의 high보다 높을 때
    Bullish Fractal (Down fractal): low[i]가 양쪽 n개 봉의 low보다 낮을 때

    Args:
        df: OHLCV DataFrame
        n: 프랙탈 주기 (양쪽 n개 봉 비교, 총 2n+1 봉)
    """
    df = df.copy()
    highs = df["high"].values
    lows = df["low"].values
    length = len(df)

    fractal_high = np.full(length, np.nan)
    fractal_low = np.full(length, np.nan)

    for i in range(n, length - n):
        # Bearish fractal (Up): 중심이 양쪽보다 높은 고점
        is_high = True
        for j in range(1, n + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_high = False
                break
        if is_high:
            fractal_high[i] = highs[i]

        # Bullish fractal (Down): 중심이 양쪽보다 낮은 저점
        is_low = True
        for j in range(1, n + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_low = False
                break
        if is_low:
            fractal_low[i] = lows[i]

    df["fractal_high"] = fractal_high
    df["fractal_low"] = fractal_low

    # 최근 유효 프랙탈 레벨 (forward fill)
    df["last_fractal_high"] = df["fractal_high"].ffill()
    df["last_fractal_low"] = df["fractal_low"].ffill()

    return df


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    프랙탈 돌파/이탈 시그널 계산.

    - 롱: 종가가 최근 bearish fractal(고점) 돌파
    - 숏: 종가가 최근 bullish fractal(저점) 이탈
    """
    df = df.copy()

    prev_high = df["last_fractal_high"].shift(1)
    prev_low = df["last_fractal_low"].shift(1)
    prev_close = df["close"].shift(1)

    # 롱: 이전 캔들은 프랙탈 고점 아래였는데 현재 캔들 종가가 돌파
    df["long_signal"] = (
        (prev_close <= prev_high) &
        (df["close"] > df["last_fractal_high"]) &
        df["last_fractal_high"].notna()
    ).fillna(False)

    # 숏: 이전 캔들은 프랙탈 저점 위였는데 현재 캔들 종가가 이탈
    df["short_signal"] = (
        (prev_close >= prev_low) &
        (df["close"] < df["last_fractal_low"]) &
        df["last_fractal_low"].notna()
    ).fillna(False)

    return df


# ─── Data Loading ─────────────────────────────────────────────

def load_data(
    loader: DataLoader, timeframe: str,
    start_dt: datetime, end_dt: datetime,
) -> Dict[str, pd.DataFrame]:
    """MAJOR_COINS 데이터 로드."""
    available = set(loader.get_available_symbols())
    symbols = [s for s in MAJOR_COINS if s in available]
    all_data: Dict[str, pd.DataFrame] = {}

    warmup_days = 120 if timeframe == "1d" else 30
    warmup = timedelta(days=warmup_days)
    start_str = (start_dt - warmup).strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    for symbol in symbols:
        tfs = loader.get_available_timeframes(symbol)
        df = None

        if timeframe in tfs:
            df = loader.load(symbol, timeframe, start=start_str, end=end_str)

        # 1d 없으면 4h → 1d 리샘플
        if df is None and timeframe == "1d" and "4h" in tfs:
            raw = loader.load(symbol, "4h", start=start_str, end=end_str)
            if raw is not None and len(raw) >= 100:
                raw = raw.copy()
                raw["timestamp"] = pd.to_datetime(raw["timestamp"])
                df = raw.set_index("timestamp").resample("1D").agg({
                    "open": "first", "high": "max", "low": "min",
                    "close": "last", "volume": "sum",
                }).dropna().reset_index()

        if df is not None and len(df) >= 30:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
            all_data[symbol] = df

    logger.info(f"  {timeframe} 데이터: {len(all_data)}/{len(symbols)} 심볼 로드")
    return all_data


# ─── Backtest Engine ──────────────────────────────────────────

def run_backtest(
    data: Dict[str, pd.DataFrame],
    config: dict,
    start_dt: datetime,
    end_dt: datetime,
    timeframe: str,
    initial_balance: float,
) -> dict:
    """프랙탈 전략 백테스트 실행."""

    n = config["fractal_period"]

    # 시그널 사전 계산
    precomputed: Dict[str, pd.DataFrame] = {}
    for symbol, df in data.items():
        df = compute_fractals(df, n)
        df = compute_signals(df)
        # 백테스트 범위 필터
        mask = (df["timestamp"] >= pd.Timestamp(start_dt)) & (df["timestamp"] <= pd.Timestamp(end_dt))
        df_range = df[mask].reset_index(drop=True)
        if len(df_range) >= 10:
            precomputed[symbol] = df_range
    logger.info(f"  시그널 계산 완료: {len(precomputed)} 심볼")

    # 시뮬레이션
    balance = initial_balance
    positions: Dict[str, dict] = {}
    trades: List[dict] = []
    cooldowns: Dict[str, int] = {}  # symbol → 남은 쿨다운 캔들 수
    equity_curve = []

    # 모든 타임스탬프 통합 (시간순)
    all_timestamps = set()
    for df in precomputed.values():
        all_timestamps.update(df["timestamp"].tolist())
    all_timestamps = sorted(all_timestamps)

    for ts in all_timestamps:
        # 1) 포지션 관리 (청산 체크)
        closed_symbols = []
        for sym, pos in list(positions.items()):
            if sym not in precomputed:
                continue
            df = precomputed[sym]
            rows = df[df["timestamp"] == ts]
            if rows.empty:
                continue
            row = rows.iloc[0]

            side = pos["side"]
            entry_price = pos["entry_price"]
            high = row["high"]
            low = row["low"]
            close = row["close"]

            # PnL 계산
            if side == "long":
                unrealized_pct = (close / entry_price - 1) * 100
                worst_pct = (low / entry_price - 1) * 100
                best_pct = (high / entry_price - 1) * 100
            else:
                unrealized_pct = (1 - close / entry_price) * 100
                worst_pct = (1 - high / entry_price) * 100
                best_pct = (1 - low / entry_price) * 100

            # 트레일링 스탑 업데이트
            if best_pct > pos.get("best_pct", 0):
                pos["best_pct"] = best_pct

            # 청산 조건 체크
            exit_reason = None
            exit_price = close

            # 손절
            if side == "long" and low <= pos["sl_price"]:
                exit_reason = "SL"
                exit_price = pos["sl_price"]
            elif side == "short" and high >= pos["sl_price"]:
                exit_reason = "SL"
                exit_price = pos["sl_price"]

            # 익절
            if exit_reason is None:
                if side == "long" and high >= pos["tp_price"]:
                    exit_reason = "TP"
                    exit_price = pos["tp_price"]
                elif side == "short" and low <= pos["tp_price"]:
                    exit_reason = "TP"
                    exit_price = pos["tp_price"]

            # 트레일링 스탑
            if exit_reason is None and pos["best_pct"] >= config["trail_start_pct"]:
                drawdown = pos["best_pct"] - unrealized_pct
                if drawdown >= config["trail_pct"]:
                    exit_reason = "TRAIL"
                    exit_price = close

            if exit_reason:
                # 수익 계산
                if side == "long":
                    pnl_pct = (exit_price / entry_price - 1) * 100
                else:
                    pnl_pct = (1 - exit_price / entry_price) * 100

                fee_cost = pos["size"] * config["fee_rate"] * 2  # 진입+청산
                pnl_usdt = pos["size"] * config["leverage"] * pnl_pct / 100 - fee_cost

                balance += pnl_usdt
                trades.append({
                    "symbol": sym,
                    "side": side,
                    "entry_time": pos["entry_time"],
                    "exit_time": ts,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct,
                    "pnl_usdt": pnl_usdt,
                    "exit_reason": exit_reason,
                    "holding_candles": pos.get("candles", 0),
                })
                closed_symbols.append(sym)
                cooldowns[sym] = config["cooldown_candles"]

        for sym in closed_symbols:
            del positions[sym]

        # 2) 쿨다운 감소
        for sym in list(cooldowns):
            cooldowns[sym] -= 1
            if cooldowns[sym] <= 0:
                del cooldowns[sym]

        # 3) 캔들 카운트 증가
        for pos in positions.values():
            pos["candles"] = pos.get("candles", 0) + 1

        # 4) 진입 체크
        if len(positions) < config["max_positions"]:
            candidates = []
            for sym, df in precomputed.items():
                if sym in positions or sym in cooldowns:
                    continue
                rows = df[df["timestamp"] == ts]
                if rows.empty:
                    continue
                row = rows.iloc[0]

                if row["long_signal"]:
                    candidates.append((sym, "long", row))
                elif row["short_signal"]:
                    candidates.append((sym, "short", row))

            # 여러 후보가 있으면 볼륨 기준 정렬
            candidates.sort(key=lambda x: x[2].get("volume", 0), reverse=True)

            for sym, side, row in candidates:
                if len(positions) >= config["max_positions"]:
                    break

                price = row["close"]
                size = balance * config["position_pct"]
                if size < 5:
                    continue

                if side == "long":
                    sl_price = price * (1 - config["sl_pct"] / 100)
                    tp_price = price * (1 + config["tp_pct"] / 100)
                else:
                    sl_price = price * (1 + config["sl_pct"] / 100)
                    tp_price = price * (1 - config["tp_pct"] / 100)

                positions[sym] = {
                    "side": side,
                    "entry_price": price,
                    "entry_time": ts,
                    "size": size,
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "best_pct": 0,
                    "candles": 0,
                }

        # 에쿼티 기록
        total_equity = balance
        for pos in positions.values():
            total_equity += pos["size"]  # 간단히 마진만 포함
        equity_curve.append({"timestamp": ts, "equity": balance})

    # 미청산 포지션 강제 청산
    for sym, pos in positions.items():
        if sym not in precomputed:
            continue
        df = precomputed[sym]
        if df.empty:
            continue
        last_row = df.iloc[-1]
        price = last_row["close"]
        side = pos["side"]
        if side == "long":
            pnl_pct = (price / pos["entry_price"] - 1) * 100
        else:
            pnl_pct = (1 - price / pos["entry_price"]) * 100
        fee_cost = pos["size"] * config["fee_rate"] * 2
        pnl_usdt = pos["size"] * config["leverage"] * pnl_pct / 100 - fee_cost
        balance += pnl_usdt
        trades.append({
            "symbol": sym,
            "side": side,
            "entry_time": pos["entry_time"],
            "exit_time": last_row["timestamp"],
            "entry_price": pos["entry_price"],
            "exit_price": price,
            "pnl_pct": pnl_pct,
            "pnl_usdt": pnl_usdt,
            "exit_reason": "FORCE_CLOSE",
            "holding_candles": pos.get("candles", 0),
        })

    return {
        "trades": trades,
        "final_balance": balance,
        "equity_curve": equity_curve,
    }


# ─── Report ───────────────────────────────────────────────────

def print_report(
    result: dict,
    timeframe: str,
    initial_balance: float,
    config: dict,
):
    """백테스트 결과 리포트 출력."""
    trades = result["trades"]
    final = result["final_balance"]

    if not trades:
        print(f"\n{'='*60}")
        print(f"  Williams Fractals Backtest [{timeframe}]")
        print(f"{'='*60}")
        print("  거래 없음")
        return

    df = pd.DataFrame(trades)
    total_pnl = final - initial_balance
    total_return = total_pnl / initial_balance * 100

    wins = df[df["pnl_usdt"] > 0]
    losses = df[df["pnl_usdt"] <= 0]
    win_rate = len(wins) / len(df) * 100 if len(df) > 0 else 0

    avg_win = wins["pnl_usdt"].mean() if len(wins) > 0 else 0
    avg_loss = losses["pnl_usdt"].mean() if len(losses) > 0 else 0
    profit_factor = abs(wins["pnl_usdt"].sum() / losses["pnl_usdt"].sum()) if len(losses) > 0 and losses["pnl_usdt"].sum() != 0 else float("inf")

    # 최대 낙폭 (MDD)
    eq = pd.DataFrame(result["equity_curve"])
    if len(eq) > 0:
        eq["peak"] = eq["equity"].cummax()
        eq["dd"] = (eq["equity"] - eq["peak"]) / eq["peak"] * 100
        mdd = eq["dd"].min()
    else:
        mdd = 0

    # 롱/숏 분리
    longs = df[df["side"] == "long"]
    shorts = df[df["side"] == "short"]

    long_pnl = longs["pnl_usdt"].sum() if len(longs) > 0 else 0
    short_pnl = shorts["pnl_usdt"].sum() if len(shorts) > 0 else 0
    long_wr = len(longs[longs["pnl_usdt"] > 0]) / len(longs) * 100 if len(longs) > 0 else 0
    short_wr = len(shorts[shorts["pnl_usdt"] > 0]) / len(shorts) * 100 if len(shorts) > 0 else 0

    # 청산 사유별
    exit_counts = df["exit_reason"].value_counts()

    # 코인별 성과
    coin_stats = df.groupby("symbol").agg(
        trades=("pnl_usdt", "count"),
        pnl=("pnl_usdt", "sum"),
        win_rate=("pnl_usdt", lambda x: (x > 0).mean() * 100),
        avg_pnl=("pnl_usdt", "mean"),
    ).sort_values("pnl", ascending=False)

    print(f"\n{'='*70}")
    print(f"  Williams Fractals Backtest [{timeframe}] | Period={config['fractal_period']}")
    print(f"{'='*70}")
    print(f"  기간          : {df['entry_time'].min().strftime('%Y-%m-%d')} ~ {df['exit_time'].max().strftime('%Y-%m-%d')}")
    print(f"  레버리지      : {config['leverage']}x | 포지션 {config['position_pct']*100:.0f}% | 최대 {config['max_positions']}개")
    print(f"  SL/TP         : {config['sl_pct']}% / {config['tp_pct']}% | Trail {config['trail_start_pct']}%→{config['trail_pct']}%")
    print(f"{'─'*70}")
    print(f"  초기 잔고     : ${initial_balance:,.0f}")
    print(f"  최종 잔고     : ${final:,.2f}")
    print(f"  총 수익       : ${total_pnl:,.2f} ({total_return:+.1f}%)")
    print(f"  MDD           : {mdd:.1f}%")
    print(f"{'─'*70}")
    print(f"  총 거래       : {len(df)}회 (롱 {len(longs)} / 숏 {len(shorts)})")
    print(f"  승률          : {win_rate:.1f}% (롱 {long_wr:.1f}% / 숏 {short_wr:.1f}%)")
    print(f"  평균 승/패    : ${avg_win:,.2f} / ${avg_loss:,.2f}")
    print(f"  Profit Factor : {profit_factor:.2f}")
    print(f"  롱 PnL        : ${long_pnl:,.2f}")
    print(f"  숏 PnL        : ${short_pnl:,.2f}")
    print(f"{'─'*70}")
    print(f"  청산 사유:")
    for reason, count in exit_counts.items():
        pnl_by_reason = df[df["exit_reason"] == reason]["pnl_usdt"].sum()
        print(f"    {reason:12s}: {count:4d}회  ${pnl_by_reason:>+10,.2f}")
    print(f"{'─'*70}")
    print(f"  평균 보유 캔들: {df['holding_candles'].mean():.1f}")
    print(f"{'─'*70}")
    print(f"  코인별 성과 (상위/하위 5):")
    print(f"    {'심볼':<20s} {'거래':>5s} {'승률':>7s} {'총PnL':>12s} {'평균PnL':>10s}")
    for sym, row in coin_stats.head(5).iterrows():
        short_sym = sym.split('/')[0]
        print(f"    {short_sym:<20s} {row['trades']:5.0f} {row['win_rate']:6.1f}% ${row['pnl']:>+10,.2f} ${row['avg_pnl']:>+8,.2f}")
    if len(coin_stats) > 5:
        print(f"    {'...'}")
        for sym, row in coin_stats.tail(5).iterrows():
            short_sym = sym.split('/')[0]
            print(f"    {short_sym:<20s} {row['trades']:5.0f} {row['win_rate']:6.1f}% ${row['pnl']:>+10,.2f} ${row['avg_pnl']:>+8,.2f}")
    print(f"{'='*70}\n")


# ─── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Williams Fractals Backtest")
    parser.add_argument("--balance", type=float, default=6500)
    parser.add_argument("--start", type=str, default="2025-01-02")
    parser.add_argument("--end", type=str, default="2026-03-22")
    parser.add_argument("--timeframe", type=str, default="both",
                        choices=["4h", "1d", "both"])
    parser.add_argument("--leverage", type=int, default=None)
    parser.add_argument("--sl", type=float, default=None, help="SL %%")
    parser.add_argument("--tp", type=float, default=None, help="TP %%")
    parser.add_argument("--period", type=int, default=5, help="Fractal period")
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")

    config = DEFAULT_CONFIG.copy()
    config["fractal_period"] = args.period
    if args.leverage is not None:
        config["leverage"] = args.leverage
    if args.sl is not None:
        config["sl_pct"] = args.sl
    if args.tp is not None:
        config["tp_pct"] = args.tp

    loader = DataLoader()
    timeframes = ["4h", "1d"] if args.timeframe == "both" else [args.timeframe]

    for tf in timeframes:
        logger.info(f"\n{'='*50}")
        logger.info(f"  Williams Fractals [{tf}] 백테스트 시작")
        logger.info(f"{'='*50}")

        data = load_data(loader, tf, start_dt, end_dt)
        if not data:
            logger.warning(f"  {tf} 데이터 없음, 스킵")
            continue

        result = run_backtest(data, config, start_dt, end_dt, tf, args.balance)
        print_report(result, tf, args.balance, config)


if __name__ == "__main__":
    main()
