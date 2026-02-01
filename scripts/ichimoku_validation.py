"""
Ichimoku 전략 종합 검증
- RSI Divergence와 동일한 검증 프레임워크 적용
- 샘플 사이즈, EV, 연속 손실, MDD, 월별 레짐 분석
"""

import sys
sys.path.insert(0, '/home/hyeon/project/ichimoku')

import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP
from collections import Counter

from src.ichimoku import calculate_ichimoku

logger = logging.getLogger(__name__)

# === 설정 ===
INITIAL_CAPITAL = 5_000_000  # 500만원
LEVERAGE = 20
POSITION_PCT = 0.05  # 5%

STRATEGY_PARAMS = {
    "min_cloud_thickness": 0.2,
    "min_sl_pct": 0.3,
    "max_sl_pct": 8.0,
    "sl_buffer": 0.2,
    "rr_ratio": 2.0,
    "trail_pct": 1.5,
    "cooldown_hours": 4,
    "max_positions": 5,
    "use_btc_filter": True,
    "short_only": True,
}

# 4코인만 테스트 (RSI Divergence와 동일)
COINS_4 = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'HYPEUSDT']

# 전체 코인 (비교용)
MAJOR_COINS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT',
    'ADAUSDT', 'DOGEUSDT', 'TONUSDT', 'TRXUSDT', 'AVAXUSDT',
    'DOTUSDT', 'LINKUSDT', 'BCHUSDT', 'SUIUSDT', 'NEARUSDT',
    'LTCUSDT', 'UNIUSDT', 'APTUSDT', 'ICPUSDT', 'ETCUSDT',
]


def fetch_klines(symbol: str, interval: int, limit: int = 2000) -> Optional[pd.DataFrame]:
    session = HTTP()
    all_data = []
    end_time = None

    while len(all_data) < limit:
        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': interval,
            'limit': 1000
        }
        if end_time:
            params['end'] = end_time

        try:
            response = session.get_kline(**params)
            klines = response['result']['list']
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            break

        if not klines:
            break

        all_data.extend(klines)
        end_time = int(klines[-1][0]) - 1

        if len(klines) < 1000:
            break
        time.sleep(0.05)

    if not all_data:
        return None

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    return df


def run_backtest(
    all_data: Dict[str, pd.DataFrame],
    params: dict = STRATEGY_PARAMS,
    initial_capital: float = INITIAL_CAPITAL,
    leverage: float = LEVERAGE,
    position_pct: float = POSITION_PCT,
) -> tuple:
    """Ichimoku SHORT 전략 백테스트"""
    all_bars = []

    btc_trends = {}
    if 'BTCUSDT' in all_data:
        btc_df = all_data['BTCUSDT'].copy()
        btc_df['sma_26'] = btc_df['close'].rolling(26).mean()
        btc_df['sma_52'] = btc_df['close'].rolling(52).mean()
        for _, row in btc_df.iterrows():
            if pd.notna(row['sma_26']) and pd.notna(row['sma_52']):
                btc_trends[row['timestamp']] = row['sma_26'] > row['sma_52']

    for symbol, df in all_data.items():
        df = calculate_ichimoku(df)
        df = df.dropna(subset=['tenkan', 'kijun', 'cloud_top', 'cloud_bottom'])

        for idx, row in df.iterrows():
            all_bars.append({
                'symbol': symbol,
                'time': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume'],
                'tenkan': row['tenkan'],
                'kijun': row['kijun'],
                'cloud_top': row['cloud_top'],
                'cloud_bottom': row['cloud_bottom'],
                'cloud_thickness': row['cloud_thickness'],
                'cloud_green': row['cloud_green'],
                'tenkan_above': row['tenkan_above'],
                'tk_cross_down': row['tk_cross_down'],
                'kijun_cross_down': row['kijun_cross_down'],
                'chikou_bearish': row.get('chikou_bearish', False),
                'above_cloud': row['above_cloud'],
                'below_cloud': row['below_cloud'],
                'in_cloud': row['in_cloud'],
            })

    all_bars.sort(key=lambda x: x['time'])

    time_groups = {}
    for bar in all_bars:
        t = bar['time']
        if t not in time_groups:
            time_groups[t] = {}
        time_groups[t][bar['symbol']] = bar

    sorted_times = sorted(time_groups.keys())

    cash = initial_capital
    positions = {}
    trades = []
    equity_curve = []
    last_exit = {}

    for t in sorted_times:
        bars = time_groups[t]
        closed = []

        btc_uptrend = btc_trends.get(t)

        for sym, pos in positions.items():
            if sym not in bars:
                continue

            bar = bars[sym]
            price = bar['close']
            high = bar['high']
            low = bar['low']
            entry = pos['entry_price']

            if low < pos['lowest']:
                pos['lowest'] = low
                if low <= pos['take_profit']:
                    pos['trailing'] = True
                    pos['trail_stop'] = min(pos['trail_stop'], low * (1 + params['trail_pct'] / 100))

            reason = None

            max_loss_price = entry * 1.02
            if high >= max_loss_price:
                reason = 'MaxLoss'
                price = max_loss_price
            elif high >= pos['stop_loss']:
                reason = 'Stop'
                price = min(pos['stop_loss'], high)
            elif pos.get('trailing') and high >= pos['trail_stop']:
                reason = 'Trail'
                price = pos['trail_stop']
            elif not pos.get('trailing') and low <= pos['take_profit']:
                reason = 'TP'
                price = pos['take_profit']
            elif bar['in_cloud'] or bar['above_cloud']:
                reason = 'Cloud'
                price = bar['close']

            if reason:
                pnl_pct = (entry - price) / entry * 100
                position_size = pos['position_size']
                realized_pnl = pnl_pct * leverage / 100 * position_size

                cash += position_size + realized_pnl

                trades.append({
                    'symbol': sym,
                    'side': 'short',
                    'entry_time': pos['entry_time'],
                    'exit_time': t,
                    'entry_price': entry,
                    'exit_price': price,
                    'pnl_pct': round(pnl_pct * leverage, 2),
                    'pnl_usd': round(realized_pnl, 2),
                    'reason': reason
                })
                closed.append(sym)
                last_exit[sym] = t

        for s in closed:
            del positions[s]

        unrealized = 0
        for sym, pos in positions.items():
            if sym in bars:
                price = bars[sym]['close']
                pnl = (pos['entry_price'] - price) / pos['entry_price'] * leverage * pos['position_size'] / 100
                unrealized += pnl

        current_equity = cash + sum(pos['position_size'] for pos in positions.values()) + unrealized
        pos_size = current_equity * position_pct

        if cash >= pos_size and len(positions) < params['max_positions']:
            candidates = []

            for sym, bar in bars.items():
                if sym in positions:
                    continue

                if sym in last_exit:
                    if (t - last_exit[sym]).total_seconds() < params['cooldown_hours'] * 3600:
                        continue

                price = bar['close']
                cloud_bottom = bar['cloud_bottom']
                thickness = bar['cloud_thickness']

                if bar['in_cloud']:
                    continue

                if thickness < params['min_cloud_thickness']:
                    continue

                if bar['below_cloud'] and not bar['tenkan_above']:
                    has_signal = bar['tk_cross_down'] or bar['kijun_cross_down']
                    if not has_signal:
                        continue

                    if btc_uptrend is False:
                        continue

                    score = 0
                    if bar.get('chikou_bearish', False):
                        score += 2
                    if not bar.get('cloud_green', True):
                        score += 1
                    if thickness > 1.0:
                        score += 1

                    stop_loss = cloud_bottom * (1 + params['sl_buffer'] / 100)
                    sl_distance_pct = (stop_loss - price) / price * 100

                    if params['min_sl_pct'] <= sl_distance_pct <= params['max_sl_pct']:
                        take_profit = price * (1 - sl_distance_pct * params['rr_ratio'] / 100)

                        candidates.append({
                            'symbol': sym,
                            'side': 'short',
                            'price': price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'score': score,
                            'thickness': thickness
                        })

            candidates.sort(key=lambda x: (-x['score'], -x['thickness']))

            for cand in candidates:
                unrealized = 0
                for sym, pos in positions.items():
                    if sym in bars:
                        price = bars[sym]['close']
                        pnl = (pos['entry_price'] - price) / pos['entry_price'] * leverage * pos['position_size'] / 100
                        unrealized += pnl

                current_equity = cash + sum(pos['position_size'] for pos in positions.values()) + unrealized
                pos_size = current_equity * position_pct

                if cash < pos_size or len(positions) >= params['max_positions']:
                    break

                positions[cand['symbol']] = {
                    'side': cand['side'],
                    'entry_price': cand['price'],
                    'entry_time': t,
                    'stop_loss': cand['stop_loss'],
                    'take_profit': cand['take_profit'],
                    'highest': cand['price'],
                    'lowest': cand['price'],
                    'trail_stop': cand['stop_loss'],
                    'trailing': False,
                    'position_size': pos_size,
                }
                cash -= pos_size

        unrealized = 0
        total_position_size = 0
        for sym, pos in positions.items():
            total_position_size += pos['position_size']
            if sym in bars:
                price = bars[sym]['close']
                pnl = (pos['entry_price'] - price) / pos['entry_price'] * leverage * pos['position_size'] / 100
                unrealized += pnl

        equity = cash + total_position_size + unrealized
        equity_curve.append({
            'time': t,
            'equity': round(equity, 2)
        })

    return trades, equity_curve


def analyze_comprehensive(trades: List[dict], equity_curve: List[dict], initial: float) -> dict:
    """RSI Divergence와 동일한 검증 분석"""
    if not trades:
        return None

    # 기본 통계
    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]

    win_rate = len(wins) / len(trades) if trades else 0
    loss_rate = 1 - win_rate

    avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
    avg_loss = abs(np.mean([t['pnl_pct'] for t in losses])) if losses else 0

    # EV 계산
    ev = win_rate * avg_win - loss_rate * avg_loss

    # 연속 손실 분석
    max_streak = 0
    current_streak = 0
    for t in sorted(trades, key=lambda x: x['entry_time']):
        if t['pnl_pct'] <= 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    # 손실당 자본 비율 (레버리지 적용)
    loss_per_trade_pct = avg_loss  # 이미 레버리지 적용됨
    survival_after_streak = 100 - (max_streak * loss_per_trade_pct * POSITION_PCT)

    # MDD 계산
    peak = initial
    max_dd = 0
    for e in equity_curve:
        if e['equity'] > peak:
            peak = e['equity']
        dd = (peak - e['equity']) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # 월별 레짐 분석
    monthly_pnl = {}
    for t in trades:
        month = t['entry_time'].strftime('%Y-%m')
        if month not in monthly_pnl:
            monthly_pnl[month] = []
        monthly_pnl[month].append(t['pnl_usd'])

    monthly_stats = {}
    for month, pnls in monthly_pnl.items():
        monthly_stats[month] = {
            'trades': len(pnls),
            'pnl': sum(pnls),
            'avg_pnl': np.mean(pnls) if pnls else 0,
            'win_rate': len([p for p in pnls if p > 0]) / len(pnls) * 100 if pnls else 0
        }

    profitable_months = sum(1 for m in monthly_stats.values() if m['pnl'] > 0)

    final_equity = equity_curve[-1]['equity'] if equity_curve else initial
    total_pnl = sum(t['pnl_usd'] for t in trades)
    return_pct = (final_equity - initial) / initial * 100

    # 수익 팩터
    total_profit = sum(t['pnl_usd'] for t in wins) if wins else 0
    total_loss = abs(sum(t['pnl_usd'] for t in losses)) if losses else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else 999

    return {
        # 기본
        'total_trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate * 100,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        # EV
        'ev_per_trade': ev,
        # 연속 손실
        'max_consecutive_losses': max_streak,
        'loss_per_trade_pct': loss_per_trade_pct,
        'survival_after_max_streak': survival_after_streak,
        # 수익
        'total_pnl': total_pnl,
        'return_pct': return_pct,
        'final_equity': final_equity,
        'profit_factor': profit_factor,
        # MDD
        'max_dd': max_dd,
        # 월별
        'monthly_stats': monthly_stats,
        'profitable_months': profitable_months,
        'total_months': len(monthly_stats),
    }


def print_validation_report(result: dict, title: str):
    """검증 리포트 출력"""
    print("\n" + "=" * 70)
    print(f"📊 {title}")
    print("=" * 70)

    print("\n[1] 샘플 사이즈 분석")
    print("-" * 40)
    print(f"  총 거래 수: {result['total_trades']:,}회")
    print(f"  승리 거래: {result['wins']:,}회")
    print(f"  손실 거래: {result['losses']:,}회")
    sample_ok = result['wins'] >= 100
    print(f"  샘플 충분성: {'✅ 충분' if sample_ok else '⚠️ 부족 (100회 미만)'}")

    print("\n[2] EV (기대값) 분석")
    print("-" * 40)
    print(f"  승률: {result['win_rate']:.1f}%")
    print(f"  평균 승리: +{result['avg_win']:.2f}%")
    print(f"  평균 손실: -{result['avg_loss']:.2f}%")
    print(f"  EV = {result['win_rate']:.1f}% × {result['avg_win']:.2f}% - {100-result['win_rate']:.1f}% × {result['avg_loss']:.2f}%")
    print(f"  EV/거래 = {result['ev_per_trade']:+.3f}%")
    ev_ok = result['ev_per_trade'] > 0
    print(f"  평가: {'✅ 양의 기대값' if ev_ok else '❌ 음의 기대값'}")

    print("\n[3] 연속 손실 분석")
    print("-" * 40)
    print(f"  최대 연속 손실: {result['max_consecutive_losses']}회")
    print(f"  1회 손실: {result['loss_per_trade_pct']:.2f}% × {POSITION_PCT*100}% 비중 = {result['loss_per_trade_pct']*POSITION_PCT:.2f}%")
    print(f"  최대 연속 손실 시 잔고: {result['survival_after_max_streak']:.1f}%")
    survive_ok = result['survival_after_max_streak'] > 50
    print(f"  평가: {'✅ 생존 가능' if survive_ok else '❌ 생존 위험'}")

    print("\n[4] MDD (최대 낙폭) 분석")
    print("-" * 40)
    print(f"  MDD: {result['max_dd']:.1f}%")
    mdd_ok = result['max_dd'] < 50
    print(f"  평가: {'✅ 수용 가능' if mdd_ok else '⚠️ 위험 (50% 초과)'}")

    print("\n[5] 월별 레짐 분석")
    print("-" * 40)
    print(f"  수익 월: {result['profitable_months']}/{result['total_months']}개월")

    print("\n  월별 상세:")
    for month, stats in sorted(result['monthly_stats'].items()):
        emoji = "🟢" if stats['pnl'] > 0 else "🔴"
        print(f"    {month}: {emoji} ₩{stats['pnl']:>+12,.0f} | {stats['trades']:>3}거래 | 승률 {stats['win_rate']:>5.1f}%")

    regime_ok = result['profitable_months'] >= result['total_months'] // 2
    print(f"\n  평가: {'✅ 다양한 레짐에서 수익' if regime_ok else '⚠️ 레짐 의존적'}")

    print("\n[6] 종합 결과")
    print("-" * 40)
    print(f"  초기 자본: ₩{INITIAL_CAPITAL:,}")
    print(f"  최종 자본: ₩{result['final_equity']:,.0f}")
    print(f"  총 수익: ₩{result['total_pnl']:,.0f}")
    print(f"  수익률: {result['return_pct']:+.1f}%")
    print(f"  Profit Factor: {result['profit_factor']:.2f}")

    print("\n[7] 최종 평가")
    print("-" * 40)
    checks = [
        ("샘플 사이즈", sample_ok),
        ("양의 EV", ev_ok),
        ("연속손실 생존", survive_ok),
        ("MDD < 50%", mdd_ok),
        ("레짐 안정성", regime_ok),
    ]
    passed = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(f"  {'✅' if ok else '❌'} {name}")

    print(f"\n  통과: {passed}/5")
    if passed >= 4:
        print("  📈 결론: 실전 적용 가능")
    elif passed >= 3:
        print("  ⚠️ 결론: 주의해서 사용")
    else:
        print("  ❌ 결론: 사용 비추천")


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)

    print("=" * 70)
    print("🔍 Ichimoku 전략 종합 검증")
    print("=" * 70)
    print(f"초기 자본: ₩{INITIAL_CAPITAL:,}")
    print(f"레버리지: {LEVERAGE}x")
    print(f"포지션 크기: {POSITION_PCT*100}%")
    print(f"손익비: 1:{STRATEGY_PARAMS['rr_ratio']}")
    print("=" * 70)

    # === 4코인 테스트 ===
    print("\n📊 테스트 1: 4코인 (BTCUSDT, ETHUSDT, BNBUSDT, HYPEUSDT)")
    print("-" * 70)

    print("\n데이터 수집 중...")
    all_data_4 = {}

    for i, symbol in enumerate(COINS_4):
        print(f"  {i+1}/{len(COINS_4)} {symbol}...", end='', flush=True)
        df = fetch_klines(symbol, 240, limit=4000)
        if df is not None and not df.empty:
            all_data_4[symbol] = df
            print(" OK")
        else:
            print(" SKIP")

    if all_data_4:
        first_df = list(all_data_4.values())[0]
        print(f"\n데이터 기간: {first_df['timestamp'].min()} ~ {first_df['timestamp'].max()}")

        print("\n백테스트 실행 중...")
        trades_4, equity_4 = run_backtest(all_data_4)
        result_4 = analyze_comprehensive(trades_4, equity_4, INITIAL_CAPITAL)

        if result_4:
            print_validation_report(result_4, "Ichimoku SHORT 전략 (4코인)")
        else:
            print("❌ 거래 없음")

    # === 20코인 테스트 (비교용) ===
    print("\n\n" + "=" * 70)
    print("📊 테스트 2: 20코인 (비교용)")
    print("-" * 70)

    print("\n데이터 수집 중...")
    all_data_20 = {}

    for i, symbol in enumerate(MAJOR_COINS):
        print(f"  {i+1}/{len(MAJOR_COINS)} {symbol}...", end='', flush=True)
        df = fetch_klines(symbol, 240, limit=4000)
        if df is not None and not df.empty:
            all_data_20[symbol] = df
            print(" OK")
        else:
            print(" SKIP")

    if all_data_20:
        first_df = list(all_data_20.values())[0]
        print(f"\n데이터 기간: {first_df['timestamp'].min()} ~ {first_df['timestamp'].max()}")

        print("\n백테스트 실행 중...")
        trades_20, equity_20 = run_backtest(all_data_20)
        result_20 = analyze_comprehensive(trades_20, equity_20, INITIAL_CAPITAL)

        if result_20:
            print_validation_report(result_20, "Ichimoku SHORT 전략 (20코인)")
        else:
            print("❌ 거래 없음")

    # === RSI Divergence와 비교 ===
    print("\n\n" + "=" * 70)
    print("📊 RSI Divergence vs Ichimoku 비교")
    print("=" * 70)
    print("\n(RSI Divergence 결과는 STRATEGY_RESULTS.md 참고)")
    print("\n| 지표 | RSI Divergence (4코인) | Ichimoku (4코인) | Ichimoku (20코인) |")
    print("|------|------------------------|------------------|-------------------|")

    if result_4 and result_20:
        # RSI Divergence 값 (STRATEGY_RESULTS.md에서)
        rsi_return = 100.8
        rsi_mdd = 37.0
        rsi_days_10pct = 38

        print(f"| 수익률 | +{rsi_return}% | {result_4['return_pct']:+.1f}% | {result_20['return_pct']:+.1f}% |")
        print(f"| MDD | {rsi_mdd}% | {result_4['max_dd']:.1f}% | {result_20['max_dd']:.1f}% |")
        print(f"| 거래 수 | 2502회 | {result_4['total_trades']}회 | {result_20['total_trades']}회 |")
        print(f"| 승률 | 31.3% | {result_4['win_rate']:.1f}% | {result_20['win_rate']:.1f}% |")
        print(f"| EV/거래 | +0.097% | {result_4['ev_per_trade']:+.3f}% | {result_20['ev_per_trade']:+.3f}% |")
