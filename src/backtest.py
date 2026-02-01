"""
일목균형표 백테스트

ichimoku_backtest.py와 동일한 로직
"""

import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP

from src.ichimoku import calculate_ichimoku
from src.strategy import LEVERAGE, POSITION_PCT, STRATEGY_PARAMS

logger = logging.getLogger(__name__)

# 백테스트 설정
INITIAL_CAPITAL = 2100

# 메이저 코인 (Bybit 심볼) - 시총 상위 50개
MAJOR_COINS = [
    # Top 1-10
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT',
    'ADAUSDT', 'DOGEUSDT', 'TONUSDT', 'TRXUSDT', 'AVAXUSDT',
    # Top 11-20
    'SHIBUSDT', 'DOTUSDT', 'LINKUSDT', 'BCHUSDT', 'SUIUSDT',
    'NEARUSDT', 'LTCUSDT', 'PEPEUSDT', 'UNIUSDT', 'APTUSDT',
    # Top 21-30
    'ICPUSDT', 'ETCUSDT', 'RENDERUSDT', 'STXUSDT', 'HBARUSDT',
    'XMRUSDT', 'ATOMUSDT', 'IMXUSDT', 'FILUSDT', 'INJUSDT',
    # Top 31-40
    'XLMUSDT', 'ARBUSDT', 'OPUSDT', 'VETUSDT', 'FTMUSDT',
    'KASUSDT', 'TIAUSDT', 'BONKUSDT', 'POLUSDT', 'SEIUSDT',
    # Top 41-50
    'RUNEUSDT', 'FLOKIUSDT', 'WIFUSDT', 'JUPUSDT', 'AAVEUSDT',
    'ALGOUSDT', 'SANDUSDT', 'AXSUSDT', 'MANAUSDT', 'THETAUSDT'
]


def fetch_klines(symbol: str, interval: int, limit: int = 2000) -> Optional[pd.DataFrame]:
    """캔들 데이터 수집"""
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
    use_btc_filter: bool = True,
    use_volume_filter: bool = True
) -> tuple:
    """
    백테스트 실행 (ichimoku_backtest.py와 동일한 로직)

    Args:
        all_data: {symbol: DataFrame} 딕셔너리
        params: 전략 파라미터
        initial_capital: 초기 자본
        use_btc_filter: BTC 도미넌스 필터 사용
        use_volume_filter: LONG 거래량 필터 사용

    Returns:
        (trades, equity_curve)
    """
    all_bars = []

    # BTC 트렌드 계산용
    btc_trends = {}
    if use_btc_filter and 'BTCUSDT' in all_data:
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
                'tk_cross_up': row['tk_cross_up'],
                'tk_cross_down': row['tk_cross_down'],
                'kijun_cross_up': row['kijun_cross_up'],
                'kijun_cross_down': row['kijun_cross_down'],
                'chikou_bullish': row.get('chikou_bullish', False),
                'chikou_bearish': row.get('chikou_bearish', False),
                'above_cloud': row['above_cloud'],
                'below_cloud': row['below_cloud'],
                'in_cloud': row['in_cloud'],
                'volume_ratio': row.get('volume_ratio', 1.0),
            })

    all_bars.sort(key=lambda x: x['time'])

    # 시간별 그룹화
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

        # BTC 트렌드
        btc_uptrend = btc_trends.get(t)

        # 포지션 청산 체크
        for sym, pos in positions.items():
            if sym not in bars:
                continue

            bar = bars[sym]
            price = bar['close']
            high = bar['high']
            low = bar['low']
            entry = pos['entry_price']

            if pos['side'] == 'long':
                if high > pos['highest']:
                    pos['highest'] = high
                    if high >= pos['take_profit']:
                        pos['trailing'] = True
                        pos['trail_stop'] = max(pos['trail_stop'], high * (1 - params['trail_pct'] / 100))

                reason = None

                max_loss_price = entry * 0.98
                if low <= max_loss_price:
                    reason = 'MaxLoss'
                    price = max_loss_price
                elif low <= pos['stop_loss']:
                    reason = 'Stop'
                    price = max(pos['stop_loss'], low)
                elif pos.get('trailing') and low <= pos['trail_stop']:
                    reason = 'Trail'
                    price = pos['trail_stop']
                elif not pos.get('trailing') and high >= pos['take_profit']:
                    reason = 'TP'
                    price = pos['take_profit']
                elif bar['in_cloud'] or bar['below_cloud']:
                    reason = 'Cloud'
                    price = bar['close']

                if reason:
                    pnl_pct = (price - entry) / entry * 100
                    position_size = pos['position_size']
                    realized_pnl = pnl_pct * LEVERAGE / 100 * position_size
                    cash += position_size + realized_pnl

                    trades.append({
                        'symbol': sym,
                        'side': 'long',
                        'entry_time': pos['entry_time'],
                        'exit_time': t,
                        'entry_price': entry,
                        'exit_price': price,
                        'pnl_pct': round(pnl_pct * LEVERAGE, 2),
                        'pnl_usd': round(realized_pnl, 2),
                        'reason': reason
                    })
                    closed.append(sym)
                    last_exit[sym] = t

            else:  # short
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
                    realized_pnl = pnl_pct * LEVERAGE / 100 * position_size
                    cash += position_size + realized_pnl

                    trades.append({
                        'symbol': sym,
                        'side': 'short',
                        'entry_time': pos['entry_time'],
                        'exit_time': t,
                        'entry_price': entry,
                        'exit_price': price,
                        'pnl_pct': round(pnl_pct * LEVERAGE, 2),
                        'pnl_usd': round(realized_pnl, 2),
                        'reason': reason
                    })
                    closed.append(sym)
                    last_exit[sym] = t

        for s in closed:
            del positions[s]

        # 현재 자산 계산
        unrealized = 0
        for sym, pos in positions.items():
            if sym in bars:
                price = bars[sym]['close']
                if pos['side'] == 'long':
                    pnl = (price - pos['entry_price']) / pos['entry_price'] * LEVERAGE * pos['position_size'] / 100
                else:
                    pnl = (pos['entry_price'] - price) / pos['entry_price'] * LEVERAGE * pos['position_size'] / 100
                unrealized += pnl

        current_equity = cash + sum(pos['position_size'] for pos in positions.values()) + unrealized
        position_size = current_equity * POSITION_PCT

        # 신규 진입
        if cash >= position_size and len(positions) < params['max_positions']:
            candidates = []

            for sym, bar in bars.items():
                if sym in positions:
                    continue

                if sym in last_exit:
                    if (t - last_exit[sym]).total_seconds() < params['cooldown_hours'] * 3600:
                        continue

                price = bar['close']
                cloud_top = bar['cloud_top']
                cloud_bottom = bar['cloud_bottom']
                thickness = bar['cloud_thickness']

                if bar['in_cloud']:
                    continue

                if thickness < params['min_cloud_thickness']:
                    continue

                # === 롱 조건 ===
                if bar['above_cloud'] and bar['tenkan_above']:
                    has_signal = bar['tk_cross_up'] or bar['kijun_cross_up']

                    if has_signal:
                        # BTC 필터
                        if use_btc_filter and btc_uptrend is True:
                            continue

                        # 후행스팬 필수
                        if params.get('long_chikou_required', True):
                            if not bar.get('chikou_bullish', False):
                                continue

                        # 거래량 필터
                        if use_volume_filter:
                            volume_ratio = bar.get('volume_ratio', 1.0)
                            if volume_ratio < params.get('long_volume_min_ratio', 1.2):
                                continue

                        score = 0
                        if bar.get('chikou_bullish', False):
                            score += 3
                        if bar.get('cloud_green', False):
                            score += 2
                        if thickness > 1.0:
                            score += 2
                        if bar.get('volume_ratio', 1.0) > 1.5:
                            score += 1

                        stop_loss = cloud_top * (1 - params['sl_buffer'] / 100)
                        sl_distance_pct = (price - stop_loss) / price * 100

                        if params['min_sl_pct'] <= sl_distance_pct <= params['max_sl_pct']:
                            take_profit = price * (1 + sl_distance_pct * params['rr_ratio'] / 100)

                            candidates.append({
                                'symbol': sym,
                                'side': 'long',
                                'price': price,
                                'stop_loss': stop_loss,
                                'take_profit': take_profit,
                                'score': score,
                                'thickness': thickness
                            })

                # === 숏 조건 ===
                elif bar['below_cloud'] and not bar['tenkan_above']:
                    has_signal = bar['tk_cross_down'] or bar['kijun_cross_down']

                    if has_signal:
                        # BTC 필터
                        if use_btc_filter and btc_uptrend is False:
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
                        if pos['side'] == 'long':
                            pnl = (price - pos['entry_price']) / pos['entry_price'] * LEVERAGE * pos['position_size'] / 100
                        else:
                            pnl = (pos['entry_price'] - price) / pos['entry_price'] * LEVERAGE * pos['position_size'] / 100
                        unrealized += pnl

                current_equity = cash + sum(pos['position_size'] for pos in positions.values()) + unrealized
                position_size = current_equity * POSITION_PCT

                if cash < position_size or len(positions) >= params['max_positions']:
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
                    'position_size': position_size,
                }
                cash -= position_size

        # 자산 기록
        unrealized = 0
        total_position_size = 0
        for sym, pos in positions.items():
            total_position_size += pos['position_size']
            if sym in bars:
                price = bars[sym]['close']
                if pos['side'] == 'long':
                    pnl = (price - pos['entry_price']) / pos['entry_price'] * LEVERAGE * pos['position_size'] / 100
                else:
                    pnl = (pos['entry_price'] - price) / pos['entry_price'] * LEVERAGE * pos['position_size'] / 100
                unrealized += pnl

        equity = cash + total_position_size + unrealized
        equity_curve.append({
            'time': int(t.timestamp()),
            'equity': round(equity, 2)
        })

    return trades, equity_curve


def calculate_stats(trades: List[dict], equity_curve: List[dict], initial: float = INITIAL_CAPITAL) -> dict:
    """통계 계산"""
    if not trades:
        return {
            'total_trades': 0, 'win_rate': 0, 'total_pnl': 0, 'avg_pnl': 0,
            'max_dd': 0, 'final_equity': initial, 'return_pct': 0,
            'profit_factor': 0, 'avg_win': 0, 'avg_loss': 0,
            'long_trades': 0, 'short_trades': 0, 'long_wr': 0, 'short_wr': 0,
            'long_pnl': 0, 'short_pnl': 0,
        }

    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]
    longs = [t for t in trades if t['side'] == 'long']
    shorts = [t for t in trades if t['side'] == 'short']
    long_wins = [t for t in longs if t['pnl_pct'] > 0]
    short_wins = [t for t in shorts if t['pnl_pct'] > 0]

    total_profit = sum(t['pnl_usd'] for t in wins) if wins else 0
    total_loss = abs(sum(t['pnl_usd'] for t in losses)) if losses else 0

    peak = initial
    max_dd = 0
    for e in equity_curve:
        if e['equity'] > peak:
            peak = e['equity']
        dd = (peak - e['equity']) / peak * 100
        if dd > max_dd:
            max_dd = dd

    final = equity_curve[-1]['equity'] if equity_curve else initial

    return {
        'total_trades': len(trades),
        'win_rate': round(len(wins) / len(trades) * 100, 1) if trades else 0,
        'total_pnl': round(sum(t['pnl_usd'] for t in trades), 2),
        'avg_pnl': round(sum(t['pnl_pct'] for t in trades) / len(trades), 2) if trades else 0,
        'max_dd': round(max_dd, 1),
        'final_equity': round(final, 2),
        'return_pct': round((final - initial) / initial * 100, 1),
        'profit_factor': round(total_profit / total_loss, 2) if total_loss > 0 else 999,
        'avg_win': round(sum(t['pnl_pct'] for t in wins) / len(wins), 2) if wins else 0,
        'avg_loss': round(sum(t['pnl_pct'] for t in losses) / len(losses), 2) if losses else 0,
        'long_trades': len(longs),
        'short_trades': len(shorts),
        'long_wr': round(len(long_wins) / len(longs) * 100, 1) if longs else 0,
        'short_wr': round(len(short_wins) / len(shorts) * 100, 1) if shorts else 0,
        'long_pnl': round(sum(t['pnl_usd'] for t in longs), 2),
        'short_pnl': round(sum(t['pnl_usd'] for t in shorts), 2),
    }


def print_report(stats: dict, trades: List[dict]):
    """결과 리포트 출력"""
    print("\n" + "=" * 60)
    print("백테스트 결과")
    print("=" * 60)
    print(f"총 거래: {stats['total_trades']}회")
    print(f"승률: {stats['win_rate']}%")
    print(f"총 수익: ${stats['total_pnl']:,.2f}")
    print(f"수익률: {stats['return_pct']}%")
    print(f"최종 자산: ${stats['final_equity']:,.2f}")
    print(f"MDD: {stats['max_dd']}%")
    print(f"Profit Factor: {stats['profit_factor']}")
    print("-" * 60)
    print(f"LONG: {stats['long_trades']}회, 승률 {stats['long_wr']}%, 수익 ${stats['long_pnl']:,.2f}")
    print(f"SHORT: {stats['short_trades']}회, 승률 {stats['short_wr']}%, 수익 ${stats['short_pnl']:,.2f}")
    print("=" * 60)
