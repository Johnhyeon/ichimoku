"""
볼린저 밴드 + RSI 평균 회귀 단타 전략 백테스트

전략 요약:
1. 횡보장 확인: 1H/4H 볼린저 밴드 중간선이 수평
2. 숏 진입: 장대 양봉이 BB 상단 절반 이상 뚫음 + RSI >= 75
3. 롱 진입: 장대 음봉이 BB 하단 절반 이상 뚫음 + RSI <= 25
4. 익절: BB 중간선
5. 손절: 진입 캔들의 고점/저점 부근
6. 필터: BB 폭이 너무 좁으면 매매 금지
"""

import sys
sys.path.insert(0, '/home/hyeon/project/ichimoku')

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP

logger = logging.getLogger(__name__)

# 백테스트 설정
INITIAL_CAPITAL = 2100
LEVERAGE = 20
POSITION_PCT = 0.05  # 자산의 5%

# 전략 파라미터 (5차 개선: 숏 전용)
STRATEGY_PARAMS = {
    # 볼린저 밴드 설정
    "bb_period": 20,
    "bb_std": 2.0,

    # RSI 설정
    "rsi_period": 14,
    "rsi_overbought": 70,  # 숏 진입 RSI
    "rsi_oversold": 30,    # 롱 진입 RSI (사용 안 함)

    # 횡보장 판단
    "sideways_slope_threshold": 1.0,  # 중간선 기울기 임계값 (%)
    "sideways_lookback": 10,

    # BB 폭 필터
    "min_bb_width_pct": 0.3,

    # 장대 캔들 정의
    "min_candle_body_pct": 0.3,

    # 손절/익절 (넓은 손절 유지)
    "sl_buffer_pct": 0.3,

    # 기타
    "cooldown_candles": 2,
    "max_positions": 5,
    "short_only": False,  # 롱+숏 모두
}

MAJOR_COINS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT',
    'ADAUSDT', 'DOGEUSDT', 'TONUSDT', 'TRXUSDT', 'AVAXUSDT',
    'DOTUSDT', 'LINKUSDT', 'BCHUSDT', 'SUIUSDT', 'NEARUSDT',
    'LTCUSDT', 'UNIUSDT', 'APTUSDT', 'ICPUSDT', 'ETCUSDT',
    'RENDERUSDT', 'STXUSDT', 'HBARUSDT', 'XMRUSDT', 'ATOMUSDT',
    'IMXUSDT', 'FILUSDT', 'INJUSDT', 'XLMUSDT', 'ARBUSDT',
    'OPUSDT', 'VETUSDT', 'FTMUSDT', 'KASUSDT', 'TIAUSDT',
    'POLUSDT', 'SEIUSDT', 'RUNEUSDT', 'WIFUSDT', 'JUPUSDT',
    'AAVEUSDT', 'ALGOUSDT', 'SANDUSDT', 'AXSUSDT', 'MANAUSDT',
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


def calculate_indicators(df: pd.DataFrame, params: dict = STRATEGY_PARAMS) -> pd.DataFrame:
    """볼린저 밴드와 RSI 계산"""
    df = df.copy()

    # 볼린저 밴드
    df['bb_mid'] = df['close'].rolling(params['bb_period']).mean()
    df['bb_std'] = df['close'].rolling(params['bb_period']).std()
    df['bb_upper'] = df['bb_mid'] + params['bb_std'] * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - params['bb_std'] * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100  # BB 폭 (%)

    # 중간선 기울기 (횡보장 판단용)
    lookback = params['sideways_lookback']
    df['bb_mid_slope'] = (df['bb_mid'] - df['bb_mid'].shift(lookback)) / df['bb_mid'].shift(lookback) * 100

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(params['rsi_period']).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(params['rsi_period']).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # RSI 반전 신호 (핵심!)
    # 이전 캔들 RSI
    df['rsi_prev'] = df['rsi'].shift(1)
    df['rsi_prev2'] = df['rsi'].shift(2)

    # 숏 반전: RSI가 70 이상에서 하락 반전 (이전 > 현재)
    # "RSI가 한번 튀어오를 때" = RSI가 고점 찍고 내려올 때
    df['rsi_short_reversal'] = (
        (df['rsi_prev'] >= params['rsi_overbought']) &  # 이전 RSI가 70 이상
        (df['rsi'] < df['rsi_prev'])  # 현재 RSI가 하락
    )

    # 롱 반전: RSI가 30 이하에서 상승 반전 (이전 < 현재)
    df['rsi_long_reversal'] = (
        (df['rsi_prev'] <= params['rsi_oversold']) &  # 이전 RSI가 30 이하
        (df['rsi'] > df['rsi_prev'])  # 현재 RSI가 상승
    )

    # 캔들 특성
    df['candle_body'] = abs(df['close'] - df['open'])
    df['candle_range'] = df['high'] - df['low']
    df['is_bullish'] = df['close'] > df['open']
    df['is_bearish'] = df['close'] < df['open']

    # BB 돌파 비율 계산
    df['upper_breach'] = (df['high'] - df['bb_upper']) / (df['bb_upper'] - df['bb_lower'])
    df['lower_breach'] = (df['bb_lower'] - df['low']) / (df['bb_upper'] - df['bb_lower'])

    return df


def check_sideways_market(df_1h: pd.DataFrame, df_4h: pd.DataFrame,
                          current_time: pd.Timestamp, params: dict) -> bool:
    """
    1H/4H에서 횡보장인지 확인
    중간선 기울기가 임계값 이하면 횡보장
    """
    threshold = params['sideways_slope_threshold']

    # 1H 체크
    df_1h_recent = df_1h[df_1h['timestamp'] <= current_time]
    if df_1h_recent.empty or pd.isna(df_1h_recent.iloc[-1]['bb_mid_slope']):
        return False
    slope_1h = abs(df_1h_recent.iloc[-1]['bb_mid_slope'])

    # 4H 체크
    df_4h_recent = df_4h[df_4h['timestamp'] <= current_time]
    if df_4h_recent.empty or pd.isna(df_4h_recent.iloc[-1]['bb_mid_slope']):
        return False
    slope_4h = abs(df_4h_recent.iloc[-1]['bb_mid_slope'])

    # 둘 다 횡보장이어야 함
    return slope_1h < threshold and slope_4h < threshold


def run_backtest(
    data_15m: Dict[str, pd.DataFrame],
    data_1h: Dict[str, pd.DataFrame],
    data_4h: Dict[str, pd.DataFrame],
    params: dict = STRATEGY_PARAMS,
    initial_capital: float = INITIAL_CAPITAL,
) -> tuple:
    """
    볼린저 밴드 + RSI 평균 회귀 백테스트 실행
    """
    # 지표 계산
    for symbol in data_15m:
        data_15m[symbol] = calculate_indicators(data_15m[symbol], params)
    for symbol in data_1h:
        data_1h[symbol] = calculate_indicators(data_1h[symbol], params)
    for symbol in data_4h:
        data_4h[symbol] = calculate_indicators(data_4h[symbol], params)

    # 15분봉 기준으로 시간순 정렬
    all_bars = []
    for symbol, df in data_15m.items():
        df = df.dropna(subset=['bb_mid', 'rsi'])
        for idx, row in df.iterrows():
            all_bars.append({
                'symbol': symbol,
                'time': row['timestamp'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'bb_mid': row['bb_mid'],
                'bb_upper': row['bb_upper'],
                'bb_lower': row['bb_lower'],
                'bb_width': row['bb_width'],
                'rsi': row['rsi'],
                'rsi_short_reversal': row.get('rsi_short_reversal', False),  # RSI 숏 반전
                'rsi_long_reversal': row.get('rsi_long_reversal', False),    # RSI 롱 반전
                'candle_body': row['candle_body'],
                'is_bullish': row['is_bullish'],
                'is_bearish': row['is_bearish'],
                'upper_breach': row['upper_breach'],
                'lower_breach': row['lower_breach'],
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

        # 포지션 청산 체크
        for sym, pos in positions.items():
            if sym not in bars:
                continue

            bar = bars[sym]
            price = bar['close']
            high = bar['high']
            low = bar['low']
            entry = pos['entry_price']

            reason = None
            exit_price = price

            if pos['side'] == 'long':
                # 손절
                if low <= pos['stop_loss']:
                    reason = 'Stop'
                    exit_price = pos['stop_loss']
                # 익절 (BB 중간선 도달)
                elif high >= pos['take_profit']:
                    reason = 'TP'
                    exit_price = pos['take_profit']
                # MaxLoss 안전장치
                elif low <= entry * 0.98:
                    reason = 'MaxLoss'
                    exit_price = entry * 0.98

            else:  # short
                # 손절
                if high >= pos['stop_loss']:
                    reason = 'Stop'
                    exit_price = pos['stop_loss']
                # 익절 (BB 중간선 도달)
                elif low <= pos['take_profit']:
                    reason = 'TP'
                    exit_price = pos['take_profit']
                # MaxLoss 안전장치
                elif high >= entry * 1.02:
                    reason = 'MaxLoss'
                    exit_price = entry * 1.02

            if reason:
                if pos['side'] == 'long':
                    pnl_pct = (exit_price - entry) / entry * 100
                else:
                    pnl_pct = (entry - exit_price) / entry * 100

                position_size = pos['position_size']
                realized_pnl = pnl_pct * LEVERAGE / 100 * position_size
                cash += position_size + realized_pnl

                trades.append({
                    'symbol': sym,
                    'side': pos['side'],
                    'entry_time': pos['entry_time'],
                    'exit_time': t,
                    'entry_price': entry,
                    'exit_price': exit_price,
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

                # 쿨다운 체크
                if sym in last_exit:
                    candles_since = (t - last_exit[sym]).total_seconds() / (15 * 60)
                    if candles_since < params['cooldown_candles']:
                        continue

                # BB 폭 필터
                if bar['bb_width'] < params['min_bb_width_pct']:
                    continue

                # 횡보장 체크 (비활성화 - 거래 빈도 증가)
                # if sym not in data_1h or sym not in data_4h:
                #     continue
                # is_sideways = check_sideways_market(data_1h[sym], data_4h[sym], t, params)
                # if not is_sideways:
                #     continue

                price = bar['close']
                bb_mid = bar['bb_mid']
                bb_upper = bar['bb_upper']
                bb_lower = bar['bb_lower']
                rsi = bar['rsi']

                # === 숏 조건 ===
                # BB 상단 돌파 + RSI 반전 (70 이상에서 하락 전환)
                if (bar['upper_breach'] >= 0.35 and
                    bar['rsi_short_reversal']):  # RSI 반전 확인!

                    stop_loss = bar['high'] * (1 + params['sl_buffer_pct'] / 100)
                    take_profit = bb_mid

                    # 손익비 체크 (최소 1:1)
                    risk = stop_loss - price
                    reward = price - take_profit
                    if reward > 0 and risk > 0:
                        candidates.append({
                            'symbol': sym,
                            'side': 'short',
                            'price': price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'rsi': rsi,
                            'bb_width': bar['bb_width'],
                        })

                # === 롱 조건 (숏 전용 모드에서는 비활성화) ===
                elif (not params.get('short_only', False) and
                      bar['lower_breach'] >= 0.35 and
                      bar['rsi_long_reversal']):

                    stop_loss = bar['low'] * (1 - params['sl_buffer_pct'] / 100)
                    take_profit = bb_mid

                    risk = price - stop_loss
                    reward = take_profit - price
                    if reward > 0 and risk > 0:
                        candidates.append({
                            'symbol': sym,
                            'side': 'long',
                            'price': price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'rsi': rsi,
                            'bb_width': bar['bb_width'],
                        })

            # RSI가 더 극단적인 순으로 정렬
            candidates.sort(key=lambda x: abs(x['rsi'] - 50), reverse=True)

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


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)

    print("=" * 70)
    print("볼린저 밴드 + RSI 평균 회귀 단타 전략 백테스트")
    print("=" * 70)
    print(f"초기 자본: ${INITIAL_CAPITAL:,}")
    print(f"레버리지: {LEVERAGE}x")
    print(f"포지션 크기: {POSITION_PCT*100}%")
    print(f"RSI 숏 진입: >= {STRATEGY_PARAMS['rsi_overbought']}")
    print(f"RSI 롱 진입: <= {STRATEGY_PARAMS['rsi_oversold']}")
    print("=" * 70)

    # 15분, 1시간, 4시간 데이터 수집
    data_15m = {}
    data_1h = {}
    data_4h = {}

    print("\n15분봉 데이터 수집 중...")
    for i, symbol in enumerate(MAJOR_COINS[:20]):  # 20개 코인
        print(f"  {i+1}/20 {symbol}...", end='', flush=True)
        df = fetch_klines(symbol, 15, limit=20000)  # 약 208일 (7개월)
        if df is not None and not df.empty:
            data_15m[symbol] = df
            print("OK")
        else:
            print("SKIP")

    print("\n1시간봉 데이터 수집 중...")
    for symbol in data_15m.keys():
        print(f"  {symbol}...", end='', flush=True)
        df = fetch_klines(symbol, 60, limit=5000)  # 7개월
        if df is not None and not df.empty:
            data_1h[symbol] = df
            print("OK")
        else:
            print("SKIP")

    print("\n4시간봉 데이터 수집 중...")
    for symbol in data_15m.keys():
        print(f"  {symbol}...", end='', flush=True)
        df = fetch_klines(symbol, 240, limit=2000)  # 7개월
        if df is not None and not df.empty:
            data_4h[symbol] = df
            print("OK")
        else:
            print("SKIP")

    print(f"\n{len(data_15m)}개 코인 로드 완료")

    if data_15m:
        first_df = list(data_15m.values())[0]
        print(f"데이터 기간: {first_df['timestamp'].min()} ~ {first_df['timestamp'].max()}")

    print("\n백테스트 실행 중...")
    trades, equity = run_backtest(data_15m, data_1h, data_4h)
    stats = calculate_stats(trades, equity)

    # 거래 빈도 계산
    if trades:
        first_trade = min(t['entry_time'] for t in trades)
        last_trade = max(t['entry_time'] for t in trades)
        days = (last_trade - first_trade).days or 1
        trades_per_day = len(trades) / days
    else:
        days = 0
        trades_per_day = 0

    print("\n" + "=" * 70)
    print("백테스트 결과")
    print("=" * 70)
    print(f"총 거래: {stats['total_trades']}회")
    print(f"거래 빈도: {trades_per_day:.2f}회/일 ({days}일간)")
    print(f"승률: {stats['win_rate']}%")
    print(f"평균 수익: {stats['avg_pnl']}%")
    print(f"평균 승리: {stats['avg_win']}%")
    print(f"평균 손실: {stats['avg_loss']}%")
    print(f"총 수익: ${stats['total_pnl']:,.2f}")
    print(f"수익률: {stats['return_pct']}%")
    print(f"최종 자산: ${stats['final_equity']:,.2f}")
    print(f"MDD: {stats['max_dd']}%")
    print(f"Profit Factor: {stats['profit_factor']}")
    print("-" * 70)
    print(f"LONG: {stats['long_trades']}회, 승률 {stats['long_wr']}%, 수익 ${stats['long_pnl']:,.2f}")
    print(f"SHORT: {stats['short_trades']}회, 승률 {stats['short_wr']}%, 수익 ${stats['short_pnl']:,.2f}")
    print("=" * 70)

    # 청산 사유별 통계
    if trades:
        from collections import Counter
        reasons = Counter(t['reason'] for t in trades)
        print("\n[청산 사유별 통계]")
        for reason, count in reasons.most_common():
            reason_trades = [t for t in trades if t['reason'] == reason]
            reason_pnl = sum(t['pnl_usd'] for t in reason_trades)
            reason_wr = len([t for t in reason_trades if t['pnl_pct'] > 0]) / len(reason_trades) * 100
            print(f"  {reason:10}: {count:4}회 ({count/len(trades)*100:5.1f}%), "
                  f"승률 {reason_wr:5.1f}%, 수익 ${reason_pnl:>12,.2f}")

    # 최근 10개 거래
    if trades:
        print("\n[최근 10개 거래]")
        print("-" * 100)
        print(f"{'시간':<16} {'코인':<12} {'방향':<6} {'진입가':>12} {'청산가':>12} {'수익률':>10} {'수익($)':>12} {'사유':<8}")
        print("-" * 100)
        for t in sorted(trades, key=lambda x: x['exit_time'], reverse=True)[:10]:
            print(f"{t['entry_time'].strftime('%m/%d %H:%M'):<16} {t['symbol']:<12} {t['side']:<6} "
                  f"{t['entry_price']:>12.4f} {t['exit_price']:>12.4f} "
                  f"{t['pnl_pct']:>9.1f}% {t['pnl_usd']:>12.2f} {t['reason']:<8}")
