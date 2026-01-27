"""
일목균형표(Ichimoku Cloud) 백테스트 v3
- 원본 전략 충실 구현
- 메이저 코인 20개, 시간대별 백테스트
- 시드 2100$, 레버리지 20배, qty 100$ 고정
"""
import sys
sys.path.insert(0, '..')
from pybit.unified_trading import HTTP
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta
import os
import warnings

# FutureWarning 무시 설정 (옵션은 제거 - pandas 동작 변경 방지)
warnings.filterwarnings('ignore', category=FutureWarning)
# pd.set_option('future.no_silent_downcasting', True)  # 이 옵션은 pandas 동작을 변경하여 결과에 영향을 줄 수 있음

session = HTTP()

# 메이저 코인 20개
MAJOR_COINS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT',
    'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT',
    'MATICUSDT', 'LTCUSDT', 'ATOMUSDT', 'UNIUSDT', 'ETCUSDT',
    'APTUSDT', 'NEARUSDT', 'FILUSDT', 'AAVEUSDT', 'INJUSDT'
]

# 시간대 설정
TIMEFRAMES = {
    # '15m': 15,
    # '30m': 30,
    # '1h': 60,
    '4h': 240,
}

# 백테스트 설정
INITIAL_CAPITAL = 2100
LEVERAGE = 20
POSITION_PCT = 0.05  # 자산의 5%

# 시간대별 데이터 길이 설정 (캔들 개수)
# 4h 기준 4000개면 대략 수년치 데이터 → 상승장/하락장 모두 포함
TIMEFRAME_LIMITS = {
    # '15m': 4000,
    # '30m': 4000,
    # '1h': 4000,
    '4h': 4000,
}


def fetch_klines(symbol, interval, limit=2000):
    """캔들 데이터 수집"""
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
            print(f"Error fetching {symbol}: {e}")
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


def calculate_ichimoku(df, tenkan_period=9, kijun_period=26, senkou_b_period=52):
    """일목균형표 지표 계산"""
    df = df.copy()

    # 전환선 (Tenkan-sen): 9일 고저 평균
    high_9 = df['high'].rolling(tenkan_period).max()
    low_9 = df['low'].rolling(tenkan_period).min()
    df['tenkan'] = (high_9 + low_9) / 2

    # 기준선 (Kijun-sen): 26일 고저 평균
    high_26 = df['high'].rolling(kijun_period).max()
    low_26 = df['low'].rolling(kijun_period).min()
    df['kijun'] = (high_26 + low_26) / 2

    # 선행스팬A (Senkou Span A): 전환선+기준선 평균, 26일 앞으로
    df['senkou_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(kijun_period)

    # 선행스팬B (Senkou Span B): 52일 고저 평균, 26일 앞으로
    high_52 = df['high'].rolling(senkou_b_period).max()
    low_52 = df['low'].rolling(senkou_b_period).min()
    df['senkou_b'] = ((high_52 + low_52) / 2).shift(kijun_period)

    # 구름 상단/하단
    df['cloud_top'] = df[['senkou_a', 'senkou_b']].max(axis=1)
    df['cloud_bottom'] = df[['senkou_a', 'senkou_b']].min(axis=1)

    # 구름 두께 (% 기준)
    df['cloud_thickness'] = (df['cloud_top'] - df['cloud_bottom']) / df['close'] * 100

    # 구름 색상 (녹색: 상승, 빨간색: 하락)
    df['cloud_green'] = df['senkou_a'] > df['senkou_b']

    # 전환선/기준선 크로스 신호
    df['tenkan_above'] = df['tenkan'] > df['kijun']
    tenkan_above_shifted = df['tenkan_above'].shift(1)
    df['tk_cross_up'] = (df['tenkan_above']) & (~tenkan_above_shifted.fillna(False).astype(bool))
    df['tk_cross_down'] = (~df['tenkan_above']) & (tenkan_above_shifted.fillna(True).astype(bool))

    # 가격 vs 기준선
    df['price_above_kijun'] = df['close'] > df['kijun']
    price_above_kijun_shifted = df['price_above_kijun'].shift(1)
    df['kijun_cross_up'] = (df['price_above_kijun']) & (~price_above_kijun_shifted.fillna(False).astype(bool))
    df['kijun_cross_down'] = (~df['price_above_kijun']) & (price_above_kijun_shifted.fillna(True).astype(bool))

    # 후행스팬 방향 (현재 종가 vs 26일 전 종가)
    df['chikou_bullish'] = df['close'] > df['close'].shift(26)
    df['chikou_bearish'] = df['close'] < df['close'].shift(26)

    # 가격 위치
    df['above_cloud'] = df['close'] > df['cloud_top']
    df['below_cloud'] = df['close'] < df['cloud_bottom']
    df['in_cloud'] = ~df['above_cloud'] & ~df['below_cloud']

    return df


def backtest_ichimoku(all_data, params):
    """
    일목균형표 백테스트 v3

    [원본 전략]
    롱 조건:
    1. 가격 > 구름 상단 (상승 추세)
    2. 전환선 > 기준선 (단기 강세)
    3. 전환선이 기준선을 상향 돌파 OR 가격이 기준선을 상향 돌파
    4. (선택) 후행스팬 상승

    숏 조건:
    1. 가격 < 구름 하단 (하락 추세)
    2. 전환선 < 기준선 (단기 약세)
    3. 전환선이 기준선을 하향 돌파 OR 가격이 기준선을 하향 돌파
    4. (선택) 후행스팬 하락

    손절: 구름 경계
    익절: 손절의 2배 (R:R 2:1)
    """
    all_bars = []

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

    cash = INITIAL_CAPITAL
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

            if pos['side'] == 'long':
                # 최고가 갱신 및 트레일링
                if high > pos['highest']:
                    pos['highest'] = high
                    # 익절 달성 후 트레일링
                    if high >= pos['take_profit']:
                        pos['trailing'] = True
                        pos['trail_stop'] = max(pos['trail_stop'], high * (1 - params['trail_pct'] / 100))

                reason = None

                # 0. -80% 손실 시 전량 손절 (최우선)
                # 레버리지 20배 기준: -80% = 가격 -4% 하락
                max_loss_price = entry * 0.98  # -4% 가격
                if low <= max_loss_price:
                    reason = 'MaxLoss'
                    price = max_loss_price  # 손절선 가격에서 체결
                # 1. 손절: 구름 안으로 진입 또는 손절선 터치
                elif low <= pos['stop_loss']:
                    reason = 'Stop'
                    price = max(pos['stop_loss'], low)
                # 2. 트레일링 스탑
                elif pos.get('trailing') and low <= pos['trail_stop']:
                    reason = 'Trail'
                    price = pos['trail_stop']
                # 3. 익절 (트레일링 없으면)
                elif not pos.get('trailing') and high >= pos['take_profit']:
                    reason = 'TP'
                    price = pos['take_profit']
                # 4. 구름 안으로 진입하면 청산
                elif bar['in_cloud'] or bar['below_cloud']:
                    reason = 'Cloud'
                    price = bar['close']  # 구름 조건에서는 종가로 청산

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

                # 0. -80% 손실 시 전량 손절 (최우선)
                # 레버리지 20배 기준: -80% = 가격 +4% 상승
                max_loss_price = entry * 1.02  # +4% 가격
                if high >= max_loss_price:
                    reason = 'MaxLoss'
                    price = max_loss_price  # 손절선 가격에서 체결
                # 1. 손절
                elif high >= pos['stop_loss']:
                    reason = 'Stop'
                    price = min(pos['stop_loss'], high)
                # 2. 트레일링
                elif pos.get('trailing') and high >= pos['trail_stop']:
                    reason = 'Trail'
                    price = pos['trail_stop']
                # 3. 익절
                elif not pos.get('trailing') and low <= pos['take_profit']:
                    reason = 'TP'
                    price = pos['take_profit']
                # 4. 구름 안으로 진입
                elif bar['in_cloud'] or bar['above_cloud']:
                    reason = 'Cloud'
                    price = bar['close']  # 구름 조건에서는 종가로 청산

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

        # 현재 자산 계산 (미실현 손익 포함)
        unrealized_before = 0
        for sym, pos in positions.items():
            if sym in bars:
                price = bars[sym]['close']
                if pos['side'] == 'long':
                    pnl = (price - pos['entry_price']) / pos['entry_price'] * LEVERAGE * pos['position_size'] / 100
                else:
                    pnl = (pos['entry_price'] - price) / pos['entry_price'] * LEVERAGE * pos['position_size'] / 100
                unrealized_before += pnl
        current_equity = cash + sum(pos['position_size'] for pos in positions.values()) + unrealized_before
        position_size = current_equity * POSITION_PCT

        # 신규 진입
        # 시간대 필터링: 특정 UTC 시간대에만 신규 진입 (예: 12시, 16시)
        allowed_hours = params.get('allowed_hours')
        can_enter_now = True
        if allowed_hours is not None:
            # t는 UTC 기준 타임스탬프 (Bybit Kline 기준)
            if t.hour not in allowed_hours:
                can_enter_now = False

        if can_enter_now and cash >= position_size and len(positions) < params['max_positions']:
            candidates = []

            for sym, bar in bars.items():
                if sym in positions:
                    continue

                # 쿨다운 체크
                if sym in last_exit:
                    if (t - last_exit[sym]).total_seconds() < params['cooldown_hours'] * 3600:
                        continue

                price = bar['close']
                cloud_top = bar['cloud_top']
                cloud_bottom = bar['cloud_bottom']
                kijun = bar['kijun']
                thickness = bar['cloud_thickness']

                # 구름 안에 있으면 무시 (횡보장)
                if bar['in_cloud']:
                    continue

                # 구름 두께 필터 (추세 강도)
                if thickness < params['min_cloud_thickness']:
                    continue

                # === 롱 조건 ===
                # 1. 가격 > 구름 상단
                # 2. 전환선 > 기준선
                # 3. 크로스 신호 (전환선/기준선 크로스 OR 가격/기준선 크로스)
                if bar['above_cloud'] and bar['tenkan_above']:
                    has_signal = bar['tk_cross_up'] or bar['kijun_cross_up']

                    if has_signal:
                        score = 0
                        # 후행스팬 상승이면 가산점
                        if bar['chikou_bullish']:
                            score += 2
                        # 녹색 구름이면 가산점
                        if bar['cloud_green']:
                            score += 1
                        # 구름 두께가 두꺼우면 가산점
                        if thickness > 1.0:
                            score += 1

                        # 손절: 구름 상단 (가격이 구름 안으로 들어오면 청산)
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
                        score = 0
                        if bar['chikou_bearish']:
                            score += 2
                        if not bar['cloud_green']:
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

            # 점수순 정렬하여 진입
            candidates.sort(key=lambda x: (-x['score'], -x['thickness']))

            for cand in candidates:
                # 현재 자산 재계산 (이전 진입 반영)
                unrealized_before = 0
                for sym, pos in positions.items():
                    if sym in bars:
                        price = bars[sym]['close']
                        if pos['side'] == 'long':
                            pnl = (price - pos['entry_price']) / pos['entry_price'] * LEVERAGE * pos['position_size'] / 100
                        else:
                            pnl = (pos['entry_price'] - price) / pos['entry_price'] * LEVERAGE * pos['position_size'] / 100
                        unrealized_before += pnl
                current_equity = cash + sum(pos['position_size'] for pos in positions.values()) + unrealized_before
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
                    'position_size': position_size,  # 진입 시점의 포지션 크기 저장
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


def calculate_stats(trades, equity_curve, initial=INITIAL_CAPITAL):
    """통계 계산"""
    if not trades:
        return {
            'total_trades': 0, 'win_rate': 0, 'total_pnl': 0, 'avg_pnl': 0,
            'max_dd': 0, 'final_equity': initial, 'return_pct': 0,
            'profit_factor': 0, 'avg_win': 0, 'avg_loss': 0,
            'long_trades': 0, 'short_trades': 0, 'long_wr': 0, 'short_wr': 0,
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
    }


def save_trades_csv(trades, filename):
    """거래 기록 CSV 저장"""
    if not trades:
        return

    df = pd.DataFrame(trades)
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"  CSV 저장: {filename}")


def generate_html_report(all_results):
    """종합 HTML 리포트 생성"""

    html = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>일목균형표 백테스트 리포트</title>
    <script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #0d1117; color: #c9d1d9; }
        .header { background: linear-gradient(135deg, #1a1f29 0%, #0d1117 100%); padding: 30px; text-align: center; border-bottom: 1px solid #30363d; }
        .header h1 { font-size: 28px; margin-bottom: 10px; }
        .header p { color: #8b949e; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }

        .config-box { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 20px; margin-bottom: 20px; }
        .config-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }
        .config-item { text-align: center; }
        .config-item .label { font-size: 12px; color: #8b949e; margin-bottom: 5px; }
        .config-item .value { font-size: 18px; font-weight: bold; color: #58a6ff; }

        .tf-tabs { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
        .tf-tab { padding: 12px 24px; background: #21262d; border: 1px solid #30363d; border-radius: 8px; cursor: pointer; transition: all 0.2s; }
        .tf-tab:hover { background: #30363d; }
        .tf-tab.active { background: #238636; border-color: #238636; }
        .tf-tab .tf-name { font-weight: bold; font-size: 16px; }
        .tf-tab .tf-return { font-size: 13px; margin-top: 3px; }
        .pos { color: #3fb950; }
        .neg { color: #f85149; }

        .tf-content { display: none; }
        .tf-content.active { display: block; }

        .summary { display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin-bottom: 20px; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 18px; text-align: center; }
        .card .val { font-size: 26px; font-weight: bold; margin: 8px 0; }
        .card .lbl { font-size: 12px; color: #8b949e; }

        .section { background: #161b22; border: 1px solid #30363d; border-radius: 10px; margin-bottom: 20px; padding: 20px; }
        .section h2 { font-size: 18px; margin-bottom: 15px; padding-bottom: 12px; border-bottom: 1px solid #30363d; }

        .chart-container { height: 350px; }
        .coin-btns { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 15px; }
        .coin-btn { padding: 8px 14px; background: #21262d; border: 1px solid #30363d; border-radius: 6px; cursor: pointer; font-size: 13px; color: #c9d1d9; }
        .coin-btn:hover { background: #30363d; }
        .coin-btn.active { background: #238636; border-color: #238636; }
        .coin-btn .pnl { font-size: 11px; margin-left: 5px; }
        .price-chart { height: 400px; }

        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #30363d; }
        th { background: #21262d; }
        .long { color: #3fb950; }
        .short { color: #f85149; }

        .stats-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-bottom: 20px; }
        .stat-group { background: #21262d; border-radius: 8px; padding: 15px; }
        .stat-group h3 { font-size: 14px; margin-bottom: 12px; color: #58a6ff; }
        .stat-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #30363d; }
        .stat-row:last-child { border-bottom: none; }
        .stat-label { color: #8b949e; }

        .strategy-box { background: #21262d; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
        .strategy-box h3 { color: #58a6ff; margin-bottom: 15px; }
        .strategy-box ul { list-style: none; }
        .strategy-box li { padding: 8px 0; border-bottom: 1px solid #30363d; }
        .strategy-box li:last-child { border-bottom: none; }
        .cond { color: #3fb950; font-weight: 600; }
    </style>
</head>
<body>
<div class="header">
    <h1>일목균형표 (Ichimoku Cloud) 백테스트 리포트</h1>
    <p>전환선/기준선 크로스 + 구름 필터 + 후행스팬 전략</p>
</div>
<div class="container">
    <div class="config-box">
        <div class="config-grid">
            <div class="config-item"><div class="label">초기 자본</div><div class="value">$''' + f"{INITIAL_CAPITAL:,}" + '''</div></div>
            <div class="config-item"><div class="label">레버리지</div><div class="value">''' + str(LEVERAGE) + '''x</div></div>
            <div class="config-item"><div class="label">포지션 크기</div><div class="value">''' + str(POSITION_PCT * 100) + '''%</div></div>
            <div class="config-item"><div class="label">코인 수</div><div class="value">''' + str(len(MAJOR_COINS)) + '''개</div></div>
        </div>
    </div>
    <div class="strategy-box">
        <h3>전략 조건</h3>
        <ul>
            <li><span class="cond">롱 진입:</span> 가격 > 구름 상단 + 전환선 > 기준선 + (전환선↗기준선 크로스 OR 가격↗기준선 돌파)</li>
            <li><span class="cond">숏 진입:</span> 가격 < 구름 하단 + 전환선 < 기준선 + (전환선↘기준선 크로스 OR 가격↘기준선 돌파)</li>
            <li><span class="cond">손절:</span> 구름 경계 이탈 (가격이 구름 안으로 진입 시)</li>
            <li><span class="cond">익절:</span> R:R 2:1 후 트레일링 스탑</li>
            <li><span class="cond">필터:</span> 구름 안에서는 진입하지 않음 (횡보장)</li>
        </ul>
    </div>
    <div class="tf-tabs" id="tf-tabs"></div>
    <div id="tf-contents"></div>
</div>
<script>
const allResults = ''' + json.dumps(all_results) + ''';
const tfs = Object.keys(allResults);

const tabsEl = document.getElementById('tf-tabs');
const contentsEl = document.getElementById('tf-contents');

tfs.forEach((tf, i) => {
    const d = allResults[tf];
    const tab = document.createElement('div');
    tab.className = 'tf-tab' + (i === 0 ? ' active' : '');
    tab.innerHTML = '<div class="tf-name">' + tf + '</div><div class="tf-return ' + (d.stats.return_pct >= 0 ? 'pos' : 'neg') + '">' + (d.stats.return_pct >= 0 ? '+' : '') + d.stats.return_pct + '%</div>';
    tab.onclick = () => switchTab(tf);
    tabsEl.appendChild(tab);

    const content = document.createElement('div');
    content.className = 'tf-content' + (i === 0 ? ' active' : '');
    content.id = 'content-' + tf;
    content.innerHTML = genContent(tf, d);
    contentsEl.appendChild(content);
});

function genContent(tf, d) {
    const s = d.stats;
    return '<div class="summary">' +
        '<div class="card"><div class="lbl">최종 자산</div><div class="val">$' + s.final_equity.toLocaleString() + '</div></div>' +
        '<div class="card"><div class="lbl">수익률</div><div class="val ' + (s.return_pct >= 0 ? 'pos' : 'neg') + '">' + (s.return_pct >= 0 ? '+' : '') + s.return_pct + '%</div></div>' +
        '<div class="card"><div class="lbl">총 거래</div><div class="val">' + s.total_trades + '</div></div>' +
        '<div class="card"><div class="lbl">승률</div><div class="val">' + s.win_rate + '%</div></div>' +
        '<div class="card"><div class="lbl">MDD</div><div class="val neg">' + s.max_dd + '%</div></div>' +
        '<div class="card"><div class="lbl">PF</div><div class="val">' + s.profit_factor + '</div></div>' +
        '</div>' +
        '<div class="stats-grid">' +
        '<div class="stat-group"><h3>롱</h3><div class="stat-row"><span class="stat-label">거래</span><span>' + s.long_trades + '</span></div><div class="stat-row"><span class="stat-label">승률</span><span>' + s.long_wr + '%</span></div><div class="stat-row"><span class="stat-label">평균익</span><span>' + s.avg_win + '%</span></div></div>' +
        '<div class="stat-group"><h3>숏</h3><div class="stat-row"><span class="stat-label">거래</span><span>' + s.short_trades + '</span></div><div class="stat-row"><span class="stat-label">승률</span><span>' + s.short_wr + '%</span></div><div class="stat-row"><span class="stat-label">평균손</span><span>' + s.avg_loss + '%</span></div></div>' +
        '</div>' +
        '<div class="section"><h2>자산 곡선</h2><div class="chart-container" id="eq-' + tf + '"></div></div>' +
        '<div class="section"><h2>코인별 차트</h2><div class="coin-btns" id="btns-' + tf + '"></div><div class="price-chart" id="price-' + tf + '"></div></div>' +
        '<div class="section"><h2>거래 내역</h2><table><thead><tr><th>코인</th><th>방향</th><th>진입</th><th>청산</th><th>진입가</th><th>청산가</th><th>수익률</th><th>수익($)</th><th>사유</th></tr></thead><tbody id="tbl-' + tf + '"></tbody></table></div>';
}

function switchTab(tf) {
    document.querySelectorAll('.tf-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tf-content').forEach(c => c.classList.remove('active'));
    event.target.closest('.tf-tab').classList.add('active');
    document.getElementById('content-' + tf).classList.add('active');
    initCharts(tf);
}

const charts = {};
function initCharts(tf) {
    const d = allResults[tf];
    const eqEl = document.getElementById('eq-' + tf);
    if (eqEl && !charts['eq-' + tf]) {
        const c = LightweightCharts.createChart(eqEl, {layout:{background:{type:'solid',color:'#161b22'},textColor:'#c9d1d9'},grid:{vertLines:{color:'#21262d'},horzLines:{color:'#21262d'}},width:eqEl.clientWidth,height:350,timeScale:{timeVisible:true}});
        const s = c.addAreaSeries({lineColor:'#3fb950',topColor:'rgba(63,185,80,0.3)',bottomColor:'rgba(63,185,80,0)',lineWidth:2});
        s.setData(d.equity_curve.map(e=>({time:e.time,value:e.equity})));
        c.timeScale().fitContent();
        charts['eq-' + tf] = c;
    }
    const btnsEl = document.getElementById('btns-' + tf);
    if (btnsEl && btnsEl.children.length === 0) {
        const coinPnl = {};
        d.trades.forEach(t => { coinPnl[t.symbol] = (coinPnl[t.symbol] || 0) + t.pnl_usd; });
        const coins = Object.keys(coinPnl).sort((a,b) => coinPnl[b] - coinPnl[a]);
        coins.forEach((coin, i) => {
            const btn = document.createElement('button');
            btn.className = 'coin-btn' + (i === 0 ? ' active' : '');
            const pnl = coinPnl[coin];
            btn.innerHTML = coin.replace('USDT','') + '<span class="pnl ' + (pnl>=0?'pos':'neg') + '">' + (pnl>=0?'+':'') + pnl.toFixed(0) + '</span>';
            btn.onclick = () => showCoin(tf, coin, btn);
            btnsEl.appendChild(btn);
        });
        if (coins.length > 0) showCoin(tf, coins[0], btnsEl.children[0]);
    }
    const tbl = document.getElementById('tbl-' + tf);
    if (tbl && tbl.children.length === 0) {
        d.trades.slice(-100).reverse().forEach(t => {
            const tr = document.createElement('tr');
            tr.innerHTML = '<td>' + t.symbol.replace('USDT','') + '</td><td class="' + t.side + '">' + t.side.toUpperCase() + '</td><td>' + new Date(t.entry_time).toLocaleString() + '</td><td>' + new Date(t.exit_time).toLocaleString() + '</td><td>' + t.entry_price.toFixed(t.entry_price<1?6:2) + '</td><td>' + t.exit_price.toFixed(t.exit_price<1?6:2) + '</td><td class="' + (t.pnl_pct>=0?'pos':'neg') + '">' + (t.pnl_pct>=0?'+':'') + t.pnl_pct + '%</td><td class="' + (t.pnl_usd>=0?'pos':'neg') + '">' + (t.pnl_usd>=0?'+':'') + '$' + t.pnl_usd + '</td><td>' + t.reason + '</td>';
            tbl.appendChild(tr);
        });
    }
}

function showCoin(tf, sym, btn) {
    document.getElementById('btns-' + tf).querySelectorAll('.coin-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const el = document.getElementById('price-' + tf);
    const d = allResults[tf];
    const cd = d.chart_data[sym];
    if (!cd) return;
    el.innerHTML = '';
    const c = LightweightCharts.createChart(el, {layout:{background:{type:'solid',color:'#161b22'},textColor:'#c9d1d9'},grid:{vertLines:{color:'#21262d'},horzLines:{color:'#21262d'}},width:el.clientWidth,height:400,timeScale:{timeVisible:true}});
    const candles = c.addCandlestickSeries({upColor:'#26a69a',downColor:'#ef5350',borderUpColor:'#26a69a',borderDownColor:'#ef5350',wickUpColor:'#26a69a',wickDownColor:'#ef5350'});
    candles.setData(cd.map(x=>({time:x.time,open:x.open,high:x.high,low:x.low,close:x.close})));
    const tenkan = c.addLineSeries({color:'#2196F3',lineWidth:1});
    tenkan.setData(cd.filter(x=>x.tenkan).map(x=>({time:x.time,value:x.tenkan})));
    const kijun = c.addLineSeries({color:'#F44336',lineWidth:1});
    kijun.setData(cd.filter(x=>x.kijun).map(x=>({time:x.time,value:x.kijun})));
    const ctop = c.addLineSeries({color:'rgba(76,175,80,0.5)',lineWidth:1});
    ctop.setData(cd.filter(x=>x.cloud_top).map(x=>({time:x.time,value:x.cloud_top})));
    const cbot = c.addLineSeries({color:'rgba(244,67,54,0.5)',lineWidth:1});
    cbot.setData(cd.filter(x=>x.cloud_bottom).map(x=>({time:x.time,value:x.cloud_bottom})));
    const trades = d.trades.filter(t => t.symbol === sym);
    const markers = [];
    trades.forEach(t => {
        markers.push({time:Math.floor(new Date(t.entry_time).getTime()/1000),position:t.side==='long'?'belowBar':'aboveBar',color:t.side==='long'?'#4CAF50':'#F44336',shape:t.side==='long'?'arrowUp':'arrowDown',text:t.side.toUpperCase()});
        markers.push({time:Math.floor(new Date(t.exit_time).getTime()/1000),position:t.side==='long'?'aboveBar':'belowBar',color:t.pnl_pct>=0?'#4CAF50':'#F44336',shape:'circle',text:(t.pnl_pct>=0?'+':'')+t.pnl_pct+'%'});
    });
    markers.sort((a,b)=>a.time-b.time);
    candles.setMarkers(markers);
    c.timeScale().fitContent();
}

initCharts(tfs[0]);
window.addEventListener('resize', () => { Object.values(charts).forEach(c => { if(c) c.applyOptions({width:c.options().container?.clientWidth||800}); }); });
</script>
</body>
</html>'''
    return html


def main():
    print("=" * 60)
    print("일목균형표 (Ichimoku Cloud) 백테스트 v3")
    print("=" * 60)
    print(f"초기 자본: ${INITIAL_CAPITAL:,}")
    print(f"레버리지: {LEVERAGE}x")
    print(f"포지션 크기: {POSITION_PCT * 100}% (자산 대비)")
    print(f"코인: {len(MAJOR_COINS)}개")
    print(f"시간대: {list(TIMEFRAMES.keys())}")
    print("=" * 60)

    # 전략 파라미터
    params = {
        'min_cloud_thickness': 0.2,   # 최소 구름 두께 %
        'min_sl_pct': 0.3,            # 최소 손절 거리 %
        'max_sl_pct': 8.0,            # 최대 손절 거리 %
        'sl_buffer': 0.2,             # 손절 버퍼 %
        'rr_ratio': 2.0,              # 손익비
        'trail_pct': 1.5,             # 트레일링 스탑 %
        'cooldown_hours': 4,          # 재진입 쿨다운 (시간)
        'max_positions': 5,           # 최대 동시 포지션
        # 시간대 필터링: UTC 기준 특정 시간에만 신규 진입 (예: 12시, 16시)
        # None 으로 두면 모든 시간대에서 진입 허용
        'allowed_hours': None,  # 필터링 없음 - 모든 시간대 진입
    }

    all_results = {}

    for tf_name, tf_interval in TIMEFRAMES.items():
        # print(f"\n[{tf_name}] 데이터 수집 중...")

        all_data = {}
        for i, symbol in enumerate(MAJOR_COINS):
            print(f"\r  {i+1}/{len(MAJOR_COINS)} {symbol}...", end='')
            # 시간대별로 더 긴 기간을 보기 위해 캔들 수를 늘림
            limit = TIMEFRAME_LIMITS.get(tf_name, 2000)
            df = fetch_klines(symbol, tf_interval, limit)
            if df is not None and len(df) > 100:
                all_data[symbol] = df
            time.sleep(0.05)

        print(f"\n  {len(all_data)}개 코인 로드 완료")

        if not all_data:
            continue

        print(f"  백테스트 실행 중...")
        trades, equity_curve = backtest_ichimoku(all_data, params)
        stats = calculate_stats(trades, equity_curve)

        print(f"  거래 수: {stats['total_trades']}")
        print(f"  승률: {stats['win_rate']}%")
        print(f"  수익률: {stats['return_pct']}%")
        print(f"  MDD: {stats['max_dd']}%")
        print(f"  PF: {stats['profit_factor']}")

        chart_data = {}
        for symbol, df in all_data.items():
            df = calculate_ichimoku(df)
            chart_data[symbol] = []
            for _, row in df.iterrows():
                chart_data[symbol].append({
                    'time': int(row['timestamp'].timestamp()),
                    'open': row['open'], 'high': row['high'], 'low': row['low'], 'close': row['close'],
                    'tenkan': row['tenkan'] if pd.notna(row['tenkan']) else None,
                    'kijun': row['kijun'] if pd.notna(row['kijun']) else None,
                    'cloud_top': row['cloud_top'] if pd.notna(row['cloud_top']) else None,
                    'cloud_bottom': row['cloud_bottom'] if pd.notna(row['cloud_bottom']) else None,
                })

        trades_conv = []
        for t in trades:
            trades_conv.append({**t,
                'entry_time': t['entry_time'].isoformat() if hasattr(t['entry_time'], 'isoformat') else t['entry_time'],
                'exit_time': t['exit_time'].isoformat() if hasattr(t['exit_time'], 'isoformat') else t['exit_time'],
            })

        all_results[tf_name] = {
            'stats': stats,
            'trades': trades_conv,
            'equity_curve': equity_curve,
            'chart_data': chart_data,
        }

        # 절대 경로로 저장
        output_dir = r'D:\project\auto_trading\exchange_bybit\backtest'
        save_trades_csv(trades, os.path.join(output_dir, f'ichimoku_trades_{tf_name}_nofilter.csv'))

    print("\n" + "=" * 60)
    print("HTML 리포트 생성 중...")
    html = generate_html_report(all_results)
    output_dir = r'D:\project\auto_trading\exchange_bybit\backtest'
    html_path = os.path.join(output_dir, 'ichimoku_backtest_report_nofilter.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"HTML 저장: {html_path}")

    print("\n" + "=" * 60)
    print("시간대별 결과 요약")
    print("=" * 60)
    print(f"{'시간대':<8} {'거래수':>8} {'승률':>8} {'수익률':>10} {'MDD':>8} {'PF':>8}")
    print("-" * 60)
    for tf, r in all_results.items():
        s = r['stats']
        print(f"{tf:<8} {s['total_trades']:>8} {s['win_rate']:>7}% {s['return_pct']:>9}% {s['max_dd']:>7}% {s['profit_factor']:>8}")

    print("\n완료!")
    print(f"\n결과 파일:")
    print(f"  - ichimoku_backtest_report.html")
    for tf in TIMEFRAMES.keys():
        print(f"  - ichimoku_trades_{tf}.csv")


if __name__ == '__main__':
    main()
