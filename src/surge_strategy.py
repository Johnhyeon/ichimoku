"""
급등 코인 탐지 전략 (롱 전용)

패턴 분석 결과 기반:
- 눌림목에서 거래량 폭발과 함께 반등하는 코인 탐지
- RSI 과매도 구간에서 반등
- 볼린저 밴드 하단에서 반등
"""

import logging
from datetime import datetime
from typing import Optional, Dict, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# 전략 파라미터 (데이터 분석 기반 v2)
# 분석 결과:
# - SMA25 대비 -10% 이하: 성공률 20%
# - Volume 5x 이상: 성공률 6.7%
# - RSI 35-40: 성공률 4.2%
# - RSI 급등 +10 이상: 성공률 4.0%
SURGE_STRATEGY_PARAMS = {
    # 진입 조건 (기본 필터 - 점수로 최종 필터링)
    "rsi_oversold": 45,
    "rsi_min": 20,
    "bb_position_max": 0.35,
    "volume_ratio_min": 1.8,      # 기본 필터 (점수에서 5x 보너스)
    "price_below_sma25_pct": -2,  # 기본 필터 (점수에서 -10% 보너스)

    # 반등 확인
    "bounce_confirm_pct": 1.0,
    "green_candle_required": True,

    # 리스크 관리
    "sl_pct": 5.0,
    "tp_pct": 12.0,
    "trail_start_pct": 8.0,
    "trail_pct": 3.0,

    # 포지션
    "leverage": 5,
    "position_pct": 0.03,
    "max_positions": 3,

    # 필터
    "min_volume_24h": 1000000,
    "min_atr_pct": 2.0,
    "max_atr_pct": 15.0,
    "min_score": 10,              # 점수 10 이상만 (성공률 20%+)
    "min_bb_width": 0.06,         # BB 폭 6% 이상
}


def calculate_surge_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """급등 탐지용 지표 계산"""
    df = df.copy()

    # 이동평균
    df['sma_7'] = df['close'].rolling(7).mean()
    df['sma_25'] = df['close'].rolling(25).mean()
    df['sma_99'] = df['close'].rolling(99).mean()

    # EMA
    df['ema_9'] = df['close'].ewm(span=9).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()

    # 볼린저 밴드
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # 거래량
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['close'] * 100

    # 가격 변화
    df['change_1h'] = df['close'].pct_change(1) * 100
    df['change_4h'] = df['close'].pct_change(4) * 100

    # 이평선 대비 위치
    df['price_vs_sma7'] = (df['close'] / df['sma_7'] - 1) * 100
    df['price_vs_sma25'] = (df['close'] / df['sma_25'] - 1) * 100
    df['price_vs_sma99'] = (df['close'] / df['sma_99'] - 1) * 100

    # 캔들 패턴
    df['is_green'] = df['close'] > df['open']
    df['candle_body'] = abs(df['close'] - df['open']) / df['open'] * 100
    df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open'] * 100

    # 최근 저점 대비
    df['low_5'] = df['low'].rolling(5).min()
    df['bounce_from_low'] = (df['close'] / df['low_5'] - 1) * 100

    # MACD
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    return df


def get_surge_entry_signal(
    symbol: str,
    df: pd.DataFrame,
    params: dict = SURGE_STRATEGY_PARAMS
) -> Optional[dict]:
    """
    급등 진입 신호 계산

    조건:
    1. RSI 과매도 구간 (20~45)
    2. 볼린저 밴드 하단 근처 (position < 0.35)
    3. 가격이 SMA25 아래
    4. 거래량 폭발 (1.5x 이상)
    5. 양봉 확인 (반등)

    Args:
        symbol: 심볼
        df: OHLCV + 지표 DataFrame
        params: 전략 파라미터

    Returns:
        진입 신호 딕셔너리 또는 None
    """
    if df is None or len(df) < 30:
        return None

    row = df.iloc[-1]
    prev = df.iloc[-2]

    price = float(row['close'])
    rsi = float(row['rsi']) if pd.notna(row['rsi']) else 50
    bb_position = float(row['bb_position']) if pd.notna(row['bb_position']) else 0.5
    volume_ratio = float(row['volume_ratio']) if pd.notna(row['volume_ratio']) else 1
    price_vs_sma25 = float(row['price_vs_sma25']) if pd.notna(row['price_vs_sma25']) else 0
    atr_pct = float(row['atr_pct']) if pd.notna(row['atr_pct']) else 3
    is_green = bool(row['is_green'])

    # === 필터 조건 ===

    # ATR 필터 (변동성)
    if atr_pct < params['min_atr_pct'] or atr_pct > params['max_atr_pct']:
        return None

    # BB Width 필터 (변동성이 너무 낮으면 제외)
    bb_width = float(row['bb_width']) if pd.notna(row['bb_width']) else 0
    min_bb_width = params.get('min_bb_width', 0.08)
    if bb_width < min_bb_width * 100:  # bb_width는 % 단위
        return None

    # === 진입 조건 ===

    # 1. RSI 과매도 구간
    if not (params['rsi_min'] <= rsi <= params['rsi_oversold']):
        return None

    # 2. 볼린저 밴드 하단 근처
    if bb_position > params['bb_position_max']:
        return None

    # 3. 가격 < SMA25 (눌림목)
    if price_vs_sma25 > params['price_below_sma25_pct']:
        return None

    # 4. 거래량 폭발
    if volume_ratio < params['volume_ratio_min']:
        return None

    # 5. 양봉 (반등 확인)
    if params['green_candle_required'] and not is_green:
        return None

    # === 점수 계산 (데이터 분석 기반 v2) ===
    score = 0

    # 1. SMA25 대비 가격 (가장 강력한 지표! 성공률 20%)
    if price_vs_sma25 < -10:
        score += 4  # 핵심 조건
    elif price_vs_sma25 < -5:
        score += 2

    # 2. 거래량 폭발 (5x 이상 성공률 6.7%)
    if volume_ratio >= 5:
        score += 4  # 폭발적 거래량
    elif volume_ratio >= 3:
        score += 2
    elif volume_ratio >= 2:
        score += 1

    # 3. RSI 범위 (35-40이 최적, 성공률 4.2%)
    if 35 <= rsi <= 40:
        score += 2  # 최적 범위
    elif 30 <= rsi < 35:
        score += 1
    elif 25 <= rsi < 30:
        score += 1
    # RSI < 25는 너무 위험 (성공률 1.3%)

    # 4. RSI 급등 (모멘텀 전환, +10 이상 성공률 4.0%)
    prev_rsi = float(prev['rsi']) if pd.notna(prev['rsi']) else rsi
    rsi_change = rsi - prev_rsi
    if rsi_change >= 10:
        score += 3  # 강한 모멘텀 전환
    elif rsi_change >= 5:
        score += 1

    # 5. BB 폭 (변동성, 성공 시 0.19 vs 실패 시 0.07)
    bb_width = float(row['bb_width']) if pd.notna(row['bb_width']) else 0
    if bb_width >= 15:  # 15% 이상
        score += 2
    elif bb_width >= 10:
        score += 1

    # 6. 볼밴 위치 (0.1-0.3이 최적)
    if 0.1 <= bb_position <= 0.25:
        score += 1
    # 0.0 근처는 오히려 성공률 낮음

    # 7. 하락 후 반등 (이전 캔들이 음봉)
    if not bool(prev['is_green']):
        score += 1

    # 최소 점수 필터
    min_score = params.get('min_score', 5)
    if score < min_score:
        return None

    # === 손절/익절 계산 ===
    stop_loss = price * (1 - params['sl_pct'] / 100)
    take_profit = price * (1 + params['tp_pct'] / 100)

    return {
        'symbol': symbol,
        'side': 'long',
        'price': price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'score': score,
        'rsi': rsi,
        'bb_position': bb_position,
        'volume_ratio': volume_ratio,
        'price_vs_sma25': price_vs_sma25,
        'atr_pct': atr_pct,
    }


def check_surge_exit_signal(
    pos: dict,
    row: pd.Series,
    params: dict = SURGE_STRATEGY_PARAMS
) -> Optional[dict]:
    """
    급등 포지션 청산 신호

    Args:
        pos: 포지션 정보
        row: 현재 캔들 데이터
        params: 전략 파라미터

    Returns:
        청산 신호 또는 None
    """
    entry = float(pos['entry_price'])
    stop_loss = float(pos['stop_loss'])
    take_profit = float(pos['take_profit'])
    trail_stop = float(pos.get('trail_stop', 0))
    trailing = bool(pos.get('trailing', False))

    price = float(row['close'])
    high = float(row['high'])
    low = float(row['low'])

    # 현재 수익률
    pnl_pct = (price - entry) / entry * 100

    # 최고가 갱신
    highest = float(pos.get('highest', entry))
    if high > highest:
        highest = high
        pos['highest'] = highest

        # 트레일링 시작 조건
        gain_from_entry = (highest - entry) / entry * 100
        if gain_from_entry >= params['trail_start_pct']:
            trailing = True
            new_trail = highest * (1 - params['trail_pct'] / 100)
            trail_stop = max(trail_stop, new_trail)
            pos['trailing'] = True
            pos['trail_stop'] = trail_stop

    # 1. 손절
    if low <= stop_loss:
        return {'action': 'close', 'reason': 'Stop', 'price': max(stop_loss, low)}

    # 2. 트레일링 스탑
    if trailing and low <= trail_stop:
        return {'action': 'close', 'reason': 'Trail', 'price': trail_stop}

    # 3. 익절 (트레일링 없으면)
    if not trailing and high >= take_profit:
        return {'action': 'close', 'reason': 'TP', 'price': take_profit}

    # 4. RSI 과매수 청산
    rsi = float(row['rsi']) if pd.notna(row.get('rsi')) else 50
    if rsi > 75 and pnl_pct > 3:
        return {'action': 'close', 'reason': 'RSI_Overbought', 'price': price}

    # 5. 급락 방어 (급격한 하락)
    change_1h = float(row.get('change_1h', 0)) if pd.notna(row.get('change_1h')) else 0
    if change_1h < -5 and pnl_pct < 0:
        return {'action': 'close', 'reason': 'Crash', 'price': price}

    return None


def scan_surge_candidates(
    symbols: List[str],
    data_fetcher,
    timeframe: str = '1h',
    params: dict = SURGE_STRATEGY_PARAMS
) -> List[dict]:
    """
    급등 후보 코인 스캔

    Args:
        symbols: 스캔할 심볼 리스트
        data_fetcher: DataFetcher 인스턴스
        timeframe: 타임프레임
        params: 전략 파라미터

    Returns:
        진입 신호 리스트 (점수순 정렬)
    """
    candidates = []

    for symbol in symbols:
        try:
            df = data_fetcher.get_ohlcv(symbol, timeframe, limit=100)
            if df is None or len(df) < 30:
                continue

            # 지표 계산
            df = calculate_surge_indicators(df)

            # 진입 신호 체크
            signal = get_surge_entry_signal(symbol, df, params)
            if signal:
                candidates.append(signal)
                logger.info(f"급등 후보: {symbol} | Score={signal['score']} RSI={signal['rsi']:.1f} Vol={signal['volume_ratio']:.1f}x")

        except Exception as e:
            logger.debug(f"스캔 실패 ({symbol}): {e}")
            continue

    # 점수순 정렬
    candidates.sort(key=lambda x: x['score'], reverse=True)

    return candidates


# 바이빗 전체 USDT 무기한 선물 종목 가져오기
def get_all_usdt_perpetuals():
    """바이빗 전체 USDT 무기한 선물 종목 조회"""
    import ccxt
    try:
        exchange = ccxt.bybit({'options': {'defaultType': 'swap'}})
        markets = exchange.load_markets()

        # USDT 무기한 선물만 필터링 (BTC, ETH 같은 대형주 제외 옵션)
        symbols = [
            symbol for symbol, market in markets.items()
            if market.get('swap') and
               market.get('quote') == 'USDT' and
               market.get('active', True)
        ]

        # 알파벳 순 정렬
        symbols.sort()
        return symbols
    except Exception as e:
        print(f"종목 조회 실패: {e}")
        return []


# 제외할 대형주 (원하면 스캔에서 제외)
EXCLUDE_LARGE_CAPS = ['BTC/USDT:USDT', 'ETH/USDT:USDT']

# 하위 호환성을 위한 기본 리스트 (동적 조회 실패시 폴백)
SURGE_WATCH_LIST_FALLBACK = [
    "ZORA/USDT:USDT", "CYS/USDT:USDT", "ZKP/USDT:USDT", "MEGA/USDT:USDT",
    "HANA/USDT:USDT", "WIF/USDT:USDT", "PEPE/USDT:USDT", "ARB/USDT:USDT",
]

# 전체 종목 리스트 (동적 로드)
_cached_watch_list = None

def get_surge_watch_list(exclude_large_caps=True):
    """스캔 대상 종목 리스트 반환 (캐싱)"""
    global _cached_watch_list

    if _cached_watch_list is None:
        _cached_watch_list = get_all_usdt_perpetuals()

    if not _cached_watch_list:
        return SURGE_WATCH_LIST_FALLBACK

    if exclude_large_caps:
        return [s for s in _cached_watch_list if s not in EXCLUDE_LARGE_CAPS]

    return _cached_watch_list

# 기존 코드 호환성 유지
SURGE_WATCH_LIST = SURGE_WATCH_LIST_FALLBACK  # 초기값, 실제 사용시 get_surge_watch_list() 호출


def check_15m_entry_timing(df_15m: pd.DataFrame) -> dict:
    """
    15분봉에서 진입 타이밍 확인

    로직:
    1. 최근 캔들 중 양봉 출현 확인
    2. 양봉 이후 음봉 확인 시 진입

    Args:
        df_15m: 15분봉 OHLCV DataFrame

    Returns:
        {'ready': True/False, 'reason': str, 'entry_price': float}
    """
    if df_15m is None or len(df_15m) < 5:
        return {'ready': False, 'reason': 'data_insufficient'}

    # 최근 10개 캔들 확인
    recent = df_15m.tail(10)

    # 양봉/음봉 체크
    recent_candles = []
    for i in range(len(recent)):
        row = recent.iloc[i]
        is_green = row['close'] > row['open']
        recent_candles.append({
            'idx': i,
            'is_green': is_green,
            'close': float(row['close']),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
        })

    # 양봉 후 음봉 패턴 찾기 (뒤에서부터)
    last_candle = recent_candles[-1]
    prev_candle = recent_candles[-2] if len(recent_candles) >= 2 else None

    # 현재 캔들이 음봉이고 이전 캔들이 양봉이면 진입
    if prev_candle and prev_candle['is_green'] and not last_candle['is_green']:
        return {
            'ready': True,
            'reason': 'green_then_red',
            'entry_price': last_candle['close'],
            'pattern': f"양봉({prev_candle['close']:.4f}) → 음봉({last_candle['close']:.4f})"
        }

    # 현재 양봉 진행 중 - 다음 음봉 대기
    if last_candle['is_green']:
        return {
            'ready': False,
            'reason': 'waiting_red_candle',
            'current_green': last_candle['close']
        }

    # 아직 양봉 안 나옴 - 양봉 대기
    # 최근 5개 캔들에서 양봉 있는지 확인
    has_recent_green = any(c['is_green'] for c in recent_candles[-5:])
    if not has_recent_green:
        return {
            'ready': False,
            'reason': 'waiting_green_candle'
        }

    # 양봉은 있었지만 연속 음봉 중
    return {
        'ready': False,
        'reason': 'consecutive_red',
    }


def get_surge_entry_signal_mtf(
    symbol: str,
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    params: dict = SURGE_STRATEGY_PARAMS
) -> Optional[dict]:
    """
    멀티 타임프레임 진입 신호 (1시간 + 15분)

    1. 1시간봉에서 기본 조건 확인
    2. 15분봉에서 양봉 후 음봉 확인

    Args:
        symbol: 심볼
        df_1h: 1시간봉 DataFrame (지표 포함)
        df_15m: 15분봉 DataFrame
        params: 전략 파라미터

    Returns:
        진입 신호 또는 None
    """
    # 1. 1시간봉 기본 신호 확인
    signal_1h = get_surge_entry_signal(symbol, df_1h, params)
    if signal_1h is None:
        return None

    # 2. 15분봉 타이밍 확인
    timing = check_15m_entry_timing(df_15m)

    if not timing['ready']:
        # 아직 진입 타이밍 아님
        return None

    # 진입 가격은 15분봉 기준
    entry_price = timing['entry_price']

    # 손절/익절 재계산
    stop_loss = entry_price * (1 - params['sl_pct'] / 100)
    take_profit = entry_price * (1 + params['tp_pct'] / 100)

    return {
        'symbol': symbol,
        'side': 'long',
        'price': entry_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'score': signal_1h['score'],
        'rsi': signal_1h['rsi'],
        'bb_position': signal_1h['bb_position'],
        'volume_ratio': signal_1h['volume_ratio'],
        'price_vs_sma25': signal_1h['price_vs_sma25'],
        'atr_pct': signal_1h['atr_pct'],
        'entry_reason': timing['reason'],
        'pattern': timing.get('pattern', ''),
    }
