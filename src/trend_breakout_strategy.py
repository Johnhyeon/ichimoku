"""
트렌드 급등 전략 (롱 전용)

급등 후 트렌드로 전환되어 지속 상승하는 코인을 초반에 포착
- 단순 펌프덤프가 아닌 진정한 트렌드 전환 감지
- 이평선 정배열 + 건강한 조정 + 거래량 지속

패턴:
1. 급등 발생 (가격 상승 + 거래량 증가)
2. 얕은 조정 (되돌림 < 50%)
3. 트렌드 지속 (이평선 정배열 유지)
"""

import logging
from datetime import datetime
from typing import Optional, Dict, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# 전략 파라미터
TREND_BREAKOUT_PARAMS = {
    # === 급등 감지 ===
    "surge_lookback_days": 7,        # 급등 감지 기간 (일)
    "min_surge_pct": 15,             # 최소 상승률 %
    "min_volume_surge": 1.5,         # 최소 거래량 증가 배수

    # === 트렌드 확인 ===
    "ema_fast": 9,                   # 빠른 EMA
    "ema_mid": 21,                   # 중간 EMA
    "ema_slow": 50,                  # 느린 EMA
    "ema_check_mode": "fast_only",   # "full": 9>21>50, "fast_only": 9>21만 체크
    "require_ema_alignment": True,   # 이평선 정배열 필수

    # === 조정 필터 ===
    "max_pullback_pct": 50,          # 최대 되돌림 (급등폭 대비 %)
    "min_pullback_pct": 10,          # 최소 조정 (너무 과열 제외)

    # === 모멘텀 필터 ===
    "rsi_min": 40,                   # RSI 하한 (상승 모멘텀) - 완화
    "rsi_max": 80,                   # RSI 상한 (과매수 제외) - 완화
    "obv_rising": False,             # OBV 상승 필수 - 비활성화

    # === 거래량 지속성 (펌프덤프 필터) ===
    "volume_consistency_days": 3,    # 거래량 지속 확인 일수
    "min_consistent_volume": 1.2,    # 최소 지속 거래량 배수

    # === 리스크 관리 ===
    "sl_pct": 5.0,                   # 손절 %
    "tp_pct": 15.0,                  # 1차 익절 %
    "trail_start_pct": 10.0,         # 트레일링 시작 %
    "trail_pct": 4.0,                # 트레일링 스탑 %

    # === 포지션 ===
    "leverage": 5,
    "position_pct": 0.05,            # 자산의 5%
    "max_positions": 3,

    # === 필터 ===
    "min_market_cap_rank": 200,      # 시총 순위 (상위 N개만)
    "min_volume_24h": 5_000_000,     # 최소 24시간 거래량
    "min_atr_pct": 3.0,              # 최소 변동성
    "max_atr_pct": 20.0,             # 최대 변동성
    "cooldown_hours": 24,            # 재진입 쿨다운
}


def calculate_trend_indicators(df: pd.DataFrame, params: dict = TREND_BREAKOUT_PARAMS) -> pd.DataFrame:
    """
    트렌드 급등 전략용 지표 계산

    Args:
        df: OHLCV DataFrame
        params: 전략 파라미터

    Returns:
        지표가 추가된 DataFrame
    """
    df = df.copy()

    # === 이동평균선 ===
    df['ema_fast'] = df['close'].ewm(span=params['ema_fast']).mean()
    df['ema_mid'] = df['close'].ewm(span=params['ema_mid']).mean()
    df['ema_slow'] = df['close'].ewm(span=params['ema_slow']).mean()

    # SMA도 추가
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()

    # === 이평선 정배열 확인 ===
    # 전체 정배열: EMA9 > EMA21 > EMA50
    df['ema_aligned'] = (
        (df['ema_fast'] > df['ema_mid']) &
        (df['ema_mid'] > df['ema_slow'])
    )

    # 빠른 정배열: EMA9 > EMA21만 (급등 초반용)
    df['ema_fast_aligned'] = df['ema_fast'] > df['ema_mid']

    # 가격이 이평선 위
    df['price_above_emas'] = (
        (df['close'] > df['ema_fast']) &
        (df['close'] > df['ema_mid'])
    )

    # === RSI ===
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # === MACD ===
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_bullish'] = df['macd'] > df['macd_signal']

    # === 거래량 분석 ===
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    # 거래량 지속성 (최근 N일 평균)
    consistency_days = params.get('volume_consistency_days', 3)
    df['volume_recent_avg'] = df['volume'].rolling(consistency_days).mean()
    df['volume_consistent'] = df['volume_recent_avg'] / df['volume_sma']

    # === OBV (On Balance Volume) ===
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_sma'] = df['obv'].rolling(20).mean()
    df['obv_rising'] = df['obv'] > df['obv_sma']

    # === ATR (변동성) ===
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['close'] * 100

    # === 최근 고점/저점 ===
    lookback = params.get('surge_lookback_days', 7)
    # 4시간봉 기준으로 변환 (하루 6개 캔들)
    candles_lookback = lookback * 6

    df['recent_high'] = df['high'].rolling(candles_lookback).max()
    df['recent_low'] = df['low'].rolling(candles_lookback).min()

    # 급등폭 계산
    df['surge_pct'] = (df['recent_high'] - df['recent_low']) / df['recent_low'] * 100

    # 현재가 대비 고점에서 얼마나 내려왔는지 (조정폭)
    df['pullback_from_high'] = (df['recent_high'] - df['close']) / df['recent_high'] * 100

    # 조정폭 / 급등폭 비율
    df['pullback_ratio'] = df['pullback_from_high'] / df['surge_pct'] * 100

    # === 캔들 패턴 ===
    df['is_green'] = df['close'] > df['open']
    df['candle_body_pct'] = abs(df['close'] - df['open']) / df['open'] * 100

    # 양봉 연속 개수
    df['consecutive_green'] = df['is_green'].rolling(3).sum()

    # === 돌파 감지 ===
    df['breakout_high'] = df['close'] > df['recent_high'].shift(1)

    # === 가격 위치 (0~1) ===
    df['price_position'] = (df['close'] - df['recent_low']) / (df['recent_high'] - df['recent_low'])

    return df


def detect_surge_pattern(df: pd.DataFrame, params: dict = TREND_BREAKOUT_PARAMS) -> Dict:
    """
    급등 패턴 감지

    Returns:
        {
            'has_surge': bool,
            'surge_pct': float,
            'pullback_ratio': float,
            'volume_surge': float,
            'is_healthy': bool  # 건강한 조정인지
        }
    """
    if df is None or len(df) < 50:
        return {'has_surge': False}

    row = df.iloc[-1]

    surge_pct = float(row['surge_pct']) if pd.notna(row['surge_pct']) else 0
    pullback_ratio = float(row['pullback_ratio']) if pd.notna(row['pullback_ratio']) else 100
    volume_consistent = float(row['volume_consistent']) if pd.notna(row['volume_consistent']) else 0

    # 급등 발생 여부
    has_surge = surge_pct >= params['min_surge_pct']

    # 건강한 조정인지 (너무 많이 빠지지 않고, 어느 정도는 조정)
    min_pb = params.get('min_pullback_pct', 10)
    max_pb = params['max_pullback_pct']
    is_healthy = min_pb <= pullback_ratio <= max_pb

    # 거래량 지속
    volume_sustained = volume_consistent >= params['min_consistent_volume']

    return {
        'has_surge': has_surge,
        'surge_pct': surge_pct,
        'pullback_ratio': pullback_ratio,
        'volume_surge': volume_consistent,
        'is_healthy': is_healthy and volume_sustained
    }


def get_trend_breakout_entry_signal(
    symbol: str,
    df: pd.DataFrame,
    params: dict = TREND_BREAKOUT_PARAMS
) -> Optional[dict]:
    """
    트렌드 급등 진입 신호 계산

    진입 조건:
    1. 급등 발생 (최근 N일간 X% 상승)
    2. 이평선 정배열 (EMA9 > EMA21 > EMA50)
    3. 건강한 조정 (되돌림 < 50%)
    4. 거래량 지속 (펌프덤프 필터)
    5. RSI 적정 범위 (45~75)
    6. OBV 상승 추세

    Args:
        symbol: 심볼
        df: OHLCV + 지표 DataFrame
        params: 전략 파라미터

    Returns:
        진입 신호 딕셔너리 또는 None
    """
    if df is None or len(df) < 60:
        return None

    row = df.iloc[-1]
    prev = df.iloc[-2]

    # 기본값 추출
    price = float(row['close'])
    rsi = float(row['rsi']) if pd.notna(row['rsi']) else 50
    atr_pct = float(row['atr_pct']) if pd.notna(row['atr_pct']) else 5
    volume_ratio = float(row['volume_ratio']) if pd.notna(row['volume_ratio']) else 1

    # === 필터 조건 ===

    # 1. 변동성 필터
    if atr_pct < params['min_atr_pct'] or atr_pct > params['max_atr_pct']:
        return None

    # 2. RSI 필터
    if not (params['rsi_min'] <= rsi <= params['rsi_max']):
        return None

    # 3. 급등 패턴 감지
    surge = detect_surge_pattern(df, params)
    if not surge['has_surge']:
        return None

    # 4. 건강한 조정 확인
    if not surge['is_healthy']:
        return None

    # 5. 이평선 정배열 확인
    if params['require_ema_alignment']:
        ema_mode = params.get('ema_check_mode', 'full')

        if ema_mode == 'fast_only':
            # 빠른 정배열만: EMA9 > EMA21 (급등 초반 진입용)
            ema_ok = bool(row['ema_fast_aligned']) if pd.notna(row.get('ema_fast_aligned')) else False
        else:
            # 전체 정배열: EMA9 > EMA21 > EMA50
            ema_ok = bool(row['ema_aligned']) if pd.notna(row['ema_aligned']) else False

        price_above = bool(row['price_above_emas']) if pd.notna(row['price_above_emas']) else False
        if not (ema_ok and price_above):
            return None

    # 6. OBV 상승 확인
    if params.get('obv_rising', True):
        obv_rising = bool(row['obv_rising']) if pd.notna(row['obv_rising']) else False
        if not obv_rising:
            return None

    # 7. MACD 확인 (선택적)
    macd_bullish = bool(row['macd_bullish']) if pd.notna(row['macd_bullish']) else True

    # 8. 반등 확인 (현재 양봉 또는 연속 양봉)
    # 급등폭이 크면 (50%+) 양봉 조건 면제 (조정 구간 진입 허용)
    is_green = bool(row['is_green'])
    consecutive_green = int(row['consecutive_green']) if pd.notna(row['consecutive_green']) else 0

    surge_pct = detect_surge_pattern(df, params)['surge_pct']
    skip_green_check = surge_pct >= 50  # 급등 50% 이상이면 양봉 체크 스킵

    if not skip_green_check:
        if not is_green and consecutive_green < 2:
            return None

    # === 점수 계산 ===
    score = 0

    # 급등폭 점수
    surge_pct = surge['surge_pct']
    if surge_pct >= 30:
        score += 4
    elif surge_pct >= 20:
        score += 2
    else:
        score += 1

    # 조정 비율 점수 (적정 조정이 베스트)
    pullback_ratio = surge['pullback_ratio']
    if 20 <= pullback_ratio <= 40:
        score += 3  # 최적의 조정
    elif 15 <= pullback_ratio <= 50:
        score += 2
    else:
        score += 1

    # 거래량 지속성 점수
    vol_surge = surge['volume_surge']
    if vol_surge >= 2.0:
        score += 3
    elif vol_surge >= 1.5:
        score += 2
    else:
        score += 1

    # 이평선 점수
    if bool(row.get('ema_aligned', False)):
        score += 2

    # MACD 점수
    if macd_bullish:
        score += 1

    # RSI 점수 (50~60이 최적)
    if 50 <= rsi <= 60:
        score += 2
    elif 45 <= rsi <= 65:
        score += 1

    # 연속 양봉 보너스
    if consecutive_green >= 2:
        score += 1

    # === 손절/익절 계산 ===
    # 손절: 최근 저점 또는 EMA slow 아래
    ema_slow = float(row['ema_slow']) if pd.notna(row['ema_slow']) else price * 0.95
    recent_low = float(row['recent_low']) if pd.notna(row['recent_low']) else price * 0.95

    # 둘 중 더 높은 값 (손실 제한)
    sl_level = max(ema_slow, recent_low * 0.98)
    sl_distance = (price - sl_level) / price * 100

    # 손절이 너무 멀거나 가까우면 기본값 사용
    if sl_distance < 2 or sl_distance > params['sl_pct'] * 1.5:
        stop_loss = price * (1 - params['sl_pct'] / 100)
    else:
        stop_loss = sl_level

    take_profit = price * (1 + params['tp_pct'] / 100)

    return {
        'symbol': symbol,
        'side': 'long',
        'price': price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'score': score,
        'surge_pct': surge_pct,
        'pullback_ratio': pullback_ratio,
        'volume_surge': vol_surge,
        'rsi': rsi,
        'ema_aligned': bool(row.get('ema_aligned', False)),
        'atr_pct': atr_pct,
    }


def check_trend_exit_signal(
    pos: dict,
    row: pd.Series,
    params: dict = TREND_BREAKOUT_PARAMS
) -> Optional[dict]:
    """
    트렌드 급등 포지션 청산 신호

    청산 조건:
    1. 손절: SL 도달
    2. 트레일링 스탑: 수익 보호
    3. 이평선 역전: 트렌드 종료
    4. RSI 과매수 후 하락: 모멘텀 약화

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

    # 4. 이평선 역전 (트렌드 종료 신호)
    ema_aligned = bool(row['ema_aligned']) if pd.notna(row.get('ema_aligned')) else True
    if not ema_aligned and pnl_pct > 0:
        # 수익 중일 때만 트렌드 종료로 청산
        return {'action': 'close', 'reason': 'EMA_Reversal', 'price': price}

    # 5. 가격이 EMA slow 아래로 이탈
    ema_slow = float(row['ema_slow']) if pd.notna(row.get('ema_slow')) else 0
    if ema_slow > 0 and price < ema_slow:
        return {'action': 'close', 'reason': 'Below_EMA50', 'price': price}

    # 6. RSI 과매수 후 하락 반전
    rsi = float(row['rsi']) if pd.notna(row.get('rsi')) else 50
    if rsi < 40 and pnl_pct < 5:  # RSI 하락 + 수익 적으면 청산
        return {'action': 'close', 'reason': 'RSI_Weakness', 'price': price}

    # 7. OBV 하락 반전 (거래량 약화)
    obv_rising = bool(row['obv_rising']) if pd.notna(row.get('obv_rising')) else True
    if not obv_rising and pnl_pct > 8:
        # 수익 중이지만 거래량 약화 - 일부 청산 고려
        # 여기서는 신호만 반환, 실제 일부 청산은 trader에서 처리
        pass

    return None


def scan_trend_breakout_candidates(
    symbols: List[str],
    data_fetcher,
    timeframe: str = '4h',
    params: dict = TREND_BREAKOUT_PARAMS
) -> List[dict]:
    """
    트렌드 급등 후보 코인 스캔

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
            if df is None or len(df) < 60:
                continue

            # 지표 계산
            df = calculate_trend_indicators(df, params)

            # 진입 신호 체크
            signal = get_trend_breakout_entry_signal(symbol, df, params)
            if signal:
                candidates.append(signal)
                logger.info(
                    f"트렌드 급등 후보: {symbol} | "
                    f"Score={signal['score']} "
                    f"Surge={signal['surge_pct']:.1f}% "
                    f"Pullback={signal['pullback_ratio']:.1f}% "
                    f"Vol={signal['volume_surge']:.1f}x"
                )

        except Exception as e:
            logger.debug(f"스캔 실패 ({symbol}): {e}")
            continue

    # 점수순 정렬
    candidates.sort(key=lambda x: x['score'], reverse=True)

    return candidates


def get_higher_timeframe_confirmation(
    symbol: str,
    data_fetcher,
    params: dict = TREND_BREAKOUT_PARAMS
) -> Dict:
    """
    상위 타임프레임 확인 (일봉)

    펌프덤프 필터 강화를 위해 일봉에서도 상승 구조 확인

    Returns:
        {'confirmed': bool, 'trend': str, 'details': dict}
    """
    try:
        df_daily = data_fetcher.get_ohlcv(symbol, '1d', limit=30)
        if df_daily is None or len(df_daily) < 20:
            return {'confirmed': False, 'trend': 'unknown'}

        df_daily = calculate_trend_indicators(df_daily, params)

        row = df_daily.iloc[-1]

        ema_aligned = bool(row['ema_aligned']) if pd.notna(row['ema_aligned']) else False
        obv_rising = bool(row['obv_rising']) if pd.notna(row['obv_rising']) else False
        rsi = float(row['rsi']) if pd.notna(row['rsi']) else 50

        # 상위 타임프레임에서 상승 구조면 확정
        confirmed = ema_aligned and obv_rising and rsi > 45

        trend = 'up' if confirmed else ('down' if rsi < 40 else 'neutral')

        return {
            'confirmed': confirmed,
            'trend': trend,
            'ema_aligned': ema_aligned,
            'obv_rising': obv_rising,
            'rsi': rsi
        }

    except Exception as e:
        logger.debug(f"상위 타임프레임 확인 실패 ({symbol}): {e}")
        return {'confirmed': False, 'trend': 'unknown'}


# 스캔 대상 종목 (동적 로드)
def get_trend_watch_list(exclude_large_caps: bool = False) -> List[str]:
    """
    트렌드 급등 스캔 대상 종목 조회

    급등 후 트렌드가 될 가능성이 높은 중소형 알트코인 위주
    """
    from src.surge_strategy import get_all_usdt_perpetuals, EXCLUDE_LARGE_CAPS

    symbols = get_all_usdt_perpetuals()

    if exclude_large_caps:
        symbols = [s for s in symbols if s not in EXCLUDE_LARGE_CAPS]

    return symbols


# 인기 있는 급등 코인들 (폴백용)
TREND_WATCH_LIST_FALLBACK = [
    # 밈코인 (급등 빈번)
    "WIF/USDT:USDT", "PEPE/USDT:USDT", "DOGE/USDT:USDT", "SHIB/USDT:USDT",
    "FLOKI/USDT:USDT", "BONK/USDT:USDT",

    # AI 관련
    "FET/USDT:USDT", "AGIX/USDT:USDT", "RNDR/USDT:USDT", "TAO/USDT:USDT",

    # 레이어2/신규
    "ARB/USDT:USDT", "OP/USDT:USDT", "STRK/USDT:USDT", "ZETA/USDT:USDT",

    # 솔라나 생태계
    "SOL/USDT:USDT", "JTO/USDT:USDT", "JUP/USDT:USDT", "PYTH/USDT:USDT",

    # 기타 급등 가능성
    "SUI/USDT:USDT", "SEI/USDT:USDT", "INJ/USDT:USDT", "TIA/USDT:USDT",
    "ORDI/USDT:USDT", "STX/USDT:USDT", "NEAR/USDT:USDT", "APT/USDT:USDT",
]
