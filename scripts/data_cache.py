"""
차트 데이터 캐싱 시스템

데이터를 로컬에 저장해두고 백테스트 시 빠르게 로드
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
from pybit.unified_trading import HTTP

logger = logging.getLogger(__name__)

# 데이터 저장 경로
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'cache')

# 코인 목록
MAJOR_COINS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT',
    'ADAUSDT', 'DOGEUSDT', 'TONUSDT', 'TRXUSDT', 'AVAXUSDT',
    'DOTUSDT', 'LINKUSDT', 'BCHUSDT', 'SUIUSDT', 'NEARUSDT',
    'LTCUSDT', 'UNIUSDT', 'APTUSDT', 'ICPUSDT', 'ETCUSDT',
    'RENDERUSDT', 'STXUSDT', 'HBARUSDT', 'XMRUSDT', 'ATOMUSDT',
    'IMXUSDT', 'FILUSDT', 'INJUSDT', 'XLMUSDT', 'ARBUSDT',
    'OPUSDT', 'VETUSDT', 'KASUSDT', 'TIAUSDT', 'POLUSDT',
    'SEIUSDT', 'RUNEUSDT', 'WIFUSDT', 'JUPUSDT', 'AAVEUSDT',
]

# 타임프레임 설정
TIMEFRAMES = {
    '5m': 5,
    '15m': 15,
    '1h': 60,
    '4h': 240,
    '1d': 'D',
}


def get_cache_path(symbol: str, interval: str) -> str:
    """캐시 파일 경로"""
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f'{symbol}_{interval}.csv')


def fetch_klines_from_api(symbol: str, interval: int, limit: int = 10000) -> Optional[pd.DataFrame]:
    """API에서 캔들 데이터 가져오기"""
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
        time.sleep(0.03)

    if not all_data:
        return None

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    return df


def save_to_cache(df: pd.DataFrame, symbol: str, interval: str):
    """데이터를 캐시에 저장 (CSV)"""
    path = get_cache_path(symbol, interval)
    df.to_csv(path, index=False)
    print(f"  저장: {symbol} {interval} ({len(df)}개 캔들)")


def load_from_cache(symbol: str, interval: str) -> Optional[pd.DataFrame]:
    """캐시에서 데이터 로드 (CSV)"""
    path = get_cache_path(symbol, interval)
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=['timestamp'])
        return df
    return None


def get_cache_info(symbol: str, interval: str) -> dict:
    """캐시 정보 조회"""
    path = get_cache_path(symbol, interval)
    if not os.path.exists(path):
        return {'exists': False}

    df = pd.read_csv(path, parse_dates=['timestamp'])
    return {
        'exists': True,
        'candles': len(df),
        'start': df['timestamp'].min(),
        'end': df['timestamp'].max(),
        'size_mb': os.path.getsize(path) / 1024 / 1024
    }


def download_all_data(
    symbols: List[str] = MAJOR_COINS,
    intervals: List[str] = ['15m', '1h', '4h'],
    limit: int = 20000
):
    """모든 코인의 데이터 다운로드"""
    print("=" * 60)
    print("차트 데이터 다운로드")
    print("=" * 60)
    print(f"코인: {len(symbols)}개")
    print(f"타임프레임: {intervals}")
    print(f"캔들 수: {limit}개")
    print("=" * 60)

    for interval in intervals:
        print(f"\n[{interval}] 다운로드 중...")
        interval_val = TIMEFRAMES.get(interval, int(interval.replace('m', '').replace('h', '')))

        for i, symbol in enumerate(symbols):
            print(f"  {i+1}/{len(symbols)} {symbol}...", end='', flush=True)

            # 이미 캐시가 있으면 스킵 (강제 다운로드 옵션 추가 가능)
            cached = load_from_cache(symbol, interval)
            if cached is not None and len(cached) >= limit * 0.9:
                print(f"캐시 있음 ({len(cached)}개)")
                continue

            df = fetch_klines_from_api(symbol, interval_val, limit)
            if df is not None and not df.empty:
                save_to_cache(df, symbol, interval)
            else:
                print("SKIP")

    print("\n" + "=" * 60)
    print("다운로드 완료!")
    print_cache_summary(symbols, intervals)


def update_cache(
    symbols: List[str] = MAJOR_COINS,
    intervals: List[str] = ['15m', '1h', '4h']
):
    """캐시 데이터 최신화 (새 캔들만 추가)"""
    print("캐시 업데이트 중...")

    for interval in intervals:
        interval_val = TIMEFRAMES.get(interval, int(interval.replace('m', '').replace('h', '')))

        for symbol in symbols:
            cached = load_from_cache(symbol, interval)
            if cached is None:
                continue

            # 최신 500개만 가져와서 병합
            new_df = fetch_klines_from_api(symbol, interval_val, 500)
            if new_df is not None:
                combined = pd.concat([cached, new_df]).drop_duplicates('timestamp')
                combined = combined.sort_values('timestamp').reset_index(drop=True)
                save_to_cache(combined, symbol, interval)


def load_cached_data(
    symbols: List[str] = MAJOR_COINS,
    interval: str = '15m'
) -> Dict[str, pd.DataFrame]:
    """캐시된 데이터 로드 (백테스트용)"""
    data = {}
    for symbol in symbols:
        df = load_from_cache(symbol, interval)
        if df is not None:
            data[symbol] = df
    return data


def print_cache_summary(
    symbols: List[str] = MAJOR_COINS,
    intervals: List[str] = ['15m', '1h', '4h']
):
    """캐시 상태 요약"""
    print("\n[캐시 상태]")
    print("-" * 70)

    total_size = 0
    for interval in intervals:
        count = 0
        candles = 0
        for symbol in symbols:
            info = get_cache_info(symbol, interval)
            if info['exists']:
                count += 1
                candles += info['candles']
                total_size += info.get('size_mb', 0)

        if count > 0:
            print(f"  {interval}: {count}개 코인, 총 {candles:,}개 캔들")

    print(f"  총 용량: {total_size:.1f} MB")
    print("-" * 70)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'download':
            download_all_data()
        elif cmd == 'update':
            update_cache()
        elif cmd == 'status':
            print_cache_summary()
    else:
        print("사용법:")
        print("  python data_cache.py download  # 전체 다운로드")
        print("  python data_cache.py update    # 최신 데이터 추가")
        print("  python data_cache.py status    # 캐시 상태 확인")
