"""
과거 데이터 로더

collect_historical_data.py로 수집한 데이터를 빠르게 로드합니다.
다양한 전략 개발/백테스트에 재사용 가능합니다.

사용 예시:
    # 단일 코인, 단일 타임프레임
    loader = DataLoader()
    df = loader.load("BTC/USDT:USDT", "5m")

    # 여러 타임프레임 동시 로드
    data = loader.load_multiple("ETH/USDT:USDT", ["5m", "1h", "4h"])

    # 날짜 범위 필터
    df = loader.load("BTC/USDT:USDT", "1h", start="2024-01-01", end="2024-12-31")

    # 전체 코인 목록
    symbols = loader.get_available_symbols()

    # 메모리 절약 (청크 단위 로드)
    for chunk in loader.load_chunks("BTC/USDT:USDT", "1m", chunk_size=10000):
        process(chunk)
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Iterator
from pathlib import Path
import pandas as pd
import json

logger = logging.getLogger(__name__)


DATA_DIR = Path("data/historical")
METADATA_FILE = DATA_DIR / "metadata.json"


class DataLoader:
    """과거 데이터 로더"""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.metadata = self._load_metadata()
        self._cache = {}  # 메모리 캐시 (선택적)

    def _load_metadata(self) -> Dict:
        """메타데이터 로드"""
        if METADATA_FILE.exists():
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
        return {}

    def _clean_symbol(self, symbol: str) -> str:
        """심볼 정규화"""
        return symbol.replace('/', '').replace(':', '')

    def get_available_symbols(self) -> List[str]:
        """사용 가능한 심볼 목록"""
        return list(self.metadata.keys())

    def get_available_timeframes(self, symbol: str) -> List[str]:
        """특정 심볼의 사용 가능한 타임프레임"""
        if symbol not in self.metadata:
            return []
        return list(self.metadata[symbol].keys())

    def get_info(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """데이터 정보 조회"""
        if symbol not in self.metadata:
            return None
        if timeframe not in self.metadata[symbol]:
            return None
        return self.metadata[symbol][timeframe]

    def load(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_cache: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        데이터 로드

        Args:
            symbol: 심볼 (예: "BTC/USDT:USDT")
            timeframe: 타임프레임 (예: "5m")
            start: 시작 날짜 (예: "2024-01-01")
            end: 종료 날짜 (예: "2024-12-31")
            use_cache: 메모리 캐시 사용 여부

        Returns:
            DataFrame (timestamp, open, high, low, close, volume)
        """
        cache_key = f"{symbol}_{timeframe}"

        # 캐시 확인
        if use_cache and cache_key in self._cache:
            df = self._cache[cache_key]
        else:
            # 파일 로드
            symbol_clean = self._clean_symbol(symbol)
            file_path = self.data_dir / symbol_clean / f"{timeframe}.parquet"

            if not file_path.exists():
                logger.warning(f"데이터 파일 없음: {file_path}")
                return None

            df = pd.read_parquet(file_path)

            if use_cache:
                self._cache[cache_key] = df

        # 날짜 필터
        if start or end:
            df = df.copy()
            if start:
                df = df[df['timestamp'] >= pd.to_datetime(start)]
            if end:
                df = df[df['timestamp'] <= pd.to_datetime(end)]

        return df

    def load_multiple(
        self,
        symbol: str,
        timeframes: List[str],
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        여러 타임프레임 동시 로드

        Args:
            symbol: 심볼
            timeframes: 타임프레임 리스트
            start: 시작 날짜
            end: 종료 날짜

        Returns:
            {타임프레임: DataFrame} 딕셔너리
        """
        result = {}
        for tf in timeframes:
            df = self.load(symbol, tf, start, end)
            if df is not None:
                result[tf] = df
        return result

    def load_chunks(
        self,
        symbol: str,
        timeframe: str,
        chunk_size: int = 10000
    ) -> Iterator[pd.DataFrame]:
        """
        메모리 절약을 위한 청크 단위 로드

        Args:
            symbol: 심볼
            timeframe: 타임프레임
            chunk_size: 청크 크기 (행 수)

        Yields:
            DataFrame 청크
        """
        symbol_clean = self._clean_symbol(symbol)
        file_path = self.data_dir / symbol_clean / f"{timeframe}.parquet"

        if not file_path.exists():
            return

        # Parquet는 기본적으로 전체 로드하므로, 로드 후 청크로 나눔
        df = pd.read_parquet(file_path)

        for i in range(0, len(df), chunk_size):
            yield df.iloc[i:i+chunk_size].copy()

    def get_stats(self, symbol: str = None) -> Dict:
        """
        데이터 통계

        Args:
            symbol: 특정 심볼만 (None이면 전체)

        Returns:
            통계 딕셔너리
        """
        if symbol:
            if symbol not in self.metadata:
                return {}

            stats = {
                'symbol': symbol,
                'timeframes': {},
            }

            for tf, info in self.metadata[symbol].items():
                stats['timeframes'][tf] = {
                    'rows': info['rows'],
                    'start': info['start'],
                    'end': info['end'],
                    'updated': info['updated_at'],
                }

            return stats

        else:
            # 전체 통계
            total_symbols = len(self.metadata)
            total_candles = 0
            timeframe_counts = {}

            for symbol_data in self.metadata.values():
                for tf, info in symbol_data.items():
                    total_candles += info['rows']
                    timeframe_counts[tf] = timeframe_counts.get(tf, 0) + 1

            return {
                'total_symbols': total_symbols,
                'total_candles': total_candles,
                'timeframes': timeframe_counts,
            }

    def clear_cache(self):
        """메모리 캐시 초기화"""
        self._cache.clear()


# 편의 함수들

def quick_load(symbol: str, timeframe: str = "5m", **kwargs) -> Optional[pd.DataFrame]:
    """빠른 데이터 로드"""
    loader = DataLoader()
    return loader.load(symbol, timeframe, **kwargs)


def get_symbols() -> List[str]:
    """사용 가능한 심볼 목록"""
    loader = DataLoader()
    return loader.get_available_symbols()


def compare_timeframes(symbol: str, timeframes: List[str], date: str) -> Dict:
    """
    특정 날짜의 여러 타임프레임 비교

    Args:
        symbol: 심볼
        timeframes: 타임프레임 리스트
        date: 날짜 (예: "2024-01-01")

    Returns:
        {타임프레임: 해당 날짜 데이터} 딕셔너리
    """
    loader = DataLoader()
    result = {}

    for tf in timeframes:
        df = loader.load(symbol, tf, start=date, end=date)
        if df is not None and len(df) > 0:
            result[tf] = df.iloc[0].to_dict()

    return result


# 백테스트 전용 헬퍼

class BacktestDataProvider:
    """백테스트용 데이터 제공자 (기존 DataFetcher 인터페이스 호환)"""

    def __init__(self, use_cache: bool = True):
        self.loader = DataLoader()
        self.use_cache = use_cache

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "4h",
        limit: int = 200,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        DataFetcher.get_ohlcv와 호환되는 인터페이스

        Args:
            symbol: 심볼
            timeframe: 타임프레임
            limit: 최대 캔들 수
            end_date: 종료 날짜 (백테스트 시점)

        Returns:
            DataFrame (index=timestamp)
        """
        df = self.loader.load(symbol, timeframe, end=end_date, use_cache=self.use_cache)

        if df is None:
            return None

        # 최신 N개만
        if len(df) > limit:
            df = df.tail(limit)

        # index를 timestamp로 설정 (기존 DataFetcher와 호환)
        df = df.set_index('timestamp')

        return df

    def has_data(self, symbol: str, timeframe: str) -> bool:
        """데이터 존재 여부 확인"""
        info = self.loader.get_info(symbol, timeframe)
        return info is not None


# 사용 예시
if __name__ == '__main__':
    # 기본 사용
    loader = DataLoader()

    # 사용 가능한 심볼 확인
    symbols = loader.get_available_symbols()
    print(f"사용 가능한 심볼: {len(symbols)}개")

    if symbols:
        symbol = symbols[0]
        print(f"\n예시: {symbol}")

        # 타임프레임 확인
        timeframes = loader.get_available_timeframes(symbol)
        print(f"타임프레임: {', '.join(timeframes)}")

        # 데이터 로드
        if '5m' in timeframes:
            df = loader.load(symbol, '5m')
            if df is not None:
                print(f"\n5분봉 데이터:")
                print(f"  행 수: {len(df):,}")
                print(f"  기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
                print(f"  컬럼: {list(df.columns)}")

        # 2024년 데이터만
        if '1h' in timeframes:
            df_2024 = loader.load(symbol, '1h', start='2024-01-01', end='2024-12-31')
            if df_2024 is not None:
                print(f"\n2024년 1시간봉:")
                print(f"  행 수: {len(df_2024):,}")

    # 전체 통계
    print(f"\n전체 통계:")
    stats = loader.get_stats()
    print(f"  심볼: {stats['total_symbols']}개")
    print(f"  총 캔들: {stats['total_candles']:,}개")
