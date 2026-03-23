"""OHLCV 데이터 수집"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)


class DataFetcher:
    """캔들 데이터 수집기 (레이트 리밋 내장)"""

    # Bybit API Rate Limit: 초당 10회
    _MIN_INTERVAL = 0.12  # 120ms 간격 = 초당 ~8회 (여유분 확보)
    _lock = threading.Lock()
    _last_call_time = 0.0

    def __init__(self, client):
        self.client = client

    def _rate_limit(self):
        """API 호출 간 최소 간격 보장."""
        with DataFetcher._lock:
            now = time.monotonic()
            elapsed = now - DataFetcher._last_call_time
            if elapsed < DataFetcher._MIN_INTERVAL:
                time.sleep(DataFetcher._MIN_INTERVAL - elapsed)
            DataFetcher._last_call_time = time.monotonic()

    def get_ohlcv(self, symbol: str, timeframe: str = "4h", limit: int = 200) -> Optional[pd.DataFrame]:
        """
        OHLCV 캔들 데이터 조회 (레이트 리밋 적용)

        Args:
            symbol: 거래쌍 (예: "BTC/USDT:USDT")
            timeframe: 타임프레임 (예: "4h", "1h", "15m")
            limit: 캔들 개수

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        self._rate_limit()

        try:
            ohlcv = self.client.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            if not ohlcv:
                return None

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')

            return df

        except Exception as e:
            # Rate limit 에러 시 1초 대기 후 1회 재시도
            if "10006" in str(e) or "Rate Limit" in str(e):
                logger.warning(f"Rate limit hit ({symbol}), 1s 대기 후 재시도")
                time.sleep(1)
                self._rate_limit()
                try:
                    ohlcv = self.client.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                    if ohlcv:
                        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df = df.set_index('timestamp')
                        return df
                except Exception:
                    pass
            logger.error(f"OHLCV 조회 실패 ({symbol}): {e}")
            return None

    def get_next_candle_time(self, timeframe: str = "4h") -> datetime:
        """
        다음 캔들 시작 시간 계산

        Args:
            timeframe: 타임프레임

        Returns:
            다음 캔들 시작 시간 (datetime)
        """
        now = datetime.utcnow()

        if timeframe == "4h":
            hours = (now.hour // 4 + 1) * 4
            if hours >= 24:
                next_candle = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            else:
                next_candle = now.replace(hour=hours, minute=0, second=0, microsecond=0)
        elif timeframe == "1h":
            next_candle = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        elif timeframe == "15m":
            minutes = (now.minute // 15 + 1) * 15
            if minutes >= 60:
                next_candle = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            else:
                next_candle = now.replace(minute=minutes, second=0, microsecond=0)
        elif timeframe == "1d":
            next_candle = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        else:
            next_candle = now + timedelta(hours=4)

        return next_candle
