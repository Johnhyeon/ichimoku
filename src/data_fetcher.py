"""OHLCV 데이터 수집"""

import logging
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)


class DataFetcher:
    """캔들 데이터 수집기"""

    def __init__(self, client):
        self.client = client

    def get_ohlcv(self, symbol: str, timeframe: str = "4h", limit: int = 200) -> Optional[pd.DataFrame]:
        """
        OHLCV 캔들 데이터 조회

        Args:
            symbol: 거래쌍 (예: "BTC/USDT:USDT")
            timeframe: 타임프레임 (예: "4h", "1h", "15m")
            limit: 캔들 개수

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.client.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            if not ohlcv:
                return None

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')

            return df

        except Exception as e:
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
        else:
            next_candle = now + timedelta(hours=4)

        return next_candle
