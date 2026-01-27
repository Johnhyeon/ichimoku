"""OHLCV 데이터 수집 모듈"""

import pandas as pd
import logging
from typing import Optional
from datetime import datetime, timedelta
from exchange.bybit_client import BybitClient

logger = logging.getLogger(__name__)


class DataFetcher:
    """OHLCV 데이터 수집"""

    # 타임프레임 → 분 변환
    TIMEFRAME_MINUTES = {
        '1m': 1,
        '3m': 3,
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '2h': 120,
        '4h': 240,
        '6h': 360,
        '12h': 720,
        '1d': 1440,
        '1w': 10080,
    }

    def __init__(self, client: BybitClient):
        """
        데이터 수집기 초기화

        Args:
            client: BybitClient 인스턴스
        """
        self.client = client
        self.exchange = client.exchange

    def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """
        캔들 데이터 조회

        Args:
            symbol: "BTC/USDT"
            timeframe: "1m", "5m", "15m", "1h", "4h", "1d"
            limit: 캔들 개수 (최대 200)

        Returns:
            DataFrame with columns: [timestamp, open, high, low, close, volume]
        """
        try:
            # CCXT로 OHLCV 데이터 조회
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit
            )

            # DataFrame으로 변환
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            # 타임스탬프를 datetime으로 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # 숫자형으로 변환
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            # 인덱스를 timestamp로 설정
            df.set_index('timestamp', inplace=True)

            logger.debug(f"OHLCV 데이터 조회 완료: {symbol} {timeframe} ({len(df)}개)")
            return df

        except Exception as e:
            logger.error(f"OHLCV 데이터 조회 실패: {e}")
            raise

    def get_ohlcv_extended(self, symbol: str, timeframe: str,
                          start_date: datetime, end_date: datetime = None) -> pd.DataFrame:
        """
        확장된 기간의 OHLCV 데이터 조회 (백테스트용)

        Args:
            symbol: 거래쌍
            timeframe: 타임프레임
            start_date: 시작 날짜
            end_date: 종료 날짜 (None이면 현재)

        Returns:
            DataFrame with OHLCV data
        """
        if end_date is None:
            end_date = datetime.now()

        all_data = []
        current_date = start_date

        # 타임프레임별 최대 조회 개수
        batch_size = 200
        minutes = self.TIMEFRAME_MINUTES.get(timeframe, 60)

        while current_date < end_date:
            since = int(current_date.timestamp() * 1000)

            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=batch_size
                )

                if not ohlcv:
                    break

                all_data.extend(ohlcv)

                # 다음 배치 시작점
                last_timestamp = ohlcv[-1][0]
                current_date = datetime.fromtimestamp(last_timestamp / 1000) + timedelta(minutes=minutes)

                logger.debug(f"데이터 수집 중: {current_date.strftime('%Y-%m-%d %H:%M')}")

            except Exception as e:
                logger.error(f"확장 OHLCV 조회 실패: {e}")
                break

        if not all_data:
            return pd.DataFrame()

        # DataFrame 생성
        df = pd.DataFrame(
            all_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        # 중복 제거
        df.drop_duplicates(subset=['timestamp'], inplace=True)

        # 타임스탬프 변환
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # end_date까지만 필터링
        df = df[df.index <= end_date]

        # 숫자형 변환
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        logger.info(f"확장 OHLCV 조회 완료: {len(df)}개 캔들")
        return df

    def get_current_price(self, symbol: str) -> float:
        """
        현재가 조회

        Args:
            symbol: 거래쌍

        Returns:
            현재 가격
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            price = float(ticker['last'])
            logger.debug(f"현재가: {symbol} = {price}")
            return price
        except Exception as e:
            logger.error(f"현재가 조회 실패: {e}")
            raise

    def get_orderbook(self, symbol: str, limit: int = 10) -> dict:
        """
        호가창 조회

        Args:
            symbol: 거래쌍
            limit: 호가 개수

        Returns:
            {"bids": [[price, amount], ...], "asks": [[price, amount], ...]}
        """
        try:
            orderbook = self.exchange.fetch_order_book(symbol, limit)
            return {
                "bids": orderbook['bids'][:limit],
                "asks": orderbook['asks'][:limit],
                "timestamp": orderbook.get('timestamp')
            }
        except Exception as e:
            logger.error(f"호가창 조회 실패: {e}")
            raise

    def get_recent_trades(self, symbol: str, limit: int = 50) -> pd.DataFrame:
        """
        최근 체결 내역 조회

        Args:
            symbol: 거래쌍
            limit: 조회 개수

        Returns:
            DataFrame with trade data
        """
        try:
            trades = self.exchange.fetch_trades(symbol, limit=limit)

            df = pd.DataFrame([{
                'timestamp': trade['timestamp'],
                'price': float(trade['price']),
                'amount': float(trade['amount']),
                'side': trade['side']
            } for trade in trades])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df

        except Exception as e:
            logger.error(f"체결 내역 조회 실패: {e}")
            raise

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        ATR (Average True Range) 계산

        Args:
            df: OHLCV DataFrame
            period: ATR 기간

        Returns:
            ATR Series
        """
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def get_next_candle_time(self, timeframe: str) -> datetime:
        """
        다음 캔들 생성 시간 계산

        Args:
            timeframe: 타임프레임

        Returns:
            다음 캔들 생성 시간
        """
        now = datetime.now()
        minutes = self.TIMEFRAME_MINUTES.get(timeframe, 60)

        # 현재 시간을 타임프레임 단위로 올림
        current_minutes = now.hour * 60 + now.minute
        next_candle_minutes = ((current_minutes // minutes) + 1) * minutes

        next_hour = next_candle_minutes // 60
        next_minute = next_candle_minutes % 60

        next_candle = now.replace(hour=next_hour % 24, minute=next_minute, second=0, microsecond=0)

        if next_candle <= now:
            next_candle += timedelta(days=1)

        return next_candle
