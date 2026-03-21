#!/usr/bin/env python3
"""
바이빗 전체 코인 과거 데이터 수집기

모든 USDT 무기한 선물의 1~2년치 멀티 타임프레임 데이터를 수집합니다.
- 타임프레임: 1m, 5m, 15m, 1h, 4h, 1d
- 저장 형식: Parquet (압축, 빠른 로드)
- 증분 업데이트 지원

사용법:
    # 전체 수집 (2년치)
    python collect_historical_data.py --years 2

    # 특정 타임프레임만
    python collect_historical_data.py --timeframes 5m 1h 4h --years 1

    # 기존 데이터 업데이트 (최신 데이터만 추가)
    python collect_historical_data.py --update

    # 특정 코인만
    python collect_historical_data.py --symbols BTC/USDT:USDT ETH/USDT:USDT
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from src.bybit_client import BybitClient
from src.data_fetcher import DataFetcher
from src.surge_strategy import get_all_usdt_perpetuals


# 데이터 저장 경로
DATA_DIR = Path("data/historical")
METADATA_FILE = DATA_DIR / "metadata.json"


class HistoricalDataCollector:
    """과거 데이터 수집 및 관리"""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.client = BybitClient()
        self.data_fetcher = DataFetcher(self.client)

        # 메타데이터 로드
        self.metadata = self.load_metadata()

    def load_metadata(self) -> Dict:
        """메타데이터 로드"""
        if METADATA_FILE.exists():
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
        return {}

    def save_metadata(self):
        """메타데이터 저장"""
        with open(METADATA_FILE, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def get_timeframe_info(self, timeframe: str) -> Dict:
        """타임프레임 정보"""
        info = {
            '1m': {'minutes': 1, 'limit': 1000, 'name': '1분봉'},
            '5m': {'minutes': 5, 'limit': 1000, 'name': '5분봉'},
            '15m': {'minutes': 15, 'limit': 1000, 'name': '15분봉'},
            '1h': {'minutes': 60, 'limit': 1000, 'name': '1시간봉'},
            '4h': {'minutes': 240, 'limit': 1000, 'name': '4시간봉'},
            '1d': {'minutes': 1440, 'limit': 1000, 'name': '1일봉'},
        }
        return info.get(timeframe, {'minutes': 60, 'limit': 1000, 'name': timeframe})

    def calculate_batches(self, timeframe: str, years: int) -> int:
        """필요한 API 호출 횟수 계산"""
        tf_info = self.get_timeframe_info(timeframe)
        candles_per_day = 1440 / tf_info['minutes']
        total_candles = candles_per_day * 365 * years
        batches = int(np.ceil(total_candles / tf_info['limit']))
        return batches

    def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        years: int = 2,
        show_progress: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        특정 심볼의 과거 데이터 수집

        Args:
            symbol: 심볼 (예: "BTC/USDT:USDT")
            timeframe: 타임프레임 (예: "5m")
            years: 수집 기간 (년)
            show_progress: 진행 상황 출력

        Returns:
            DataFrame (timestamp, open, high, low, close, volume)
        """
        tf_info = self.get_timeframe_info(timeframe)
        batches = self.calculate_batches(timeframe, years)

        all_data = []
        end_time = datetime.now()

        if show_progress:
            print(f"  {symbol} {tf_info['name']}: {batches}회 수집 필요")

        for i in range(batches):
            try:
                # 현재 시점부터 과거로
                since = int((end_time - timedelta(days=365*years)).timestamp() * 1000) + (i * tf_info['limit'] * tf_info['minutes'] * 60 * 1000)

                ohlcv = self.client.exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    since=since,
                    limit=tf_info['limit']
                )

                if ohlcv:
                    all_data.extend(ohlcv)

                # Rate limit 준수
                time.sleep(0.12)  # 초당 8회 제한

                if show_progress and (i + 1) % 10 == 0:
                    print(f"    진행: {i+1}/{batches} ({(i+1)/batches*100:.1f}%)")

            except Exception as e:
                print(f"    에러 (batch {i}): {e}")
                time.sleep(1)
                continue

        if not all_data:
            return None

        # DataFrame 생성
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # 중복 제거 및 정렬
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

        if show_progress:
            print(f"    완료: {len(df):,}개 캔들, {df['timestamp'].min()} ~ {df['timestamp'].max()}")

        return df

    def save_data(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """데이터 저장 (Parquet 형식)"""
        # 파일명: data/historical/BTCUSDT/5m.parquet
        symbol_clean = symbol.replace('/', '').replace(':', '')
        symbol_dir = self.data_dir / symbol_clean
        symbol_dir.mkdir(exist_ok=True)

        file_path = symbol_dir / f"{timeframe}.parquet"
        df.to_parquet(file_path, compression='snappy', index=False)

        # 메타데이터 업데이트
        if symbol not in self.metadata:
            self.metadata[symbol] = {}

        self.metadata[symbol][timeframe] = {
            'file': str(file_path),
            'rows': len(df),
            'start': str(df['timestamp'].min()),
            'end': str(df['timestamp'].max()),
            'updated_at': str(datetime.now()),
        }

        self.save_metadata()

    def load_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """저장된 데이터 로드"""
        symbol_clean = symbol.replace('/', '').replace(':', '')
        file_path = self.data_dir / symbol_clean / f"{timeframe}.parquet"

        if not file_path.exists():
            return None

        return pd.read_parquet(file_path)

    def update_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """기존 데이터 업데이트 (최신 데이터만 추가)"""
        existing_df = self.load_data(symbol, timeframe)

        if existing_df is None:
            # 기존 데이터 없으면 전체 수집
            return None

        last_timestamp = existing_df['timestamp'].max()
        days_old = (datetime.now() - last_timestamp).days

        if days_old < 1:
            print(f"  {symbol} {timeframe}: 최신 상태 (업데이트 불필요)")
            return existing_df

        print(f"  {symbol} {timeframe}: {days_old}일 업데이트 필요")

        # 최신 데이터 수집
        tf_info = self.get_timeframe_info(timeframe)
        limit = min(days_old * int(1440 / tf_info['minutes']) + 100, 1000)

        try:
            new_df = self.data_fetcher.get_ohlcv(symbol, timeframe, limit=limit)

            if new_df is not None:
                new_df = new_df.reset_index()

                # 기존 데이터와 병합
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

                print(f"    추가: {len(combined) - len(existing_df)}개 캔들")
                return combined

        except Exception as e:
            print(f"    업데이트 실패: {e}")

        return existing_df

    def _collect_one(self, symbol: str, timeframe: str, years: int, update_mode: bool, skip_existing: bool = True) -> tuple:
        """단일 심볼/타임프레임 데이터 수집 (병렬 처리용)"""
        try:
            # === 스킵 로직: 이미 수집된 파일 체크 ===
            if skip_existing and not update_mode:
                symbol_clean = symbol.replace('/', '').replace(':', '')
                file_path = self.data_dir / symbol_clean / f"{timeframe}.parquet"

                if file_path.exists():
                    try:
                        # 파일 크기 확인 (최소 1KB 이상)
                        file_size = file_path.stat().st_size
                        if file_size > 1024:
                            existing_df = pd.read_parquet(file_path)

                            # 충분한 데이터가 있는지 확인 (최소 100개 캔들)
                            if len(existing_df) > 100:
                                # 스킵!
                                return (symbol, timeframe, len(existing_df), "Skipped (already exists)")
                    except Exception as e:
                        # 파일 손상 시 다시 수집
                        print(f"    경고: {symbol} {timeframe} 파일 손상, 재수집 - {e}")
                        pass

            # === 기존 수집 로직 ===
            if update_mode:
                # 업데이트 모드
                df = self.update_data(symbol, timeframe)
                if df is None:
                    # 기존 데이터 없으면 전체 수집
                    df = self.fetch_historical_data(symbol, timeframe, years, show_progress=False)
            else:
                # 전체 수집 모드
                df = self.fetch_historical_data(symbol, timeframe, years, show_progress=False)

            if df is not None and len(df) > 0:
                self.save_data(symbol, timeframe, df)
                return (symbol, timeframe, len(df), None)
            else:
                return (symbol, timeframe, 0, "No data")

        except Exception as e:
            return (symbol, timeframe, 0, str(e))

    def collect_all(
        self,
        symbols: List[str],
        timeframes: List[str],
        years: int = 2,
        update_mode: bool = False,
        max_workers: int = 8,  # 병렬 처리 워커 수 (권장: 4~8)
        skip_existing: bool = True  # 이미 수집된 파일 스킵
    ):
        """
        모든 심볼/타임프레임 데이터 수집 (병렬 처리)

        Args:
            symbols: 심볼 리스트
            timeframes: 타임프레임 리스트
            years: 수집 기간
            update_mode: True면 기존 데이터 업데이트만
            max_workers: 병렬 처리 워커 수 (4~8 권장)
            skip_existing: True면 이미 수집된 파일 스킵 (기본: True)
        """
        print(f"\n{'='*70}")
        print(f"  {'데이터 업데이트' if update_mode else '전체 데이터 수집'} (병렬 처리)")
        print(f"{'='*70}")
        print(f"심볼: {len(symbols)}개")
        print(f"타임프레임: {', '.join(timeframes)}")
        if not update_mode:
            print(f"기간: {years}년")
        print(f"병렬 워커: {max_workers}개")
        print(f"기존 파일 스킵: {'예' if skip_existing else '아니오 (강제 재다운로드)'}")
        print()

        # 모든 작업 생성
        tasks = [(symbol, tf) for symbol in symbols for tf in timeframes]
        total_tasks = len(tasks)
        completed = 0
        success = 0
        failed = 0

        print(f"총 작업 수: {total_tasks}개")
        print(f"예상 시간: {total_tasks * 0.15 / max_workers / 60:.1f}분 (병렬 처리)")
        print()

        # 병렬 처리
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 작업 제출
            futures = {
                executor.submit(self._collect_one, symbol, tf, years, update_mode, skip_existing): (symbol, tf)
                for symbol, tf in tasks
            }

            # 완료되는 대로 처리
            for future in as_completed(futures):
                symbol, tf = futures[future]
                completed += 1

                try:
                    sym, timeframe, rows, error = future.result()

                    if error is None:
                        success += 1
                        status = f"OK {rows:,} candles"
                    elif error == "Skipped (already exists)":
                        success += 1
                        status = f"SKIP ({rows:,} candles exist)"
                    else:
                        failed += 1
                        status = f"FAIL {error[:30]}"

                    # 진행 상황 출력 (10개마다)
                    if completed % 10 == 0 or completed == total_tasks:
                        print(f"  [{completed:4d}/{total_tasks}] {sym:20s} {timeframe:4s}: {status} | Success: {success}, Failed: {failed}")
                    elif "Skipped" not in error if error else False:
                        # 스킵 외의 에러는 바로 출력
                        print(f"  [{completed:4d}/{total_tasks}] {sym:20s} {timeframe:4s}: {status}")

                except Exception as e:
                    failed += 1
                    print(f"  [{completed:4d}/{total_tasks}] {symbol:20s} {tf:4s}: ERROR processing - {e}")

        print(f"\n{'='*70}")
        print(f"  수집 완료!")
        print(f"{'='*70}")
        print(f"총 작업: {total_tasks}개")
        print(f"성공: {success}개")
        print(f"실패: {failed}개")
        print(f"성공률: {success/total_tasks*100:.1f}%")
        print()
        print(f"저장 경로: {self.data_dir}")
        print(f"메타데이터: {METADATA_FILE}")

    def get_stats(self):
        """수집된 데이터 통계"""
        print(f"\n{'='*70}")
        print(f"  데이터 통계")
        print(f"{'='*70}")

        if not self.metadata:
            print("수집된 데이터 없음")
            return

        total_symbols = len(self.metadata)
        total_files = 0
        total_candles = 0
        total_size = 0

        timeframe_stats = {}

        for symbol, tf_data in self.metadata.items():
            for tf, info in tf_data.items():
                total_files += 1
                total_candles += info['rows']

                file_path = Path(info['file'])
                if file_path.exists():
                    total_size += file_path.stat().st_size

                if tf not in timeframe_stats:
                    timeframe_stats[tf] = {'symbols': 0, 'candles': 0}
                timeframe_stats[tf]['symbols'] += 1
                timeframe_stats[tf]['candles'] += info['rows']

        print(f"심볼 수: {total_symbols}개")
        print(f"파일 수: {total_files}개")
        print(f"총 캔들 수: {total_candles:,}개")
        print(f"디스크 사용량: {total_size / 1024 / 1024:.1f} MB")

        print(f"\n타임프레임별 통계:")
        for tf in ['1m', '5m', '15m', '1h', '4h', '1d']:
            if tf in timeframe_stats:
                stats = timeframe_stats[tf]
                print(f"  {tf:4s}: {stats['symbols']:3d}개 심볼, {stats['candles']:8,d}개 캔들")

        print(f"\n최근 업데이트:")
        recent = sorted(
            [(s, tf, info['updated_at']) for s, tfs in self.metadata.items() for tf, info in tfs.items()],
            key=lambda x: x[2],
            reverse=True
        )[:5]

        for symbol, tf, updated in recent:
            print(f"  {symbol:20s} {tf:4s}: {updated}")


def main():
    parser = argparse.ArgumentParser(description='바이빗 과거 데이터 수집 (병렬 처리)')
    parser.add_argument('--years', '-y', type=int, default=2, help='수집 기간 (년, 기본 2년)')
    parser.add_argument('--timeframes', '-t', nargs='+',
                       default=['1m', '5m', '15m', '1h', '4h', '1d'],
                       help='타임프레임 (기본: 전체)')
    parser.add_argument('--symbols', '-s', nargs='+', help='특정 심볼만 (기본: 전체)')
    parser.add_argument('--update', '-u', action='store_true', help='업데이트 모드 (기존 데이터 갱신만)')
    parser.add_argument('--stats', action='store_true', help='통계만 출력')
    parser.add_argument('--limit', '-l', type=int, help='코인 수 제한 (테스트용)')
    parser.add_argument('--workers', '-w', type=int, default=8, help='병렬 워커 수 (기본: 8, 권장: 4~12)')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='이미 수집된 파일 스킵 (기본: True)')
    parser.add_argument('--force-redownload', action='store_true',
                       help='모든 파일 강제 재다운로드 (--skip-existing 무시)')

    args = parser.parse_args()

    collector = HistoricalDataCollector()

    # 통계 출력
    if args.stats:
        collector.get_stats()
        return

    # 심볼 목록
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = get_all_usdt_perpetuals()
        if args.limit:
            symbols = symbols[:args.limit]

    # 수집 시작
    collector.collect_all(
        symbols=symbols,
        timeframes=args.timeframes,
        years=args.years,
        update_mode=args.update,
        max_workers=args.workers,
        skip_existing=not args.force_redownload  # force가 True면 skip False
    )

    # 통계 출력
    collector.get_stats()


if __name__ == '__main__':
    main()
