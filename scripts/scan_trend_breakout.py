#!/usr/bin/env python3
"""
트렌드 급등 코인 스캐너

급등 후 트렌드로 전환되어 지속 상승하는 코인 탐지
- 급등 발생 + 건강한 조정 + 이평선 정배열

사용법:
    python scripts/scan_trend_breakout.py
    python scripts/scan_trend_breakout.py --timeframe 1h
    python scripts/scan_trend_breakout.py --top 20
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime

from src.bybit_client import BybitClient
from src.data_fetcher import DataFetcher
from src.trend_breakout_strategy import (
    calculate_trend_indicators,
    get_trend_breakout_entry_signal,
    detect_surge_pattern,
    get_higher_timeframe_confirmation,
    get_trend_watch_list,
    TREND_BREAKOUT_PARAMS,
    TREND_WATCH_LIST_FALLBACK,
)


def print_header():
    """헤더 출력"""
    print("\n" + "=" * 80)
    print("  트렌드 급등 코인 스캐너")
    print("  급등 → 건강한 조정 → 트렌드 지속 패턴 탐지")
    print("=" * 80)
    print(f"  시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")


def scan_all_coins(data_fetcher, timeframe: str = '4h', top_n: int = 10, use_fallback: bool = False):
    """
    전체 코인 스캔

    Args:
        data_fetcher: DataFetcher 인스턴스
        timeframe: 타임프레임
        top_n: 상위 N개만 표시
        use_fallback: 폴백 리스트 사용 여부
    """
    if use_fallback:
        symbols = TREND_WATCH_LIST_FALLBACK
        print(f"폴백 리스트 사용 ({len(symbols)}개 코인)")
    else:
        try:
            symbols = get_trend_watch_list(exclude_large_caps=False)
            print(f"전체 USDT 선물 스캔 ({len(symbols)}개 코인)")
        except Exception as e:
            print(f"종목 조회 실패, 폴백 사용: {e}")
            symbols = TREND_WATCH_LIST_FALLBACK

    print(f"타임프레임: {timeframe}\n")

    candidates = []
    scanned = 0
    errors = 0

    for i, symbol in enumerate(symbols):
        try:
            df = data_fetcher.get_ohlcv(symbol, timeframe, limit=100)
            if df is None or len(df) < 60:
                continue

            scanned += 1

            # 지표 계산
            df = calculate_trend_indicators(df, TREND_BREAKOUT_PARAMS)

            # 급등 패턴 먼저 확인
            surge = detect_surge_pattern(df, TREND_BREAKOUT_PARAMS)

            if surge['has_surge']:
                # 진입 신호 체크
                signal = get_trend_breakout_entry_signal(symbol, df, TREND_BREAKOUT_PARAMS)

                if signal:
                    # 상위 타임프레임 확인
                    htf = get_higher_timeframe_confirmation(symbol, data_fetcher)
                    signal['htf_confirmed'] = htf['confirmed']
                    signal['htf_trend'] = htf['trend']

                    candidates.append(signal)
                else:
                    # 급등은 있지만 진입 조건 미충족
                    candidates.append({
                        'symbol': symbol,
                        'score': 0,
                        'surge_pct': surge['surge_pct'],
                        'pullback_ratio': surge['pullback_ratio'],
                        'is_healthy': surge['is_healthy'],
                        'reason': '진입조건 미충족',
                        'htf_confirmed': False,
                    })

            # 진행 상황 표시
            if (i + 1) % 50 == 0:
                print(f"  스캔 중... {i + 1}/{len(symbols)}")

        except Exception as e:
            errors += 1
            continue

    print(f"\n스캔 완료: {scanned}개 분석, {errors}개 오류")

    # 점수순 정렬
    candidates.sort(key=lambda x: x.get('score', 0), reverse=True)

    return candidates[:top_n * 2]  # 여유있게 반환


def print_candidates(candidates, top_n: int = 10):
    """후보 출력"""
    if not candidates:
        print("\n발견된 후보 없음")
        return

    # 진입 신호 있는 것만
    entry_signals = [c for c in candidates if c.get('score', 0) > 0]
    watching = [c for c in candidates if c.get('score', 0) == 0 and c.get('surge_pct', 0) > 0]

    print("\n" + "=" * 80)
    print("  [진입 가능] 트렌드 급등 신호")
    print("=" * 80)

    if entry_signals:
        print(f"\n{'심볼':<18} {'점수':>5} {'급등':>8} {'조정':>8} {'거래량':>8} {'RSI':>6} {'일봉확인':>8}")
        print("-" * 80)

        for c in entry_signals[:top_n]:
            htf_status = "O" if c.get('htf_confirmed') else "X"
            print(
                f"{c['symbol']:<18} "
                f"{c['score']:>5} "
                f"{c['surge_pct']:>7.1f}% "
                f"{c['pullback_ratio']:>7.1f}% "
                f"{c.get('volume_surge', 0):>7.1f}x "
                f"{c.get('rsi', 0):>5.1f} "
                f"{htf_status:>8}"
            )

        print("-" * 80)
        print("\n진입 조건:")
        print("  - 급등: 최근 7일 15%+ 상승")
        print("  - 조정: 되돌림 10~50% (건강한 조정)")
        print("  - 이평선: EMA 정배열 (9>21>50)")
        print("  - 거래량: 지속적으로 높음")
        print("  - 일봉확인: O = 일봉에서도 상승 구조")

    else:
        print("\n진입 조건 충족 코인 없음")

    # 관찰 대상
    if watching:
        print("\n" + "-" * 80)
        print("  [관찰 대상] 급등 발생했지만 조건 미충족")
        print("-" * 80)
        print(f"\n{'심볼':<18} {'급등':>8} {'조정':>8} {'상태':<20}")
        print("-" * 60)

        for c in watching[:5]:
            health = "건강한 조정" if c.get('is_healthy') else "조정 과다/부족"
            print(
                f"{c['symbol']:<18} "
                f"{c['surge_pct']:>7.1f}% "
                f"{c['pullback_ratio']:>7.1f}% "
                f"{health:<20}"
            )


def analyze_single_coin(data_fetcher, symbol: str, timeframe: str = '4h'):
    """단일 코인 상세 분석"""
    print(f"\n{'=' * 60}")
    print(f"  {symbol} 상세 분석")
    print(f"{'=' * 60}\n")

    df = data_fetcher.get_ohlcv(symbol, timeframe, limit=100)
    if df is None or len(df) < 60:
        print("데이터 부족")
        return

    df = calculate_trend_indicators(df, TREND_BREAKOUT_PARAMS)
    row = df.iloc[-1]

    # 기본 정보
    print(f"현재가: ${row['close']:.4f}")
    print(f"RSI: {row['rsi']:.1f}")
    print(f"ATR%: {row['atr_pct']:.2f}%")
    print()

    # 급등 패턴
    surge = detect_surge_pattern(df, TREND_BREAKOUT_PARAMS)
    print("[급등 패턴]")
    print(f"  급등 발생: {'O' if surge['has_surge'] else 'X'}")
    print(f"  급등폭: {surge['surge_pct']:.1f}%")
    print(f"  조정 비율: {surge['pullback_ratio']:.1f}%")
    print(f"  건강한 조정: {'O' if surge['is_healthy'] else 'X'}")
    print()

    # 이평선
    print("[이평선 상태]")
    print(f"  EMA9: ${row['ema_fast']:.4f}")
    print(f"  EMA21: ${row['ema_mid']:.4f}")
    print(f"  EMA50: ${row['ema_slow']:.4f}")
    print(f"  정배열: {'O' if row['ema_aligned'] else 'X'}")
    print(f"  가격 위치: {'이평선 위' if row['price_above_emas'] else '이평선 아래'}")
    print()

    # 거래량
    print("[거래량]")
    print(f"  현재/평균: {row['volume_ratio']:.2f}x")
    print(f"  최근 지속: {row['volume_consistent']:.2f}x")
    print(f"  OBV 상승: {'O' if row['obv_rising'] else 'X'}")
    print()

    # 진입 신호
    signal = get_trend_breakout_entry_signal(symbol, df, TREND_BREAKOUT_PARAMS)
    print("[진입 신호]")
    if signal:
        print(f"  상태: 진입 가능!")
        print(f"  점수: {signal['score']}")
        print(f"  진입가: ${signal['price']:.4f}")
        print(f"  손절가: ${signal['stop_loss']:.4f} ({(signal['stop_loss']/signal['price']-1)*100:.1f}%)")
        print(f"  익절가: ${signal['take_profit']:.4f} ({(signal['take_profit']/signal['price']-1)*100:.1f}%)")
    else:
        print(f"  상태: 진입 조건 미충족")

    # 상위 타임프레임
    print("\n[일봉 확인]")
    htf = get_higher_timeframe_confirmation(symbol, data_fetcher)
    print(f"  트렌드: {htf['trend']}")
    print(f"  확정: {'O' if htf['confirmed'] else 'X'}")


def main():
    parser = argparse.ArgumentParser(description='트렌드 급등 코인 스캐너')
    parser.add_argument('--timeframe', '-t', default='4h', help='타임프레임 (기본: 4h)')
    parser.add_argument('--top', '-n', type=int, default=10, help='상위 N개 표시 (기본: 10)')
    parser.add_argument('--symbol', '-s', help='특정 심볼 분석')
    parser.add_argument('--fallback', '-f', action='store_true', help='폴백 리스트 사용 (빠름)')

    args = parser.parse_args()

    # 초기화
    client = BybitClient()
    data_fetcher = DataFetcher(client)

    print_header()

    if args.symbol:
        # 특정 심볼 분석
        analyze_single_coin(data_fetcher, args.symbol, args.timeframe)
    else:
        # 전체 스캔
        candidates = scan_all_coins(
            data_fetcher,
            timeframe=args.timeframe,
            top_n=args.top,
            use_fallback=args.fallback
        )
        print_candidates(candidates, args.top)

    print("\n" + "=" * 80)
    print("  스캔 완료")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
