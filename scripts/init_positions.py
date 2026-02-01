#!/usr/bin/env python3
"""
현재 포지션을 동기화하고 상태 파일에 저장하는 스크립트
봇 첫 실행 전에 한번 실행하면 됨
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import datetime

from src.bybit_client import BybitClient
from src.data_fetcher import DataFetcher
from src.ichimoku import calculate_ichimoku
from src.strategy import STRATEGY_PARAMS, MAJOR_COINS, LEVERAGE


def main():
    print("=" * 50)
    print("포지션 초기화 및 저장")
    print("=" * 50)

    # 클라이언트 초기화
    client = BybitClient(testnet=False)
    data_fetcher = DataFetcher(client)

    # 거래소에서 포지션 조회
    print("\n거래소 포지션 조회 중...")
    exchange_positions = client.get_all_positions()

    if not exchange_positions:
        print("보유 중인 포지션이 없습니다.")
        return

    positions = {}
    params = STRATEGY_PARAMS

    for pos in exchange_positions:
        symbol = pos["symbol"]

        if symbol not in MAJOR_COINS:
            print(f"  - {symbol}: 운용 대상 아님 (스킵)")
            continue

        entry_price = pos["entry_price"]
        side = pos["side"]
        size = pos["size"]

        print(f"\n{'='*40}")
        print(f"코인: {symbol}")
        print(f"방향: {side.upper()}")
        print(f"진입가: ${entry_price:,.2f}")
        print(f"수량: {size}")

        # 일목 데이터 조회
        print("일목 데이터 계산 중...")
        df = data_fetcher.get_ohlcv(symbol, "4h", limit=200)
        if df is None or df.empty:
            print("  데이터 조회 실패!")
            continue

        df = df.reset_index()
        df = calculate_ichimoku(df)
        df = df.dropna(subset=["tenkan", "kijun", "cloud_top", "cloud_bottom"])

        if df.empty:
            print("  일목 계산 실패!")
            continue

        row = df.iloc[-1]
        cloud_top = float(row["cloud_top"])
        cloud_bottom = float(row["cloud_bottom"])

        print(f"현재 구름 상단: ${cloud_top:,.2f}")
        print(f"현재 구름 하단: ${cloud_bottom:,.2f}")

        # 손절가/익절가 계산
        if side == "long":
            stop_loss = cloud_top * (1 - params["sl_buffer"] / 100)
            sl_distance_pct = (entry_price - stop_loss) / entry_price * 100
            take_profit = entry_price * (1 + sl_distance_pct * params["rr_ratio"] / 100)
        else:
            stop_loss = cloud_bottom * (1 + params["sl_buffer"] / 100)
            sl_distance_pct = (stop_loss - entry_price) / entry_price * 100
            take_profit = entry_price * (1 - sl_distance_pct * params["rr_ratio"] / 100)

        print(f"\n계산된 손절가: ${stop_loss:,.2f} ({sl_distance_pct:.2f}%)")
        print(f"계산된 익절가: ${take_profit:,.2f} (RR 1:{params['rr_ratio']})")

        # 수동 조정 옵션
        print("\n[옵션] 손절가/익절가를 수동으로 설정하시겠습니까?")
        print("  Enter: 계산된 값 사용")
        print("  sl,tp: 직접 입력 (예: 3450,3600)")

        user_input = input("> ").strip()
        if user_input:
            try:
                parts = user_input.split(",")
                if len(parts) == 2:
                    stop_loss = float(parts[0])
                    take_profit = float(parts[1])
                    print(f"수동 설정: SL=${stop_loss:,.2f}, TP=${take_profit:,.2f}")
            except:
                print("입력 오류, 계산된 값 사용")

        positions[symbol] = {
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "entry_time": datetime.utcnow().isoformat(),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "highest": entry_price,
            "lowest": entry_price,
            "trail_stop": stop_loss,
            "trailing": False,
            "size": size,
            "pnl": pos.get("pnl", 0),
        }

    if not positions:
        print("\n저장할 포지션이 없습니다.")
        return

    # 상태 파일 저장
    state_file = "data/bot_state.json"
    os.makedirs(os.path.dirname(state_file), exist_ok=True)

    state = {
        "positions": positions,
        "last_exit_times": {},
        "trade_history": [],
        "saved_at": datetime.utcnow().isoformat()
    }

    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)

    print("\n" + "=" * 50)
    print(f"✅ 상태 저장 완료: {state_file}")
    print("=" * 50)

    for sym, p in positions.items():
        short_sym = sym.split('/')[0]
        print(f"\n{short_sym} {p['side'].upper()}")
        print(f"  진입가: ${p['entry_price']:,.2f}")
        print(f"  손절가: ${p['stop_loss']:,.2f}")
        print(f"  익절가: ${p['take_profit']:,.2f}")


if __name__ == "__main__":
    main()
