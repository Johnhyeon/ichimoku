"""
급등 코인 전략: 급락 후 계단식 상승 패턴

패턴:
1. 급락으로 깊은 골 생성 (N일간 X% 하락)
2. 1차 반등 (저점에서 Y% 상승)
3. 1차 조정 (고점에서 Z% 하락, 저점 유지)
4. 2차 상승 시작 → 진입!

진입: 2차 상승 시작점
손절: 급락 저점 아래
익절: 트레일링 또는 목표가
"""

import sys
sys.path.insert(0, '/home/hyeon/project/ichimoku')

import time
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
import numpy as np
from collections import defaultdict
from pybit.unified_trading import HTTP

FEE_RATE = 0.00055  # 바이빗 테이커

def fetch_klines(symbol: str, interval: str, limit: int = 2000):
    """바이빗에서 캔들 데이터 가져오기"""
    session = HTTP()
    all_data = []
    end_time = None

    # interval 변환
    interval_map = {'15m': 15, '1h': 60, '4h': 240, '1d': 'D'}
    interval_val = interval_map.get(interval, interval)

    while len(all_data) < limit:
        params = {'category': 'linear', 'symbol': symbol, 'interval': interval_val, 'limit': 1000}
        if end_time:
            params['end'] = end_time
        try:
            response = session.get_kline(**params)
            klines = response['result']['list']
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            break
        if not klines:
            break
        all_data.extend(klines)
        end_time = int(klines[-1][0]) - 1
        if len(klines) < 1000:
            break
        time.sleep(0.05)

    if not all_data:
        return None

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    return df


def detect_surge_pattern(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    급등 패턴 감지

    1. 급락: lookback 기간 내 drop_pct% 이상 하락한 저점 찾기
    2. 1차 반등: 저점에서 bounce_pct% 상승
    3. 1차 조정: 반등 고점에서 pullback_pct% 하락 (저점은 유지)
    4. 2차 상승 시작: 조정 저점에서 다시 상승 시작
    """
    df = df.copy()

    lookback = cfg.get('lookback', 20)  # 급락 탐색 기간
    drop_pct = cfg.get('drop_pct', 15)  # 급락 기준 %
    bounce_pct = cfg.get('bounce_pct', 10)  # 1차 반등 %
    pullback_pct = cfg.get('pullback_pct', 5)  # 1차 조정 %

    df['rolling_high'] = df['high'].rolling(lookback).max()
    df['rolling_low'] = df['low'].rolling(lookback).min()

    # 급락 감지: 최근 고점 대비 현재 저점이 drop_pct% 이상 하락
    df['drop_from_high'] = (df['rolling_high'] - df['low']) / df['rolling_high'] * 100
    df['is_deep_drop'] = df['drop_from_high'] >= drop_pct

    # 저점에서의 반등률
    df['bounce_from_low'] = (df['close'] - df['rolling_low']) / df['rolling_low'] * 100

    # 패턴 상태 추적
    df['pattern_state'] = 0  # 0: 없음, 1: 급락 후, 2: 1차 반등 중, 3: 1차 조정 중, 4: 2차 상승 시작!
    df['pattern_low'] = np.nan
    df['pattern_high1'] = np.nan
    df['entry_signal'] = False

    pattern_low = np.nan
    pattern_high1 = np.nan
    pullback_low = np.nan
    state = 0

    for i in range(lookback, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]

        if state == 0:
            # 급락 감지
            if row['is_deep_drop']:
                pattern_low = row['rolling_low']
                state = 1

        elif state == 1:
            # 1차 반등 대기
            current_bounce = (row['close'] - pattern_low) / pattern_low * 100
            if current_bounce >= bounce_pct:
                pattern_high1 = row['high']
                state = 2
            elif row['low'] < pattern_low:
                # 새로운 저점 → 리셋
                pattern_low = row['low']

        elif state == 2:
            # 1차 반등 중 → 고점 갱신 또는 조정 시작
            if row['high'] > pattern_high1:
                pattern_high1 = row['high']
            else:
                pullback = (pattern_high1 - row['low']) / pattern_high1 * 100
                if pullback >= pullback_pct:
                    pullback_low = row['low']
                    state = 3

        elif state == 3:
            # 1차 조정 중 → 저점 유지하면서 2차 상승 시작 감지
            if row['low'] < pattern_low * 0.98:  # 급락 저점 이탈 → 실패
                state = 0
                pattern_low = np.nan
            elif row['low'] < pullback_low:
                pullback_low = row['low']
            elif row['close'] > prev_row['high']:  # 전일 고점 돌파 → 2차 상승 시작!
                df.iloc[i, df.columns.get_loc('entry_signal')] = True
                df.iloc[i, df.columns.get_loc('pattern_low')] = pattern_low
                df.iloc[i, df.columns.get_loc('pattern_high1')] = pattern_high1
                state = 0  # 리셋
                pattern_low = np.nan

    return df


def backtest_surge(all_data: Dict[str, pd.DataFrame], cfg: dict) -> dict:
    """급등 전략 백테스트"""

    trade_cfg = {
        'initial': 5_000_000,
        'leverage': 5,  # 낮은 레버리지
        'pos_pct': 0.05,  # 작은 포지션
        'sl_pct': 0.05,  # 진입가 대비 5% 손절
        'tp_mult': 3.0,  # 손절폭의 3배
        'trail_pct': 2.0,  # 트레일링 2%
        'max_pos': 3,
        'cooldown_hours': 4,
        **cfg.get('trade', {})
    }

    pattern_cfg = cfg.get('pattern', {})

    # 패턴 감지
    for sym in all_data:
        all_data[sym] = detect_surge_pattern(all_data[sym], pattern_cfg)

    # 바 정렬
    bars = []
    for sym, df in all_data.items():
        df = df.dropna(subset=['rolling_high'])
        for _, row in df.iterrows():
            bars.append({'symbol': sym, **row.to_dict()})
    bars.sort(key=lambda x: x['timestamp'])

    # 시간별 그룹
    tg = {}
    for b in bars:
        t = b['timestamp']
        if t not in tg:
            tg[t] = {}
        tg[t][b['symbol']] = b

    times = sorted(tg.keys())

    cash = trade_cfg['initial']
    positions = {}
    trades = []
    equity = []
    last_exit = {}
    total_fees = 0

    for t in times:
        current_bars = tg[t]
        closed = []

        # 청산 체크
        for sym, pos in positions.items():
            if sym not in current_bars:
                continue
            b = current_bars[sym]
            h, l = b['high'], b['low']
            entry = pos['entry']
            reason = None
            exit_p = None

            # 최고가 갱신
            if h > pos['highest']:
                pos['highest'] = h
                # 트레일링 업데이트
                if h >= pos['tp']:
                    pos['trailing'] = True
                    pos['trail_stop'] = max(pos['trail_stop'], h * (1 - trade_cfg['trail_pct'] / 100))

            # 손절
            if l <= pos['sl']:
                reason = 'SL'
                exit_p = pos['sl']
            # 트레일링
            elif pos['trailing'] and l <= pos['trail_stop']:
                reason = 'Trail'
                exit_p = pos['trail_stop']
            # 익절
            elif not pos['trailing'] and h >= pos['tp']:
                reason = 'TP'
                exit_p = pos['tp']

            if reason:
                notional = pos['size'] * trade_cfg['leverage']
                fee = notional * FEE_RATE * 2
                total_fees += fee

                pnl_pct = (exit_p - entry) / entry * 100
                realized = pnl_pct * trade_cfg['leverage'] / 100 * pos['size'] - fee

                cash += pos['size'] + realized

                trades.append({
                    'symbol': sym,
                    'entry_time': pos['entry_time'],
                    'exit_time': t,
                    'entry': entry,
                    'exit': exit_p,
                    'sl': pos['sl'],
                    'tp': pos['tp'],
                    'pnl_pct': round(pnl_pct * trade_cfg['leverage'], 2),
                    'pnl_after_fee': round(pnl_pct * trade_cfg['leverage'] - fee / pos['size'] * 100, 2),
                    'pnl_krw': round(realized, 0),
                    'reason': reason,
                    'highest': pos['highest']
                })
                closed.append(sym)
                last_exit[sym] = t

        for s in closed:
            del positions[s]

        # 자산 계산
        unreal = sum(
            (current_bars[s]['close'] - p['entry']) / p['entry'] * trade_cfg['leverage'] * p['size'] / 100
            for s, p in positions.items() if s in current_bars
        )
        eq = cash + sum(p['size'] for p in positions.values()) + unreal

        # 진입 체크
        pos_size = eq * trade_cfg['pos_pct']

        if cash >= pos_size and len(positions) < trade_cfg['max_pos']:
            for sym, b in current_bars.items():
                if sym in positions:
                    continue
                if sym in last_exit:
                    if (t - last_exit[sym]).total_seconds() < trade_cfg['cooldown_hours'] * 3600:
                        continue

                if b.get('entry_signal', False):
                    price = b['close']
                    pattern_low = b['pattern_low']

                    # 손절: 진입가 대비 고정 %
                    sl_pct = trade_cfg.get('sl_pct', 0.05)
                    sl = price * (1 - sl_pct)

                    # 익절: 손절폭의 N배
                    tp = price * (1 + sl_pct * trade_cfg['tp_mult'])

                    positions[sym] = {
                        'entry': price,
                        'entry_time': t,
                        'sl': sl,
                        'tp': tp,
                        'size': pos_size,
                        'highest': price,
                        'trailing': False,
                        'trail_stop': sl,
                        'pattern_low': pattern_low
                    }
                    cash -= pos_size

                    if len(positions) >= trade_cfg['max_pos']:
                        break

        equity.append({'time': t, 'equity': eq})

    # 결과 계산
    if not trades:
        return {'trades': 0, 'ev': -999, 'ret': -100}

    wins = [t for t in trades if t['pnl_krw'] > 0]
    losses = [t for t in trades if t['pnl_krw'] <= 0]

    win_rate = len(wins) / len(trades)
    avg_win = np.mean([t['pnl_after_fee'] for t in wins]) if wins else 0
    avg_loss = abs(np.mean([t['pnl_after_fee'] for t in losses])) if losses else 0
    ev = win_rate * avg_win - (1 - win_rate) * avg_loss

    profit = sum(t['pnl_krw'] for t in wins) if wins else 0
    loss = abs(sum(t['pnl_krw'] for t in losses)) if losses else 0

    final = equity[-1]['equity'] if equity else trade_cfg['initial']

    # MDD
    peak = trade_cfg['initial']
    max_dd = 0
    for e in equity:
        if e['equity'] > peak:
            peak = e['equity']
        max_dd = max(max_dd, (peak - e['equity']) / peak * 100)

    return {
        'trades': len(trades),
        'wins': len(wins),
        'wr': round(win_rate * 100, 1),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'ev': round(ev, 3),
        'ret': round((final - trade_cfg['initial']) / trade_cfg['initial'] * 100, 1),
        'pf': round(profit / loss, 2) if loss > 0 else 999,
        'mdd': round(max_dd, 1),
        'fees': round(total_fees, 0),
        'trades_list': trades
    }


if __name__ == '__main__':
    print("=" * 90)
    print("급등 코인 전략: 급락 후 계단식 상승")
    print("=" * 90)
    print("패턴: 급락 → 1차 반등 → 1차 조정 → 2차 상승 시작 (진입)")
    print("=" * 90)

    # 다양한 코인 테스트 (저시총 밈코인 포함)
    COINS = [
        # 메이저
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
        # 알트
        'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT',
        'SUIUSDT', 'APTUSDT', 'INJUSDT', 'ARBUSDT', 'OPUSDT',
        # 밈/저시총 (1000x 심볼)
        'WIFUSDT', '1000PEPEUSDT', '1000SHIBUSDT', '1000BONKUSDT', '1000FLOKIUSDT',
        'MEMEUSDT', 'PEOPLEUSDT', 'LUNCUSDT', 'BOMEUSDT', 'NOTUSDT',
        'WLDUSDT', 'TURBOUSDT', 'NEIROUSDT', 'POPCATUSDT',
    ]

    print(f"\n{len(COINS)}개 코인 데이터 수집 중...")
    all_data = {}
    for i, sym in enumerate(COINS):
        print(f"  {i+1}/{len(COINS)} {sym}...", end='', flush=True)
        df = fetch_klines(sym, '4h', limit=3000)  # 4시간봉
        if df is not None and len(df) > 100:
            all_data[sym] = df
            print(f" OK ({len(df)}개)")
        else:
            print(" SKIP")

    if not all_data:
        print("데이터 없음")
        exit()

    print(f"\n{len(all_data)}개 코인 로드 완료")
    first_df = list(all_data.values())[0]
    print(f"기간: {first_df['timestamp'].min()} ~ {first_df['timestamp'].max()}")

    # ============================================================
    # 1. 패턴 파라미터 테스트
    # ============================================================
    print("\n" + "=" * 90)
    print("[1] 패턴 파라미터 테스트")
    print("=" * 90)

    pattern_configs = [
        {'name': '약한급락 10/5/3', 'cfg': {'lookback': 20, 'drop_pct': 10, 'bounce_pct': 5, 'pullback_pct': 3}},
        {'name': '중간급락 15/8/4', 'cfg': {'lookback': 20, 'drop_pct': 15, 'bounce_pct': 8, 'pullback_pct': 4}},
        {'name': '강한급락 20/10/5', 'cfg': {'lookback': 20, 'drop_pct': 20, 'bounce_pct': 10, 'pullback_pct': 5}},
        {'name': '극단급락 25/12/6', 'cfg': {'lookback': 30, 'drop_pct': 25, 'bounce_pct': 12, 'pullback_pct': 6}},
        {'name': '짧은주기 10/5/3', 'cfg': {'lookback': 10, 'drop_pct': 10, 'bounce_pct': 5, 'pullback_pct': 3}},
        {'name': '긴주기 20/10/5', 'cfg': {'lookback': 40, 'drop_pct': 20, 'bounce_pct': 10, 'pullback_pct': 5}},
    ]

    print(f"\n{'패턴':<20} {'거래':>6} {'승률':>8} {'평균승':>10} {'평균손':>10} {'EV':>10} {'수익률':>10} {'MDD':>8}")
    print("-" * 90)

    best_pattern = None
    best_ev = -999

    for pc in pattern_configs:
        r = backtest_surge(
            {k: v.copy() for k, v in all_data.items()},
            {'pattern': pc['cfg']}
        )
        ev_ok = "✅" if r['ev'] > 0 else "❌"
        print(f"{pc['name']:<20} {r['trades']:>6} {r['wr']:>7.1f}% {r['avg_win']:>9.1f}% {r['avg_loss']:>9.1f}% "
              f"{r['ev']:>+9.3f}% {r['ret']:>9.1f}% {r['mdd']:>7.1f}% {ev_ok}")

        if r['ev'] > best_ev and r['trades'] >= 20:
            best_ev = r['ev']
            best_pattern = pc

    if best_pattern:
        print(f"\n최고 패턴: {best_pattern['name']} (EV: {best_ev:+.3f}%)")

    # ============================================================
    # 2. 손절/손익비 테스트
    # ============================================================
    print("\n" + "=" * 90)
    print("[2] 손절/손익비 테스트")
    print("=" * 90)

    if best_pattern:
        sl_tp_combos = [
            (0.03, 2.0), (0.03, 3.0), (0.03, 4.0),
            (0.05, 2.0), (0.05, 3.0), (0.05, 4.0),
            (0.07, 2.0), (0.07, 3.0), (0.07, 4.0),
            (0.10, 2.0), (0.10, 3.0), (0.10, 4.0),
        ]

        print(f"\n{'SL/TP':<12} {'거래':>6} {'승률':>8} {'평균승':>10} {'평균손':>10} {'EV':>10} {'수익률':>10} {'MDD':>8}")
        print("-" * 90)

        best_combo = (0.05, 3.0)
        best_combo_ev = -999

        for sl_pct, tp_mult in sl_tp_combos:
            r = backtest_surge(
                {k: v.copy() for k, v in all_data.items()},
                {'pattern': best_pattern['cfg'], 'trade': {'sl_pct': sl_pct, 'tp_mult': tp_mult}}
            )
            ev_ok = "✅" if r['ev'] > 0 else "❌"
            print(f"{sl_pct*100:.0f}%/1:{tp_mult:<5} {r['trades']:>6} {r['wr']:>7.1f}% {r['avg_win']:>9.1f}% {r['avg_loss']:>9.1f}% "
                  f"{r['ev']:>+9.3f}% {r['ret']:>9.1f}% {r['mdd']:>7.1f}% {ev_ok}")

            if r['ev'] > best_combo_ev:
                best_combo_ev = r['ev']
                best_combo = (sl_pct, tp_mult)

        best_sl, best_tp = best_combo
        print(f"\n최고 설정: SL {best_sl*100:.0f}%, TP 1:{best_tp} (EV: {best_combo_ev:+.3f}%)")

    # ============================================================
    # 3. 최종 결과
    # ============================================================
    print("\n" + "=" * 90)
    print("[3] 최종 결과")
    print("=" * 90)

    if best_pattern:
        final_r = backtest_surge(
            {k: v.copy() for k, v in all_data.items()},
            {'pattern': best_pattern['cfg'], 'trade': {'sl_pct': best_sl, 'tp_mult': best_tp}}
        )

        print(f"\n최적 설정:")
        print(f"  패턴: {best_pattern['name']}")
        print(f"  - 급락 탐색: {best_pattern['cfg']['lookback']}봉")
        print(f"  - 급락 기준: {best_pattern['cfg']['drop_pct']}% 이상 하락")
        print(f"  - 1차 반등: {best_pattern['cfg']['bounce_pct']}% 이상 상승")
        print(f"  - 1차 조정: {best_pattern['cfg']['pullback_pct']}% 이상 조정")
        print(f"  손절: {best_sl*100:.0f}%")
        print(f"  손익비: 1:{best_tp}")

        print(f"\n결과:")
        print(f"  거래 수: {final_r['trades']}회")
        print(f"  승률: {final_r['wr']}%")
        print(f"  평균 승리: +{final_r['avg_win']:.1f}%")
        print(f"  평균 손실: -{final_r['avg_loss']:.1f}%")
        print(f"  EV/거래: {final_r['ev']:+.3f}%")
        print(f"  수익률: {final_r['ret']:+.1f}%")
        print(f"  MDD: {final_r['mdd']:.1f}%")
        print(f"  PF: {final_r['pf']:.2f}")
        print(f"  총 수수료: ₩{final_r['fees']:,.0f}")

        if final_r['ev'] > 0:
            print(f"\n✅ 양의 EV! 전략 유효")
        else:
            print(f"\n❌ 음의 EV. 추가 최적화 필요")

        # 최근 거래 출력
        if final_r['trades_list']:
            print(f"\n[최근 10개 거래]")
            print("-" * 100)
            print(f"{'코인':<10} {'진입일':>12} {'진입가':>12} {'청산가':>12} {'수익률':>10} {'사유':>8}")
            print("-" * 100)
            for t in sorted(final_r['trades_list'], key=lambda x: x['exit_time'], reverse=True)[:10]:
                print(f"{t['symbol']:<10} {str(t['entry_time'])[:10]:>12} {t['entry']:>12.4f} "
                      f"{t['exit']:>12.4f} {t['pnl_after_fee']:>+9.1f}% {t['reason']:>8}")
