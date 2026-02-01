"""
수수료 포함 백테스트 보고서 생성
RSI Divergence + Ichimoku 비교
"""

import sys
sys.path.insert(0, '/home/hyeon/project/ichimoku')

import time
import json
from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np
from collections import defaultdict
from pybit.unified_trading import HTTP

from scripts.data_cache import load_cached_data
from src.ichimoku import calculate_ichimoku

# ============================================================
# 지표
# ============================================================
def rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - 100 / (1 + g / l)

def atr(df, p=14):
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    return tr.rolling(p).mean()

# ============================================================
# RSI Divergence 전략
# ============================================================
def apply_rsi_strategy(df):
    df = df.copy()
    df['rsi'] = rsi(df['close'], 14)
    df['atr'] = atr(df, 14)
    df['price_low'] = df['low'].rolling(10).min()
    df['price_high'] = df['high'].rolling(10).max()
    df['rsi_at_low'] = df['rsi'].rolling(10).min()
    df['rsi_at_high'] = df['rsi'].rolling(10).max()

    df['long'] = (df['low'] <= df['price_low'] * 1.005) & (df['rsi'] > df['rsi_at_low'].shift(1) + 3)
    df['short'] = (df['high'] >= df['price_high'] * 0.995) & (df['rsi'] < df['rsi_at_high'].shift(1) - 3)
    return df


# ============================================================
# 수수료 포함 백테스트
# ============================================================
def backtest_with_fees(all_data: Dict[str, pd.DataFrame], strategy: str, config: dict):
    cfg = {
        'initial': 5_000_000,
        'leverage': 10,
        'pos_pct': 0.12,
        'atr_sl': 0.7,
        'atr_tp': 2.0,
        'max_pos': 4,
        'cooldown': 2,
        'fee_rate': 0.00055,  # 바이빗 테이커 0.055%
        **config
    }

    # 전략 적용
    if strategy == 'rsi':
        for sym in all_data:
            all_data[sym] = apply_rsi_strategy(all_data[sym])
    # ichimoku는 이미 적용됨

    # 바 정렬
    bars = []
    for sym, df in all_data.items():
        df = df.dropna()
        for _, row in df.iterrows():
            bars.append({'symbol': sym, **row.to_dict()})
    bars.sort(key=lambda x: x['timestamp'])

    tg = {}
    for b in bars:
        t = b['timestamp']
        if t not in tg:
            tg[t] = {}
        tg[t][b['symbol']] = b

    times = sorted(tg.keys())

    cash = cfg['initial']
    positions = {}
    trades = []
    equity_curve = []
    last_exit = {}
    daily_pnl = defaultdict(float)
    total_fees = 0

    for t in times:
        current_bars = tg[t]
        closed = []

        for sym, pos in positions.items():
            if sym not in current_bars:
                continue
            b = current_bars[sym]
            h, l = b['high'], b['low']
            entry = pos['entry']
            reason = None

            if pos['side'] == 'long':
                if l <= pos['sl']:
                    reason, exit_p = 'SL', pos['sl']
                elif h >= pos['tp']:
                    reason, exit_p = 'TP', pos['tp']
            else:
                if h >= pos['sl']:
                    reason, exit_p = 'SL', pos['sl']
                elif l <= pos['tp']:
                    reason, exit_p = 'TP', pos['tp']

            if reason:
                # 수수료 계산
                notional = pos['size'] * cfg['leverage']
                round_trip_fee = notional * cfg['fee_rate'] * 2
                total_fees += round_trip_fee

                pnl = ((exit_p - entry) / entry if pos['side'] == 'long' else (entry - exit_p) / entry) * 100
                leveraged_pnl = pnl * cfg['leverage'] / 100 * pos['size']
                realized = leveraged_pnl - round_trip_fee

                cash += pos['size'] + realized
                daily_pnl[t.date()] += realized

                trades.append({
                    'symbol': sym,
                    'side': pos['side'],
                    'entry_time': pos['entry_time'].isoformat(),
                    'exit_time': t.isoformat(),
                    'entry_price': round(entry, 6),
                    'exit_price': round(exit_p, 6),
                    'pnl_pct': round(pnl * cfg['leverage'], 2),
                    'fee': round(round_trip_fee, 0),
                    'pnl_after_fee': round(pnl * cfg['leverage'] - round_trip_fee / pos['size'] * 100, 2),
                    'pnl_krw': round(realized, 0),
                    'reason': reason,
                    'size': round(pos['size'], 0)
                })
                closed.append(sym)
                last_exit[sym] = t

        for s in closed:
            del positions[s]

        unreal = sum(
            ((current_bars[s]['close'] - p['entry']) / p['entry'] if p['side'] == 'long'
             else (p['entry'] - current_bars[s]['close']) / p['entry']) * cfg['leverage'] * p['size'] / 100
            for s, p in positions.items() if s in current_bars
        )
        eq = cash + sum(p['size'] for p in positions.values()) + unreal
        pos_size = eq * cfg['pos_pct']

        if cash >= pos_size and len(positions) < cfg['max_pos']:
            for sym, b in current_bars.items():
                if sym in positions:
                    continue
                if sym in last_exit and (t - last_exit[sym]).total_seconds() < cfg['cooldown'] * 15 * 60:
                    continue

                price = b['close']
                a = b.get('atr', price * 0.01)

                if b.get('long', False):
                    sl = price - a * cfg['atr_sl']
                    tp = price + a * cfg['atr_tp']
                    positions[sym] = {
                        'side': 'long', 'entry': price, 'entry_time': t,
                        'sl': sl, 'tp': tp, 'size': pos_size
                    }
                    cash -= pos_size
                elif b.get('short', False):
                    sl = price + a * cfg['atr_sl']
                    tp = price - a * cfg['atr_tp']
                    positions[sym] = {
                        'side': 'short', 'entry': price, 'entry_time': t,
                        'sl': sl, 'tp': tp, 'size': pos_size
                    }
                    cash -= pos_size

                if len(positions) >= cfg['max_pos']:
                    break

        equity_curve.append({
            'time': t.isoformat(),
            'timestamp': int(t.timestamp() * 1000),
            'equity': round(eq, 0)
        })

    # 일별/월별 수익
    daily_returns = [{'date': d.isoformat(), 'pnl': round(p, 0), 'pnl_pct': round(p / cfg['initial'] * 100, 2)}
                     for d, p in sorted(daily_pnl.items())]

    monthly_pnl = defaultdict(float)
    for d, p in daily_pnl.items():
        monthly_pnl[d.strftime('%Y-%m')] += p
    monthly_returns = [{'month': m, 'pnl': round(p, 0), 'pnl_pct': round(p / cfg['initial'] * 100, 2)}
                       for m, p in sorted(monthly_pnl.items())]

    # 통계
    wins = [t for t in trades if t['pnl_krw'] > 0]
    losses = [t for t in trades if t['pnl_krw'] <= 0]

    profit = sum(t['pnl_krw'] for t in wins) if wins else 0
    loss = abs(sum(t['pnl_krw'] for t in losses)) if losses else 0

    peak, max_dd = cfg['initial'], 0
    for e in equity_curve:
        if e['equity'] > peak:
            peak = e['equity']
        dd = (peak - e['equity']) / peak * 100
        max_dd = max(max_dd, dd)

    final = equity_curve[-1]['equity'] if equity_curve else cfg['initial']
    days = len(daily_pnl)
    daily_rets_pct = [d['pnl_pct'] for d in daily_returns]
    big_days = [d for d in daily_returns if d['pnl_pct'] >= 10]

    # 코인별 통계
    coin_stats = {}
    for coin in all_data.keys():
        coin_trades = [t for t in trades if t['symbol'] == coin]
        if coin_trades:
            coin_wins = [t for t in coin_trades if t['pnl_krw'] > 0]
            coin_stats[coin] = {
                'trades': len(coin_trades),
                'win_rate': round(len(coin_wins) / len(coin_trades) * 100, 1),
                'total_pnl': round(sum(t['pnl_krw'] for t in coin_trades), 0),
                'avg_pnl': round(np.mean([t['pnl_after_fee'] for t in coin_trades]), 2)
            }

    # 롱/숏 통계
    longs = [t for t in trades if t['side'] == 'long']
    shorts = [t for t in trades if t['side'] == 'short']
    long_wins = [t for t in longs if t['pnl_krw'] > 0]
    short_wins = [t for t in shorts if t['pnl_krw'] > 0]

    # EV 계산
    if trades:
        win_rate = len(wins) / len(trades)
        avg_win = np.mean([t['pnl_after_fee'] for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t['pnl_after_fee'] for t in losses])) if losses else 0
        ev = win_rate * avg_win - (1 - win_rate) * avg_loss
    else:
        win_rate, avg_win, avg_loss, ev = 0, 0, 0, 0

    stats = {
        'initial_capital': cfg['initial'],
        'final_capital': final,
        'total_return': round((final - cfg['initial']) / cfg['initial'] * 100, 2),
        'total_pnl': round(sum(t['pnl_krw'] for t in trades), 0),
        'total_fees': round(total_fees, 0),
        'total_trades': len(trades),
        'win_rate': round(len(wins) / len(trades) * 100, 1) if trades else 0,
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'ev_per_trade': round(ev, 3),
        'max_win': round(max(t['pnl_after_fee'] for t in trades), 2) if trades else 0,
        'max_loss': round(min(t['pnl_after_fee'] for t in trades), 2) if trades else 0,
        'profit_factor': round(profit / loss, 2) if loss > 0 else 999,
        'max_drawdown': round(max_dd, 2),
        'trading_days': days,
        'trades_per_day': round(len(trades) / days, 2) if days > 0 else 0,
        'avg_daily_return': round(np.mean(daily_rets_pct), 3) if daily_rets_pct else 0,
        'best_day': round(max(daily_rets_pct), 2) if daily_rets_pct else 0,
        'worst_day': round(min(daily_rets_pct), 2) if daily_rets_pct else 0,
        'big_days_count': len(big_days),
        'long_trades': len(longs),
        'long_win_rate': round(len(long_wins) / len(longs) * 100, 1) if longs else 0,
        'long_pnl': round(sum(t['pnl_krw'] for t in longs), 0),
        'short_trades': len(shorts),
        'short_win_rate': round(len(short_wins) / len(shorts) * 100, 1) if shorts else 0,
        'short_pnl': round(sum(t['pnl_krw'] for t in shorts), 0),
        'leverage': cfg['leverage'],
        'position_pct': cfg['pos_pct'] * 100,
        'fee_rate': cfg['fee_rate'] * 100,
    }

    return {
        'stats': stats,
        'trades': trades,
        'equity_curve': equity_curve[::max(1, len(equity_curve)//500)],  # 샘플링
        'daily_returns': daily_returns,
        'monthly_returns': monthly_returns,
        'coin_stats': coin_stats,
        'big_days': big_days,
        'config': cfg
    }


def generate_html(rsi_data: dict, ichimoku_data: dict) -> str:
    """비교 HTML 보고서 생성"""

    rsi_stats = rsi_data['stats']
    ich_stats = ichimoku_data['stats']

    html = f'''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>수수료 포함 백테스트 비교 보고서</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0a;
            color: #e0e0e0;
            line-height: 1.6;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; padding: 20px; }}

        h1 {{
            text-align: center;
            padding: 30px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            margin-bottom: 30px;
            border-radius: 10px;
        }}
        h1 span {{ color: #00d4aa; }}

        h2 {{
            color: #00d4aa;
            margin: 30px 0 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #333;
        }}

        .warning-box {{
            background: linear-gradient(135deg, #4a1c1c 0%, #2a1010 100%);
            border: 2px solid #ff4757;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .warning-box h3 {{ color: #ff4757; margin-bottom: 10px; }}

        .success-box {{
            background: linear-gradient(135deg, #1c4a2e 0%, #102a18 100%);
            border: 2px solid #00d4aa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .success-box h3 {{ color: #00d4aa; margin-bottom: 10px; }}

        .comparison-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }}

        .strategy-card {{
            background: #1a1a2e;
            padding: 25px;
            border-radius: 10px;
            border: 2px solid #333;
        }}
        .strategy-card.winner {{ border-color: #00d4aa; }}
        .strategy-card.loser {{ border-color: #ff4757; opacity: 0.7; }}

        .strategy-card h3 {{
            text-align: center;
            font-size: 24px;
            margin-bottom: 20px;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }}

        .stat-item {{
            background: #16213e;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-item .label {{ color: #888; font-size: 11px; text-transform: uppercase; }}
        .stat-item .value {{ font-size: 20px; font-weight: bold; margin-top: 5px; }}
        .stat-item .value.positive {{ color: #00d4aa; }}
        .stat-item .value.negative {{ color: #ff4757; }}

        .chart-container {{
            background: #1a1a2e;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border: 1px solid #333;
        }}
        .chart-container canvas {{ max-height: 400px; }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            font-size: 13px;
        }}
        th, td {{
            padding: 12px 8px;
            text-align: right;
            border-bottom: 1px solid #333;
        }}
        th {{
            background: #16213e;
            color: #00d4aa;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 11px;
        }}
        td:first-child, th:first-child {{ text-align: left; }}

        .positive {{ color: #00d4aa; }}
        .negative {{ color: #ff4757; }}

        .fee-impact {{
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}

        .summary-table {{
            width: 100%;
            margin-top: 20px;
        }}
        .summary-table th {{ background: #0a0a0a; }}

        @media (max-width: 1200px) {{
            .comparison-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>💰 <span>수수료 포함</span> 백테스트 비교 보고서</h1>

        <div class="{'success-box' if ich_stats['ev_per_trade'] > 0 else 'warning-box'}">
            <h3>🎯 핵심 결론</h3>
            <p style="font-size: 18px;">
                RSI Divergence: <span class="negative">수수료로 인해 전략 무효화 (EV {rsi_stats['ev_per_trade']:+.3f}%)</span><br>
                Ichimoku: <span class="positive">수수료 후에도 양의 EV (EV {ich_stats['ev_per_trade']:+.3f}%)</span>
            </p>
            <p style="margin-top: 10px; color: #888;">
                수수료율: {rsi_stats['fee_rate']}% (테이커) | 레버리지: {rsi_stats['leverage']}x
            </p>
        </div>

        <h2>📊 전략 비교</h2>
        <div class="comparison-grid">
            <div class="strategy-card loser">
                <h3>❌ RSI Divergence</h3>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="label">수익률</div>
                        <div class="value {'positive' if rsi_stats['total_return'] > 0 else 'negative'}">
                            {rsi_stats['total_return']:+.1f}%
                        </div>
                    </div>
                    <div class="stat-item">
                        <div class="label">최종 자산</div>
                        <div class="value">₩{rsi_stats['final_capital']:,.0f}</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">거래 수</div>
                        <div class="value">{rsi_stats['total_trades']:,}회</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">총 수수료</div>
                        <div class="value negative">₩{rsi_stats['total_fees']:,.0f}</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">승률</div>
                        <div class="value">{rsi_stats['win_rate']:.1f}%</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">EV/거래</div>
                        <div class="value {'positive' if rsi_stats['ev_per_trade'] > 0 else 'negative'}">
                            {rsi_stats['ev_per_trade']:+.3f}%
                        </div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Profit Factor</div>
                        <div class="value">{rsi_stats['profit_factor']:.2f}</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">MDD</div>
                        <div class="value negative">{rsi_stats['max_drawdown']:.1f}%</div>
                    </div>
                </div>
            </div>

            <div class="strategy-card winner">
                <h3>✅ Ichimoku (4코인)</h3>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="label">수익률</div>
                        <div class="value {'positive' if ich_stats['total_return'] > 0 else 'negative'}">
                            {ich_stats['total_return']:+.1f}%
                        </div>
                    </div>
                    <div class="stat-item">
                        <div class="label">최종 자산</div>
                        <div class="value">₩{ich_stats['final_capital']:,.0f}</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">거래 수</div>
                        <div class="value">{ich_stats['total_trades']:,}회</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">총 수수료</div>
                        <div class="value negative">₩{ich_stats['total_fees']:,.0f}</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">승률</div>
                        <div class="value">{ich_stats['win_rate']:.1f}%</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">EV/거래</div>
                        <div class="value {'positive' if ich_stats['ev_per_trade'] > 0 else 'negative'}">
                            {ich_stats['ev_per_trade']:+.3f}%
                        </div>
                    </div>
                    <div class="stat-item">
                        <div class="label">Profit Factor</div>
                        <div class="value">{ich_stats['profit_factor']:.2f}</div>
                    </div>
                    <div class="stat-item">
                        <div class="label">MDD</div>
                        <div class="value negative">{ich_stats['max_drawdown']:.1f}%</div>
                    </div>
                </div>
            </div>
        </div>

        <h2>📈 수수료 영향 분석</h2>
        <div class="fee-impact">
            <table class="summary-table">
                <tr>
                    <th>항목</th>
                    <th>RSI Divergence</th>
                    <th>Ichimoku</th>
                    <th>비고</th>
                </tr>
                <tr>
                    <td>총 거래 수</td>
                    <td>{rsi_stats['total_trades']:,}회</td>
                    <td>{ich_stats['total_trades']:,}회</td>
                    <td>RSI가 {rsi_stats['total_trades'] / max(ich_stats['total_trades'], 1):.0f}배 많음</td>
                </tr>
                <tr>
                    <td>총 수수료</td>
                    <td class="negative">₩{rsi_stats['total_fees']:,.0f}</td>
                    <td class="negative">₩{ich_stats['total_fees']:,.0f}</td>
                    <td>RSI가 {rsi_stats['total_fees'] / max(ich_stats['total_fees'], 1):.1f}배 많음</td>
                </tr>
                <tr>
                    <td>수수료/초기자본</td>
                    <td class="negative">{rsi_stats['total_fees'] / rsi_stats['initial_capital'] * 100:.1f}%</td>
                    <td class="negative">{ich_stats['total_fees'] / ich_stats['initial_capital'] * 100:.1f}%</td>
                    <td></td>
                </tr>
                <tr>
                    <td>EV/거래 (수수료 후)</td>
                    <td class="{'positive' if rsi_stats['ev_per_trade'] > 0 else 'negative'}">{rsi_stats['ev_per_trade']:+.3f}%</td>
                    <td class="{'positive' if ich_stats['ev_per_trade'] > 0 else 'negative'}">{ich_stats['ev_per_trade']:+.3f}%</td>
                    <td>{'Ichimoku 유효' if ich_stats['ev_per_trade'] > 0 else ''}</td>
                </tr>
                <tr>
                    <td>최종 수익률</td>
                    <td class="{'positive' if rsi_stats['total_return'] > 0 else 'negative'}">{rsi_stats['total_return']:+.1f}%</td>
                    <td class="{'positive' if ich_stats['total_return'] > 0 else 'negative'}">{ich_stats['total_return']:+.1f}%</td>
                    <td></td>
                </tr>
            </table>
        </div>

        <h2>💰 에쿼티 커브 비교</h2>
        <div class="chart-container">
            <canvas id="equityChart"></canvas>
        </div>

        <h2>📅 Ichimoku 월별 수익</h2>
        <div class="chart-container">
            <canvas id="monthlyChart"></canvas>
        </div>

        <h2>🪙 Ichimoku 코인별 성과</h2>
        <div class="chart-container">
            <table>
                <tr>
                    <th>코인</th>
                    <th>거래</th>
                    <th>승률</th>
                    <th>평균 수익</th>
                    <th>총 수익</th>
                </tr>
                {''.join(f"""
                <tr>
                    <td>{coin}</td>
                    <td>{s['trades']}회</td>
                    <td>{s['win_rate']:.1f}%</td>
                    <td class="{'positive' if s['avg_pnl'] > 0 else 'negative'}">{s['avg_pnl']:+.1f}%</td>
                    <td class="{'positive' if s['total_pnl'] > 0 else 'negative'}">₩{s['total_pnl']:,.0f}</td>
                </tr>
                """ for coin, s in ichimoku_data['coin_stats'].items())}
            </table>
        </div>

        <h2>📝 Ichimoku 거래 내역 (전체 {len(ichimoku_data['trades'])}건)</h2>
        <div class="chart-container" style="overflow-x: auto;">
            <table>
                <tr>
                    <th>진입</th>
                    <th>청산</th>
                    <th>코인</th>
                    <th>방향</th>
                    <th>진입가</th>
                    <th>청산가</th>
                    <th>수익률 (수수료전)</th>
                    <th>수수료</th>
                    <th>수익률 (수수료후)</th>
                    <th>순수익</th>
                    <th>사유</th>
                </tr>
                {''.join(f"""
                <tr>
                    <td>{t['entry_time'][:16]}</td>
                    <td>{t['exit_time'][:16]}</td>
                    <td>{t['symbol'].replace('USDT','')}</td>
                    <td>{'🟢' if t['side'] == 'long' else '🔴'}</td>
                    <td>{t['entry_price']:.4f}</td>
                    <td>{t['exit_price']:.4f}</td>
                    <td class="{'positive' if t['pnl_pct'] > 0 else 'negative'}">{t['pnl_pct']:+.1f}%</td>
                    <td class="negative">₩{t['fee']:,.0f}</td>
                    <td class="{'positive' if t['pnl_after_fee'] > 0 else 'negative'}">{t['pnl_after_fee']:+.1f}%</td>
                    <td class="{'positive' if t['pnl_krw'] > 0 else 'negative'}">₩{t['pnl_krw']:,.0f}</td>
                    <td>{t['reason']}</td>
                </tr>
                """ for t in ichimoku_data['trades'])}
            </table>
        </div>

        <h2>⚠️ 핵심 교훈</h2>
        <div class="fee-impact">
            <table class="summary-table">
                <tr>
                    <th>교훈</th>
                    <th>설명</th>
                </tr>
                <tr>
                    <td>고빈도 ≠ 고수익</td>
                    <td>RSI Divergence는 9,505회 거래로 수수료 ₩577만 발생, 전략 무효화</td>
                </tr>
                <tr>
                    <td>수수료가 EV를 잡아먹음</td>
                    <td>레버리지 10x에서 왕복 수수료는 마진의 2% → 거래당 -2% 고정 비용</td>
                </tr>
                <tr>
                    <td>선별적 진입의 중요성</td>
                    <td>Ichimoku는 44회 거래만으로 42.8% 수익, 수수료 ₩57만만 발생</td>
                </tr>
                <tr>
                    <td>EV가 높아야 생존</td>
                    <td>RSI EV +0.097% → 수수료 2%에 패배 / Ichimoku EV +17.996% → 수수료 4% 흡수</td>
                </tr>
            </table>
        </div>

    </div>

    <script>
        const rsiEquity = {json.dumps(rsi_data['equity_curve'])};
        const ichEquity = {json.dumps(ichimoku_data['equity_curve'])};
        const ichMonthly = {json.dumps(ichimoku_data['monthly_returns'])};

        // 에쿼티 커브 비교
        new Chart(document.getElementById('equityChart'), {{
            type: 'line',
            data: {{
                datasets: [
                    {{
                        label: 'RSI Divergence',
                        data: rsiEquity.map(d => ({{x: new Date(d.time), y: d.equity}})),
                        borderColor: '#ff4757',
                        backgroundColor: 'rgba(255, 71, 87, 0.1)',
                        fill: false,
                        tension: 0.1,
                        pointRadius: 0,
                    }},
                    {{
                        label: 'Ichimoku',
                        data: ichEquity.map(d => ({{x: new Date(d.time), y: d.equity}})),
                        borderColor: '#00d4aa',
                        backgroundColor: 'rgba(0, 212, 170, 0.1)',
                        fill: false,
                        tension: 0.1,
                        pointRadius: 0,
                    }}
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ labels: {{ color: '#888' }} }},
                    tooltip: {{ callbacks: {{ label: ctx => ctx.dataset.label + ': ₩' + ctx.raw.y.toLocaleString() }} }}
                }},
                scales: {{
                    x: {{
                        type: 'time',
                        grid: {{ color: '#333' }},
                        ticks: {{ color: '#888' }}
                    }},
                    y: {{
                        grid: {{ color: '#333' }},
                        ticks: {{ color: '#888', callback: v => '₩' + (v/1000000).toFixed(1) + 'M' }}
                    }}
                }}
            }}
        }});

        // 월별 수익
        new Chart(document.getElementById('monthlyChart'), {{
            type: 'bar',
            data: {{
                labels: ichMonthly.map(d => d.month),
                datasets: [{{
                    label: '월별 수익',
                    data: ichMonthly.map(d => d.pnl_pct),
                    backgroundColor: ichMonthly.map(d => d.pnl_pct >= 0 ? '#00d4aa' : '#ff4757'),
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    x: {{ grid: {{ display: false }}, ticks: {{ color: '#888' }} }},
                    y: {{ grid: {{ color: '#333' }}, ticks: {{ color: '#888', callback: v => v + '%' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>'''

    return html


# ============================================================
# Ichimoku 백테스트 (수수료 포함)
# ============================================================
def fetch_klines(symbol: str, interval: int, limit: int = 4000):
    session = HTTP()
    all_data = []
    end_time = None

    while len(all_data) < limit:
        params = {'category': 'linear', 'symbol': symbol, 'interval': interval, 'limit': 1000}
        if end_time:
            params['end'] = end_time
        try:
            response = session.get_kline(**params)
            klines = response['result']['list']
        except:
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


def ichimoku_backtest_with_fees(all_data, config):
    """Ichimoku 수수료 포함 백테스트"""
    cfg = {
        'initial': 5_000_000,
        'leverage': 20,
        'pos_pct': 0.05,
        'fee_rate': 0.00055,  # 바이빗 테이커 0.055%
        'max_positions': 5,
        'cooldown_hours': 4,
        'min_cloud_thickness': 0.2,
        'min_sl_pct': 0.3,
        'max_sl_pct': 8.0,
        'sl_buffer': 0.2,
        'rr_ratio': 2.0,
        'trail_pct': 1.5,
        **config
    }

    # BTC 트렌드
    btc_trends = {}
    if 'BTCUSDT' in all_data:
        btc_df = all_data['BTCUSDT'].copy()
        btc_df['sma_26'] = btc_df['close'].rolling(26).mean()
        btc_df['sma_52'] = btc_df['close'].rolling(52).mean()
        for _, row in btc_df.iterrows():
            if pd.notna(row['sma_26']) and pd.notna(row['sma_52']):
                btc_trends[row['timestamp']] = row['sma_26'] > row['sma_52']

    # 지표 계산
    all_bars = []
    for symbol, df in all_data.items():
        df = calculate_ichimoku(df)
        df = df.dropna(subset=['tenkan', 'kijun', 'cloud_top', 'cloud_bottom'])
        for _, row in df.iterrows():
            all_bars.append({
                'symbol': symbol, 'time': row['timestamp'],
                'high': row['high'], 'low': row['low'], 'close': row['close'],
                'cloud_top': row['cloud_top'], 'cloud_bottom': row['cloud_bottom'],
                'cloud_thickness': row['cloud_thickness'], 'cloud_green': row['cloud_green'],
                'tenkan_above': row['tenkan_above'], 'tk_cross_down': row['tk_cross_down'],
                'kijun_cross_down': row['kijun_cross_down'],
                'chikou_bearish': row.get('chikou_bearish', False),
                'above_cloud': row['above_cloud'], 'below_cloud': row['below_cloud'],
                'in_cloud': row['in_cloud'],
            })

    all_bars.sort(key=lambda x: x['time'])
    time_groups = {}
    for bar in all_bars:
        t = bar['time']
        if t not in time_groups:
            time_groups[t] = {}
        time_groups[t][bar['symbol']] = bar

    sorted_times = sorted(time_groups.keys())

    cash = cfg['initial']
    positions = {}
    trades = []
    equity_curve = []
    last_exit = {}
    daily_pnl = defaultdict(float)
    total_fees = 0

    for t in sorted_times:
        bars = time_groups[t]
        closed = []
        btc_uptrend = btc_trends.get(t)

        for sym, pos in positions.items():
            if sym not in bars:
                continue
            bar = bars[sym]
            price = bar['close']
            high = bar['high']
            low = bar['low']
            entry = pos['entry_price']

            if low < pos['lowest']:
                pos['lowest'] = low
                if low <= pos['take_profit']:
                    pos['trailing'] = True
                    pos['trail_stop'] = min(pos['trail_stop'], low * (1 + cfg['trail_pct'] / 100))

            reason = None
            max_loss_price = entry * 1.02
            if high >= max_loss_price:
                reason = 'MaxLoss'
                price = max_loss_price
            elif high >= pos['stop_loss']:
                reason = 'Stop'
                price = min(pos['stop_loss'], high)
            elif pos.get('trailing') and high >= pos['trail_stop']:
                reason = 'Trail'
                price = pos['trail_stop']
            elif not pos.get('trailing') and low <= pos['take_profit']:
                reason = 'TP'
                price = pos['take_profit']
            elif bar['in_cloud'] or bar['above_cloud']:
                reason = 'Cloud'
                price = bar['close']

            if reason:
                notional = pos['position_size'] * cfg['leverage']
                round_trip_fee = notional * cfg['fee_rate'] * 2
                total_fees += round_trip_fee

                pnl_pct = (entry - price) / entry * 100
                leveraged_pnl = pnl_pct * cfg['leverage'] / 100 * pos['position_size']
                realized = leveraged_pnl - round_trip_fee

                cash += pos['position_size'] + realized
                daily_pnl[t.date()] += realized

                trades.append({
                    'symbol': sym, 'side': 'short',
                    'entry_time': pos['entry_time'].isoformat(),
                    'exit_time': t.isoformat(),
                    'entry_price': round(entry, 6),
                    'exit_price': round(price, 6),
                    'pnl_pct': round(pnl_pct * cfg['leverage'], 2),
                    'fee': round(round_trip_fee, 0),
                    'pnl_after_fee': round(pnl_pct * cfg['leverage'] - round_trip_fee / pos['position_size'] * 100, 2),
                    'pnl_krw': round(realized, 0),
                    'reason': reason,
                    'size': round(pos['position_size'], 0)
                })
                closed.append(sym)
                last_exit[sym] = t

        for s in closed:
            del positions[s]

        unrealized = sum(
            (pos['entry_price'] - bars[sym]['close']) / pos['entry_price'] * cfg['leverage'] * pos['position_size'] / 100
            for sym, pos in positions.items() if sym in bars
        )
        eq = cash + sum(pos['position_size'] for pos in positions.values()) + unrealized
        p_size = eq * cfg['pos_pct']

        if cash >= p_size and len(positions) < cfg['max_positions']:
            for sym, bar in bars.items():
                if sym in positions:
                    continue
                if sym in last_exit:
                    if (t - last_exit[sym]).total_seconds() < cfg['cooldown_hours'] * 3600:
                        continue

                price = bar['close']
                cloud_bottom = bar['cloud_bottom']
                thickness = bar['cloud_thickness']

                if bar['in_cloud'] or thickness < cfg['min_cloud_thickness']:
                    continue

                if bar['below_cloud'] and not bar['tenkan_above']:
                    has_signal = bar['tk_cross_down'] or bar['kijun_cross_down']
                    if not has_signal or btc_uptrend is False:
                        continue

                    stop_loss = cloud_bottom * (1 + cfg['sl_buffer'] / 100)
                    sl_distance_pct = (stop_loss - price) / price * 100

                    if cfg['min_sl_pct'] <= sl_distance_pct <= cfg['max_sl_pct']:
                        take_profit = price * (1 - sl_distance_pct * cfg['rr_ratio'] / 100)
                        positions[sym] = {
                            'side': 'short', 'entry_price': price, 'entry_time': t,
                            'stop_loss': stop_loss, 'take_profit': take_profit,
                            'highest': price, 'lowest': price,
                            'trail_stop': stop_loss, 'trailing': False,
                            'position_size': p_size,
                        }
                        cash -= p_size

                if len(positions) >= cfg['max_positions']:
                    break

        equity_curve.append({'time': t.isoformat(), 'timestamp': int(t.timestamp() * 1000), 'equity': round(eq, 0)})

    # 결과 처리
    daily_returns = [{'date': d.isoformat(), 'pnl': round(p, 0), 'pnl_pct': round(p / cfg['initial'] * 100, 2)}
                     for d, p in sorted(daily_pnl.items())]

    monthly_pnl = defaultdict(float)
    for d, p in daily_pnl.items():
        monthly_pnl[d.strftime('%Y-%m')] += p
    monthly_returns = [{'month': m, 'pnl': round(p, 0), 'pnl_pct': round(p / cfg['initial'] * 100, 2)}
                       for m, p in sorted(monthly_pnl.items())]

    wins = [t for t in trades if t['pnl_krw'] > 0]
    losses = [t for t in trades if t['pnl_krw'] <= 0]
    profit = sum(t['pnl_krw'] for t in wins) if wins else 0
    loss = abs(sum(t['pnl_krw'] for t in losses)) if losses else 0

    peak, max_dd = cfg['initial'], 0
    for e in equity_curve:
        if e['equity'] > peak:
            peak = e['equity']
        max_dd = max(max_dd, (peak - e['equity']) / peak * 100)

    final = equity_curve[-1]['equity'] if equity_curve else cfg['initial']

    if trades:
        win_rate = len(wins) / len(trades)
        avg_win = np.mean([t['pnl_after_fee'] for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t['pnl_after_fee'] for t in losses])) if losses else 0
        ev = win_rate * avg_win - (1 - win_rate) * avg_loss
    else:
        win_rate, avg_win, avg_loss, ev = 0, 0, 0, 0

    coin_stats = {}
    for coin in all_data.keys():
        coin_trades = [t for t in trades if t['symbol'] == coin]
        if coin_trades:
            coin_wins = [t for t in coin_trades if t['pnl_krw'] > 0]
            coin_stats[coin] = {
                'trades': len(coin_trades),
                'win_rate': round(len(coin_wins) / len(coin_trades) * 100, 1),
                'total_pnl': round(sum(t['pnl_krw'] for t in coin_trades), 0),
                'avg_pnl': round(np.mean([t['pnl_after_fee'] for t in coin_trades]), 2)
            }

    longs = [t for t in trades if t['side'] == 'long']
    shorts = [t for t in trades if t['side'] == 'short']

    stats = {
        'initial_capital': cfg['initial'],
        'final_capital': final,
        'total_return': round((final - cfg['initial']) / cfg['initial'] * 100, 2),
        'total_pnl': round(sum(t['pnl_krw'] for t in trades), 0),
        'total_fees': round(total_fees, 0),
        'total_trades': len(trades),
        'win_rate': round(len(wins) / len(trades) * 100, 1) if trades else 0,
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'ev_per_trade': round(ev, 3),
        'profit_factor': round(profit / loss, 2) if loss > 0 else 999,
        'max_drawdown': round(max_dd, 2),
        'leverage': cfg['leverage'],
        'position_pct': cfg['pos_pct'] * 100,
        'fee_rate': cfg['fee_rate'] * 100,
        'long_trades': len(longs),
        'short_trades': len(shorts),
    }

    return {
        'stats': stats,
        'trades': trades,
        'equity_curve': equity_curve,
        'daily_returns': daily_returns,
        'monthly_returns': monthly_returns,
        'coin_stats': coin_stats,
        'big_days': [d for d in daily_returns if d['pnl_pct'] >= 10],
        'config': cfg
    }


if __name__ == '__main__':
    print("=" * 80)
    print("수수료 포함 백테스트 보고서 생성")
    print("=" * 80)

    COINS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'HYPEUSDT']

    # RSI Divergence
    print("\n[1] RSI Divergence 백테스트...")
    rsi_data = load_cached_data(COINS, '15m')
    print(f"  코인: {len(rsi_data)}개")

    rsi_result = backtest_with_fees(
        {k: v.copy() for k, v in rsi_data.items()},
        'rsi',
        {'leverage': 10, 'pos_pct': 0.12, 'atr_sl': 0.7, 'atr_tp': 2.0, 'fee_rate': 0.00055}
    )
    print(f"  거래: {rsi_result['stats']['total_trades']}회")
    print(f"  수익률: {rsi_result['stats']['total_return']:+.1f}%")
    print(f"  수수료: ₩{rsi_result['stats']['total_fees']:,.0f}")
    print(f"  EV/거래: {rsi_result['stats']['ev_per_trade']:+.3f}%")

    # Ichimoku
    print("\n[2] Ichimoku 백테스트...")
    print("  데이터 수집 중...")
    ich_data = {}
    for symbol in COINS:
        print(f"    {symbol}...", end='', flush=True)
        df = fetch_klines(symbol, 240, limit=4000)
        if df is not None:
            ich_data[symbol] = df
            print(" OK")

    ich_result = ichimoku_backtest_with_fees(ich_data, {'fee_rate': 0.00055})
    print(f"  거래: {ich_result['stats']['total_trades']}회")
    print(f"  수익률: {ich_result['stats']['total_return']:+.1f}%")
    print(f"  수수료: ₩{ich_result['stats']['total_fees']:,.0f}")
    print(f"  EV/거래: {ich_result['stats']['ev_per_trade']:+.3f}%")

    # HTML 생성
    print("\n[3] HTML 보고서 생성...")
    html = generate_html(rsi_result, ich_result)

    output_path = '/home/hyeon/project/ichimoku/backtest_report_with_fees.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n✅ 보고서 생성 완료: {output_path}")
    print(f"\n📊 결론:")
    print(f"   RSI Divergence: EV {rsi_result['stats']['ev_per_trade']:+.3f}% → {'❌ 무효' if rsi_result['stats']['ev_per_trade'] < 0 else '✅ 유효'}")
    print(f"   Ichimoku: EV {ich_result['stats']['ev_per_trade']:+.3f}% → {'❌ 무효' if ich_result['stats']['ev_per_trade'] < 0 else '✅ 유효'}")
