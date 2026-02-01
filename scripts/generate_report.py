"""
RSI Divergence 전략 상세 백테스트 보고서 생성
HTML 파일로 차트, 통계, 거래 내역 포함
"""

import sys
sys.path.insert(0, '/home/hyeon/project/ichimoku')

import json
from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np
from collections import defaultdict

from scripts.data_cache import load_cached_data

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

def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

# ============================================================
# 전략
# ============================================================
def apply_strategy(df):
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
# 상세 백테스트
# ============================================================
def detailed_backtest(all_data: Dict[str, pd.DataFrame], config: dict):
    cfg = {
        'initial': 5_000_000,
        'leverage': 10,
        'pos_pct': 0.12,
        'atr_sl': 0.7,
        'atr_tp': 2.0,
        'max_pos': 4,
        'cooldown': 2,
        **config
    }

    # 전략 적용
    for sym in all_data:
        all_data[sym] = apply_strategy(all_data[sym])

    # 바 정렬
    bars = []
    for sym, df in all_data.items():
        df = df.dropna()
        for _, row in df.iterrows():
            bars.append({'symbol': sym, **row.to_dict()})
    bars.sort(key=lambda x: x['timestamp'])

    # 시간 그룹
    tg = {}
    for b in bars:
        t = b['timestamp']
        if t not in tg:
            tg[t] = {}
        tg[t][b['symbol']] = b

    times = sorted(tg.keys())

    # 시뮬레이션
    cash = cfg['initial']
    positions = {}
    trades = []
    equity_curve = []
    last_exit = {}
    daily_pnl = defaultdict(float)

    for t in times:
        current_bars = tg[t]
        closed = []

        # 청산
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
                pnl = ((exit_p - entry) / entry if pos['side'] == 'long' else (entry - exit_p) / entry) * 100
                realized = pnl * cfg['leverage'] / 100 * pos['size']
                cash += pos['size'] + realized
                daily_pnl[t.date()] += realized

                trades.append({
                    'symbol': sym,
                    'side': pos['side'],
                    'entry_time': pos['entry_time'].isoformat(),
                    'exit_time': t.isoformat(),
                    'entry_price': round(entry, 6),
                    'exit_price': round(exit_p, 6),
                    'sl': round(pos['sl'], 6),
                    'tp': round(pos['tp'], 6),
                    'pnl_pct': round(pnl * cfg['leverage'], 2),
                    'pnl_krw': round(realized, 0),
                    'reason': reason,
                    'size': round(pos['size'], 0)
                })
                closed.append(sym)
                last_exit[sym] = t

        for s in closed:
            del positions[s]

        # 자산
        unreal = sum(
            ((current_bars[s]['close'] - p['entry']) / p['entry'] if p['side'] == 'long'
             else (p['entry'] - current_bars[s]['close']) / p['entry']) * cfg['leverage'] * p['size'] / 100
            for s, p in positions.items() if s in current_bars
        )
        eq = cash + sum(p['size'] for p in positions.values()) + unreal
        pos_size = eq * cfg['pos_pct']

        # 진입
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

    # 일별 수익 계산
    daily_returns = []
    for date, pnl in sorted(daily_pnl.items()):
        daily_returns.append({
            'date': date.isoformat(),
            'pnl': round(pnl, 0),
            'pnl_pct': round(pnl / cfg['initial'] * 100, 2)
        })

    # 월별 수익
    monthly_pnl = defaultdict(float)
    for date, pnl in daily_pnl.items():
        month_key = date.strftime('%Y-%m')
        monthly_pnl[month_key] += pnl

    monthly_returns = []
    for month, pnl in sorted(monthly_pnl.items()):
        monthly_returns.append({
            'month': month,
            'pnl': round(pnl, 0),
            'pnl_pct': round(pnl / cfg['initial'] * 100, 2)
        })

    # 통계 계산
    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]

    profit = sum(t['pnl_krw'] for t in wins) if wins else 0
    loss = abs(sum(t['pnl_krw'] for t in losses)) if losses else 0

    peak, max_dd = cfg['initial'], 0
    dd_start, dd_end = None, None
    for e in equity_curve:
        if e['equity'] > peak:
            peak = e['equity']
            dd_start = e['time']
        dd = (peak - e['equity']) / peak * 100
        if dd > max_dd:
            max_dd = dd
            dd_end = e['time']

    final = equity_curve[-1]['equity'] if equity_curve else cfg['initial']
    days = len(daily_pnl)

    daily_rets_pct = [d['pnl_pct'] for d in daily_returns]
    big_days = [d for d in daily_returns if d['pnl_pct'] >= 10]

    # 코인별 통계
    coin_stats = {}
    for coin in all_data.keys():
        coin_trades = [t for t in trades if t['symbol'] == coin]
        if coin_trades:
            coin_wins = [t for t in coin_trades if t['pnl_pct'] > 0]
            coin_stats[coin] = {
                'trades': len(coin_trades),
                'win_rate': round(len(coin_wins) / len(coin_trades) * 100, 1),
                'total_pnl': round(sum(t['pnl_krw'] for t in coin_trades), 0),
                'avg_pnl': round(np.mean([t['pnl_pct'] for t in coin_trades]), 2)
            }

    # 롱/숏 통계
    longs = [t for t in trades if t['side'] == 'long']
    shorts = [t for t in trades if t['side'] == 'short']
    long_wins = [t for t in longs if t['pnl_pct'] > 0]
    short_wins = [t for t in shorts if t['pnl_pct'] > 0]

    stats = {
        'initial_capital': cfg['initial'],
        'final_capital': final,
        'total_return': round((final - cfg['initial']) / cfg['initial'] * 100, 2),
        'total_pnl': round(sum(t['pnl_krw'] for t in trades), 0),
        'total_trades': len(trades),
        'win_rate': round(len(wins) / len(trades) * 100, 1) if trades else 0,
        'avg_win': round(np.mean([t['pnl_pct'] for t in wins]), 2) if wins else 0,
        'avg_loss': round(np.mean([t['pnl_pct'] for t in losses]), 2) if losses else 0,
        'max_win': round(max(t['pnl_pct'] for t in trades), 2) if trades else 0,
        'max_loss': round(min(t['pnl_pct'] for t in trades), 2) if trades else 0,
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
    }

    # 차트용 가격 데이터 (샘플링)
    price_data = {}
    for sym, df in all_data.items():
        df = df.reset_index(drop=True)
        # 1시간 단위로 샘플링 (15분봉 4개당 1개)
        sampled = df.iloc[::4].copy()
        price_data[sym] = [
            {
                'time': row['timestamp'].isoformat(),
                'timestamp': int(row['timestamp'].timestamp() * 1000),
                'open': round(row['open'], 6),
                'high': round(row['high'], 6),
                'low': round(row['low'], 6),
                'close': round(row['close'], 6),
            }
            for _, row in sampled.iterrows()
        ]

    return {
        'stats': stats,
        'trades': trades,
        'equity_curve': equity_curve[::4],  # 샘플링
        'daily_returns': daily_returns,
        'monthly_returns': monthly_returns,
        'coin_stats': coin_stats,
        'big_days': big_days,
        'price_data': price_data,
        'config': {
            'coins': list(all_data.keys()),
            'leverage': cfg['leverage'],
            'position_pct': cfg['pos_pct'],
            'atr_sl': cfg['atr_sl'],
            'atr_tp': cfg['atr_tp'],
            'strategy': 'RSI Divergence (LB10, DIV3)'
        }
    }


def generate_html(data: dict) -> str:
    """HTML 보고서 생성"""

    stats = data['stats']
    config = data['config']

    html = f'''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RSI Divergence 백테스트 보고서</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0a;
            color: #e0e0e0;
            line-height: 1.6;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}

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

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}

        .stat-card {{
            background: #1a1a2e;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border: 1px solid #333;
        }}
        .stat-card .label {{ color: #888; font-size: 12px; text-transform: uppercase; }}
        .stat-card .value {{ font-size: 28px; font-weight: bold; margin-top: 5px; }}
        .stat-card .value.positive {{ color: #00d4aa; }}
        .stat-card .value.negative {{ color: #ff4757; }}

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
        tr:hover {{ background: #1a1a2e; }}

        .positive {{ color: #00d4aa; }}
        .negative {{ color: #ff4757; }}

        .trade-long {{ background: rgba(0, 212, 170, 0.1); }}
        .trade-short {{ background: rgba(255, 71, 87, 0.1); }}

        .config-box {{
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        .config-item {{ }}
        .config-item .label {{ color: #888; font-size: 11px; }}
        .config-item .value {{ color: #00d4aa; font-size: 16px; font-weight: bold; }}

        .tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }}
        .tab {{
            padding: 10px 20px;
            background: #1a1a2e;
            border: 1px solid #333;
            border-radius: 5px;
            cursor: pointer;
            color: #888;
        }}
        .tab.active {{ background: #00d4aa; color: #000; border-color: #00d4aa; }}

        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}

        .summary-row {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 20px;
        }}

        @media (max-width: 768px) {{
            .summary-row {{ grid-template-columns: 1fr; }}
            .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 <span>RSI Divergence</span> 백테스트 보고서</h1>

        <div class="config-box">
            <div class="config-item">
                <div class="label">전략</div>
                <div class="value">{config['strategy']}</div>
            </div>
            <div class="config-item">
                <div class="label">코인</div>
                <div class="value">{', '.join(config['coins'])}</div>
            </div>
            <div class="config-item">
                <div class="label">레버리지</div>
                <div class="value">{config['leverage']}x</div>
            </div>
            <div class="config-item">
                <div class="label">포지션 비율</div>
                <div class="value">{config['position_pct']*100:.0f}%</div>
            </div>
            <div class="config-item">
                <div class="label">손익비 (ATR)</div>
                <div class="value">{config['atr_sl']} : {config['atr_tp']}</div>
            </div>
        </div>

        <h2>📈 핵심 지표</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="label">총 수익률</div>
                <div class="value {'positive' if stats['total_return'] > 0 else 'negative'}">
                    {stats['total_return']:+.1f}%
                </div>
            </div>
            <div class="stat-card">
                <div class="label">총 수익</div>
                <div class="value {'positive' if stats['total_pnl'] > 0 else 'negative'}">
                    ₩{stats['total_pnl']:,.0f}
                </div>
            </div>
            <div class="stat-card">
                <div class="label">최종 자산</div>
                <div class="value">₩{stats['final_capital']:,.0f}</div>
            </div>
            <div class="stat-card">
                <div class="label">승률</div>
                <div class="value">{stats['win_rate']:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="label">총 거래</div>
                <div class="value">{stats['total_trades']:,}회</div>
            </div>
            <div class="stat-card">
                <div class="label">Profit Factor</div>
                <div class="value">{stats['profit_factor']:.2f}</div>
            </div>
            <div class="stat-card">
                <div class="label">최대 낙폭 (MDD)</div>
                <div class="value negative">{stats['max_drawdown']:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="label">일평균 수익</div>
                <div class="value {'positive' if stats['avg_daily_return'] > 0 else 'negative'}">
                    {stats['avg_daily_return']:+.2f}%
                </div>
            </div>
            <div class="stat-card">
                <div class="label">10%+ 수익 일</div>
                <div class="value positive">{stats['big_days_count']}일</div>
            </div>
        </div>

        <h2>💰 에쿼티 커브</h2>
        <div class="chart-container">
            <canvas id="equityChart"></canvas>
        </div>

        <h2>📅 일별 수익</h2>
        <div class="chart-container">
            <canvas id="dailyChart"></canvas>
        </div>

        <h2>📊 월별 수익</h2>
        <div class="chart-container" style="max-height: 300px;">
            <canvas id="monthlyChart"></canvas>
        </div>

        <div class="summary-row">
            <div>
                <h2>🟢 롱 vs 🔴 숏</h2>
                <div class="chart-container">
                    <table>
                        <tr>
                            <th></th>
                            <th>거래 수</th>
                            <th>승률</th>
                            <th>총 수익</th>
                        </tr>
                        <tr class="trade-long">
                            <td>🟢 LONG</td>
                            <td>{stats['long_trades']:,}회</td>
                            <td>{stats['long_win_rate']:.1f}%</td>
                            <td class="{'positive' if stats['long_pnl'] > 0 else 'negative'}">₩{stats['long_pnl']:,.0f}</td>
                        </tr>
                        <tr class="trade-short">
                            <td>🔴 SHORT</td>
                            <td>{stats['short_trades']:,}회</td>
                            <td>{stats['short_win_rate']:.1f}%</td>
                            <td class="{'positive' if stats['short_pnl'] > 0 else 'negative'}">₩{stats['short_pnl']:,.0f}</td>
                        </tr>
                    </table>
                </div>
            </div>
            <div>
                <h2>🪙 코인별 성과</h2>
                <div class="chart-container">
                    <table>
                        <tr>
                            <th>코인</th>
                            <th>거래</th>
                            <th>승률</th>
                            <th>평균</th>
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
                        """ for coin, s in data['coin_stats'].items())}
                    </table>
                </div>
            </div>
        </div>

        <h2>🔥 10%+ 수익 날 ({len(data['big_days'])}일)</h2>
        <div class="chart-container">
            <table>
                <tr>
                    <th>날짜</th>
                    <th>수익</th>
                    <th>수익률</th>
                </tr>
                {''.join(f"""
                <tr>
                    <td>{d['date']}</td>
                    <td class="positive">₩{d['pnl']:,.0f}</td>
                    <td class="positive">{d['pnl_pct']:+.1f}%</td>
                </tr>
                """ for d in sorted(data['big_days'], key=lambda x: x['pnl_pct'], reverse=True)[:20])}
            </table>
        </div>

        <h2>📝 최근 거래 (100개)</h2>
        <div class="chart-container" style="overflow-x: auto;">
            <table>
                <tr>
                    <th>진입</th>
                    <th>청산</th>
                    <th>코인</th>
                    <th>방향</th>
                    <th>진입가</th>
                    <th>청산가</th>
                    <th>수익률</th>
                    <th>수익</th>
                    <th>사유</th>
                </tr>
                {''.join(f"""
                <tr class="{'trade-long' if t['side'] == 'long' else 'trade-short'}">
                    <td>{t['entry_time'][:16]}</td>
                    <td>{t['exit_time'][:16]}</td>
                    <td>{t['symbol'].replace('USDT','')}</td>
                    <td>{'🟢' if t['side'] == 'long' else '🔴'}</td>
                    <td>{t['entry_price']:.4f}</td>
                    <td>{t['exit_price']:.4f}</td>
                    <td class="{'positive' if t['pnl_pct'] > 0 else 'negative'}">{t['pnl_pct']:+.1f}%</td>
                    <td class="{'positive' if t['pnl_krw'] > 0 else 'negative'}">₩{t['pnl_krw']:,.0f}</td>
                    <td>{t['reason']}</td>
                </tr>
                """ for t in data['trades'][-100:][::-1])}
            </table>
        </div>

        <h2>📉 손익 분포</h2>
        <div class="chart-container">
            <canvas id="pnlDistChart"></canvas>
        </div>

    </div>

    <script>
        // 데이터
        const equityData = {json.dumps(data['equity_curve'])};
        const dailyData = {json.dumps(data['daily_returns'])};
        const monthlyData = {json.dumps(data['monthly_returns'])};
        const trades = {json.dumps(data['trades'])};

        // 에쿼티 커브
        new Chart(document.getElementById('equityChart'), {{
            type: 'line',
            data: {{
                labels: equityData.map(d => d.time.slice(0, 10)),
                datasets: [{{
                    label: '자산',
                    data: equityData.map(d => d.equity),
                    borderColor: '#00d4aa',
                    backgroundColor: 'rgba(0, 212, 170, 0.1)',
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0,
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{
                            label: ctx => '₩' + ctx.raw.toLocaleString()
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        display: true,
                        grid: {{ color: '#333' }},
                        ticks: {{ color: '#888', maxTicksLimit: 10 }}
                    }},
                    y: {{
                        grid: {{ color: '#333' }},
                        ticks: {{
                            color: '#888',
                            callback: v => '₩' + (v/1000000).toFixed(1) + 'M'
                        }}
                    }}
                }}
            }}
        }});

        // 일별 수익
        new Chart(document.getElementById('dailyChart'), {{
            type: 'bar',
            data: {{
                labels: dailyData.map(d => d.date),
                datasets: [{{
                    label: '일별 수익률',
                    data: dailyData.map(d => d.pnl_pct),
                    backgroundColor: dailyData.map(d => d.pnl_pct >= 0 ? 'rgba(0, 212, 170, 0.7)' : 'rgba(255, 71, 87, 0.7)'),
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    x: {{
                        display: true,
                        grid: {{ display: false }},
                        ticks: {{ display: false }}
                    }},
                    y: {{
                        grid: {{ color: '#333' }},
                        ticks: {{
                            color: '#888',
                            callback: v => v + '%'
                        }}
                    }}
                }}
            }}
        }});

        // 월별 수익
        new Chart(document.getElementById('monthlyChart'), {{
            type: 'bar',
            data: {{
                labels: monthlyData.map(d => d.month),
                datasets: [{{
                    label: '월별 수익',
                    data: monthlyData.map(d => d.pnl_pct),
                    backgroundColor: monthlyData.map(d => d.pnl_pct >= 0 ? '#00d4aa' : '#ff4757'),
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    x: {{ grid: {{ display: false }}, ticks: {{ color: '#888' }} }},
                    y: {{
                        grid: {{ color: '#333' }},
                        ticks: {{ color: '#888', callback: v => v + '%' }}
                    }}
                }}
            }}
        }});

        // 손익 분포
        const pnlValues = trades.map(t => t.pnl_pct);
        const bins = [];
        for (let i = -50; i <= 50; i += 5) {{
            bins.push({{
                range: i + '~' + (i+5),
                count: pnlValues.filter(v => v >= i && v < i + 5).length
            }});
        }}

        new Chart(document.getElementById('pnlDistChart'), {{
            type: 'bar',
            data: {{
                labels: bins.map(b => b.range + '%'),
                datasets: [{{
                    label: '거래 수',
                    data: bins.map(b => b.count),
                    backgroundColor: bins.map((b, i) => i < 10 ? '#ff4757' : '#00d4aa'),
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    x: {{ grid: {{ display: false }}, ticks: {{ color: '#888' }} }},
                    y: {{ grid: {{ color: '#333' }}, ticks: {{ color: '#888' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>'''

    return html


if __name__ == '__main__':
    print("RSI Divergence 백테스트 보고서 생성 중...")

    # 데이터 로드
    COINS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'HYPEUSDT']
    data = load_cached_data(COINS, '15m')
    print(f"코인: {len(data)}개")

    # 상세 백테스트
    print("백테스트 실행 중...")
    result = detailed_backtest(data, {
        'leverage': 10,
        'pos_pct': 0.12,
        'atr_sl': 0.7,
        'atr_tp': 2.0
    })

    print(f"거래: {result['stats']['total_trades']}회")
    print(f"수익: {result['stats']['total_return']}%")

    # HTML 생성
    print("HTML 보고서 생성 중...")
    html = generate_html(result)

    # 저장
    output_path = '/home/hyeon/project/ichimoku/backtest_report.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n✅ 보고서 생성 완료: {output_path}")
    print(f"\n📊 주요 지표:")
    print(f"   수익률: {result['stats']['total_return']:+.1f}%")
    print(f"   승률: {result['stats']['win_rate']:.1f}%")
    print(f"   MDD: {result['stats']['max_drawdown']:.1f}%")
    print(f"   PF: {result['stats']['profit_factor']:.2f}")
    print(f"   10%+ 날: {result['stats']['big_days_count']}일")
