#!/usr/bin/env python3
"""
백테스트 결과 HTML 리포트 생성기
- 거래 내역 + 차트 + 진입/청산 마커
"""

import sys
sys.path.insert(0, '/home/hyeon/project/ichimoku')

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

from src.surge_strategy import calculate_surge_indicators
import warnings
warnings.filterwarnings('ignore')


def get_exchange():
    return ccxt.bybit({'options': {'defaultType': 'swap'}})


def fetch_ohlcv(exchange, symbol, timeframe='1h', limit=1000):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        return df
    except:
        return None


def prepare_chart_data(df, trades):
    """차트용 데이터 준비"""
    df_calc = calculate_surge_indicators(df.copy())
    df_calc = df_calc.reset_index()

    def clean_series(s):
        return [None if pd.isna(x) else float(x) for x in s]

    # 거래 데이터 정리
    trade_markers = []
    for t in trades:
        trade_markers.append({
            'entry_time': t['entry_time'].isoformat() if hasattr(t['entry_time'], 'isoformat') else str(t['entry_time']),
            'exit_time': t['exit_time'].isoformat() if hasattr(t['exit_time'], 'isoformat') else str(t['exit_time']),
            'entry_price': float(t['entry_price']),
            'exit_price': float(t['exit_price']),
            'pnl_pct': float(t['pnl_pct']),
            'pnl_with_lev': float(t['pnl_with_lev']),
            'reason': t['reason'],
            'score': int(t['score']),
            'is_win': t['pnl_pct'] > 0,
        })

    return {
        'timestamps': [t.isoformat() for t in df_calc['timestamp']],
        'open': clean_series(df_calc['open']),
        'high': clean_series(df_calc['high']),
        'low': clean_series(df_calc['low']),
        'close': clean_series(df_calc['close']),
        'volume': clean_series(df_calc['volume']),
        'rsi': clean_series(df_calc['rsi']),
        'bb_upper': clean_series(df_calc['bb_upper']),
        'bb_lower': clean_series(df_calc['bb_lower']),
        'sma_25': clean_series(df_calc['sma_25']),
        'trades': trade_markers,
    }


def generate_report():
    print("=" * 60)
    print("백테스트 HTML 리포트 생성 중...")
    print("=" * 60)

    # 백테스트 결과 로드
    csv_path = '/home/hyeon/project/ichimoku/data/surge_backtest_results.csv'
    if not os.path.exists(csv_path):
        print(f"❌ 백테스트 결과 파일 없음: {csv_path}")
        return

    trades_df = pd.read_csv(csv_path)
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

    # 통계 계산
    total_trades = len(trades_df)
    wins = len(trades_df[trades_df['pnl_pct'] > 0])
    losses = len(trades_df[trades_df['pnl_pct'] < 0])
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    total_pnl = trades_df['pnl_with_lev'].sum()
    profit_factor = abs(trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].sum() / trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].sum()) if losses > 0 else float('inf')
    avg_hold = trades_df['hold_hours'].mean()

    # 거래가 있는 종목별로 차트 데이터 수집
    exchange = get_exchange()
    traded_symbols = trades_df['symbol'].unique()
    charts_data = {}

    print(f"\n{len(traded_symbols)}개 종목 차트 데이터 수집 중...")

    for symbol in traded_symbols:
        coin = symbol.replace('/USDT:USDT', '')
        print(f"  {coin}...")

        df = fetch_ohlcv(exchange, symbol, '1h', 1000)
        if df is None:
            continue

        symbol_trades = trades_df[trades_df['symbol'] == symbol].to_dict('records')
        charts_data[coin] = prepare_chart_data(df, symbol_trades)

    # 진입 타입별 분석
    entry_type_stats = {}
    if 'entry_reason' in trades_df.columns:
        for reason in trades_df['entry_reason'].dropna().unique():
            subset = trades_df[trades_df['entry_reason'] == reason]
            wins_r = len(subset[subset['pnl_pct'] > 0])
            entry_type_stats[reason] = {
                'count': len(subset),
                'wins': wins_r,
                'rate': wins_r / len(subset) * 100 if len(subset) > 0 else 0,
                'avg': subset['pnl_pct'].mean(),
            }

    # 점수별 분석
    score_stats = {}
    for score in sorted(trades_df['score'].unique()):
        subset = trades_df[trades_df['score'] == score]
        wins_s = len(subset[subset['pnl_pct'] > 0])
        score_stats[int(score)] = {
            'count': len(subset),
            'wins': wins_s,
            'rate': wins_s / len(subset) * 100 if len(subset) > 0 else 0,
            'avg': subset['pnl_pct'].mean(),
        }

    # 청산 사유별 분석
    exit_stats = {}
    for reason in trades_df['reason'].unique():
        subset = trades_df[trades_df['reason'] == reason]
        exit_stats[reason] = {
            'count': len(subset),
            'avg': subset['pnl_pct'].mean(),
        }

    # 누적 수익 계산
    trades_df_sorted = trades_df.sort_values('entry_time')
    cumulative_pnl = trades_df_sorted['pnl_with_lev'].cumsum().tolist()
    cumulative_times = [t.isoformat() for t in trades_df_sorted['entry_time']]

    # 코인별 통계
    coin_stats = trades_df.groupby('symbol').agg({
        'pnl_pct': ['count', 'sum', 'mean'],
        'pnl_with_lev': 'sum'
    }).round(2)
    coin_stats.columns = ['trades', 'total_pnl', 'avg_pnl', 'total_pnl_lev']
    coin_stats = coin_stats.sort_values('total_pnl', ascending=False)

    # HTML 생성
    charts_json = json.dumps(charts_data)

    # 거래 테이블 HTML
    trades_table_html = ""
    for _, t in trades_df.sort_values('entry_time', ascending=False).iterrows():
        coin = t['symbol'].replace('/USDT:USDT', '')
        pnl = t['pnl_with_lev']
        pnl_color = '#00d4aa' if pnl > 0 else '#ff4757'
        emoji = '🟢' if pnl > 0 else '🔴'
        entry_reason = t.get('entry_reason', '-')

        trades_table_html += f"""<tr onclick="showChart('{coin}')" style="cursor:pointer">
            <td>{emoji} <strong>{coin}</strong></td>
            <td>{t['entry_time'].strftime('%m/%d %H:%M')}</td>
            <td>${t['entry_price']:.4f}</td>
            <td>${t['exit_price']:.4f}</td>
            <td style="color:{pnl_color}">{pnl:+.1f}%</td>
            <td>{t['reason']}</td>
            <td>{entry_reason}</td>
            <td>{int(t['score'])}</td>
            <td>{t['hold_hours']:.1f}h</td>
        </tr>"""

    # 진입 타입별 HTML
    entry_type_html = ""
    for reason, stats in entry_type_stats.items():
        rate_color = '#00d4aa' if stats['rate'] >= 60 else '#f39c12' if stats['rate'] >= 40 else '#ff4757'
        entry_type_html += f"""
        <div class="stat-card">
            <div class="stat-label">{reason}</div>
            <div class="stat-value" style="color:{rate_color}">{stats['rate']:.0f}%</div>
            <div class="stat-sub">{stats['wins']}/{stats['count']}건 | {stats['avg']:.2f}%</div>
        </div>"""

    # 점수별 HTML
    score_html = ""
    for score, stats in score_stats.items():
        rate_color = '#00d4aa' if stats['rate'] >= 60 else '#f39c12' if stats['rate'] >= 40 else '#ff4757'
        score_html += f"""
        <div class="stat-card">
            <div class="stat-label">Score {score}</div>
            <div class="stat-value" style="color:{rate_color}">{stats['rate']:.0f}%</div>
            <div class="stat-sub">{stats['wins']}/{stats['count']}건</div>
        </div>"""

    # 청산 사유별 HTML
    exit_html = ""
    for reason, stats in exit_stats.items():
        avg_color = '#00d4aa' if stats['avg'] > 0 else '#ff4757'
        exit_html += f"""
        <div class="stat-card">
            <div class="stat-label">{reason}</div>
            <div class="stat-value" style="color:{avg_color}">{stats['avg']:.2f}%</div>
            <div class="stat-sub">{stats['count']}건</div>
        </div>"""

    # 코인별 HTML
    coin_table_html = ""
    for symbol, row in coin_stats.head(20).iterrows():
        coin = symbol.replace('/USDT:USDT', '')
        pnl_color = '#00d4aa' if row['total_pnl'] > 0 else '#ff4757'
        coin_table_html += f"""<tr onclick="showChart('{coin}')" style="cursor:pointer">
            <td><strong>{coin}</strong></td>
            <td>{int(row['trades'])}</td>
            <td style="color:{pnl_color}">{row['total_pnl']:.2f}%</td>
            <td>{row['avg_pnl']:.2f}%</td>
            <td style="color:{pnl_color}">{row['total_pnl_lev']:.1f}%</td>
        </tr>"""

    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Results - MTF Strategy</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #16213e 100%);
            color: #e0e0e0; min-height: 100vh; padding: 20px;
        }}
        .container {{ max-width: 1800px; margin: 0 auto; }}
        h1 {{ text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
             border-radius: 20px; margin-bottom: 30px; font-size: 2.5em; }}
        h2 {{ color: #667eea; margin: 30px 0 20px; padding-bottom: 10px; border-bottom: 2px solid #667eea; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .summary-card {{ background: rgba(255,255,255,0.05); border-radius: 15px; padding: 25px; text-align: center;
                        border: 1px solid rgba(255,255,255,0.1); }}
        .summary-card h3 {{ color: #888; font-size: 0.85em; margin-bottom: 10px; }}
        .summary-card .value {{ font-size: 2.2em; font-weight: bold; }}
        .value.green {{ color: #00d4aa; }}
        .value.yellow {{ color: #f39c12; }}
        .value.red {{ color: #ff4757; }}
        .section {{ background: rgba(255,255,255,0.03); border-radius: 15px; padding: 25px; margin-bottom: 30px;
                   border: 1px solid rgba(255,255,255,0.08); }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 12px 10px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.08); font-size: 0.9em; }}
        th {{ background: rgba(102, 126, 234, 0.2); color: #fff; font-weight: 600; }}
        tr:hover {{ background: rgba(255,255,255,0.05); }}
        .chart-container {{ background: rgba(0,0,0,0.3); border-radius: 15px; padding: 20px; margin-bottom: 20px; display: none; }}
        .chart-container.active {{ display: block; }}
        .chart-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }}
        .chart-header h3 {{ color: #667eea; font-size: 1.3em; }}
        .close-btn {{ background: #ff4757; color: white; border: none; padding: 8px 16px; border-radius: 8px; cursor: pointer; }}
        .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 15px; }}
        .stat-card {{ background: rgba(255,255,255,0.05); border-radius: 12px; padding: 15px; text-align: center; }}
        .stat-label {{ font-size: 0.9em; color: #888; margin-bottom: 8px; }}
        .stat-value {{ font-size: 1.5em; font-weight: bold; }}
        .stat-sub {{ font-size: 0.8em; color: #666; margin-top: 5px; }}
        .highlight {{ background: linear-gradient(135deg, rgba(0,212,170,0.1) 0%, rgba(0,184,148,0.1) 100%);
                     border: 1px solid rgba(0,212,170,0.3); }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Backtest Results (MTF Strategy)</h1>
        <p style="text-align:center; color:#888; margin-top:-20px; margin-bottom:30px;">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | 1H Signal + 15M Entry Timing
        </p>

        <div class="summary-grid">
            <div class="summary-card highlight">
                <h3>총 수익률 (5x)</h3>
                <div class="value green">{total_pnl:+.1f}%</div>
            </div>
            <div class="summary-card">
                <h3>총 거래</h3>
                <div class="value yellow">{total_trades}</div>
            </div>
            <div class="summary-card">
                <h3>승률</h3>
                <div class="value {'green' if win_rate >= 50 else 'yellow'}">{win_rate:.1f}%</div>
            </div>
            <div class="summary-card">
                <h3>승/패</h3>
                <div class="value"><span style="color:#00d4aa">{wins}W</span> / <span style="color:#ff4757">{losses}L</span></div>
            </div>
            <div class="summary-card">
                <h3>Profit Factor</h3>
                <div class="value {'green' if profit_factor >= 1.5 else 'yellow'}">{profit_factor:.2f}</div>
            </div>
            <div class="summary-card">
                <h3>평균 보유</h3>
                <div class="value yellow">{avg_hold:.1f}h</div>
            </div>
        </div>

        <!-- 누적 수익 차트 -->
        <div class="section">
            <h2>📈 누적 수익률</h2>
            <div id="equity-chart" style="height:300px;"></div>
        </div>

        <!-- 차트 영역 -->
        <div id="chart-area" class="chart-container">
            <div class="chart-header">
                <h3 id="chart-title">차트</h3>
                <button class="close-btn" onclick="hideChart()">닫기</button>
            </div>
            <div id="chart-plot" style="height:500px;"></div>
        </div>

        <div class="section">
            <h2>🎯 진입 타입별 성과</h2>
            <div class="stat-grid">
                {entry_type_html}
            </div>
        </div>

        <div class="section">
            <h2>📊 점수별 성과</h2>
            <div class="stat-grid">
                {score_html}
            </div>
        </div>

        <div class="section">
            <h2>🚪 청산 사유별 분석</h2>
            <div class="stat-grid">
                {exit_html}
            </div>
        </div>

        <div class="section">
            <h2>🏆 코인별 성과 (Top 20)</h2>
            <table>
                <tr><th>코인</th><th>거래수</th><th>총 수익</th><th>평균</th><th>레버리지</th></tr>
                {coin_table_html}
            </table>
        </div>

        <div class="section">
            <h2>📋 전체 거래 내역 ({total_trades}건)</h2>
            <table>
                <tr><th>코인</th><th>진입시간</th><th>진입가</th><th>청산가</th><th>PnL (5x)</th><th>청산사유</th><th>진입타입</th><th>Score</th><th>보유</th></tr>
                {trades_table_html}
            </table>
        </div>
    </div>

    <script>
        const chartsData = {charts_json};
        const cumulativePnl = {json.dumps(cumulative_pnl)};
        const cumulativeTimes = {json.dumps(cumulative_times)};

        // 누적 수익 차트
        Plotly.newPlot('equity-chart', [{{
            x: cumulativeTimes,
            y: cumulativePnl,
            type: 'scatter',
            fill: 'tozeroy',
            fillcolor: 'rgba(0, 212, 170, 0.2)',
            line: {{ color: '#00d4aa', width: 2 }},
            name: 'Cumulative PnL'
        }}], {{
            template: 'plotly_dark',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: {{ l: 50, r: 30, t: 20, b: 40 }},
            xaxis: {{ showgrid: true, gridcolor: 'rgba(255,255,255,0.1)' }},
            yaxis: {{ showgrid: true, gridcolor: 'rgba(255,255,255,0.1)', title: 'PnL (%)' }},
        }}, {{ responsive: true }});

        function showChart(coin) {{
            const data = chartsData[coin];
            if (!data) {{
                alert('차트 데이터 없음: ' + coin);
                return;
            }}

            const container = document.getElementById('chart-area');
            const plotDiv = document.getElementById('chart-plot');
            document.getElementById('chart-title').textContent = coin + ' - Trade History';

            const winTrades = data.trades.filter(t => t.is_win);
            const loseTrades = data.trades.filter(t => !t.is_win);

            const traces = [
                // 캔들스틱
                {{
                    x: data.timestamps,
                    open: data.open,
                    high: data.high,
                    low: data.low,
                    close: data.close,
                    type: 'candlestick',
                    name: 'Price',
                    increasing: {{ line: {{ color: '#26a69a' }} }},
                    decreasing: {{ line: {{ color: '#ef5350' }} }},
                    yaxis: 'y'
                }},
                // BB
                {{ x: data.timestamps, y: data.bb_upper, type: 'scatter', name: 'BB Upper',
                   line: {{ color: 'rgba(255,255,255,0.2)', width: 1 }}, yaxis: 'y' }},
                {{ x: data.timestamps, y: data.bb_lower, type: 'scatter', name: 'BB Lower',
                   line: {{ color: 'rgba(255,255,255,0.2)', width: 1 }}, fill: 'tonexty',
                   fillcolor: 'rgba(255,255,255,0.05)', yaxis: 'y' }},
                // SMA25
                {{ x: data.timestamps, y: data.sma_25, type: 'scatter', name: 'SMA25',
                   line: {{ color: '#f39c12', width: 1.5 }}, yaxis: 'y' }},
                // 승리 진입
                {{ x: winTrades.map(t => t.entry_time), y: winTrades.map(t => t.entry_price),
                   type: 'scatter', mode: 'markers', name: 'Win Entry',
                   marker: {{ symbol: 'triangle-up', size: 14, color: '#00d4aa' }}, yaxis: 'y',
                   hovertemplate: 'Entry: $%{{y:.4f}}<br>PnL: %{{customdata:.1f}}%<extra></extra>',
                   customdata: winTrades.map(t => t.pnl_with_lev) }},
                // 승리 청산
                {{ x: winTrades.map(t => t.exit_time), y: winTrades.map(t => t.exit_price),
                   type: 'scatter', mode: 'markers', name: 'Win Exit',
                   marker: {{ symbol: 'star', size: 12, color: '#00d4aa' }}, yaxis: 'y' }},
                // 패배 진입
                {{ x: loseTrades.map(t => t.entry_time), y: loseTrades.map(t => t.entry_price),
                   type: 'scatter', mode: 'markers', name: 'Lose Entry',
                   marker: {{ symbol: 'triangle-up', size: 14, color: '#ff4757' }}, yaxis: 'y',
                   hovertemplate: 'Entry: $%{{y:.4f}}<br>PnL: %{{customdata:.1f}}%<extra></extra>',
                   customdata: loseTrades.map(t => t.pnl_with_lev) }},
                // 패배 청산
                {{ x: loseTrades.map(t => t.exit_time), y: loseTrades.map(t => t.exit_price),
                   type: 'scatter', mode: 'markers', name: 'Lose Exit',
                   marker: {{ symbol: 'x', size: 12, color: '#ff4757' }}, yaxis: 'y' }},
                // RSI
                {{ x: data.timestamps, y: data.rsi, type: 'scatter', name: 'RSI',
                   line: {{ color: '#9b59b6', width: 1.5 }}, yaxis: 'y2' }},
            ];

            Plotly.newPlot(plotDiv, traces, {{
                template: 'plotly_dark',
                height: 500,
                showlegend: true,
                legend: {{ orientation: 'h', y: 1.02 }},
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                margin: {{ l: 60, r: 60, t: 40, b: 40 }},
                xaxis: {{ showgrid: true, gridcolor: 'rgba(255,255,255,0.1)', rangeslider: {{ visible: false }} }},
                yaxis: {{ title: 'Price', showgrid: true, gridcolor: 'rgba(255,255,255,0.1)', domain: [0.25, 1] }},
                yaxis2: {{ title: 'RSI', showgrid: true, gridcolor: 'rgba(255,255,255,0.1)', domain: [0, 0.2], range: [0, 100] }},
            }}, {{ responsive: true }});

            container.classList.add('active');
            container.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
        }}

        function hideChart() {{
            document.getElementById('chart-area').classList.remove('active');
        }}
    </script>
</body>
</html>
"""

    # 파일 저장
    output_path = '/home/hyeon/project/ichimoku/data/backtest_report.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print("\n" + "=" * 60)
    print("✅ 리포트 생성 완료!")
    print("=" * 60)
    print(f"  파일: {output_path}")
    print(f"  거래: {total_trades}건")
    print(f"  승률: {win_rate:.1f}%")
    print(f"  총 수익: {total_pnl:+.1f}%")
    print(f"\n  file://{output_path}")


if __name__ == "__main__":
    generate_report()
