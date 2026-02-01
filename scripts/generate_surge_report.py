#!/usr/bin/env python3
"""
급등 신호 분석 HTML 리포트 생성기 (v2 - Lazy Loading)
"""

import sys
sys.path.insert(0, '/home/hyeon/project/ichimoku')

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

from src.surge_strategy import (
    calculate_surge_indicators,
    get_surge_entry_signal,
    SURGE_STRATEGY_PARAMS,
    get_surge_watch_list,
)
import warnings
warnings.filterwarnings('ignore')


def get_exchange():
    return ccxt.bybit({'options': {'defaultType': 'swap'}})


def fetch_ohlcv(exchange, symbol, timeframe='4h', limit=500):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        return df
    except:
        return None


def find_signals_in_data(symbol, df, params):
    """데이터에서 모든 신호 찾기 + 이후 성과 측정"""
    df_calc = calculate_surge_indicators(df.copy())
    signals = []

    for i in range(30, len(df_calc) - 10):
        prev_rows = df_calc.iloc[:i+1]
        signal = get_surge_entry_signal(symbol, prev_rows, params)

        if signal:
            entry_price = signal['price']
            future_data = df_calc.iloc[i+1:i+11]

            if len(future_data) > 0:
                max_gain = ((future_data['high'].max() / entry_price) - 1) * 100
                max_dd = ((future_data['low'].min() / entry_price) - 1) * 100
            else:
                max_gain = 0
                max_dd = 0

            signals.append({
                'idx': i,
                'time': df_calc.index[i].isoformat(),
                'price': float(entry_price),
                'score': int(signal['score']),
                'rsi': float(signal['rsi']),
                'bb_position': float(signal['bb_position']),
                'volume_ratio': float(signal['volume_ratio']),
                'max_gain': float(max_gain),
                'max_dd': float(max_dd),
                'success': bool(max_gain >= 12),
            })

    return signals


def prepare_chart_data(symbol, df, signals):
    """차트용 데이터 준비 (JSON 직렬화 가능)"""
    df_calc = calculate_surge_indicators(df.copy())
    df_calc = df_calc.reset_index()

    # NaN을 None으로 변환
    def clean_series(s):
        return [None if pd.isna(x) else float(x) for x in s]

    return {
        'timestamps': [t.isoformat() for t in df_calc['timestamp']],
        'close': clean_series(df_calc['close']),
        'open': clean_series(df_calc['open']),
        'high': clean_series(df_calc['high']),
        'low': clean_series(df_calc['low']),
        'volume': clean_series(df_calc['volume']),
        'volume_sma': clean_series(df_calc['volume_sma']),
        'rsi': clean_series(df_calc['rsi']),
        'bb_upper': clean_series(df_calc['bb_upper']),
        'bb_lower': clean_series(df_calc['bb_lower']),
        'sma_25': clean_series(df_calc['sma_25']),
        'signals': signals,
    }


def generate_report():
    print("=" * 60)
    print("HTML 리포트 생성 중...")
    print("=" * 60)

    exchange = get_exchange()
    params = SURGE_STRATEGY_PARAMS.copy()

    test_symbols = get_surge_watch_list()

    all_data = []
    charts_data = {}  # 차트 데이터를 딕셔너리로 저장
    current_signals = []
    all_historical_signals = []

    for symbol in test_symbols:
        coin = symbol.replace('/USDT:USDT', '')
        print(f"  {coin} 분석 중...")

        df = fetch_ohlcv(exchange, symbol, '1h', 200)
        if df is None or len(df) < 50:
            continue

        df_calc = calculate_surge_indicators(df.copy())
        row = df_calc.iloc[-1]

        # 현재 상태
        price = float(row['close'])
        rsi = float(row['rsi']) if pd.notna(row['rsi']) else 50
        bb_position = float(row['bb_position']) if pd.notna(row['bb_position']) else 0.5
        volume_ratio = float(row['volume_ratio']) if pd.notna(row['volume_ratio']) else 1
        change_24h = ((price / df_calc.iloc[-24]['close']) - 1) * 100 if len(df_calc) >= 24 else 0

        # 현재 신호
        current_signal = get_surge_entry_signal(symbol, df_calc, params)
        has_signal = current_signal is not None

        if has_signal:
            current_signals.append({
                'symbol': symbol,
                'price': current_signal['price'],
                'score': current_signal['score'],
                'rsi': current_signal['rsi'],
                'bb_position': current_signal['bb_position'],
                'volume_ratio': current_signal['volume_ratio'],
                'sl': current_signal['stop_loss'],
                'tp': current_signal['take_profit'],
            })

        # 과거 신호 분석
        signals = find_signals_in_data(symbol, df, params)
        signal_count = len(signals)
        success_count = len([s for s in signals if s['success']])
        success_rate = success_count / signal_count * 100 if signal_count > 0 else 0

        all_historical_signals.extend(signals)

        all_data.append({
            'symbol': symbol,
            'coin': coin,
            'price': price,
            'change_24h': change_24h,
            'rsi': rsi,
            'bb_position': bb_position,
            'volume_ratio': volume_ratio,
            'signal_count': signal_count,
            'success_count': success_count,
            'success_rate': success_rate,
            'has_signal': has_signal,
        })

        # 시그널이 있는 종목만 차트 데이터 저장
        if signal_count > 0:
            charts_data[coin] = prepare_chart_data(symbol, df, signals)

    # 통계
    total_signals = len(all_historical_signals)
    total_success = len([s for s in all_historical_signals if s['success']])
    overall_success_rate = total_success / total_signals * 100 if total_signals > 0 else 0

    # 점수별 통계
    score_stats = {}
    for score in range(7, 18):
        score_signals = [s for s in all_historical_signals if s['score'] == score]
        if score_signals:
            score_success = len([s for s in score_signals if s['success']])
            score_stats[score] = {
                'count': len(score_signals),
                'success': score_success,
                'rate': score_success / len(score_signals) * 100,
                'avg_gain': np.mean([s['max_gain'] for s in score_signals]),
            }

    # HTML 생성
    current_signals_html = ""
    if current_signals:
        rows = ""
        for s in current_signals:
            coin = s['symbol'].replace('/USDT:USDT', '')
            rows += f"""<tr onclick="showChart('{coin}')" style="cursor:pointer">
                <td><strong style="color:#f39c12">{coin}</strong></td>
                <td>${s['price']:.4f}</td>
                <td><span class="signal-badge signal-active">{s['score']}</span></td>
                <td style="color:#9b59b6">{s['rsi']:.1f}</td>
                <td>{s['bb_position']:.2f}</td>
                <td style="color:#f39c12">{s['volume_ratio']:.1f}x</td>
                <td style="color:#ff4757">${s['sl']:.4f}</td>
                <td style="color:#00d4aa">${s['tp']:.4f}</td>
            </tr>"""
        current_signals_html = f"""
        <div class="highlight-box">
            <p style="margin-bottom:15px; color:#00d4aa;">
                <strong>{len(current_signals)}개 코인에서 진입 신호 발생! (클릭하면 차트 표시)</strong>
            </p>
            <table>
                <tr><th>코인</th><th>가격</th><th>Score</th><th>RSI</th><th>BB Pos</th><th>Volume</th><th>SL</th><th>TP</th></tr>
                {rows}
            </table>
        </div>"""
    else:
        current_signals_html = """<div class="highlight-box warning">
            <p style="color:#888;">현재 진입 조건을 충족하는 코인이 없습니다.</p>
        </div>"""

    # 점수별 통계 HTML
    score_html = ""
    for score, stats in sorted(score_stats.items()):
        rate_color = '#00d4aa' if stats['rate'] >= 50 else '#f39c12' if stats['rate'] >= 30 else '#ff4757'
        score_html += f"""
        <div class="accuracy-card">
            <div class="score">Score {score}</div>
            <div class="rate" style="color:{rate_color}">{stats['rate']:.0f}%</div>
            <div class="count">{stats['success']}/{stats['count']}건 | +{stats['avg_gain']:.1f}%</div>
        </div>"""

    # 코인 테이블 HTML (시그널 있는 것만)
    coin_table_html = ""
    signal_coins = [d for d in all_data if d['signal_count'] > 0]
    for d in sorted(signal_coins, key=lambda x: (-x['has_signal'], -x['success_rate'], x['rsi'])):
        coin = d['coin']
        change_color = '#00d4aa' if d['change_24h'] >= 0 else '#ff4757'
        rate_color = '#00d4aa' if d['success_rate'] >= 50 else '#f39c12' if d['success_rate'] >= 30 else '#888'
        badge_class = 'signal-active' if d['has_signal'] else 'signal-none'
        badge_text = 'ENTRY!' if d['has_signal'] else 'wait'

        coin_table_html += f"""<tr onclick="showChart('{coin}')" style="cursor:pointer">
            <td><strong>{coin}</strong></td>
            <td>${d['price']:.4f}</td>
            <td style="color:{change_color}">{d['change_24h']:+.1f}%</td>
            <td>{d['rsi']:.1f}</td>
            <td>{d['bb_position']:.2f}</td>
            <td>{d['volume_ratio']:.1f}x</td>
            <td>{d['signal_count']}</td>
            <td style="color:{rate_color}">{d['success_rate']:.0f}%</td>
            <td><span class="signal-badge {badge_class}">{badge_text}</span></td>
        </tr>"""

    # 차트 데이터를 JSON으로 변환
    charts_json = json.dumps(charts_data)

    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Surge Signal Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #16213e 100%);
            color: #e0e0e0; min-height: 100vh; padding: 20px;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        h1 {{ text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
             border-radius: 20px; margin-bottom: 30px; font-size: 2.5em; }}
        h2 {{ color: #667eea; margin: 30px 0 20px; padding-bottom: 10px; border-bottom: 2px solid #667eea; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .summary-card {{ background: rgba(255,255,255,0.05); border-radius: 15px; padding: 25px; text-align: center;
                        border: 1px solid rgba(255,255,255,0.1); }}
        .summary-card h3 {{ color: #888; font-size: 0.85em; margin-bottom: 10px; }}
        .summary-card .value {{ font-size: 2.5em; font-weight: bold; }}
        .value.green {{ color: #00d4aa; }}
        .value.yellow {{ color: #f39c12; }}
        .value.purple {{ color: #9b59b6; }}
        .section {{ background: rgba(255,255,255,0.03); border-radius: 15px; padding: 25px; margin-bottom: 30px;
                   border: 1px solid rgba(255,255,255,0.08); }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 14px 12px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.08); }}
        th {{ background: rgba(102, 126, 234, 0.2); color: #fff; font-weight: 600; font-size: 0.85em; }}
        tr:hover {{ background: rgba(255,255,255,0.05); }}
        .signal-badge {{ display: inline-block; padding: 6px 14px; border-radius: 20px; font-size: 0.85em; font-weight: bold; }}
        .signal-active {{ background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%); color: #fff; }}
        .signal-none {{ background: rgba(255,255,255,0.1); color: #888; }}
        .chart-container {{ background: rgba(0,0,0,0.3); border-radius: 15px; padding: 20px; margin-bottom: 20px; display: none; }}
        .chart-container.active {{ display: block; }}
        .chart-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; flex-wrap: wrap; gap: 10px; }}
        .chart-header h3 {{ color: #667eea; font-size: 1.3em; }}
        .chart-stats {{ display: flex; gap: 15px; }}
        .chart-stat {{ background: rgba(255,255,255,0.08); padding: 8px 15px; border-radius: 10px; font-size: 0.9em; }}
        .chart-stat.success {{ background: rgba(0, 212, 170, 0.2); color: #00d4aa; }}
        .accuracy-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 15px; margin-top: 20px; }}
        .accuracy-card {{ background: rgba(255,255,255,0.05); border-radius: 12px; padding: 15px; text-align: center; }}
        .accuracy-card .score {{ font-size: 1.5em; font-weight: bold; color: #667eea; }}
        .accuracy-card .rate {{ font-size: 1.2em; margin-top: 5px; }}
        .accuracy-card .count {{ font-size: 0.8em; color: #888; margin-top: 5px; }}
        .highlight-box {{ background: rgba(0, 212, 170, 0.1); border: 1px solid rgba(0, 212, 170, 0.3);
                         border-radius: 15px; padding: 20px; margin-bottom: 20px; }}
        .highlight-box.warning {{ background: rgba(255, 71, 87, 0.1); border-color: rgba(255, 71, 87, 0.3); }}
        .close-btn {{ background: #ff4757; color: white; border: none; padding: 8px 16px; border-radius: 8px; cursor: pointer; }}
        .close-btn:hover {{ background: #ff6b7a; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 Surge Signal Analysis</h1>
        <p style="text-align:center; color:#888; margin-top:-20px; margin-bottom:30px;">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | 테이블 행 클릭 시 차트 표시
        </p>

        <div class="summary-grid">
            <div class="summary-card">
                <h3>분석 종목</h3>
                <div class="value purple">{len(all_data)}</div>
            </div>
            <div class="summary-card">
                <h3>현재 신호</h3>
                <div class="value green">{len(current_signals)}</div>
            </div>
            <div class="summary-card">
                <h3>과거 신호</h3>
                <div class="value yellow">{total_signals}</div>
            </div>
            <div class="summary-card">
                <h3>전체 성공률</h3>
                <div class="value green">{overall_success_rate:.1f}%</div>
            </div>
        </div>

        <div class="section">
            <h2>🎯 현재 진입 신호</h2>
            {current_signals_html}
        </div>

        <!-- 차트 영역 (클릭 시 표시) -->
        <div id="chart-area" class="chart-container">
            <div class="chart-header">
                <h3 id="chart-title">차트</h3>
                <button class="close-btn" onclick="hideChart()">닫기</button>
            </div>
            <div id="chart-plot" style="height:500px;"></div>
        </div>

        <div class="section">
            <h2>📊 점수별 성공률</h2>
            <div class="accuracy-grid">
                {score_html}
            </div>
        </div>

        <div class="section">
            <h2>📋 시그널 발생 종목 ({len(signal_coins)}개)</h2>
            <p style="color:#888; margin-bottom:15px;">행을 클릭하면 차트가 표시됩니다.</p>
            <table>
                <tr><th>코인</th><th>가격</th><th>24h</th><th>RSI</th><th>BB</th><th>Vol</th><th>신호</th><th>성공률</th><th>상태</th></tr>
                {coin_table_html}
            </table>
        </div>
    </div>

    <script>
        const chartsData = {charts_json};

        function showChart(coin) {{
            const data = chartsData[coin];
            if (!data) {{
                alert('차트 데이터가 없습니다: ' + coin);
                return;
            }}

            const container = document.getElementById('chart-area');
            const plotDiv = document.getElementById('chart-plot');
            const title = document.getElementById('chart-title');

            title.textContent = coin + ' - Signal Analysis';

            // 시그널 분리
            const successSignals = data.signals.filter(s => s.success);
            const failSignals = data.signals.filter(s => !s.success);

            const traces = [
                // 가격 라인
                {{
                    x: data.timestamps,
                    y: data.close,
                    type: 'scatter',
                    name: 'Price',
                    line: {{ color: '#26a69a', width: 1.5 }},
                    yaxis: 'y'
                }},
                // BB Upper
                {{
                    x: data.timestamps,
                    y: data.bb_upper,
                    type: 'scatter',
                    name: 'BB Upper',
                    line: {{ color: 'rgba(255,255,255,0.2)', width: 1 }},
                    yaxis: 'y'
                }},
                // BB Lower
                {{
                    x: data.timestamps,
                    y: data.bb_lower,
                    type: 'scatter',
                    name: 'BB Lower',
                    line: {{ color: 'rgba(255,255,255,0.2)', width: 1 }},
                    fill: 'tonexty',
                    fillcolor: 'rgba(255,255,255,0.05)',
                    yaxis: 'y'
                }},
                // SMA25
                {{
                    x: data.timestamps,
                    y: data.sma_25,
                    type: 'scatter',
                    name: 'SMA25',
                    line: {{ color: '#f39c12', width: 1.5 }},
                    yaxis: 'y'
                }},
                // 성공 시그널
                {{
                    x: successSignals.map(s => s.time),
                    y: successSignals.map(s => s.price),
                    type: 'scatter',
                    mode: 'markers',
                    name: 'Success (+12%)',
                    marker: {{ symbol: 'triangle-up', size: 14, color: '#00d4aa' }},
                    yaxis: 'y',
                    hovertemplate: 'Score: %{{text}}<br>Gain: %{{customdata:.1f}}%<extra></extra>',
                    text: successSignals.map(s => s.score),
                    customdata: successSignals.map(s => s.max_gain)
                }},
                // 실패 시그널
                {{
                    x: failSignals.map(s => s.time),
                    y: failSignals.map(s => s.price),
                    type: 'scatter',
                    mode: 'markers',
                    name: 'Failed',
                    marker: {{ symbol: 'triangle-up', size: 12, color: '#ff4757' }},
                    yaxis: 'y',
                    hovertemplate: 'Score: %{{text}}<br>Gain: %{{customdata:.1f}}%<extra></extra>',
                    text: failSignals.map(s => s.score),
                    customdata: failSignals.map(s => s.max_gain)
                }},
                // RSI
                {{
                    x: data.timestamps,
                    y: data.rsi,
                    type: 'scatter',
                    name: 'RSI',
                    line: {{ color: '#9b59b6', width: 1.5 }},
                    yaxis: 'y2'
                }},
            ];

            const layout = {{
                template: 'plotly_dark',
                height: 500,
                showlegend: true,
                legend: {{ orientation: 'h', y: 1.02 }},
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                margin: {{ l: 60, r: 60, t: 40, b: 40 }},
                xaxis: {{
                    showgrid: true,
                    gridcolor: 'rgba(255,255,255,0.1)',
                    rangeslider: {{ visible: false }}
                }},
                yaxis: {{
                    title: 'Price',
                    showgrid: true,
                    gridcolor: 'rgba(255,255,255,0.1)',
                    domain: [0.25, 1]
                }},
                yaxis2: {{
                    title: 'RSI',
                    showgrid: true,
                    gridcolor: 'rgba(255,255,255,0.1)',
                    domain: [0, 0.2],
                    range: [0, 100]
                }},
                shapes: [
                    {{ type: 'line', y0: 45, y1: 45, x0: 0, x1: 1, xref: 'paper', yref: 'y2',
                       line: {{ color: 'rgba(255,0,0,0.5)', dash: 'dash' }} }},
                    {{ type: 'line', y0: 30, y1: 30, x0: 0, x1: 1, xref: 'paper', yref: 'y2',
                       line: {{ color: 'rgba(0,255,0,0.5)', dash: 'dash' }} }},
                ]
            }};

            Plotly.newPlot(plotDiv, traces, layout, {{ responsive: true }});
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
    output_path = '/home/hyeon/project/ichimoku/data/surge_report.html'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print("\n" + "=" * 60)
    print("Report generated!")
    print("=" * 60)
    print(f"  File: {output_path}")
    print(f"  Coins: {len(all_data)}")
    print(f"  Current signals: {len(current_signals)}")
    print(f"  Historical signals: {total_signals}")
    print(f"  Success rate: {overall_success_rate:.1f}%")
    print(f"\nOpen in browser:")
    print(f"   file://{output_path}")


if __name__ == "__main__":
    generate_report()
