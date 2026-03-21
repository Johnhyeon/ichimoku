"""
Monte Carlo Simulation for Combined Backtest Results

백테스트 트레이드 결과를 기반으로 몬테카를로 시뮬레이션을 수행합니다.
트레이드를 부트스트랩 리샘플링하여 다양한 시나리오의 자산 곡선, 드로다운,
파산 확률 등을 분석합니다.

사용법:
    # 백테스트 먼저 실행 (trades JSON 자동 저장)
    python scripts/backtest_combined.py --balance 1000 --start 2024-02-02 --end 2026-02-02

    # 몬테카를로 시뮬레이션
    python scripts/monte_carlo.py --trades data/backtest_combined_trades.json
    python scripts/monte_carlo.py --trades data/backtest_combined_trades.json --sims 20000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def load_trades(filepath: str) -> dict:
    """Load trades from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "trades" in data:
        return data
    elif isinstance(data, list):
        return {"trades": data, "initial_balance": 1000.0}

    raise ValueError(f"Invalid trades JSON format in {filepath}")


def compute_trade_returns(trades: list, initial_balance: float) -> np.ndarray:
    """각 트레이드의 계좌 대비 수익률을 계산합니다.

    트레이드를 시간순으로 정렬한 뒤, 닫힐 시점의 잔고 대비
    pnl_usd 비율을 리턴합니다.
    """
    sorted_trades = sorted(trades, key=lambda t: str(t.get("exit_time", "")))
    returns = []
    balance = initial_balance

    for t in sorted_trades:
        pnl = t["pnl_usd"]
        if balance > 0:
            returns.append(pnl / balance)
        balance += pnl

    return np.array(returns, dtype=np.float64)


def run_monte_carlo(
    returns: np.ndarray,
    initial_balance: float,
    num_trades: int,
    num_sims: int = 10000,
    seed: int = 42,
) -> dict:
    """Monte Carlo 시뮬레이션 실행.

    Returns:
        equity_curves: (num_sims, num_trades+1)
        final_balances: (num_sims,)
        max_drawdowns_pct: (num_sims,) — 최대 드로다운 (%)
        min_equities: (num_sims,) — 시뮬레이션 중 최저 자산
    """
    rng = np.random.default_rng(seed)

    equity_curves = np.empty((num_sims, num_trades + 1), dtype=np.float64)
    equity_curves[:, 0] = initial_balance
    max_drawdowns = np.zeros(num_sims, dtype=np.float64)
    min_equities = np.full(num_sims, initial_balance, dtype=np.float64)

    for i in range(num_sims):
        sampled_idx = rng.integers(0, len(returns), size=num_trades)
        balance = initial_balance
        peak = initial_balance
        max_dd = 0.0

        for j, idx in enumerate(sampled_idx):
            r = returns[idx]
            balance *= 1.0 + r

            if balance <= 0:
                balance = 0.0
                equity_curves[i, j + 1 :] = 0.0
                max_dd = 100.0
                min_equities[i] = 0.0
                break

            equity_curves[i, j + 1] = balance

            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100.0
            if dd > max_dd:
                max_dd = dd
            if balance < min_equities[i]:
                min_equities[i] = balance

        max_drawdowns[i] = max_dd

    return {
        "equity_curves": equity_curves,
        "final_balances": equity_curves[:, -1],
        "max_drawdowns": max_drawdowns,
        "min_equities": min_equities,
    }


def generate_html_report(
    results: dict,
    trades: list,
    initial_balance: float,
    num_sims: int,
    num_trades: int,
    period_str: str,
    output_path: str,
) -> str:
    """HTML 리포트 생성 (Chart.js)."""
    ec = results["equity_curves"]
    fb = results["final_balances"]
    mdd = results["max_drawdowns"]

    # ── Percentile curves for fan chart ──
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    pct_curves = {}
    for p in percentiles:
        pct_curves[p] = np.percentile(ec, p, axis=0)

    # Downsample for rendering performance
    max_points = 400
    total_steps = num_trades + 1
    if total_steps > max_points:
        step = total_steps // max_points
        indices = list(range(0, total_steps, step))
        if indices[-1] != total_steps - 1:
            indices.append(total_steps - 1)
    else:
        indices = list(range(total_steps))
        step = 1

    x_labels = json.dumps(indices)
    ds_curves = {}
    for p in percentiles:
        ds_curves[p] = json.dumps([round(float(pct_curves[p][i]), 2) for i in indices])

    # ── Summary statistics ──
    median_final = float(np.median(fb))
    mean_final = float(np.mean(fb))
    std_final = float(np.std(fb))
    p5_final = float(np.percentile(fb, 5))
    p25_final = float(np.percentile(fb, 25))
    p75_final = float(np.percentile(fb, 75))
    p95_final = float(np.percentile(fb, 95))
    min_final = float(np.min(fb))
    max_final = float(np.max(fb))

    median_return = (median_final - initial_balance) / initial_balance * 100
    mean_return = (mean_final - initial_balance) / initial_balance * 100
    p5_return = (p5_final - initial_balance) / initial_balance * 100
    p95_return = (p95_final - initial_balance) / initial_balance * 100

    median_mdd = float(np.median(mdd))
    mean_mdd = float(np.mean(mdd))
    p95_mdd = float(np.percentile(mdd, 95))
    max_mdd = float(np.max(mdd))

    profitable_pct = float(np.sum(fb > initial_balance) / num_sims * 100)
    loss_pct = 100.0 - profitable_pct

    # ── Risk of ruin ──
    ruin_thresholds = [
        (0.95, "5% 손실"),
        (0.90, "10% 손실"),
        (0.75, "25% 손실"),
        (0.50, "50% 손실"),
        (0.25, "75% 손실"),
    ]
    ruin_rows = ""
    for th, label in ruin_thresholds:
        # Check lowest point in each simulation
        count = int(np.sum(results["min_equities"] < initial_balance * th))
        prob = count / num_sims * 100
        cls = "positive" if prob < 5 else ("" if prob < 20 else "negative")
        ruin_rows += f'<tr><td>{label} (${initial_balance * th:,.0f} 이하)</td><td class="{cls}">{prob:.2f}%</td><td>{count:,} / {num_sims:,}</td></tr>\n'

    # ── Final balance histogram ──
    hist_bins = 60
    fb_hist, fb_edges = np.histogram(fb, bins=hist_bins)
    fb_centers = [round(float((fb_edges[i] + fb_edges[i + 1]) / 2), 2) for i in range(hist_bins)]
    fb_hist_list = fb_hist.tolist()
    fb_colors = json.dumps(
        ["rgba(34,197,94,0.7)" if c >= initial_balance else "rgba(239,68,68,0.7)" for c in fb_centers]
    )

    # ── Max drawdown histogram ──
    mdd_hist, mdd_edges = np.histogram(mdd, bins=50)
    mdd_centers = [round(float((mdd_edges[i] + mdd_edges[i + 1]) / 2), 1) for i in range(len(mdd_hist))]
    mdd_hist_list = mdd_hist.tolist()

    # ── Original trades summary ──
    orig_wins = sum(1 for t in trades if t["pnl_usd"] > 0)
    orig_losses = len(trades) - orig_wins
    orig_total_pnl = sum(t["pnl_usd"] for t in trades)
    orig_wr = orig_wins / len(trades) * 100 if trades else 0
    orig_return = orig_total_pnl / initial_balance * 100

    # ── Strategy breakdown ──
    strat_stats = {}
    for t in trades:
        s = t.get("strategy", "unknown")
        if s not in strat_stats:
            strat_stats[s] = {"count": 0, "wins": 0, "pnl": 0.0}
        strat_stats[s]["count"] += 1
        strat_stats[s]["pnl"] += t["pnl_usd"]
        if t["pnl_usd"] > 0:
            strat_stats[s]["wins"] += 1

    strat_labels_map = {
        "ichimoku": "Ichimoku Short",
        "mirror_short": "Mirror Short",
        "ma100": "MA100 V2",
    }
    strat_colors_map = {
        "ichimoku": "#a855f7",
        "mirror_short": "#ef4444",
        "ma100": "#3b82f6",
    }

    strat_cards = ""
    for s, st in strat_stats.items():
        label = strat_labels_map.get(s, s)
        color = strat_colors_map.get(s, "#888")
        wr = st["wins"] / st["count"] * 100 if st["count"] > 0 else 0
        pnl_cls = "positive" if st["pnl"] >= 0 else "negative"
        strat_cards += f"""
    <div class="card" style="border-left: 4px solid {color};">
      <div class="card-title">{label}</div>
      <div class="card-value {pnl_cls}">${st['pnl']:+,.2f}</div>
      <div class="card-sub">{st['count']} trades | WR={wr:.0f}% ({st['wins']}W/{st['count'] - st['wins']}L)</div>
    </div>"""

    # ── Percentile table for final balance ──
    pct_table_rows = ""
    for p, val, ret in [
        ("5th", p5_final, p5_return),
        ("25th", p25_final, (p25_final - initial_balance) / initial_balance * 100),
        ("50th (Median)", median_final, median_return),
        ("75th", p75_final, (p75_final - initial_balance) / initial_balance * 100),
        ("95th", p95_final, p95_return),
    ]:
        cls = "positive" if ret >= 0 else "negative"
        pct_table_rows += f'<tr><td>{p}</td><td>${val:,.2f}</td><td class="{cls}">{ret:+.2f}%</td></tr>\n'

    pnl_cls = "positive" if median_return >= 0 else "negative"
    orig_pnl_cls = "positive" if orig_total_pnl >= 0 else "negative"

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Monte Carlo Simulation Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0f1117; --card: #1a1d28; --border: #2a2d3a;
    --text: #e1e4ea; --muted: #8b8fa3;
    --green: #22c55e; --red: #ef4444; --blue: #3b82f6;
    --purple: #a855f7; --yellow: #eab308; --orange: #f97316;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--text); padding: 24px; line-height: 1.6;
  }}
  .container {{ max-width: 1300px; margin: 0 auto; }}
  h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 4px; }}
  .subtitle {{ color: var(--muted); margin-bottom: 24px; font-size: 14px; }}
  .grid {{ display: grid; gap: 16px; margin-bottom: 24px; }}
  .grid-4 {{ grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }}
  .grid-3 {{ grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
  .grid-2 {{ grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); }}
  .card {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px;
  }}
  .card-title {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--muted); margin-bottom: 8px; }}
  .card-value {{ font-size: 28px; font-weight: 700; }}
  .card-sub {{ font-size: 13px; color: var(--muted); margin-top: 4px; }}
  .positive {{ color: var(--green); }}
  .negative {{ color: var(--red); }}
  .chart-card {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px; margin-bottom: 24px;
  }}
  .chart-card h3 {{ font-size: 16px; margin-bottom: 16px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{
    text-align: left; padding: 10px 12px;
    border-bottom: 2px solid var(--border);
    color: var(--muted); font-weight: 600; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.5px;
  }}
  td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); }}
  tr:hover {{ background: rgba(255,255,255,0.02); }}
  .section-title {{
    font-size: 18px; font-weight: 600;
    margin: 32px 0 16px; padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }}
  canvas {{ max-height: 360px; }}
  .note {{ font-size: 12px; color: var(--muted); margin-top: 8px; }}
  .highlight {{ background: rgba(168,85,247,0.1); border-left: 3px solid var(--purple); padding: 12px 16px; border-radius: 8px; margin-bottom: 24px; }}
  .highlight strong {{ color: var(--purple); }}
</style>
</head>
<body>
<div class="container">

<h1>Monte Carlo Simulation</h1>
<p class="subtitle">{period_str} &nbsp;|&nbsp; {num_sims:,} simulations &times; {num_trades} trades (bootstrap resampling)</p>

<div class="highlight">
  <strong>Method:</strong> 원본 백테스트의 {len(trades)}개 트레이드를 부트스트랩 리샘플링하여 {num_sims:,}개의 시나리오를 생성합니다.
  각 시뮬레이션은 원본 트레이드 풀에서 {num_trades}개를 복원추출하여 순차적으로 적용합니다.
</div>

<!-- Summary Cards -->
<div class="grid grid-4">
  <div class="card">
    <div class="card-title">Median Final Balance</div>
    <div class="card-value {pnl_cls}">${median_final:,.2f}</div>
    <div class="card-sub">{median_return:+.2f}% return</div>
  </div>
  <div class="card">
    <div class="card-title">Profitable Simulations</div>
    <div class="card-value {'positive' if profitable_pct >= 50 else 'negative'}">{profitable_pct:.1f}%</div>
    <div class="card-sub">{loss_pct:.1f}% lost money</div>
  </div>
  <div class="card">
    <div class="card-title">Median Max Drawdown</div>
    <div class="card-value negative">-{median_mdd:.1f}%</div>
    <div class="card-sub">95th pct: -{p95_mdd:.1f}%</div>
  </div>
  <div class="card">
    <div class="card-title">5th~95th Range</div>
    <div class="card-value">${p5_final:,.0f}~${p95_final:,.0f}</div>
    <div class="card-sub">Initial: ${initial_balance:,.0f}</div>
  </div>
</div>

<!-- Original Backtest Summary -->
<h2 class="section-title">Original Backtest</h2>
<div class="grid grid-4">
  <div class="card">
    <div class="card-title">Total Trades</div>
    <div class="card-value">{len(trades)}</div>
    <div class="card-sub">{orig_wins}W / {orig_losses}L ({orig_wr:.1f}% WR)</div>
  </div>
  <div class="card">
    <div class="card-title">Original PnL</div>
    <div class="card-value {orig_pnl_cls}">${orig_total_pnl:+,.2f}</div>
    <div class="card-sub">{orig_return:+.2f}% return</div>
  </div>
{strat_cards}
</div>

<!-- Fan Chart -->
<div class="chart-card">
  <h3>Equity Curve Fan Chart (5th / 10th / 25th / 50th / 75th / 90th / 95th Percentile)</h3>
  <canvas id="fanChart"></canvas>
  <p class="note">진한 영역은 25th~75th 구간, 연한 영역은 5th~95th 구간</p>
</div>

<!-- Histograms -->
<div class="grid grid-2">
  <div class="chart-card">
    <h3>Final Balance Distribution</h3>
    <canvas id="balanceHist"></canvas>
    <p class="note">빨간 점선: 원금 (${initial_balance:,.0f})</p>
  </div>
  <div class="chart-card">
    <h3>Max Drawdown Distribution</h3>
    <canvas id="mddHist"></canvas>
    <p class="note">Median: {median_mdd:.1f}% | 95th: {p95_mdd:.1f}%</p>
  </div>
</div>

<!-- Percentile Table -->
<h2 class="section-title">Final Balance Percentiles</h2>
<div class="card" style="margin-bottom: 24px;">
  <table>
    <thead><tr><th>Percentile</th><th>Final Balance</th><th>Return</th></tr></thead>
    <tbody>
      <tr><td>Worst Case (Min)</td><td>${min_final:,.2f}</td><td class="negative">{(min_final - initial_balance) / initial_balance * 100:+.2f}%</td></tr>
      {pct_table_rows}
      <tr><td>Best Case (Max)</td><td>${max_final:,.2f}</td><td class="positive">{(max_final - initial_balance) / initial_balance * 100:+.2f}%</td></tr>
      <tr style="font-weight:700;"><td>Mean</td><td>${mean_final:,.2f}</td><td class="{'positive' if mean_return >= 0 else 'negative'}">{mean_return:+.2f}%</td></tr>
      <tr><td>Std Dev</td><td colspan="2">${std_final:,.2f}</td></tr>
    </tbody>
  </table>
</div>

<!-- Risk of Ruin -->
<h2 class="section-title">Risk of Ruin</h2>
<div class="card" style="margin-bottom: 24px;">
  <table>
    <thead><tr><th>Threshold</th><th>Probability</th><th>Count</th></tr></thead>
    <tbody>
      {ruin_rows}
    </tbody>
  </table>
  <p class="note">시뮬레이션 중 한 번이라도 해당 수준 이하로 떨어진 경우를 카운트합니다.</p>
</div>

<!-- Max Drawdown Percentiles -->
<h2 class="section-title">Max Drawdown Percentiles</h2>
<div class="card" style="margin-bottom: 24px;">
  <table>
    <thead><tr><th>Percentile</th><th>Max Drawdown</th></tr></thead>
    <tbody>
      <tr><td>25th</td><td>-{float(np.percentile(mdd, 25)):.1f}%</td></tr>
      <tr><td>50th (Median)</td><td>-{median_mdd:.1f}%</td></tr>
      <tr><td>75th</td><td>-{float(np.percentile(mdd, 75)):.1f}%</td></tr>
      <tr><td>90th</td><td>-{float(np.percentile(mdd, 90)):.1f}%</td></tr>
      <tr><td>95th</td><td>-{p95_mdd:.1f}%</td></tr>
      <tr><td>99th</td><td>-{float(np.percentile(mdd, 99)):.1f}%</td></tr>
      <tr><td>Worst</td><td class="negative">-{max_mdd:.1f}%</td></tr>
    </tbody>
  </table>
</div>

</div>

<script>
// ── Fan Chart ──
const fanCtx = document.getElementById('fanChart');
new Chart(fanCtx, {{
  type: 'line',
  data: {{
    labels: {x_labels},
    datasets: [
      // 5-95 fill
      {{
        label: '95th',
        data: {ds_curves[95]},
        borderColor: 'rgba(168,85,247,0.3)',
        backgroundColor: 'rgba(168,85,247,0.05)',
        fill: '+6',
        borderWidth: 1,
        pointRadius: 0,
        tension: 0.3,
      }},
      // 10-90 fill
      {{
        label: '90th',
        data: {ds_curves[90]},
        borderColor: 'rgba(168,85,247,0.3)',
        backgroundColor: 'rgba(168,85,247,0.08)',
        fill: '+4',
        borderWidth: 1,
        pointRadius: 0,
        tension: 0.3,
      }},
      // 25-75 fill
      {{
        label: '75th',
        data: {ds_curves[75]},
        borderColor: 'rgba(59,130,246,0.5)',
        backgroundColor: 'rgba(59,130,246,0.12)',
        fill: '+2',
        borderWidth: 1,
        pointRadius: 0,
        tension: 0.3,
      }},
      // Median
      {{
        label: '50th (Median)',
        data: {ds_curves[50]},
        borderColor: '#22c55e',
        borderWidth: 2.5,
        pointRadius: 0,
        tension: 0.3,
        fill: false,
      }},
      // 25th
      {{
        label: '25th',
        data: {ds_curves[25]},
        borderColor: 'rgba(59,130,246,0.5)',
        backgroundColor: 'transparent',
        borderWidth: 1,
        pointRadius: 0,
        tension: 0.3,
        fill: false,
      }},
      // 10th
      {{
        label: '10th',
        data: {ds_curves[10]},
        borderColor: 'rgba(168,85,247,0.3)',
        backgroundColor: 'transparent',
        borderWidth: 1,
        pointRadius: 0,
        tension: 0.3,
        fill: false,
      }},
      // 5th
      {{
        label: '5th',
        data: {ds_curves[5]},
        borderColor: 'rgba(168,85,247,0.3)',
        backgroundColor: 'transparent',
        borderWidth: 1,
        pointRadius: 0,
        tension: 0.3,
        fill: false,
      }},
      // Initial balance line
      {{
        label: 'Initial (${initial_balance:,.0f})',
        data: Array({len(indices)}).fill({initial_balance}),
        borderColor: 'rgba(255,255,255,0.3)',
        borderWidth: 1,
        borderDash: [5, 5],
        pointRadius: 0,
        fill: false,
      }},
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{
        position: 'bottom',
        labels: {{
          padding: 16,
          color: '#8b8fa3',
          usePointStyle: true,
          filter: function(item) {{
            return ['5th','50th (Median)','75th','95th','Initial (${initial_balance:,.0f})'].includes(item.text);
          }}
        }}
      }},
      tooltip: {{
        mode: 'index',
        intersect: false,
        callbacks: {{
          label: function(ctx) {{ return ctx.dataset.label + ': $' + ctx.raw.toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}}); }}
        }}
      }}
    }},
    scales: {{
      x: {{
        title: {{ display: true, text: 'Trade #', color: '#8b8fa3' }},
        ticks: {{ color: '#8b8fa3', maxTicksLimit: 15 }},
        grid: {{ color: 'rgba(42,45,58,0.5)' }},
      }},
      y: {{
        title: {{ display: true, text: 'Balance ($)', color: '#8b8fa3' }},
        ticks: {{ color: '#8b8fa3', callback: function(v) {{ return '$' + v.toLocaleString(); }} }},
        grid: {{ color: 'rgba(42,45,58,0.5)' }},
      }}
    }},
    interaction: {{ mode: 'nearest', axis: 'x', intersect: false }},
  }}
}});

// ── Final Balance Histogram ──
new Chart(document.getElementById('balanceHist'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(fb_centers)},
    datasets: [{{
      label: 'Frequency',
      data: {json.dumps(fb_hist_list)},
      backgroundColor: {fb_colors},
      borderRadius: 2,
      barPercentage: 1.0,
      categoryPercentage: 1.0,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ display: false }},
      annotation: {{ annotations: {{ line1: {{ type: 'line', xMin: {initial_balance}, xMax: {initial_balance}, borderColor: '#ef4444', borderWidth: 2, borderDash: [5,5] }} }} }},
    }},
    scales: {{
      x: {{
        title: {{ display: true, text: 'Final Balance ($)', color: '#8b8fa3' }},
        ticks: {{
          color: '#8b8fa3',
          maxTicksLimit: 10,
          callback: function(v,i) {{ return '$' + {json.dumps(fb_centers)}[i]?.toLocaleString(undefined,{{maximumFractionDigits:0}}) || ''; }}
        }},
        grid: {{ display: false }},
      }},
      y: {{
        title: {{ display: true, text: 'Frequency', color: '#8b8fa3' }},
        ticks: {{ color: '#8b8fa3' }},
        grid: {{ color: 'rgba(42,45,58,0.5)' }},
      }}
    }}
  }}
}});

// ── Max Drawdown Histogram ──
new Chart(document.getElementById('mddHist'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(mdd_centers)},
    datasets: [{{
      label: 'Frequency',
      data: {json.dumps(mdd_hist_list)},
      backgroundColor: 'rgba(239,68,68,0.6)',
      borderRadius: 2,
      barPercentage: 1.0,
      categoryPercentage: 1.0,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{
        title: {{ display: true, text: 'Max Drawdown (%)', color: '#8b8fa3' }},
        ticks: {{
          color: '#8b8fa3',
          maxTicksLimit: 10,
          callback: function(v,i) {{ return {json.dumps(mdd_centers)}[i]?.toFixed(0) + '%' || ''; }}
        }},
        grid: {{ display: false }},
      }},
      y: {{
        title: {{ display: true, text: 'Frequency', color: '#8b8fa3' }},
        ticks: {{ color: '#8b8fa3' }},
        grid: {{ color: 'rgba(42,45,58,0.5)' }},
      }}
    }}
  }}
}});
</script>
</body>
</html>"""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return str(out)


def print_summary(results: dict, trades: list, initial_balance: float, num_sims: int, num_trades: int):
    """콘솔 요약 출력."""
    fb = results["final_balances"]
    mdd = results["max_drawdowns"]

    median_final = np.median(fb)
    mean_final = np.mean(fb)
    p5 = np.percentile(fb, 5)
    p95 = np.percentile(fb, 95)
    profitable = np.sum(fb > initial_balance) / num_sims * 100

    median_mdd = np.median(mdd)
    p95_mdd = np.percentile(mdd, 95)

    print()
    print("=" * 60)
    print("  Monte Carlo Simulation Results")
    print("=" * 60)
    print(f"  Simulations : {num_sims:,}")
    print(f"  Trades/sim  : {num_trades}")
    print(f"  Source trades: {len(trades)}")
    print(f"  Initial     : ${initial_balance:,.2f}")
    print()

    print("--- Final Balance ---")
    print(f"  Median       : ${median_final:,.2f} ({(median_final-initial_balance)/initial_balance*100:+.2f}%)")
    print(f"  Mean         : ${mean_final:,.2f} ({(mean_final-initial_balance)/initial_balance*100:+.2f}%)")
    print(f"  5th pct      : ${p5:,.2f}")
    print(f"  95th pct     : ${p95:,.2f}")
    print(f"  Min          : ${np.min(fb):,.2f}")
    print(f"  Max          : ${np.max(fb):,.2f}")
    print(f"  Profitable   : {profitable:.1f}%")
    print()

    print("--- Max Drawdown ---")
    print(f"  Median       : -{median_mdd:.1f}%")
    print(f"  95th pct     : -{p95_mdd:.1f}%")
    print(f"  Worst        : -{np.max(mdd):.1f}%")
    print()

    print("--- Risk of Ruin ---")
    for th, label in [(0.9, "10% loss"), (0.5, "50% loss"), (0.25, "75% loss")]:
        count = np.sum(results["min_equities"] < initial_balance * th)
        print(f"  {label:12s}: {count/num_sims*100:.2f}%")

    print()
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo Simulation for Backtest Results")
    parser.add_argument(
        "--trades",
        default="data/backtest_combined_trades.json",
        help="Path to trades JSON (default: data/backtest_combined_trades.json)",
    )
    parser.add_argument("--sims", type=int, default=10000, help="Number of simulations (default: 10000)")
    parser.add_argument("--balance", type=float, default=None, help="Override initial balance")
    parser.add_argument("--num-trades", type=int, default=None, help="Number of trades per simulation (default: same as source)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", default="data/monte_carlo_report.html", help="Output HTML path")
    args = parser.parse_args()

    trades_path = Path(args.trades)
    if not trades_path.exists():
        print(f"Error: {trades_path} not found.")
        print("먼저 백테스트를 실행하세요:")
        print("  python scripts/backtest_combined.py --balance 1000 --start 2024-02-02 --end 2026-02-02")
        sys.exit(1)

    print(f"Loading trades from {trades_path}...")
    data = load_trades(str(trades_path))
    trades = data["trades"]
    initial_balance = args.balance or data.get("initial_balance", 1000.0)
    period_str = f"{data.get('start', '?')} ~ {data.get('end', '?')}"

    if not trades:
        print("Error: No trades found.")
        sys.exit(1)

    num_trades = args.num_trades or len(trades)
    print(f"  {len(trades)} trades loaded (Initial: ${initial_balance:,.2f})")

    # Compute per-trade return rates
    returns = compute_trade_returns(trades, initial_balance)
    print(f"  Returns: mean={np.mean(returns)*100:.3f}%, std={np.std(returns)*100:.3f}%")

    # Run simulation
    print(f"\nRunning {args.sims:,} Monte Carlo simulations ({num_trades} trades each)...")
    t0 = time.time()
    results = run_monte_carlo(returns, initial_balance, num_trades, args.sims, args.seed)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    # Print summary
    print_summary(results, trades, initial_balance, args.sims, num_trades)

    # Generate HTML report
    html_path = generate_html_report(
        results, trades, initial_balance, args.sims, num_trades, period_str, args.output,
    )
    print(f"HTML Report: {html_path}")

    import webbrowser
    abs_path = Path(html_path).resolve()
    webbrowser.open(f"file:///{abs_path}")


if __name__ == "__main__":
    main()
