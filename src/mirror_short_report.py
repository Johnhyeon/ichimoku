import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def to_raw_number_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    text = str(value)
    if "e" in text.lower():
        return format(float(value), ".20f").rstrip("0").rstrip(".")
    return text


def _to_fixed_2(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.2f}"


def _nearest_index(ts_series: pd.Series, target: pd.Timestamp) -> int:
    arr = pd.to_datetime(ts_series).values
    pos = arr.searchsorted(target.to_datetime64())
    if pos <= 0:
        return 0
    if pos >= len(arr):
        return len(arr) - 1
    before = pd.Timestamp(arr[pos - 1])
    after = pd.Timestamp(arr[pos])
    if abs((target - before).total_seconds()) <= abs((after - target).total_seconds()):
        return pos - 1
    return pos


def _build_trade_chart_data(trades: List[dict], all_data: Dict[str, pd.DataFrame], window: int = 24) -> Dict[str, dict]:
    trade_chart_data = {}
    for i, t in enumerate(trades):
        symbol = t["symbol"]
        df = all_data.get(symbol)
        if df is None or df.empty:
            continue
        local = df.sort_values("timestamp").reset_index(drop=True)
        local["timestamp"] = pd.to_datetime(local["timestamp"])
        if local.empty:
            continue

        entry_ts = pd.to_datetime(t["entry_time"])
        exit_ts = pd.to_datetime(t["exit_time"])
        entry_idx = _nearest_index(local["timestamp"], entry_ts)
        exit_idx = _nearest_index(local["timestamp"], exit_ts)
        if exit_idx < entry_idx:
            entry_idx, exit_idx = exit_idx, entry_idx
        start_idx = max(0, entry_idx - window)
        end_idx = min(len(local) - 1, exit_idx + window)
        cut = local.iloc[start_idx : end_idx + 1]
        if len(cut) < 3:
            continue

        candles = []
        for _, row in cut.iterrows():
            ts = int(pd.Timestamp(row["timestamp"]).timestamp())
            candles.append(
                {
                    "time": ts,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                }
            )

        trade_chart_data[str(i)] = {
            "symbol": symbol,
            "side": "short",
            "entry_time": entry_ts.isoformat(),
            "exit_time": exit_ts.isoformat(),
            "entry_price": to_raw_number_str(t["entry_price"]),
            "exit_price": to_raw_number_str(t["exit_price"]),
            "reason": t.get("reason", ""),
            "pnl_usd": _to_fixed_2(t.get("pnl_usd")),
            "pnl_pct_notional": _to_fixed_2(t.get("pnl_pct_notional")),
            "candles": candles,
            "entry_marker_time": int(pd.Timestamp(local.iloc[entry_idx]["timestamp"]).timestamp()),
            "exit_marker_time": int(pd.Timestamp(local.iloc[exit_idx]["timestamp"]).timestamp()),
        }
    return trade_chart_data


def generate_html_report(
    result: dict,
    all_data: Dict[str, pd.DataFrame],
    out_path: Path,
    page_size: int = 100,
    chart_window: int = 24,
) -> str:
    trades = result.get("trades", [])
    equity_curve = result.get("equity_curve", [])
    total_pnl = result.get("total_pnl", 0)
    win_rate = result.get("win_rate", 0)
    pf = result.get("pf", 0)
    max_dd_pct = result.get("max_dd_pct", 0)
    balance = result.get("balance", 0)
    initial_seed = result.get("initial_seed", 1000.0)

    trade_rows = []
    for i, t in enumerate(trades):
        trade_rows.append(
            {
                "idx": i,
                "symbol": t["symbol"],
                "entry_time": pd.to_datetime(t["entry_time"]).isoformat(sep=" "),
                "exit_time": pd.to_datetime(t["exit_time"]).isoformat(sep=" "),
                "entry_price": to_raw_number_str(t["entry_price"]),
                "exit_price": to_raw_number_str(t["exit_price"]),
                "reason": t.get("reason", ""),
                "pnl_usd": _to_fixed_2(t.get("pnl_usd")),
                "pnl_pct_notional": _to_fixed_2(t.get("pnl_pct_notional")),
                "_sort_pnl_usd": float(t.get("pnl_usd", 0.0)),
                "_sort_pnl_pct": float(t.get("pnl_pct_notional", 0.0)),
            }
        )

    chart_data = _build_trade_chart_data(trades, all_data, window=chart_window)
    equity_labels = [pd.to_datetime(e["timestamp"]).isoformat(sep=" ") for e in equity_curve]
    equity_values = [float(e["equity"]) for e in equity_curve]

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Live Surge Mirror Short Report</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
  <script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    body {{ font-family: "Segoe UI", Arial, sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }}
    .wrap {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
    .cards {{ display: grid; grid-template-columns: repeat(6, minmax(150px, 1fr)); gap: 12px; }}
    .card {{ background: #111827; border: 1px solid #1f2937; border-radius: 10px; padding: 14px; }}
    .k {{ font-size: 12px; color: #94a3b8; }}
    .v {{ font-size: 20px; font-weight: 700; margin-top: 4px; }}
    .panel {{ margin-top: 16px; background: #111827; border: 1px solid #1f2937; border-radius: 10px; padding: 14px; }}
    .toolbar {{ display: flex; gap: 10px; align-items: center; margin-bottom: 10px; }}
    input, select {{ background: #0b1220; color: #e2e8f0; border: 1px solid #334155; border-radius: 8px; padding: 8px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px; border-bottom: 1px solid #1f2937; text-align: left; }}
    th {{ color: #cbd5e1; position: sticky; top: 0; background: #111827; }}
    a {{ color: #60a5fa; text-decoration: none; }}
    .pager {{ display: flex; gap: 8px; align-items: center; margin-top: 10px; }}
    button {{ background: #1e293b; color: #e2e8f0; border: 1px solid #334155; border-radius: 8px; padding: 6px 10px; cursor: pointer; }}
    .modal-overlay {{ position: fixed; inset: 0; display: none; background: rgba(2,6,23,.7); z-index: 999; }}
    .modal-overlay.active {{ display: block; }}
    .modal {{ max-width: 1100px; margin: 30px auto; background: #0b1220; border: 1px solid #334155; border-radius: 10px; padding: 12px; }}
    #tradeChartContainer {{ height: 520px; }}
    .mono {{ font-family: Consolas, "Courier New", monospace; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Live Surge Mirror Short Backtest (parquet-only)</h1>
    <div class="cards">
      <div class="card"><div class="k">Trades</div><div class="v">{len(trades)}</div></div>
      <div class="card"><div class="k">Initial Seed</div><div class="v mono">{to_raw_number_str(initial_seed)}</div></div>
      <div class="card"><div class="k">Win Rate</div><div class="v">{to_raw_number_str(win_rate)}</div></div>
      <div class="card"><div class="k">Total PnL</div><div class="v mono">{to_raw_number_str(total_pnl)}</div></div>
      <div class="card"><div class="k">Final Balance</div><div class="v mono">{to_raw_number_str(balance)}</div></div>
      <div class="card"><div class="k">PF / MDD%</div><div class="v">{to_raw_number_str(pf)} / {to_raw_number_str(max_dd_pct)}</div></div>
    </div>
    <div class="panel">
      <canvas id="equityChart" height="100"></canvas>
    </div>
    <div class="panel">
      <div class="toolbar">
        <input id="tradeSearch" placeholder="symbol / reason search" />
        <label for="pageSizeSelect">rows</label>
        <select id="pageSizeSelect">
          <option>50</option>
          <option selected>{page_size}</option>
          <option>200</option>
          <option>500</option>
        </select>
      </div>
      <table>
        <thead>
          <tr>
            <th>#</th><th>Symbol</th><th>Entry Time</th><th>Exit Time</th>
            <th>Entry Price</th><th>Exit Price</th><th>Reason</th>
            <th><a href="#" onclick="toggleSort('pnl_usd'); return false;">PnL USD</a></th>
            <th><a href="#" onclick="toggleSort('pnl_pct'); return false;">PnL % (Notional)</a></th>
          </tr>
        </thead>
        <tbody id="tradeTbody"></tbody>
      </table>
      <div class="pager">
        <button id="prevPage">Prev</button>
        <span id="pageInfo"></span>
        <button id="nextPage">Next</button>
      </div>
    </div>
  </div>

  <div class="modal-overlay" id="tradeModal">
    <div class="modal">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <h3 id="modalTitle">Trade Chart</h3>
        <button onclick="closeTradeChart()">Close</button>
      </div>
      <div id="modalInfo" class="mono" style="margin-bottom:8px;"></div>
      <div id="tradeChartContainer"></div>
    </div>
  </div>

  <script>
    const equityLabels = {json.dumps(equity_labels)};
    const equityValues = {json.dumps(equity_values)};
    new Chart(document.getElementById('equityChart'), {{
      type: 'line',
      data: {{ labels: equityLabels, datasets: [{{ data: equityValues, borderColor: '#38bdf8', pointRadius: 0 }}] }},
      options: {{ plugins: {{ legend: {{ display: false }} }}, scales: {{ x: {{ display: false }} }} }}
    }});

    const allTrades = {json.dumps(trade_rows)};
    const tradeChartData = {json.dumps(chart_data)};

    const tbody = document.getElementById('tradeTbody');
    const searchEl = document.getElementById('tradeSearch');
    const pageSizeEl = document.getElementById('pageSizeSelect');
    const pageInfoEl = document.getElementById('pageInfo');
    let filteredTrades = allTrades.slice();
    let currentPage = 1;
    let currentChart = null;
    let pageSize = parseInt(pageSizeEl.value, 10);
    let sortKey = null;
    let sortDir = null;

    function applyFilter() {{
      const q = searchEl.value.trim().toLowerCase();
      filteredTrades = allTrades.filter(t =>
        t.symbol.toLowerCase().includes(q) || String(t.reason).toLowerCase().includes(q)
      );
      applySort();
      currentPage = 1;
      renderPage();
    }}

    function applySort() {{
      if (!sortKey || !sortDir) return;
      const key = sortKey === 'pnl_usd' ? '_sort_pnl_usd' : '_sort_pnl_pct';
      const sign = sortDir === 'asc' ? 1 : -1;
      filteredTrades.sort((a, b) => (a[key] - b[key]) * sign);
    }}

    function toggleSort(key) {{
      if (sortKey !== key) {{
        sortKey = key;
        sortDir = 'desc';
      }} else if (sortDir === 'desc') {{
        sortDir = 'asc';
      }} else if (sortDir === 'asc') {{
        sortDir = null;
      }} else {{
        sortDir = 'desc';
      }}
      filteredTrades = allTrades.filter(t =>
        t.symbol.toLowerCase().includes(searchEl.value.trim().toLowerCase()) ||
        String(t.reason).toLowerCase().includes(searchEl.value.trim().toLowerCase())
      );
      applySort();
      currentPage = 1;
      renderPage();
    }}

    function renderPage() {{
      pageSize = parseInt(pageSizeEl.value, 10);
      const totalPages = Math.max(1, Math.ceil(filteredTrades.length / pageSize));
      if (currentPage > totalPages) currentPage = totalPages;
      const start = (currentPage - 1) * pageSize;
      const end = start + pageSize;
      const rows = filteredTrades.slice(start, end);
      tbody.innerHTML = rows.map(t => `
        <tr>
          <td>${{t.idx + 1}}</td>
          <td><a href="#" onclick="openTradeChart(${{t.idx}}); return false;">${{t.symbol}}</a></td>
          <td class="mono">${{t.entry_time}}</td>
          <td class="mono">${{t.exit_time}}</td>
          <td class="mono">${{t.entry_price}}</td>
          <td class="mono">${{t.exit_price}}</td>
          <td>${{t.reason}}</td>
          <td class="mono">${{t.pnl_usd}}</td>
          <td class="mono">${{t.pnl_pct_notional}}</td>
        </tr>
      `).join('');
      pageInfoEl.textContent = `Page ${{currentPage}} / ${{totalPages}} (rows=${{filteredTrades.length}})`;
    }}

    document.getElementById('prevPage').onclick = () => {{ if (currentPage > 1) {{ currentPage--; renderPage(); }} }};
    document.getElementById('nextPage').onclick = () => {{
      const totalPages = Math.max(1, Math.ceil(filteredTrades.length / pageSize));
      if (currentPage < totalPages) {{ currentPage++; renderPage(); }}
    }};
    searchEl.addEventListener('input', applyFilter);
    pageSizeEl.addEventListener('change', renderPage);

    function openTradeChart(idx) {{
      const data = tradeChartData[String(idx)];
      if (!data) {{
        alert('Trade chart data not available for this row.');
        return;
      }}
      const modal = document.getElementById('tradeModal');
      const info = document.getElementById('modalInfo');
      info.textContent = `${{data.symbol}} | Entry=${{data.entry_price}} | Exit=${{data.exit_price}} | Reason=${{data.reason}} | PnL=${{data.pnl_usd}}`;

      const container = document.getElementById('tradeChartContainer');
      container.innerHTML = '';
      if (currentChart) currentChart.remove();

      if (typeof LightweightCharts === 'undefined') {{
        modal.classList.add('active');
        return;
      }}

      const chart = LightweightCharts.createChart(container, {{
        width: container.clientWidth,
        height: 520,
        layout: {{ background: {{ color: '#0b1220' }}, textColor: '#cbd5e1' }},
        grid: {{ vertLines: {{ color: '#1f2937' }}, horzLines: {{ color: '#1f2937' }} }}
      }});
      const series = chart.addCandlestickSeries();
      series.setData(data.candles);
      series.setMarkers([
        {{ time: data.entry_marker_time, position: 'aboveBar', color: '#ef4444', shape: 'arrowDown', text: 'Entry' }},
        {{ time: data.exit_marker_time, position: 'belowBar', color: '#22c55e', shape: 'arrowUp', text: 'Exit' }}
      ]);
      chart.timeScale().fitContent();
      currentChart = chart;
      modal.classList.add('active');
      modal.onclick = (e) => {{ if (e.target === modal) closeTradeChart(); }};
    }}

    function closeTradeChart() {{
      document.getElementById('tradeModal').classList.remove('active');
      if (currentChart) {{ currentChart.remove(); currentChart = null; }}
    }}

    renderPage();
  </script>
</body>
</html>
"""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return str(out_path)


def save_trades_csv(result: dict, out_path: Path) -> str:
    trades = result.get("trades", [])
    rows = []
    for t in trades:
        rows.append(
            {
                "symbol": t.get("symbol"),
                "entry_time": pd.to_datetime(t.get("entry_time")).isoformat(sep=" "),
                "exit_time": pd.to_datetime(t.get("exit_time")).isoformat(sep=" "),
                "entry_price": to_raw_number_str(t.get("entry_price")),
                "exit_price": to_raw_number_str(t.get("exit_price")),
                "reason": t.get("reason", ""),
                "pnl_usd": _to_fixed_2(t.get("pnl_usd")),
                "pnl_pct_notional": _to_fixed_2(t.get("pnl_pct_notional")),
            }
        )
    df = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return str(out_path)
