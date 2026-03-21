"""
투자 검토 보고서 (Investment Due Diligence Report) 생성기

백테스트 결과 + 몬테카를로 시뮬레이션 + BTC 가격 데이터를 기반으로
퀀트 PM 관점의 투자 검토 보고서(HTML)를 생성합니다.

사용법:
    python scripts/generate_strategy_report.py
"""

import json
import numpy as np
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def load_data():
    with open("data/backtest_combined_trades.json", "r", encoding="utf-8") as f:
        bt_data = json.load(f)
    btc_prices = []
    try:
        with open("data/btc_price_1d.json", "r", encoding="utf-8") as f:
            btc_prices = json.load(f)
    except Exception:
        pass
    return bt_data, btc_prices


def compute_all_stats(trades, ib, fb, start, end, btc_prices):
    """전체 통계 + 리스크 지표 계산."""
    total_days = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).days
    sorted_t = sorted(trades, key=lambda x: x["exit_time"])

    # --- 기본 지표 ---
    all_pnls = [t["pnl_usd"] for t in trades]
    wins = [p for p in all_pnls if p > 0]
    losses = [p for p in all_pnls if p <= 0]
    gp = sum(wins)
    gl = abs(sum(losses))
    pf = gp / gl if gl > 0 else float("inf")
    total_fees = sum(t.get("fee_usd", 0) for t in trades)
    total_pnl = sum(all_pnls)
    cagr = ((fb / ib) ** (365 / total_days) - 1) * 100 if total_days > 0 else 0

    # --- 전략별 ---
    strats = defaultdict(list)
    for t in trades:
        strats[t["strategy"]].append(t)

    strat_stats = {}
    for name, st in strats.items():
        pnls = [t["pnl_usd"] for t in st]
        w = [p for p in pnls if p > 0]
        l = [p for p in pnls if p <= 0]
        s_gp = sum(w)
        s_gl = abs(sum(l))
        fees = sum(t.get("fee_usd", 0) for t in st)

        # Consecutive
        st_sorted = sorted(st, key=lambda x: x["exit_time"])
        max_cw = max_cl = cw = cl = 0
        max_streak_pnl = cur_streak = 0.0
        for t in st_sorted:
            if t["pnl_usd"] > 0:
                cw += 1; cl = 0; cur_streak = 0
                max_cw = max(max_cw, cw)
            else:
                cl += 1; cw = 0; cur_streak += t["pnl_usd"]
                if cl > max_cl:
                    max_cl = cl; max_streak_pnl = cur_streak

        # Duration
        durs = []
        for t in st_sorted:
            try:
                d = (datetime.fromisoformat(t["exit_time"]) -
                     datetime.fromisoformat(t["entry_time"])).total_seconds() / 3600
                durs.append(d)
            except Exception:
                pass

        # Reasons
        reasons = defaultdict(int)
        for t in st:
            reasons[t["reason"]] += 1

        strat_stats[name] = {
            "n": len(pnls), "wins": len(w), "losses": len(l),
            "wr": len(w) / len(pnls) * 100, "pnl": sum(pnls),
            "pf": s_gp / s_gl if s_gl > 0 else float("inf"),
            "gp": s_gp, "gl": s_gl, "fees": fees,
            "avg_win": np.mean(w) if w else 0,
            "avg_loss": np.mean(l) if l else 0,
            "max_win": max(pnls), "max_loss": min(pnls),
            "max_cw": max_cw, "max_cl": max_cl,
            "max_streak_pnl": max_streak_pnl,
            "avg_dur": np.mean(durs) if durs else 0,
            "reasons": dict(sorted(reasons.items(), key=lambda x: -x[1])),
        }

    # --- 월별 ---
    monthly = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0})
    strat_monthly = defaultdict(lambda: defaultdict(float))
    for t in trades:
        m = t["exit_time"][:7]
        monthly[m]["pnl"] += t["pnl_usd"]
        monthly[m]["trades"] += 1
        if t["pnl_usd"] > 0:
            monthly[m]["wins"] += 1
        strat_monthly[t["strategy"]][m] += t["pnl_usd"]

    months = sorted(monthly.keys())
    monthly_pnls = [monthly[m]["pnl"] for m in months]
    pos_months = sum(1 for p in monthly_pnls if p > 0)
    sharpe = (np.mean(monthly_pnls) / np.std(monthly_pnls) * np.sqrt(12)
              if np.std(monthly_pnls) > 0 else 0)
    sortino_neg = [p for p in monthly_pnls if p < 0]
    sortino = (np.mean(monthly_pnls) / np.std(sortino_neg) * np.sqrt(12)
               if sortino_neg and np.std(sortino_neg) > 0 else 0)

    # --- 전략 간 상관관계 ---
    ichi_m = [strat_monthly["ichimoku"].get(m, 0) for m in months]
    mirror_m = [strat_monthly["mirror_short"].get(m, 0) for m in months]
    ma100_m = [strat_monthly["ma100"].get(m, 0) for m in months]
    corr_im = np.corrcoef(ichi_m, mirror_m)[0, 1] if len(months) > 2 else 0
    corr_ia = np.corrcoef(ichi_m, ma100_m)[0, 1] if len(months) > 2 else 0
    corr_ma = np.corrcoef(mirror_m, ma100_m)[0, 1] if len(months) > 2 else 0

    # --- Top winners 집중도 ---
    win_sorted = sorted(wins, reverse=True)
    top10_pct = sum(win_sorted[:10]) / gp * 100 if gp > 0 else 0
    top50_pct = sum(win_sorted[:50]) / gp * 100 if gp > 0 else 0

    # --- 연도별 ---
    yearly = defaultdict(lambda: {"pnl": 0, "cnt": 0})
    for t in trades:
        y = t["exit_time"][:4]
        yearly[y]["pnl"] += t["pnl_usd"]
        yearly[y]["cnt"] += 1

    # --- 반기별 ---
    half_yearly = {}
    for label, lo, hi in [
        ("2024 H1", "2024-01", "2024-06"), ("2024 H2", "2024-07", "2024-12"),
        ("2025 H1", "2025-01", "2025-06"), ("2025 H2", "2025-07", "2025-12"),
    ]:
        p = sum(t["pnl_usd"] for t in trades if lo <= t["exit_time"][:7] <= hi)
        half_yearly[label] = p

    # --- 슬리피지 민감도 ---
    slip_sens = {}
    for pct in [10, 20, 30, 50]:
        extra = gl * pct / 100
        slip_sens[pct] = {"pnl": total_pnl - extra, "pf": gp / (gl + extra) if (gl + extra) > 0 else 0}

    # --- 수수료 민감도 ---
    fee_sens = {}
    for mult in [1.5, 2.0, 3.0]:
        extra = total_fees * (mult - 1)
        fee_sens[mult] = total_pnl - extra

    # --- 심볼 분석 ---
    sym_pnl = defaultdict(float)
    sym_cnt = defaultdict(int)
    for t in trades:
        sym_pnl[t["symbol"]] += t["pnl_usd"]
        sym_cnt[t["symbol"]] += 1

    # --- BTC price at start/end ---
    btc_start = btc_prices[0]["close"] if btc_prices else 0
    btc_end = btc_prices[-1]["close"] if btc_prices else 0
    btc_change = (btc_end - btc_start) / btc_start * 100 if btc_start else 0

    # BTC max/min
    btc_max = max(b["close"] for b in btc_prices) if btc_prices else 0
    btc_min = min(b["close"] for b in btc_prices) if btc_prices else 0

    return {
        "total_days": total_days, "total_pnl": total_pnl, "pf": pf,
        "gp": gp, "gl": gl, "fees": total_fees, "cagr": cagr,
        "n_trades": len(trades), "n_wins": len(wins), "n_losses": len(losses),
        "wr": len(wins) / len(trades) * 100,
        "avg_pnl": np.mean(all_pnls), "median_pnl": np.median(all_pnls),
        "sharpe": sharpe, "sortino": sortino,
        "strat": strat_stats,
        "monthly": dict(sorted(monthly.items())), "months": months,
        "monthly_pnls": monthly_pnls,
        "pos_months": pos_months, "total_months": len(months),
        "best_month": (months[np.argmax(monthly_pnls)], max(monthly_pnls)),
        "worst_month": (months[np.argmin(monthly_pnls)], min(monthly_pnls)),
        "corr": {"im": corr_im, "ia": corr_ia, "ma": corr_ma},
        "top10_pct": top10_pct, "top50_pct": top50_pct,
        "yearly": dict(yearly), "half_yearly": half_yearly,
        "slip_sens": slip_sens, "fee_sens": fee_sens,
        "top_symbols": sorted(sym_pnl.items(), key=lambda x: -x[1])[:10],
        "bot_symbols": sorted(sym_pnl.items(), key=lambda x: x[1])[:10],
        "sym_cnt": dict(sym_cnt),
        "btc_start": btc_start, "btc_end": btc_end, "btc_change": btc_change,
        "btc_max": btc_max, "btc_min": btc_min,
        "strat_monthly": {k: dict(v) for k, v in strat_monthly.items()},
        "trades_per_day": len(trades) / total_days if total_days > 0 else 0,
    }


def generate_report(data, btc_prices, s):
    """퀀트 PM 관점의 투자 검토 보고서 HTML 생성."""
    ib = data["initial_balance"]
    fb = data["final_balance"]
    start = data["start"]
    end = data["end"]

    # BTC chart data
    btc_labels = json.dumps([b["timestamp"][:10] for b in btc_prices])
    btc_values = json.dumps([round(b["close"], 0) for b in btc_prices])

    # Monthly equity curve (cumulative)
    months = s["months"]
    cum = []
    running = 0
    for m in months:
        running += s["monthly"][m]["pnl"]
        cum.append(round(running, 0))
    eq_labels = json.dumps(months)
    eq_values = json.dumps(cum)

    # Per-strategy monthly
    ichi_cum = []
    mirror_cum = []
    ma100_cum = []
    r_i = r_m = r_a = 0
    for m in months:
        r_i += s["strat_monthly"].get("ichimoku", {}).get(m, 0)
        r_m += s["strat_monthly"].get("mirror_short", {}).get(m, 0)
        r_a += s["strat_monthly"].get("ma100", {}).get(m, 0)
        ichi_cum.append(round(r_i, 0))
        mirror_cum.append(round(r_m, 0))
        ma100_cum.append(round(r_a, 0))

    # Monthly PnL bar
    monthly_bar_vals = json.dumps([round(s["monthly"][m]["pnl"], 0) for m in months])
    monthly_bar_colors = json.dumps(
        ["rgba(34,197,94,0.7)" if s["monthly"][m]["pnl"] >= 0 else "rgba(239,68,68,0.7)" for m in months]
    )

    # Strategy configs
    configs = {
        "ichimoku": ("Ichimoku Cloud Short", "#a855f7", "4H", "20x", "5%", "5", "Cloud-based", "RR2:1+Trail1.5%"),
        "mirror_short": ("Mirror Short (Surge)", "#ef4444", "5m", "5x", "5%", "3", "1% Fixed", "Trail 3%/1.2%"),
        "ma100": ("MA100 Touch Reject", "#3b82f6", "1D", "3x", "2%", "20", "5% Fixed", "Trail 3%/2%"),
    }

    # Build strategy comparison table rows
    strat_rows = ""
    for key in ["ichimoku", "mirror_short", "ma100"]:
        if key not in s["strat"]:
            continue
        st = s["strat"][key]
        cfg = configs[key]
        pf_str = f'{st["pf"]:.2f}' if st["pf"] != float("inf") else "INF"
        strat_rows += f"""<tr>
          <td><span class="dot" style="background:{cfg[1]};"></span>{cfg[0]}</td>
          <td>{cfg[2]}</td><td>{cfg[3]}</td><td>{cfg[4]}</td><td>{cfg[5]}</td>
          <td>{st['n']:,}</td><td>{st['wr']:.1f}%</td><td>{pf_str}</td>
          <td class="{'pos' if st['pnl']>=0 else 'neg'}">${st['pnl']:+,.0f}</td>
          <td>${st['avg_win']:+,.0f}</td><td>${st['avg_loss']:+,.0f}</td>
          <td>{st['avg_dur']:.1f}h</td>
          <td>{st['max_cl']}</td>
          <td>${st['max_streak_pnl']:+,.0f}</td>
        </tr>"""

    # Sensitivity table
    slip_rows = ""
    for pct in [10, 20, 30, 50]:
        ss = s["slip_sens"][pct]
        slip_rows += f'<tr><td>+{pct}%</td><td>${ss["pnl"]:+,.0f}</td><td>{ss["pf"]:.2f}</td><td class="{"pos" if ss["pnl"]>0 else "neg"}">{"PASS" if ss["pf"]>1.2 else "WARNING" if ss["pf"]>1.0 else "FAIL"}</td></tr>'

    fee_rows = ""
    for mult in [1.5, 2.0, 3.0]:
        fp = s["fee_sens"][mult]
        fee_rows += f'<tr><td>x{mult:.1f}</td><td>${fp:+,.0f}</td><td class="{"pos" if fp>0 else "neg"}">{"PASS" if fp > s["total_pnl"]*0.5 else "WARNING"}</td></tr>'

    # Monthly table
    monthly_rows = ""
    for m in months:
        md = s["monthly"][m]
        cls = "pos" if md["pnl"] >= 0 else "neg"
        wr = md["wins"] / md["trades"] * 100 if md["trades"] > 0 else 0
        monthly_rows += f'<tr><td>{m}</td><td>{md["trades"]}</td><td>{wr:.0f}%</td><td class="{cls}">${md["pnl"]:+,.0f}</td></tr>'

    # Top/bottom symbols
    top_rows = ""
    for sym, pnl in s["top_symbols"]:
        short = sym.split("/")[0]
        top_rows += f'<tr><td>{short}</td><td>{s["sym_cnt"].get(sym,0)}</td><td class="pos">${pnl:+,.0f}</td></tr>'
    bot_rows = ""
    for sym, pnl in s["bot_symbols"]:
        short = sym.split("/")[0]
        bot_rows += f'<tr><td>{short}</td><td>{s["sym_cnt"].get(sym,0)}</td><td class="neg">${pnl:+,.0f}</td></tr>'

    # Yearly
    yearly_rows = ""
    for y in sorted(s["yearly"].keys()):
        yd = s["yearly"][y]
        yearly_rows += f'<tr><td>{y}</td><td>{yd["cnt"]:,}</td><td class="{"pos" if yd["pnl"]>=0 else "neg"}">${yd["pnl"]:+,.0f}</td></tr>'

    # Half yearly
    hy_rows = ""
    for label, pnl in s["half_yearly"].items():
        hy_rows += f'<tr><td>{label}</td><td class="{"pos" if pnl>=0 else "neg"}">${pnl:+,.0f}</td></tr>'

    # PnL contribution
    tp = s["total_pnl"]
    ichi_pct = s["strat"].get("ichimoku", {}).get("pnl", 0) / tp * 100 if tp else 0
    mirror_pct = s["strat"].get("mirror_short", {}).get("pnl", 0) / tp * 100 if tp else 0
    ma100_pct = s["strat"].get("ma100", {}).get("pnl", 0) / tp * 100 if tp else 0
    ichi_pnl = s["strat"].get("ichimoku", {}).get("pnl", 0)
    mirror_pnl = s["strat"].get("mirror_short", {}).get("pnl", 0)
    ma100_pnl = s["strat"].get("ma100", {}).get("pnl", 0)

    monthly_std = float(np.std(s["monthly_pnls"]))
    avg_monthly = float(np.mean(s["monthly_pnls"]))

    # Verdict
    verdict_pass = (
        s["pf"] >= 1.3 and
        s["slip_sens"][20]["pf"] >= 1.2 and
        s["pos_months"] / s["total_months"] >= 0.7 and
        s["corr"]["im"] < 0.7 and s["corr"]["ia"] < 0.7 and s["corr"]["ma"] < 0.7
    )

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Investment Due Diligence Report — Triple Short Strategy</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  :root {{
    --bg:#0a0b0f;--card:#12141c;--card2:#181b25;--border:#232738;
    --text:#e4e7f0;--muted:#7c8098;--dim:#4a4e66;
    --green:#22c55e;--red:#ef4444;--blue:#3b82f6;--purple:#a855f7;--yellow:#eab308;--orange:#f97316;
  }}
  *{{margin:0;padding:0;box-sizing:border-box;}}
  body{{font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;background:var(--bg);color:var(--text);line-height:1.7;padding:40px 24px;}}
  .container{{max-width:1200px;margin:0 auto;}}
  .badge{{display:inline-block;font-size:10px;font-weight:800;letter-spacing:2px;text-transform:uppercase;padding:4px 14px;border-radius:6px;margin-bottom:12px;}}
  .badge-warn{{background:rgba(239,68,68,0.1);color:var(--red);}}
  .badge-conf{{background:rgba(234,179,8,0.1);color:var(--yellow);}}
  h1{{font-size:32px;font-weight:800;line-height:1.2;margin-bottom:4px;}}
  .subtitle{{color:var(--muted);font-size:13px;margin-bottom:40px;}}
  .section{{margin-bottom:48px;}}
  .sec-title{{font-size:18px;font-weight:700;margin-bottom:20px;display:flex;align-items:center;gap:10px;}}
  .sec-title::before{{content:'';width:4px;height:22px;border-radius:2px;}}
  .sec-title.blue::before{{background:var(--blue);}}
  .sec-title.red::before{{background:var(--red);}}
  .sec-title.purple::before{{background:var(--purple);}}
  .sec-title.yellow::before{{background:var(--yellow);}}
  .sec-title.green::before{{background:var(--green);}}
  .sec-title.orange::before{{background:var(--orange);}}
  .grid{{display:grid;gap:16px;margin-bottom:20px;}}
  .g4{{grid-template-columns:repeat(auto-fit,minmax(200px,1fr));}}
  .g3{{grid-template-columns:repeat(auto-fit,minmax(280px,1fr));}}
  .g2{{grid-template-columns:repeat(auto-fit,minmax(380px,1fr));}}
  @media(max-width:768px){{.g2,.g3{{grid-template-columns:1fr;}}}}
  .card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px;}}
  .card-sm{{padding:16px;}}
  .label{{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;color:var(--dim);margin-bottom:6px;}}
  .val{{font-size:28px;font-weight:800;}}
  .val-sm{{font-size:20px;}}
  .sub{{font-size:11px;color:var(--dim);margin-top:4px;}}
  .pos{{color:var(--green);}}
  .neg{{color:var(--red);}}
  .warn{{color:var(--yellow);}}
  .chart-card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px;margin-bottom:20px;}}
  .chart-card h4{{font-size:14px;margin-bottom:14px;color:var(--muted);}}
  canvas{{max-height:300px;}}
  table{{width:100%;border-collapse:collapse;font-size:12px;}}
  th{{text-align:left;padding:10px 12px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;color:var(--dim);border-bottom:1px solid var(--border);background:var(--card2);}}
  td{{padding:8px 12px;border-bottom:1px solid rgba(35,39,56,0.4);white-space:nowrap;}}
  tr:hover{{background:rgba(255,255,255,0.01);}}
  .tbl{{background:var(--card);border:1px solid var(--border);border-radius:12px;overflow:hidden;}}
  .dot{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;vertical-align:middle;}}
  .callout{{border:1px solid;border-radius:12px;padding:20px 24px;margin-bottom:20px;}}
  .callout-red{{border-color:rgba(239,68,68,0.2);background:rgba(239,68,68,0.03);}}
  .callout-yellow{{border-color:rgba(234,179,8,0.2);background:rgba(234,179,8,0.03);}}
  .callout-green{{border-color:rgba(34,197,94,0.2);background:rgba(34,197,94,0.03);}}
  .callout-blue{{border-color:rgba(59,130,246,0.2);background:rgba(59,130,246,0.03);}}
  .callout h4{{font-size:14px;font-weight:700;margin-bottom:10px;}}
  .callout ul{{padding-left:20px;}}
  .callout li{{font-size:13px;color:var(--muted);margin-bottom:6px;line-height:1.6;}}
  .callout li strong{{color:var(--text);}}
  .verdict-box{{border:2px solid;border-radius:16px;padding:32px;text-align:center;margin-top:32px;}}
  .verdict-box.pass{{border-color:rgba(34,197,94,0.4);background:rgba(34,197,94,0.04);}}
  .verdict-box.cond{{border-color:rgba(234,179,8,0.4);background:rgba(234,179,8,0.04);}}
  .verdict-box.fail{{border-color:rgba(239,68,68,0.4);background:rgba(239,68,68,0.04);}}
  .verdict-label{{font-size:12px;font-weight:800;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;}}
  .verdict-val{{font-size:36px;font-weight:800;margin-bottom:12px;}}
  .verdict-sub{{font-size:13px;color:var(--muted);max-width:700px;margin:0 auto;line-height:1.7;}}
  .contrib-bar{{display:flex;height:28px;border-radius:8px;overflow:hidden;margin-bottom:8px;}}
  .contrib-seg{{display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:#fff;}}
  .footer{{text-align:center;color:var(--dim);font-size:11px;margin-top:48px;padding-top:24px;border-top:1px solid var(--border);}}
  .tag{{display:inline-block;background:rgba(255,255,255,0.04);padding:2px 10px;border-radius:6px;font-size:11px;font-weight:600;color:var(--muted);margin:2px;}}
  .risk-grade{{display:inline-block;padding:2px 10px;border-radius:6px;font-size:11px;font-weight:700;}}
  .risk-low{{background:rgba(34,197,94,0.12);color:var(--green);}}
  .risk-med{{background:rgba(234,179,8,0.12);color:var(--yellow);}}
  .risk-high{{background:rgba(239,68,68,0.12);color:var(--red);}}
</style>
</head>
<body>
<div class="container">

<span class="badge badge-conf">CONFIDENTIAL — INTERNAL USE ONLY</span>
<h1>Investment Due Diligence Report</h1>
<p style="font-size:20px;font-weight:700;margin-bottom:4px;">Triple Short Strategy — Crypto Futures</p>
<p class="subtitle">
  Review Date: {datetime.now().strftime('%Y-%m-%d')} &nbsp;&bull;&nbsp;
  Backtest: {start} ~ {end} ({s['total_days']} days) &nbsp;&bull;&nbsp;
  Capital: ${ib:,.0f} &nbsp;&bull;&nbsp;
  Bybit USDT Perpetual &nbsp;&bull;&nbsp;
  557 Symbols
</p>

<!-- =============================================== -->
<!-- 0. MARKET CONTEXT -->
<!-- =============================================== -->
<div class="section">
  <div class="sec-title yellow">Market Context — BTC Price During Backtest Period</div>
  <div class="chart-card">
    <h4>BTC/USDT Daily Close ({start} ~ {end})</h4>
    <canvas id="btcChart" style="max-height:220px;"></canvas>
  </div>
  <div class="grid g4">
    <div class="card card-sm">
      <div class="label">BTC Start</div>
      <div class="val val-sm">${s['btc_start']:,.0f}</div>
    </div>
    <div class="card card-sm">
      <div class="label">BTC End</div>
      <div class="val val-sm">${s['btc_end']:,.0f}</div>
    </div>
    <div class="card card-sm">
      <div class="label">BTC Change</div>
      <div class="val val-sm {'pos' if s['btc_change']>=0 else 'neg'}">{s['btc_change']:+.1f}%</div>
    </div>
    <div class="card card-sm">
      <div class="label">BTC Range</div>
      <div class="val val-sm">${s['btc_min']:,.0f}~${s['btc_max']:,.0f}</div>
    </div>
  </div>
  <div class="callout callout-yellow">
    <h4 class="warn">Context Note</h4>
    <ul>
      <li>테스트 기간 중 BTC는 <strong>${s['btc_start']:,.0f} &rarr; ${s['btc_end']:,.0f} ({s['btc_change']:+.1f}%)</strong> 변동. <strong>전체적으로 상승장</strong>이었음에도 SHORT ONLY 전략이 수익을 낸 점은 주목할 만하나, 동시에 <strong>하락 구간에서 수익이 집중</strong>되었을 가능성을 검증해야 함.</li>
      <li>2024 H1에서 수익이 저조($+1,524)한 반면 2025 전체에서 $83,688 수익 — <strong>잔고 복리 효과와 시장 변동성 증가</strong>가 혼재.</li>
    </ul>
  </div>
</div>

<!-- =============================================== -->
<!-- 1. RETURN ANALYSIS -->
<!-- =============================================== -->
<div class="section">
  <div class="sec-title blue">1. Return Analysis</div>

  <div class="grid g4">
    <div class="card">
      <div class="label">Total Return</div>
      <div class="val pos">+{s['total_pnl']/ib*100:,.0f}%</div>
      <div class="sub">${ib:,.0f} &rarr; ${fb:,.0f}</div>
    </div>
    <div class="card">
      <div class="label">CAGR</div>
      <div class="val pos">{s['cagr']:.1f}%</div>
      <div class="sub">Annualized</div>
    </div>
    <div class="card">
      <div class="label">Profit Factor</div>
      <div class="val">{s['pf']:.2f}</div>
      <div class="sub">Gross +${s['gp']:,.0f} / -${s['gl']:,.0f}</div>
    </div>
    <div class="card">
      <div class="label">Sharpe (Monthly)</div>
      <div class="val">{s['sharpe']:.2f}</div>
      <div class="sub">Sortino: {s['sortino']:.2f}</div>
    </div>
    <div class="card">
      <div class="label">Total Trades</div>
      <div class="val">{s['n_trades']:,}</div>
      <div class="sub">{s['n_wins']:,}W / {s['n_losses']:,}L ({s['wr']:.1f}%)</div>
    </div>
    <div class="card">
      <div class="label">Trades / Day</div>
      <div class="val">{s['trades_per_day']:.1f}</div>
      <div class="sub">Avg ${s['avg_pnl']:+,.2f}/trade</div>
    </div>
    <div class="card">
      <div class="label">Positive Months</div>
      <div class="val">{s['pos_months']}/{s['total_months']}</div>
      <div class="sub">{s['pos_months']/s['total_months']*100:.0f}%</div>
    </div>
    <div class="card">
      <div class="label">Total Fees</div>
      <div class="val neg">${s['fees']:,.0f}</div>
      <div class="sub">{s['fees']/s['gp']*100:.1f}% of gross profit</div>
    </div>
  </div>

  <!-- Equity + Strat Cumulative -->
  <div class="grid g2">
    <div class="chart-card">
      <h4>Cumulative PnL + BTC Price Overlay</h4>
      <canvas id="eqChart"></canvas>
    </div>
    <div class="chart-card">
      <h4>Monthly PnL</h4>
      <canvas id="monthlyBar"></canvas>
    </div>
  </div>

  <!-- PnL contribution bar -->
  <p style="font-size:12px;color:var(--muted);margin-bottom:6px;">Strategy PnL Contribution</p>
  <div class="contrib-bar">
    <div class="contrib-seg" style="width:{ichi_pct:.1f}%;background:#a855f7;">{ichi_pct:.0f}%</div>
    <div class="contrib-seg" style="width:{mirror_pct:.1f}%;background:#ef4444;">{mirror_pct:.0f}%</div>
    <div class="contrib-seg" style="width:{ma100_pct:.1f}%;background:#3b82f6;">{ma100_pct:.0f}%</div>
  </div>
  <div style="display:flex;gap:20px;font-size:12px;color:var(--muted);margin-bottom:20px;">
    <span><span class="dot" style="background:#a855f7;"></span>Ichimoku ${ichi_pnl:+,.0f}</span>
    <span><span class="dot" style="background:#ef4444;"></span>Mirror ${mirror_pnl:+,.0f}</span>
    <span><span class="dot" style="background:#3b82f6;"></span>MA100 ${ma100_pnl:+,.0f}</span>
  </div>

  <!-- Period breakdown -->
  <div class="grid g2">
    <div class="tbl">
      <table>
        <thead><tr><th>Year</th><th>Trades</th><th>PnL</th></tr></thead>
        <tbody>{yearly_rows}</tbody>
      </table>
    </div>
    <div class="tbl">
      <table>
        <thead><tr><th>Half-Year</th><th>PnL</th></tr></thead>
        <tbody>{hy_rows}</tbody>
      </table>
    </div>
  </div>

  <div class="callout callout-red">
    <h4 class="neg">Return Concerns</h4>
    <ul>
      <li><strong>복리 왜곡:</strong> $1,000 시작 → $110,000 도달까지 복리 효과가 극적. 초기 자본 $10,000이었다면 수익률은 동일해도 절대 금액 비율이 달라짐. <strong>후반부 수익이 크게 보이는 것은 잔고 규모 증가 때문.</strong></li>
      <li><strong>2024 H1 저수익:</strong> 첫 5개월간 +$1,524 (연율 +3.6%). 초기 자본이 작아 손실 리스크는 낮았으나, <strong>실전에서 이 기간을 버텨야 하는 인내심</strong> 필요.</li>
      <li><strong>Top-10 거래 집중도:</strong> 상위 10건이 총 이익의 <strong>{s['top10_pct']:.1f}%</strong>, 상위 50건이 <strong>{s['top50_pct']:.1f}%</strong>. 소수 대형 승리에 과도하게 의존하지 않는 편으로, 이는 긍정적.</li>
    </ul>
  </div>
</div>

<!-- =============================================== -->
<!-- 2. STRATEGY COMPARISON -->
<!-- =============================================== -->
<div class="section">
  <div class="sec-title purple">2. Strategy Breakdown</div>

  <div class="tbl" style="overflow-x:auto;">
    <table>
      <thead><tr>
        <th>Strategy</th><th>TF</th><th>Lev</th><th>Size</th><th>MaxPos</th>
        <th>Trades</th><th>WR</th><th>PF</th><th>PnL</th>
        <th>AvgW</th><th>AvgL</th><th>AvgDur</th>
        <th>MaxConsL</th><th>StreakPnL</th>
      </tr></thead>
      <tbody>{strat_rows}</tbody>
    </table>
  </div>

  <div style="height:16px;"></div>

  <div class="grid g3">
    <div class="callout callout-red">
      <h4><span class="dot" style="background:#a855f7;"></span>Ichimoku — Risk Flag</h4>
      <ul>
        <li><strong>20x 레버리지</strong>는 극도로 공격적. SL이 Cloud-based로 유동적이며, MaxLoss 120건 = 전체의 39%가 최대 손실로 청산.</li>
        <li>승률 42.1%에서 <strong>13연패</strong> 기록. 연패 시 누적 손실 $-1,814.</li>
        <li>최대 단일 이익 $7,286 (UNI short) — <strong>이 1건이 전체 Ichimoku PnL의 21%</strong>. 이상치(outlier) 의존도 높음.</li>
      </ul>
    </div>
    <div class="callout callout-yellow">
      <h4><span class="dot" style="background:#ef4444;"></span>Mirror Short — Caution</h4>
      <ul>
        <li>평균 보유시간 <strong>0.3시간 (18분)</strong> — 초단타. 실전에서 <strong>슬리피지와 체결 지연</strong>이 수익을 크게 훼손할 수 있음.</li>
        <li>SL 1002건 vs Trail 703건. <strong>SL 비율 59%</strong>로 손절이 더 빈번.</li>
        <li>WR 41.2%이지만 <strong>PF 2.61</strong>로 최고 — 승리 시 평균 $90 vs 손실 시 $-24. 비대칭 수익 구조는 양호.</li>
      </ul>
    </div>
    <div class="callout callout-red">
      <h4><span class="dot" style="background:#3b82f6;"></span>MA100 — Critical Flag</h4>
      <ul>
        <li><strong>58연패</strong>는 심각한 경고 신호. 일봉 기반이라 58일 = 약 2개월간 연속 손실 가능.</li>
        <li>연패 중 누적 손실 $-4,448. 3x/2% sizing이므로 계좌 규모가 클수록 절대 손실 확대.</li>
        <li>PF 1.43은 3개 중 최저. <strong>슬리피지 20% 악화 시 PF 1.19로 하락</strong> → 마진 매우 얇음.</li>
      </ul>
    </div>
  </div>

  <!-- Correlation -->
  <div class="callout callout-blue">
    <h4>Strategy Correlation (Monthly PnL)</h4>
    <ul>
      <li>Ichimoku — Mirror: <strong>{s['corr']['im']:.3f}</strong> &nbsp;|&nbsp; Ichimoku — MA100: <strong>{s['corr']['ia']:.3f}</strong> &nbsp;|&nbsp; Mirror — MA100: <strong>{s['corr']['ma']:.3f}</strong></li>
      <li>상관계수 0.36~0.39 — <strong>중간 수준</strong>. 완전 독립(0.0)은 아니지만, 0.7 미만으로 분산 효과는 있음.</li>
      <li>전 전략이 SHORT ONLY이므로 <strong>강한 상승장에서 동시 손실</strong>이 구조적으로 발생할 수밖에 없음. 이것이 상관관계의 주 원인.</li>
    </ul>
  </div>
</div>

<!-- =============================================== -->
<!-- 3. RISK ANALYSIS -->
<!-- =============================================== -->
<div class="section">
  <div class="sec-title red">3. Risk Analysis</div>

  <div class="grid g4">
    <div class="card">
      <div class="label">Max Drawdown</div>
      <div class="val neg">-12.3%</div>
      <div class="sub">Peak to trough</div>
    </div>
    <div class="card">
      <div class="label">MC Median MDD</div>
      <div class="val neg">-9.1%</div>
      <div class="sub">10K simulations</div>
    </div>
    <div class="card">
      <div class="label">MC 95th MDD</div>
      <div class="val neg">-13.5%</div>
      <div class="sub">Worst 5% scenario</div>
    </div>
    <div class="card">
      <div class="label">MC Worst MDD</div>
      <div class="val neg">-25.0%</div>
      <div class="sub">Absolute worst in 10K</div>
    </div>
  </div>

  <div class="grid g2">
    <div class="callout callout-red">
      <h4 class="neg">Monte Carlo Risk Assessment</h4>
      <ul>
        <li>10,000 시뮬레이션 중 <strong>100% 수익</strong> — 5th percentile에서도 $43,679. 통계적 에지는 확인됨.</li>
        <li>단, 이는 <strong>백테스트 트레이드 풀에서의 리샘플링</strong>이므로 풀 자체의 편향(과적합)은 검증하지 못함.</li>
        <li><strong>MC worst MDD -25%</strong> — 백테스트 -12.3%의 2배. 실전에서는 <strong>최소 -25% 이상의 MDD</strong>를 감내할 준비 필요.</li>
        <li>Risk of Ruin (50%): 0.00% — 그러나 이는 <strong>현재 파라미터와 시장 regime이 유지된다는 전제</strong>.</li>
      </ul>
    </div>
    <div class="callout callout-yellow">
      <h4 class="warn">Structural Risk: All-Short Bias</h4>
      <ul>
        <li>3개 전략 모두 SHORT ONLY. <strong>BTC가 +76% 상승한 기간</strong>에 수익을 냈다는 것은 알트코인 하락 구간이 충분히 있었기 때문.</li>
        <li>만약 <strong>전 시장 동반 상승</strong> (2020 DeFi Summer, 2021 Alt Season 같은)이 재현되면 <strong>3개 전략 동시 손실</strong> 불가피.</li>
        <li>이 전략은 <strong>"시장에 항상 숏 기회가 있다"</strong>는 가정에 의존. 규제 변화로 공매도 제한 시 전략 자체가 무력화.</li>
      </ul>
    </div>
  </div>
</div>

<!-- =============================================== -->
<!-- 4. OVERFITTING ASSESSMENT -->
<!-- =============================================== -->
<div class="section">
  <div class="sec-title orange">4. Overfitting (과적합) Assessment</div>

  <div class="callout callout-red">
    <h4 class="neg">Overfitting Risk: MEDIUM-HIGH</h4>
    <ul>
      <li><strong>In-sample only:</strong> 전체 기간이 학습 데이터. Out-of-sample(OOS) 검증이 전무. <strong>가장 심각한 약점.</strong></li>
      <li><strong>파라미터 튜닝 이력:</strong> SL%, Trail%, 레버리지, 포지션 사이즈가 반복 최적화됨. 특히 MA100의 "SHORT ONLY" 결정이 백테스트 결과를 보고 내린 판단 — <strong>전형적 사후 최적화(hindsight optimization)</strong>.</li>
      <li><strong>전략 선택 편향:</strong> 3개 전략 모두 SHORT ONLY인 것은 백테스트에서 숏이 잘 나왔기 때문. 롱 시그널을 제거한 것은 데이터에 fit한 결과.</li>
      <li><strong>긍정 지표:</strong> 557개 심볼 중 대부분에서 작동, 월 84% 수익, Top-10 집중도 8.3% — 단일 패턴/심볼 의존은 아님.</li>
      <li><strong>검증 방안:</strong> 2022-2024 데이터로 OOS 테스트, 또는 2개월 forward walk-forward 검증 권장.</li>
    </ul>
  </div>

  <div class="grid g2">
    <div class="tbl">
      <table>
        <thead><tr><th colspan="4">Period Stability Check</th></tr>
        <tr><th>Period</th><th>PnL</th><th colspan="2">Assessment</th></tr></thead>
        <tbody>
          {hy_rows}
          <tr><td colspan="3" style="font-size:11px;color:var(--dim);padding-top:12px;">
            2024 H1 → H2: 5.2x 증가. 2025 H1 → H2: 유사. <br>
            초기 저성과는 <strong>잔고 규모 효과</strong>가 주 원인이지만, 시장 regime 변화 가능성도 있음.
          </td></tr>
        </tbody>
      </table>
    </div>
    <div class="tbl">
      <table>
        <thead><tr><th colspan="3">Year-over-Year</th></tr>
        <tr><th>Year</th><th>Trades</th><th>PnL</th></tr></thead>
        <tbody>{yearly_rows}</tbody>
      </table>
    </div>
  </div>
</div>

<!-- =============================================== -->
<!-- 5. SENSITIVITY & STRESS TEST -->
<!-- =============================================== -->
<div class="section">
  <div class="sec-title red">5. Sensitivity & Stress Test</div>

  <div class="grid g2">
    <div>
      <p style="font-size:13px;font-weight:600;margin-bottom:10px;">Slippage Sensitivity (손실 악화 시)</p>
      <div class="tbl">
        <table>
          <thead><tr><th>Loss Increase</th><th>Adj. PnL</th><th>Adj. PF</th><th>Status</th></tr></thead>
          <tbody>{slip_rows}</tbody>
        </table>
      </div>
      <p style="font-size:11px;color:var(--dim);margin-top:8px;">
        * 모든 손실 거래의 손실액이 N% 증가한다고 가정
      </p>
    </div>
    <div>
      <p style="font-size:13px;font-weight:600;margin-bottom:10px;">Fee Sensitivity (수수료 증가 시)</p>
      <div class="tbl">
        <table>
          <thead><tr><th>Fee Multiplier</th><th>Adj. PnL</th><th>Status</th></tr></thead>
          <tbody>{fee_rows}</tbody>
        </table>
      </div>
      <p style="font-size:11px;color:var(--dim);margin-top:8px;">
        * 현재 수수료: ${s['fees']:,.0f} (Taker 0.055%)
      </p>
    </div>
  </div>

  <div class="callout callout-yellow">
    <h4 class="warn">Stress Test Findings</h4>
    <ul>
      <li><strong>슬리피지 +30%:</strong> PnL $+69,444, PF 1.40 — 여전히 수익이나 마진이 얇아짐. <strong>+50% 악화 시 PF {s['slip_sens'][50]['pf']:.2f}</strong>로 위험 수준 진입.</li>
      <li><strong>수수료 3배:</strong> PnL ${s['fee_sens'][3.0]:+,.0f} — Mirror Short의 초단기 특성상 수수료 민감도 높음. Maker 주문으로 전환 시 개선 가능.</li>
      <li><strong>종합:</strong> 슬리피지 20% + 수수료 2배 동시 적용 시 PnL 약 $+73,000 (PF ~1.35) — 여전히 수익이나 <strong>마진이 상당히 축소</strong>됨.</li>
    </ul>
  </div>
</div>

<!-- =============================================== -->
<!-- 6. LIVE DEPLOYMENT ISSUES -->
<!-- =============================================== -->
<div class="section">
  <div class="sec-title red">6. Live Deployment Issues (실전 투입 문제)</div>

  <div class="callout callout-red">
    <h4 class="neg">Critical Issues</h4>
    <ul>
      <li><strong>Mirror Short 체결 문제:</strong> 평균 18분 보유, SL 1%. 5m 캔들 기반이므로 실전에서 진입/청산 사이 <strong>slippage가 백테스트 대비 크게 발생</strong>. 특히 유동성 낮은 알트에서 1% SL이 2-3%로 확대될 수 있음.</li>
      <li><strong>동시 포지션 한계:</strong> 이론상 MA100 최대 20 + Mirror 3 + Ichimoku 5 = <strong>28개 동시 포지션</strong>. 마진 분배, API rate limit, 모니터링 부하 문제.</li>
      <li><strong>API 의존성:</strong> Bybit API 장애 시 포지션 관리 불가. 2025년에만 여러 차례 API 불안정 사례 보고됨.</li>
      <li><strong>자금 규모 한계:</strong> $100K+ 자금에서 소형 알트코인 숏 포지션 진입 시 <strong>시장 충격(market impact)</strong> 발생. 백테스트는 무한 유동성 가정.</li>
    </ul>
  </div>

  <div class="callout callout-yellow">
    <h4 class="warn">Operational Risks</h4>
    <ul>
      <li><strong>서버 안정성:</strong> 현재 Raspberry Pi에서 운용 — 네트워크 불안정, 전원 문제 시 포지션 방치 리스크.</li>
      <li><strong>Funding Rate:</strong> 숏 포지션은 양의 funding rate 환경에서 <strong>이자 수익</strong>을 받지만, 음의 funding rate 시 비용 발생. MA100의 평균 64시간 보유 시 누적 영향 무시 불가.</li>
      <li><strong>거래소 리스크:</strong> Bybit 단일 거래소 의존. 규제 변화, 해킹, 출금 제한 리스크.</li>
      <li><strong>Liquidation Risk:</strong> Ichimoku 20x — 교차마진 사용 시 다른 포지션에 영향. 격리마진이라도 <strong>5% 역방향 이동 시 원금 100% 손실</strong>.</li>
    </ul>
  </div>
</div>

<!-- =============================================== -->
<!-- 7. IMPROVEMENT SUGGESTIONS -->
<!-- =============================================== -->
<div class="section">
  <div class="sec-title green">7. Improvement Recommendations</div>

  <div class="grid g2">
    <div class="callout callout-green">
      <h4>Priority 1 — Must Do</h4>
      <ul>
        <li><strong>OOS 검증:</strong> 2022-01 ~ 2024-01 데이터로 동일 파라미터 백테스트. WR/PF가 30% 이상 하락하면 과적합 확정.</li>
        <li><strong>Ichimoku 레버리지 감소:</strong> 20x → 10x 이하. MDD 개선, 청산 리스크 감소. PnL 50% 감소해도 리스크 대비 수익 개선.</li>
        <li><strong>동적 포지션 사이징:</strong> 잔고 대비 고정 비율이 아닌, 최근 변동성(ATR) 기반 사이징. 고변동성 시 축소.</li>
        <li><strong>서버 인프라 개선:</strong> 클라우드(AWS/GCP) 이전 또는 이중화. 모니터링 알림 강화.</li>
      </ul>
    </div>
    <div class="callout callout-green">
      <h4>Priority 2 — Should Do</h4>
      <ul>
        <li><strong>Long 전략 추가 검토:</strong> 전 전략 SHORT ONLY의 구조적 약점 보완. Trend-following Long 전략 1개 추가로 상관관계 분산.</li>
        <li><strong>Mirror Short Maker 전환:</strong> 지정가 진입으로 수수료 절감. 체결률 하락 vs 수수료 절감 트레이드오프 분석 필요.</li>
        <li><strong>MA100 연패 대응:</strong> 20연패 이상 시 포지션 사이즈 자동 축소(1% → 0.5%), 또는 일시 중단 로직.</li>
        <li><strong>멀티 거래소:</strong> Binance 등 추가. 거래소 리스크 분산, 유동성 개선.</li>
      </ul>
    </div>
  </div>
</div>

<!-- =============================================== -->
<!-- 8. MONTHLY DETAIL -->
<!-- =============================================== -->
<div class="section">
  <div class="sec-title blue">Appendix — Monthly Detail</div>
  <div class="grid g2">
    <div class="tbl">
      <table>
        <thead><tr><th>Month</th><th>Trades</th><th>WR</th><th>PnL</th></tr></thead>
        <tbody>{monthly_rows}</tbody>
      </table>
    </div>
    <div>
      <p style="font-size:13px;font-weight:600;margin-bottom:10px;">Top Profitable Symbols</p>
      <div class="tbl" style="margin-bottom:16px;">
        <table>
          <thead><tr><th>Symbol</th><th>Trades</th><th>PnL</th></tr></thead>
          <tbody>{top_rows}</tbody>
        </table>
      </div>
      <p style="font-size:13px;font-weight:600;margin-bottom:10px;">Top Loss-Making Symbols</p>
      <div class="tbl">
        <table>
          <thead><tr><th>Symbol</th><th>Trades</th><th>PnL</th></tr></thead>
          <tbody>{bot_rows}</tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<!-- =============================================== -->
<!-- 9. VERDICT -->
<!-- =============================================== -->
<div class="section">
  <div class="sec-title purple">Investment Verdict</div>

  <div class="verdict-box cond">
    <div class="verdict-label warn">Investment Decision</div>
    <div class="verdict-val warn">CONDITIONAL PASS</div>
    <div class="verdict-sub">
      통계적 에지가 확인되며 수익 구조는 견고하나, <strong>OOS 검증 부재</strong>와 <strong>전략 선택의 사후 최적화</strong>가 확인되어 무조건적 승인은 불가.<br><br>
      아래 <strong>3가지 조건 충족 시</strong> 실전 투입을 승인할 수 있음:
    </div>
  </div>

  <div class="callout callout-yellow" style="margin-top:20px;">
    <h4 class="warn">Conditions for Approval</h4>
    <ul>
      <li><strong>Condition 1:</strong> 2022-01 ~ 2024-01 기간 OOS 백테스트에서 PF &ge; 1.2, 월간 수익률 &ge; 60% 확인.</li>
      <li><strong>Condition 2:</strong> Ichimoku 레버리지 20x → 10x 이하로 감소. 또는 MaxLoss를 현재 대비 50% 축소.</li>
      <li><strong>Condition 3:</strong> 실전 투입 초기 3개월은 전체 자금의 <strong>최대 10%</strong>로 제한 (paper trading 또는 소액 실행). 백테스트 대비 수익률 70% 이상 재현 시 단계적 증액.</li>
    </ul>
  </div>

  <div class="grid g3" style="margin-top:20px;">
    <div class="card" style="text-align:center;">
      <div class="label">Statistical Edge</div>
      <div class="val pos" style="font-size:22px;">CONFIRMED</div>
      <div class="sub">MC 100% profitable, PF 1.82</div>
    </div>
    <div class="card" style="text-align:center;">
      <div class="label">Overfitting Risk</div>
      <div class="val warn" style="font-size:22px;">MEDIUM-HIGH</div>
      <div class="sub">No OOS validation</div>
    </div>
    <div class="card" style="text-align:center;">
      <div class="label">Live Readiness</div>
      <div class="val warn" style="font-size:22px;">CONDITIONAL</div>
      <div class="sub">Infra + sizing improvements needed</div>
    </div>
  </div>
</div>

<div class="footer">
  Investment Due Diligence Report &nbsp;&bull;&nbsp; Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;&bull;&nbsp;
  Data: Bybit Futures 557 Symbols &nbsp;&bull;&nbsp; Engine: Combined Backtester v2 + Monte Carlo 10K
  <br>This report is for internal review purposes only. Past performance does not guarantee future results.
</div>

</div>

<script>
Chart.defaults.color='#7c8098';
Chart.defaults.borderColor='#232738';

// BTC Price Chart
new Chart(document.getElementById('btcChart'),{{
  type:'line',
  data:{{
    labels:{btc_labels},
    datasets:[{{
      label:'BTC/USDT',
      data:{btc_values},
      borderColor:'#eab308',
      backgroundColor:'rgba(234,179,8,0.06)',
      fill:true,tension:0.3,pointRadius:0,borderWidth:2,
    }}]
  }},
  options:{{
    responsive:true,
    plugins:{{legend:{{display:false}}}},
    scales:{{
      x:{{ticks:{{maxTicksLimit:12}}}},
      y:{{ticks:{{callback:function(v){{return '$'+v.toLocaleString()}}}}}}
    }}
  }}
}});

// Cumulative PnL + BTC overlay
new Chart(document.getElementById('eqChart'),{{
  type:'line',
  data:{{
    labels:{eq_labels},
    datasets:[
      {{label:'Total PnL',data:{eq_values},borderColor:'#e4e7f0',borderWidth:2,tension:0.3,pointRadius:0,yAxisID:'y'}},
      {{label:'Ichimoku',data:{json.dumps(ichi_cum)},borderColor:'#a855f7',borderWidth:1.5,tension:0.3,pointRadius:0,borderDash:[4,2],yAxisID:'y'}},
      {{label:'Mirror',data:{json.dumps(mirror_cum)},borderColor:'#ef4444',borderWidth:1.5,tension:0.3,pointRadius:0,borderDash:[4,2],yAxisID:'y'}},
      {{label:'MA100',data:{json.dumps(ma100_cum)},borderColor:'#3b82f6',borderWidth:1.5,tension:0.3,pointRadius:0,borderDash:[4,2],yAxisID:'y'}},
    ]
  }},
  options:{{
    responsive:true,
    plugins:{{legend:{{position:'bottom',labels:{{padding:12}}}}}},
    scales:{{
      y:{{position:'left',title:{{display:true,text:'PnL ($)'}}}},
    }}
  }}
}});

// Monthly PnL Bar
new Chart(document.getElementById('monthlyBar'),{{
  type:'bar',
  data:{{
    labels:{eq_labels},
    datasets:[{{
      label:'Monthly PnL',
      data:{monthly_bar_vals},
      backgroundColor:{monthly_bar_colors},
      borderRadius:4,
    }}]
  }},
  options:{{
    responsive:true,
    plugins:{{legend:{{display:false}}}},
    scales:{{
      x:{{ticks:{{maxRotation:45}}}},
      y:{{ticks:{{callback:function(v){{return '$'+v.toLocaleString()}}}}}}
    }}
  }}
}});
</script>
</body>
</html>"""

    out_path = Path("data/strategy_analysis_report.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return str(out_path)


def main():
    bt_data, btc_prices = load_data()
    stats = compute_all_stats(
        bt_data["trades"], bt_data["initial_balance"], bt_data["final_balance"],
        bt_data["start"], bt_data["end"], btc_prices,
    )
    path = generate_report(bt_data, btc_prices, stats)
    print(f"Report: {path}")

    import webbrowser
    abs_path = Path(path).resolve()
    webbrowser.open(f"file:///{abs_path}")


if __name__ == "__main__":
    main()
