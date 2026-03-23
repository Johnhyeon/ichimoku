#!/usr/bin/env python3
"""
대시보드 데이터 내보내기

봇 상태 파일들을 읽어 docs/data.json 생성 후 GitHub Pages에 푸시.
run_unified.py에서 주기적으로 호출하거나 cron으로 실행.

사용:
    python scripts/export_dashboard_data.py                        # 로컬 생성만
    python scripts/export_dashboard_data.py --push                 # 생성 + git push
    python scripts/export_dashboard_data.py --data-dir /path/data  # 데이터 경로 지정
"""

import json
import os
import sys
import subprocess
import logging
from datetime import datetime, timedelta

sys.path.insert(0, ".")

logger = logging.getLogger(__name__)

OUTPUT = "docs/data.json"


def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return None


def collect_trades(state, strategy_name):
    """state에서 trade_history 추출, strategy 라벨 부여."""
    trades = []
    for t in state.get("trade_history", []):
        trade = {
            "strategy": strategy_name,
            "symbol": t.get("symbol", ""),
            "side": t.get("side", ""),
            "entry_price": t.get("entry_price", 0),
            "exit_price": t.get("exit_price", 0),
            "pnl_usd": t.get("pnl_usd", 0),
            "pnl_pct": t.get("pnl_pct", 0),
            "reason": t.get("reason", ""),
            "closed_at": t.get("closed_at", ""),
        }
        trades.append(trade)
    return trades


def collect_positions(state, strategy_name):
    """state에서 열린 포지션 추출."""
    positions = []
    for sym, p in state.get("positions", {}).items():
        # DCA 정보: pending_dca + filled_entries 기반
        pending_dca = p.get("pending_dca", [])
        filled_entries = p.get("filled_entries", [])
        dca_total = len(filled_entries) + len(pending_dca)
        dca_filled = len(filled_entries)

        pos = {
            "strategy": strategy_name,
            "symbol": sym,
            "side": p.get("side", ""),
            "entry_price": p.get("entry_price", 0),
            "current_price": p.get("current_price", 0),
            "stop_loss": p.get("stop_loss", 0),
            "take_profit": p.get("take_profit", 0),
            "pnl_usd": p.get("pnl"),
            "dca_count": dca_total if dca_total > 1 else 0,
            "dca_filled": dca_filled if dca_total > 1 else 0,
            "size": p.get("size", 0),
            "leverage": p.get("leverage", 0),
        }
        positions.append(pos)
    return positions


def export_data(paper=False, data_dir="data"):
    """모든 상태 파일을 읽어 data.json 생성."""
    now = datetime.utcnow()

    state_files = {
        "ichimoku": os.path.join(data_dir, "bot_state.json"),
        "mirror_short": os.path.join(data_dir, "mirror_short_bot_state.json"),
        "ma100": os.path.join(data_dir, "ma100_bot_state.json"),
    }
    dca_path = os.path.join(data_dir, "dca_state.json")
    balance_path = os.path.join(data_dir, "balance_history.json")

    all_trades = []
    all_positions = []
    strategies = {}

    for name, path in state_files.items():
        state = load_json(path) or {}
        all_trades.extend(collect_trades(state, name))
        all_positions.extend(collect_positions(state, name))
        strategies[name] = {
            "running": os.path.exists(path),
            "positions": len(state.get("positions", {})),
            "total_trades": len(state.get("trade_history", [])),
        }

    # 시간순 정렬 (최신 먼저)
    all_trades.sort(key=lambda t: t.get("closed_at", ""), reverse=True)

    # DCA
    dca_state = load_json(dca_path) or {}
    dca_data = {
        "accumulated": dca_state.get("accumulated", {}),
        "last_dca_time": dca_state.get("last_dca_time"),
        "next_dca_time": None,
    }
    if dca_state.get("last_dca_time"):
        try:
            last = datetime.fromisoformat(dca_state["last_dca_time"])
            interval = dca_state.get("interval_hours", 8)
            dca_data["next_dca_time"] = (last + timedelta(hours=interval)).isoformat()
        except Exception:
            pass

    # Balance history (equity curve) - 0값 필터링 + 다운샘플링
    balance_raw = load_json(balance_path) or []
    equity_full = []
    for entry in balance_raw:
        eq = entry.get("equity", 0)
        bal = entry.get("balance", 0)
        if eq <= 0 and bal <= 0:
            continue  # 봇 꺼져있던 기간 제외
        equity_full.append({
            "time": entry.get("timestamp", ""),
            "balance": eq if eq > 0 else bal,
        })

    # 다운샘플링: 최대 200포인트 (GitHub Pages 로딩 최적화)
    if len(equity_full) > 200:
        step = len(equity_full) // 200
        equity_history = equity_full[::step]
        # 마지막 포인트 항상 포함
        if equity_history[-1] != equity_full[-1]:
            equity_history.append(equity_full[-1])
    else:
        equity_history = equity_full

    # Current balance from latest entry
    balance = {}
    if equity_full:
        latest = equity_full[-1]
        balance["total"] = latest["balance"]
        # Daily change: 24시간 전과 비교
        cutoff = (now - timedelta(hours=24)).isoformat()
        older = [e for e in equity_full if e["time"] < cutoff]
        if older:
            prev = older[-1]["balance"]
            if prev > 0:
                balance["daily_change"] = latest["balance"] - prev
                balance["daily_change_pct"] = (latest["balance"] - prev) / prev * 100

    # Unrealized PnL from positions
    total_upnl = sum(p.get("pnl_usd", 0) or 0 for p in all_positions)
    balance["unrealized_pnl"] = total_upnl

    data = {
        "updated_at": now.isoformat(),
        "mode": "paper" if paper else "live",
        "balance": balance,
        "strategies": strategies,
        "positions": all_positions,
        "trades": all_trades,
        "dca": dca_data,
        "equity_history": equity_history,
    }

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, default=str)

    logger.info(f"Dashboard data exported: {len(all_positions)} positions, {len(all_trades)} trades")
    return data


def git_push():
    """docs/data.json을 git에 커밋하고 push."""
    try:
        subprocess.run(
            ["git", "add", "-f", OUTPUT],
            cwd=".", capture_output=True, check=True
        )
        result = subprocess.run(
            ["git", "status", "--porcelain", OUTPUT],
            cwd=".", capture_output=True, text=True
        )
        if not result.stdout.strip():
            logger.debug("No changes to push")
            return

        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        subprocess.run(
            ["git", "commit", "-m", f"dashboard: update data {now}"],
            cwd=".", capture_output=True, check=True
        )
        subprocess.run(
            ["git", "push"],
            cwd=".", capture_output=True, check=True
        )
        logger.info("Dashboard data pushed to GitHub")
    except subprocess.CalledProcessError as e:
        logger.error(f"Git push failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Git push error: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    push = "--push" in sys.argv

    # --data-dir 옵션
    data_dir = "data"
    for i, arg in enumerate(sys.argv):
        if arg == "--data-dir" and i + 1 < len(sys.argv):
            data_dir = sys.argv[i + 1]

    export_data(data_dir=data_dir)
    if push:
        git_push()
    print(f"Exported to {OUTPUT}")
