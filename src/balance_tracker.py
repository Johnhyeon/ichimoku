"""잔고 추이 기록 모듈"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class BalanceTracker:
    """잔고 스냅샷을 기록하고 조회하는 트래커"""

    FILE = "data/balance_history.json"
    MIN_INTERVAL_SEC = 300  # 5분

    def __init__(self):
        os.makedirs(os.path.dirname(self.FILE), exist_ok=True)
        self._history = self._load()

    def _load(self) -> list:
        """파일에서 이력 로드"""
        try:
            if os.path.exists(self.FILE):
                with open(self.FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"잔고 이력 로드 실패: {e}")
        return []

    def _save(self):
        """이력을 파일에 저장"""
        try:
            with open(self.FILE, "w", encoding="utf-8") as f:
                json.dump(self._history, f, ensure_ascii=False, indent=1)
        except Exception as e:
            logger.error(f"잔고 이력 저장 실패: {e}")

    def record(self, balance_dict: dict):
        """
        잔고 스냅샷 기록 (최소 5분 간격)

        Args:
            balance_dict: {total, free, used, unrealized_pnl, equity, ...}
        """
        now = datetime.utcnow()
        now_iso = now.isoformat()

        # 중복 기록 방지 (마지막 기록으로부터 5분 이내면 스킵)
        if self._history:
            try:
                last_ts = datetime.fromisoformat(self._history[-1]["timestamp"])
                if (now - last_ts).total_seconds() < self.MIN_INTERVAL_SEC:
                    return
            except (KeyError, ValueError):
                pass

        equity = float(balance_dict.get("equity", balance_dict.get("total", 0)))
        balance = float(balance_dict.get("total", 0))
        unrealized_pnl = float(balance_dict.get("unrealized_pnl", 0))

        snapshot = {
            "timestamp": now_iso,
            "equity": equity,
            "balance": balance,
            "unrealized_pnl": unrealized_pnl,
        }
        self._history.append(snapshot)
        self._save()

    def get_history(self, days: int = 7) -> list:
        """
        최근 N일 잔고 이력 반환

        Args:
            days: 조회할 일수 (기본 7일)

        Returns:
            [{timestamp, equity, balance, unrealized_pnl}, ...]
        """
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        return [h for h in self._history if h["timestamp"] >= cutoff]
