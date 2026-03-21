"""
스팟 DCA(Dollar Cost Averaging) 자동 적립 봇

BTC/ETH를 일정 주기로 자동 매수하여 장기 보유합니다.
선물 숏 전략과 자연스러운 헷지 효과를 제공합니다.

구조:
  - 기본 DCA: 8시간마다 $10 고정 매수 (BTC 40% / ETH 60%)
  - 주간 보너스: 일요일 00시(KST)에 한 주 선물 수익 정산 → 수익의 10% 스팟 매수
"""

import logging
import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.bybit_client import BybitClient
from src.telegram_bot import TelegramNotifier

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))

DCA_PARAMS = {
    'enabled': True,
    'interval_hours': 8,          # DCA 주기
    'base_amount_usdt': 10.0,     # 기본 매수액
    'btc_ratio': 0.4,             # BTC 40%
    'eth_ratio': 0.6,             # ETH 60%
    'weekly_bonus_pct': 0.10,     # 주간 선물 수익의 10%를 보너스 매수
    'weekly_bonus_day': 6,        # 0=월 ... 6=일 (일요일)
    'weekly_bonus_hour_kst': 0,   # KST 기준 시간 (00시)
    'min_futures_reserve': 500.0, # 선물 마진 최소 유보액
    'min_order_usdt': 1.0,        # Bybit 스팟 최소 주문 ($1)
}

STATE_FILE = "data/dca_state.json"


class SpotDCA:
    """스팟 DCA 자동 적립 봇"""

    def __init__(
        self,
        paper: bool = False,
        client: BybitClient = None,
        notifier: TelegramNotifier = None,
        interval_hours: float = None,
        base_amount: float = None,
        min_reserve: float = None,
    ):
        self.paper = paper
        self.client = client
        self.notifier = notifier
        self.running = False

        # 파라미터 (CLI 오버라이드 가능)
        self.params = DCA_PARAMS.copy()
        if interval_hours is not None:
            self.params['interval_hours'] = interval_hours
        if base_amount is not None:
            self.params['base_amount_usdt'] = base_amount
        if min_reserve is not None:
            self.params['min_futures_reserve'] = min_reserve

        # 상태 로드
        self.state = self._load_state()

        logger.info(
            f"[DCA] 초기화 완료 (paper={paper}, "
            f"주기={self.params['interval_hours']}h, "
            f"기본액=${self.params['base_amount_usdt']}, "
            f"유보=${self.params['min_futures_reserve']})"
        )

    def _load_state(self) -> dict:
        """상태 파일 로드"""
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"[DCA] 상태 파일 로드 실패: {e}")

        return {
            "last_dca_time": None,
            "last_weekly_bonus_time": None,
            "last_weekly_pnl_check_ms": 0,
            "accumulated": {
                "BTC": {"total_qty": 0.0, "total_invested_usdt": 0.0, "buy_count": 0},
                "ETH": {"total_qty": 0.0, "total_invested_usdt": 0.0, "buy_count": 0},
            },
            "history": [],
        }

    def _save_state(self):
        """상태 파일 저장"""
        try:
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
            with open(STATE_FILE, 'w') as f:
                json.dump(self.state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"[DCA] 상태 저장 실패: {e}")

    def is_time_for_dca(self) -> bool:
        """DCA 실행 시간인지 확인"""
        last_time = self.state.get("last_dca_time")
        if not last_time:
            return True

        try:
            last_dt = datetime.fromisoformat(last_time)
        except (ValueError, TypeError):
            return True

        interval = timedelta(hours=self.params['interval_hours'])
        return datetime.utcnow() >= last_dt + interval

    def is_time_for_weekly_bonus(self) -> bool:
        """주간 보너스 실행 시간인지 확인 (일요일 00시 KST)"""
        now_kst = datetime.now(KST)
        target_day = self.params['weekly_bonus_day']
        target_hour = self.params['weekly_bonus_hour_kst']

        # 요일/시간 체크
        if now_kst.weekday() != target_day or now_kst.hour != target_hour:
            return False

        # 이번 주에 이미 실행했는지 확인
        last_time = self.state.get("last_weekly_bonus_time")
        if not last_time:
            return True

        try:
            last_dt = datetime.fromisoformat(last_time)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            # 마지막 실행 후 6일 이상 지났으면 OK
            return (datetime.now(timezone.utc) - last_dt) > timedelta(days=6)
        except (ValueError, TypeError):
            return True

    def _get_weekly_futures_profit(self) -> float:
        """지난 주간 선물 실현손익 합산"""
        try:
            last_check_ms = self.state.get("last_weekly_pnl_check_ms", 0)
            records = self.client.get_closed_pnl(limit=200)

            profit = 0.0
            for r in records:
                created_at = r.get('created_at', 0)
                if created_at > last_check_ms:
                    profit += r.get('closed_pnl', 0)

            return profit
        except Exception as e:
            logger.warning(f"[DCA] 주간 선물 손익 조회 실패: {e}")
            return 0.0

    def _ensure_balance(self, required_usdt: float) -> bool:
        """DCA에 필요한 잔고 확보 (Unified 가용 → Funding 이체 순)"""
        try:
            balance = self.client.get_balance()
            free = balance.get('free', 0)
            reserve = self.params['min_futures_reserve']
            available = free - reserve

            if available >= required_usdt:
                return True

            # Unified 잔고 부족 → Funding에서 이체 시도
            shortfall = required_usdt - max(available, 0)
            funding_bal = self.client.get_funding_balance('USDT')
            logger.info(f"[DCA] Unified 부족 (가용=${available:.2f}), Funding=${funding_bal:.2f}")

            if funding_bal >= shortfall:
                ok = self.client.internal_transfer('USDT', shortfall, 'FUND', 'UNIFIED')
                if ok:
                    logger.info(f"[DCA] Funding → Unified ${shortfall:.2f} 이체 완료")
                    return True

            logger.warning(
                f"[DCA] 잔고 부족: 필요=${required_usdt:.2f}, "
                f"Unified 가용=${available:.2f}, Funding=${funding_bal:.2f}"
            )
            return False
        except Exception as e:
            logger.error(f"[DCA] 잔고 확인 실패: {e}")
            return False

    def _execute_spot_buy(self, asset: str, usdt_amount: float) -> Optional[dict]:
        """스팟 시장가 매수"""
        symbol = f"{asset}/USDT"
        min_order = self.params['min_order_usdt']

        if usdt_amount < min_order:
            logger.info(f"[DCA] {asset} 매수 금액 ${usdt_amount:.2f} < 최소 ${min_order} → 스킵")
            return None

        if self.paper:
            try:
                ticker = self.client.get_ticker(f"{asset}/USDT:USDT")
                price = ticker['last']
                qty = usdt_amount / price
                logger.info(f"[DCA][PAPER] {asset} 매수: ${usdt_amount:.2f} @ ${price:,.2f} = {qty:.6f}")
                return {
                    'symbol': symbol,
                    'side': 'buy',
                    'amount': qty,
                    'price': price,
                    'cost': usdt_amount,
                    'status': 'paper',
                }
            except Exception as e:
                logger.error(f"[DCA][PAPER] {asset} 시세 조회 실패: {e}")
                return None
        else:
            try:
                result = self.client.spot_market_buy(symbol, usdt_amount)
                logger.info(
                    f"[DCA] {asset} 스팟 매수 완료: "
                    f"${usdt_amount:.2f} @ ${result.get('price', 0):,.2f}"
                )
                return result
            except Exception as e:
                logger.error(f"[DCA] {asset} 스팟 매수 실패: {e}")
                return None

    def _update_accumulation(self, asset: str, qty: float, cost: float):
        """적립 누적 업데이트"""
        acc = self.state["accumulated"].setdefault(
            asset, {"total_qty": 0.0, "total_invested_usdt": 0.0, "buy_count": 0}
        )
        acc["total_qty"] += qty
        acc["total_invested_usdt"] += cost
        acc["buy_count"] += 1

    def _do_buy(self, total_amount: float, label: str) -> dict:
        """BTC/ETH 매수 실행 (기본 DCA, 주간 보너스 공용)"""
        btc_amount = total_amount * self.params['btc_ratio']
        eth_amount = total_amount * self.params['eth_ratio']

        results = {}
        history_entry = {
            "time": datetime.utcnow().isoformat(),
            "type": label,
            "total_usdt": total_amount,
            "buys": {},
        }

        for asset, amount in [("BTC", btc_amount), ("ETH", eth_amount)]:
            result = self._execute_spot_buy(asset, amount)
            if result:
                qty = result.get('amount', 0)
                price = result.get('price', 0)
                cost = result.get('cost', amount)
                self._update_accumulation(asset, qty, cost)
                results[asset] = result
                history_entry["buys"][asset] = {
                    "qty": qty,
                    "price": price,
                    "cost": cost,
                }

        # 히스토리 추가 (최근 100건만 유지)
        if history_entry["buys"]:
            self.state.setdefault("history", []).append(history_entry)
            self.state["history"] = self.state["history"][-100:]

        return results

    def run_once(self):
        """기본 DCA 1사이클 실행 (고정 금액)"""
        if not self.params['enabled']:
            return

        if not self.is_time_for_dca():
            return

        total_amount = self.params['base_amount_usdt']
        logger.info(f"[DCA] === 기본 DCA 시작: ${total_amount:.2f} ===")

        if not self._ensure_balance(total_amount):
            msg = (
                f"⚠️ [DCA] 잔고 부족으로 스킵\n"
                f"필요: ${total_amount:.2f}\n"
                f"유보: ${self.params['min_futures_reserve']:.0f}"
            )
            logger.warning(msg)
            if self.notifier:
                self.notifier.send_sync(msg)
            return

        results = self._do_buy(total_amount, "dca")

        self.state["last_dca_time"] = datetime.utcnow().isoformat()
        self._save_state()

        if results and self.notifier:
            summary = self._get_buy_summary(results, total_amount, "DCA 적립")
            self.notifier.send_sync(summary)

        logger.info("[DCA] === 기본 DCA 완료 ===")

    def run_weekly_bonus(self):
        """주간 보너스 실행 (일요일 00시 KST)"""
        if not self.params['enabled']:
            return

        if not self.is_time_for_weekly_bonus():
            return

        # 주간 선물 수익 조회
        profit = self._get_weekly_futures_profit()
        logger.info(f"[DCA] === 주간 보너스 체크: 선물 수익 ${profit:.2f} ===")

        if profit <= 0:
            logger.info("[DCA] 주간 선물 수익 없음 → 보너스 스킵")
            self.state["last_weekly_bonus_time"] = datetime.utcnow().isoformat()
            self.state["last_weekly_pnl_check_ms"] = int(time.time() * 1000)
            self._save_state()
            return

        bonus_amount = profit * self.params['weekly_bonus_pct']

        if bonus_amount < self.params['min_order_usdt']:
            logger.info(f"[DCA] 보너스 ${bonus_amount:.2f} < 최소주문 → 스킵")
            self.state["last_weekly_bonus_time"] = datetime.utcnow().isoformat()
            self.state["last_weekly_pnl_check_ms"] = int(time.time() * 1000)
            self._save_state()
            return

        if not self._ensure_balance(bonus_amount):
            logger.warning(f"[DCA] 주간 보너스 잔고 부족: ${bonus_amount:.2f}")
            return

        results = self._do_buy(bonus_amount, "weekly_bonus")

        self.state["last_weekly_bonus_time"] = datetime.utcnow().isoformat()
        self.state["last_weekly_pnl_check_ms"] = int(time.time() * 1000)
        self._save_state()

        if results and self.notifier:
            summary = self._get_buy_summary(
                results, bonus_amount,
                f"주간 보너스 (수익 ${profit:.0f}의 {self.params['weekly_bonus_pct']*100:.0f}%)"
            )
            self.notifier.send_sync(summary)

        logger.info(f"[DCA] === 주간 보너스 완료: ${bonus_amount:.2f} ===")

    def _get_buy_summary(self, results: dict, total_amount: float, title: str) -> str:
        """매수 결과 텔레그램 요약"""
        mode = "[PAPER] " if self.paper else ""
        text = f"🛒 <b>{mode}{title}</b>\n\n"

        for asset, result in results.items():
            qty = result.get('amount', 0)
            price = result.get('price', 0)
            cost = result.get('cost', 0)
            text += f"{'₿' if asset == 'BTC' else 'Ξ'} <b>{asset}</b>: "
            text += f"${cost:.2f} @ ${price:,.2f} = {qty:.6f}\n"

        text += f"\n💵 이번 총액: <code>${total_amount:.2f}</code>\n"

        # 누적 현황
        acc = self.state.get("accumulated", {})
        text += "\n📊 <b>누적 현황</b>\n"
        for asset in ["BTC", "ETH"]:
            a = acc.get(asset, {})
            total_qty = a.get("total_qty", 0)
            total_inv = a.get("total_invested_usdt", 0)
            count = a.get("buy_count", 0)
            avg_price = total_inv / total_qty if total_qty > 0 else 0
            emoji = "₿" if asset == "BTC" else "Ξ"
            text += (
                f"{emoji} {asset}: {total_qty:.6f} "
                f"(${total_inv:.2f}, {count}회, 평균 ${avg_price:,.2f})\n"
            )

        return text.strip()

    def get_accumulation_summary(self) -> str:
        """적립 현황 요약 텍스트 (텔레그램 대시보드용)"""
        acc = self.state.get("accumulated", {})

        lines = []
        for asset in ["BTC", "ETH"]:
            a = acc.get(asset, {})
            total_qty = a.get("total_qty", 0)
            total_inv = a.get("total_invested_usdt", 0)
            count = a.get("buy_count", 0)

            if count == 0:
                continue

            avg_price = total_inv / total_qty if total_qty > 0 else 0
            emoji = "₿" if asset == "BTC" else "Ξ"

            try:
                ticker = self.client.get_ticker(f"{asset}/USDT:USDT")
                current_price = ticker['last']
                current_value = total_qty * current_price
                pnl = current_value - total_inv
                pnl_pct = (pnl / total_inv * 100) if total_inv > 0 else 0
                pnl_sign = "+" if pnl >= 0 else ""
                lines.append(
                    f"{emoji} {asset}: {total_qty:.6f} "
                    f"(평균${avg_price:,.0f}, {pnl_sign}{pnl_pct:.1f}%)"
                )
            except Exception:
                lines.append(
                    f"{emoji} {asset}: {total_qty:.6f} "
                    f"(${total_inv:.2f}, {count}회)"
                )

        if not lines:
            return "적립 내역 없음"

        return "\n".join(lines)

    def get_params(self) -> dict:
        """현재 DCA 파라미터 반환"""
        return self.params.copy()

    def set_param(self, key: str, value) -> str:
        """DCA 파라미터 변경. 변경 결과 텍스트 반환."""
        if key not in self.params:
            return f"❌ 알 수 없는 파라미터: {key}"
        old = self.params[key]
        self.params[key] = type(old)(value)
        logger.info(f"[DCA] 파라미터 변경: {key} = {old} → {self.params[key]}")
        return f"✅ {key}: {old} → {self.params[key]}"

    def get_detailed_status(self) -> str:
        """상세 DCA 현황 (텔레그램용)"""
        p = self.params
        acc = self.state.get("accumulated", {})
        last_time = self.state.get("last_dca_time")
        last_bonus = self.state.get("last_weekly_bonus_time")

        # 설정 정보
        lines = [
            f"⚙️ <b>설정</b>",
            f"  기본: ${p['base_amount_usdt']:.0f} / {p['interval_hours']}시간",
            f"  비율: BTC {p['btc_ratio']*100:.0f}% / ETH {p['eth_ratio']*100:.0f}%",
            f"  주간보너스: 일요일 00시 선물수익의 {p['weekly_bonus_pct']*100:.0f}%",
            f"  마진유보: ${p['min_futures_reserve']:,.0f}",
            "",
        ]

        # 적립 현황
        total_invested = 0
        total_value = 0
        for asset in ["BTC", "ETH"]:
            a = acc.get(asset, {})
            qty = a.get("total_qty", 0)
            inv = a.get("total_invested_usdt", 0)
            count = a.get("buy_count", 0)
            total_invested += inv

            if count == 0:
                continue

            avg = inv / qty if qty > 0 else 0
            emoji = "₿" if asset == "BTC" else "Ξ"

            try:
                ticker = self.client.get_ticker(f"{asset}/USDT:USDT")
                price = ticker['last']
                val = qty * price
                total_value += val
                pnl = val - inv
                pnl_pct = pnl / inv * 100 if inv > 0 else 0
                sign = "+" if pnl >= 0 else ""
                lines.append(
                    f"{emoji} <b>{asset}</b>\n"
                    f"  보유: {qty:.6f} ({count}회 매수)\n"
                    f"  평균단가: ${avg:,.0f}\n"
                    f"  투자: ${inv:,.0f} → 평가: ${val:,.0f} ({sign}{pnl_pct:.1f}%)"
                )
            except Exception:
                lines.append(
                    f"{emoji} <b>{asset}</b>: {qty:.6f} (${inv:,.0f}, {count}회)"
                )

        if total_invested > 0:
            total_pnl = total_value - total_invested
            sign = "+" if total_pnl >= 0 else ""
            lines.append(f"\n💰 합계: ${total_invested:,.0f} → ${total_value:,.0f} ({sign}${total_pnl:,.0f})")

        if last_time:
            lines.append(f"\n🕐 마지막 DCA: {last_time}")
        if last_bonus:
            lines.append(f"🎁 마지막 보너스: {last_bonus}")

        return "\n".join(lines)

    def stop(self):
        """DCA 중지"""
        self.running = False
        logger.info("[DCA] 중지됨")

    def resume(self):
        """DCA 재개"""
        self.running = True
        logger.info("[DCA] 재개됨")
