"""
스팟 DCA(Dollar Cost Averaging) 자동 적립 봇

BTC/ETH를 일정 주기로 자동 매수하여 장기 보유합니다.
선물 숏 전략과 자연스러운 헷지 효과를 제공합니다.

파라미터:
  - interval_hours: DCA 주기 (기본 8시간)
  - base_amount_usdt: 기본 매수액 (기본 $10)
  - btc_ratio / eth_ratio: BTC 40%, ETH 60%
  - profit_bonus_pct: 선물 수익의 10%를 추가 매수
  - min_futures_reserve: 선물 마진 최소 유보액 ($500)
"""

import logging
import json
import os
import time
from datetime import datetime, timedelta
from typing import Optional

from src.bybit_client import BybitClient
from src.telegram_bot import TelegramNotifier

logger = logging.getLogger(__name__)

DCA_PARAMS = {
    'enabled': True,
    'interval_hours': 8,          # DCA 주기
    'base_amount_usdt': 10.0,     # 기본 매수액
    'btc_ratio': 0.4,             # BTC 40%
    'eth_ratio': 0.6,             # ETH 60%
    'profit_bonus_pct': 0.10,     # 선물 수익의 10%를 추가 매수
    'min_futures_reserve': 500.0, # 선물 마진 최소 유보액
    'min_order_usdt': 5.0,        # Bybit 최소 주문
    'min_balance_to_start': 10000.0,  # 이 잔고 이상일 때만 DCA 시작
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
            "last_pnl_check_ms": 0,
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

    def _get_futures_profit_since_last(self) -> float:
        """마지막 DCA 이후 선물 실현손익 합산"""
        try:
            last_check_ms = self.state.get("last_pnl_check_ms", 0)
            records = self.client.get_closed_pnl(limit=50)

            profit = 0.0
            for r in records:
                created_at = r.get('created_at', 0)
                if created_at > last_check_ms:
                    profit += r.get('closed_pnl', 0)

            return profit
        except Exception as e:
            logger.warning(f"[DCA] 선물 손익 조회 실패: {e}")
            return 0.0

    def _calculate_dca_amount(self) -> float:
        """DCA 매수 금액 계산 (기본 + 보너스)"""
        base = self.params['base_amount_usdt']

        # 선물 수익 보너스 계산
        bonus = 0.0
        futures_profit = self._get_futures_profit_since_last()
        if futures_profit > 0:
            bonus = futures_profit * self.params['profit_bonus_pct']
            logger.info(f"[DCA] 선물 수익 ${futures_profit:.2f} → 보너스 ${bonus:.2f}")

        total = base + bonus
        return total

    def _check_balance_sufficient(self, required_usdt: float) -> bool:
        """잔고 충분한지 확인 (선물 마진 유보 고려)"""
        try:
            balance = self.client.get_balance()
            free = balance.get('free', 0)
            reserve = self.params['min_futures_reserve']

            available = free - reserve
            if available < required_usdt:
                logger.warning(
                    f"[DCA] 잔고 부족: 가용=${free:.2f}, "
                    f"유보=${reserve:.2f}, 필요=${required_usdt:.2f}"
                )
                return False
            return True
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
            # Paper 모드: 현재가 기준 시뮬레이션
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
            # 실거래
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

    def run_once(self):
        """DCA 1사이클 실행"""
        if not self.params['enabled']:
            return

        if not self.is_time_for_dca():
            return

        # 0. 최소 잔고 확인
        min_bal = self.params.get('min_balance_to_start', 0)
        if min_bal > 0:
            try:
                balance = self.client.get_balance()
                total_bal = balance.get('total', 0)
                if total_bal < min_bal:
                    logger.info(f"[DCA] 잔고 ${total_bal:,.0f} < ${min_bal:,.0f} → DCA 대기 중")
                    return
            except Exception as e:
                logger.warning(f"[DCA] 잔고 확인 실패: {e}")
                return

        logger.info("[DCA] ═══ DCA 사이클 시작 ═══")

        # 1. DCA 금액 계산
        total_amount = self._calculate_dca_amount()
        logger.info(f"[DCA] 이번 매수 총액: ${total_amount:.2f}")

        # 2. 잔고 확인
        if not self._check_balance_sufficient(total_amount):
            msg = (
                f"⚠️ [DCA] 잔고 부족으로 스킵\n"
                f"필요: ${total_amount:.2f}\n"
                f"유보: ${self.params['min_futures_reserve']:.0f}"
            )
            logger.warning(msg)
            if self.notifier:
                self.notifier.send_sync(msg)
            return

        # 3. BTC/ETH 각각 매수
        btc_amount = total_amount * self.params['btc_ratio']
        eth_amount = total_amount * self.params['eth_ratio']

        results = {}
        now = datetime.utcnow()
        history_entry = {
            "time": now.isoformat(),
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

        # 4. 상태 업데이트
        self.state["last_dca_time"] = now.isoformat()
        self.state["last_pnl_check_ms"] = int(time.time() * 1000)

        # 히스토리 추가 (최근 100건만 유지)
        if history_entry["buys"]:
            self.state["history"].append(history_entry)
            self.state["history"] = self.state["history"][-100:]

        self._save_state()

        # 5. 텔레그램 알림
        if results and self.notifier:
            summary = self._get_buy_summary(results, total_amount)
            self.notifier.send_sync(summary)

        logger.info("[DCA] ═══ DCA 사이클 완료 ═══")

    def _get_buy_summary(self, results: dict, total_amount: float) -> str:
        """매수 결과 텔레그램 요약"""
        mode = "[PAPER] " if self.paper else ""
        text = f"🛒 <b>{mode}DCA 적립 완료</b>\n\n"

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
        last_time = self.state.get("last_dca_time")

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

            # 현재 평가액 계산
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

        # 설정 정보
        lines = [
            f"⚙️ <b>설정</b>",
            f"  매수액: ${p['base_amount_usdt']:.0f} / {p['interval_hours']}시간",
            f"  비율: BTC {p['btc_ratio']*100:.0f}% / ETH {p['eth_ratio']*100:.0f}%",
            f"  보너스: 선물수익의 {p['profit_bonus_pct']*100:.0f}%",
            f"  최소잔고: ${p['min_balance_to_start']:,.0f}",
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

        # 마지막 매수 시간
        if last_time:
            lines.append(f"\n🕐 마지막 매수: {last_time}")

        return "\n".join(lines)

    def stop(self):
        """DCA 중지"""
        self.running = False
        logger.info("[DCA] 중지됨")

    def resume(self):
        """DCA 재개"""
        self.running = True
        logger.info("[DCA] 재개됨")
