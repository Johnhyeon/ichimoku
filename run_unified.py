#!/usr/bin/env python3
"""
통합 봇 - 이치모쿠 + 미러숏 + MA100 전략 동시 실행

하나의 프로세스에서 세 전략을 함께 실행합니다:
  - 이치모쿠: 4시간봉 기반 SHORT 전략 (레버리지 20x)
  - 미러숏: 5분봉 기반 SHORT 전략 (레버리지 5x)
  - MA100: 일봉 기반 LONG+SHORT 전략 (레버리지 5x)

텔레그램 봇 1개로 통합 관리하므로 409 Conflict 없이 동작합니다.

실행 예시:
    python run_unified.py --paper      # 페이퍼 모드
    python run_unified.py              # 실거래 (메인넷)
    python run_unified.py --testnet    # 테스트넷
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from typing import Dict

from src.bybit_client import BybitClient
from src.telegram_bot import TelegramNotifier, TelegramBot
from src.trader import IchimokuTrader
from src.surge_trader import SurgeTrader
from src.ma100_trader import MA100Trader
from src.balance_tracker import BalanceTracker
from src.chart_generator import ChartGenerator

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("unified_bot.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


class UnifiedTrader:
    """이치모쿠 + 미러숏 + MA100 통합 트레이더"""

    def __init__(
        self,
        paper: bool = False,
        testnet: bool = False,
        initial_balance: float = 1000.0,
        daily_loss_limit_pct: float = 20.0,
        surge_max_positions: int = 3,
        ma100_max_positions: int = 5
    ):
        self.paper = paper
        self.testnet = testnet

        # 공유 리소스
        self.client = BybitClient(testnet=testnet)
        self.notifier = TelegramNotifier()
        self.telegram_bot = TelegramBot(self.notifier)
        self.balance_tracker = BalanceTracker()
        self.chart_generator = ChartGenerator()

        # 이치모쿠 전략 (공유 리소스 주입)
        self.ichimoku = IchimokuTrader(
            paper=paper,
            testnet=testnet,
            client=self.client,
            notifier=self.notifier,
            telegram_bot=self.telegram_bot
        )

        # MA100 전략 (공유 리소스 주입, 이치모쿠+미러숏 심볼 제외)
        self.ma100 = MA100Trader(
            paper=paper,
            testnet=testnet,
            max_positions=ma100_max_positions,
            client=self.client,
            notifier=self.notifier,
            telegram_bot=self.telegram_bot,
            get_excluded_symbols=lambda: set(self.ichimoku.positions.keys()) | set(self.surge.positions.keys())
        )

        # 미러숏 전략 (공유 리소스 주입, 이치모쿠+MA100 심볼 제외)
        self.surge = SurgeTrader(
            paper=paper,
            testnet=testnet,
            initial_balance=initial_balance,
            daily_loss_limit_pct=daily_loss_limit_pct,
            max_positions=surge_max_positions,
            client=self.client,
            notifier=self.notifier,
            telegram_bot=self.telegram_bot,
            get_excluded_symbols=lambda: set(self.ichimoku.positions.keys()) | set(self.ma100.positions.keys())
        )

        # 텔레그램 콜백을 통합 메서드로 재등록
        self.telegram_bot.set_callbacks(
            get_balance=self._get_balance,
            get_positions=self._get_all_positions,
            get_trade_history=self._get_all_trade_history,
            stop_bot=self._stop_all,
            start_bot=self._resume_all,
            sync_positions=self._sync_all
        )

        # 이치모쿠 분석 콜백은 그대로 유지
        self.telegram_bot.set_analysis_callbacks(
            get_market_report=self.ichimoku._get_market_report,
            get_no_entry_report=self.ichimoku._get_no_entry_report,
            get_watch_report=self.ichimoku._get_watch_report,
            get_chart=self.ichimoku._get_chart,
            get_overview_chart=self.ichimoku._get_overview_chart,
            chat_response=self.ichimoku._chat_response
        )

        # 거래정보 콜백도 이치모쿠에서 유지 (공유 client 사용)
        self.telegram_bot.set_trading_callbacks(
            get_funding_rates=self.ichimoku._get_funding_rates,
            get_position_sl_tp=self.ichimoku._get_position_sl_tp,
            set_position_sl_tp=self.ichimoku._set_position_sl_tp,
            get_account_stats=self.ichimoku._get_account_stats,
            get_trade_history_exchange=self.ichimoku._get_trade_history_from_exchange,
            get_transaction_log=self.ichimoku._get_transaction_log
        )

        # 잔고 차트 콜백
        self.telegram_bot.get_balance_chart_callback = self._get_balance_chart

        # 전략별 제어 콜백
        self.telegram_bot.set_strategy_callbacks(
            get_strategy_status=self._get_strategy_status,
            stop_ichimoku=self.ichimoku.stop,
            start_ichimoku=self.ichimoku.resume,
            stop_surge=self.surge.stop,
            start_surge=self.surge.resume,
            stop_ma100=self.ma100.stop,
            start_ma100=self.ma100.resume
        )

        # 설정 콜백
        self.telegram_bot.set_settings_callbacks(
            get_settings=self._get_settings,
            set_leverage=self._set_leverage,
            set_position_pct=self._set_position_pct
        )

    def _get_strategy_status(self) -> dict:
        """전략별 상태 조회"""
        return {
            'ichimoku_running': self.ichimoku.running,
            'surge_running': self.surge.running,
            'ma100_running': self.ma100.running,
            'surge_daily_pnl': self.surge.daily_pnl,
            'surge_daily_limit': self.surge.daily_loss_limit,
            'surge_positions': len(self.surge.positions),
            'surge_max_positions': self.surge.max_positions,
            'ichimoku_positions': len(self.ichimoku.positions),
            'ma100_positions': len(self.ma100.positions),
            'ma100_max_positions': self.ma100.max_positions,
        }

    def _get_settings(self) -> dict:
        """현재 레버리지/진입비율 반환"""
        return {
            'ich_leverage': self.ichimoku.leverage,
            'ich_pct': self.ichimoku.position_pct * 100,
            'surge_leverage': self.surge.params['leverage'],
            'surge_pct': self.surge.params['position_pct'] * 100,
            'ma100_leverage': self.ma100.params['leverage'],
            'ma100_pct': self.ma100.params['position_pct'] * 100,
        }

    def _set_leverage(self, strategy: str, value: int):
        """전략별 레버리지 변경"""
        if strategy == 'ichimoku':
            self.ichimoku.leverage = value
            logger.info(f"[설정] 이치모쿠 레버리지 → {value}x")
        elif strategy == 'surge':
            self.surge.params['leverage'] = value
            logger.info(f"[설정] 미러숏 레버리지 → {value}x")
        elif strategy == 'ma100':
            self.ma100.params['leverage'] = value
            logger.info(f"[설정] MA100 레버리지 → {value}x")

    def _set_position_pct(self, strategy: str, value: float):
        """전략별 진입비율 변경"""
        if strategy == 'ichimoku':
            self.ichimoku.position_pct = value
            logger.info(f"[설정] 이치모쿠 진입비율 → {value*100:.0f}%")
        elif strategy == 'surge':
            self.surge.params['position_pct'] = value
            logger.info(f"[설정] 미러숏 진입비율 → {value*100:.0f}%")
        elif strategy == 'ma100':
            self.ma100.params['position_pct'] = value
            logger.info(f"[설정] MA100 진입비율 → {value*100:.0f}%")

    def _get_balance(self) -> dict:
        """잔고 조회 (공유 client)"""
        return self.ichimoku._get_balance_full()

    def _get_all_positions(self) -> list:
        """세 전략의 포지션 합산"""
        positions = []
        for p in self.ichimoku._get_positions_list():
            p['strategy'] = 'ichimoku'
            positions.append(p)
        for p in self.surge._get_positions_list():
            p['strategy'] = 'mirror_short'
            positions.append(p)
        for p in self.ma100._get_positions_list():
            p['strategy'] = 'ma100'
            positions.append(p)
        return positions

    def _get_all_trade_history(self) -> list:
        """세 전략의 거래 이력 합산"""
        history = []
        for h in self.ichimoku._get_trade_history():
            h_copy = h.copy()
            h_copy['strategy'] = 'ichimoku'
            history.append(h_copy)
        for h in self.surge._get_trade_history():
            h_copy = h.copy()
            h_copy['strategy'] = 'mirror_short'
            history.append(h_copy)
        for h in self.ma100._get_trade_history():
            h_copy = h.copy()
            h_copy['strategy'] = 'ma100'
            history.append(h_copy)
        # 시간순 정렬
        history.sort(key=lambda x: x.get('closed_at') or datetime.min, reverse=True)
        return history

    def _stop_all(self):
        """세 전략 모두 중지"""
        self.ichimoku.stop()
        self.surge.stop()
        self.ma100.stop()

    def _resume_all(self):
        """세 전략 모두 재개"""
        self.ichimoku.resume()
        self.surge.resume()
        self.ma100.resume()

    def _sync_all(self) -> dict:
        """세 전략 모두 포지션 동기화"""
        result1 = self.ichimoku._check_manual_closes() or {"synced": 0, "positions": 0}
        result2 = self.surge._check_manual_closes() or {"synced": 0, "positions": 0}
        result3 = self.ma100._check_manual_closes() or {"synced": 0, "positions": 0}
        return {
            "synced": result1.get("synced", 0) + result2.get("synced", 0) + result3.get("synced", 0),
            "positions": result1.get("positions", 0) + result2.get("positions", 0) + result3.get("positions", 0)
        }

    def _record_balance(self):
        """현재 잔고를 트래커에 기록"""
        try:
            balance = self._get_balance()
            if balance:
                self.balance_tracker.record(balance)
        except Exception as e:
            logger.debug(f"잔고 기록 실패 (무시): {e}")

    async def _get_balance_chart(self) -> bytes:
        """잔고 추이 차트 생성"""
        history = self.balance_tracker.get_history(days=7)
        return self.chart_generator.generate_balance_chart(history)

    async def _ichimoku_loop(self):
        """이치모쿠 루프 (4시간봉 갱신 시마다)"""
        while True:
            if self.ichimoku.running:
                try:
                    self.ichimoku.run_once()

                    # 시황 리포트 전송
                    await self.ichimoku._send_periodic_report()

                    # 잔고 기록
                    self._record_balance()

                except Exception as e:
                    logger.error(f"[이치모쿠] 루프 오류: {e}")
                    self.notifier.send_sync(f"⚠️ 이치모쿠 오류: {e}")

            # 다음 4시간봉 캔들까지 대기
            next_candle = self.ichimoku.data_fetcher.get_next_candle_time("4h")
            now = datetime.utcnow()
            sleep_seconds = max(60, (next_candle - now).total_seconds())
            logger.info(f"[이치모쿠] 다음 캔들까지 {sleep_seconds/60:.1f}분 대기")
            await asyncio.sleep(sleep_seconds)

    async def _surge_loop(self):
        """미러숏 루프 (5분마다)"""
        while True:
            if self.surge.running:
                try:
                    self.surge.run_once()

                    # 잔고 기록
                    self._record_balance()

                except Exception as e:
                    logger.error(f"[미러숏] 루프 오류: {e}")
                    self.notifier.send_sync(f"⚠️ 미러숏 오류: {e}")

            logger.info("[미러숏] 5분 대기...")
            await asyncio.sleep(300)

    async def _ma100_scan_loop(self):
        """MA100 시그널 스캔 루프 (하루 1회, 일봉 갱신 시)"""
        while True:
            if self.ma100.running:
                try:
                    self.ma100.run_once()
                    self._record_balance()
                except Exception as e:
                    logger.error(f"[MA100] 스캔 오류: {e}")
                    self.notifier.send_sync(f"⚠️ MA100 스캔 오류: {e}")

            next_candle = self.ma100.data_fetcher.get_next_candle_time("1d")
            now = datetime.utcnow()
            sleep_seconds = max(60, (next_candle - now).total_seconds())
            logger.info(f"[MA100] 다음 일봉까지 {sleep_seconds/3600:.1f}시간 대기")
            await asyncio.sleep(sleep_seconds)

    async def _ma100_position_loop(self):
        """MA100 포지션 모니터링 루프 (5분마다 청산 감지)"""
        while True:
            if self.ma100.running and self.ma100.positions:
                try:
                    self.ma100.check_positions()
                except Exception as e:
                    logger.error(f"[MA100] 포지션 체크 오류: {e}")

            await asyncio.sleep(300)  # 5분

    async def run_async(self):
        """세 전략을 하나의 asyncio 루프에서 실행"""
        mode = "PAPER" if self.paper else "LIVE"
        logger.info(f"통합 봇 시작 [{mode}]")

        # 텔레그램 봇 시작 (1개만)
        await self.telegram_bot.start_polling()

        # 시작 알림
        self.notifier.send_sync(
            f"🚀 <b>통합 봇 시작</b> [{mode}]\n\n"
            f"⛩️ 이치모쿠: 4시간봉 SHORT (20x)\n"
            f"📉 미러숏: 5분봉 SHORT (5x)\n"
            f"📊 MA100: 일봉 LONG+SHORT (5x)"
        )

        self.ichimoku.running = True
        self.surge.running = True
        self.ma100.running = True

        # 전략을 별도 asyncio Task로 실행
        ichimoku_task = asyncio.create_task(self._ichimoku_loop())
        surge_task = asyncio.create_task(self._surge_loop())
        ma100_scan_task = asyncio.create_task(self._ma100_scan_loop())
        ma100_pos_task = asyncio.create_task(self._ma100_position_loop())

        try:
            await asyncio.gather(ichimoku_task, surge_task, ma100_scan_task, ma100_pos_task)
        except asyncio.CancelledError:
            logger.info("통합 봇 종료")
        finally:
            await self.telegram_bot.stop_polling()

    def run(self):
        """동기 실행"""
        asyncio.run(self.run_async())


def main():
    parser = argparse.ArgumentParser(
        description="통합 봇 - 이치모쿠 + 미러숏 + MA100 전략 동시 실행"
    )
    parser.add_argument(
        "--paper", action="store_true",
        help="페이퍼 모드 (실제 주문 안 보냄)"
    )
    parser.add_argument(
        "--testnet", action="store_true",
        help="Bybit 테스트넷 사용"
    )
    parser.add_argument(
        "--initial", type=float, default=1000.0,
        help="미러숏 초기 운용 자금 (기본: 1000 USDT)"
    )
    parser.add_argument(
        "--loss-limit", type=float, default=20.0,
        help="미러숏 일일 손실 한도 %% (기본: 20%%)"
    )
    parser.add_argument(
        "--surge-max-positions", type=int, default=3,
        help="미러숏 최대 동시 포지션 수 (기본: 3개)"
    )
    parser.add_argument(
        "--ma100-max-positions", type=int, default=5,
        help="MA100 최대 동시 포지션 수 (기본: 5개)"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("통합 봇 - 이치모쿠 + 미러숏 + MA100 전략")
    logger.info("=" * 60)

    mode = "PAPER" if args.paper else "LIVE"
    net = "TESTNET" if args.testnet else "MAINNET"
    logger.info(f"모드: {mode}, 네트워크: {net}")

    try:
        trader = UnifiedTrader(
            paper=args.paper,
            testnet=args.testnet,
            initial_balance=args.initial,
            daily_loss_limit_pct=args.loss_limit,
            surge_max_positions=args.surge_max_positions,
            ma100_max_positions=args.ma100_max_positions
        )
        trader.run()

    except KeyboardInterrupt:
        logger.info("사용자 인터럽트로 종료")
    except Exception as e:
        logger.error(f"치명적 오류: {e}")
        raise


if __name__ == "__main__":
    main()
