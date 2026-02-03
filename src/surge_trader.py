"""
Early Surge 전략 실시간 트레이더

백테스트에서 검증된 초기 급등 감지 전략을 실시간으로 운영합니다.

주요 특징:
  - 5분봉 기반 초기 급등 감지
  - 거래량 폭발 + 가격 급등 필터링
  - 트레일링 스톱으로 수익 보호
  - 일일 손실 한도 안전장치
  - 텔레그램 실시간 알림

안전장치:
  - 일일 최대 손실 한도 (초기 자금 대비 %)
  - 최대 동시 포지션 수 제한
  - 레버리지 제한 (기본 5x)
  - 포지션 크기 제한 (자산의 3%)

실행 예시:
    # 페이퍼 모드 (시뮬레이션)
    python live_surge.py --paper

    # 실제 거래 (신중히!)
    python live_surge.py

    # 테스트넷
    python live_surge.py --testnet
"""

import logging
import time
import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import pandas as pd

from src.bybit_client import BybitClient
from src.data_fetcher import DataFetcher
from src.early_surge_detector import EarlySurgeDetector, EARLY_SURGE_PARAMS
from src.telegram_bot import TelegramNotifier, TelegramBot
from src.strategy import MAJOR_COINS

logger = logging.getLogger(__name__)


class SurgeTrader:
    """Early Surge 전략 실시간 트레이더"""

    def __init__(
        self,
        paper: bool = False,
        testnet: bool = False,
        initial_balance: float = 1000.0,
        daily_loss_limit_pct: float = 20.0,
        max_positions: int = 3
    ):
        """
        Args:
            paper: 페이퍼 모드 (시뮬레이션)
            testnet: 테스트넷 사용
            initial_balance: 초기 잔고 (실제 거래 시 참고용)
            daily_loss_limit_pct: 일일 손실 한도 % (초기 자금 대비)
            max_positions: 최대 동시 포지션 수
        """
        self.paper = paper
        self.testnet = testnet
        self.running = False

        # 안전장치 설정
        self.initial_balance = initial_balance
        self.daily_loss_limit = initial_balance * (daily_loss_limit_pct / 100)
        self.max_positions = max_positions
        self.daily_loss_limit_pct = daily_loss_limit_pct

        # 일일 손실 추적
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.utcnow().date()

        # 바이빗 클라이언트
        self.client = BybitClient(testnet=testnet)
        self.data_fetcher = DataFetcher(self.client)

        # Early Surge Detector
        self.detector = EarlySurgeDetector(self.data_fetcher, EARLY_SURGE_PARAMS)
        self.params = EARLY_SURGE_PARAMS

        # 텔레그램
        self.notifier = TelegramNotifier()
        self.telegram_bot = TelegramBot(self.notifier)
        self.telegram_bot.set_callbacks(
            get_balance=self._get_balance_full,
            get_positions=self._get_positions_list,
            get_trade_history=self._get_trade_history,
            stop_bot=self.stop,
            start_bot=self.resume,
            sync_positions=self._check_manual_closes
        )

        # 포지션 상태
        self.positions: Dict[str, dict] = {}
        self.last_exit_times: Dict[str, datetime] = {}

        # 거래 이력
        self.trade_history: list = []

        # 상태 저장 파일 경로
        self.state_file = "data/surge_bot_state.json"

        # 시작 로그
        mode = "PAPER" if self.paper else "LIVE"
        net = "TESTNET" if self.testnet else "MAINNET"
        logger.info(f"SurgeTrader 시작 - 모드: {mode}, 네트워크: {net}")
        logger.info(f"초기 자금: ${self.initial_balance:,.0f}")
        logger.info(f"일일 손실 한도: ${self.daily_loss_limit:,.0f} ({self.daily_loss_limit_pct}%)")
        logger.info(f"최대 포지션: {self.max_positions}개")
        logger.info(f"레버리지: {self.params['leverage']}x, 포지션 크기: {self.params['position_pct']*100}%")

        # 거래소에서 기존 포지션 동기화
        self._sync_positions()

    def _save_state(self):
        """상태를 파일에 저장"""
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)

            # datetime을 문자열로 변환
            positions_to_save = {}
            for symbol, pos in self.positions.items():
                pos_copy = pos.copy()
                if pos_copy.get("entry_time"):
                    pos_copy["entry_time"] = pos_copy["entry_time"].isoformat()
                positions_to_save[symbol] = pos_copy

            last_exits_to_save = {}
            for symbol, dt in self.last_exit_times.items():
                last_exits_to_save[symbol] = dt.isoformat()

            history_to_save = []
            for h in self.trade_history:
                h_copy = h.copy()
                if h_copy.get("closed_at"):
                    h_copy["closed_at"] = h_copy["closed_at"].isoformat()
                history_to_save.append(h_copy)

            state = {
                "positions": positions_to_save,
                "last_exit_times": last_exits_to_save,
                "trade_history": history_to_save,
                "daily_pnl": self.daily_pnl,
                "last_reset_date": self.last_reset_date.isoformat(),
                "saved_at": datetime.utcnow().isoformat()
            }

            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)

            logger.debug("상태 저장 완료")

        except Exception as e:
            logger.error(f"상태 저장 실패: {e}")

    def _load_state(self):
        """저장된 상태 불러오기"""
        if not os.path.exists(self.state_file):
            logger.info("저장된 상태 파일 없음")
            return False

        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            # 포지션 복원
            for symbol, pos in state.get("positions", {}).items():
                if pos.get("entry_time"):
                    pos["entry_time"] = datetime.fromisoformat(pos["entry_time"])
                self.positions[symbol] = pos

            # 마지막 청산 시간 복원
            for symbol, dt_str in state.get("last_exit_times", {}).items():
                self.last_exit_times[symbol] = datetime.fromisoformat(dt_str)

            # 거래 이력 복원
            for h in state.get("trade_history", []):
                if h.get("closed_at"):
                    h["closed_at"] = datetime.fromisoformat(h["closed_at"])
                self.trade_history.append(h)

            # 일일 PnL 복원
            self.daily_pnl = state.get("daily_pnl", 0.0)
            last_reset = state.get("last_reset_date", datetime.utcnow().date().isoformat())
            self.last_reset_date = datetime.fromisoformat(last_reset).date()

            saved_at = state.get("saved_at", "알 수 없음")
            logger.info(f"저장된 상태 불러옴 (저장 시각: {saved_at})")
            logger.info(f"  - 포지션 {len(self.positions)}개")
            logger.info(f"  - 일일 PnL: ${self.daily_pnl:+.2f}")
            logger.info(f"  - 거래 이력 {len(self.trade_history)}건")

            return True

        except Exception as e:
            logger.error(f"상태 불러오기 실패: {e}")
            return False

    def _sync_positions(self):
        """거래소에서 기존 포지션 동기화"""
        # 저장된 상태 먼저 불러오기
        saved_state_loaded = self._load_state()
        saved_positions = self.positions.copy()

        if self.paper:
            logger.info("페이퍼 모드 - 거래소 동기화 스킵")
            return

        try:
            # 거래소에서 실제 포지션 조회
            exchange_positions = self.client.get_all_positions()
            exchange_symbols = {pos["symbol"] for pos in exchange_positions}

            # 거래소에 없는 저장된 포지션 제거
            for symbol in list(self.positions.keys()):
                if symbol not in exchange_symbols:
                    logger.info(f"거래소에 없는 포지션 제거: {symbol}")
                    self.positions.pop(symbol, None)

            # 거래소 포지션과 동기화
            for pos in exchange_positions:
                symbol = pos["symbol"]

                # Early Surge 전략은 알트코인 전체 대상
                entry_price = pos["entry_price"]
                side = pos["side"]
                size = pos["size"]
                pnl = pos["pnl"]

                # 저장된 포지션이 있으면 그 정보 유지
                if symbol in saved_positions:
                    saved = saved_positions[symbol]
                    # 진입가가 같으면 저장된 설정 유지
                    if abs(saved.get("entry_price", 0) - entry_price) < 0.01:
                        self.positions[symbol] = saved
                        self.positions[symbol]["size"] = size
                        self.positions[symbol]["pnl"] = pnl
                        logger.info(f"저장된 포지션 복원: {symbol}")
                        continue

                # 저장된 정보 없으면 기본값 설정
                stop_loss = entry_price * (1 - self.params["sl_pct"] / 100)
                take_profit = entry_price * (1 + self.params["tp_pct"] / 100)

                self.positions[symbol] = {
                    "symbol": symbol,
                    "side": side,
                    "entry_price": entry_price,
                    "entry_time": None,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "highest": entry_price,
                    "lowest": entry_price,
                    "trail_stop": stop_loss,
                    "trailing": False,
                    "size": size,
                    "pnl": pnl,
                }
                logger.info(f"새 포지션 동기화: {symbol}")

            if self.positions:
                logger.info(f"포지션 {len(self.positions)}개 동기화 완료")
            else:
                logger.info("동기화된 포지션 없음")

            self._save_state()

        except Exception as e:
            logger.error(f"포지션 동기화 실패: {e}")

    def _reset_daily_pnl_if_needed(self):
        """자정이 지나면 일일 PnL 리셋"""
        today = datetime.utcnow().date()
        if today > self.last_reset_date:
            logger.info(f"일일 PnL 리셋: ${self.daily_pnl:+.2f} → $0.00")
            self.daily_pnl = 0.0
            self.last_reset_date = today
            self._save_state()

    def _check_daily_loss_limit(self) -> bool:
        """일일 손실 한도 체크"""
        self._reset_daily_pnl_if_needed()

        if self.daily_pnl <= -self.daily_loss_limit:
            logger.warning(f"⚠️  일일 손실 한도 도달! PnL: ${self.daily_pnl:+.2f} / 한도: ${-self.daily_loss_limit:.2f}")
            self.notifier.send_sync(
                f"🚨 <b>일일 손실 한도 도달</b>\n\n"
                f"오늘 손실: ${self.daily_pnl:+.2f}\n"
                f"한도: ${-self.daily_loss_limit:.2f}\n\n"
                f"오늘은 더 이상 거래하지 않습니다.\n"
                f"내일 자동으로 재개됩니다."
            )
            return True
        return False

    def _get_balance_full(self) -> dict:
        """USDT 전체 잔고 정보"""
        try:
            return self.client.get_balance()
        except Exception as e:
            logger.error(f"잔고 조회 실패: {e}")
            return {"total": 0, "free": 0, "used": 0, "unrealized_pnl": 0, "equity": 0}

    def _get_balance_free(self) -> float:
        """USDT 사용 가능 잔고"""
        try:
            balance = self.client.get_balance()
            return float(balance.get("free", 0))
        except Exception as e:
            logger.error(f"잔고 조회 실패: {e}")
            return 0.0

    def _get_positions_list(self) -> list:
        """현재 포지션 목록 (실시간 PnL 포함)"""
        if not self.positions:
            return []

        if self.paper:
            return list(self.positions.values())

        try:
            exchange_positions = self.client.get_all_positions()
            pos_map = {p["symbol"]: p for p in exchange_positions}

            result = []
            for pos in self.positions.values():
                pos_copy = pos.copy()
                ex_pos = pos_map.get(pos["symbol"], {})

                entry = float(pos.get("entry_price", 0))
                current_price = float(ex_pos.get("mark_price", 0))
                pnl_usd = float(ex_pos.get("pnl", 0))
                side = pos.get("side", "long")
                sl = float(pos.get("stop_loss", 0))
                tp = float(pos.get("take_profit", 0))

                # 수익률 계산
                if entry > 0 and current_price > 0:
                    if side == "long":
                        pnl_pct = (current_price - entry) / entry * 100 * self.params['leverage']
                    else:
                        pnl_pct = (entry - current_price) / entry * 100 * self.params['leverage']
                else:
                    pnl_pct = 0

                pos_copy["pnl"] = pnl_usd
                pos_copy["pnl_pct"] = pnl_pct
                pos_copy["current_price"] = current_price

                result.append(pos_copy)

            return result
        except Exception as e:
            logger.warning(f"실시간 PnL 조회 실패: {e}")
            return list(self.positions.values())

    def _get_trade_history(self) -> list:
        """거래 이력 반환"""
        return self.trade_history

    def _check_manual_closes(self) -> dict:
        """수동 청산 감지 및 거래 이력 기록"""
        result = {"synced": 0, "positions": len(self.positions)}

        if self.paper or not self.positions:
            return result

        try:
            exchange_positions = self.client.get_all_positions()
            exchange_symbols = {pos["symbol"] for pos in exchange_positions}

            closed_symbols = []
            for symbol in list(self.positions.keys()):
                if symbol not in exchange_symbols:
                    closed_symbols.append(symbol)

            if not closed_symbols:
                return result

            closed_pnl_list = self.client.get_closed_pnl(limit=50)
            closed_pnl_map = {}
            for pnl in closed_pnl_list:
                sym = pnl['symbol']
                if sym not in closed_pnl_map:
                    closed_pnl_map[sym] = pnl

            for symbol in closed_symbols:
                pos = self.positions[symbol]
                side = pos["side"]
                entry = float(pos["entry_price"])
                qty = float(pos.get("size", 0))

                if symbol in closed_pnl_map:
                    pnl_record = closed_pnl_map[symbol]
                    exit_price = pnl_record['exit_price']
                    pnl_usd = pnl_record['closed_pnl']
                    if entry > 0 and qty > 0:
                        pnl_pct = pnl_usd / (entry * qty / self.params['leverage']) * 100
                    else:
                        pnl_pct = 0
                    reason = "수동 청산"
                else:
                    try:
                        ticker = self.client.get_ticker(symbol)
                        exit_price = ticker["last"]
                    except:
                        exit_price = entry

                    if side == "long":
                        pnl_pct = (exit_price - entry) / entry * 100 * self.params['leverage']
                    else:
                        pnl_pct = (entry - exit_price) / entry * 100 * self.params['leverage']
                    pnl_usd = pnl_pct / 100 * (entry * qty) / self.params['leverage']
                    reason = "수동 청산 (추정)"

                # 일일 PnL 업데이트
                self.daily_pnl += pnl_usd

                short_sym = symbol.split('/')[0]
                logger.info(f"[MANUAL CLOSE] {short_sym} 수동 청산 감지 | PnL: {pnl_pct:+.2f}% (${pnl_usd:+.2f})")

                self.notifier.notify_exit(symbol, side, entry, exit_price, pnl_pct, pnl_usd, reason)

                self.trade_history.append({
                    "symbol": symbol,
                    "side": side,
                    "entry_price": entry,
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct,
                    "pnl_usd": pnl_usd,
                    "reason": reason,
                    "closed_at": datetime.utcnow(),
                })

                if len(self.trade_history) > 20:
                    self.trade_history = self.trade_history[-20:]

                self.positions.pop(symbol, None)
                self.last_exit_times[symbol] = datetime.utcnow()
                result["synced"] += 1

            if closed_symbols:
                self._save_state()

            result["positions"] = len(self.positions)
            return result

        except Exception as e:
            logger.error(f"수동 청산 감지 실패: {e}")
            return result

    def _calc_order_quantity(self, price: float, free_balance: float) -> float:
        """주문 수량 계산"""
        if free_balance <= 0:
            return 0.0

        margin = free_balance * self.params['position_pct']
        position_value = margin * self.params['leverage']
        qty = position_value / price
        qty = round(qty, 3)
        return qty

    def _open_position(self, signal: dict, free_balance: float) -> float:
        """포지션 오픈"""
        symbol = signal["symbol"]
        side = signal["side"]
        price = signal["price"]
        stop_loss = signal["stop_loss"]
        take_profit = signal["take_profit"]

        qty = self._calc_order_quantity(price, free_balance)
        if qty <= 0:
            logger.warning("주문 수량이 0 이하입니다.")
            return 0.0

        logger.info(f"[ENTRY] {symbol} {side.upper()} | Price=${price:.4f}, Qty={qty}, SL=${stop_loss:.4f}, TP=${take_profit:.4f}")

        if not self.paper:
            try:
                self.client.set_leverage(symbol, self.params['leverage'])
            except Exception as e:
                logger.warning(f"레버리지 설정 실패: {e}")

            try:
                order_side = "buy" if side == "long" else "sell"
                self.client.market_order_with_sl_tp(
                    symbol, order_side, qty,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
            except Exception as e:
                logger.error(f"진입 실패 ({symbol}): {e}")
                self.notifier.notify_error(f"진입 실패: {symbol}\n{e}")
                return 0.0

        # 포지션 상태 업데이트
        self.positions[symbol] = {
            "symbol": symbol,
            "side": side,
            "entry_price": price,
            "entry_time": datetime.utcnow(),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "highest": price,
            "lowest": price,
            "trail_stop": stop_loss,
            "trailing": False,
            "size": qty,
        }

        # 텔레그램 알림
        surge_info = signal.get('surge_info', {})
        vol_ratio = surge_info.get('volume_ratio', 0)
        price_change = surge_info.get('price_change', 0)

        short_sym = symbol.split('/')[0]
        message = (
            f"🚀 <b>급등 진입: {short_sym}</b>\n\n"
            f"진입가: ${price:.4f}\n"
            f"수량: {qty}\n"
            f"레버리지: {self.params['leverage']}x\n\n"
            f"📊 급등 시그널\n"
            f"거래량: {vol_ratio:.1f}배\n"
            f"가격 상승: +{price_change:.1f}%\n\n"
            f"손절: ${stop_loss:.4f} (-{self.params['sl_pct']}%)\n"
            f"익절: ${take_profit:.4f} (+{self.params['tp_pct']}%)"
        )
        self.notifier.send_sync(message)

        # 상태 저장
        self._save_state()

        used_margin = (price * qty) / self.params['leverage']
        return used_margin

    def _close_position(self, symbol: str, exit_info: dict):
        """포지션 청산"""
        pos = self.positions.get(symbol)
        if not pos:
            return

        side = pos["side"]
        qty = float(pos["size"])
        entry = float(pos["entry_price"])
        price = exit_info["price"]
        reason = exit_info["reason"]

        logger.info(f"[EXIT] {symbol} | Reason={reason}, Price=${price:.4f}")

        if not self.paper:
            try:
                order_side = "sell" if side == "long" else "buy"
                self.client.market_order(symbol, order_side, qty)
            except Exception as e:
                logger.error(f"청산 실패 ({symbol}): {e}")
                self.notifier.notify_error(f"청산 실패: {symbol}\n{e}")
                return

        # PnL 계산
        if side == "long":
            pnl_pct = (price - entry) / entry * 100 * self.params['leverage']
        else:
            pnl_pct = (entry - price) / entry * 100 * self.params['leverage']

        pnl_usd = pnl_pct / 100 * (entry * qty) / self.params['leverage']

        # 일일 PnL 업데이트
        self.daily_pnl += pnl_usd

        # 텔레그램 알림
        short_sym = symbol.split('/')[0]
        emoji = "💰" if pnl_pct >= 0 else "💸"
        message = (
            f"{emoji} <b>청산: {short_sym}</b>\n\n"
            f"진입가: ${entry:.4f}\n"
            f"청산가: ${price:.4f}\n"
            f"사유: {reason}\n\n"
            f"수익률: {pnl_pct:+.2f}%\n"
            f"수익: ${pnl_usd:+.2f}\n\n"
            f"오늘 총 수익: ${self.daily_pnl:+.2f}"
        )
        self.notifier.send_sync(message)

        # 거래 이력 저장
        self.trade_history.append({
            "symbol": symbol,
            "side": side,
            "entry_price": entry,
            "exit_price": price,
            "pnl_pct": pnl_pct,
            "pnl_usd": pnl_usd,
            "reason": reason,
            "closed_at": datetime.utcnow(),
        })

        if len(self.trade_history) > 20:
            self.trade_history = self.trade_history[-20:]

        # 상태 업데이트
        self.positions.pop(symbol, None)
        self.last_exit_times[symbol] = datetime.utcnow()

        # 상태 저장
        self._save_state()

    def _check_exit_signals(self, symbol: str, df: pd.DataFrame):
        """청산 신호 체크 (트레일링 스톱 포함)"""
        pos = self.positions.get(symbol)
        if not pos:
            return

        current_price = float(df.iloc[-1]['close'])
        entry_price = float(pos['entry_price'])
        stop_loss = float(pos['stop_loss'])
        take_profit = float(pos['take_profit'])
        side = pos['side']

        # 현재 수익률
        if side == "long":
            pnl_pct = (current_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100

        # 손절 체크
        if side == "long" and current_price <= stop_loss:
            self._close_position(symbol, {"price": current_price, "reason": "손절"})
            return
        elif side == "short" and current_price >= stop_loss:
            self._close_position(symbol, {"price": current_price, "reason": "손절"})
            return

        # 익절 체크
        if side == "long" and current_price >= take_profit:
            self._close_position(symbol, {"price": current_price, "reason": "익절"})
            return
        elif side == "short" and current_price <= take_profit:
            self._close_position(symbol, {"price": current_price, "reason": "익절"})
            return

        # 트레일링 스톱 활성화
        trail_start_pct = self.params['trail_start_pct']
        trail_pct = self.params['trail_pct']

        if pnl_pct >= trail_start_pct:
            if not pos.get('trailing'):
                pos['trailing'] = True
                logger.info(f"[TRAIL] {symbol} 트레일링 스톱 활성화 (수익률: {pnl_pct:.1f}%)")

            # 최고가 업데이트
            if side == "long":
                if current_price > pos['highest']:
                    pos['highest'] = current_price
                    pos['trail_stop'] = current_price * (1 - trail_pct / 100)
                    logger.debug(f"[TRAIL] {symbol} 트레일링 스톱 업데이트: ${pos['trail_stop']:.4f}")

                # 트레일링 스톱 도달
                if current_price <= pos['trail_stop']:
                    self._close_position(symbol, {"price": current_price, "reason": "트레일링 스톱"})
                    return
            else:
                if current_price < pos['lowest']:
                    pos['lowest'] = current_price
                    pos['trail_stop'] = current_price * (1 + trail_pct / 100)
                    logger.debug(f"[TRAIL] {symbol} 트레일링 스톱 업데이트: ${pos['trail_stop']:.4f}")

                if current_price >= pos['trail_stop']:
                    self._close_position(symbol, {"price": current_price, "reason": "트레일링 스톱"})
                    return

    def run_once(self):
        """한 번 스캔 및 실행"""
        # 일일 손실 한도 체크
        if self._check_daily_loss_limit():
            logger.info("[WAIT] 일일 손실 한도 도달 - 거래 중지")
            return

        # 수동 청산 감지
        self._check_manual_closes()

        # 기존 포지션 청산 체크
        for symbol in list(self.positions.keys()):
            df = self.detector.get_5m_data(symbol, limit=50)
            if df is not None:
                self._check_exit_signals(symbol, df)

        # 최대 포지션 수 체크
        if len(self.positions) >= self.max_positions:
            logger.info(f"[WAIT] 최대 포지션 수 도달 ({self.max_positions}개)")
            return

        # 잔고 확인
        try:
            free_balance = self._get_balance_free()
        except Exception:
            free_balance = 0.0

        if free_balance <= 0:
            logger.warning("[WAIT] 사용 가능 잔고 없음")
            return

        # Early Surge 스캔
        logger.info("[SCAN] 초기 급등 스캔 시작...")

        # USDT 무기한 선물 전체 스캔 (빠른 스캔)
        try:
            markets = self.client.exchange.fetch_markets()
            usdt_perps = [
                m['symbol'] for m in markets
                if m.get('settle') == 'USDT'
                and m.get('type') == 'swap'
                and m.get('active', True)
            ]
            logger.info(f"스캔 대상: {len(usdt_perps)}개 코인")
        except:
            usdt_perps = list(MAJOR_COINS)[:50]  # 실패 시 주요 코인만

        # 빠른 스캔 (샘플링)
        import random
        scan_symbols = random.sample(usdt_perps, min(100, len(usdt_perps)))

        signals = []
        for symbol in scan_symbols:
            # 이미 포지션 있으면 스킵
            if symbol in self.positions:
                continue

            # 최근 청산한 코인은 재진입 쿨타임
            last_exit = self.last_exit_times.get(symbol)
            if last_exit:
                cooldown = timedelta(minutes=30)
                if datetime.utcnow() - last_exit < cooldown:
                    continue

            try:
                signal = self.detector.get_entry_signal(symbol)
                if signal:
                    signals.append(signal)
                    surge_info = signal['surge_info']
                    logger.info(
                        f"[SURGE] {symbol} | "
                        f"Vol={surge_info['volume_ratio']:.1f}x "
                        f"Change={surge_info['price_change']:.1f}%"
                    )
            except Exception as e:
                logger.debug(f"스캔 실패 ({symbol}): {e}")
                continue

        if not signals:
            logger.info("[WAIT] 급등 신호 없음")
            return

        # 거래량 비율 순으로 정렬
        signals.sort(key=lambda s: -s['surge_info']['volume_ratio'])

        # 최대 포지션 수까지만 진입
        for signal in signals:
            if len(self.positions) >= self.max_positions:
                break

            used_margin = self._open_position(signal, free_balance)
            if used_margin > 0:
                free_balance -= used_margin

            if free_balance <= 0:
                break

    def stop(self):
        """봇 중지"""
        self.running = False
        logger.info("봇 중지됨")

    def resume(self):
        """봇 재시작"""
        self.running = True
        logger.info("봇 재시작됨")

    async def run_async(self):
        """비동기 실행 루프"""
        mode = "PAPER" if self.paper else "LIVE"
        logger.info(f"SurgeTrader 루프 시작 [{mode}]")

        # 텔레그램 봇 시작
        await self.telegram_bot.start_polling()

        # 시작 알림
        self.notifier.send_sync(
            f"🚀 <b>Early Surge Bot 시작</b>\n\n"
            f"모드: {mode}\n"
            f"초기 자금: ${self.initial_balance:,.0f}\n"
            f"일일 손실 한도: ${self.daily_loss_limit:,.0f} ({self.daily_loss_limit_pct}%)\n"
            f"최대 포지션: {self.max_positions}개\n"
            f"레버리지: {self.params['leverage']}x"
        )

        self.running = True

        try:
            while True:
                if self.running:
                    try:
                        self.run_once()
                    except Exception as e:
                        logger.error(f"루프 오류: {e}")
                        self.notifier.notify_error(str(e))

                # 5분마다 스캔
                logger.info("[SLEEP] 5분 대기...")
                await asyncio.sleep(300)  # 5분

        except asyncio.CancelledError:
            logger.info("봇 종료")
        finally:
            await self.telegram_bot.stop_polling()

    def run(self):
        """동기 실행"""
        asyncio.run(self.run_async())
