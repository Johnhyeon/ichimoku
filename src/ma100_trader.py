"""
MA100 터치 거부 전략 실시간 트레이더 (SHORT ONLY)

일봉 MA100 하향 기울기에서 터치 거부 시그널로 숏 진입하는 전략입니다.
백테스트: 1D 숏온리 609일간 +1001%, PF 1.45, 승률 55.6%.

주요 특징:
  - 일봉 MA100 터치 감지 (하락 추세 숏만 진입)
  - 트레일링 스톱으로 수익 보호
  - 텔레그램 실시간 알림

실행 예시:
    python run_unified.py --paper
"""

import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional
import pandas as pd

from src.bybit_client import BybitClient
from src.data_fetcher import DataFetcher
from src.strategy import STABLECOINS, fmt_price as _fmt_price
from src.telegram_bot import TelegramNotifier, TelegramBot
from src.trade_logger import TradeLogger

logger = logging.getLogger(__name__)


MA100_PARAMS = {
    'ma_period': 100,
    'slope_lookback': 3,
    'touch_buffer_pct': 1.0,
    'leverage': 5,
    'position_pct': 0.02,
    'max_positions': 20,
    'max_margin': 500,
    'sl_pct': 5.0,
    'trail_start_pct': 3.0,
    'trail_pct': 3.0,
    'cooldown_days': 3,
    'fee_rate': 0.00055,
    # DCA 비활성화 (백테스트 결과: 풀사이즈 진입이 더 우수)
    'dca_ratios': [1],
    'dca_interval_pct': 0,
}


class MA100Trader:
    """MA100 터치 반등 전략 실시간 트레이더"""

    def __init__(
        self,
        paper: bool = False,
        testnet: bool = False,
        max_positions: int = 5,
        client=None,
        notifier=None,
        telegram_bot=None,
        get_excluded_symbols=None
    ):
        self.paper = paper
        self.testnet = testnet
        self.running = False
        self.get_excluded_symbols = get_excluded_symbols
        self.max_positions = max_positions

        # 바이빗 클라이언트
        self.client = client or BybitClient(testnet=testnet)
        self.data_fetcher = DataFetcher(self.client)

        # 파라미터
        self.params = MA100_PARAMS.copy()

        # 텔레그램
        self.notifier = notifier or TelegramNotifier()
        self.telegram_bot = telegram_bot
        self.trade_logger = TradeLogger()

        # 포지션 상태
        self.positions: Dict[str, dict] = {}
        self.last_exit_times: Dict[str, datetime] = {}

        # 거래 이력
        self.trade_history: list = []

        # 상태 저장 파일
        self.state_file = "data/ma100_bot_state.json"

        # 시작 로그
        mode = "PAPER" if self.paper else "LIVE"
        net = "TESTNET" if self.testnet else "MAINNET"
        logger.info(f"MA100 시작 - 모드: {mode}, 네트워크: {net}")
        logger.info(f"최대 포지션: {self.max_positions}개")
        logger.info(f"레버리지: {self.params['leverage']}x, 포지션 크기: {self.params['position_pct']*100}%")

        # 거래소에서 기존 포지션 동기화
        self._sync_positions()

    # ==================== 상태 저장/복원 ====================

    def _save_state(self):
        """상태를 파일에 저장"""
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)

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
                "saved_at": datetime.utcnow().isoformat()
            }

            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)

            logger.debug("MA100 상태 저장 완료")

        except Exception as e:
            logger.error(f"MA100 상태 저장 실패: {e}")

    def _load_state(self):
        """저장된 상태 불러오기"""
        if not os.path.exists(self.state_file):
            logger.info("MA100 저장된 상태 파일 없음")
            return False

        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            for symbol, pos in state.get("positions", {}).items():
                if pos.get("entry_time"):
                    pos["entry_time"] = datetime.fromisoformat(pos["entry_time"])
                self.positions[symbol] = pos

            for symbol, dt_str in state.get("last_exit_times", {}).items():
                self.last_exit_times[symbol] = datetime.fromisoformat(dt_str)

            for h in state.get("trade_history", []):
                if h.get("closed_at"):
                    h["closed_at"] = datetime.fromisoformat(h["closed_at"])
                self.trade_history.append(h)

            saved_at = state.get("saved_at", "알 수 없음")
            logger.info(f"MA100 상태 불러옴 (저장 시각: {saved_at})")
            logger.info(f"  - 포지션 {len(self.positions)}개")
            logger.info(f"  - 거래 이력 {len(self.trade_history)}건")

            return True

        except Exception as e:
            logger.error(f"MA100 상태 불러오기 실패: {e}")
            return False

    def _sync_positions(self):
        """거래소에서 기존 포지션 동기화 (이 봇이 관리하는 포지션만)"""
        saved_state_loaded = self._load_state()
        saved_positions = self.positions.copy()

        if self.paper:
            logger.info("MA100 페이퍼 모드 - 거래소 동기화 스킵")
            return

        try:
            exchange_positions = self.client.get_all_positions()
            exchange_symbols = {pos["symbol"] for pos in exchange_positions}

            for symbol in list(self.positions.keys()):
                if symbol not in exchange_symbols:
                    logger.info(f"MA100 거래소에 없는 포지션 제거: {symbol}")
                    self.positions.pop(symbol, None)

            for pos in exchange_positions:
                symbol = pos["symbol"]
                entry_price = pos["entry_price"]
                side = pos["side"]
                size = pos["size"]
                pnl = pos["pnl"]

                if symbol in saved_positions:
                    saved = saved_positions[symbol]
                    if abs(saved.get("entry_price", 0) - entry_price) / max(entry_price, 0.01) < 0.01:
                        self.positions[symbol] = saved
                        self.positions[symbol]["size"] = size
                        self.positions[symbol]["pnl"] = pnl
                        logger.info(f"MA100 저장된 포지션 복원: {symbol}")
                    else:
                        logger.info(f"MA100 포지션 변경 감지, 관리 대상에서 제거: {symbol}")
                        self.positions.pop(symbol, None)

            if self.positions:
                logger.info(f"MA100 관리 포지션 {len(self.positions)}개 동기화 완료")
                # DCA 비활성화: 기존 DCA 주문 취소 및 정리
                self._cancel_pending_dca_orders()
                # 기존 포지션 SL을 진입가 기준으로 재계산
                self._recalc_sl_full_dca()
            else:
                logger.info("MA100 관리 중인 포지션 없음")

            self._save_state()

        except Exception as e:
            logger.error(f"MA100 포지션 동기화 실패: {e}")

    def _cancel_pending_dca_orders(self):
        """기존 DCA 지정가 주문 취소 및 pending_dca 정리"""
        for symbol, pos in self.positions.items():
            pending = pos.get("pending_dca")
            if not pending:
                continue

            short_sym = symbol.split('/')[0]

            if not self.paper:
                # 거래소 미체결 주문 취소
                for dca in pending:
                    order_id = dca.get("order_id")
                    if order_id:
                        try:
                            self.client.exchange.cancel_order(order_id, symbol)
                            logger.info(f"[MA100] {short_sym} DCA 주문 취소: {order_id}")
                        except Exception as e:
                            logger.debug(f"DCA 주문 취소 실패 (이미 체결/취소됨): {e}")

            pos["pending_dca"] = []
            logger.info(f"[MA100] {short_sym} DCA 비활성화 → 풀사이즈 모드")

    def _recalc_sl_full_dca(self):
        """기존 포지션 SL을 전체 DCA 평균단가 기준으로 재계산 (재시작 시)"""
        dca_ratios = self.params.get('dca_ratios', [1])
        dca_interval = self.params.get('dca_interval_pct', 4.0)
        total_ratio = sum(dca_ratios)
        sl_pct = self.params['sl_pct'] / 100

        for symbol, pos in self.positions.items():
            side = pos["side"]
            # 1차 진입가 = filled_entries의 첫 번째
            entries = pos.get("filled_entries", [])
            if not entries:
                continue
            first_price = entries[0]["price"]

            # 전체 DCA 가중평균 계산
            weighted_sum = 0.0
            for i, ratio in enumerate(dca_ratios):
                if side == "short":
                    tranche_price = first_price * (1 + i * dca_interval / 100)
                else:
                    tranche_price = first_price * (1 - i * dca_interval / 100)
                weighted_sum += tranche_price * ratio
            avg_full = weighted_sum / total_ratio

            if side == "long":
                new_sl = avg_full * (1 - sl_pct)
            else:
                new_sl = avg_full * (1 + sl_pct)

            old_sl = pos["stop_loss"]
            short_sym = symbol.split('/')[0]
            if abs(old_sl - new_sl) / old_sl > 0.001:
                logger.info(f"[MA100 SL] {short_sym} SL 재계산: {_fmt_price(old_sl)} → {_fmt_price(new_sl)}")
            pos["stop_loss"] = new_sl

            # 거래소 SL 항상 동기화 (이전 실패 복구)
            if not self.paper:
                try:
                    self.client.set_stop_loss(symbol, new_sl)
                    logger.info(f"[MA100 SL] {short_sym} 거래소 SL 동기화: {_fmt_price(new_sl)}")
                except Exception as e:
                    logger.warning(f"MA100 SL 거래소 업데이트 실패 ({short_sym}): {e}")

        # 거래소 트레일링 스탑도 최신 파라미터로 동기화
        for symbol, pos in self.positions.items():
            if not self.paper:
                try:
                    entry_price = pos.get("entry_price", 0)
                    side = pos["side"]
                    trail_dist = entry_price * self.params['trail_pct'] / 100
                    if side == "long":
                        active_price = entry_price * (1 + self.params['trail_start_pct'] / 100)
                    else:
                        active_price = entry_price * (1 - self.params['trail_start_pct'] / 100)
                    self.client.set_trailing_stop(symbol, trail_dist, active_price)
                except Exception as e:
                    short_sym = symbol.split('/')[0]
                    logger.warning(f"MA100 트레일링 동기화 실패 ({short_sym}): {e}")

        self._save_state()

    # ==================== 데이터 조회 ====================

    def _get_1d_data(self, symbol: str, limit: int = 150) -> Optional[pd.DataFrame]:
        """일봉 데이터 조회"""
        try:
            df = self.data_fetcher.get_ohlcv(symbol, '1d', limit=limit)
            if df is not None:
                df = df.reset_index()
            return df
        except Exception as e:
            logger.debug(f"일봉 데이터 조회 실패 ({symbol}): {e}")
            return None

    # ==================== 시그널 감지 ====================

    def _detect_signal(self, symbol: str, df: pd.DataFrame) -> Optional[dict]:
        """MA100 터치 반등 시그널 감지"""
        if df is None or len(df) < self.params['ma_period'] + 5:
            return None

        ma_period = self.params['ma_period']
        slope_lookback = self.params['slope_lookback']
        touch_buf = self.params['touch_buffer_pct'] / 100

        work = df.copy()
        work['ma100'] = work['close'].rolling(ma_period).mean()
        work['slope'] = (
            (work['ma100'] - work['ma100'].shift(slope_lookback))
            / work['ma100'].shift(slope_lookback) * 100
        )

        # 직전 봉 기준 (iloc[-2])
        if len(work) < 2:
            return None

        row = work.iloc[-2]

        if pd.isna(row['ma100']) or pd.isna(row['slope']):
            return None

        ma100_val = float(row['ma100'])
        slope_val = float(row['slope'])
        low_val = float(row['low'])
        high_val = float(row['high'])
        close_val = float(row['close'])

        # 현재가 (최신 봉)
        current_price = float(work.iloc[-1]['close'])

        side = None

        # SHORT ONLY: slope < 0, high >= ma100*(1-buf), close < ma100
        if (slope_val < 0
              and high_val >= ma100_val * (1 - touch_buf)
              and close_val < ma100_val):
            side = "short"

        if side is None:
            return None

        # 손절가 계산 (DCA 전체 평균단가 기준)
        # 모든 DCA 트랜치가 체결된다고 가정한 가중평균 진입가로 SL 설정
        dca_ratios = self.params.get('dca_ratios', [1])
        dca_interval = self.params.get('dca_interval_pct', 4.0)
        total_ratio = sum(dca_ratios)
        weighted_sum = 0.0
        for i, ratio in enumerate(dca_ratios):
            if side == "short":
                tranche_price = current_price * (1 + i * dca_interval / 100)
            else:
                tranche_price = current_price * (1 - i * dca_interval / 100)
            weighted_sum += tranche_price * ratio
        avg_entry_full = weighted_sum / total_ratio

        sl_pct = self.params['sl_pct'] / 100
        if side == "long":
            stop_loss = avg_entry_full * (1 - sl_pct)
        else:
            stop_loss = avg_entry_full * (1 + sl_pct)

        return {
            "symbol": symbol,
            "side": side,
            "price": current_price,
            "stop_loss": stop_loss,
            "take_profit": 0,
            "signal_info": {
                "slope": slope_val,
                "ma100": ma100_val,
            }
        }

    # ==================== 포지션 관리 ====================

    def _calc_order_quantity(self, price: float, total_balance: float) -> float:
        """주문 수량 계산 (total 잔고 기준)"""
        if total_balance <= 0:
            return 0.0

        margin = min(
            total_balance * self.params['position_pct'],
            self.params['max_margin']
        )
        position_value = margin * self.params['leverage']
        qty = position_value / price
        return qty

    def _open_position(self, signal: dict, free_balance: float) -> float:
        """포지션 오픈 (DCA 1차 진입)"""
        symbol = signal["symbol"]
        side = signal["side"]
        price = signal["price"]
        stop_loss = signal["stop_loss"]

        # DCA 분할매수 설정
        dca_ratios = self.params.get('dca_ratios', [1])
        dca_interval = self.params.get('dca_interval_pct', 2.0)
        total_ratio = sum(dca_ratios)

        total_qty = self._calc_order_quantity(price, free_balance)
        if total_qty <= 0:
            logger.warning("MA100 주문 수량이 0 이하입니다.")
            return 0.0

        tranche_sizes = [
            self.client.amount_to_precision(symbol, total_qty * r / total_ratio)
            for r in dca_ratios
        ]
        first_qty = tranche_sizes[0]

        # DCA 대기 주문 정보 생성
        pending_dca = []
        for i in range(1, len(dca_ratios)):
            if side == "short":
                dca_price = price * (1 + i * dca_interval / 100)
            else:
                dca_price = price * (1 - i * dca_interval / 100)
            pending_dca.append({"price": dca_price, "size": tranche_sizes[i]})

        dca_desc = ":".join(str(r) for r in dca_ratios)
        logger.info(f"[MA100 ENTRY] {symbol} {side.upper()} | Price={_fmt_price(price)}, Qty={first_qty}/{total_qty} (DCA {dca_desc}), SL={_fmt_price(stop_loss)}")

        if not self.paper:
            try:
                self.client.set_leverage(symbol, self.params['leverage'])
            except Exception as e:
                logger.warning(f"MA100 레버리지 설정 실패: {e}")

            try:
                order_side = "buy" if side == "long" else "sell"
                self.client.market_order_with_sl_tp(
                    symbol, order_side, first_qty,
                    stop_loss=stop_loss,
                    take_profit=None
                )
            except Exception as e:
                logger.error(f"MA100 진입 실패 ({symbol}): {e}")
                self.notifier.notify_error(f"MA100 진입 실패: {symbol}\n{e}")
                return 0.0

            # 거래소 레벨 트레일링 스톱 설정
            try:
                trail_dist = price * self.params['trail_pct'] / 100
                if side == "long":
                    active_price = price * (1 + self.params['trail_start_pct'] / 100)
                else:
                    active_price = price * (1 - self.params['trail_start_pct'] / 100)

                self.client.set_trailing_stop(symbol, trail_dist, active_price)
            except Exception as e:
                logger.warning(f"MA100 트레일링 스톱 설정 실패 (봇 체크로 대체): {e}")

            # DCA 지정가 주문 등록
            dca_order_ids = []
            for j, dca in enumerate(pending_dca):
                try:
                    dca_order = self.client.limit_order(
                        symbol, order_side, dca["size"], dca["price"]
                    )
                    dca["order_id"] = dca_order["id"]
                    dca_order_ids.append(dca_order["id"])
                    logger.info(f"[MA100 DCA] {symbol} {j+2}차 지정가 등록: {dca['size']} @ {_fmt_price(dca['price'])} (id={dca_order['id']})")
                except Exception as e:
                    logger.error(f"MA100 DCA {j+2}차 지정가 주문 실패 ({symbol}): {e}")

        # 포지션 상태 업데이트
        self.positions[symbol] = {
            "symbol": symbol,
            "side": side,
            "entry_price": price,
            "entry_time": datetime.utcnow(),
            "stop_loss": stop_loss,
            "take_profit": 0,
            "highest": price,
            "lowest": price,
            "trail_stop": None,
            "trailing": False,
            "size": first_qty,
            "strategy": "ma100",
            "leverage": self.params['leverage'],
            "pending_dca": pending_dca,
            "filled_entries": [{"price": price, "size": first_qty}],
        }

        # 텔레그램 알림
        signal_info = signal.get('signal_info', {})
        slope = signal_info.get('slope', 0)
        ma100_val = signal_info.get('ma100', 0)
        short_sym = symbol.split('/')[0]
        side_emoji = "📈" if side == "long" else "📉"

        dca_info = ""
        if len(dca_ratios) > 1:
            dca_info = f"\n\n📊 분할매수 ({dca_desc})\n"
            dca_info += f"1차: {first_qty} @ {_fmt_price(price)}\n"
            for j, dca in enumerate(pending_dca):
                dca_info += f"{j+2}차: {dca['size']} @ {_fmt_price(dca['price'])} (대기)\n"

        message = (
            f"{side_emoji} <b>MA100 진입: {short_sym}</b>\n\n"
            f"방향: {side.upper()}\n"
            f"진입가: {_fmt_price(price)}\n"
            f"수량: {first_qty} (전체 {total_qty})\n"
            f"레버리지: {self.params['leverage']}x\n\n"
            f"📊 시그널 정보\n"
            f"MA100: {_fmt_price(ma100_val)}\n"
            f"기울기: {slope:.3f}%\n\n"
            f"손절: {_fmt_price(stop_loss)} ({self.params['sl_pct']}%)\n"
            f"트레일링: {self.params['trail_start_pct']}% 수익 시 활성화 → {self.params['trail_pct']}% 되돌림 청산 (거래소)"
            f"{dca_info}"
        )
        self.notifier.send_sync(message)

        self._save_state()

        used_margin = (price * first_qty) / self.params['leverage']
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

        logger.info(f"[MA100 EXIT] {symbol} | Reason={reason}, Price={_fmt_price(price)}")

        if not self.paper:
            # 미체결 DCA 지정가 주문 취소
            pending = pos.get("pending_dca", [])
            if pending:
                cancelled = self.client.cancel_all_orders(symbol)
                if cancelled > 0:
                    logger.info(f"[MA100] {symbol} DCA 미체결 주문 {cancelled}건 취소")

            try:
                order_side = "sell" if side == "long" else "buy"
                self.client.market_order(symbol, order_side, qty)
            except Exception as e:
                logger.error(f"MA100 청산 실패 ({symbol}): {e}")
                self.notifier.notify_error(f"MA100 청산 실패: {symbol}\n{e}")
                return

        # PnL 계산
        if side == "long":
            pnl_pct = (price - entry) / entry * 100 * self.params['leverage']
        else:
            pnl_pct = (entry - price) / entry * 100 * self.params['leverage']

        pnl_usd = pnl_pct / 100 * (entry * qty) / self.params['leverage']

        # 텔레그램 알림
        short_sym = symbol.split('/')[0]
        emoji = "💰" if pnl_pct >= 0 else "💸"
        message = (
            f"{emoji} <b>MA100 청산: {short_sym}</b>\n\n"
            f"진입가: {_fmt_price(entry)}\n"
            f"청산가: {_fmt_price(price)}\n"
            f"사유: {reason}\n\n"
            f"수익률: {pnl_pct:+.2f}%\n"
            f"수익: ${pnl_usd:+.2f}"
        )
        self.notifier.send_sync(message)

        # 거래 로그 (전건 누적)
        self.trade_logger.log_exit(
            symbol, side, entry, price, pnl_pct, pnl_usd, reason,
            strategy="ma100", leverage=self.params["leverage"],
            qty=qty, entry_time=pos.get("entry_time"),
            best_pnl_pct=0, worst_pnl_pct=0,
            best_price=pos.get("highest", entry),
            worst_price=pos.get("lowest", entry),
            trailing_activated=pos.get("trailing", False),
            trail_stop_price=pos.get("trail_stop", 0),
        )

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
            "strategy": "ma100",
        })

        if len(self.trade_history) > 20:
            self.trade_history = self.trade_history[-20:]

        self.positions.pop(symbol, None)
        self.last_exit_times[symbol] = datetime.utcnow()
        self._save_state()

    def _ensure_trailing_stop(self, symbol: str):
        """거래소에 트레일링 스톱이 없으면 설정"""
        if self.paper:
            return

        pos = self.positions.get(symbol)
        if not pos:
            return

        try:
            sl_tp = self.client.get_position_sl_tp(symbol)
            if float(sl_tp.get('trailing_stop', 0)) > 0:
                return  # 이미 설정됨

            entry_price = float(pos['entry_price'])
            side = pos['side']
            trail_dist = entry_price * self.params['trail_pct'] / 100

            if side == "long":
                active_price = entry_price * (1 + self.params['trail_start_pct'] / 100)
            else:
                active_price = entry_price * (1 - self.params['trail_start_pct'] / 100)

            self.client.set_trailing_stop(symbol, trail_dist, active_price)
            short_sym = symbol.split('/')[0]
            logger.info(f"[MA100] {short_sym} 트레일링 스톱 보완 설정 완료")

        except Exception as e:
            logger.warning(f"[MA100] {symbol} 트레일링 스톱 보완 실패: {e}")

    def _check_dca_fills(self, symbol: str):
        """DCA 지정가 주문 체결 여부 확인 및 평균단가/SL 업데이트"""
        pos = self.positions.get(symbol)
        if not pos:
            return

        pending = pos.get("pending_dca")
        if not pending:
            return

        side = pos["side"]
        filled_new = []
        remaining = []

        if self.paper:
            # 페이퍼: 기존 로직 (현재가 비교)
            try:
                ticker = self.client.exchange.fetch_ticker(symbol)
                current_price = float(ticker['last'])
            except Exception:
                return
            for dca in pending:
                if side == "short" and current_price >= dca["price"]:
                    filled_new.append(dca)
                elif side == "long" and current_price <= dca["price"]:
                    filled_new.append(dca)
                else:
                    remaining.append(dca)
        else:
            # 실전: 거래소 미체결 주문 확인
            open_orders = self.client.get_open_orders(symbol)
            open_ids = {o["id"] for o in open_orders}
            for dca in pending:
                order_id = dca.get("order_id")
                if order_id and order_id not in open_ids:
                    # 주문이 사라짐 = 체결됨
                    filled_new.append(dca)
                else:
                    remaining.append(dca)

        if not filled_new:
            return

        short_sym = symbol.split('/')[0]

        for dca in filled_new:
            logger.info(f"[MA100 DCA] {short_sym} 분할매수 체결 @ {_fmt_price(dca['price'])} (수량: {dca['size']})")
            entries = pos.get("filled_entries", [])
            entries.append({"price": dca["price"], "size": dca["size"]})
            pos["filled_entries"] = entries

        # 평균단가 재계산
        entries = pos["filled_entries"]
        total_size = sum(e["size"] for e in entries)
        avg_price = sum(e["price"] * e["size"] for e in entries) / total_size

        old_entry = pos["entry_price"]
        pos["entry_price"] = avg_price
        pos["size"] = total_size
        pos["pending_dca"] = remaining

        # SL은 처음부터 전체 DCA 평균단가 기준으로 설정됨 → 변경 불필요

        # 텔레그램 알림
        n_filled = len(entries)
        n_total = n_filled + len(remaining)
        self.notifier.send_sync(
            f"📊 <b>MA100 분할매수: {short_sym}</b>\n\n"
            f"체결: {n_filled}/{n_total}차\n"
            f"평균단가: {_fmt_price(old_entry)} → {_fmt_price(avg_price)}\n"
            f"총수량: {total_size}\n"
            f"손절가: {_fmt_price(pos['stop_loss'])} (변동없음)"
        )

        self._save_state()

    def _check_exit_signals(self, symbol: str, df: pd.DataFrame):
        """청산 신호 체크 (DCA → SL, 시그널 반전, 트레일링)"""
        pos = self.positions.get(symbol)
        if not pos:
            return

        current_price = float(df.iloc[-1]['close'])

        # DCA 체결 확인
        self._check_dca_fills(symbol)

        # DCA 후 업데이트된 값 사용
        entry_price = float(pos['entry_price'])
        stop_loss = float(pos['stop_loss'])
        side = pos['side']

        # SL 체크
        if side == "long" and current_price <= stop_loss:
            self._close_position(symbol, {"price": current_price, "reason": "손절"})
            return
        elif side == "short" and current_price >= stop_loss:
            self._close_position(symbol, {"price": current_price, "reason": "손절"})
            return

        # 시그널 반전 체크
        signal = self._detect_signal(symbol, df)
        if signal:
            if side == "long" and signal["side"] == "short":
                self._close_position(symbol, {"price": current_price, "reason": "시그널 반전 (숏)"})
                return
            elif side == "short" and signal["side"] == "long":
                self._close_position(symbol, {"price": current_price, "reason": "시그널 반전 (롱)"})
                return

        # 트레일링 스톱
        trail_start = self.params['trail_start_pct']
        trail_pct = self.params['trail_pct']

        if side == "long":
            pnl_pct = (current_price - entry_price) / entry_price * 100

            # highest 추적
            if current_price > pos.get('highest', 0):
                pos['highest'] = current_price

            if pnl_pct >= trail_start:
                if not pos.get('trailing'):
                    pos['trailing'] = True
                    pos['trail_activated_at'] = current_price
                    logger.info(f"[MA100 TRAIL] {symbol} 트레일링 활성화 (수익률: {pnl_pct:.1f}%)")
                    self._save_state()

                # 최고가 대비 trail_pct% 하락 시 청산
                highest = pos['highest']
                trail_stop = highest * (1 - trail_pct / 100)
                pos['trail_stop'] = trail_stop

                if current_price <= trail_stop:
                    self._close_position(symbol, {"price": current_price, "reason": "트레일링 스톱"})
                    return

        else:  # short
            pnl_pct = (entry_price - current_price) / entry_price * 100

            # lowest 추적
            if current_price < pos.get('lowest', float('inf')):
                pos['lowest'] = current_price

            if pnl_pct >= trail_start:
                if not pos.get('trailing'):
                    pos['trailing'] = True
                    pos['trail_activated_at'] = current_price
                    logger.info(f"[MA100 TRAIL] {symbol} 트레일링 활성화 (수익률: {pnl_pct:.1f}%)")
                    self._save_state()

                # 최저가 대비 trail_pct% 상승 시 청산
                lowest = pos['lowest']
                trail_stop = lowest * (1 + trail_pct / 100)
                pos['trail_stop'] = trail_stop

                if current_price >= trail_stop:
                    self._close_position(symbol, {"price": current_price, "reason": "트레일링 스톱"})
                    return

    # ==================== 수동 청산 감지 ====================

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

            for symbol in closed_symbols:
                pos = self.positions[symbol]
                side = pos["side"]
                entry = float(pos["entry_price"])
                qty = float(pos.get("size", 0))

                pnl_record = None
                for pnl in closed_pnl_list:
                    if pnl['symbol'] == symbol:
                        pnl_entry = float(pnl.get('entry_price', 0))
                        if entry > 0 and abs(pnl_entry - entry) / entry < 0.001:
                            pnl_record = pnl
                            break

                if pnl_record:
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

                short_sym = symbol.split('/')[0]
                logger.info(f"[MA100 MANUAL CLOSE] {short_sym} 수동 청산 감지 | PnL: {pnl_pct:+.2f}% (${pnl_usd:+.2f})")

                # 미체결 DCA 지정가 주문 취소
                pending = pos.get("pending_dca", [])
                if pending:
                    cancelled = self.client.cancel_all_orders(symbol)
                    if cancelled > 0:
                        logger.info(f"[MA100] {symbol} DCA 미체결 주문 {cancelled}건 취소")

                self.notifier.notify_exit(symbol, side, entry, exit_price, pnl_pct, pnl_usd, reason, strategy="ma100")

                # 거래 로그 (전건 누적 CSV)
                self.trade_logger.log_exit(
                    symbol, side, entry, exit_price, pnl_pct, pnl_usd, reason,
                    strategy="ma100", leverage=self.params["leverage"],
                    qty=qty, entry_time=pos.get("entry_time"),
                    best_pnl_pct=0, worst_pnl_pct=0,
                    best_price=pos.get("highest", entry),
                    worst_price=pos.get("lowest", entry),
                    trailing_activated=pos.get("trailing", False),
                    trail_stop_price=pos.get("trail_stop", 0),
                )

                self.trade_history.append({
                    "symbol": symbol,
                    "side": side,
                    "entry_price": entry,
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct,
                    "pnl_usd": pnl_usd,
                    "reason": reason,
                    "closed_at": datetime.utcnow(),
                    "strategy": "ma100",
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
            logger.error(f"MA100 수동 청산 감지 실패: {e}")
            return result

    # ==================== 텔레그램 콜백 ====================

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

    def _get_balance_total(self) -> float:
        """USDT 전체 잔고 (포지션 비율 계산용)"""
        try:
            balance = self.client.get_balance()
            return float(balance.get("total", 0))
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

                ex_entry = float(ex_pos.get("entry_price", 0))
                entry = ex_entry if ex_entry > 0 else float(pos.get("entry_price", 0))
                current_price = float(ex_pos.get("mark_price", 0))
                pnl_usd = float(ex_pos.get("pnl", 0))
                side = pos.get("side", "long")
                size = float(ex_pos.get("size", 0)) or float(pos.get("size", 0))
                leverage = float(ex_pos.get("leverage", self.params['leverage']))

                # 마진 대비 PnL (가장 정확)
                if size > 0 and entry > 0 and leverage > 0:
                    margin = size * entry / leverage
                    pnl_pct = (pnl_usd / margin * 100) if margin > 0 else 0
                elif entry > 0 and current_price > 0:
                    if side == "long":
                        pnl_pct = (current_price - entry) / entry * 100 * leverage
                    else:
                        pnl_pct = (entry - current_price) / entry * 100 * leverage
                else:
                    pnl_pct = 0

                pos_copy["pnl"] = pnl_usd
                pos_copy["pnl_pct"] = pnl_pct
                pos_copy["current_price"] = current_price

                result.append(pos_copy)

            return result
        except Exception as e:
            logger.warning(f"MA100 실시간 PnL 조회 실패: {e}")
            return list(self.positions.values())

    def _get_trade_history(self) -> list:
        """거래 이력 반환"""
        return self.trade_history

    # ==================== 메인 실행 ====================

    def check_positions(self):
        """포지션 상태만 체크 (수동/거래소 청산 감지 + 가격 추적 + 트레일링). 5분마다 호출."""
        if not self.positions:
            return

        self._check_manual_closes()

        state_changed = False
        for symbol in list(self.positions.keys()):
            pos = self.positions.get(symbol)
            if not pos:
                continue

            self._ensure_trailing_stop(symbol)

            # 현재가 조회 → lowest/highest 업데이트
            try:
                ticker = self.client.get_ticker(symbol)
                current_price = float(ticker["last"])
                side = pos["side"]
                entry = float(pos["entry_price"])

                # highest/lowest 추적
                if current_price > pos.get("highest", 0):
                    pos["highest"] = current_price
                    state_changed = True
                if current_price < pos.get("lowest", float("inf")):
                    pos["lowest"] = current_price
                    state_changed = True

                # 봇 레벨 트레일링 체크 (거래소 트레일링 백업)
                trail_start = self.params["trail_start_pct"]
                trail_pct = self.params["trail_pct"]

                if side == "short":
                    pnl_pct = (entry - current_price) / entry * 100
                    if pnl_pct >= trail_start:
                        if not pos.get("trailing"):
                            pos["trailing"] = True
                            pos["trail_activated_at"] = current_price
                            logger.info(f"[MA100 TRAIL] {symbol.split('/')[0]} 트레일링 활성화 (수익률: {pnl_pct:.1f}%)")
                            state_changed = True
                        lowest = pos.get("lowest", entry)
                        trail_stop = lowest * (1 + trail_pct / 100)
                        pos["trail_stop"] = trail_stop
                        if current_price >= trail_stop:
                            self._close_position(symbol, {"price": current_price, "reason": "트레일링 스톱"})
                            continue
                else:  # long
                    pnl_pct = (current_price - entry) / entry * 100
                    if pnl_pct >= trail_start:
                        if not pos.get("trailing"):
                            pos["trailing"] = True
                            pos["trail_activated_at"] = current_price
                            logger.info(f"[MA100 TRAIL] {symbol.split('/')[0]} 트레일링 활성화 (수익률: {pnl_pct:.1f}%)")
                            state_changed = True
                        highest = pos.get("highest", entry)
                        trail_stop = highest * (1 - trail_pct / 100)
                        pos["trail_stop"] = trail_stop
                        if current_price <= trail_stop:
                            self._close_position(symbol, {"price": current_price, "reason": "트레일링 스톱"})
                            continue

            except Exception as e:
                logger.debug(f"MA100 가격 추적 실패 ({symbol}): {e}")

            # DCA 지정가 주문 체결 여부 확인
            if pos and pos.get("pending_dca"):
                try:
                    self._check_dca_fills(symbol)
                except Exception as e:
                    logger.debug(f"MA100 DCA 체결 확인 실패 ({symbol}): {e}")

        if state_changed:
            self._save_state()

    def run_once(self):
        """일봉 갱신 시 전체 스캔 + 시그널 진입"""
        # 포지션 체크 (청산 감지 + 트레일링 보완)
        self.check_positions()

        # 기존 포지션 exit 체크 (시그널 반전 등 일봉 데이터 필요한 로직)
        for symbol in list(self.positions.keys()):
            df = self._get_1d_data(symbol, limit=150)
            if df is not None:
                self._check_exit_signals(symbol, df)

        # 최대 포지션 수 체크
        if len(self.positions) >= self.max_positions:
            logger.info(f"[MA100 WAIT] 최대 포지션 수 도달 ({self.max_positions}개)")
            return

        # 잔고 확인 (total 기준으로 포지션 비율 계산)
        try:
            total_balance = self._get_balance_total()
        except Exception:
            total_balance = 0.0

        if total_balance <= 0:
            logger.warning("[MA100 WAIT] 잔고 없음")
            return

        # 전체 USDT 무기한 선물 스캔
        logger.info("[MA100 SCAN] 일봉 MA100 스캔 시작...")

        try:
            markets = self.client.exchange.fetch_markets()
            usdt_perps = [
                m['symbol'] for m in markets
                if m.get('settle') == 'USDT'
                and m.get('type') == 'swap'
                and m.get('active', True)
                and m.get('base') not in STABLECOINS
            ]
            logger.info(f"MA100 스캔 대상: {len(usdt_perps)}개 코인")
        except Exception:
            logger.error("MA100 마켓 리스트 조회 실패")
            return

        # 다른 전략 보유 심볼 제외
        excluded = set()
        if self.get_excluded_symbols:
            try:
                excluded = self.get_excluded_symbols()
            except Exception:
                pass

        signals = []
        now = datetime.utcnow()
        for symbol in usdt_perps:
            # 이미 포지션 있거나 다른 전략이 보유 중이면 스킵
            if symbol in self.positions or symbol in excluded:
                continue

            # 쿨다운 체크
            last_exit = self.last_exit_times.get(symbol)
            if last_exit:
                cooldown = timedelta(days=self.params['cooldown_days'])
                if now - last_exit < cooldown:
                    continue

            try:
                df = self._get_1d_data(symbol, limit=150)
                if df is None:
                    continue

                # 최신 캔들 시간 체크: 일봉이 30시간 이내인 경우만 (24h + 6h 여유)
                if "timestamp" in df.columns and len(df) > 0:
                    last_ts = pd.Timestamp(df.iloc[-1]["timestamp"])
                    if last_ts.tzinfo is not None:
                        last_ts = last_ts.tz_localize(None)
                    age_hours = (now - last_ts.to_pydatetime()).total_seconds() / 3600
                    if age_hours > 30:
                        continue

                signal = self._detect_signal(symbol, df)
                if signal:
                    signals.append(signal)
                    info = signal['signal_info']
                    logger.info(
                        f"[MA100 SIGNAL] {symbol} {signal['side'].upper()} | "
                        f"Slope={info['slope']:.3f}% MA100={_fmt_price(info['ma100'])}"
                    )
            except Exception as e:
                logger.debug(f"MA100 스캔 실패 ({symbol}): {e}")
                continue

        if not signals:
            logger.info("[MA100 WAIT] 시그널 없음")
            return

        # slope 절대값 순으로 정렬 (강한 추세 우선)
        signals.sort(key=lambda s: -abs(s['signal_info']['slope']))

        # 최대 포지션 수까지만 진입
        for signal in signals:
            if len(self.positions) >= self.max_positions:
                break

            used_margin = self._open_position(signal, total_balance)
            if used_margin > 0:
                total_balance -= used_margin

            if total_balance <= 0:
                break

    def stop(self):
        """봇 중지"""
        self.running = False
        logger.info("MA100 봇 중지됨")

    def resume(self):
        """봇 재시작"""
        self.running = True
        logger.info("MA100 봇 재시작됨")
