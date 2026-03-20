"""
일목균형표 자동매매 트레이더

ichimoku_live.py의 IchimokuLiveBot과 동일한 로직
"""

import logging
import time
import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Optional
import pandas as pd

from src.bybit_client import BybitClient
from src.data_fetcher import DataFetcher
from src.ichimoku import calculate_ichimoku
from src.strategy import (
    get_entry_signal, check_exit_signal, update_btc_trend,
    LEVERAGE, POSITION_PCT, STRATEGY_PARAMS, MAJOR_COINS
)
from src.telegram_bot import TelegramNotifier, TelegramBot
from src.market_analyzer import MarketAnalyzer
from src.chart_generator import ChartGenerator

logger = logging.getLogger(__name__)


class IchimokuTrader:
    """일목균형표 자동매매 트레이더"""

    def __init__(self, paper: bool = False, testnet: bool = False,
                 client=None, notifier=None, telegram_bot=None):
        self.paper = paper
        self.testnet = testnet
        self.running = False

        # 바이빗 클라이언트 (외부 주입 또는 자체 생성)
        self.client = client or BybitClient(testnet=testnet)
        self.data_fetcher = DataFetcher(self.client)

        # 텔레그램 (외부 주입 또는 자체 생성)
        self.notifier = notifier or TelegramNotifier()
        self.telegram_bot = telegram_bot or TelegramBot(self.notifier)
        self.telegram_bot.set_callbacks(
            get_balance=self._get_balance_full,
            get_positions=self._get_positions_list,
            get_trade_history=self._get_trade_history,
            stop_bot=self.stop,
            start_bot=self.resume,
            sync_positions=self._check_manual_closes
        )

        # 시장 분석 & 차트
        self.market_analyzer = MarketAnalyzer(self.data_fetcher)
        self.chart_generator = ChartGenerator()

        # 분석 콜백 설정
        self.telegram_bot.set_analysis_callbacks(
            get_market_report=self._get_market_report,
            get_no_entry_report=self._get_no_entry_report,
            get_watch_report=self._get_watch_report,
            get_chart=self._get_chart,
            get_overview_chart=self._get_overview_chart,
            chat_response=self._chat_response
        )

        # 거래정보 콜백 설정
        self.telegram_bot.set_trading_callbacks(
            get_funding_rates=self._get_funding_rates,
            get_position_sl_tp=self._get_position_sl_tp,
            set_position_sl_tp=self._set_position_sl_tp,
            get_account_stats=self._get_account_stats,
            get_trade_history_exchange=self._get_trade_history_from_exchange,
            get_transaction_log=self._get_transaction_log
        )

        # 타임프레임
        self.timeframe = "4h"

        # 캐시된 코인 데이터 (명령어 응답용)
        self._cached_coin_data: Dict[str, pd.DataFrame] = {}

        # 포지션 상태
        self.positions: Dict[str, dict] = {}
        self.last_exit_times: Dict[str, datetime] = {}

        # 거래 이력
        self.trade_history: list = []

        # 상태 저장 파일 경로
        self.state_file = "data/bot_state.json"

        # BTC 트렌드
        self.btc_uptrend: Optional[bool] = None

        # 레버리지/진입비율 (런타임 변경 가능)
        self.leverage = LEVERAGE
        self.position_pct = POSITION_PCT

        # 시작 로그
        mode = "PAPER" if self.paper else "LIVE"
        logger.info(f"IchimokuTrader 시작 - 모드: {mode}, 테스트넷: {self.testnet}")
        logger.info(f"레버리지: {self.leverage}x, 포지션 크기: {self.position_pct*100}%")

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

            saved_at = state.get("saved_at", "알 수 없음")
            logger.info(f"저장된 상태 불러옴 (저장 시각: {saved_at})")
            logger.info(f"  - 포지션 {len(self.positions)}개")
            logger.info(f"  - 거래 이력 {len(self.trade_history)}건")

            return True

        except Exception as e:
            logger.error(f"상태 불러오기 실패: {e}")
            return False

    def _sync_positions(self):
        """거래소에서 기존 포지션 동기화 (저장된 상태 우선)"""

        # 1. 저장된 상태 먼저 불러오기
        saved_state_loaded = self._load_state()
        saved_positions = self.positions.copy()

        if self.paper:
            logger.info("페이퍼 모드 - 거래소 동기화 스킵")
            return

        try:
            # 2. 거래소에서 실제 포지션 조회
            exchange_positions = self.client.get_all_positions()
            exchange_symbols = {pos["symbol"] for pos in exchange_positions}

            # 3. 거래소에 없는 저장된 포지션 제거
            for symbol in list(self.positions.keys()):
                if symbol not in exchange_symbols:
                    logger.info(f"거래소에 없는 포지션 제거: {symbol}")
                    self.positions.pop(symbol, None)

            # 4. 거래소 포지션과 동기화
            for pos in exchange_positions:
                symbol = pos["symbol"]

                # 운용 대상 코인만
                if symbol not in MAJOR_COINS:
                    continue

                entry_price = pos["entry_price"]
                side = pos["side"]

                # 저장된 포지션이 있으면 그 정보 유지
                if symbol in saved_positions:
                    saved = saved_positions[symbol]
                    # 진입가가 같으면 저장된 설정 유지
                    if abs(saved.get("entry_price", 0) - entry_price) < 0.01:
                        self.positions[symbol] = saved
                        self.positions[symbol]["size"] = pos["size"]  # 수량만 업데이트
                        self.positions[symbol]["pnl"] = pos["pnl"]
                        logger.info(f"저장된 포지션 복원: {symbol} (SL: ${saved.get('stop_loss', 0):,.2f}, TP: ${saved.get('take_profit', 0):,.2f}, 트레일링: {saved.get('trailing', False)})")
                        continue

                # 저장된 정보 없으면 새로 계산
                stop_loss = 0
                take_profit = 0
                try:
                    df = self._fetch_ichimoku_df(symbol, limit=200)
                    if df is not None and not df.empty:
                        row = df.iloc[-1]
                        cloud_top = float(row["cloud_top"])
                        cloud_bottom = float(row["cloud_bottom"])
                        params = STRATEGY_PARAMS

                        if side == "long":
                            stop_loss = cloud_top * (1 - params["sl_buffer"] / 100)
                            sl_distance_pct = (entry_price - stop_loss) / entry_price * 100
                            take_profit = entry_price * (1 + sl_distance_pct * params["rr_ratio"] / 100)
                        else:
                            stop_loss = cloud_bottom * (1 + params["sl_buffer"] / 100)
                            sl_distance_pct = (stop_loss - entry_price) / entry_price * 100
                            take_profit = entry_price * (1 - sl_distance_pct * params["rr_ratio"] / 100)
                except Exception as e:
                    logger.warning(f"손절가 계산 실패 ({symbol}): {e}")

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
                    "size": pos["size"],
                    "pnl": pos["pnl"],
                }
                logger.info(f"새 포지션 동기화: {symbol} (SL/TP 재계산)")

            if self.positions:
                logger.info(f"포지션 {len(self.positions)}개 동기화 완료")
                for sym, p in self.positions.items():
                    short_sym = sym.split('/')[0]
                    trail_str = " [트레일링]" if p.get('trailing') else ""
                    logger.info(f"  - {short_sym}: {p['side'].upper()} @ ${p['entry_price']:,.2f} | SL: ${p['stop_loss']:,.2f} | TP: ${p['take_profit']:,.2f}{trail_str}")
            else:
                logger.info("동기화된 포지션 없음")

            # 5. 현재 상태 저장
            self._save_state()

        except Exception as e:
            logger.error(f"포지션 동기화 실패: {e}")

    def _get_balance_full(self) -> dict:
        """USDT 전체 잔고 정보"""
        try:
            return self.client.get_balance()
        except Exception as e:
            logger.error(f"잔고 조회 실패: {e}")
            return {"total": 0, "free": 0, "used": 0, "unrealized_pnl": 0, "equity": 0}

    def _get_trade_history(self) -> list:
        """거래 이력 반환"""
        return self.trade_history

    def _get_funding_rates(self) -> dict:
        """펀딩비 조회"""
        try:
            from src.strategy import MAJOR_COINS
            return self.client.get_funding_rates(list(MAJOR_COINS))
        except Exception as e:
            logger.error(f"펀딩비 조회 실패: {e}")
            return {}

    def _get_position_sl_tp(self, symbol: str) -> dict:
        """포지션 SL/TP 조회"""
        try:
            return self.client.get_position_sl_tp(symbol)
        except Exception as e:
            logger.error(f"SL/TP 조회 실패: {e}")
            return {}

    def _set_position_sl_tp(self, symbol: str, stop_loss: float = None, take_profit: float = None) -> bool:
        """포지션 SL/TP 수정"""
        try:
            result = self.client.set_position_sl_tp(symbol, stop_loss, take_profit)

            # 봇 메모리도 업데이트
            if result and symbol in self.positions:
                if stop_loss is not None:
                    self.positions[symbol]['stop_loss'] = stop_loss
                if take_profit is not None:
                    self.positions[symbol]['take_profit'] = take_profit
                self._save_state()

            return result
        except Exception as e:
            logger.error(f"SL/TP 수정 실패: {e}")
            return False

    def _get_account_stats(self, days: int = 30) -> dict:
        """계정 통계 조회"""
        try:
            return self.client.get_account_stats(days)
        except Exception as e:
            logger.error(f"계정 통계 조회 실패: {e}")
            return {}

    def _get_trade_history_from_exchange(self, days: int = 7) -> list:
        """바이빗에서 거래 이력 직접 조회"""
        try:
            return self.client.get_trade_history_from_exchange(days, limit=20)
        except Exception as e:
            logger.error(f"거래 이력 조회 실패: {e}")
            return []

    def _get_transaction_log(self, days: int = 7) -> dict:
        """펀딩비/수수료 내역 조회"""
        try:
            return self.client.get_transaction_log(days)
        except Exception as e:
            logger.error(f"거래 내역 조회 실패: {e}")
            return {}

    def _get_balance_free(self) -> float:
        """USDT 사용 가능 잔고"""
        try:
            balance = self.client.get_balance()
            return float(balance.get("free", 0))
        except Exception as e:
            logger.error(f"잔고 조회 실패: {e}")
            return 0.0

    def _get_positions_list(self) -> list:
        """현재 포지션 목록 (실시간 PnL, 수익률, 현재가 포함)"""
        if not self.positions:
            return []

        # 페이퍼 모드면 메모리 정보만 반환
        if self.paper:
            return list(self.positions.values())

        # 거래소에서 실시간 정보 조회
        try:
            exchange_positions = self.client.get_all_positions()
            pos_map = {p["symbol"]: p for p in exchange_positions}

            result = []
            for pos in self.positions.values():
                pos_copy = pos.copy()
                ex_pos = pos_map.get(pos["symbol"], {})

                # 거래소 평균단가 우선, 없으면 봇 메모리
                ex_entry = float(ex_pos.get("entry_price", 0))
                entry = ex_entry if ex_entry > 0 else float(pos.get("entry_price", 0))
                current_price = float(ex_pos.get("mark_price", 0))
                pnl_usd = float(ex_pos.get("pnl", 0))
                side = pos.get("side", "long")
                size = float(ex_pos.get("size", 0)) or float(pos.get("size", 0))
                sl = float(pos.get("stop_loss", 0))
                tp = float(pos.get("take_profit", 0))

                # 수익률: 마진 대비 PnL (가장 정확)
                if size > 0 and entry > 0:
                    margin = size * entry / float(ex_pos.get("leverage", self.leverage))
                    pnl_pct = (pnl_usd / margin * 100) if margin > 0 else 0
                elif entry > 0 and current_price > 0:
                    if side == "long":
                        pnl_pct = (current_price - entry) / entry * 100 * self.leverage
                    else:
                        pnl_pct = (entry - current_price) / entry * 100 * self.leverage
                else:
                    pnl_pct = 0

                # 손절/익절 거리 %
                sl_pct = abs((sl - entry) / entry * 100) if entry > 0 and sl > 0 else 0
                tp_pct = abs((tp - entry) / entry * 100) if entry > 0 and tp > 0 else 0

                pos_copy["pnl"] = pnl_usd
                pos_copy["pnl_pct"] = pnl_pct
                pos_copy["current_price"] = current_price
                pos_copy["sl_pct"] = sl_pct
                pos_copy["tp_pct"] = tp_pct
                pos_copy["leverage"] = ex_pos.get("leverage", self.leverage)

                result.append(pos_copy)

            return result
        except Exception as e:
            logger.warning(f"실시간 PnL 조회 실패: {e}")
            return list(self.positions.values())

    def _update_btc_trend(self):
        """BTC 트렌드 업데이트"""
        try:
            btc_df = self.data_fetcher.get_ohlcv("BTC/USDT:USDT", self.timeframe, limit=100)
            if btc_df is None or btc_df.empty:
                self.btc_uptrend = None
                return

            btc_df = btc_df.reset_index()
            self.btc_uptrend = update_btc_trend(btc_df)

            if self.btc_uptrend is not None:
                trend_str = "상승" if self.btc_uptrend else "하락"
                logger.info(f"[BTC TREND] {trend_str}")

        except Exception as e:
            logger.error(f"BTC 트렌드 계산 실패: {e}")
            self.btc_uptrend = None

    def _fetch_ichimoku_df(self, symbol: str, limit: int = 200) -> Optional[pd.DataFrame]:
        """일목 지표 계산"""
        try:
            df = self.data_fetcher.get_ohlcv(symbol, self.timeframe, limit=limit)
        except Exception as e:
            logger.error(f"OHLCV 조회 실패 ({symbol}): {e}")
            return None

        if df is None or df.empty:
            return None

        df = df.reset_index()
        df = calculate_ichimoku(df)
        df = df.dropna(subset=["tenkan", "kijun", "cloud_top", "cloud_bottom", "cloud_thickness"])

        if df.empty:
            return None

        return df

    def _calc_order_quantity(self, price: float, free_balance: float) -> float:
        """주문 수량 계산"""
        if free_balance <= 0:
            return 0.0

        margin = free_balance * self.position_pct
        position_value = margin * self.leverage
        qty = position_value / price
        qty = round(qty, 3)
        return qty

    def _open_position(self, signal: dict, free_balance: float, df: pd.DataFrame = None) -> float:
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

        logger.info(f"[ENTRY] {symbol} {side.upper()} | Price={price:.2f}, Qty={qty}")

        if not self.paper:
            try:
                self.client.set_leverage(symbol, self.leverage)
            except Exception as e:
                logger.warning(f"레버리지 설정 실패: {e}")

            try:
                order_side = "buy" if side == "long" else "sell"
                # SL/TP를 바이빗 서버에 함께 등록 (봇 다운 시에도 작동)
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
            "strategy": "ichimoku",
        }

        # 텔레그램 알림
        self.notifier.notify_entry(symbol, side, price, qty, stop_loss, take_profit)

        # AI 진입 이유 분석 (비동기)
        if df is not None:
            asyncio.create_task(self._send_entry_analysis(symbol, df, side))

        # 상태 저장
        self._save_state()

        used_margin = (price * qty) / self.leverage
        return used_margin

    async def _send_entry_analysis(self, symbol: str, df: pd.DataFrame, side: str):
        """진입 이유 AI 분석 전송"""
        try:
            analysis = await self.market_analyzer.analyze_entry_reason(
                symbol, df, side, self.btc_uptrend
            )
            if analysis:
                short_symbol = symbol.split('/')[0]
                emoji = "🟢" if side == "long" else "🔴"
                message = f"{emoji} <b>{short_symbol} 진입 분석</b>\n\n{analysis}"
                self.notifier.send_sync(message)
        except Exception as e:
            logger.warning(f"진입 분석 전송 실패: {e}")

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

        logger.info(f"[EXIT] {symbol} | Reason={reason}, Price={price:.2f}")

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
            pnl_pct = (price - entry) / entry * 100 * self.leverage
        else:
            pnl_pct = (entry - price) / entry * 100 * self.leverage

        pnl_usd = pnl_pct / 100 * (entry * qty) / self.leverage

        # 텔레그램 알림
        self.notifier.notify_exit(symbol, side, entry, price, pnl_pct, pnl_usd, reason)

        # AI 청산 분석 (비동기)
        asyncio.create_task(self._send_exit_analysis(symbol, side, entry, price, reason, pnl_pct))

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

        # 최근 20개만 유지
        if len(self.trade_history) > 20:
            self.trade_history = self.trade_history[-20:]

        # 상태 업데이트
        self.positions.pop(symbol, None)
        self.last_exit_times[symbol] = datetime.utcnow()

        # 상태 저장
        self._save_state()

    async def _send_exit_analysis(self, symbol: str, side: str, entry: float, exit_price: float, reason: str, pnl_pct: float):
        """청산 이유 AI 분석 전송"""
        try:
            analysis = await self.market_analyzer.analyze_exit_reason(
                symbol, side, entry, exit_price, reason, pnl_pct
            )
            if analysis:
                short_symbol = symbol.split('/')[0]
                emoji = "💰" if pnl_pct >= 0 else "💸"
                message = f"{emoji} <b>{short_symbol} 청산 분석</b>\n\n{analysis}"
                self.notifier.send_sync(message)
        except Exception as e:
            logger.warning(f"청산 분석 전송 실패: {e}")

    def _check_manual_closes(self) -> dict:
        """수동 청산 감지 및 거래 이력 기록 (바이빗 실제 체결 기록 사용)

        거래소에는 없지만 봇 메모리에 있는 포지션을 찾아
        바이빗의 실제 청산 기록에서 정확한 청산가와 PnL을 가져옵니다.

        Returns:
            동기화 결과 {"synced": 청산 감지 수, "positions": 현재 포지션 수}
        """
        result = {"synced": 0, "positions": len(self.positions)}

        if self.paper or not self.positions:
            return result

        try:
            # 거래소의 실제 포지션 조회
            exchange_positions = self.client.get_all_positions()
            exchange_symbols = {pos["symbol"] for pos in exchange_positions}

            # 봇 메모리에는 있지만 거래소에 없는 포지션 찾기
            closed_symbols = []
            for symbol in list(self.positions.keys()):
                if symbol not in exchange_symbols:
                    closed_symbols.append(symbol)

            if not closed_symbols:
                return result

            # 바이빗에서 최근 청산 이력 조회
            closed_pnl_list = self.client.get_closed_pnl(limit=50)

            # 수동 청산된 포지션 처리
            for symbol in closed_symbols:
                pos = self.positions[symbol]
                side = pos["side"]
                entry = float(pos["entry_price"])
                qty = float(pos.get("size", 0))

                # 진입가가 일치하는 청산 기록만 매칭 (다른 전략 기록 방지)
                pnl_record = None
                for pnl in closed_pnl_list:
                    if pnl['symbol'] == symbol:
                        pnl_entry = float(pnl.get('entry_price', 0))
                        if entry > 0 and abs(pnl_entry - entry) / entry < 0.001:
                            pnl_record = pnl
                            break

                # 바이빗 청산 기록에서 실제 청산가와 PnL 가져오기
                if pnl_record:
                    exit_price = pnl_record['exit_price']
                    pnl_usd = pnl_record['closed_pnl']
                    # 실제 PnL에서 수익률 역산
                    if entry > 0 and qty > 0:
                        pnl_pct = pnl_usd / (entry * qty / self.leverage) * 100
                    else:
                        pnl_pct = 0
                    reason = "수동 청산"
                    logger.info(f"[SYNC] {symbol} 바이빗 청산 기록 발견 | 청산가: ${exit_price:.2f}, PnL: ${pnl_usd:.2f}")
                else:
                    # 청산 기록이 없으면 현재가로 추정
                    try:
                        ticker = self.client.get_ticker(symbol)
                        exit_price = ticker["last"]
                    except:
                        exit_price = entry

                    if side == "long":
                        pnl_pct = (exit_price - entry) / entry * 100 * self.leverage
                    else:
                        pnl_pct = (entry - exit_price) / entry * 100 * self.leverage
                    pnl_usd = pnl_pct / 100 * (entry * qty) / self.leverage
                    reason = "수동 청산 (추정)"
                    logger.info(f"[SYNC] {symbol} 청산 기록 없음, 현재가로 추정 | PnL: {pnl_pct:+.2f}%")

                short_sym = symbol.split('/')[0]
                logger.info(f"[MANUAL CLOSE] {short_sym} 수동 청산 감지 | PnL: {pnl_pct:+.2f}% (${pnl_usd:+.2f})")

                # 텔레그램 알림
                self.notifier.notify_exit(
                    symbol, side, entry, exit_price,
                    pnl_pct, pnl_usd, reason
                )

                # 거래 이력 저장
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

                # 최근 20개만 유지
                if len(self.trade_history) > 20:
                    self.trade_history = self.trade_history[-20:]

                # 상태 업데이트
                self.positions.pop(symbol, None)
                self.last_exit_times[symbol] = datetime.utcnow()
                result["synced"] += 1

            # 변경사항 저장
            if closed_symbols:
                self._save_state()

            result["positions"] = len(self.positions)
            return result

        except Exception as e:
            logger.error(f"수동 청산 감지 실패: {e}")

    async def _get_market_report(self) -> str:
        """시황 리포트 생성"""
        coin_data = self._get_all_coin_data()
        return await self.market_analyzer.generate_market_report(
            coin_data, self.btc_uptrend, self.positions
        )

    async def _get_no_entry_report(self) -> str:
        """진입 없는 이유 리포트"""
        coin_data = self._get_all_coin_data()
        return await self.market_analyzer.generate_no_entry_report(
            coin_data, self.btc_uptrend, self.last_exit_times
        )

    async def _get_watch_report(self) -> str:
        """진입 예상 코인 리포트"""
        coin_data = self._get_all_coin_data()
        return await self.market_analyzer.generate_watch_report(
            coin_data, self.btc_uptrend
        )

    async def _get_chart(self, symbol: str) -> Optional[bytes]:
        """특정 코인 차트 생성"""
        # 심볼 정규화
        if '/' not in symbol:
            symbol = f"{symbol}/USDT:USDT"

        df = self._fetch_ichimoku_df(symbol, limit=200)
        if df is None:
            return None

        position = self.positions.get(symbol)
        return self.chart_generator.generate_ichimoku_chart(df, symbol, position)

    async def _get_overview_chart(self) -> Optional[bytes]:
        """주요 코인 차트 생성"""
        coin_data = self._get_all_coin_data()
        main_symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT"]
        return self.chart_generator.generate_multi_chart(coin_data, main_symbols)

    async def _chat_response(self, user_message: str) -> str:
        """사용자 채팅에 AI 응답"""
        coin_data = self._get_all_coin_data()
        positions = self._get_positions_list()
        balance = self._get_balance_full()

        return await self.market_analyzer.chat_response(
            user_message=user_message,
            coin_data=coin_data,
            btc_uptrend=self.btc_uptrend,
            positions=positions,
            balance=balance
        )

    def _get_all_coin_data(self) -> Dict[str, pd.DataFrame]:
        """모든 코인 데이터 가져오기"""
        coin_data = {}
        for symbol in MAJOR_COINS:
            df = self._fetch_ichimoku_df(symbol, limit=200)
            if df is not None:
                coin_data[symbol] = df
        self._cached_coin_data = coin_data
        return coin_data

    async def _send_periodic_report(self):
        """4시간봉 갱신 시 자동 시황 리포트 전송"""
        try:
            report = await self._get_market_report()
            self.notifier.send_sync(report)
            logger.info("시황 리포트 전송 완료")
        except Exception as e:
            logger.error(f"시황 리포트 전송 실패: {e}")

    def _ensure_exchange_trailing(self):
        """목표가 50% 이상 도달 시 거래소 레벨 트레일링 스톱 설정"""
        if self.paper or not self.positions:
            return

        params = STRATEGY_PARAMS
        trail_pct = params.get("trail_pct", 1.5)

        try:
            exchange_positions = self.client.get_all_positions()
            pos_map = {p["symbol"]: p for p in exchange_positions}
        except Exception as e:
            logger.error(f"트레일링 체크용 포지션 조회 실패: {e}")
            return

        for symbol, pos in list(self.positions.items()):
            # 이미 거래소 트레일링 설정됨
            if pos.get("exchange_trailing_set"):
                continue

            entry = float(pos.get("entry_price", 0))
            tp = float(pos.get("take_profit", 0))
            side = pos.get("side", "short")

            if entry <= 0 or tp <= 0:
                continue

            # 거래소에서 현재가 조회
            ex_pos = pos_map.get(symbol)
            if not ex_pos:
                continue
            current_price = float(ex_pos.get("mark_price", 0))
            if current_price <= 0:
                continue

            # 목표까지의 전체 거리와 50% 지점 계산
            target_distance = abs(entry - tp)
            half_target = target_distance * 0.5

            # 50% 도달 여부 체크
            reached_half = False
            if side == "short":
                # 숏: 가격이 entry보다 half_target 이상 내려갔으면
                reached_half = current_price <= entry - half_target
                active_price = entry - half_target  # 활성화 가격
            else:
                # 롱: 가격이 entry보다 half_target 이상 올라갔으면
                reached_half = current_price >= entry + half_target
                active_price = entry + half_target

            if not reached_half:
                continue

            # 트레일링 스톱 거리 계산
            trail_dist = round(current_price * trail_pct / 100, 2)

            short_sym = symbol.split('/')[0]
            logger.info(f"[TRAILING] {short_sym} 목표가 50%+ 도달 → 거래소 트레일링 설정 (거리: ${trail_dist:.2f})")

            try:
                success = self.client.set_trailing_stop(
                    symbol,
                    trailing_stop=trail_dist,
                    active_price=active_price
                )
                if success:
                    pos["exchange_trailing_set"] = True
                    self._save_state()

                    pnl_pct = abs(current_price - entry) / entry * 100 * self.leverage
                    self.notifier.send_sync(
                        f"⛩️ <b>{short_sym} 트레일링 스톱 설정</b>\n"
                        f"현재 수익: {pnl_pct:+.1f}%\n"
                        f"트레일링 거리: ${trail_dist:.2f} ({trail_pct}%)\n"
                        f"활성화 가격: ${active_price:,.2f}"
                    )
                    logger.info(f"[TRAILING] {short_sym} 거래소 트레일링 설정 완료")
            except Exception as e:
                logger.error(f"[TRAILING] {short_sym} 거래소 트레일링 설정 실패: {e}")

    def check_positions(self):
        """포지션 상태만 체크 (수동/거래소 청산 감지 + 트레일링). 자주 호출용."""
        if not self.positions:
            return
        self._check_manual_closes()
        self._ensure_exchange_trailing()

    def run_once(self):
        """한 번 스캔 및 실행"""
        params = STRATEGY_PARAMS

        # BTC 트렌드 업데이트
        if params.get("use_btc_filter", True):
            self._update_btc_trend()

        # 수동 청산 감지 (거래소에 없지만 봇 메모리에 있는 포지션)
        self._check_manual_closes()

        # 각 심볼 데이터 로드
        latest_rows: Dict[str, pd.Series] = {}
        symbol_dfs: Dict[str, pd.DataFrame] = {}  # AI 분석용 df 저장
        for symbol in MAJOR_COINS:
            df = self._fetch_ichimoku_df(symbol, limit=200)
            if df is None or df.empty:
                continue
            latest_rows[symbol] = df.iloc[-1]
            symbol_dfs[symbol] = df

        if not latest_rows:
            logger.warning("유효한 심볼 데이터가 없습니다.")
            return

        # 기존 포지션 청산 체크
        positions_updated = False
        for symbol, pos in list(self.positions.items()):
            row = latest_rows.get(symbol)
            if row is None:
                continue

            # 트레일링 상태 변경 감지
            old_trailing = pos.get("trailing", False)
            old_trail_stop = pos.get("trail_stop", 0)

            exit_info = check_exit_signal(pos, row, params)

            # 트레일링 상태가 변경되었으면 저장 필요
            if pos.get("trailing") != old_trailing or pos.get("trail_stop") != old_trail_stop:
                positions_updated = True

            if exit_info:
                self._close_position(symbol, exit_info)

        # 트레일링 상태 변경 시 저장
        if positions_updated:
            self._save_state()

        # 진입 후보 생성
        candidates = []
        for symbol, row in latest_rows.items():
            if symbol in self.positions:
                continue
            sig = get_entry_signal(
                symbol, row, self.btc_uptrend,
                self.last_exit_times.get(symbol), params
            )
            if sig:
                candidates.append(sig)

        if not candidates:
            logger.info("[WAIT] 진입 후보 없음")
            return

        # 점수순 정렬
        candidates.sort(key=lambda x: (-x["score"], -x["thickness"]))

        # 잔고 기반 진입
        try:
            balance = self._get_balance_free()
        except Exception:
            balance = 0.0

        if balance <= 0:
            logger.warning("USDT 잔고가 없습니다.")
            return

        free = balance
        for cand in candidates:
            if len(self.positions) >= params["max_positions"]:
                logger.info("[LIMIT] 최대 포지션 수 도달")
                break

            # AI 분석용 df 전달
            df = symbol_dfs.get(cand["symbol"])
            used_margin = self._open_position(cand, free, df)
            if used_margin <= 0:
                continue

            free -= used_margin
            if free <= 0:
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
        logger.info(f"IchimokuTrader 루프 시작 [{mode}]")

        # 텔레그램 봇 시작
        await self.telegram_bot.start_polling()

        # 시작 알림
        self.notifier.send_sync(f"🚀 봇 시작됨 [{mode}]")

        self.running = True

        try:
            while True:
                if self.running:
                    try:
                        self.run_once()

                        # 4시간봉 갱신 시 시황 리포트 자동 전송
                        await self._send_periodic_report()

                    except Exception as e:
                        logger.error(f"루프 오류: {e}")
                        self.notifier.notify_error(str(e))

                # 다음 캔들까지 대기
                next_candle = self.data_fetcher.get_next_candle_time(self.timeframe)
                now = datetime.utcnow()
                sleep_seconds = max(60, (next_candle - now).total_seconds())
                logger.info(f"[SLEEP] 다음 캔들까지 {sleep_seconds/60:.1f}분 대기")

                await asyncio.sleep(sleep_seconds)

        except asyncio.CancelledError:
            logger.info("봇 종료")
        finally:
            await self.telegram_bot.stop_polling()

    def run(self):
        """동기 실행"""
        asyncio.run(self.run_async())
