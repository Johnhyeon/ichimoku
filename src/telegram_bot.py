"""텔레그램 봇 - Trojan 스타일 인라인 버튼 UI"""

import logging
import asyncio
import io
from datetime import datetime
from typing import Optional, Callable, Dict
from telegram import Update, Bot, InputFile, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, ContextTypes, filters

from src.config import settings
from src.strategy import fmt_price

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """텔레그램 알림 전송"""

    def __init__(self):
        self.token = settings.TELEGRAM_BOT_TOKEN
        self.chat_id = settings.TELEGRAM_CHAT_ID
        self.bot: Optional[Bot] = None

        if self.token and self.chat_id:
            self.bot = Bot(token=self.token)
            logger.info("텔레그램 봇 초기화 완료")
        else:
            logger.warning("텔레그램 설정이 없습니다")

    async def send_message(self, text: str, reply_markup=None):
        """메시지 전송"""
        if not self.bot:
            return None

        try:
            msg = await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode='HTML',
                reply_markup=reply_markup
            )
            return msg
        except Exception as e:
            logger.error(f"텔레그램 전송 실패: {e}")
            return None

    async def send_photo(self, photo_bytes: bytes, caption: str = "", reply_markup=None):
        """사진 전송"""
        if not self.bot:
            return None

        try:
            msg = await self.bot.send_photo(
                chat_id=self.chat_id,
                photo=io.BytesIO(photo_bytes),
                caption=caption,
                parse_mode='HTML',
                reply_markup=reply_markup
            )
            return msg
        except Exception as e:
            logger.error(f"텔레그램 사진 전송 실패: {e}")
            return None

    def send_sync(self, text: str):
        """동기식 메시지 전송"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.send_message(text))
            else:
                loop.run_until_complete(self.send_message(text))
        except RuntimeError:
            asyncio.run(self.send_message(text))

    def notify_entry(self, symbol: str, side: str, price: float, qty: float, sl: float, tp: float):
        """진입 알림"""
        emoji = "🟢" if side == "long" else "🔴"
        short_sym = symbol.split('/')[0]
        sl_pct = abs((sl - price) / price * 100)
        tp_pct = abs((tp - price) / price * 100)

        text = f"""
{emoji} <b>{side.upper()} 진입</b>

코인: {short_sym}
가격: {fmt_price(price)}
수량: {qty:.4f}
손절: {fmt_price(sl)} (-{sl_pct:.1f}%)
익절: {fmt_price(tp)} (+{tp_pct:.1f}%)
"""
        self.send_sync(text.strip())

    def notify_exit(self, symbol: str, side: str, entry: float, exit_price: float, pnl_pct: float, pnl_usd: float, reason: str):
        """청산 알림"""
        emoji = "💰" if pnl_pct >= 0 else "💸"
        short_sym = symbol.split('/')[0]
        sign = "+" if pnl_pct >= 0 else ""

        text = f"""
{emoji} <b>청산 완료</b>

코인: {short_sym}
방향: {side.upper()}
진입가: {fmt_price(entry)}
청산가: {fmt_price(exit_price)}
수익: {sign}{pnl_pct:.1f}% ({sign}${pnl_usd:.2f})
사유: {reason}
"""
        self.send_sync(text.strip())

    def notify_error(self, message: str):
        """오류 알림"""
        text = f"⚠️ <b>오류 발생</b>\n\n{message}"
        self.send_sync(text)


class TelegramBot:
    """텔레그램 Trojan 스타일 봇"""

    def __init__(self, notifier: TelegramNotifier):
        self.notifier = notifier
        self.app: Optional[Application] = None
        self.running = False

        # 메인 메시지 ID 저장 (채팅별)
        self.main_message_ids: Dict[int, int] = {}

        # 콜백 함수들
        self.get_balance_callback: Optional[Callable] = None
        self.get_positions_callback: Optional[Callable] = None
        self.get_trade_history_callback: Optional[Callable] = None
        self.stop_callback: Optional[Callable] = None
        self.start_callback: Optional[Callable] = None
        self.sync_positions_callback: Optional[Callable] = None

        # 분석 관련 콜백
        self.get_market_report_callback: Optional[Callable] = None
        self.get_no_entry_report_callback: Optional[Callable] = None
        self.get_watch_report_callback: Optional[Callable] = None
        self.get_chart_callback: Optional[Callable] = None
        self.get_overview_chart_callback: Optional[Callable] = None
        self.chat_response_callback: Optional[Callable] = None

        # 잔고 차트 콜백
        self.get_balance_chart_callback: Optional[Callable] = None

        # 거래 정보 콜백
        self.get_funding_rates_callback: Optional[Callable] = None
        self.get_position_sl_tp_callback: Optional[Callable] = None
        self.set_position_sl_tp_callback: Optional[Callable] = None
        self.get_account_stats_callback: Optional[Callable] = None
        self.get_trade_history_exchange_callback: Optional[Callable] = None
        self.get_transaction_log_callback: Optional[Callable] = None

        # 전략별 제어 콜백
        self.get_strategy_status_callback: Optional[Callable] = None
        self.stop_ichimoku_callback: Optional[Callable] = None
        self.start_ichimoku_callback: Optional[Callable] = None
        self.stop_surge_callback: Optional[Callable] = None
        self.start_surge_callback: Optional[Callable] = None
        self.stop_ma100_callback: Optional[Callable] = None
        self.start_ma100_callback: Optional[Callable] = None

        # DCA 콜백
        self.stop_dca_callback: Optional[Callable] = None
        self.start_dca_callback: Optional[Callable] = None
        self.get_dca_summary_callback: Optional[Callable] = None
        self.get_dca_detail_callback: Optional[Callable] = None
        self.get_dca_params_callback: Optional[Callable] = None
        self.set_dca_param_callback: Optional[Callable] = None

        # 설정 콜백
        self.get_settings_callback: Optional[Callable] = None
        self.set_leverage_callback: Optional[Callable] = None
        self.set_position_pct_callback: Optional[Callable] = None

    def set_callbacks(
        self,
        get_balance: Callable,
        get_positions: Callable,
        get_trade_history: Callable = None,
        stop_bot: Callable = None,
        start_bot: Callable = None,
        sync_positions: Callable = None
    ):
        """콜백 설정"""
        self.get_balance_callback = get_balance
        self.get_positions_callback = get_positions
        self.get_trade_history_callback = get_trade_history
        self.stop_callback = stop_bot
        self.start_callback = start_bot
        self.sync_positions_callback = sync_positions

    def set_analysis_callbacks(
        self,
        get_market_report: Callable = None,
        get_no_entry_report: Callable = None,
        get_watch_report: Callable = None,
        get_chart: Callable = None,
        get_overview_chart: Callable = None,
        chat_response: Callable = None
    ):
        """분석 관련 콜백 설정"""
        self.get_market_report_callback = get_market_report
        self.get_no_entry_report_callback = get_no_entry_report
        self.get_watch_report_callback = get_watch_report
        self.get_chart_callback = get_chart
        self.get_overview_chart_callback = get_overview_chart
        self.chat_response_callback = chat_response

    def set_trading_callbacks(
        self,
        get_funding_rates: Callable = None,
        get_position_sl_tp: Callable = None,
        set_position_sl_tp: Callable = None,
        get_account_stats: Callable = None,
        get_trade_history_exchange: Callable = None,
        get_transaction_log: Callable = None
    ):
        """거래 정보 콜백 설정"""
        self.get_funding_rates_callback = get_funding_rates
        self.get_position_sl_tp_callback = get_position_sl_tp
        self.set_position_sl_tp_callback = set_position_sl_tp
        self.get_account_stats_callback = get_account_stats
        self.get_trade_history_exchange_callback = get_trade_history_exchange
        self.get_transaction_log_callback = get_transaction_log

    def set_strategy_callbacks(
        self,
        get_strategy_status: Callable = None,
        stop_ichimoku: Callable = None,
        start_ichimoku: Callable = None,
        stop_surge: Callable = None,
        start_surge: Callable = None,
        stop_ma100: Callable = None,
        start_ma100: Callable = None,
        stop_dca: Callable = None,
        start_dca: Callable = None,
        get_dca_summary: Callable = None,
        get_dca_detail: Callable = None,
        get_dca_params: Callable = None,
        set_dca_param: Callable = None,
    ):
        """전략별 제어 콜백 설정"""
        self.get_strategy_status_callback = get_strategy_status
        self.stop_ichimoku_callback = stop_ichimoku
        self.start_ichimoku_callback = start_ichimoku
        self.stop_surge_callback = stop_surge
        self.start_surge_callback = start_surge
        self.stop_ma100_callback = stop_ma100
        self.start_ma100_callback = start_ma100
        self.stop_dca_callback = stop_dca
        self.start_dca_callback = start_dca
        self.get_dca_summary_callback = get_dca_summary
        self.get_dca_detail_callback = get_dca_detail
        self.get_dca_params_callback = get_dca_params
        self.set_dca_param_callback = set_dca_param

    def set_settings_callbacks(
        self,
        get_settings: Callable = None,
        set_leverage: Callable = None,
        set_position_pct: Callable = None
    ):
        """설정 관련 콜백 등록"""
        self.get_settings_callback = get_settings
        self.set_leverage_callback = set_leverage
        self.set_position_pct_callback = set_position_pct

    async def _safe_edit_message(self, query, text: str, reply_markup=None):
        """메시지 편집 (이미지 메시지인 경우 삭제 후 새 메시지 전송)"""
        try:
            await query.edit_message_text(
                text,
                parse_mode='HTML',
                reply_markup=reply_markup
            )
        except Exception as e:
            if "no text in the message" in str(e).lower():
                # 이미지 메시지인 경우: 삭제하고 새 메시지 전송
                try:
                    await query.message.delete()
                except:
                    pass
                await self.notifier.send_message(text, reply_markup=reply_markup)
            else:
                raise

    # ==================== 키보드 레이아웃 ====================

    def _get_main_keyboard(self) -> InlineKeyboardMarkup:
        """메인 대시보드 키보드"""
        keyboard = [
            [
                InlineKeyboardButton("💰 잔고", callback_data="balance"),
                InlineKeyboardButton("📋 포지션", callback_data="positions"),
            ],
            [
                InlineKeyboardButton("📊 시황분석", callback_data="menu_analysis"),
                InlineKeyboardButton("📈 차트", callback_data="menu_chart"),
            ],
            [
                InlineKeyboardButton("📜 거래이력", callback_data="history"),
                InlineKeyboardButton("📉 거래정보", callback_data="menu_trading"),
            ],
            [
                InlineKeyboardButton("🛒 DCA 현황", callback_data="dca_status"),
                InlineKeyboardButton("🔧 설정", callback_data="menu_settings"),
            ],
            [
                InlineKeyboardButton("⚙️ 봇 제어", callback_data="menu_control"),
                InlineKeyboardButton("🔄 새로고침", callback_data="refresh"),
            ],
        ]
        return InlineKeyboardMarkup(keyboard)

    def _get_analysis_keyboard(self) -> InlineKeyboardMarkup:
        """시황 분석 메뉴 키보드"""
        keyboard = [
            [
                InlineKeyboardButton("🤖 AI 시황", callback_data="market"),
                InlineKeyboardButton("❓ 미진입 이유", callback_data="why"),
            ],
            [
                InlineKeyboardButton("🔭 관심 코인", callback_data="watch"),
            ],
            [
                InlineKeyboardButton("← 뒤로", callback_data="back_main"),
            ],
        ]
        return InlineKeyboardMarkup(keyboard)

    def _get_chart_keyboard(self) -> InlineKeyboardMarkup:
        """차트 메뉴 키보드"""
        keyboard = [
            [
                InlineKeyboardButton("BTC", callback_data="chart_BTC"),
                InlineKeyboardButton("ETH", callback_data="chart_ETH"),
                InlineKeyboardButton("SOL", callback_data="chart_SOL"),
                InlineKeyboardButton("XRP", callback_data="chart_XRP"),
            ],
            [
                InlineKeyboardButton("BNB", callback_data="chart_BNB"),
                InlineKeyboardButton("DOGE", callback_data="chart_DOGE"),
                InlineKeyboardButton("ADA", callback_data="chart_ADA"),
                InlineKeyboardButton("AVAX", callback_data="chart_AVAX"),
            ],
            [
                InlineKeyboardButton("📊 전체 차트", callback_data="overview"),
            ],
            [
                InlineKeyboardButton("← 뒤로", callback_data="back_main"),
            ],
        ]
        return InlineKeyboardMarkup(keyboard)

    def _get_control_keyboard(self) -> InlineKeyboardMarkup:
        """봇 제어 메뉴 키보드"""
        # 전략별 제어가 가능한 경우 (통합 봇)
        if self.get_strategy_status_callback:
            try:
                status = self.get_strategy_status_callback()
            except:
                status = {}

            ich_running = status.get('ichimoku_running', False)
            surge_running = status.get('surge_running', False)
            ma100_running = status.get('ma100_running', False)

            ich_btn = "🔷 Vertex ⏸" if ich_running else "🔷 Vertex ▶️"
            ich_data = "ctrl_ich_stop" if ich_running else "ctrl_ich_start"
            surge_btn = "📉 미러숏 ⏸" if surge_running else "📉 미러숏 ▶️"
            surge_data = "ctrl_surge_stop" if surge_running else "ctrl_surge_start"
            ma100_btn = "📊 MA100 ⏸" if ma100_running else "📊 MA100 ▶️"
            ma100_data = "ctrl_ma100_stop" if ma100_running else "ctrl_ma100_start"

            dca_running = status.get('dca_running', False)
            dca_btn = "🛒 DCA ⏸" if dca_running else "🛒 DCA ▶️"
            dca_data = "ctrl_dca_stop" if dca_running else "ctrl_dca_start"

            keyboard = [
                [InlineKeyboardButton(ich_btn, callback_data=ich_data)],
                [InlineKeyboardButton(surge_btn, callback_data=surge_data)],
                [InlineKeyboardButton(ma100_btn, callback_data=ma100_data)],
                [InlineKeyboardButton(dca_btn, callback_data=dca_data)],
                [InlineKeyboardButton("🔄 동기화", callback_data="sync_positions")],
                [InlineKeyboardButton("← 뒤로", callback_data="back_main")],
            ]
        else:
            # 단독 실행 모드
            status_btn = "⏸ 중지" if self.running else "▶️ 시작"
            status_data = "bot_stop" if self.running else "bot_start"

            keyboard = [
                [
                    InlineKeyboardButton(status_btn, callback_data=status_data),
                    InlineKeyboardButton("🔄 동기화", callback_data="sync_positions"),
                ],
                [InlineKeyboardButton("← 뒤로", callback_data="back_main")],
            ]
        return InlineKeyboardMarkup(keyboard)

    def _get_settings_keyboard(self) -> InlineKeyboardMarkup:
        """설정 메뉴 키보드"""
        settings = {}
        if self.get_settings_callback:
            try:
                settings = self.get_settings_callback()
            except:
                pass

        ich_lev = settings.get('ich_leverage', 20)
        ich_pct = int(settings.get('ich_pct', 5))
        surge_lev = settings.get('surge_leverage', 5)
        surge_pct = int(settings.get('surge_pct', 5))
        ma100_lev = settings.get('ma100_leverage', 5)
        ma100_pct = int(settings.get('ma100_pct', 5))

        keyboard = [
            [InlineKeyboardButton(
                f"🔷 Vertex: {ich_lev}x / {ich_pct}%",
                callback_data="settings_ich"
            )],
            [InlineKeyboardButton(
                f"📉 미러숏: {surge_lev}x / {surge_pct}%",
                callback_data="settings_surge"
            )],
            [InlineKeyboardButton(
                f"📊 MA100: {ma100_lev}x / {ma100_pct}%",
                callback_data="settings_ma100"
            )],
            [InlineKeyboardButton("← 뒤로", callback_data="back_main")],
        ]
        return InlineKeyboardMarkup(keyboard)

    def _get_settings_ich_keyboard(self) -> InlineKeyboardMarkup:
        """Vertex 설정 키보드"""
        settings = {}
        if self.get_settings_callback:
            try:
                settings = self.get_settings_callback()
            except:
                pass

        cur_lev = settings.get('ich_leverage', 20)
        cur_pct = int(settings.get('ich_pct', 5))

        lev_options = [10, 15, 20, 25]
        pct_options = [3, 5, 10, 20]

        lev_buttons = []
        for v in lev_options:
            label = f"{'→' if v == cur_lev else ''}{v}x"
            lev_buttons.append(InlineKeyboardButton(label, callback_data=f"set_ich_lev_{v}"))

        pct_buttons = []
        for v in pct_options:
            label = f"{'→' if v == cur_pct else ''}{v}%"
            pct_buttons.append(InlineKeyboardButton(label, callback_data=f"set_ich_pct_{v}"))

        keyboard = [
            [InlineKeyboardButton("📐 레버리지", callback_data="_noop")],
            lev_buttons,
            [InlineKeyboardButton("💰 진입비율", callback_data="_noop")],
            pct_buttons,
            [InlineKeyboardButton("← 뒤로", callback_data="menu_settings")],
        ]
        return InlineKeyboardMarkup(keyboard)

    def _get_settings_surge_keyboard(self) -> InlineKeyboardMarkup:
        """미러숏 설정 키보드"""
        settings = {}
        if self.get_settings_callback:
            try:
                settings = self.get_settings_callback()
            except:
                pass

        cur_lev = settings.get('surge_leverage', 5)
        cur_pct = int(settings.get('surge_pct', 5))

        lev_options = [3, 5, 7, 10]
        pct_options = [3, 5, 10, 20]

        lev_buttons = []
        for v in lev_options:
            label = f"{'→' if v == cur_lev else ''}{v}x"
            lev_buttons.append(InlineKeyboardButton(label, callback_data=f"set_surge_lev_{v}"))

        pct_buttons = []
        for v in pct_options:
            label = f"{'→' if v == cur_pct else ''}{v}%"
            pct_buttons.append(InlineKeyboardButton(label, callback_data=f"set_surge_pct_{v}"))

        keyboard = [
            [InlineKeyboardButton("📐 레버리지", callback_data="_noop")],
            lev_buttons,
            [InlineKeyboardButton("💰 진입비율", callback_data="_noop")],
            pct_buttons,
            [InlineKeyboardButton("← 뒤로", callback_data="menu_settings")],
        ]
        return InlineKeyboardMarkup(keyboard)

    def _get_settings_ma100_keyboard(self) -> InlineKeyboardMarkup:
        """MA100 설정 키보드"""
        settings = {}
        if self.get_settings_callback:
            try:
                settings = self.get_settings_callback()
            except:
                pass

        cur_lev = settings.get('ma100_leverage', 5)
        cur_pct = int(settings.get('ma100_pct', 5))

        lev_options = [3, 5, 7, 10]
        pct_options = [3, 5, 10, 20]

        lev_buttons = []
        for v in lev_options:
            label = f"{'→' if v == cur_lev else ''}{v}x"
            lev_buttons.append(InlineKeyboardButton(label, callback_data=f"set_ma100_lev_{v}"))

        pct_buttons = []
        for v in pct_options:
            label = f"{'→' if v == cur_pct else ''}{v}%"
            pct_buttons.append(InlineKeyboardButton(label, callback_data=f"set_ma100_pct_{v}"))

        keyboard = [
            [InlineKeyboardButton("📐 레버리지", callback_data="_noop")],
            lev_buttons,
            [InlineKeyboardButton("💰 진입비율", callback_data="_noop")],
            pct_buttons,
            [InlineKeyboardButton("← 뒤로", callback_data="menu_settings")],
        ]
        return InlineKeyboardMarkup(keyboard)

    def _get_back_keyboard(self) -> InlineKeyboardMarkup:
        """뒤로가기만 있는 키보드"""
        keyboard = [[InlineKeyboardButton("← 뒤로", callback_data="back_main")]]
        return InlineKeyboardMarkup(keyboard)

    def _get_trading_keyboard(self) -> InlineKeyboardMarkup:
        """거래정보 메뉴 키보드"""
        keyboard = [
            [
                InlineKeyboardButton("💸 펀딩비", callback_data="funding_rates"),
                InlineKeyboardButton("🎯 SL/TP", callback_data="sl_tp_info"),
            ],
            [
                InlineKeyboardButton("💰 펀딩/수수료", callback_data="fees_info"),
            ],
            [
                InlineKeyboardButton("📊 통계 (7일)", callback_data="stats_7"),
                InlineKeyboardButton("📊 통계 (30일)", callback_data="stats_30"),
            ],
            [
                InlineKeyboardButton("← 뒤로", callback_data="back_main"),
            ],
        ]
        return InlineKeyboardMarkup(keyboard)

    def _get_sl_tp_keyboard(self, positions: list) -> InlineKeyboardMarkup:
        """SL/TP 수정용 키보드"""
        keyboard = []

        for pos in positions:
            symbol = pos.get('symbol', '')
            short_sym = symbol.split('/')[0]
            keyboard.append([
                InlineKeyboardButton(f"✏️ {short_sym} SL/TP 수정", callback_data=f"edit_sltp_{short_sym}")
            ])

        keyboard.append([InlineKeyboardButton("← 뒤로", callback_data="menu_trading")])
        return InlineKeyboardMarkup(keyboard)

    # ==================== 대시보드 생성 ====================

    def _format_position_line(self, p: dict) -> str:
        """포지션 한 줄 포맷"""
        emoji = "📈" if p['side'] == 'long' else "📉"
        short_sym = p['symbol'].split('/')[0]
        pnl_pct = float(p.get('pnl_pct', 0))
        pnl_sign = "+" if pnl_pct >= 0 else ""
        pnl_emoji = "🟢" if pnl_pct >= 0 else "🔴"
        leverage = int(p.get('leverage', 20))
        return f"\n{emoji} <b>{short_sym}</b> {p['side'].upper()} {pnl_emoji} <code>{pnl_sign}{pnl_pct:.1f}%</code> (x{leverage})"

    def _build_dashboard_text(self) -> str:
        """메인 대시보드 텍스트 생성"""
        now = datetime.utcnow().strftime('%H:%M:%S')

        # 전략별 상태 (통합 봇)
        strategy_status_text = ""
        if self.get_strategy_status_callback:
            try:
                st = self.get_strategy_status_callback()
                ich_emoji = "🟢" if st.get('ichimoku_running') else "🔴"
                surge_emoji = "🟢" if st.get('surge_running') else "🔴"
                ma100_emoji = "🟢" if st.get('ma100_running') else "🔴"
                surge_pnl = st.get('surge_daily_pnl', 0)
                surge_limit = st.get('surge_daily_limit', 0)
                pnl_sign = "+" if surge_pnl >= 0 else ""

                dca_emoji = "🟢" if st.get('dca_running') else "🔴"

                strategy_status_text = f"""
🔷{ich_emoji} 📉{surge_emoji} 📊{ma100_emoji} 🛒{dca_emoji}
🕐 갱신: {now} UTC

📊 <b>미러숏 오늘</b>
├ 손익: <code>{pnl_sign}${surge_pnl:,.2f}</code>
└ 한도: <code>${surge_limit:,.0f}</code>"""

                # DCA 적립 현황
                if self.get_dca_summary_callback:
                    try:
                        dca_summary = self.get_dca_summary_callback()
                        if dca_summary and dca_summary != "적립 내역 없음":
                            strategy_status_text += f"\n\n🛒 <b>DCA 적립</b>\n{dca_summary}"
                    except:
                        pass
            except:
                strategy_status_text = f"\n🕐 갱신: {now} UTC"
        else:
            status_emoji = "🟢" if self.running else "🔴"
            status_text = "실행중" if self.running else "중지됨"
            strategy_status_text = f"\n{status_emoji} 상태: <b>{status_text}</b>\n🕐 갱신: {now} UTC"

        # 잔고 정보
        balance_text = ""
        if self.get_balance_callback:
            try:
                balance = self.get_balance_callback()
                if isinstance(balance, dict):
                    total = balance.get("total", 0)
                    unrealized = balance.get("unrealized_pnl", 0)
                    equity = balance.get("equity", total)
                    pnl_sign = "+" if unrealized >= 0 else ""
                    balance_text = f"""
💰 <b>자산</b>
├ 잔고: <code>${total:,.2f}</code>
├ 미실현: <code>{pnl_sign}${unrealized:,.2f}</code>
└ 평가: <code>${equity:,.2f}</code>"""
                else:
                    balance_text = f"\n💰 잔고: <code>${balance:,.2f}</code>"
            except Exception as e:
                balance_text = f"\n💰 잔고: 조회 실패"

        # 포지션 정보
        positions_text = ""
        if self.get_positions_callback:
            try:
                positions = self.get_positions_callback()
                if positions:
                    # 전략별 그룹화
                    ichimoku_pos = [p for p in positions if p.get('strategy') in ('fractals', 'ichimoku')]
                    surge_pos = [p for p in positions if p.get('strategy') in ('surge', 'mirror_short')]
                    ma100_pos = [p for p in positions if p.get('strategy') == 'ma100']
                    other_pos = [p for p in positions if p.get('strategy') not in ('fractals', 'ichimoku', 'surge', 'mirror_short', 'ma100')]

                    has_groups = bool(ichimoku_pos) or bool(surge_pos) or bool(ma100_pos)

                    if has_groups:
                        positions_text = "\n\n📋 <b>포지션</b>"
                        if ichimoku_pos:
                            positions_text += "\n\n🔷 <b>Vertex</b>"
                            for p in ichimoku_pos:
                                positions_text += self._format_position_line(p)
                        if surge_pos:
                            positions_text += "\n\n📉 <b>미러숏</b>"
                            for p in surge_pos:
                                positions_text += self._format_position_line(p)
                        if ma100_pos:
                            positions_text += "\n\n📊 <b>MA100</b>"
                            for p in ma100_pos:
                                positions_text += self._format_position_line(p)
                        if other_pos:
                            for p in other_pos:
                                positions_text += self._format_position_line(p)
                    else:
                        positions_text = "\n\n📋 <b>포지션</b>"
                        for p in positions:
                            positions_text += self._format_position_line(p)
                else:
                    positions_text = "\n\n📋 <b>포지션</b>\n없음"
            except:
                positions_text = "\n\n📋 포지션: 조회 실패"

        text = f"""
🤖 <b>Trading Bot</b>
{strategy_status_text}
{balance_text}{positions_text}

━━━━━━━━━━━━━━━━━━
아래 버튼을 눌러 기능을 선택하세요
"""
        return text.strip()

    # ==================== 명령어 핸들러 ====================

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """/start - 메인 대시보드 표시"""
        chat_id = update.effective_chat.id

        # 기존 메인 메시지 삭제 시도
        if chat_id in self.main_message_ids:
            try:
                await context.bot.delete_message(chat_id, self.main_message_ids[chat_id])
            except:
                pass

        # 새 대시보드 메시지 전송
        text = self._build_dashboard_text()
        msg = await update.message.reply_text(
            text,
            parse_mode='HTML',
            reply_markup=self._get_main_keyboard()
        )

        if msg:
            self.main_message_ids[chat_id] = msg.message_id

    async def cmd_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """/menu - 메인 대시보드 표시 (별칭)"""
        await self.cmd_start(update, context)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """일반 텍스트 메시지 처리 - AI 챗봇 응답"""
        user_message = update.message.text

        if not user_message:
            return

        # 명령어는 무시 (/ 로 시작)
        if user_message.startswith('/'):
            return

        if not self.chat_response_callback:
            await update.message.reply_text(
                "💬 AI 채팅 기능을 사용할 수 없습니다.",
                parse_mode='HTML'
            )
            return

        # 타이핑 표시
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )

        try:
            response = await self.chat_response_callback(user_message)

            # 응답 길이 제한
            if len(response) > 4000:
                response = response[:4000] + "\n\n... (생략)"

            await update.message.reply_text(
                f"🤖 {response}",
                parse_mode='HTML'
            )
        except Exception as e:
            logger.error(f"채팅 응답 실패: {e}")
            await update.message.reply_text(
                f"❌ 응답 생성 중 오류가 발생했습니다: {e}",
                parse_mode='HTML'
            )

    # ==================== 콜백 핸들러 ====================

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """인라인 버튼 콜백 처리"""
        query = update.callback_query
        await query.answer()

        data = query.data
        chat_id = update.effective_chat.id
        message_id = query.message.message_id

        # 메인으로 돌아가기
        if data == "back_main" or data == "refresh":
            text = self._build_dashboard_text()
            await self._safe_edit_message(query, text, self._get_main_keyboard())
            return

        # 잔고 상세
        if data == "balance":
            await self._show_balance(query)
            return

        # 잔고 차트 (7일)
        if data == "balance_chart":
            await self._show_balance_chart(query, days=7)
            return

        # 잔고 차트 (전체)
        if data == "balance_chart_all":
            await self._show_balance_chart(query, days=365)
            return

        # 포지션 상세
        if data == "positions":
            await self._show_positions(query)
            return

        # 거래 이력
        if data == "history":
            await self._show_trade_history(query)
            return

        # DCA 현황
        if data == "dca_status":
            await self._show_dca_status(query)
            return

        if data == "dca_settings":
            await self._show_dca_settings(query)
            return

        if data == "dca_set_amount":
            await self._show_dca_param_options(
                query, "base_amount_usdt", "💵 매수금액",
                [3, 5, 7, 10, 15, 20, 30, 50], fmt="${}")
            return

        if data == "dca_set_interval":
            await self._show_dca_param_options(
                query, "interval_hours", "⏱ 인터벌",
                [4, 6, 8, 12, 24, 48], fmt="{}시간")
            return

        if data == "dca_set_ratio":
            # BTC:ETH 비율 프리셋
            ratio_options = [
                ("BTC 20/ETH 80", 0.2), ("BTC 30/ETH 70", 0.3),
                ("BTC 40/ETH 60", 0.4), ("BTC 50/ETH 50", 0.5),
                ("BTC 60/ETH 40", 0.6), ("BTC 70/ETH 30", 0.7),
                ("BTC 80/ETH 20", 0.8), ("BTC 100", 1.0),
                ("ETH 100", 0.0),
            ]
            buttons = []
            row = []
            for label, btc_r in ratio_options:
                row.append(InlineKeyboardButton(label, callback_data=f"dca_ratio_{btc_r}"))
                if len(row) >= 2:
                    buttons.append(row)
                    row = []
            if row:
                buttons.append(row)
            buttons.append([InlineKeyboardButton("◀️ 설정으로", callback_data="dca_settings")])
            current = ""
            if self.get_dca_params_callback:
                p = self.get_dca_params_callback()
                current = f"\n현재: BTC {p.get('btc_ratio',0.4)*100:.0f}% / ETH {p.get('eth_ratio',0.6)*100:.0f}%"
            await self._safe_edit_message(
                query, f"📊 <b>BTC/ETH 비율 변경</b>{current}",
                InlineKeyboardMarkup(buttons))
            return

        if data == "dca_set_reserve":
            await self._show_dca_param_options(
                query, "min_futures_reserve", "🏦 선물 마진 유보액",
                [200, 300, 500, 1000, 2000, 3000], fmt="${:,}")
            return

        if data == "dca_set_bonus":
            await self._show_dca_param_options(
                query, "weekly_bonus_pct", "🎁 주간 보너스 비율 (선물수익의 %)",
                [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5], fmt="{:.0%}")
            return

        # DCA 비율 변경 처리
        if data.startswith("dca_ratio_"):
            btc_ratio = float(data.split("_")[2])
            eth_ratio = round(1.0 - btc_ratio, 2)
            if self.set_dca_param_callback:
                self.set_dca_param_callback("btc_ratio", btc_ratio)
                self.set_dca_param_callback("eth_ratio", eth_ratio)
            text = f"✅ 비율 변경: BTC {btc_ratio*100:.0f}% / ETH {eth_ratio*100:.0f}%"
            await self._safe_edit_message(query, text, InlineKeyboardMarkup([
                [InlineKeyboardButton("◀️ DCA 설정", callback_data="dca_settings")]
            ]))
            return

        # DCA 파라미터 값 변경 처리
        if data.startswith("dca_val_"):
            parts = data.split("_", 3)  # dca_val_key_value
            param_key = parts[2]
            value = float(parts[3])
            if self.set_dca_param_callback:
                result = self.set_dca_param_callback(param_key, value)
            else:
                result = "❌ 설정 변경 불가"
            await self._safe_edit_message(query, result, InlineKeyboardMarkup([
                [InlineKeyboardButton("◀️ DCA 설정", callback_data="dca_settings")]
            ]))
            return

        # 시황 분석 메뉴
        if data == "menu_analysis":
            text = "📊 <b>시황 분석</b>\n\n원하는 분석을 선택하세요"
            await self._safe_edit_message(query, text, self._get_analysis_keyboard())
            return

        # 차트 메뉴
        if data == "menu_chart":
            text = "📈 <b>차트</b>\n\n코인을 선택하세요"
            await self._safe_edit_message(query, text, self._get_chart_keyboard())
            return

        # 거래정보 메뉴
        if data == "menu_trading":
            text = "📉 <b>거래정보</b>\n\n바이빗 거래 정보를 조회합니다"
            await self._safe_edit_message(query, text, self._get_trading_keyboard())
            return

        # 펀딩비 조회
        if data == "funding_rates":
            await self._show_funding_rates(query)
            return

        # SL/TP 정보
        if data == "sl_tp_info":
            await self._show_sl_tp_info(query)
            return

        # SL/TP 수정
        if data.startswith("edit_sltp_"):
            symbol = data.replace("edit_sltp_", "") + "/USDT:USDT"
            await self._edit_sl_tp(query, symbol)
            return

        # 통계 조회
        if data == "stats_7":
            await self._show_account_stats(query, 7)
            return

        if data == "stats_30":
            await self._show_account_stats(query, 30)
            return

        # 펀딩/수수료 내역
        if data == "fees_info":
            await self._show_fees_info(query)
            return

        # 설정 메뉴
        if data == "menu_settings":
            text = "🔧 <b>설정</b>\n\n전략을 선택하여 레버리지/진입비율을 변경하세요"
            await self._safe_edit_message(query, text, self._get_settings_keyboard())
            return

        # Vertex 설정 화면
        if data == "settings_ich":
            settings = {}
            if self.get_settings_callback:
                try:
                    settings = self.get_settings_callback()
                except:
                    pass
            cur_lev = settings.get('ich_leverage', 20)
            cur_pct = int(settings.get('ich_pct', 5))
            text = f"🔷 <b>Vertex 설정</b>\n\n현재: {cur_lev}x / {cur_pct}%"
            await self._safe_edit_message(query, text, self._get_settings_ich_keyboard())
            return

        # 미러숏 설정 화면
        if data == "settings_surge":
            settings = {}
            if self.get_settings_callback:
                try:
                    settings = self.get_settings_callback()
                except:
                    pass
            cur_lev = settings.get('surge_leverage', 5)
            cur_pct = int(settings.get('surge_pct', 5))
            text = f"📉 <b>미러숏 설정</b>\n\n현재: {cur_lev}x / {cur_pct}%"
            await self._safe_edit_message(query, text, self._get_settings_surge_keyboard())
            return

        # MA100 설정 화면
        if data == "settings_ma100":
            settings = {}
            if self.get_settings_callback:
                try:
                    settings = self.get_settings_callback()
                except:
                    pass
            cur_lev = settings.get('ma100_leverage', 5)
            cur_pct = int(settings.get('ma100_pct', 5))
            text = f"📊 <b>MA100 설정</b>\n\n현재: {cur_lev}x / {cur_pct}%"
            await self._safe_edit_message(query, text, self._get_settings_ma100_keyboard())
            return

        # 설정 변경 콜백
        if data.startswith("set_ich_lev_") or data.startswith("set_surge_lev_") or data.startswith("set_ma100_lev_"):
            await self._handle_set_leverage(query, data)
            return

        if data.startswith("set_ich_pct_") or data.startswith("set_surge_pct_") or data.startswith("set_ma100_pct_"):
            await self._handle_set_position_pct(query, data)
            return

        # noop (라벨용 버튼)
        if data == "_noop":
            return

        # 봇 제어 메뉴
        if data == "menu_control":
            if self.get_strategy_status_callback:
                try:
                    st = self.get_strategy_status_callback()
                    ich_status = "🟢 실행중" if st.get('ichimoku_running') else "🔴 중지됨"
                    surge_status = "🟢 실행중" if st.get('surge_running') else "🔴 중지됨"
                    text = f"⚙️ <b>봇 제어</b>\n\n🔷 Vertex: {ich_status}\n📉 미러숏: {surge_status}"
                except:
                    text = "⚙️ <b>봇 제어</b>"
            else:
                status = "🟢 실행중" if self.running else "🔴 중지됨"
                text = f"⚙️ <b>봇 제어</b>\n\n현재 상태: {status}"
            await self._safe_edit_message(query, text, self._get_control_keyboard())
            return

        # 전략별 개별 제어 (통합 봇)
        if data == "ctrl_ich_stop":
            if self.stop_ichimoku_callback:
                self.stop_ichimoku_callback()
            text = "⏸ Vertex 전략 중지됨"
            await self._safe_edit_message(query, text, self._get_control_keyboard())
            return

        if data == "ctrl_ich_start":
            if self.start_ichimoku_callback:
                self.start_ichimoku_callback()
            text = "▶️ Vertex 전략 시작됨"
            await self._safe_edit_message(query, text, self._get_control_keyboard())
            return

        if data == "ctrl_surge_stop":
            if self.stop_surge_callback:
                self.stop_surge_callback()
            text = "⏸ 미러숏 전략 중지됨"
            await self._safe_edit_message(query, text, self._get_control_keyboard())
            return

        if data == "ctrl_surge_start":
            if self.start_surge_callback:
                self.start_surge_callback()
            text = "▶️ 미러숏 전략 시작됨"
            await self._safe_edit_message(query, text, self._get_control_keyboard())
            return

        if data == "ctrl_ma100_stop":
            if self.stop_ma100_callback:
                self.stop_ma100_callback()
            text = "⏸ MA100 전략 중지됨"
            await self._safe_edit_message(query, text, self._get_control_keyboard())
            return

        if data == "ctrl_ma100_start":
            if self.start_ma100_callback:
                self.start_ma100_callback()
            text = "▶️ MA100 전략 시작됨"
            await self._safe_edit_message(query, text, self._get_control_keyboard())
            return

        if data == "ctrl_dca_stop":
            if self.stop_dca_callback:
                self.stop_dca_callback()
            text = "⏸ DCA 적립 중지됨"
            await self._safe_edit_message(query, text, self._get_control_keyboard())
            return

        if data == "ctrl_dca_start":
            if self.start_dca_callback:
                self.start_dca_callback()
            text = "▶️ DCA 적립 시작됨"
            await self._safe_edit_message(query, text, self._get_control_keyboard())
            return

        # 봇 시작/중지 (단독 실행 모드)
        if data == "bot_start":
            if self.start_callback:
                self.start_callback()
                self.running = True
            text = "✅ 봇이 시작되었습니다"
            await self._safe_edit_message(query, text, self._get_control_keyboard())
            return

        if data == "bot_stop":
            if self.stop_callback:
                self.stop_callback()
                self.running = False
            text = "⏸ 봇이 중지되었습니다"
            await self._safe_edit_message(query, text, self._get_control_keyboard())
            return

        # 포지션 동기화
        if data == "sync_positions":
            await self._sync_positions(query)
            return

        # AI 시황 분석
        if data == "market":
            await self._show_market_analysis(query)
            return

        # 미진입 이유
        if data == "why":
            await self._show_why_no_entry(query)
            return

        # 관심 코인
        if data == "watch":
            await self._show_watch_list(query)
            return

        # 개별 차트
        if data.startswith("chart_"):
            symbol = data.replace("chart_", "")
            await self._show_chart(query, symbol)
            return

        # 전체 차트
        if data == "overview":
            await self._show_overview_chart(query)
            return

    # ==================== 세부 화면 ====================

    def _get_balance_keyboard(self) -> InlineKeyboardMarkup:
        """잔고 상세 화면 키보드"""
        keyboard = [
            [
                InlineKeyboardButton("📈 최근 7일", callback_data="balance_chart"),
                InlineKeyboardButton("📊 전체 추이", callback_data="balance_chart_all"),
            ],
            [InlineKeyboardButton("← 뒤로", callback_data="back_main")],
        ]
        return InlineKeyboardMarkup(keyboard)

    async def _show_balance(self, query):
        """잔고 상세 표시"""
        if not self.get_balance_callback:
            await self._safe_edit_message(query, "❌ 잔고 조회 불가", self._get_back_keyboard())
            return

        try:
            balance = self.get_balance_callback()
            if isinstance(balance, dict):
                total = balance.get("total", 0)
                free = balance.get("free", 0)
                used = balance.get("used", 0)
                unrealized = balance.get("unrealized_pnl", 0)
                equity = balance.get("equity", total)
                pnl_sign = "+" if unrealized >= 0 else ""

                text = f"""
💰 <b>잔고 상세</b>

┌ 총 잔고: <code>${total:,.2f}</code>
├ 가용: <code>${free:,.2f}</code>
├ 마진 사용: <code>${used:,.2f}</code>
├ 미실현 손익: <code>{pnl_sign}${unrealized:,.2f}</code>
└ 평가 자산: <code>${equity:,.2f}</code>
"""
            else:
                text = f"💰 잔고: <code>${balance:,.2f}</code>"

            await self._safe_edit_message(query, text.strip(), self._get_balance_keyboard())
        except Exception as e:
            await self._safe_edit_message(query, f"❌ 조회 실패: {e}", self._get_back_keyboard())

    async def _show_balance_chart(self, query, days: int = 7):
        """잔고 추이 차트 표시"""
        label = "전체" if days > 30 else f"최근 {days}일"
        await self._safe_edit_message(query, f"📈 잔고 추이 차트 생성 중... ({label})")

        if not self.get_balance_chart_callback:
            await self._safe_edit_message(query, "❌ 잔고 차트 기능 사용 불가", self._get_balance_keyboard())
            return

        try:
            chart_bytes = await self.get_balance_chart_callback(days=days)
            if chart_bytes:
                await self.notifier.send_photo(
                    chart_bytes,
                    caption=f"📈 잔고 추이 ({label})",
                    reply_markup=self._get_back_keyboard()
                )
                await self._safe_edit_message(
                    query,
                    f"📈 <b>잔고 추이 ({label})</b>\n\n차트가 전송되었습니다.",
                    self._get_balance_keyboard()
                )
            else:
                await self._safe_edit_message(
                    query,
                    "❌ 잔고 이력이 부족합니다.\n봇 실행 후 데이터가 쌓이면 차트를 볼 수 있습니다.",
                    self._get_balance_keyboard()
                )
        except Exception as e:
            await self._safe_edit_message(query, f"❌ 차트 생성 실패: {e}", self._get_balance_keyboard())

    async def _show_positions(self, query):
        """포지션 상세 표시"""
        if not self.get_positions_callback:
            await self._safe_edit_message(query, "❌ 포지션 조회 불가", self._get_back_keyboard())
            return

        try:
            positions = self.get_positions_callback()
            if positions:
                text = "📋 <b>포지션 상세</b>\n"

                # 전략별 그룹화
                ichimoku_pos = [p for p in positions if p.get('strategy') in ('fractals', 'ichimoku')]
                surge_pos = [p for p in positions if p.get('strategy') in ('surge', 'mirror_short')]
                ma100_pos = [p for p in positions if p.get('strategy') == 'ma100']
                other_pos = [p for p in positions if p.get('strategy') not in ('fractals', 'ichimoku', 'surge', 'mirror_short', 'ma100')]
                has_groups = bool(ichimoku_pos) or bool(surge_pos) or bool(ma100_pos)

                ordered_positions = []
                if has_groups:
                    if ichimoku_pos:
                        ordered_positions.append(("🔷 <b>Vertex</b>", ichimoku_pos))
                    if surge_pos:
                        ordered_positions.append(("📉 <b>미러숏</b>", surge_pos))
                    if ma100_pos:
                        ordered_positions.append(("📊 <b>MA100</b>", ma100_pos))
                    if other_pos:
                        ordered_positions.append(("📋 <b>기타</b>", other_pos))
                else:
                    ordered_positions.append((None, positions))

                for group_label, group_positions in ordered_positions:
                    if group_label:
                        text += f"\n{group_label}\n"
                    for p in group_positions:
                        emoji = "📈" if p['side'] == 'long' else "📉"
                        short_sym = p['symbol'].split('/')[0]
                        side = p.get('side', 'long')

                        pnl_usd = float(p.get('pnl', 0))
                        pnl_pct = float(p.get('pnl_pct', 0))  # 레버리지 적용
                        leverage = int(p.get('leverage', 20))
                        pnl_sign = "+" if pnl_pct >= 0 else ""
                        pnl_emoji = "🟢" if pnl_pct >= 0 else "🔴"

                        entry = float(p.get('entry_price', 0))
                        current = float(p.get('current_price', 0))
                        size = float(p.get('size', 0))
                        sl = float(p.get('stop_loss', 0))
                        tp = float(p.get('take_profit', 0))

                        # 실제 가격 변동률 (레버리지 미적용)
                        if entry > 0 and current > 0:
                            if side == 'long':
                                price_change = (current - entry) / entry * 100
                            else:
                                price_change = (entry - current) / entry * 100
                        else:
                            price_change = pnl_pct / leverage if leverage > 0 else 0

                        price_sign = "+" if price_change >= 0 else ""

                        # 손절/익절까지 거리 (현재가 기준, 레버리지 적용)
                        if current > 0 and sl > 0:
                            if side == 'long':
                                sl_dist = (current - sl) / current * 100 * leverage
                            else:
                                sl_dist = (sl - current) / current * 100 * leverage
                        else:
                            sl_dist = 0

                        if current > 0 and tp > 0:
                            if side == 'long':
                                tp_dist = (tp - current) / current * 100 * leverage
                            else:
                                tp_dist = (current - tp) / current * 100 * leverage
                        else:
                            tp_dist = 0

                        text += f"\n{emoji} <b>{short_sym}</b> {p['side'].upper()} (x{leverage})"
                        text += f"\n┌ 진입: <code>{fmt_price(entry)}</code>"
                        if current > 0:
                            text += f" → 현재: <code>{fmt_price(current)}</code>"
                        text += f"\n├ 가격변동: <code>{price_sign}{price_change:.2f}%</code>"
                        text += f"\n├ {pnl_emoji} 수익률: <code>{pnl_sign}{pnl_pct:.1f}%</code> ({pnl_sign}${pnl_usd:.2f})"
                        text += f"\n├ 수량: <code>{size:.4f}</code>"
                        if sl > 0:
                            sl_emoji = "🟡" if sl_dist > 0 else "🔴"
                            text += f"\n├ {sl_emoji} 손절: <code>{fmt_price(sl)}</code> ({sl_dist:+.1f}%)"
                        if tp > 0:
                            tp_emoji = "🟡" if tp_dist > 0 else "🟢"
                            text += f"\n└ {tp_emoji} 익절: <code>{fmt_price(tp)}</code> ({tp_dist:+.1f}%)"
                        text += "\n"
            else:
                text = "📋 <b>포지션</b>\n\n현재 보유중인 포지션이 없습니다"

            await self._safe_edit_message(query, text.strip(), self._get_back_keyboard())
        except Exception as e:
            await self._safe_edit_message(query, f"❌ 조회 실패: {e}", self._get_back_keyboard())

    async def _show_trade_history(self, query):
        """거래 이력 표시 (바이빗 API에서 직접 조회)"""
        await self._safe_edit_message(query, "📜 거래 이력 조회 중...")

        # 바이빗 API에서 직접 조회 시도
        history = []
        if self.get_trade_history_exchange_callback:
            try:
                history = self.get_trade_history_exchange_callback(7)  # 최근 7일
            except:
                pass

        # 실패하면 봇 메모리에서 조회
        if not history and self.get_trade_history_callback:
            try:
                history = self.get_trade_history_callback()
            except:
                pass

        try:
            if history:
                # 최근 순으로 정렬
                history = sorted(
                    history,
                    key=lambda x: x.get('closed_at') or '',
                    reverse=True
                )

                # 통계 계산
                total_pnl = sum(h.get('pnl_usd', 0) for h in history)
                wins = sum(1 for h in history if h.get('pnl_usd', 0) > 0)
                losses = sum(1 for h in history if h.get('pnl_usd', 0) < 0)
                win_rate = (wins / len(history) * 100) if history else 0

                text = f"📜 <b>거래 이력</b> (최근 {len(history)}건)\n\n"
                text += f"📊 승률: <code>{win_rate:.0f}%</code> ({wins}승 {losses}패)\n"
                total_sign = "+" if total_pnl >= 0 else ""
                text += f"💵 총 손익: <code>{total_sign}${total_pnl:.2f}</code>\n"
                text += "━━━━━━━━━━━━━━━━\n"

                # 전략별 통계
                strat_map = {'fractals': '🔷', 'ichimoku': '🔷', 'mirror_short': '📉', 'surge': '📉', 'ma100': '📊'}
                strat_groups = {}
                for h in history:
                    s = h.get('strategy', '')
                    strat_groups.setdefault(s, []).append(h)

                for s_name, s_trades in strat_groups.items():
                    s_emoji = strat_map.get(s_name, '📋')
                    s_label = {'fractals': 'Vertex', 'ichimoku': 'Vertex', 'mirror_short': '미러숏', 'surge': '미러숏', 'ma100': 'MA100'}.get(s_name, s_name or '기타')
                    s_pnl = sum(t.get('pnl_usd', 0) for t in s_trades)
                    s_wins = sum(1 for t in s_trades if t.get('pnl_usd', 0) > 0)
                    s_total = len(s_trades)
                    s_wr = (s_wins / s_total * 100) if s_total else 0
                    s_sign = "+" if s_pnl >= 0 else ""
                    text += f"{s_emoji} {s_label}: {s_wins}/{s_total} ({s_wr:.0f}%) <code>{s_sign}${s_pnl:.2f}</code>\n"

                text += "━━━━━━━━━━━━━━━━\n"

                # 전략별로 그룹화하여 표시
                strat_order = [
                    ('fractals', '🔷 <b>Vertex</b>'),
                    ('ichimoku', '🔷 <b>Vertex</b>'),
                    ('mirror_short', '📉 <b>미러숏</b>'),
                    ('surge', '📉 <b>미러숏</b>'),
                    ('ma100', '📊 <b>MA100</b>'),
                ]
                shown_labels = set()

                for strat_key, strat_label in strat_order:
                    group = [h for h in history if h.get('strategy', '') == strat_key]
                    if not group:
                        continue

                    # mirror_short / surge 중복 라벨 방지
                    label_key = strat_label
                    if label_key in shown_labels:
                        continue
                    shown_labels.add(label_key)

                    # surge도 합산
                    if strat_key == 'mirror_short':
                        group += [h for h in history if h.get('strategy', '') == 'surge']
                        group.sort(key=lambda x: x.get('closed_at') or '', reverse=True)

                    text += f"\n{strat_label}\n"

                    for h in group[:7]:
                        symbol = h.get('symbol', '')
                        short_sym = symbol.split('/')[0] if '/' in symbol else symbol
                        pnl_pct = float(h.get('pnl_pct', 0))
                        pnl_usd = float(h.get('pnl_usd', 0))
                        reason = h.get('reason', '')
                        closed_at = h.get('closed_at')
                        side = h.get('side', 'long')
                        side_emoji = "📈" if side == 'long' else "📉"

                        emoji = "✅" if pnl_usd >= 0 else "❌"
                        pnl_sign = "+" if pnl_pct >= 0 else ""

                        time_str = ""
                        if closed_at:
                            if hasattr(closed_at, 'strftime'):
                                time_str = closed_at.strftime("%m/%d %H:%M")
                            else:
                                time_str = str(closed_at)[:16]

                        entry = float(h.get('entry_price', 0))
                        exit_p = float(h.get('exit_price', 0))

                        text += f"\n{emoji} {side_emoji} <b>{short_sym}</b> {side.upper()}"
                        if entry > 0 and exit_p > 0:
                            text += f"\n   {fmt_price(entry)} → {fmt_price(exit_p)}"
                        text += f"\n   {pnl_sign}{pnl_pct:.1f}% (<code>{pnl_sign}${pnl_usd:.2f}</code>)"
                        if reason:
                            text += f" | {reason}"
                        if time_str:
                            text += f"\n   <code>{time_str}</code>"

                # 전략 태그 없는 거래
                other = [h for h in history if h.get('strategy', '') not in ('fractals', 'ichimoku', 'mirror_short', 'surge')]
                if other:
                    text += "\n\n📋 <b>기타</b>\n"
                    for h in other[:5]:
                        symbol = h.get('symbol', '')
                        short_sym = symbol.split('/')[0] if '/' in symbol else symbol
                        pnl_usd = float(h.get('pnl_usd', 0))
                        pnl_pct = float(h.get('pnl_pct', 0))
                        pnl_sign = "+" if pnl_pct >= 0 else ""
                        emoji = "✅" if pnl_usd >= 0 else "❌"
                        text += f"\n{emoji} <b>{short_sym}</b> {pnl_sign}{pnl_pct:.1f}% (<code>{pnl_sign}${pnl_usd:.2f}</code>)"
            else:
                text = "📜 <b>거래 이력</b>\n\n최근 7일간 거래 이력이 없습니다"

            await self._safe_edit_message(query, text.strip(), self._get_back_keyboard())
        except Exception as e:
            await self._safe_edit_message(query, f"❌ 조회 실패: {e}", self._get_back_keyboard())

    async def _show_market_analysis(self, query):
        """AI 시황 분석"""
        await self._safe_edit_message(query, "🤖 AI 시황 분석 중...")

        if not self.get_market_report_callback:
            await self._safe_edit_message(query, "❌ 시황 분석 기능 사용 불가", self._get_back_keyboard())
            return

        try:
            report = await self.get_market_report_callback()
            await self._safe_edit_message(query, report, self._get_back_keyboard())
        except Exception as e:
            await self._safe_edit_message(query, f"❌ 분석 실패: {e}", self._get_back_keyboard())

    async def _show_why_no_entry(self, query):
        """미진입 이유"""
        await self._safe_edit_message(query, "🔍 분석 중...")

        if not self.get_no_entry_report_callback:
            await self._safe_edit_message(query, "❌ 분석 기능 사용 불가", self._get_back_keyboard())
            return

        try:
            report = await self.get_no_entry_report_callback()
            # 텔레그램 메시지 길이 제한 (4096자)
            if len(report) > 4000:
                report = report[:4000] + "\n\n... (생략)"
            await self._safe_edit_message(query, report, self._get_back_keyboard())
        except Exception as e:
            await self._safe_edit_message(query, f"❌ 분석 실패: {e}", self._get_back_keyboard())

    async def _show_watch_list(self, query):
        """관심 코인 목록"""
        await self._safe_edit_message(query, "🔭 분석 중...")

        if not self.get_watch_report_callback:
            await self._safe_edit_message(query, "❌ 분석 기능 사용 불가", self._get_back_keyboard())
            return

        try:
            report = await self.get_watch_report_callback()
            await self._safe_edit_message(query, report, self._get_back_keyboard())
        except Exception as e:
            await self._safe_edit_message(query, f"❌ 분석 실패: {e}", self._get_back_keyboard())

    async def _show_chart(self, query, symbol: str):
        """개별 차트 표시"""
        await self._safe_edit_message(query, f"📈 {symbol} 차트 생성 중...")

        if not self.get_chart_callback:
            await self._safe_edit_message(query, "❌ 차트 기능 사용 불가", self._get_back_keyboard())
            return

        try:
            chart_bytes = await self.get_chart_callback(symbol)
            if chart_bytes:
                # 차트는 새 메시지로 전송 (이미지는 edit 불가)
                await self.notifier.send_photo(
                    chart_bytes,
                    caption=f"📈 {symbol}/USDT Vertex 차트",
                    reply_markup=self._get_back_keyboard()
                )
                # 원래 메시지는 메뉴로 복귀
                await self._safe_edit_message(
                    query,
                    "📈 <b>차트</b>\n\n차트가 전송되었습니다. 다른 코인을 선택하세요.",
                    self._get_chart_keyboard()
                )
            else:
                await self._safe_edit_message(query, f"❌ {symbol} 차트 생성 실패", self._get_back_keyboard())
        except Exception as e:
            await self._safe_edit_message(query, f"❌ 차트 생성 실패: {e}", self._get_back_keyboard())

    async def _show_overview_chart(self, query):
        """전체 차트 표시"""
        await self._safe_edit_message(query, "📊 전체 차트 생성 중...")

        if not self.get_overview_chart_callback:
            await self._safe_edit_message(query, "❌ 차트 기능 사용 불가", self._get_back_keyboard())
            return

        try:
            chart_bytes = await self.get_overview_chart_callback()
            if chart_bytes:
                await self.notifier.send_photo(
                    chart_bytes,
                    caption="📊 주요 코인 Vertex 차트",
                    reply_markup=self._get_back_keyboard()
                )
                await self._safe_edit_message(
                    query,
                    "📈 <b>차트</b>\n\n전체 차트가 전송되었습니다.",
                    self._get_chart_keyboard()
                )
            else:
                await self._safe_edit_message(query, "❌ 전체 차트 생성 실패", self._get_back_keyboard())
        except Exception as e:
            await self._safe_edit_message(query, f"❌ 차트 생성 실패: {e}", self._get_back_keyboard())

    async def _sync_positions(self, query):
        """포지션 동기화 (바이빗 실제 거래 기록과 동기화)"""
        await self._safe_edit_message(query, "🔄 바이빗과 동기화 중...")

        if not self.sync_positions_callback:
            await self._safe_edit_message(
                query,
                "❌ 동기화 기능 사용 불가",
                self._get_control_keyboard()
            )
            return

        try:
            result = self.sync_positions_callback()
            synced = result.get("synced", 0)
            positions = result.get("positions", 0)

            if synced > 0:
                text = f"""
✅ <b>동기화 완료</b>

🔄 청산 감지: <code>{synced}건</code>
📋 현재 포지션: <code>{positions}개</code>

바이빗 청산 기록이 거래 이력에 반영되었습니다.
"""
            else:
                text = f"""
✅ <b>동기화 완료</b>

📋 현재 포지션: <code>{positions}개</code>

새로 감지된 청산이 없습니다.
"""
            await self._safe_edit_message(query, text.strip(), self._get_control_keyboard())

        except Exception as e:
            await self._safe_edit_message(
                query,
                f"❌ 동기화 실패: {e}",
                self._get_control_keyboard()
            )

    async def _show_funding_rates(self, query):
        """펀딩비 조회"""
        await self._safe_edit_message(query, "💸 펀딩비 조회 중...")

        if not self.get_funding_rates_callback:
            await self._safe_edit_message(query, "❌ 펀딩비 조회 불가", self._get_trading_keyboard())
            return

        try:
            funding_data = self.get_funding_rates_callback()

            if not funding_data:
                await self._safe_edit_message(query, "❌ 펀딩비 데이터 없음", self._get_trading_keyboard())
                return

            text = "💸 <b>펀딩비 현황</b>\n"
            text += "━━━━━━━━━━━━━━━━\n"

            # 펀딩비 기준으로 정렬 (절대값이 큰 순)
            sorted_data = sorted(
                funding_data.items(),
                key=lambda x: abs(x[1].get('funding_rate', 0)),
                reverse=True
            )

            for symbol, data in sorted_data:
                short_sym = symbol.split('/')[0]
                rate = data.get('funding_rate_pct', 0)

                # 펀딩비 방향 표시
                if rate > 0.01:
                    emoji = "🔴"  # 롱이 숏에게 지불
                    direction = "롱→숏"
                elif rate < -0.01:
                    emoji = "🟢"  # 숏이 롱에게 지불
                    direction = "숏→롱"
                else:
                    emoji = "⚪"
                    direction = "중립"

                text += f"\n{emoji} <b>{short_sym}</b>: <code>{rate:+.4f}%</code> ({direction})"

            text += "\n\n━━━━━━━━━━━━━━━━"
            text += "\n<i>🔴 양수: 롱 보유 시 수수료 지불</i>"
            text += "\n<i>🟢 음수: 롱 보유 시 수수료 수령</i>"
            text += "\n<i>펀딩은 8시간마다 정산</i>"

            await self._safe_edit_message(query, text, self._get_trading_keyboard())

        except Exception as e:
            await self._safe_edit_message(query, f"❌ 펀딩비 조회 실패: {e}", self._get_trading_keyboard())

    async def _show_sl_tp_info(self, query):
        """SL/TP 정보 표시"""
        await self._safe_edit_message(query, "🎯 SL/TP 조회 중...")

        if not self.get_positions_callback or not self.get_position_sl_tp_callback:
            await self._safe_edit_message(query, "❌ SL/TP 조회 불가", self._get_trading_keyboard())
            return

        try:
            positions = self.get_positions_callback()

            if not positions:
                text = "🎯 <b>SL/TP 설정</b>\n\n현재 보유 포지션이 없습니다"
                await self._safe_edit_message(query, text, self._get_trading_keyboard())
                return

            text = "🎯 <b>바이빗 SL/TP 설정 현황</b>\n"
            text += "━━━━━━━━━━━━━━━━\n"

            for pos in positions:
                symbol = pos.get('symbol', '')
                short_sym = symbol.split('/')[0]

                # 바이빗에서 실제 SL/TP 조회
                sl_tp = self.get_position_sl_tp_callback(symbol)

                side = pos.get('side', 'long')
                entry = float(pos.get('entry_price', 0))
                current = float(pos.get('current_price', 0))
                sl = float(sl_tp.get('stop_loss', 0))
                tp = float(sl_tp.get('take_profit', 0))

                emoji = "📈" if side == "long" else "📉"
                text += f"\n{emoji} <b>{short_sym}</b> {side.upper()}"
                text += f"\n├ 진입: <code>{fmt_price(entry)}</code>"
                if current > 0:
                    text += f" → 현재: <code>{fmt_price(current)}</code>"

                if sl > 0:
                    sl_dist = abs(sl - entry) / entry * 100
                    text += f"\n├ 🛑 손절: <code>{fmt_price(sl)}</code> ({sl_dist:.2f}%)"
                else:
                    text += f"\n├ 🛑 손절: <code>미설정</code>"

                if tp > 0:
                    tp_dist = abs(tp - entry) / entry * 100
                    text += f"\n└ 🎯 익절: <code>{fmt_price(tp)}</code> ({tp_dist:.2f}%)"
                else:
                    text += f"\n└ 🎯 익절: <code>미설정</code>"

                text += "\n"

            await self._safe_edit_message(query, text.strip(), self._get_sl_tp_keyboard(positions))

        except Exception as e:
            await self._safe_edit_message(query, f"❌ SL/TP 조회 실패: {e}", self._get_trading_keyboard())

    async def _edit_sl_tp(self, query, symbol: str):
        """SL/TP 수정 안내"""
        # 현재는 수정 기능 안내만 제공 (실제 수정은 채팅으로)
        short_sym = symbol.split('/')[0]
        text = f"""
✏️ <b>{short_sym} SL/TP 수정</b>

아래 형식으로 채팅을 보내주세요:

<code>/sltp {short_sym} SL=가격 TP=가격</code>

예시:
<code>/sltp {short_sym} SL=95000 TP=105000</code>
<code>/sltp {short_sym} SL=95000</code> (SL만 수정)
<code>/sltp {short_sym} TP=0</code> (TP 취소)
"""
        await self._safe_edit_message(query, text.strip(), self._get_trading_keyboard())

    def _calc_strategy_stats(self, trades: list) -> dict:
        """거래 목록에서 통계 계산"""
        if not trades:
            return {'total_pnl': 0, 'total_trades': 0, 'win_count': 0,
                    'loss_count': 0, 'win_rate': 0, 'avg_win': 0,
                    'avg_loss': 0, 'max_win': 0, 'max_loss': 0, 'profit_factor': 0}

        total_pnl = sum(t.get('pnl_usd', 0) for t in trades)
        wins = [t for t in trades if t.get('pnl_usd', 0) > 0]
        losses = [t for t in trades if t.get('pnl_usd', 0) < 0]
        win_count = len(wins)
        loss_count = len(losses)
        total_trades = len(trades)
        win_rate = (win_count / total_trades * 100) if total_trades else 0

        win_amts = [t.get('pnl_usd', 0) for t in wins]
        loss_amts = [t.get('pnl_usd', 0) for t in losses]

        avg_win = sum(win_amts) / len(win_amts) if win_amts else 0
        avg_loss = sum(loss_amts) / len(loss_amts) if loss_amts else 0
        max_win = max(win_amts) if win_amts else 0
        max_loss = min(loss_amts) if loss_amts else 0

        total_loss_abs = abs(sum(loss_amts)) if loss_amts else 0
        profit_factor = sum(win_amts) / total_loss_abs if total_loss_abs > 0 else float('inf')

        return {
            'total_pnl': total_pnl, 'total_trades': total_trades,
            'win_count': win_count, 'loss_count': loss_count,
            'win_rate': win_rate, 'avg_win': avg_win, 'avg_loss': avg_loss,
            'max_win': max_win, 'max_loss': max_loss, 'profit_factor': profit_factor,
        }

    def _format_stats_block(self, stats: dict, label: str = "") -> str:
        """통계 블록 포맷"""
        total_pnl = stats['total_pnl']
        pnl_emoji = "📈" if total_pnl >= 0 else "📉"
        pnl_sign = "+" if total_pnl >= 0 else ""

        text = ""
        if label:
            text += f"\n{label}\n"

        text += f"{pnl_emoji} 손익: <code>{pnl_sign}${total_pnl:,.2f}</code>\n"
        text += f"📋 {stats['total_trades']}건 ({stats['win_count']}승 {stats['loss_count']}패)"
        text += f" 승률 <code>{stats['win_rate']:.0f}%</code>\n"

        if stats['total_trades'] > 0:
            text += f"💰 평균수익: <code>+${stats['avg_win']:,.2f}</code>"
            text += f" | 평균손실: <code>${stats['avg_loss']:,.2f}</code>\n"
            text += f"🏆 최대수익: <code>+${stats['max_win']:,.2f}</code>"
            text += f" | 최대손실: <code>${stats['max_loss']:,.2f}</code>\n"

            pf = stats['profit_factor']
            if pf != float('inf'):
                text += f"📐 PF: <code>{pf:.2f}</code>\n"

        return text

    async def _show_account_stats(self, query, days: int):
        """계정 통계 표시 (전략별)"""
        await self._safe_edit_message(query, f"📊 {days}일 통계 조회 중...")

        # 봇 메모리 거래이력에서 전략별 통계 계산
        history = []
        if self.get_trade_history_exchange_callback:
            try:
                history = self.get_trade_history_exchange_callback(days)
            except:
                pass
        if not history and self.get_trade_history_callback:
            try:
                history = self.get_trade_history_callback()
            except:
                pass

        # 바이빗 API 통계도 조회 (전체 기준)
        api_stats = None
        if self.get_account_stats_callback:
            try:
                api_stats = self.get_account_stats_callback(days)
            except:
                pass

        try:
            text = f"📊 <b>최근 {days}일 거래 통계</b>\n"
            text += "━━━━━━━━━━━━━━━━\n"

            if history:
                # 전체 통계
                all_stats = self._calc_strategy_stats(history)
                text += self._format_stats_block(all_stats, "📋 <b>전체</b>")

                # 전략별 분리
                ich_trades = [h for h in history if h.get('strategy') in ('fractals', 'ichimoku')]
                surge_trades = [h for h in history if h.get('strategy') in ('mirror_short', 'surge')]
                ma100_trades = [h for h in history if h.get('strategy') == 'ma100']

                if ich_trades:
                    text += "━━━━━━━━━━━━━━━━\n"
                    ich_stats = self._calc_strategy_stats(ich_trades)
                    text += self._format_stats_block(ich_stats, "🔷 <b>Vertex</b>")

                if surge_trades:
                    text += "━━━━━━━━━━━━━━━━\n"
                    surge_stats = self._calc_strategy_stats(surge_trades)
                    text += self._format_stats_block(surge_stats, "📉 <b>미러숏</b>")

                if ma100_trades:
                    text += "━━━━━━━━━━━━━━━━\n"
                    ma100_stats = self._calc_strategy_stats(ma100_trades)
                    text += self._format_stats_block(ma100_stats, "📊 <b>MA100</b>")

            elif api_stats:
                # 봇 이력이 없으면 바이빗 API 통계만 표시
                text += self._format_stats_block(api_stats, "📋 <b>전체 (바이빗)</b>")
            else:
                text += "\n거래 이력이 없습니다"

            await self._safe_edit_message(query, text.strip(), self._get_trading_keyboard())

        except Exception as e:
            await self._safe_edit_message(query, f"❌ 통계 조회 실패: {e}", self._get_trading_keyboard())

    async def _show_fees_info(self, query):
        """펀딩비/수수료 내역 표시"""
        await self._safe_edit_message(query, "💰 펀딩/수수료 조회 중...")

        if not self.get_transaction_log_callback:
            await self._safe_edit_message(query, "❌ 조회 불가", self._get_trading_keyboard())
            return

        try:
            data = self.get_transaction_log_callback(7)  # 최근 7일

            total_funding = data.get('total_funding', 0)
            total_fee = data.get('total_trading_fee', 0)
            funding_fees = data.get('funding_fees', [])
            trading_fees = data.get('trading_fees', [])
            funding_count = data.get('funding_count', 0)
            trade_count = data.get('trade_count', 0)

            funding_emoji = "🟢" if total_funding >= 0 else "🔴"
            funding_sign = "+" if total_funding >= 0 else ""

            text = "💰 <b>최근 7일 펀딩/수수료</b>\n"
            text += "━━━━━━━━━━━━━━━━\n\n"

            # 요약
            text += f"{funding_emoji} <b>펀딩비 합계</b>: <code>{funding_sign}${total_funding:,.2f}</code>\n"
            text += f"   ({funding_count}건)\n\n"
            text += f"💸 <b>거래 수수료</b>: <code>-${abs(total_fee):,.2f}</code>\n"
            text += f"   ({trade_count}건)\n\n"

            total_cost = total_funding - abs(total_fee)
            cost_emoji = "📈" if total_cost >= 0 else "📉"
            cost_sign = "+" if total_cost >= 0 else ""
            text += f"{cost_emoji} <b>총 비용</b>: <code>{cost_sign}${total_cost:,.2f}</code>\n"
            text += "━━━━━━━━━━━━━━━━\n"

            # 펀딩비 내역 (최근 5건)
            if funding_fees:
                text += "\n<b>📋 펀딩비 내역</b>\n"
                for f in funding_fees[:5]:
                    sym = f.get('symbol', '')
                    amt = f.get('amount', 0)
                    created = f.get('created_at')

                    amt_sign = "+" if amt >= 0 else ""
                    time_str = ""
                    if created and hasattr(created, 'strftime'):
                        time_str = created.strftime("%m/%d %H:%M")

                    emoji = "🟢" if amt >= 0 else "🔴"
                    text += f"{emoji} {sym}: <code>{amt_sign}${amt:.4f}</code>"
                    if time_str:
                        text += f" ({time_str})"
                    text += "\n"

            text += "\n<i>💡 음수: 지불, 양수: 수령</i>"

            await self._safe_edit_message(query, text.strip(), self._get_trading_keyboard())

        except Exception as e:
            await self._safe_edit_message(query, f"❌ 조회 실패: {e}", self._get_trading_keyboard())

    async def _show_dca_status(self, query):
        """DCA 적립 현황 표시"""
        if not self.get_dca_summary_callback:
            await self._safe_edit_message(query, "❌ DCA 기능 사용 불가", self._get_back_keyboard())
            return

        try:
            # 상세 현황 우선, 없으면 요약
            if self.get_dca_detail_callback:
                detail = self.get_dca_detail_callback()
            else:
                detail = self.get_dca_summary_callback()

            dca_running = False
            if self.get_strategy_status_callback:
                st = self.get_strategy_status_callback()
                dca_running = st.get('dca_running', False)

            status_emoji = "🟢 실행중" if dca_running else "🔴 중지됨"
            text = f"🛒 <b>DCA 적립 현황</b>\n\n상태: {status_emoji}\n\n{detail}"

            keyboard = [
                [InlineKeyboardButton("⚙️ DCA 설정 변경", callback_data="dca_settings")],
                [InlineKeyboardButton("🔄 새로고침", callback_data="dca_status"),
                 InlineKeyboardButton("◀️ 메인", callback_data="back_main")],
            ]
            await self._safe_edit_message(query, text, InlineKeyboardMarkup(keyboard))
        except Exception as e:
            await self._safe_edit_message(query, f"❌ DCA 조회 실패: {e}", self._get_back_keyboard())

    async def _show_dca_settings(self, query):
        """DCA 설정 변경 메뉴"""
        keyboard = [
            [InlineKeyboardButton("💵 매수금액", callback_data="dca_set_amount"),
             InlineKeyboardButton("⏱ 인터벌", callback_data="dca_set_interval")],
            [InlineKeyboardButton("📊 BTC/ETH 비율", callback_data="dca_set_ratio"),
             InlineKeyboardButton("🏦 마진유보", callback_data="dca_set_reserve")],
            [InlineKeyboardButton("🎁 주간보너스%", callback_data="dca_set_bonus")],
            [InlineKeyboardButton("◀️ DCA 현황", callback_data="dca_status")],
        ]

        # 현재 설정 표시
        text = "⚙️ <b>DCA 설정 변경</b>\n\n변경할 항목을 선택하세요."
        if self.get_dca_params_callback:
            p = self.get_dca_params_callback()
            text = (
                f"⚙️ <b>DCA 설정 변경</b>\n\n"
                f"💵 매수금액: <b>${p.get('base_amount_usdt', 10):.0f}</b>/회\n"
                f"⏱ 인터벌: <b>{p.get('interval_hours', 8)}시간</b>\n"
                f"📊 비율: BTC <b>{p.get('btc_ratio', 0.4)*100:.0f}%</b> / ETH <b>{p.get('eth_ratio', 0.6)*100:.0f}%</b>\n"
                f"🏦 마진유보: <b>${p.get('min_futures_reserve', 500):,.0f}</b>\n"
                f"🎁 주간보너스: 일요일 00시 선물수익의 <b>{p.get('weekly_bonus_pct', 0.1)*100:.0f}%</b>\n\n"
                f"변경할 항목을 선택하세요."
            )

        await self._safe_edit_message(query, text, InlineKeyboardMarkup(keyboard))

    async def _show_dca_param_options(self, query, param_key: str, label: str, options: list, fmt: str = "{}"):
        """DCA 파라미터 선택 옵션 표시"""
        buttons = []
        row = []
        for val in options:
            display = fmt.format(val)
            row.append(InlineKeyboardButton(display, callback_data=f"dca_val_{param_key}_{val}"))
            if len(row) >= 3:
                buttons.append(row)
                row = []
        if row:
            buttons.append(row)
        buttons.append([InlineKeyboardButton("◀️ 설정으로", callback_data="dca_settings")])

        # 현재 값 표시
        current = ""
        if self.get_dca_params_callback:
            p = self.get_dca_params_callback()
            cur_val = p.get(param_key, "?")
            current = f"\n현재: <b>{fmt.format(cur_val)}</b>"

        text = f"⚙️ <b>{label} 변경</b>{current}\n\n원하는 값을 선택하세요."
        await self._safe_edit_message(query, text, InlineKeyboardMarkup(buttons))

    # ==================== 설정 변경 핸들러 ====================

    def _get_strategy_name(self, strategy: str) -> str:
        """전략 키 → 표시 이름"""
        names = {"ich": "Vertex", "surge": "미러숏", "ma100": "MA100"}
        return names.get(strategy, strategy)

    def _get_strategy_key(self, strategy: str) -> str:
        """전략 약어 → 콜백용 키"""
        keys = {"ich": "ichimoku", "surge": "surge", "ma100": "ma100"}
        return keys.get(strategy, strategy)

    def _get_strategy_settings_keyboard(self, strategy: str):
        """전략별 설정 키보드 반환"""
        if strategy == "ich":
            return self._get_settings_ich_keyboard()
        elif strategy == "surge":
            return self._get_settings_surge_keyboard()
        elif strategy == "ma100":
            return self._get_settings_ma100_keyboard()
        return self._get_settings_surge_keyboard()

    async def _handle_set_leverage(self, query, data: str):
        """레버리지 변경 처리"""
        # data: set_ich_lev_20 or set_surge_lev_5 or set_ma100_lev_5
        parts = data.split("_")
        strategy = parts[1]  # ich or surge or ma100
        value = int(parts[3])

        strategy_name = self._get_strategy_name(strategy)
        strategy_key = self._get_strategy_key(strategy)

        old_value = None
        if self.get_settings_callback:
            try:
                settings = self.get_settings_callback()
                old_value = settings.get(f'{strategy}_leverage')
            except:
                pass

        if self.set_leverage_callback:
            try:
                self.set_leverage_callback(strategy_key, value)
                old_str = f"{old_value}x → " if old_value is not None else ""
                text = f"✅ {strategy_name} 레버리지: {old_str}{value}x"
            except Exception as e:
                text = f"❌ 변경 실패: {e}"
        else:
            text = "❌ 설정 기능 사용 불가"

        keyboard = self._get_strategy_settings_keyboard(strategy)
        await self._safe_edit_message(query, text, keyboard)

    async def _handle_set_position_pct(self, query, data: str):
        """진입비율 변경 처리"""
        # data: set_ich_pct_5 or set_surge_pct_10 or set_ma100_pct_5
        parts = data.split("_")
        strategy = parts[1]  # ich or surge or ma100
        value = int(parts[3])

        strategy_name = self._get_strategy_name(strategy)
        strategy_key = self._get_strategy_key(strategy)

        old_value = None
        if self.get_settings_callback:
            try:
                settings = self.get_settings_callback()
                old_value = int(settings.get(f'{strategy}_pct', 0))
            except:
                pass

        if self.set_position_pct_callback:
            try:
                self.set_position_pct_callback(strategy_key, value / 100)
                old_str = f"{old_value}% → " if old_value is not None else ""
                text = f"✅ {strategy_name} 진입비율: {old_str}{value}%"
            except Exception as e:
                text = f"❌ 변경 실패: {e}"
        else:
            text = "❌ 설정 기능 사용 불가"

        keyboard = self._get_strategy_settings_keyboard(strategy)
        await self._safe_edit_message(query, text, keyboard)

    # ==================== 봇 시작/종료 ====================

    async def start_polling(self):
        """봇 폴링 시작"""
        if not self.notifier.token:
            logger.warning("텔레그램 토큰이 없어 봇을 시작하지 않습니다")
            return

        # 기존 폴링 세션 해제 (다른 인스턴스 충돌 방지)
        import httpx
        try:
            async with httpx.AsyncClient() as client:
                url = f"https://api.telegram.org/bot{self.notifier.token}/deleteWebhook?drop_pending_updates=true"
                await client.post(url)
                logger.info("기존 텔레그램 웹훅/폴링 세션 해제 완료")
        except Exception as e:
            logger.warning(f"텔레그램 세션 해제 실패 (무시): {e}")

        self.app = Application.builder().token(self.notifier.token).build()

        # 명령어 핸들러
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("menu", self.cmd_menu))

        # 콜백 쿼리 핸들러 (인라인 버튼)
        self.app.add_handler(CallbackQueryHandler(self.handle_callback))

        # 일반 텍스트 메시지 핸들러 (AI 챗봇)
        self.app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self.handle_message
        ))

        self.running = True
        logger.info("텔레그램 봇 폴링 시작")

        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)

    async def stop_polling(self):
        """봇 폴링 중지"""
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            logger.info("텔레그램 봇 폴링 중지")
