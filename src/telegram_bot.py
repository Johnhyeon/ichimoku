"""텔레그램 봇 - Trojan 스타일 인라인 버튼 UI"""

import logging
import asyncio
import io
from datetime import datetime
from typing import Optional, Callable, Dict
from telegram import Update, Bot, InputFile, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, ContextTypes, filters

from src.config import settings

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
가격: ${price:,.2f}
수량: {qty:.4f}
손절: ${sl:,.2f} (-{sl_pct:.1f}%)
익절: ${tp:,.2f} (+{tp_pct:.1f}%)
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
진입가: ${entry:,.2f}
청산가: ${exit_price:,.2f}
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

        # 거래 정보 콜백
        self.get_funding_rates_callback: Optional[Callable] = None
        self.get_position_sl_tp_callback: Optional[Callable] = None
        self.set_position_sl_tp_callback: Optional[Callable] = None
        self.get_account_stats_callback: Optional[Callable] = None
        self.get_trade_history_exchange_callback: Optional[Callable] = None
        self.get_transaction_log_callback: Optional[Callable] = None

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
        status_btn = "⏸ 중지" if self.running else "▶️ 시작"
        status_data = "bot_stop" if self.running else "bot_start"

        keyboard = [
            [
                InlineKeyboardButton(status_btn, callback_data=status_data),
                InlineKeyboardButton("🔄 동기화", callback_data="sync_positions"),
            ],
            [
                InlineKeyboardButton("← 뒤로", callback_data="back_main"),
            ],
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

    def _build_dashboard_text(self) -> str:
        """메인 대시보드 텍스트 생성"""
        now = datetime.utcnow().strftime('%H:%M:%S')
        status_emoji = "🟢" if self.running else "🔴"
        status_text = "실행중" if self.running else "중지됨"

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
                    positions_text = "\n\n📋 <b>포지션</b>"
                    for p in positions:
                        emoji = "📈" if p['side'] == 'long' else "📉"
                        short_sym = p['symbol'].split('/')[0]
                        pnl_usd = float(p.get('pnl', 0))
                        pnl_pct = float(p.get('pnl_pct', 0))  # 레버리지 적용
                        pnl_sign = "+" if pnl_pct >= 0 else ""
                        pnl_emoji = "🟢" if pnl_pct >= 0 else "🔴"
                        leverage = int(p.get('leverage', 20))
                        positions_text += f"\n{emoji} <b>{short_sym}</b> {p['side'].upper()}"
                        positions_text += f" {pnl_emoji} <code>{pnl_sign}{pnl_pct:.1f}%</code> (x{leverage})"
                else:
                    positions_text = "\n\n📋 <b>포지션</b>\n없음"
            except:
                positions_text = "\n\n📋 포지션: 조회 실패"

        text = f"""
🤖 <b>Ichimoku Trading Bot</b>

{status_emoji} 상태: <b>{status_text}</b>
🕐 갱신: {now} UTC
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

        # 포지션 상세
        if data == "positions":
            await self._show_positions(query)
            return

        # 거래 이력
        if data == "history":
            await self._show_trade_history(query)
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

        # 봇 제어 메뉴
        if data == "menu_control":
            status = "🟢 실행중" if self.running else "🔴 중지됨"
            text = f"⚙️ <b>봇 제어</b>\n\n현재 상태: {status}"
            await self._safe_edit_message(query, text, self._get_control_keyboard())
            return

        # 봇 시작/중지
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

            await self._safe_edit_message(query, text.strip(), self._get_back_keyboard())
        except Exception as e:
            await self._safe_edit_message(query, f"❌ 조회 실패: {e}", self._get_back_keyboard())

    async def _show_positions(self, query):
        """포지션 상세 표시"""
        if not self.get_positions_callback:
            await self._safe_edit_message(query, "❌ 포지션 조회 불가", self._get_back_keyboard())
            return

        try:
            positions = self.get_positions_callback()
            if positions:
                text = "📋 <b>포지션 상세</b>\n"
                for p in positions:
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
                    text += f"\n┌ 진입: <code>${entry:,.2f}</code>"
                    if current > 0:
                        text += f" → 현재: <code>${current:,.2f}</code>"
                    text += f"\n├ 가격변동: <code>{price_sign}{price_change:.2f}%</code>"
                    text += f"\n├ {pnl_emoji} 수익률: <code>{pnl_sign}{pnl_pct:.1f}%</code> ({pnl_sign}${pnl_usd:.2f})"
                    text += f"\n├ 수량: <code>{size:.4f}</code>"
                    if sl > 0:
                        sl_emoji = "🟡" if sl_dist > 0 else "🔴"
                        text += f"\n├ {sl_emoji} 손절: <code>${sl:,.2f}</code> ({sl_dist:+.1f}%)"
                    if tp > 0:
                        tp_emoji = "🟡" if tp_dist > 0 else "🟢"
                        text += f"\n└ {tp_emoji} 익절: <code>${tp:,.2f}</code> ({tp_dist:+.1f}%)"
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

                for h in history[:10]:  # 최근 10건만
                    symbol = h.get('symbol', '')
                    short_sym = symbol.split('/')[0] if '/' in symbol else symbol
                    pnl_pct = float(h.get('pnl_pct', 0))
                    pnl_usd = float(h.get('pnl_usd', 0))
                    reason = h.get('reason', '')
                    closed_at = h.get('closed_at')
                    leverage = h.get('leverage', 20)

                    emoji = "✅" if pnl_usd >= 0 else "❌"
                    pnl_sign = "+" if pnl_pct >= 0 else ""
                    side = h.get('side', 'long')
                    side_emoji = "📈" if side == 'long' else "📉"

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
                        text += f"\n   ${entry:,.0f} → ${exit_p:,.0f}"
                    text += f"\n   {pnl_sign}{pnl_pct:.1f}% (<code>{pnl_sign}${pnl_usd:.2f}</code>)"
                    if reason:
                        text += f" | {reason}"
                    if time_str:
                        text += f"\n   <code>{time_str}</code>"
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
                    caption=f"📈 {symbol}/USDT 일목균형표 차트",
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
                    caption="📊 주요 코인 일목균형표 차트",
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
                text += f"\n├ 진입: <code>${entry:,.2f}</code>"
                if current > 0:
                    text += f" → 현재: <code>${current:,.2f}</code>"

                if sl > 0:
                    sl_dist = abs(sl - entry) / entry * 100
                    text += f"\n├ 🛑 손절: <code>${sl:,.2f}</code> ({sl_dist:.2f}%)"
                else:
                    text += f"\n├ 🛑 손절: <code>미설정</code>"

                if tp > 0:
                    tp_dist = abs(tp - entry) / entry * 100
                    text += f"\n└ 🎯 익절: <code>${tp:,.2f}</code> ({tp_dist:.2f}%)"
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

    async def _show_account_stats(self, query, days: int):
        """계정 통계 표시"""
        await self._safe_edit_message(query, f"📊 {days}일 통계 조회 중...")

        if not self.get_account_stats_callback:
            await self._safe_edit_message(query, "❌ 통계 조회 불가", self._get_trading_keyboard())
            return

        try:
            stats = self.get_account_stats_callback(days)

            total_pnl = stats.get('total_pnl', 0)
            total_trades = stats.get('total_trades', 0)
            win_count = stats.get('win_count', 0)
            loss_count = stats.get('loss_count', 0)
            win_rate = stats.get('win_rate', 0)
            avg_win = stats.get('avg_win', 0)
            avg_loss = stats.get('avg_loss', 0)
            max_win = stats.get('max_win', 0)
            max_loss = stats.get('max_loss', 0)
            profit_factor = stats.get('profit_factor', 0)

            pnl_emoji = "📈" if total_pnl >= 0 else "📉"
            pnl_sign = "+" if total_pnl >= 0 else ""

            text = f"📊 <b>최근 {days}일 거래 통계</b>\n"
            text += "━━━━━━━━━━━━━━━━\n\n"

            text += f"{pnl_emoji} <b>총 손익</b>: <code>{pnl_sign}${total_pnl:,.2f}</code>\n\n"

            text += f"📋 총 거래: <code>{total_trades}건</code>\n"
            text += f"✅ 승리: <code>{win_count}건</code>\n"
            text += f"❌ 패배: <code>{loss_count}건</code>\n"
            text += f"🎯 승률: <code>{win_rate:.1f}%</code>\n\n"

            if total_trades > 0:
                text += f"💰 평균 수익: <code>+${avg_win:,.2f}</code>\n"
                text += f"💸 평균 손실: <code>${avg_loss:,.2f}</code>\n"
                text += f"🏆 최대 수익: <code>+${max_win:,.2f}</code>\n"
                text += f"😢 최대 손실: <code>${max_loss:,.2f}</code>\n\n"

                if profit_factor != float('inf'):
                    text += f"📐 Profit Factor: <code>{profit_factor:.2f}</code>\n"
                    text += "<i>(1 이상이면 수익, 2 이상이면 우수)</i>"

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

    # ==================== 봇 시작/종료 ====================

    async def start_polling(self):
        """봇 폴링 시작"""
        if not self.notifier.token:
            logger.warning("텔레그램 토큰이 없어 봇을 시작하지 않습니다")
            return

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
        await self.app.updater.start_polling()

    async def stop_polling(self):
        """봇 폴링 중지"""
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            logger.info("텔레그램 봇 폴링 중지")
