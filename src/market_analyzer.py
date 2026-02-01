"""
시장 분석 모듈 - Gemini AI 연동
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from src.config import settings
from src.ichimoku import calculate_ichimoku
from src.strategy import STRATEGY_PARAMS, MAJOR_COINS, get_entry_signal

logger = logging.getLogger(__name__)


class MarketAnalyzer:
    """시장 분석기"""

    def __init__(self, data_fetcher, timeframe: str = "4h"):
        self.data_fetcher = data_fetcher
        self.timeframe = timeframe
        self.gemini_model = None

        # Gemini 초기화
        if GEMINI_AVAILABLE and settings.GEMINI_API_KEY:
            try:
                genai.configure(api_key=settings.GEMINI_API_KEY)
                # 사용 가능한 모델: gemini-2.0-flash, gemini-1.5-pro, gemini-pro
                self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
                logger.info("Gemini AI 초기화 완료")
            except Exception as e:
                logger.warning(f"Gemini 초기화 실패: {e}")
        else:
            logger.warning("Gemini API 키가 없거나 패키지가 설치되지 않음")

    def _get_coin_status(self, symbol: str, df: pd.DataFrame) -> Dict:
        """개별 코인 상태 분석"""
        if df is None or df.empty:
            return None

        row = df.iloc[-1]
        prev_row = df.iloc[-2] if len(df) > 1 else row

        price = float(row['close'])
        cloud_top = float(row['cloud_top'])
        cloud_bottom = float(row['cloud_bottom'])

        # 위치 판단
        if price > cloud_top:
            position = "구름 위"
            position_emoji = "🟢"
        elif price < cloud_bottom:
            position = "구름 아래"
            position_emoji = "🔴"
        else:
            position = "구름 안"
            position_emoji = "🟡"

        # 추세 판단
        tenkan = float(row['tenkan'])
        kijun = float(row['kijun'])
        if tenkan > kijun:
            trend = "상승"
            trend_emoji = "📈"
        else:
            trend = "하락"
            trend_emoji = "📉"

        # 신호 체크
        signals = []
        if bool(row.get('tk_cross_up', False)):
            signals.append("전환선↑")
        if bool(row.get('tk_cross_down', False)):
            signals.append("전환선↓")
        if bool(row.get('kijun_cross_up', False)):
            signals.append("기준선 돌파↑")
        if bool(row.get('kijun_cross_down', False)):
            signals.append("기준선 돌파↓")

        # 구름 색상
        cloud_color = "녹색(상승)" if bool(row.get('cloud_green', False)) else "적색(하락)"

        # 24시간 변화율 (6개 캔들)
        if len(df) >= 6:
            price_24h_ago = float(df.iloc[-6]['close'])
            change_24h = (price - price_24h_ago) / price_24h_ago * 100
        else:
            change_24h = 0

        return {
            'symbol': symbol,
            'price': price,
            'position': position,
            'position_emoji': position_emoji,
            'trend': trend,
            'trend_emoji': trend_emoji,
            'cloud_top': cloud_top,
            'cloud_bottom': cloud_bottom,
            'cloud_thickness': float(row['cloud_thickness']),
            'cloud_color': cloud_color,
            'tenkan': tenkan,
            'kijun': kijun,
            'signals': signals,
            'change_24h': change_24h,
            'volume_ratio': float(row.get('volume_ratio', 1.0)),
            'chikou_bullish': bool(row.get('chikou_bullish', False)),
            'chikou_bearish': bool(row.get('chikou_bearish', False)),
        }

    def _check_entry_reasons(
        self,
        symbol: str,
        row: pd.Series,
        btc_uptrend: Optional[bool],
        last_exit_time: Optional[datetime]
    ) -> List[str]:
        """진입 불가 사유 분석"""
        reasons = []
        params = STRATEGY_PARAMS

        price = float(row['close'])
        thickness = float(row['cloud_thickness'])

        # 구름 안 체크
        if bool(row.get('in_cloud', False)):
            reasons.append("구름 안에 있음 (횡보장)")

        # 구름 두께 체크
        if thickness < params['min_cloud_thickness']:
            reasons.append(f"구름 두께 부족 ({thickness:.2f}% < {params['min_cloud_thickness']}%)")

        # 쿨다운 체크
        if last_exit_time:
            hours_since = (datetime.utcnow() - last_exit_time).total_seconds() / 3600
            if hours_since < params['cooldown_hours']:
                remaining = params['cooldown_hours'] - hours_since
                reasons.append(f"쿨다운 중 ({remaining:.1f}시간 남음)")

        # 롱 조건 체크
        if bool(row.get('above_cloud', False)):
            if not bool(row.get('tenkan_above', False)):
                reasons.append("전환선이 기준선 아래")

            has_signal = bool(row.get('tk_cross_up', False)) or bool(row.get('kijun_cross_up', False))
            if not has_signal:
                reasons.append("진입 신호 없음 (크로스 대기)")

            if params.get('use_btc_filter', True) and btc_uptrend is True:
                reasons.append("BTC 상승 추세 (롱 필터)")

            if params.get('long_chikou_required', True):
                if not bool(row.get('chikou_bullish', False)):
                    reasons.append("후행스팬 약세")

            volume_ratio = float(row.get('volume_ratio', 0))
            min_vol = params.get('long_volume_min_ratio', 1.2)
            if volume_ratio < min_vol:
                reasons.append(f"거래량 부족 ({volume_ratio:.2f}x < {min_vol}x)")

        # 숏 조건 체크
        elif bool(row.get('below_cloud', False)):
            if bool(row.get('tenkan_above', False)):
                reasons.append("전환선이 기준선 위")

            has_signal = bool(row.get('tk_cross_down', False)) or bool(row.get('kijun_cross_down', False))
            if not has_signal:
                reasons.append("진입 신호 없음 (크로스 대기)")

            if params.get('use_btc_filter', True) and btc_uptrend is False:
                reasons.append("BTC 하락 추세 (숏 필터)")

        if not reasons:
            reasons.append("조건 충족 - 손익비 계산 중")

        return reasons

    def _find_watch_candidates(
        self,
        coin_data: Dict[str, pd.DataFrame],
        btc_uptrend: Optional[bool]
    ) -> List[Dict]:
        """진입 예상 코인 탐색"""
        candidates = []

        for symbol, df in coin_data.items():
            if df is None or df.empty:
                continue

            row = df.iloc[-1]
            score = 0
            reasons = []

            # 구름 위 + 상승 추세 = 롱 후보
            if bool(row.get('above_cloud', False)) and bool(row.get('tenkan_above', False)):
                score += 3
                reasons.append("구름 위 + 상승 추세")

                if bool(row.get('chikou_bullish', False)):
                    score += 2
                    reasons.append("후행스팬 강세")

                # 크로스 임박 체크
                tenkan = float(row['tenkan'])
                kijun = float(row['kijun'])
                diff_pct = abs(tenkan - kijun) / kijun * 100
                if diff_pct < 0.5:
                    score += 2
                    reasons.append("크로스 임박")

                if float(row.get('volume_ratio', 0)) > 1.0:
                    score += 1
                    reasons.append("거래량 증가")

                candidates.append({
                    'symbol': symbol,
                    'direction': 'LONG',
                    'score': score,
                    'price': float(row['close']),
                    'reasons': reasons
                })

            # 구름 아래 + 하락 추세 = 숏 후보
            elif bool(row.get('below_cloud', False)) and not bool(row.get('tenkan_above', False)):
                score += 3
                reasons.append("구름 아래 + 하락 추세")

                if bool(row.get('chikou_bearish', False)):
                    score += 2
                    reasons.append("후행스팬 약세")

                tenkan = float(row['tenkan'])
                kijun = float(row['kijun'])
                diff_pct = abs(tenkan - kijun) / kijun * 100
                if diff_pct < 0.5:
                    score += 2
                    reasons.append("크로스 임박")

                candidates.append({
                    'symbol': symbol,
                    'direction': 'SHORT',
                    'score': score,
                    'price': float(row['close']),
                    'reasons': reasons
                })

        # 점수순 정렬
        candidates.sort(key=lambda x: -x['score'])
        return candidates[:5]  # 상위 5개

    async def generate_market_report(
        self,
        coin_data: Dict[str, pd.DataFrame],
        btc_uptrend: Optional[bool],
        positions: Dict = None
    ) -> str:
        """시장 리포트 생성"""
        statuses = []
        for symbol, df in coin_data.items():
            if df is not None and not df.empty:
                df = calculate_ichimoku(df.reset_index() if 'timestamp' in df.columns else df)
                status = self._get_coin_status(symbol, df)
                if status:
                    statuses.append(status)

        if not statuses:
            return "데이터를 가져올 수 없습니다."

        # 기본 통계
        above_cloud = sum(1 for s in statuses if s['position'] == '구름 위')
        below_cloud = sum(1 for s in statuses if s['position'] == '구름 아래')
        in_cloud = sum(1 for s in statuses if s['position'] == '구름 안')

        btc_status = next((s for s in statuses if 'BTC' in s['symbol']), None)
        eth_status = next((s for s in statuses if 'ETH' in s['symbol']), None)

        # 템플릿 기반 리포트
        report = f"""📊 <b>시황 분석</b> ({datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC)

<b>🌐 전체 시장</b>
• 구름 위: {above_cloud}개 | 구름 안: {in_cloud}개 | 구름 아래: {below_cloud}개
• BTC 추세: {'상승 📈' if btc_uptrend else '하락 📉' if btc_uptrend is False else '중립 ➖'}

<b>💰 주요 코인</b>"""

        if btc_status:
            report += f"""
• BTC: ${btc_status['price']:,.0f} ({btc_status['change_24h']:+.1f}%) {btc_status['position_emoji']} {btc_status['position']}"""

        if eth_status:
            report += f"""
• ETH: ${eth_status['price']:,.0f} ({eth_status['change_24h']:+.1f}%) {eth_status['position_emoji']} {eth_status['position']}"""

        # 신호 발생 코인
        signal_coins = [s for s in statuses if s['signals']]
        if signal_coins:
            report += "\n\n<b>⚡ 신호 발생</b>"
            for s in signal_coins[:5]:
                short_symbol = s['symbol'].split('/')[0]
                report += f"\n• {short_symbol}: {', '.join(s['signals'])}"

        # Gemini AI 분석 추가
        if self.gemini_model:
            try:
                ai_analysis = await self._get_gemini_analysis(statuses, btc_uptrend)
                if ai_analysis:
                    report += f"\n\n<b>🤖 AI 분석</b>\n{ai_analysis}"
            except Exception as e:
                logger.warning(f"Gemini 분석 실패: {e}")

        return report

    async def _get_gemini_analysis(
        self,
        statuses: List[Dict],
        btc_uptrend: Optional[bool]
    ) -> str:
        """Gemini AI로 시장 분석"""
        if not self.gemini_model:
            return ""

        # 요약 데이터 준비
        summary = []
        for s in statuses[:10]:  # 상위 10개만
            short_symbol = s['symbol'].split('/')[0]
            summary.append(
                f"{short_symbol}: {s['position']}, {s['trend']}, "
                f"24h {s['change_24h']:+.1f}%, 구름두께 {s['cloud_thickness']:.1f}%"
            )

        prompt = f"""당신은 암호화폐 기술적 분석 전문가입니다.
일목균형표 데이터를 바탕으로 현재 시장 상황을 2-3문장으로 간결하게 분석해주세요.

BTC 추세: {'상승' if btc_uptrend else '하락' if btc_uptrend is False else '중립'}

코인별 상태:
{chr(10).join(summary)}

분석 시 주의사항:
- 한국어로 답변
- 2-3문장으로 핵심만 간결하게
- 구체적인 투자 조언은 하지 않음
- 현재 시장 분위기와 주의할 점만 언급"""

        try:
            response = await self.gemini_model.generate_content_async(prompt)
            return response.text.strip()
        except Exception as e:
            logger.warning(f"Gemini API 호출 실패: {e}")
            return ""

    async def generate_no_entry_report(
        self,
        coin_data: Dict[str, pd.DataFrame],
        btc_uptrend: Optional[bool],
        last_exit_times: Dict[str, datetime] = None
    ) -> str:
        """진입 없는 이유 리포트"""
        last_exit_times = last_exit_times or {}

        report = """🔍 <b>진입 없는 이유</b>

"""
        reasons_count = {}

        for symbol, df in coin_data.items():
            if df is None or df.empty:
                continue

            df = calculate_ichimoku(df.reset_index() if 'timestamp' in df.columns else df)
            row = df.iloc[-1]

            reasons = self._check_entry_reasons(
                symbol, row, btc_uptrend,
                last_exit_times.get(symbol)
            )

            short_symbol = symbol.split('/')[0]
            report += f"<b>{short_symbol}</b>\n"
            for r in reasons:
                report += f"  • {r}\n"
                reasons_count[r] = reasons_count.get(r, 0) + 1
            report += "\n"

        # 가장 흔한 이유
        if reasons_count:
            top_reason = max(reasons_count, key=reasons_count.get)
            report += f"📌 <b>주요 원인:</b> {top_reason} ({reasons_count[top_reason]}개 코인)"

        return report

    async def analyze_entry_reason(
        self,
        symbol: str,
        df: pd.DataFrame,
        side: str,
        btc_uptrend: Optional[bool]
    ) -> str:
        """진입 이유 AI 분석"""
        if not self.gemini_model or df is None or df.empty:
            return ""

        row = df.iloc[-1]
        short_symbol = symbol.split('/')[0]

        # 지표 데이터 추출
        data = {
            'price': float(row['close']),
            'tenkan': float(row['tenkan']),
            'kijun': float(row['kijun']),
            'cloud_top': float(row['cloud_top']),
            'cloud_bottom': float(row['cloud_bottom']),
            'cloud_thickness': float(row['cloud_thickness']),
            'cloud_green': bool(row.get('cloud_green', False)),
            'above_cloud': bool(row.get('above_cloud', False)),
            'below_cloud': bool(row.get('below_cloud', False)),
            'tk_cross_up': bool(row.get('tk_cross_up', False)),
            'tk_cross_down': bool(row.get('tk_cross_down', False)),
            'kijun_cross_up': bool(row.get('kijun_cross_up', False)),
            'kijun_cross_down': bool(row.get('kijun_cross_down', False)),
            'chikou_bullish': bool(row.get('chikou_bullish', False)),
            'chikou_bearish': bool(row.get('chikou_bearish', False)),
            'volume_ratio': float(row.get('volume_ratio', 1.0)),
        }

        prompt = f"""당신은 일목균형표 전문 트레이더입니다.
아래 지표 데이터를 바탕으로 왜 {side.upper()} 진입 신호가 발생했는지 간결하게 설명해주세요.

코인: {short_symbol}
포지션: {side.upper()}
BTC 추세: {'상승' if btc_uptrend else '하락' if btc_uptrend is False else '중립'}

【현재 지표】
• 가격: ${data['price']:,.2f}
• 전환선(9): {data['tenkan']:.2f}
• 기준선(26): {data['kijun']:.2f}
• 구름 상단: {data['cloud_top']:.2f}
• 구름 하단: {data['cloud_bottom']:.2f}
• 구름 두께: {data['cloud_thickness']:.2f}%
• 구름 색상: {'녹색(상승)' if data['cloud_green'] else '적색(하락)'}

【신호】
• 가격 위치: {'구름 위' if data['above_cloud'] else '구름 아래' if data['below_cloud'] else '구름 안'}
• 전환선/기준선 크로스: {'상향 돌파 ✓' if data['tk_cross_up'] else '하향 돌파 ✓' if data['tk_cross_down'] else '없음'}
• 기준선 돌파: {'상향 ✓' if data['kijun_cross_up'] else '하향 ✓' if data['kijun_cross_down'] else '없음'}
• 후행스팬: {'강세 ✓' if data['chikou_bullish'] else '약세 ✓' if data['chikou_bearish'] else '중립'}
• 거래량: {data['volume_ratio']:.1f}x (평균 대비)

요구사항:
- 한국어로 3-4문장으로 핵심만 설명
- 어떤 지표가 근거가 되었는지 구체적으로 언급
- 이모지 사용 가능
- 투자 권유가 아닌 기술적 분석임을 명시하지 않아도 됨"""

        try:
            response = await self.gemini_model.generate_content_async(prompt)
            return response.text.strip()
        except Exception as e:
            logger.warning(f"진입 분석 실패: {e}")
            return ""

    async def analyze_exit_reason(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        reason: str,
        pnl_pct: float
    ) -> str:
        """청산 이유 AI 분석"""
        if not self.gemini_model:
            return ""

        short_symbol = symbol.split('/')[0]

        reason_map = {
            'Stop': '손절가 도달',
            'TP': '익절가 도달',
            'Trail': '트레일링 스탑',
            'CloudExit': '구름 이탈',
            'MaxLoss': '최대 손실 한도',
        }
        reason_kr = reason_map.get(reason, reason)

        prompt = f"""당신은 일목균형표 전문 트레이더입니다.
아래 청산 내역을 바탕으로 왜 청산되었는지 간결하게 설명해주세요.

코인: {short_symbol}
포지션: {side.upper()}
진입가: ${entry_price:,.2f}
청산가: ${exit_price:,.2f}
수익률: {pnl_pct:+.1f}%
청산 사유: {reason_kr}

요구사항:
- 한국어로 2-3문장으로 핵심만 설명
- 청산 이유와 결과를 분석
- 이모지 사용 가능"""

        try:
            response = await self.gemini_model.generate_content_async(prompt)
            return response.text.strip()
        except Exception as e:
            logger.warning(f"청산 분석 실패: {e}")
            return ""

    async def chat_response(
        self,
        user_message: str,
        coin_data: Dict[str, pd.DataFrame],
        btc_uptrend: Optional[bool],
        positions: List[Dict],
        balance: Dict
    ) -> str:
        """사용자 채팅에 AI 응답 생성"""
        if not self.gemini_model:
            return "AI 기능을 사용할 수 없습니다. Gemini API 키를 확인해주세요."

        # 현재 시장 상태 요약
        market_summary = []
        for symbol, df in list(coin_data.items())[:10]:
            if df is None or df.empty:
                continue
            row = df.iloc[-1]
            short_sym = symbol.split('/')[0]
            price = float(row['close'])

            if bool(row.get('above_cloud', False)):
                position = "구름 위"
            elif bool(row.get('below_cloud', False)):
                position = "구름 아래"
            else:
                position = "구름 안"

            trend = "상승" if bool(row.get('tenkan_above', False)) else "하락"
            market_summary.append(f"{short_sym}: ${price:,.2f}, {position}, {trend}추세")

        # 포지션 정보
        positions_text = "없음"
        if positions:
            pos_list = []
            for p in positions:
                short_sym = p['symbol'].split('/')[0]
                pnl = float(p.get('pnl', 0))
                pos_list.append(f"{short_sym} {p['side'].upper()} (PnL: ${pnl:+.2f})")
            positions_text = ", ".join(pos_list)

        # 잔고 정보
        balance_text = "조회 불가"
        if isinstance(balance, dict):
            total = balance.get('total', 0)
            equity = balance.get('equity', total)
            balance_text = f"잔고: ${total:,.2f}, 평가자산: ${equity:,.2f}"

        # 전략 파라미터
        from src.strategy import STRATEGY_PARAMS, LEVERAGE, POSITION_PCT
        params = STRATEGY_PARAMS

        system_prompt = f"""너는 "이치봇" - 일목균형표 자동매매 봇이야.

【말투】
- 반말로 친근하게 (ㅋㅋ, ㅎㅎ 적절히 사용)
- 핵심만 짧게 2-3문장
- 이모지 자연스럽게 사용
- 트레이더 친구처럼 편하게

【전략 설정】
- 레버리지: {LEVERAGE}배
- 포지션 크기: 자산의 {POSITION_PCT*100}%
- 최소 구름 두께: {params['min_cloud_thickness']}%
- 손절 범위: {params['min_sl_pct']}% ~ {params['max_sl_pct']}%
- 손익비: 1:{params['rr_ratio']}
- 트레일링: {params['trail_pct']}%
- 쿨다운: {params['cooldown_hours']}시간
- 최대 포지션: {params['max_positions']}개
- BTC 필터: {'ON (역추세)' if params['use_btc_filter'] else 'OFF'}

【진입 로직】
- 롱: 구름 위 + 전환선>기준선 + (TK크로스 or 기준선돌파) + BTC하락시만
- 숏: 구름 아래 + 전환선<기준선 + (TK크로스 or 기준선돌파) + BTC상승시만
- 손절: 구름 경계 + 버퍼 {params['sl_buffer']}%

【청산 로직】
- 손절/익절 도달
- 트레일링 스탑 (TP 도달 후 활성화)
- 구름 진입 시 청산
- 최대손실 -4% (강제청산)

【현재 상태】
- BTC: {'상승 📈' if btc_uptrend else '하락 📉' if btc_uptrend is False else '중립'}
- {balance_text}
- 포지션: {positions_text}

【코인 현황】
{chr(10).join(market_summary)}

【규칙】
- 핵심만! 장황하게 X
- 모르면 모른다고
- 매수/매도 추천 X (봇이 알아서 함)
- 전략 질문엔 위 설정 기반으로 답변"""

        try:
            chat = self.gemini_model.start_chat(history=[])
            response = await chat.send_message_async(
                f"{system_prompt}\n\n사용자 질문: {user_message}"
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"채팅 응답 생성 실패: {e}")
            return f"죄송합니다, 응답 생성 중 오류가 발생했습니다: {e}"

    async def generate_watch_report(
        self,
        coin_data: Dict[str, pd.DataFrame],
        btc_uptrend: Optional[bool]
    ) -> str:
        """진입 예상 코인 리포트"""
        # 데이터 전처리
        processed_data = {}
        for symbol, df in coin_data.items():
            if df is not None and not df.empty:
                df = calculate_ichimoku(df.reset_index() if 'timestamp' in df.columns else df)
                processed_data[symbol] = df

        candidates = self._find_watch_candidates(processed_data, btc_uptrend)

        if not candidates:
            return "🔭 현재 진입 예상 코인이 없습니다."

        report = """🔭 <b>진입 예상 코인</b>

"""
        for i, c in enumerate(candidates, 1):
            short_symbol = c['symbol'].split('/')[0]
            direction_emoji = "🟢" if c['direction'] == 'LONG' else "🔴"
            report += f"{i}. {direction_emoji} <b>{short_symbol}</b> ({c['direction']})\n"
            report += f"   가격: ${c['price']:,.2f} | 점수: {c['score']}/8\n"
            report += f"   사유: {', '.join(c['reasons'])}\n\n"

        report += "⚠️ 크로스 발생 시 진입 신호가 생성됩니다."

        return report
