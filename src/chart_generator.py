"""
차트 생성 모듈 - 일목균형표 차트
"""

import logging
import io
from typing import Optional
from datetime import datetime
import pandas as pd
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # 헤드리스 모드
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from datetime import timedelta

from src.ichimoku import calculate_ichimoku

logger = logging.getLogger(__name__)


class ChartGenerator:
    """일목균형표 차트 생성기"""

    def __init__(self):
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib가 설치되지 않음")

        # 한글 폰트 설정 시도
        try:
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass

    def generate_ichimoku_chart(
        self,
        df: pd.DataFrame,
        symbol: str,
        position: Optional[dict] = None
    ) -> Optional[bytes]:
        """
        일목균형표 차트 생성

        Args:
            df: OHLCV DataFrame (ichimoku 지표 포함)
            symbol: 심볼명
            position: 현재 포지션 정보 (있을 경우 표시)

        Returns:
            PNG 이미지 바이트
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        try:
            # 데이터 준비
            df = df.copy()
            if 'timestamp' in df.columns:
                df = df.reset_index()

            # 일목균형표 계산 (없으면)
            if 'tenkan' not in df.columns:
                df = calculate_ichimoku(df)

            # 최근 100개 캔들만
            df = df.tail(100).reset_index(drop=True)

            if len(df) < 20:
                return None

            # Figure 생성
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(14, 10),
                gridspec_kw={'height_ratios': [3, 1]},
                facecolor='#1a1a2e'
            )

            # 배경색 설정
            ax1.set_facecolor('#16213e')
            ax2.set_facecolor('#16213e')

            x = range(len(df))

            # === 캔들스틱 차트 ===
            for i in range(len(df)):
                row = df.iloc[i]
                open_price = row['open']
                close_price = row['close']
                high_price = row['high']
                low_price = row['low']

                # 색상 결정
                if close_price >= open_price:
                    color = '#00ff88'  # 상승 (녹색)
                else:
                    color = '#ff4757'  # 하락 (적색)

                # 심지
                ax1.plot([i, i], [low_price, high_price], color=color, linewidth=0.8)

                # 몸통
                body_bottom = min(open_price, close_price)
                body_height = abs(close_price - open_price)
                rect = Rectangle(
                    (i - 0.35, body_bottom),
                    0.7, body_height,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0.5
                )
                ax1.add_patch(rect)

            # === 일목균형표 라인 ===
            # 전환선 (Tenkan-sen)
            ax1.plot(x, df['tenkan'], color='#00d4ff', linewidth=1.2, label='Tenkan (9)')

            # 기준선 (Kijun-sen)
            ax1.plot(x, df['kijun'], color='#ff6b6b', linewidth=1.2, label='Kijun (26)')

            # 선행스팬 A, B (구름)
            senkou_a = df['senkou_a'].values
            senkou_b = df['senkou_b'].values

            # 구름 채우기
            for i in range(len(df) - 1):
                if pd.notna(senkou_a[i]) and pd.notna(senkou_b[i]):
                    if senkou_a[i] >= senkou_b[i]:
                        color = '#00ff8833'  # 녹색 구름
                    else:
                        color = '#ff475733'  # 적색 구름

                    ax1.fill_between(
                        [i, i + 1],
                        [senkou_a[i], senkou_a[i + 1]] if i + 1 < len(senkou_a) else [senkou_a[i], senkou_a[i]],
                        [senkou_b[i], senkou_b[i + 1]] if i + 1 < len(senkou_b) else [senkou_b[i], senkou_b[i]],
                        color=color
                    )

            # 선행스팬 라인
            ax1.plot(x, senkou_a, color='#00ff88', linewidth=0.8, alpha=0.7, linestyle='--')
            ax1.plot(x, senkou_b, color='#ff4757', linewidth=0.8, alpha=0.7, linestyle='--')

            # === 포지션 표시 ===
            if position:
                entry_price = position.get('entry_price', 0)
                stop_loss = position.get('stop_loss', 0)
                take_profit = position.get('take_profit', 0)
                side = position.get('side', '')

                # 진입가
                ax1.axhline(y=entry_price, color='#ffd93d', linestyle='-', linewidth=1.5, alpha=0.8)
                ax1.annotate(
                    f'Entry ${entry_price:,.0f}',
                    xy=(len(df) - 1, entry_price),
                    xytext=(len(df) + 2, entry_price),
                    color='#ffd93d',
                    fontsize=9,
                    fontweight='bold'
                )

                # 손절가
                ax1.axhline(y=stop_loss, color='#ff4757', linestyle='--', linewidth=1.2, alpha=0.8)
                ax1.annotate(
                    f'SL ${stop_loss:,.0f}',
                    xy=(len(df) - 1, stop_loss),
                    xytext=(len(df) + 2, stop_loss),
                    color='#ff4757',
                    fontsize=9
                )

                # 익절가
                ax1.axhline(y=take_profit, color='#00ff88', linestyle='--', linewidth=1.2, alpha=0.8)
                ax1.annotate(
                    f'TP ${take_profit:,.0f}',
                    xy=(len(df) - 1, take_profit),
                    xytext=(len(df) + 2, take_profit),
                    color='#00ff88',
                    fontsize=9
                )

            # 현재가 표시
            current_price = df.iloc[-1]['close']
            ax1.annotate(
                f'${current_price:,.2f}',
                xy=(len(df) - 1, current_price),
                xytext=(len(df) + 1, current_price),
                color='white',
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#333', alpha=0.8)
            )

            # === 거래량 차트 ===
            colors = ['#00ff88' if df.iloc[i]['close'] >= df.iloc[i]['open'] else '#ff4757'
                      for i in range(len(df))]
            ax2.bar(x, df['volume'], color=colors, alpha=0.7, width=0.8)

            # 거래량 SMA
            if 'volume_sma' in df.columns:
                ax2.plot(x, df['volume_sma'], color='#ffd93d', linewidth=1, label='Vol SMA(20)')

            # === 스타일링 ===
            short_symbol = symbol.split('/')[0]
            ax1.set_title(
                f'{short_symbol} - Ichimoku Cloud (4H)',
                color='white',
                fontsize=14,
                fontweight='bold',
                pad=15
            )

            # Y축 가격 포맷
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))

            # 격자
            ax1.grid(True, alpha=0.2, color='gray')
            ax2.grid(True, alpha=0.2, color='gray')

            # 축 색상
            for ax in [ax1, ax2]:
                ax.tick_params(colors='white')
                ax.spines['bottom'].set_color('gray')
                ax.spines['top'].set_color('gray')
                ax.spines['left'].set_color('gray')
                ax.spines['right'].set_color('gray')

            # 범례
            ax1.legend(loc='upper left', facecolor='#333', edgecolor='gray', labelcolor='white')

            # X축 라벨 숨기기 (상단 차트)
            ax1.set_xticklabels([])

            # 타임스탬프 표시 (하단 차트)
            if 'timestamp' in df.columns:
                # 10개 간격으로 라벨 표시
                tick_positions = range(0, len(df), 10)
                tick_labels = [df.iloc[i]['timestamp'].strftime('%m/%d %H:%M')
                               if i < len(df) else '' for i in tick_positions]
                ax2.set_xticks(list(tick_positions))
                ax2.set_xticklabels(tick_labels, rotation=45, ha='right', color='white', fontsize=8)

            # 여백 조정
            plt.tight_layout()

            # 이미지로 저장
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, facecolor='#1a1a2e', edgecolor='none')
            buf.seek(0)
            plt.close(fig)

            return buf.getvalue()

        except Exception as e:
            logger.error(f"차트 생성 실패: {e}")
            plt.close('all')
            return None

    def generate_balance_chart(self, history: list) -> Optional[bytes]:
        """
        잔고 추이 차트 생성

        Args:
            history: [{timestamp, equity, balance, unrealized_pnl}, ...]

        Returns:
            PNG 이미지 바이트
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        if not history or len(history) < 2:
            return None

        try:
            # 데이터 준비
            timestamps = []
            equities = []
            unrealized_pnls = []

            for h in history:
                ts = h.get("timestamp", "")
                try:
                    dt = datetime.fromisoformat(ts)
                except (ValueError, TypeError):
                    continue
                timestamps.append(dt)
                equities.append(float(h.get("equity", 0)))
                unrealized_pnls.append(float(h.get("unrealized_pnl", 0)))

            if len(timestamps) < 2:
                return None

            # Figure 생성
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(14, 8),
                gridspec_kw={'height_ratios': [3, 1]},
                facecolor='#1a1a2e'
            )

            ax1.set_facecolor('#16213e')
            ax2.set_facecolor('#16213e')

            # === 상단: Equity 라인 차트 ===
            ax1.plot(timestamps, equities, color='#00d4ff', linewidth=2, label='Equity')

            # 영역 그라데이션
            ax1.fill_between(
                timestamps, equities,
                min(equities) * 0.998,
                alpha=0.3,
                color='#00d4ff'
            )

            # 시작/종료 값
            start_equity = equities[0]
            end_equity = equities[-1]
            change_amt = end_equity - start_equity
            change_pct = (change_amt / start_equity * 100) if start_equity != 0 else 0

            change_color = '#00ff88' if change_amt >= 0 else '#ff4757'
            change_sign = '+' if change_amt >= 0 else ''

            # 시작점/끝점 마커
            ax1.plot(timestamps[0], start_equity, 'o', color='#ffd93d', markersize=8, zorder=5)
            ax1.plot(timestamps[-1], end_equity, 'o', color=change_color, markersize=8, zorder=5)

            # 현재 평가자산 라벨
            ax1.annotate(
                f'${end_equity:,.2f}',
                xy=(timestamps[-1], end_equity),
                xytext=(15, 10),
                textcoords='offset points',
                color='white',
                fontsize=11,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#333', alpha=0.9)
            )

            # 변화율/변화액 표시
            ax1.set_title(
                f'Balance Trend  |  {change_sign}{change_pct:.2f}% ({change_sign}${change_amt:,.2f})',
                color=change_color,
                fontsize=14,
                fontweight='bold',
                pad=15
            )

            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            ax1.grid(True, alpha=0.2, color='gray')
            ax1.legend(loc='upper left', facecolor='#333', edgecolor='gray', labelcolor='white')

            # === 하단: 미실현 손익 바 차트 ===
            bar_colors = ['#00ff88' if pnl >= 0 else '#ff4757' for pnl in unrealized_pnls]

            # 바 너비 계산 (데이터 간격 기반)
            if len(timestamps) > 1:
                avg_delta = (timestamps[-1] - timestamps[0]) / len(timestamps)
                bar_width = avg_delta * 0.8
            else:
                bar_width = timedelta(minutes=4)

            ax2.bar(timestamps, unrealized_pnls, color=bar_colors, alpha=0.8, width=bar_width)
            ax2.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)

            ax2.set_ylabel('Unrealized PnL', color='white', fontsize=10)
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            ax2.grid(True, alpha=0.2, color='gray')

            # X축 날짜 포맷
            for ax in [ax1, ax2]:
                ax.tick_params(colors='white')
                ax.spines['bottom'].set_color('gray')
                ax.spines['top'].set_color('gray')
                ax.spines['left'].set_color('gray')
                ax.spines['right'].set_color('gray')

            ax1.set_xticklabels([])

            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            fig.autofmt_xdate(rotation=30, ha='right')

            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, facecolor='#1a1a2e', edgecolor='none')
            buf.seek(0)
            plt.close(fig)

            return buf.getvalue()

        except Exception as e:
            logger.error(f"잔고 차트 생성 실패: {e}")
            plt.close('all')
            return None

    def generate_multi_chart(
        self,
        coin_data: dict,
        symbols: list = None,
        max_charts: int = 4
    ) -> Optional[bytes]:
        """
        여러 코인 미니 차트 생성

        Args:
            coin_data: {symbol: DataFrame} 딕셔너리
            symbols: 표시할 심볼 목록 (None이면 상위 4개)
            max_charts: 최대 차트 수

        Returns:
            PNG 이미지 바이트
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        if symbols is None:
            symbols = list(coin_data.keys())[:max_charts]

        symbols = symbols[:max_charts]

        if not symbols:
            return None

        try:
            # 그리드 크기 결정
            n = len(symbols)
            if n <= 2:
                rows, cols = 1, n
            else:
                rows, cols = 2, 2

            fig, axes = plt.subplots(rows, cols, figsize=(12, 8), facecolor='#1a1a2e')

            if n == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

            for idx, symbol in enumerate(symbols):
                if idx >= len(axes):
                    break

                ax = axes[idx]
                ax.set_facecolor('#16213e')

                df = coin_data.get(symbol)
                if df is None or df.empty:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', color='white')
                    continue

                df = df.copy()
                if 'timestamp' in df.columns:
                    df = df.reset_index()
                if 'tenkan' not in df.columns:
                    df = calculate_ichimoku(df)

                df = df.tail(50).reset_index(drop=True)
                x = range(len(df))

                # 캔들 (간소화)
                for i in range(len(df)):
                    row = df.iloc[i]
                    color = '#00ff88' if row['close'] >= row['open'] else '#ff4757'
                    ax.plot([i, i], [row['low'], row['high']], color=color, linewidth=0.5)

                # 종가 라인
                ax.plot(x, df['close'], color='white', linewidth=1)

                # 구름
                ax.fill_between(x, df['senkou_a'], df['senkou_b'],
                                where=df['senkou_a'] >= df['senkou_b'],
                                color='#00ff8833', alpha=0.5)
                ax.fill_between(x, df['senkou_a'], df['senkou_b'],
                                where=df['senkou_a'] < df['senkou_b'],
                                color='#ff475733', alpha=0.5)

                # 제목
                short_symbol = symbol.split('/')[0]
                current_price = df.iloc[-1]['close']

                # 변화율 계산
                if len(df) > 1:
                    prev_price = df.iloc[0]['close']
                    change_pct = (current_price - prev_price) / prev_price * 100
                    change_str = f'{change_pct:+.1f}%'
                    change_color = '#00ff88' if change_pct >= 0 else '#ff4757'
                else:
                    change_str = ''
                    change_color = 'white'

                ax.set_title(
                    f'{short_symbol} ${current_price:,.0f} {change_str}',
                    color=change_color,
                    fontsize=11,
                    fontweight='bold'
                )

                ax.grid(True, alpha=0.2, color='gray')
                ax.tick_params(colors='white', labelsize=8)
                ax.set_xticklabels([])

            # 빈 축 숨기기
            for idx in range(len(symbols), len(axes)):
                axes[idx].set_visible(False)

            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, facecolor='#1a1a2e')
            buf.seek(0)
            plt.close(fig)

            return buf.getvalue()

        except Exception as e:
            logger.error(f"멀티 차트 생성 실패: {e}")
            plt.close('all')
            return None
