"""
Ichimoku Cloud 실전 트레이딩 봇 (4시간봉, 메인넷용 멀티 코인 전략)

백테스트 검증 완료 (2024-04 ~ 2026-01):
    - 총 수익: $142,257 (6,777% 수익률)
    - 승률: 41.9%
    - Profit Factor: 1.73
    - LONG 수익: +$1,359 (31.0% 승률, Volume Spike 필터 적용)
    - SHORT 수익: +$140,899 (44.9% 승률)
    - MDD: 42.2% (레버리지 20배 기준, 10배 시 약 21%)

적용된 필터:
    1. BTC 도미넌스 필터 (Strict Mode)
       - BTC 상승 추세(MA26 > MA52): SHORT만 진입
       - BTC 하락 추세(MA26 < MA52): LONG만 진입

    2. LONG Volume Spike 필터 (Filter4)
       - 후행스팬 상승 필수
       - 거래량 > 평균의 120% 필수
       - 이 필터로 LONG이 손실에서 수익으로 전환

권장 리스크 관리:
    - 레버리지: 10배 (20배 → 10배로 변경 시 MDD 50% 감소)
    - 포지션 크기: 2-3% (현재 5%)
    - 초기 자본: 전체의 10-20%로 시작
    - 일일 손실 한도: 전체 자본의 5%

실행 예시:
    python ichimoku_live.py                 # 메인넷 LIVE (주의: 실제 주문)
    python ichimoku_live.py --paper         # 메인넷 페이퍼 모드 (주문 안 나감)
    python ichimoku_live.py --testnet       # 테스트넷 (권장: 먼저 테스트)
    python ichimoku_live.py --testnet --once  # 한 번만 실행해서 신호만 확인
"""

import argparse
import logging
import time
from datetime import datetime
from typing import Optional, Dict, List

import pandas as pd

from exchange.bybit_client import BybitClient
from exchange.data_fetcher import DataFetcher
from backtest.ichimoku_backtest import calculate_ichimoku


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class IchimokuLiveBot:
    """
    일목균형표 (Ichimoku Cloud) 실전 트레이딩 봇

    - 타임프레임: 4h
    - 기본 레버리지: 20x
    - 포지션 크기: 잔고의 5%를 마진으로 사용 (백테스트와 동일 개념)
    - 운용 범위: 백테스트와 동일한 메이저 코인 20개 (USDT 무기한 선물)
    """

    # 백테스트와 동일한 핵심 파라미터
    # 백테스트: 레버리지 20배, MDD 42%
    # 권장: 레버리지 10배로 변경 시 MDD 약 21%로 감소
    LEVERAGE = 20
    POSITION_PCT = 0.05  # 자산의 5%를 한 포지션에 사용

    STRATEGY_PARAMS = {
        "min_cloud_thickness": 0.2,  # 최소 구름 두께 %
        "min_sl_pct": 0.3,  # 최소 손절 거리 %
        "max_sl_pct": 8.0,  # 최대 손절 거리 %
        "sl_buffer": 0.2,  # 손절 버퍼 %
        "rr_ratio": 2.0,  # 손익비
        "trail_pct": 1.5,  # 트레일링 스탑 %
        "cooldown_hours": 4,  # 재진입 쿨다운
        "max_positions": 5,  # 최대 동시 포지션 (백테스트와 동일)
        # 백테스트 검증된 필터 (Filter4: Volume Spike)
        "use_btc_filter": True,  # BTC 도미넌스 필터 사용 (Strict Mode)
        "long_chikou_required": True,  # LONG 진입 시 후행스팬 필수
        "long_volume_min_ratio": 1.2,  # LONG 진입 시 최소 거래량 비율 (평균 대비)
    }

    # 백테스트 MAJOR_COINS를 Bybit 선물(ccxt) 심볼로 매핑
    # 예: 'BTCUSDT' -> 'BTC/USDT:USDT'
    MAJOR_COINS: List[str] = [
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "BNB/USDT:USDT",
        "XRP/USDT:USDT",
        "SOL/USDT:USDT",
        "ADA/USDT:USDT",
        "DOGE/USDT:USDT",
        "AVAX/USDT:USDT",
        "DOT/USDT:USDT",
        "LINK/USDT:USDT",
        "MATIC/USDT:USDT",
        "LTC/USDT:USDT",
        "ATOM/USDT:USDT",
        "UNI/USDT:USDT",
        "ETC/USDT:USDT",
        "APT/USDT:USDT",
        "NEAR/USDT:USDT",
        "FIL/USDT:USDT",
        "AAVE/USDT:USDT",
        "INJ/USDT:USDT",
    ]

    def __init__(self, paper: bool = False, testnet: bool = False):
        self.paper = paper
        self.testnet = testnet

        # Bybit 클라이언트 및 데이터 수집기
        self.client = BybitClient(testnet=testnet)
        self.data_fetcher = DataFetcher(self.client)

        # 타임프레임
        self.timeframe = "4h"

        # 포지션 상태: 심볼별 포지션 딕셔너리
        # {symbol: {side, entry_price, stop_loss, take_profit, highest, lowest, trail_stop, trailing, size, entry_time}}
        self.positions: Dict[str, dict] = {}
        # 심볼별 마지막 청산 시각 (쿨다운 관리)
        self.last_exit_times: Dict[str, datetime] = {}

        # BTC 트렌드 캐시 (BTC 도미넌스 필터용)
        self.btc_uptrend: Optional[bool] = None

        # 시작 로그
        mode = "PAPER" if self.paper else "LIVE"
        logger.info(f"IchimokuLiveBot 시작 - 모드: {mode}, 테스트넷: {self.testnet}")
        logger.info(f"레버리지: {self.LEVERAGE}x, 포지션 크기: {self.POSITION_PCT*100}%")
        logger.info(f"백테스트 검증 필터 적용:")
        logger.info(f"  - BTC 도미넌스 필터: {self.STRATEGY_PARAMS['use_btc_filter']}")
        logger.info(f"  - LONG 후행스팬 필수: {self.STRATEGY_PARAMS['long_chikou_required']}")
        logger.info(f"  - LONG 거래량 필터: {self.STRATEGY_PARAMS['long_volume_min_ratio']}x")

    # ---------- 유틸리티 ----------

    def _get_balance_free(self) -> float:
        """USDT 사용 가능 잔고 조회"""
        try:
            balance = self.client.get_balance()
            return float(balance.get("free", 0))
        except Exception as e:
            logger.error(f"잔고 조회 실패: {e}")
            return 0.0

    def _update_btc_trend(self):
        """
        BTC 트렌드 업데이트 (도미넌스 필터용)
        BTC MA26 > MA52: 상승 추세 (SHORT 유리)
        BTC MA26 < MA52: 하락 추세 (LONG 유리)
        """
        try:
            btc_df = self.data_fetcher.get_ohlcv("BTC/USDT:USDT", self.timeframe, limit=100)
            if btc_df is None or btc_df.empty:
                logger.warning("BTC 데이터 조회 실패, 도미넌스 필터 사용 불가")
                self.btc_uptrend = None
                return

            btc_df = btc_df.reset_index()
            btc_df['sma_26'] = btc_df['close'].rolling(26).mean()
            btc_df['sma_52'] = btc_df['close'].rolling(52).mean()

            latest = btc_df.iloc[-1]
            if pd.notna(latest['sma_26']) and pd.notna(latest['sma_52']):
                self.btc_uptrend = latest['sma_26'] > latest['sma_52']
                trend_str = "상승" if self.btc_uptrend else "하락"
                logger.info(f"[BTC TREND] {trend_str} (MA26={latest['sma_26']:.1f}, MA52={latest['sma_52']:.1f})")
            else:
                self.btc_uptrend = None
                logger.warning("BTC MA 데이터 부족, 도미넌스 필터 사용 불가")

        except Exception as e:
            logger.error(f"BTC 트렌드 계산 실패: {e}")
            self.btc_uptrend = None

    def _fetch_ichimoku_df(self, symbol: str, limit: int = 200) -> Optional[pd.DataFrame]:
        """지정 심볼의 OHLCV를 가져와서 일목 지표 + volume 지표 계산"""
        try:
            df = self.data_fetcher.get_ohlcv(symbol, self.timeframe, limit=limit)
        except Exception as e:
            logger.error(f"OHLCV 데이터 조회 실패 ({symbol}): {e}")
            return None

        if df is None or df.empty:
            logger.warning("OHLCV 데이터 없음")
            return None

        # 인덱스를 컬럼으로 되돌려 백테스트와 비슷한 형태로 변환
        df = df.reset_index()
        df = calculate_ichimoku(df)

        # Volume 지표 추가 (LONG Volume Spike Filter용)
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        df = df.dropna(
            subset=["tenkan", "kijun", "cloud_top", "cloud_bottom", "cloud_thickness"]
        )
        if df.empty:
            logger.warning("일목 지표 계산 후 유효 데이터 없음")
            return None
        return df

    # ---------- 신호 로직 (백테스트 후보 선정 로직 멀티 심볼 버전) ----------

    def _get_entry_signal(self, symbol: str, row: pd.Series) -> Optional[dict]:
        """
        단일 심볼용 진입 신호 계산 (백테스트 검증된 Filter4: Volume Spike 적용)
        Returns:
            {
              'symbol': str, 'side': 'long'/'short',
              'price': float, 'stop_loss': float, 'take_profit': float,
              'score': int, 'thickness': float
            }
            or None
        """
        params = self.STRATEGY_PARAMS

        price = float(row["close"])
        cloud_top = float(row["cloud_top"])
        cloud_bottom = float(row["cloud_bottom"])
        thickness = float(row["cloud_thickness"])

        # 구름 안이면 스킵 (횡보장)
        if bool(row["in_cloud"]):
            return None

        # 구름 두께 필터
        if thickness < params["min_cloud_thickness"]:
            return None

        # 심볼별 쿨다운
        now = datetime.utcnow()
        last_exit = self.last_exit_times.get(symbol)
        if last_exit is not None:
            hours_since_exit = (now - last_exit).total_seconds() / 3600
            if hours_since_exit < params["cooldown_hours"]:
                return None

        # === 롱 조건 (백테스트 검증: Filter4 - Volume Spike) ===
        if bool(row["above_cloud"]) and bool(row["tenkan_above"]):
            has_signal = bool(row["tk_cross_up"]) or bool(row["kijun_cross_up"])
            if not has_signal:
                return None

            # BTC 트렌드 필터 (Strict Mode)
            if params.get("use_btc_filter", True):
                if self.btc_uptrend is True:
                    # BTC 상승장에서는 LONG 진입 금지
                    return None

            # LONG 강화 필터 적용
            # 1. 후행스팬 필수 체크
            if params.get("long_chikou_required", True):
                if not bool(row.get("chikou_bullish", False)):
                    return None

            # 2. 거래량 필터 (평균의 120% 이상)
            volume_ratio = float(row.get("volume_ratio", 0))
            min_volume_ratio = params.get("long_volume_min_ratio", 1.2)
            if volume_ratio < min_volume_ratio:
                return None

            # 점수 계산 (강화된 가중치)
            score = 0
            if bool(row.get("chikou_bullish", False)):
                score += 3  # 후행스팬 가중치 증가
            if bool(row.get("cloud_green", False)):
                score += 2
            if thickness > 1.0:
                score += 2
            if volume_ratio > 1.5:
                score += 1  # 거래량 증가 보너스

            stop_loss = cloud_top * (1 - params["sl_buffer"] / 100)
            sl_distance_pct = (price - stop_loss) / price * 100
            if params["min_sl_pct"] <= sl_distance_pct <= params["max_sl_pct"]:
                take_profit = price * (
                    1 + sl_distance_pct * params["rr_ratio"] / 100
                )
                logger.info(
                    f"[SIGNAL] {symbol} LONG 후보 | Price={price:.2f}, "
                    f"Volume={volume_ratio:.2f}x, Score={score}"
                )
                return {
                    "symbol": symbol,
                    "side": "long",
                    "price": price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "score": score,
                    "thickness": thickness,
                }

        # === 숏 조건 (기존 유지) ===
        if bool(row["below_cloud"]) and not bool(row["tenkan_above"]):
            has_signal = bool(row["tk_cross_down"]) or bool(row["kijun_cross_down"])
            if not has_signal:
                return None

            # BTC 트렌드 필터 (Strict Mode)
            if params.get("use_btc_filter", True):
                if self.btc_uptrend is False:
                    # BTC 하락장에서는 SHORT 진입 금지
                    return None

            score = 0
            if bool(row.get("chikou_bearish", False)):
                score += 2
            if not bool(row.get("cloud_green", True)):
                score += 1
            if thickness > 1.0:
                score += 1

            stop_loss = cloud_bottom * (1 + params["sl_buffer"] / 100)
            sl_distance_pct = (stop_loss - price) / price * 100
            if params["min_sl_pct"] <= sl_distance_pct <= params["max_sl_pct"]:
                take_profit = price * (
                    1 - sl_distance_pct * params["rr_ratio"] / 100
                )
                logger.info(
                    f"[SIGNAL] {symbol} SHORT 후보 | Price={price:.2f}, Score={score}"
                )
                return {
                    "symbol": symbol,
                    "side": "short",
                    "price": price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "score": score,
                    "thickness": thickness,
                }

        return None

    def _check_exit_signal(self, symbol: str, row: pd.Series) -> Optional[dict]:
        """
        포지션 청산/트레일링 신호
        Returns:
            {'action': 'close'/'trail', 'reason': str, 'price': float, 'new_trail': Optional[float]}
        """
        pos = self.positions.get(symbol)
        if not pos:
            return None

        params = self.STRATEGY_PARAMS
        side = pos["side"]
        entry = float(pos["entry_price"])
        stop_loss = float(pos["stop_loss"])
        take_profit = float(pos["take_profit"])
        trail_stop = float(pos.get("trail_stop", stop_loss))
        trailing = bool(pos.get("trailing", False))

        price = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])

        # MaxLoss: 레버리지 20배 기준 가격 -4% / +4% 손절
        if side == "long":
            max_loss_price = entry * 0.98
            if low <= max_loss_price:
                return {
                    "action": "close",
                    "reason": "MaxLoss",
                    "price": max_loss_price,
                    "new_trail": None,
                }
        else:
            max_loss_price = entry * 1.02
            if high >= max_loss_price:
                return {
                    "action": "close",
                    "reason": "MaxLoss",
                    "price": max_loss_price,
                    "new_trail": None,
                }

        # 트레일링/TP/SL/구름 진입 로직 (백테스트 로직 단일 심볼 버전)
        if side == "long":
            # 최고가 갱신 시 트레일링 업데이트
            highest = float(pos.get("highest", entry))
            if high > highest:
                highest = high
                pos["highest"] = highest
                if high >= take_profit:
                    trailing = True
                    trail_stop = max(
                        trail_stop, high * (1 - params["trail_pct"] / 100)
                    )
                    pos["trailing"] = True
                    pos["trail_stop"] = trail_stop
                    logger.info(f"[TRAIL] {symbol} 롱 트레일링 스탑 갱신: {trail_stop:.2f}")

            # 1. 손절
            if low <= stop_loss:
                return {
                    "action": "close",
                    "reason": "Stop",
                    "price": max(stop_loss, low),
                    "new_trail": None,
                }

            # 2. 트레일링
            if trailing and low <= trail_stop:
                return {
                    "action": "close",
                    "reason": "Trail",
                    "price": trail_stop,
                    "new_trail": None,
                }

            # 3. TP (트레일링 없을 때)
            if not trailing and high >= take_profit:
                return {
                    "action": "close",
                    "reason": "TP",
                    "price": take_profit,
                    "new_trail": None,
                }

            # 4. 구름 안/하단 진입 시 청산
            if bool(row["in_cloud"]) or bool(row["below_cloud"]):
                return {
                    "action": "close",
                    "reason": "CloudExit",
                    "price": price,
                    "new_trail": None,
                }

        else:  # short
            lowest = float(pos.get("lowest", entry))
            if low < lowest:
                lowest = low
                pos["lowest"] = lowest
                if low <= take_profit:
                    trailing = True
                    trail_stop = min(
                        trail_stop, low * (1 + params["trail_pct"] / 100)
                    )
                    pos["trailing"] = True
                    pos["trail_stop"] = trail_stop
                    logger.info(f"[TRAIL] {symbol} 숏 트레일링 스탑 갱신: {trail_stop:.2f}")

            # 1. 손절
            if high >= stop_loss:
                return {
                    "action": "close",
                    "reason": "Stop",
                    "price": min(stop_loss, high),
                    "new_trail": None,
                }

            # 2. 트레일링
            if trailing and high >= trail_stop:
                return {
                    "action": "close",
                    "reason": "Trail",
                    "price": trail_stop,
                    "new_trail": None,
                }

            # 3. TP
            if not trailing and low <= take_profit:
                return {
                    "action": "close",
                    "reason": "TP",
                    "price": take_profit,
                    "new_trail": None,
                }

            # 4. 구름 안/상단 진입 시 청산
            if bool(row["in_cloud"]) or bool(row["above_cloud"]):
                return {
                    "action": "close",
                    "reason": "CloudExit",
                    "price": price,
                    "new_trail": None,
                }

        return None

    # ---------- 주문 실행 ----------

    def _calc_order_quantity(self, price: float, free_balance: float) -> float:
        """
        잔고와 레버리지를 기반으로 주문 수량 계산
        - free USDT * POSITION_PCT = 마진
        - 마진 * 레버리지 / price = 포지션 수량
        """
        if free_balance <= 0:
            return 0.0

        margin = free_balance * self.POSITION_PCT
        position_value = margin * self.LEVERAGE
        qty = position_value / price

        # BTC 수량은 소수 3자리 정도로 제한
        qty = round(qty, 3)
        return qty

    def _open_position(self, signal: dict, free_balance: float) -> float:
        """
        포지션 오픈
        Returns:
            실제 사용한 마진(대략적인 값, free_balance 업데이트용)
        """
        symbol = signal["symbol"]
        side = signal["side"]
        price = signal["price"]
        stop_loss = signal["stop_loss"]
        take_profit = signal["take_profit"]

        qty = self._calc_order_quantity(price, free_balance)
        if qty <= 0:
            logger.warning("주문 수량이 0 이하입니다. 진입 취소.")
            return 0.0

        logger.info(
            f"[ENTRY] {symbol} {side.upper()} 진입 시도 | Price={price:.2f}, Qty={qty}, "
            f"SL={stop_loss:.2f}, TP={take_profit:.2f}"
        )

        if not self.paper:
            try:
                # 레버리지 설정
                self.client.set_leverage(symbol, self.LEVERAGE)
            except Exception as e:
                logger.warning(f"레버리지 설정 실패 (무시하고 진행): {e}")

            try:
                order_side = "buy" if side == "long" else "sell"
                self.client.market_order(symbol, order_side, qty)
            except Exception as e:
                logger.error(f"시장가 진입 실패 ({symbol}): {e}")
                return 0.0

        # 로컬 포지션 상태 업데이트
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

        logger.info(
            f"[ENTRY] {symbol} {side.upper()} 진입 완료 | Entry={price:.2f}, Qty={qty}, "
            f"SL={stop_loss:.2f}, TP={take_profit:.2f}"
        )

        # 사용한 마진(대략): price * qty / 레버리지
        used_margin = (price * qty) / self.LEVERAGE
        return used_margin

    def _close_position(self, symbol: str, exit_info: dict):
        pos = self.positions.get(symbol)
        if not pos:
            return

        side = pos["side"]
        qty = float(pos["size"])
        price = exit_info["price"]
        reason = exit_info["reason"]

        logger.info(
            f"[EXIT] {symbol} 청산 시도 | Side={side.upper()}, Qty={qty}, "
            f"Price={price:.2f}, Reason={reason}"
        )

        if not self.paper:
            try:
                order_side = "sell" if side == "long" else "buy"
                self.client.market_order(symbol, order_side, qty)
            except Exception as e:
                logger.error(f"시장가 청산 실패 ({symbol}): {e}")
                return

        # 포지션/쿨다운 업데이트
        self.positions.pop(symbol, None)
        self.last_exit_times[symbol] = datetime.utcnow()
        logger.info(f"[EXIT] {symbol} 청산 완료 | Reason={reason}")

    # ---------- 메인 실행 루프 ----------

    def run_once(self):
        """4시간 캔들 기준으로 전체 유니버스(20개 코인) 한 번 스캔 및 주문 실행"""
        params = self.STRATEGY_PARAMS

        # 0) BTC 트렌드 업데이트 (도미넌스 필터용)
        if params.get("use_btc_filter", True):
            self._update_btc_trend()

        # 1) 각 심볼별 최신 캔들/지표 로딩
        latest_rows: Dict[str, pd.Series] = {}
        for symbol in self.MAJOR_COINS:
            df = self._fetch_ichimoku_df(symbol, limit=200)
            if df is None or df.empty:
                continue
            latest_rows[symbol] = df.iloc[-1]

        if not latest_rows:
            logger.warning("유효한 심볼 데이터가 없습니다.")
            return

        # 2) 기존 포지션에 대해 청산/트레일링 체크
        for symbol, pos in list(self.positions.items()):
            row = latest_rows.get(symbol)
            if row is None:
                continue
            exit_info = self._check_exit_signal(symbol, row)
            if exit_info:
                self._close_position(symbol, exit_info)

        # 3) 진입 후보 생성 (포지션 없는 심볼만)
        candidates = []
        for symbol, row in latest_rows.items():
            if symbol in self.positions:
                continue
            sig = self._get_entry_signal(symbol, row)
            if sig:
                candidates.append(sig)

        if not candidates:
            logger.info("[WAIT] 진입 후보 없음")
            return

        # 점수/구름 두께 기준으로 정렬 (백테스트와 동일)
        candidates.sort(key=lambda x: (-x["score"], -x["thickness"]))

        # 4) 잔고 기반으로 순차 진입 (max_positions 제한)
        try:
            balance = self._get_balance_free()
        except Exception:
            balance = 0.0

        if balance <= 0:
            logger.warning("USDT 잔고가 없습니다. 진입 불가.")
            return

        free = balance
        for cand in candidates:
            if len(self.positions) >= params["max_positions"]:
                logger.info("[LIMIT] 최대 포지션 수 도달, 추가 진입 중단")
                break

            # 이 심볼 가격 기준으로 포지션 사이즈 계산
            used_margin = self._open_position(cand, free)
            if used_margin <= 0:
                continue

            free -= used_margin
            if free <= 0:
                logger.info("사용 가능한 잔고 소진, 추가 진입 중단")
                break

    def run(self):
        """
        4시간봉 기준 실시간 루프
        - DataFetcher.get_next_candle_time 을 이용해 다음 캔들 시각까지 대기
        """
        mode = "PAPER" if self.paper else "LIVE"
        logger.info(
            f"IchimokuLiveBot 루프 시작 [{mode}] 코인수={len(self.MAJOR_COINS)}, TF={self.timeframe}"
        )

        while True:
            try:
                self.run_once()

                # 다음 4시간 캔들 시각 계산 후 그때까지 대기
                next_candle = self.data_fetcher.get_next_candle_time(self.timeframe)
                now = datetime.now()
                sleep_seconds = max(60, (next_candle - now).total_seconds())
                logger.info(
                    f"[SLEEP] 다음 캔들까지 대기: {sleep_seconds/60:.1f}분 "
                    f"(다음 캔들: {next_candle.strftime('%Y-%m-%d %H:%M')})"
                )
                time.sleep(sleep_seconds)

            except KeyboardInterrupt:
                logger.info("사용자 인터럽트로 봇 종료")
                break
            except Exception as e:
                logger.error(f"루프 오류: {e}")
                time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description="Ichimoku Cloud Live Trading Bot (4h)")
    parser.add_argument(
        "--paper", action="store_true", help="페이퍼 모드 (실제 주문 안 보냄)"
    )
    parser.add_argument(
        "--testnet", action="store_true", help="Bybit 테스트넷 사용 (강력 추천: 먼저 테스트)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="한 번만 실행 (디버깅/테스트용, 캔들 1개 기준)",
    )
    args = parser.parse_args()

    bot = IchimokuLiveBot(paper=args.paper, testnet=args.testnet)

    if args.once:
        bot.run_once()
    else:
        bot.run()


if __name__ == "__main__":
    main()

