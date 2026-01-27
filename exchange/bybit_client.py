"""바이빗 거래소 API 클라이언트"""

import ccxt
import logging
from typing import Optional
from config.settings import settings

logger = logging.getLogger(__name__)


class BybitClient:
    """바이빗 테스트넷/메인넷 API 클라이언트"""

    def __init__(self, testnet: bool = True):
        """
        바이빗 클라이언트 초기화

        Args:
            testnet: True면 테스트넷, False면 메인넷 사용
        """
        self.testnet = testnet

        # CCXT로 바이빗 연결
        self.exchange = ccxt.bybit({
            'apiKey': settings.BYBIT_API_KEY,
            'secret': settings.BYBIT_SECRET,
            'sandbox': testnet,  # 테스트넷 모드
            'options': {
                'defaultType': 'swap',  # 무기한 선물
            }
        })

        # 테스트넷 URL 설정
        if testnet:
            self.exchange.set_sandbox_mode(True)

        logger.info(f"바이빗 클라이언트 초기화 완료 (테스트넷: {testnet})")

    def get_balance(self) -> dict:
        """
        USDT 잔고 조회

        Returns:
            {"total": 1000.0, "free": 800.0, "used": 200.0}
        """
        try:
            balance = self.exchange.fetch_balance()
            usdt = balance.get('USDT', {})
            return {
                "total": float(usdt.get('total', 0) or 0),
                "free": float(usdt.get('free', 0) or 0),
                "used": float(usdt.get('used', 0) or 0)
            }
        except Exception as e:
            logger.error(f"잔고 조회 실패: {e}")
            raise

    def get_position(self, symbol: str) -> dict:
        """
        현재 포지션 조회

        Args:
            symbol: 거래쌍 (예: "BTC/USDT")

        Returns:
            {"side": "long", "size": 0.1, "entry_price": 50000, "pnl": 10.5}
            포지션 없으면: {"side": None, "size": 0, "entry_price": 0, "pnl": 0}
        """
        try:
            positions = self.exchange.fetch_positions([symbol])

            for pos in positions:
                if pos['symbol'] == symbol and float(pos['contracts'] or 0) > 0:
                    side = 'long' if pos['side'] == 'long' else 'short'
                    return {
                        "side": side,
                        "size": float(pos['contracts'] or 0),
                        "entry_price": float(pos['entryPrice'] or 0),
                        "pnl": float(pos['unrealizedPnl'] or 0),
                        "liquidation_price": float(pos['liquidationPrice'] or 0)
                    }

            return {"side": None, "size": 0, "entry_price": 0, "pnl": 0, "liquidation_price": 0}

        except Exception as e:
            logger.error(f"포지션 조회 실패: {e}")
            raise

    def market_order(self, symbol: str, side: str, amount: float) -> dict:
        """
        시장가 주문

        Args:
            symbol: 거래쌍 (예: "BTC/USDT")
            side: "buy" 또는 "sell"
            amount: 주문 수량

        Returns:
            주문 결과 딕셔너리
        """
        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount
            )
            logger.info(f"시장가 주문 체결: {side} {amount} {symbol}")
            return {
                "id": order['id'],
                "symbol": order['symbol'],
                "side": order['side'],
                "amount": float(order['amount'] or 0),
                "price": float(order['average'] or order['price'] or 0),
                "status": order['status']
            }
        except Exception as e:
            logger.error(f"시장가 주문 실패: {e}")
            raise

    def limit_order(self, symbol: str, side: str, amount: float, price: float) -> dict:
        """
        지정가 주문

        Args:
            symbol: 거래쌍
            side: "buy" 또는 "sell"
            amount: 주문 수량
            price: 주문 가격

        Returns:
            주문 결과 딕셔너리
        """
        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side=side,
                amount=amount,
                price=price
            )
            logger.info(f"지정가 주문 생성: {side} {amount} {symbol} @ {price}")
            return {
                "id": order['id'],
                "symbol": order['symbol'],
                "side": order['side'],
                "amount": float(order['amount'] or 0),
                "price": float(order['price'] or 0),
                "status": order['status']
            }
        except Exception as e:
            logger.error(f"지정가 주문 실패: {e}")
            raise

    def close_position(self, symbol: str) -> Optional[dict]:
        """
        현재 포지션 청산

        Args:
            symbol: 거래쌍

        Returns:
            주문 결과 또는 None (포지션 없을 경우)
        """
        position = self.get_position(symbol)

        if position['side'] is None:
            logger.info("청산할 포지션이 없습니다.")
            return None

        # 포지션 반대 방향으로 주문
        close_side = 'sell' if position['side'] == 'long' else 'buy'
        return self.market_order(symbol, close_side, position['size'])

    def set_leverage(self, symbol: str, leverage: int):
        """
        레버리지 설정

        Args:
            symbol: 거래쌍
            leverage: 레버리지 배수
        """
        try:
            self.exchange.set_leverage(leverage, symbol)
            logger.info(f"레버리지 설정: {symbol} {leverage}x")
        except Exception as e:
            # "leverage not modified" = 이미 동일한 레버리지로 설정됨 (정상)
            if 'not modified' in str(e):
                logger.debug(f"레버리지 이미 설정됨: {symbol} {leverage}x")
                return
            logger.error(f"레버리지 설정 실패: {e}")
            raise

    def set_stop_loss(self, symbol: str, stop_price: float, side: str = None):
        """
        손절가 설정 (조건부 주문)

        Args:
            symbol: 거래쌍
            stop_price: 손절 가격
            side: 포지션 방향 (None이면 현재 포지션에서 자동 감지)
        """
        try:
            if side is None:
                position = self.get_position(symbol)
                if position['side'] is None:
                    logger.warning("포지션이 없어 손절가를 설정할 수 없습니다.")
                    return
                side = position['side']
                amount = position['size']
            else:
                position = self.get_position(symbol)
                amount = position['size']

            # 롱 포지션이면 매도 손절, 숏 포지션이면 매수 손절
            trigger_side = 'sell' if side == 'long' else 'buy'

            self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=trigger_side,
                amount=amount,
                params={
                    'stopLoss': {
                        'triggerPrice': stop_price,
                        'type': 'market'
                    }
                }
            )
            logger.info(f"손절가 설정: {symbol} @ {stop_price}")

        except Exception as e:
            logger.error(f"손절가 설정 실패: {e}")
            raise

    def set_take_profit(self, symbol: str, take_price: float, side: str = None):
        """
        익절가 설정 (조건부 주문)

        Args:
            symbol: 거래쌍
            take_price: 익절 가격
            side: 포지션 방향 (None이면 현재 포지션에서 자동 감지)
        """
        try:
            if side is None:
                position = self.get_position(symbol)
                if position['side'] is None:
                    logger.warning("포지션이 없어 익절가를 설정할 수 없습니다.")
                    return
                side = position['side']
                amount = position['size']
            else:
                position = self.get_position(symbol)
                amount = position['size']

            # 롱 포지션이면 매도 익절, 숏 포지션이면 매수 익절
            trigger_side = 'sell' if side == 'long' else 'buy'

            self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=trigger_side,
                amount=amount,
                params={
                    'takeProfit': {
                        'triggerPrice': take_price,
                        'type': 'market'
                    }
                }
            )
            logger.info(f"익절가 설정: {symbol} @ {take_price}")

        except Exception as e:
            logger.error(f"익절가 설정 실패: {e}")
            raise

    def set_tp_sl(self, symbol: str, take_profit: float = None, stop_loss: float = None):
        """
        TP/SL 동시 설정

        Args:
            symbol: 거래쌍
            take_profit: 익절가
            stop_loss: 손절가
        """
        position = self.get_position(symbol)
        if position['side'] is None:
            logger.warning("포지션이 없어 TP/SL을 설정할 수 없습니다.")
            return

        try:
            params = {}
            if stop_loss:
                params['stopLoss'] = {'triggerPrice': stop_loss}
            if take_profit:
                params['takeProfit'] = {'triggerPrice': take_profit}

            self.exchange.set_position_mode(False, symbol)  # 단방향 모드

            # 포지션에 TP/SL 설정
            self.exchange.private_post_v5_position_trading_stop({
                'category': 'linear',
                'symbol': symbol.replace('/', ''),
                'takeProfit': str(take_profit) if take_profit else '',
                'stopLoss': str(stop_loss) if stop_loss else '',
                'positionIdx': 0  # 단방향 모드
            })

            logger.info(f"TP/SL 설정 완료: TP={take_profit}, SL={stop_loss}")

        except Exception as e:
            logger.error(f"TP/SL 설정 실패: {e}")
            raise

    def get_ticker(self, symbol: str) -> dict:
        """
        현재 시세 조회

        Args:
            symbol: 거래쌍

        Returns:
            {"last": 50000, "bid": 49999, "ask": 50001, "high": 51000, "low": 49000}
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                "last": float(ticker['last'] or 0),
                "bid": float(ticker['bid'] or 0),
                "ask": float(ticker['ask'] or 0),
                "high": float(ticker['high'] or 0),
                "low": float(ticker['low'] or 0),
                "volume": float(ticker['baseVolume'] or 0)
            }
        except Exception as e:
            logger.error(f"시세 조회 실패: {e}")
            raise

    def cancel_all_orders(self, symbol: str):
        """모든 미체결 주문 취소"""
        try:
            self.exchange.cancel_all_orders(symbol)
            logger.info(f"모든 미체결 주문 취소: {symbol}")
        except Exception as e:
            logger.error(f"주문 취소 실패: {e}")
            raise
