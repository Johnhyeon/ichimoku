"""
페이퍼 트레이딩 클라이언트
- 실제 시세는 바이빗 메인넷에서 가져옴
- 주문/포지션은 가상으로 관리
- 실제 돈 없이 봇 테스트 가능
"""

import ccxt
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, field, asdict
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class PaperPosition:
    """가상 포지션"""
    symbol: str
    side: str  # 'long' or 'short'
    size: float  # 계약 수량
    entry_price: float
    entry_time: datetime
    leverage: int = 1

    # 추가 정보
    tier: str = ""
    stop_loss: float = 0
    take_profit: float = 0
    trailing_activated: bool = False
    highest_price: float = 0  # 트레일링용

    def calc_pnl(self, current_price: float) -> tuple:
        """현재 손익 계산"""
        if self.side == 'long':
            pnl_percent = (current_price - self.entry_price) / self.entry_price * 100
        else:
            pnl_percent = (self.entry_price - current_price) / self.entry_price * 100

        # USDT 손익 (레버리지 적용)
        position_value = self.size * self.entry_price
        pnl_usdt = position_value * (pnl_percent / 100)

        return pnl_usdt, pnl_percent


@dataclass
class PaperTrade:
    """거래 기록"""
    symbol: str
    side: str
    type: str  # 'entry' or 'exit'
    price: float
    size: float
    pnl: float = 0
    pnl_percent: float = 0
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


class PaperTradingClient:
    """페이퍼 트레이딩 클라이언트"""

    def __init__(self, initial_balance: float = 680.0, save_path: str = None):
        """
        Args:
            initial_balance: 초기 시드 (USDT)
            save_path: 상태 저장 경로
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions: Dict[str, PaperPosition] = {}
        self.trades: List[PaperTrade] = []
        self.lock = Lock()

        # 저장 경로
        if save_path:
            self.save_path = Path(save_path)
        else:
            self.save_path = Path(__file__).parent.parent / "data" / "paper_trading.json"
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        # 실제 시세용 CCXT (메인넷)
        self.exchange = ccxt.bybit({
            'options': {'defaultType': 'swap'}
        })

        # 통계
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.start_time = datetime.now()

        # 기존 상태 로드
        self._load_state()

        logger.info(f"페이퍼 트레이딩 초기화: 잔고 {self.balance:.2f} USDT")

    # ==================== 시세 조회 (실제 데이터) ====================

    def get_ticker(self, symbol: str) -> dict:
        """실제 시세 조회"""
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

    def fetch_ohlcv(self, symbol: str, timeframe: str = '15m', limit: int = 100):
        """OHLCV 데이터 조회"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return ohlcv
        except Exception as e:
            logger.error(f"OHLCV 조회 실패: {e}")
            raise

    def fetch_tickers(self, symbols: List[str] = None) -> dict:
        """여러 심볼 시세 조회"""
        try:
            return self.exchange.fetch_tickers(symbols)
        except Exception as e:
            logger.error(f"시세 일괄 조회 실패: {e}")
            return {}

    # ==================== 잔고/포지션 ====================

    def get_balance(self) -> dict:
        """잔고 조회"""
        used = sum(
            pos.size * pos.entry_price / pos.leverage
            for pos in self.positions.values()
        )
        return {
            "total": self.balance,
            "free": self.balance - used,
            "used": used
        }

    def get_position(self, symbol: str) -> dict:
        """포지션 조회"""
        if symbol not in self.positions:
            return {"side": None, "size": 0, "entry_price": 0, "pnl": 0, "pnl_percent": 0}

        pos = self.positions[symbol]
        try:
            ticker = self.get_ticker(symbol)
            current_price = ticker['last']
            pnl, pnl_percent = pos.calc_pnl(current_price)

            return {
                "side": pos.side,
                "size": pos.size,
                "entry_price": pos.entry_price,
                "current_price": current_price,
                "pnl": pnl,
                "pnl_percent": pnl_percent,
                "tier": pos.tier,
                "entry_time": pos.entry_time
            }
        except:
            return {
                "side": pos.side,
                "size": pos.size,
                "entry_price": pos.entry_price,
                "pnl": 0,
                "pnl_percent": 0
            }

    def get_all_positions(self) -> List[dict]:
        """모든 포지션 조회"""
        result = []
        for symbol in list(self.positions.keys()):
            pos_info = self.get_position(symbol)
            if pos_info['side']:
                pos_info['symbol'] = symbol
                result.append(pos_info)
        return result

    # ==================== 주문 ====================

    def market_order(self, symbol: str, side: str, amount: float,
                    tier: str = "", leverage: int = 1) -> dict:
        """
        시장가 주문 (가상)

        Args:
            symbol: 거래쌍
            side: 'buy' or 'sell'
            amount: USDT 금액
            tier: 티어 정보
            leverage: 레버리지
        """
        with self.lock:
            try:
                ticker = self.get_ticker(symbol)
                price = ticker['ask'] if side == 'buy' else ticker['bid']

                # 수수료 (0.055% taker)
                fee_rate = 0.00055
                fee = amount * fee_rate

                # 실제 사용 금액
                net_amount = amount - fee

                # 포지션 크기 (코인 수량)
                size = net_amount / price * leverage

                if side == 'buy':
                    # 롱 포지션 진입
                    if symbol in self.positions:
                        # 기존 포지션에 추가 (평균 단가 계산)
                        pos = self.positions[symbol]
                        if pos.side == 'long':
                            total_value = pos.size * pos.entry_price + size * price
                            total_size = pos.size + size
                            pos.entry_price = total_value / total_size
                            pos.size = total_size
                        else:
                            # 숏 포지션 청산
                            return self._close_position(symbol, price, "reverse")
                    else:
                        self.positions[symbol] = PaperPosition(
                            symbol=symbol,
                            side='long',
                            size=size,
                            entry_price=price,
                            entry_time=datetime.now(),
                            leverage=leverage,
                            tier=tier,
                            highest_price=price
                        )

                    self.balance -= (amount / leverage)

                else:
                    # 숏 포지션 또는 롱 청산
                    if symbol in self.positions and self.positions[symbol].side == 'long':
                        return self._close_position(symbol, price, "manual")
                    else:
                        # 숏 진입
                        self.positions[symbol] = PaperPosition(
                            symbol=symbol,
                            side='short',
                            size=size,
                            entry_price=price,
                            entry_time=datetime.now(),
                            leverage=leverage,
                            tier=tier
                        )
                        self.balance -= (amount / leverage)

                # 거래 기록
                trade = PaperTrade(
                    symbol=symbol,
                    side='long' if side == 'buy' else 'short',
                    type='entry',
                    price=price,
                    size=size
                )
                self.trades.append(trade)
                self.total_trades += 1

                self._save_state()

                logger.info(f"[PAPER] 진입: {symbol} {side} @ {price:.4f}, 크기: {amount:.2f} USDT")

                return {
                    "id": f"paper_{len(self.trades)}",
                    "symbol": symbol,
                    "side": side,
                    "amount": size,
                    "price": price,
                    "cost": amount,
                    "fee": fee,
                    "status": "filled"
                }

            except Exception as e:
                logger.error(f"[PAPER] 주문 실패: {e}")
                raise

    def _close_position(self, symbol: str, exit_price: float, reason: str = "") -> dict:
        """포지션 청산"""
        if symbol not in self.positions:
            return {"status": "no_position"}

        pos = self.positions[symbol]
        pnl, pnl_percent = pos.calc_pnl(exit_price)

        # 수수료
        fee = pos.size * exit_price * 0.00055
        pnl -= fee

        # 잔고 업데이트
        margin = pos.size * pos.entry_price / pos.leverage
        self.balance += margin + pnl

        # 통계 업데이트
        self.total_pnl += pnl
        if pnl > 0:
            self.winning_trades += 1

        # 거래 기록
        trade = PaperTrade(
            symbol=symbol,
            side=pos.side,
            type='exit',
            price=exit_price,
            size=pos.size,
            pnl=pnl,
            pnl_percent=pnl_percent,
            reason=reason
        )
        self.trades.append(trade)

        # 보유 시간 계산
        hold_time = datetime.now() - pos.entry_time
        hours = hold_time.seconds // 3600
        minutes = (hold_time.seconds % 3600) // 60
        hold_time_str = f"{hold_time.days}일 {hours}시간 {minutes}분" if hold_time.days else f"{hours}시간 {minutes}분"

        logger.info(f"[PAPER] 청산: {symbol} @ {exit_price:.4f}, PnL: {pnl:+.2f} ({pnl_percent:+.1f}%), 보유: {hold_time_str}")

        # 포지션 제거
        del self.positions[symbol]

        self._save_state()

        return {
            "symbol": symbol,
            "side": pos.side,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "size": pos.size,
            "pnl": pnl,
            "pnl_percent": pnl_percent,
            "reason": reason,
            "hold_time": hold_time_str,
            "status": "closed"
        }

    def close_position(self, symbol: str, reason: str = "manual") -> Optional[dict]:
        """포지션 청산"""
        if symbol not in self.positions:
            return None

        try:
            ticker = self.get_ticker(symbol)
            price = ticker['last']
            return self._close_position(symbol, price, reason)
        except Exception as e:
            logger.error(f"[PAPER] 청산 실패: {e}")
            return None

    def close_all_positions(self, reason: str = "close_all") -> List[dict]:
        """모든 포지션 청산"""
        results = []
        for symbol in list(self.positions.keys()):
            result = self.close_position(symbol, reason)
            if result:
                results.append(result)
        return results

    # ==================== TP/SL ====================

    def set_stop_loss(self, symbol: str, stop_price: float):
        """손절가 설정"""
        if symbol in self.positions:
            self.positions[symbol].stop_loss = stop_price
            self._save_state()

    def set_take_profit(self, symbol: str, take_price: float):
        """익절가 설정"""
        if symbol in self.positions:
            self.positions[symbol].take_profit = take_price
            self._save_state()

    def check_tp_sl(self) -> List[dict]:
        """TP/SL 체크 및 자동 청산"""
        closed = []

        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]

            try:
                ticker = self.get_ticker(symbol)
                price = ticker['last']

                # 롱 포지션
                if pos.side == 'long':
                    # 손절
                    if pos.stop_loss > 0 and price <= pos.stop_loss:
                        result = self._close_position(symbol, price, "stop_loss")
                        closed.append(result)
                        continue

                    # 익절
                    if pos.take_profit > 0 and price >= pos.take_profit:
                        result = self._close_position(symbol, price, "take_profit")
                        closed.append(result)
                        continue

                # 숏 포지션
                else:
                    if pos.stop_loss > 0 and price >= pos.stop_loss:
                        result = self._close_position(symbol, price, "stop_loss")
                        closed.append(result)
                        continue

                    if pos.take_profit > 0 and price <= pos.take_profit:
                        result = self._close_position(symbol, price, "take_profit")
                        closed.append(result)
                        continue

                # 트레일링 업데이트
                if pos.side == 'long' and price > pos.highest_price:
                    pos.highest_price = price

            except Exception as e:
                logger.error(f"TP/SL 체크 오류 ({symbol}): {e}")

        return closed

    # ==================== 통계 ====================

    def get_stats(self) -> dict:
        """거래 통계"""
        win_rate = self.winning_trades / max(1, self.total_trades) * 100
        roi = (self.balance - self.initial_balance) / self.initial_balance * 100

        return {
            "initial_balance": self.initial_balance,
            "current_balance": self.balance,
            "total_pnl": self.total_pnl,
            "roi": roi,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.total_trades - self.winning_trades,
            "win_rate": win_rate,
            "open_positions": len(self.positions),
            "start_time": self.start_time.isoformat()
        }

    def print_stats(self):
        """통계 출력"""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("[PAPER TRADING STATS]")
        print("="*50)
        print(f"Initial : {stats['initial_balance']:.2f} USDT")
        print(f"Balance : {stats['current_balance']:.2f} USDT")
        print(f"PnL     : {stats['total_pnl']:+.2f} USDT ({stats['roi']:+.1f}%)")
        print(f"Trades  : {stats['total_trades']}")
        print(f"WinRate : {stats['win_rate']:.1f}% ({stats['winning_trades']}W {stats['losing_trades']}L)")
        print(f"Open Pos: {stats['open_positions']}")
        print("="*50 + "\n")

    # ==================== 상태 저장/로드 ====================

    def _save_state(self):
        """상태 저장"""
        try:
            state = {
                "balance": self.balance,
                "initial_balance": self.initial_balance,
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "total_pnl": self.total_pnl,
                "start_time": self.start_time.isoformat(),
                "positions": {
                    symbol: {
                        "symbol": pos.symbol,
                        "side": pos.side,
                        "size": pos.size,
                        "entry_price": pos.entry_price,
                        "entry_time": pos.entry_time.isoformat(),
                        "leverage": pos.leverage,
                        "tier": pos.tier,
                        "stop_loss": pos.stop_loss,
                        "take_profit": pos.take_profit,
                        "highest_price": pos.highest_price
                    }
                    for symbol, pos in self.positions.items()
                },
                "trades": [
                    {
                        "symbol": t.symbol,
                        "side": t.side,
                        "type": t.type,
                        "price": t.price,
                        "size": t.size,
                        "pnl": t.pnl,
                        "pnl_percent": t.pnl_percent,
                        "reason": t.reason,
                        "timestamp": t.timestamp.isoformat()
                    }
                    for t in self.trades[-100:]  # 최근 100개만
                ],
                "last_updated": datetime.now().isoformat()
            }

            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"상태 저장 실패: {e}")

    def _load_state(self):
        """상태 로드"""
        if not self.save_path.exists():
            return

        try:
            with open(self.save_path, 'r', encoding='utf-8') as f:
                state = json.load(f)

            self.balance = state.get('balance', self.initial_balance)
            self.initial_balance = state.get('initial_balance', self.initial_balance)
            self.total_trades = state.get('total_trades', 0)
            self.winning_trades = state.get('winning_trades', 0)
            self.total_pnl = state.get('total_pnl', 0)

            if 'start_time' in state:
                self.start_time = datetime.fromisoformat(state['start_time'])

            # 포지션 복원
            for symbol, pos_data in state.get('positions', {}).items():
                self.positions[symbol] = PaperPosition(
                    symbol=pos_data['symbol'],
                    side=pos_data['side'],
                    size=pos_data['size'],
                    entry_price=pos_data['entry_price'],
                    entry_time=datetime.fromisoformat(pos_data['entry_time']),
                    leverage=pos_data.get('leverage', 1),
                    tier=pos_data.get('tier', ''),
                    stop_loss=pos_data.get('stop_loss', 0),
                    take_profit=pos_data.get('take_profit', 0),
                    highest_price=pos_data.get('highest_price', 0)
                )

            logger.info(f"상태 복원: 잔고 {self.balance:.2f}, 포지션 {len(self.positions)}개")

        except Exception as e:
            logger.error(f"상태 로드 실패: {e}")

    def reset(self):
        """초기화 (리셋)"""
        self.balance = self.initial_balance
        self.positions.clear()
        self.trades.clear()
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0
        self.start_time = datetime.now()

        if self.save_path.exists():
            self.save_path.unlink()

        logger.info(f"페이퍼 트레이딩 리셋: {self.initial_balance} USDT")


# 테스트
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 680 USDT로 시작
    client = PaperTradingClient(initial_balance=680.0)

    # 잔고 확인
    print(f"잔고: {client.get_balance()}")

    # BTC 시세 확인
    ticker = client.get_ticker("BTC/USDT:USDT")
    print(f"BTC 현재가: ${ticker['last']:,.2f}")

    # 테스트 주문
    print("\n[테스트 주문]")
    order = client.market_order("BTC/USDT:USDT", "buy", 100, tier="test", leverage=5)
    print(f"주문 결과: {order}")

    # 포지션 확인
    pos = client.get_position("BTC/USDT:USDT")
    print(f"포지션: {pos}")

    # 통계
    client.print_stats()
