"""바이빗 거래소 API 클라이언트"""

import ccxt
import logging
from typing import Optional
from src.config import settings

logger = logging.getLogger(__name__)


class BybitClient:
    """바이빗 테스트넷/메인넷 API 클라이언트"""

    def __init__(self, testnet: bool = False):
        self.testnet = testnet

        self.exchange = ccxt.bybit({
            'apiKey': settings.BYBIT_API_KEY,
            'secret': settings.BYBIT_SECRET,
            'sandbox': testnet,
            'options': {
                'defaultType': 'swap',
            }
        })

        if testnet:
            self.exchange.set_sandbox_mode(True)

        logger.info(f"바이빗 클라이언트 초기화 완료 (테스트넷: {testnet})")

    def get_balance(self) -> dict:
        """USDT 잔고 조회"""
        try:
            balance = self.exchange.fetch_balance()

            # ccxt bybit 응답 구조 파싱
            # balance['total'], balance['free'], balance['used'] 또는
            # balance['USDT'] 형태로 올 수 있음

            total = 0
            free = 0
            used = 0

            # 방법 1: 직접 USDT 키 접근
            if 'USDT' in balance:
                usdt = balance['USDT']
                total = float(usdt.get('total', 0) or 0)
                free = float(usdt.get('free', 0) or 0)
                used = float(usdt.get('used', 0) or 0)

            # 방법 2: total/free/used 딕셔너리에서 USDT 찾기
            if total == 0:
                if 'total' in balance and isinstance(balance['total'], dict):
                    total = float(balance['total'].get('USDT', 0) or 0)
                if 'free' in balance and isinstance(balance['free'], dict):
                    free = float(balance['free'].get('USDT', 0) or 0)
                if 'used' in balance and isinstance(balance['used'], dict):
                    used = float(balance['used'].get('USDT', 0) or 0)

            # 미실현 손익 조회
            unrealized_pnl = 0
            try:
                positions = self.exchange.fetch_positions()
                for pos in positions:
                    if float(pos.get('contracts', 0) or 0) > 0:
                        unrealized_pnl += float(pos.get('unrealizedPnl', 0) or 0)
            except:
                pass

            return {
                "total": total,
                "free": free,
                "used": used,
                "unrealized_pnl": unrealized_pnl,
                "equity": total + unrealized_pnl,  # 실제 자산 가치
            }
        except Exception as e:
            logger.error(f"잔고 조회 실패: {e}")
            raise

    def get_position(self, symbol: str) -> dict:
        """현재 포지션 조회"""
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

    def get_all_positions(self) -> list:
        """모든 포지션 조회"""
        try:
            positions = self.exchange.fetch_positions()
            result = []

            for pos in positions:
                contracts = float(pos.get('contracts', 0) or 0)
                if contracts == 0:
                    continue

                # ccxt 표준화된 필드 사용
                side = pos.get('side', '').lower()

                # side가 없거나 이상한 경우 info에서 직접 확인
                if side not in ['long', 'short']:
                    info = pos.get('info', {})
                    raw_side = info.get('side', '') or info.get('positionIdx', '')
                    if str(raw_side).lower() in ['buy', '1']:
                        side = 'long'
                    elif str(raw_side).lower() in ['sell', '2']:
                        side = 'short'

                entry_price = float(pos.get('entryPrice', 0) or 0)
                unrealized_pnl = float(pos.get('unrealizedPnl', 0) or 0)

                # info에서 직접 PnL 가져오기 (더 정확함)
                info = pos.get('info', {})
                if 'unrealisedPnl' in info:
                    unrealized_pnl = float(info['unrealisedPnl'] or 0)

                result.append({
                    "symbol": pos['symbol'],
                    "side": side,
                    "size": contracts,
                    "entry_price": entry_price,
                    "pnl": unrealized_pnl,
                    "mark_price": float(pos.get('markPrice', 0) or 0),
                    "leverage": float(pos.get('leverage', 1) or 1),
                })

                logger.debug(f"포지션: {pos['symbol']} {side} size={contracts} entry={entry_price} pnl={unrealized_pnl}")

            return result

        except Exception as e:
            logger.error(f"포지션 조회 실패: {e}")
            raise

    def market_order(self, symbol: str, side: str, amount: float) -> dict:
        """시장가 주문"""
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

    def market_order_with_sl_tp(
        self,
        symbol: str,
        side: str,
        amount: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> dict:
        """SL/TP 포함 시장가 주문

        바이빗 서버에 조건부 주문(Stop Loss / Take Profit)을 함께 등록합니다.
        봇이 다운되어도 바이빗 서버에서 SL/TP가 자동 실행됩니다.

        Args:
            symbol: 거래 심볼 (예: "BTC/USDT:USDT")
            side: 주문 방향 ("buy" 또는 "sell")
            amount: 주문 수량
            stop_loss: 손절가 (None이면 설정 안함)
            take_profit: 익절가 (None이면 설정 안함)

        Returns:
            주문 결과 dict
        """
        try:
            params = {}

            if stop_loss is not None:
                params['stopLoss'] = {
                    'triggerPrice': stop_loss,
                    'type': 'market',
                    'triggerBy': 'MarkPrice'
                }
                logger.info(f"SL 설정: {stop_loss:.2f}")

            if take_profit is not None:
                params['takeProfit'] = {
                    'triggerPrice': take_profit,
                    'type': 'market',
                    'triggerBy': 'MarkPrice'
                }
                logger.info(f"TP 설정: {take_profit:.2f}")

            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount,
                params=params
            )

            sl_str = f", SL=${stop_loss:.2f}" if stop_loss else ""
            tp_str = f", TP=${take_profit:.2f}" if take_profit else ""
            logger.info(f"시장가 주문 체결 (SL/TP): {side} {amount} {symbol}{sl_str}{tp_str}")

            return {
                "id": order['id'],
                "symbol": order['symbol'],
                "side": order['side'],
                "amount": float(order['amount'] or 0),
                "price": float(order['average'] or order['price'] or 0),
                "status": order['status'],
                "stop_loss": stop_loss,
                "take_profit": take_profit
            }
        except Exception as e:
            logger.error(f"시장가 주문 (SL/TP) 실패: {e}")
            raise

    def set_leverage(self, symbol: str, leverage: int):
        """레버리지 설정"""
        try:
            self.exchange.set_leverage(leverage, symbol)
            logger.info(f"레버리지 설정: {symbol} {leverage}x")
        except Exception as e:
            if 'not modified' in str(e):
                logger.debug(f"레버리지 이미 설정됨: {symbol} {leverage}x")
                return
            logger.error(f"레버리지 설정 실패: {e}")
            raise

    def get_ticker(self, symbol: str) -> dict:
        """현재 시세 조회"""
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

    def get_closed_pnl(self, symbol: str = None, limit: int = 50) -> list:
        """청산된 포지션의 PnL 이력 조회 (바이빗 Closed PnL API)

        Args:
            symbol: 특정 심볼만 조회 (None이면 전체)
            limit: 조회 개수

        Returns:
            청산 이력 리스트
        """
        try:
            params = {
                'category': 'linear',
                'limit': limit
            }
            if symbol:
                # ccxt 심볼을 바이빗 심볼로 변환 (BTC/USDT:USDT -> BTCUSDT)
                bybit_symbol = symbol.replace('/USDT:USDT', 'USDT')
                params['symbol'] = bybit_symbol

            response = self.exchange.privateGetV5PositionClosedPnl(params)

            result = []
            if response and response.get('result') and response['result'].get('list'):
                for item in response['result']['list']:
                    # 바이빗 심볼을 ccxt 심볼로 변환 (BTCUSDT -> BTC/USDT:USDT)
                    raw_symbol = item.get('symbol', '')
                    if raw_symbol.endswith('USDT'):
                        ccxt_symbol = raw_symbol[:-4] + '/USDT:USDT'
                    else:
                        ccxt_symbol = raw_symbol

                    result.append({
                        'symbol': ccxt_symbol,
                        'side': 'long' if item.get('side') == 'Buy' else 'short',
                        'qty': float(item.get('qty', 0)),
                        'entry_price': float(item.get('avgEntryPrice', 0)),
                        'exit_price': float(item.get('avgExitPrice', 0)),
                        'closed_pnl': float(item.get('closedPnl', 0)),
                        'created_at': int(item.get('createdTime', 0)),
                        'updated_at': int(item.get('updatedTime', 0)),
                    })

            logger.debug(f"청산 이력 {len(result)}건 조회")
            return result

        except Exception as e:
            logger.error(f"청산 이력 조회 실패: {e}")
            return []

    def get_funding_rates(self, symbols: list = None) -> dict:
        """펀딩비 조회

        Args:
            symbols: 조회할 심볼 리스트 (None이면 주요 코인)

        Returns:
            {symbol: {rate, next_time, ...}} 형태의 딕셔너리
        """
        if symbols is None:
            symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
                       "XRP/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT"]

        result = {}
        try:
            for symbol in symbols:
                try:
                    # ccxt의 fetch_funding_rate 사용
                    funding = self.exchange.fetch_funding_rate(symbol)

                    result[symbol] = {
                        'symbol': symbol,
                        'funding_rate': float(funding.get('fundingRate', 0) or 0),
                        'funding_rate_pct': float(funding.get('fundingRate', 0) or 0) * 100,
                        'next_funding_time': funding.get('fundingTimestamp'),
                        'mark_price': float(funding.get('markPrice', 0) or 0),
                        'index_price': float(funding.get('indexPrice', 0) or 0),
                    }
                except Exception as e:
                    logger.warning(f"펀딩비 조회 실패 ({symbol}): {e}")
                    continue

            return result

        except Exception as e:
            logger.error(f"펀딩비 조회 실패: {e}")
            return {}

    def get_position_sl_tp(self, symbol: str) -> dict:
        """포지션의 SL/TP 설정 조회

        Args:
            symbol: 심볼

        Returns:
            {stop_loss, take_profit, trailing_stop, ...}
        """
        try:
            positions = self.exchange.fetch_positions([symbol])

            for pos in positions:
                if pos['symbol'] == symbol and float(pos['contracts'] or 0) > 0:
                    info = pos.get('info', {})

                    return {
                        'symbol': symbol,
                        'side': pos.get('side', ''),
                        'size': float(pos['contracts'] or 0),
                        'entry_price': float(pos['entryPrice'] or 0),
                        'stop_loss': float(info.get('stopLoss', 0) or 0),
                        'take_profit': float(info.get('takeProfit', 0) or 0),
                        'trailing_stop': float(info.get('trailingStop', 0) or 0),
                        'sl_trigger_by': info.get('slTriggerBy', ''),
                        'tp_trigger_by': info.get('tpTriggerBy', ''),
                    }

            return {'symbol': symbol, 'stop_loss': 0, 'take_profit': 0}

        except Exception as e:
            logger.error(f"SL/TP 조회 실패: {e}")
            return {'symbol': symbol, 'stop_loss': 0, 'take_profit': 0}

    def set_position_sl_tp(self, symbol: str, stop_loss: float = None, take_profit: float = None) -> bool:
        """포지션의 SL/TP 수정

        Args:
            symbol: 심볼
            stop_loss: 손절가 (None이면 변경 안함, 0이면 취소)
            take_profit: 익절가 (None이면 변경 안함, 0이면 취소)

        Returns:
            성공 여부
        """
        try:
            # 바이빗 심볼로 변환
            bybit_symbol = symbol.replace('/USDT:USDT', 'USDT')

            params = {
                'category': 'linear',
                'symbol': bybit_symbol,
                'positionIdx': 0,  # 단방향 모드
            }

            if stop_loss is not None:
                params['stopLoss'] = str(stop_loss) if stop_loss > 0 else ''
                params['slTriggerBy'] = 'MarkPrice'

            if take_profit is not None:
                params['takeProfit'] = str(take_profit) if take_profit > 0 else ''
                params['tpTriggerBy'] = 'MarkPrice'

            response = self.exchange.privatePostV5PositionTradingStop(params)

            if response and response.get('retCode') == 0:
                sl_str = f"SL=${stop_loss:.2f}" if stop_loss else "SL=취소"
                tp_str = f"TP=${take_profit:.2f}" if take_profit else "TP=취소"
                logger.info(f"SL/TP 수정 완료: {symbol} | {sl_str}, {tp_str}")
                return True
            else:
                logger.error(f"SL/TP 수정 실패: {response}")
                return False

        except Exception as e:
            logger.error(f"SL/TP 수정 실패: {e}")
            return False

    def get_account_stats(self, days: int = 30) -> dict:
        """계정 거래 통계 조회

        Args:
            days: 조회 기간 (일)

        Returns:
            {total_pnl, win_count, loss_count, win_rate, ...}
        """
        try:
            import time

            # 기간 설정
            end_time = int(time.time() * 1000)
            start_time = end_time - (days * 24 * 60 * 60 * 1000)

            params = {
                'category': 'linear',
                'limit': 100,
                'startTime': start_time,
                'endTime': end_time,
            }

            all_records = []
            cursor = None

            # 페이지네이션으로 모든 기록 조회
            for _ in range(10):  # 최대 1000건
                if cursor:
                    params['cursor'] = cursor

                response = self.exchange.privateGetV5PositionClosedPnl(params)

                if response and response.get('result'):
                    records = response['result'].get('list', [])
                    all_records.extend(records)

                    cursor = response['result'].get('nextPageCursor')
                    if not cursor or not records:
                        break
                else:
                    break

            # 통계 계산
            total_pnl = 0
            win_count = 0
            loss_count = 0
            win_pnl = 0
            loss_pnl = 0
            max_win = 0
            max_loss = 0

            for record in all_records:
                pnl = float(record.get('closedPnl', 0))
                total_pnl += pnl

                if pnl > 0:
                    win_count += 1
                    win_pnl += pnl
                    max_win = max(max_win, pnl)
                elif pnl < 0:
                    loss_count += 1
                    loss_pnl += pnl
                    max_loss = min(max_loss, pnl)

            total_trades = win_count + loss_count
            win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
            avg_win = (win_pnl / win_count) if win_count > 0 else 0
            avg_loss = (loss_pnl / loss_count) if loss_count > 0 else 0
            profit_factor = abs(win_pnl / loss_pnl) if loss_pnl != 0 else float('inf')

            return {
                'days': days,
                'total_trades': total_trades,
                'total_pnl': total_pnl,
                'win_count': win_count,
                'loss_count': loss_count,
                'win_rate': win_rate,
                'win_pnl': win_pnl,
                'loss_pnl': loss_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'max_win': max_win,
                'max_loss': max_loss,
                'profit_factor': profit_factor,
            }

        except Exception as e:
            logger.error(f"계정 통계 조회 실패: {e}")
            return {
                'days': days,
                'total_trades': 0,
                'total_pnl': 0,
                'win_count': 0,
                'loss_count': 0,
                'win_rate': 0,
            }

    def get_trade_history_from_exchange(self, days: int = 7, limit: int = 20) -> list:
        """바이빗에서 직접 거래 이력 조회

        Args:
            days: 조회 기간 (일)
            limit: 최대 조회 개수

        Returns:
            거래 이력 리스트
        """
        try:
            import time
            from datetime import datetime

            end_time = int(time.time() * 1000)
            start_time = end_time - (days * 24 * 60 * 60 * 1000)

            params = {
                'category': 'linear',
                'limit': min(limit, 100),
                'startTime': start_time,
                'endTime': end_time,
            }

            response = self.exchange.privateGetV5PositionClosedPnl(params)

            result = []
            if response and response.get('result') and response['result'].get('list'):
                for item in response['result']['list']:
                    raw_symbol = item.get('symbol', '')
                    if raw_symbol.endswith('USDT'):
                        symbol = raw_symbol[:-4] + '/USDT:USDT'
                    else:
                        symbol = raw_symbol

                    # 시간 변환
                    created_ts = int(item.get('createdTime', 0))
                    closed_at = datetime.fromtimestamp(created_ts / 1000) if created_ts else None

                    closed_pnl = float(item.get('closedPnl', 0))
                    entry_price = float(item.get('avgEntryPrice', 0))
                    exit_price = float(item.get('avgExitPrice', 0))
                    qty = float(item.get('qty', 0))

                    # 수익률 계산
                    if entry_price > 0:
                        pnl_pct = (exit_price - entry_price) / entry_price * 100
                        if item.get('side') == 'Sell':  # Short
                            pnl_pct = -pnl_pct
                    else:
                        pnl_pct = 0

                    result.append({
                        'symbol': symbol,
                        'side': 'long' if item.get('side') == 'Buy' else 'short',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'qty': qty,
                        'pnl_usd': closed_pnl,
                        'pnl_pct': pnl_pct,
                        'closed_at': closed_at,
                        'leverage': int(item.get('leverage', 1)),
                    })

            return result[:limit]

        except Exception as e:
            logger.error(f"거래 이력 조회 실패: {e}")
            return []

    def get_transaction_log(self, days: int = 7, tx_type: str = None) -> dict:
        """거래 내역 로그 조회 (펀딩비, 수수료 등)

        Args:
            days: 조회 기간 (일)
            tx_type: 거래 유형 필터 (None이면 전체)
                - TRADE: 거래 수수료
                - FUNDING_FEE: 펀딩비
                - SETTLEMENT: 정산

        Returns:
            {funding_fees: [...], trading_fees: [...], total_funding, total_fees}
        """
        try:
            import time
            from datetime import datetime

            end_time = int(time.time() * 1000)
            start_time = end_time - (days * 24 * 60 * 60 * 1000)

            params = {
                'accountType': 'UNIFIED',
                'category': 'linear',
                'limit': 50,
                'startTime': start_time,
                'endTime': end_time,
            }

            if tx_type:
                params['type'] = tx_type

            response = self.exchange.privateGetV5AccountTransactionLog(params)

            funding_fees = []
            trading_fees = []
            total_funding = 0
            total_trading_fee = 0

            if response and response.get('result') and response['result'].get('list'):
                for item in response['result']['list']:
                    tx_type_raw = item.get('type', '')
                    amount = float(item.get('cashFlow', 0) or 0)
                    fee = float(item.get('fee', 0) or 0)

                    raw_symbol = item.get('symbol', '')
                    if raw_symbol.endswith('USDT'):
                        symbol = raw_symbol[:-4]
                    else:
                        symbol = raw_symbol

                    created_ts = int(item.get('transactionTime', 0))
                    created_at = datetime.fromtimestamp(created_ts / 1000) if created_ts else None

                    record = {
                        'symbol': symbol,
                        'type': tx_type_raw,
                        'amount': amount,
                        'fee': fee,
                        'created_at': created_at,
                    }

                    if tx_type_raw == 'FUNDING_FEE':
                        funding_fees.append(record)
                        total_funding += amount
                    elif tx_type_raw == 'TRADE':
                        trading_fees.append(record)
                        total_trading_fee += fee

            return {
                'days': days,
                'funding_fees': funding_fees[:20],
                'trading_fees': trading_fees[:20],
                'total_funding': total_funding,
                'total_trading_fee': total_trading_fee,
                'funding_count': len(funding_fees),
                'trade_count': len(trading_fees),
            }

        except Exception as e:
            logger.error(f"거래 내역 조회 실패: {e}")
            return {
                'days': days,
                'funding_fees': [],
                'trading_fees': [],
                'total_funding': 0,
                'total_trading_fee': 0,
            }
