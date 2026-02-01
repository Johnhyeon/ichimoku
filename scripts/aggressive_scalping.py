"""
공격적 스캘핑 전략 백테스트
목표: 하루 10% 수익
기록: 모든 시도와 결과를 로그에 남김
"""

import sys
sys.path.insert(0, '/home/hyeon/project/ichimoku')

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from collections import defaultdict

from scripts.data_cache import load_cached_data, MAJOR_COINS, fetch_klines_from_api, save_to_cache

logger = logging.getLogger(__name__)

# ============================================================
# 실험 기록
# ============================================================
EXPERIMENT_LOG = []

def log_experiment(name: str, params: dict, results: dict):
    """실험 결과 기록"""
    EXPERIMENT_LOG.append({
        'timestamp': datetime.now().isoformat(),
        'name': name,
        'params': params,
        'results': results
    })

# ============================================================
# 기본 설정
# ============================================================
INITIAL_CAPITAL = 5_000_000  # 500만원
TARGET_COINS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'HYPEUSDT']

# ============================================================
# 지표 계산 함수들
# ============================================================
def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_bollinger(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    df['bb_mid'] = df['close'].rolling(period).mean()
    bb_std = df['close'].rolling(period).std()
    df['bb_upper'] = df['bb_mid'] + std * bb_std
    df['bb_lower'] = df['bb_mid'] - std * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100
    return df

def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calc_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calc_stoch_rsi(series: pd.Series, rsi_period: int = 14, stoch_period: int = 14, k_period: int = 3, d_period: int = 3):
    rsi = calc_rsi(series, rsi_period)
    rsi_min = rsi.rolling(stoch_period).min()
    rsi_max = rsi.rolling(stoch_period).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min) * 100
    k = stoch_rsi.rolling(k_period).mean()
    d = k.rolling(d_period).mean()
    return k, d

# ============================================================
# 전략 클래스
# ============================================================
class Strategy:
    def __init__(self, name: str, params: dict):
        self.name = name
        self.params = params

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

class RSIMomentumStrategy(Strategy):
    """RSI 모멘텀 전략: RSI 극단값에서 반전 시 진입"""

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['rsi'] = calc_rsi(df['close'], self.params.get('rsi_period', 7))
        df['rsi_prev'] = df['rsi'].shift(1)
        df['ema_fast'] = calc_ema(df['close'], self.params.get('ema_fast', 5))
        df['ema_slow'] = calc_ema(df['close'], self.params.get('ema_slow', 20))
        df['atr'] = calc_atr(df, 14)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        rsi_high = self.params.get('rsi_overbought', 80)
        rsi_low = self.params.get('rsi_oversold', 20)

        # 롱: RSI가 과매도에서 반등
        df['long_signal'] = (df['rsi_prev'] <= rsi_low) & (df['rsi'] > df['rsi_prev'])

        # 숏: RSI가 과매수에서 하락
        df['short_signal'] = (df['rsi_prev'] >= rsi_high) & (df['rsi'] < df['rsi_prev'])

        return df

class BreakoutStrategy(Strategy):
    """브레이크아웃 전략: 볼린저 밴드 돌파 시 추세 따라가기"""

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = calc_bollinger(df, self.params.get('bb_period', 20), self.params.get('bb_std', 2.0))
        df['rsi'] = calc_rsi(df['close'], 14)
        df['atr'] = calc_atr(df, 14)
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        min_volume = self.params.get('min_volume_ratio', 1.5)

        # 롱: BB 상단 돌파 + 거래량 증가 (추세 추종)
        df['long_signal'] = (df['close'] > df['bb_upper']) & (df['volume_ratio'] > min_volume)

        # 숏: BB 하단 이탈 + 거래량 증가 (추세 추종)
        df['short_signal'] = (df['close'] < df['bb_lower']) & (df['volume_ratio'] > min_volume)

        return df

class MACDCrossStrategy(Strategy):
    """MACD 크로스 전략"""

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['macd'], df['signal'], df['histogram'] = calc_macd(df['close'])
        df['macd_prev'] = df['macd'].shift(1)
        df['signal_prev'] = df['signal'].shift(1)
        df['atr'] = calc_atr(df, 14)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 롱: MACD가 시그널을 상향 돌파
        df['long_signal'] = (df['macd_prev'] < df['signal_prev']) & (df['macd'] > df['signal'])

        # 숏: MACD가 시그널을 하향 돌파
        df['short_signal'] = (df['macd_prev'] > df['signal_prev']) & (df['macd'] < df['signal'])

        return df

class StochRSIStrategy(Strategy):
    """Stochastic RSI 전략"""

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['stoch_k'], df['stoch_d'] = calc_stoch_rsi(df['close'])
        df['stoch_k_prev'] = df['stoch_k'].shift(1)
        df['stoch_d_prev'] = df['stoch_d'].shift(1)
        df['atr'] = calc_atr(df, 14)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        oversold = self.params.get('oversold', 20)
        overbought = self.params.get('overbought', 80)

        # 롱: K가 D를 상향돌파 + 과매도 영역
        df['long_signal'] = (df['stoch_k_prev'] < df['stoch_d_prev']) & \
                           (df['stoch_k'] > df['stoch_d']) & \
                           (df['stoch_k'] < oversold + 20)

        # 숏: K가 D를 하향돌파 + 과매수 영역
        df['short_signal'] = (df['stoch_k_prev'] > df['stoch_d_prev']) & \
                            (df['stoch_k'] < df['stoch_d']) & \
                            (df['stoch_k'] > overbought - 20)

        return df

class ComboStrategy(Strategy):
    """복합 전략: 여러 지표 조합"""

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['rsi'] = calc_rsi(df['close'], self.params.get('rsi_period', 7))
        df['rsi_prev'] = df['rsi'].shift(1)
        df = calc_bollinger(df, self.params.get('bb_period', 20), self.params.get('bb_std', 2.0))
        df['ema_fast'] = calc_ema(df['close'], self.params.get('ema_fast', 5))
        df['ema_slow'] = calc_ema(df['close'], self.params.get('ema_slow', 20))
        df['atr'] = calc_atr(df, 14)
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        rsi_low = self.params.get('rsi_oversold', 25)
        rsi_high = self.params.get('rsi_overbought', 75)

        # 롱: RSI 과매도 반전 + BB 하단 근처 + EMA 골든크로스 가능성
        df['long_signal'] = (df['rsi_prev'] <= rsi_low) & \
                           (df['rsi'] > df['rsi_prev']) & \
                           (df['close'] < df['bb_mid'])

        # 숏: RSI 과매수 반전 + BB 상단 근처
        df['short_signal'] = (df['rsi_prev'] >= rsi_high) & \
                            (df['rsi'] < df['rsi_prev']) & \
                            (df['close'] > df['bb_mid'])

        return df

# ============================================================
# 백테스트 엔진
# ============================================================
class Backtester:
    def __init__(self, strategy: Strategy, config: dict):
        self.strategy = strategy
        self.config = config
        self.initial_capital = config.get('initial_capital', 5_000_000)
        self.leverage = config.get('leverage', 10)
        self.position_pct = config.get('position_pct', 0.10)
        self.sl_pct = config.get('sl_pct', 2.0)  # 손절 %
        self.tp_pct = config.get('tp_pct', 5.0)  # 익절 %
        self.max_positions = config.get('max_positions', 4)
        self.cooldown_minutes = config.get('cooldown_minutes', 15)

    def run(self, all_data: Dict[str, pd.DataFrame]) -> dict:
        """백테스트 실행"""
        # 지표 계산
        for symbol in all_data:
            all_data[symbol] = self.strategy.calculate_indicators(all_data[symbol])
            all_data[symbol] = self.strategy.generate_signals(all_data[symbol])

        # 시간순 정렬
        all_bars = []
        for symbol, df in all_data.items():
            df = df.dropna()
            for _, row in df.iterrows():
                all_bars.append({'symbol': symbol, **row.to_dict()})

        all_bars.sort(key=lambda x: x['timestamp'])

        # 시간별 그룹화
        time_groups = {}
        for bar in all_bars:
            t = bar['timestamp']
            if t not in time_groups:
                time_groups[t] = {}
            time_groups[t][bar['symbol']] = bar

        sorted_times = sorted(time_groups.keys())

        # 시뮬레이션
        cash = self.initial_capital
        positions = {}
        trades = []
        equity_curve = []
        last_exit = {}
        daily_pnl = defaultdict(float)

        for t in sorted_times:
            bars = time_groups[t]
            closed = []

            # 청산 체크
            for sym, pos in positions.items():
                if sym not in bars:
                    continue

                bar = bars[sym]
                high, low, close = bar['high'], bar['low'], bar['close']
                entry = pos['entry_price']
                reason, exit_price = None, close

                if pos['side'] == 'long':
                    pnl_pct = (close - entry) / entry * 100
                    if low <= pos['stop_loss']:
                        reason, exit_price = 'SL', pos['stop_loss']
                    elif high >= pos['take_profit']:
                        reason, exit_price = 'TP', pos['take_profit']
                else:
                    pnl_pct = (entry - close) / entry * 100
                    if high >= pos['stop_loss']:
                        reason, exit_price = 'SL', pos['stop_loss']
                    elif low <= pos['take_profit']:
                        reason, exit_price = 'TP', pos['take_profit']

                if reason:
                    if pos['side'] == 'long':
                        final_pnl_pct = (exit_price - entry) / entry * 100
                    else:
                        final_pnl_pct = (entry - exit_price) / entry * 100

                    realized_pnl = final_pnl_pct * self.leverage / 100 * pos['position_size']
                    cash += pos['position_size'] + realized_pnl

                    trade_date = t.date()
                    daily_pnl[trade_date] += realized_pnl

                    trades.append({
                        'symbol': sym, 'side': pos['side'],
                        'entry_time': pos['entry_time'], 'exit_time': t,
                        'entry_price': entry, 'exit_price': exit_price,
                        'pnl_pct': round(final_pnl_pct * self.leverage, 2),
                        'pnl_krw': round(realized_pnl, 0),
                        'reason': reason
                    })
                    closed.append(sym)
                    last_exit[sym] = t

            for s in closed:
                del positions[s]

            # 자산 계산
            unrealized = 0
            for s, p in positions.items():
                if s in bars:
                    if p['side'] == 'long':
                        unrealized += (bars[s]['close'] - p['entry_price']) / p['entry_price'] * self.leverage * p['position_size'] / 100
                    else:
                        unrealized += (p['entry_price'] - bars[s]['close']) / p['entry_price'] * self.leverage * p['position_size'] / 100

            current_equity = cash + sum(p['position_size'] for p in positions.values()) + unrealized
            position_size = current_equity * self.position_pct

            # 진입
            if cash >= position_size and len(positions) < self.max_positions:
                for sym, bar in bars.items():
                    if sym in positions:
                        continue
                    if sym in last_exit:
                        if (t - last_exit[sym]).total_seconds() < self.cooldown_minutes * 60:
                            continue

                    price = bar['close']
                    atr = bar.get('atr', price * 0.01)

                    if bar.get('long_signal', False):
                        sl = price * (1 - self.sl_pct / 100)
                        tp = price * (1 + self.tp_pct / 100)
                        positions[sym] = {
                            'side': 'long', 'entry_price': price, 'entry_time': t,
                            'stop_loss': sl, 'take_profit': tp,
                            'position_size': position_size
                        }
                        cash -= position_size

                    elif bar.get('short_signal', False):
                        sl = price * (1 + self.sl_pct / 100)
                        tp = price * (1 - self.tp_pct / 100)
                        positions[sym] = {
                            'side': 'short', 'entry_price': price, 'entry_time': t,
                            'stop_loss': sl, 'take_profit': tp,
                            'position_size': position_size
                        }
                        cash -= position_size

                    if len(positions) >= self.max_positions:
                        break

            equity_curve.append({'time': t, 'equity': current_equity})

        return self._calculate_stats(trades, equity_curve, daily_pnl)

    def _calculate_stats(self, trades, equity_curve, daily_pnl) -> dict:
        if not trades:
            return {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0, 'return_pct': 0,
                    'max_dd': 0, 'profit_factor': 0, 'avg_daily_return': 0, 'trades': []}

        wins = [t for t in trades if t['pnl_pct'] > 0]
        losses = [t for t in trades if t['pnl_pct'] <= 0]

        total_profit = sum(t['pnl_krw'] for t in wins) if wins else 0
        total_loss = abs(sum(t['pnl_krw'] for t in losses)) if losses else 0

        peak, max_dd = self.initial_capital, 0
        for e in equity_curve:
            if e['equity'] > peak:
                peak = e['equity']
            dd = (peak - e['equity']) / peak * 100
            max_dd = max(max_dd, dd)

        final = equity_curve[-1]['equity'] if equity_curve else self.initial_capital
        days = len(daily_pnl)

        daily_returns = [v / self.initial_capital * 100 for v in daily_pnl.values()]
        avg_daily = np.mean(daily_returns) if daily_returns else 0

        # 하루 10% 이상 수익 낸 날 수
        big_days = len([d for d in daily_returns if d >= 10])

        return {
            'total_trades': len(trades),
            'win_rate': round(len(wins) / len(trades) * 100, 1),
            'total_pnl': round(sum(t['pnl_krw'] for t in trades), 0),
            'return_pct': round((final - self.initial_capital) / self.initial_capital * 100, 1),
            'max_dd': round(max_dd, 1),
            'profit_factor': round(total_profit / total_loss, 2) if total_loss > 0 else 999,
            'avg_win': round(np.mean([t['pnl_pct'] for t in wins]), 2) if wins else 0,
            'avg_loss': round(np.mean([t['pnl_pct'] for t in losses]), 2) if losses else 0,
            'avg_daily_return': round(avg_daily, 2),
            'big_days_10pct': big_days,
            'total_days': days,
            'trades_per_day': round(len(trades) / max(days, 1), 2),
            'final_equity': round(final, 0),
            'trades': trades[-10:]  # 최근 10개 거래
        }

# ============================================================
# 메인 실행
# ============================================================
def run_experiment(name: str, strategy: Strategy, config: dict, data: Dict[str, pd.DataFrame]) -> dict:
    """실험 실행 및 기록"""
    data_copy = {k: v.copy() for k, v in data.items()}

    bt = Backtester(strategy, config)
    results = bt.run(data_copy)

    log_experiment(name, {**strategy.params, **config}, results)

    return results

def print_results(name: str, results: dict):
    """결과 출력"""
    print(f"\n{'='*70}")
    print(f"[{name}]")
    print(f"{'='*70}")
    print(f"거래: {results['total_trades']}회 ({results['trades_per_day']}회/일)")
    print(f"승률: {results['win_rate']}%")
    print(f"평균 승리: {results['avg_win']}% | 평균 손실: {results['avg_loss']}%")
    print(f"총 수익: ₩{results['total_pnl']:,.0f} ({results['return_pct']}%)")
    print(f"최종 자산: ₩{results['final_equity']:,.0f}")
    print(f"MDD: {results['max_dd']}% | PF: {results['profit_factor']}")
    print(f"일평균 수익: {results['avg_daily_return']}%")
    print(f"10%+ 수익 낸 날: {results['big_days_10pct']}일 / {results['total_days']}일")

if __name__ == '__main__':
    import time as time_module

    print("=" * 70)
    print("공격적 스캘핑 전략 테스트")
    print("목표: 하루 10% 수익")
    print("=" * 70)

    # 데이터 로드
    print("\n데이터 로드 중...")
    data_15m = load_cached_data(TARGET_COINS, '15m')

    # 5분봉 데이터도 다운로드 (더 빠른 스캘핑용)
    print("5분봉 데이터 확인 중...")
    data_5m = {}
    for coin in TARGET_COINS:
        df = load_cached_data([coin], '5m').get(coin)
        if df is None:
            print(f"  {coin} 5분봉 다운로드...")
            df = fetch_klines_from_api(coin, 5, limit=20000)
            if df is not None:
                save_to_cache(df, coin, '5m')
        if df is not None:
            data_5m[coin] = df

    print(f"\n15분봉: {len(data_15m)}개 코인")
    print(f"5분봉: {len(data_5m)}개 코인")

    if data_15m:
        first_df = list(data_15m.values())[0]
        print(f"기간: {first_df['timestamp'].min()} ~ {first_df['timestamp'].max()}")

    # ============================================================
    # 실험 1: RSI 모멘텀 (기본)
    # ============================================================
    config_base = {
        'initial_capital': 5_000_000,
        'leverage': 10,
        'position_pct': 0.10,
        'sl_pct': 2.0,
        'tp_pct': 5.0,
        'max_positions': 4,
        'cooldown_minutes': 15
    }

    strategy1 = RSIMomentumStrategy('RSI_Momentum', {
        'rsi_period': 7,
        'rsi_overbought': 80,
        'rsi_oversold': 20,
        'ema_fast': 5,
        'ema_slow': 20
    })

    print("\n[실험 1] RSI 모멘텀 - 15분봉")
    results1 = run_experiment('RSI_Momentum_15m', strategy1, config_base, data_15m)
    print_results('RSI 모멘텀 15분봉', results1)

    # ============================================================
    # 실험 2: RSI 모멘텀 (5분봉)
    # ============================================================
    if data_5m:
        print("\n[실험 2] RSI 모멘텀 - 5분봉")
        results2 = run_experiment('RSI_Momentum_5m', strategy1, config_base, data_5m)
        print_results('RSI 모멘텀 5분봉', results2)

    # ============================================================
    # 실험 3: 공격적 설정 (높은 레버리지)
    # ============================================================
    config_aggressive = {
        'initial_capital': 5_000_000,
        'leverage': 20,
        'position_pct': 0.15,
        'sl_pct': 1.5,
        'tp_pct': 4.0,
        'max_positions': 3,
        'cooldown_minutes': 10
    }

    print("\n[실험 3] RSI 모멘텀 - 공격적 설정")
    results3 = run_experiment('RSI_Aggressive', strategy1, config_aggressive, data_15m)
    print_results('RSI 공격적', results3)

    # ============================================================
    # 실험 4: Stochastic RSI
    # ============================================================
    strategy4 = StochRSIStrategy('StochRSI', {
        'oversold': 20,
        'overbought': 80
    })

    print("\n[실험 4] Stochastic RSI")
    results4 = run_experiment('StochRSI', strategy4, config_base, data_15m)
    print_results('Stochastic RSI', results4)

    # ============================================================
    # 실험 5: 복합 전략
    # ============================================================
    strategy5 = ComboStrategy('Combo', {
        'rsi_period': 7,
        'rsi_overbought': 75,
        'rsi_oversold': 25,
        'bb_period': 20,
        'bb_std': 2.0,
        'ema_fast': 5,
        'ema_slow': 20
    })

    print("\n[실험 5] 복합 전략")
    results5 = run_experiment('Combo', strategy5, config_base, data_15m)
    print_results('복합 전략', results5)

    # ============================================================
    # 실험 6: 매우 공격적 (하루 10% 목표)
    # ============================================================
    config_ultra = {
        'initial_capital': 5_000_000,
        'leverage': 25,
        'position_pct': 0.20,
        'sl_pct': 1.0,
        'tp_pct': 3.0,
        'max_positions': 2,
        'cooldown_minutes': 5
    }

    strategy6 = RSIMomentumStrategy('RSI_Ultra', {
        'rsi_period': 5,
        'rsi_overbought': 85,
        'rsi_oversold': 15,
        'ema_fast': 3,
        'ema_slow': 10
    })

    print("\n[실험 6] 초공격적 설정")
    results6 = run_experiment('Ultra_Aggressive', strategy6, config_ultra, data_5m if data_5m else data_15m)
    print_results('초공격적', results6)

    # ============================================================
    # 결과 요약
    # ============================================================
    print("\n" + "=" * 70)
    print("실험 결과 요약")
    print("=" * 70)
    print(f"{'실험명':<25} {'수익%':>8} {'일평균%':>8} {'10%+일':>6} {'MDD':>6} {'PF':>6}")
    print("-" * 70)

    all_results = [
        ('RSI 모멘텀 15m', results1),
        ('RSI 모멘텀 5m', results2) if data_5m else None,
        ('RSI 공격적', results3),
        ('Stochastic RSI', results4),
        ('복합 전략', results5),
        ('초공격적', results6),
    ]

    for item in all_results:
        if item:
            name, r = item
            print(f"{name:<25} {r['return_pct']:>7.1f}% {r['avg_daily_return']:>7.2f}% "
                  f"{r['big_days_10pct']:>5}일 {r['max_dd']:>5.1f}% {r['profit_factor']:>6.2f}")

    print("=" * 70)

    # 로그 저장
    import json
    log_path = '/home/hyeon/project/ichimoku/data/experiment_log.json'
    with open(log_path, 'w') as f:
        json.dump(EXPERIMENT_LOG, f, indent=2, default=str)
    print(f"\n실험 로그 저장: {log_path}")
