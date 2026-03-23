"""
Williams Fractals Strategy Test Cases

실전 투입 전/후 검증용 테스트.
    .venv/Scripts/python.exe -m pytest tests/test_fractals_strategy.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fractals_strategy import (
    compute_fractals,
    get_fractals_entry_signal,
    check_fractals_exit,
    FRACTALS_LEVERAGE,
    FRACTALS_POSITION_PCT,
    FRACTALS_PARAMS,
)


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def sample_ohlcv():
    """프랙탈이 형성되는 30봉 샘플 데이터."""
    np.random.seed(42)
    n = 30
    base = 100 + np.cumsum(np.random.randn(n) * 0.5)
    # 인위적으로 봉 10에 고점 프랙탈, 봉 15에 저점 프랙탈 생성
    base[10] = base[8:13].max() + 2
    base[15] = base[13:18].min() - 2
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="4h"),
        "open": base - 0.3,
        "high": base + 1.0,
        "low": base - 1.0,
        "close": base,
        "volume": np.random.randint(1000, 5000, n).astype(float),
    })
    return df


@pytest.fixture
def long_entry_rows():
    """롱 진입 조건을 충족하는 prev/curr 행."""
    prev = pd.Series({
        "close": 100.0, "last_fractal_high": 102.0, "last_fractal_low": 98.0,
        "ema_fast": 101.0, "ema_slow": 100.0, "rsi": 50.0, "adx": 25.0,
    })
    curr = pd.Series({
        "close": 103.0, "high": 103.5, "low": 100.5,
        "last_fractal_high": 102.0, "last_fractal_low": 98.0,
        "ema_fast": 101.5, "ema_slow": 100.5, "rsi": 55.0, "adx": 28.0,
    })
    return prev, curr


@pytest.fixture
def short_entry_rows():
    """숏 진입 조건을 충족하는 prev/curr 행."""
    prev = pd.Series({
        "close": 100.0, "last_fractal_high": 102.0, "last_fractal_low": 99.0,
        "ema_fast": 99.0, "ema_slow": 100.0, "rsi": 50.0, "adx": 25.0,
    })
    curr = pd.Series({
        "close": 98.0, "high": 100.5, "low": 97.5,
        "last_fractal_high": 102.0, "last_fractal_low": 99.0,
        "ema_fast": 98.5, "ema_slow": 100.0, "rsi": 45.0, "adx": 28.0,
    })
    return prev, curr


# ═══════════════════════════════════════════════════════════════
# TC-01: Parameters
# ═══════════════════════════════════════════════════════════════

class TestParameters:
    """TC-01: 전략 파라미터 검증."""

    def test_leverage(self):
        assert FRACTALS_LEVERAGE == 10

    def test_position_pct(self):
        assert FRACTALS_POSITION_PCT == 0.05

    def test_sl_pct(self):
        assert FRACTALS_PARAMS["sl_pct"] == 3.0

    def test_tp_pct(self):
        assert FRACTALS_PARAMS["tp_pct"] == 10.0

    def test_trail_start(self):
        assert FRACTALS_PARAMS["trail_start_pct"] == 2.0

    def test_trail_pct(self):
        assert FRACTALS_PARAMS["trail_pct"] == 1.5

    def test_cooldown(self):
        assert FRACTALS_PARAMS["cooldown_candles"] == 2

    def test_max_positions(self):
        assert FRACTALS_PARAMS["max_positions"] == 5

    def test_ema_filter(self):
        assert FRACTALS_PARAMS["ema_fast"] == 20
        assert FRACTALS_PARAMS["ema_slow"] == 50

    def test_rsi_filter(self):
        assert FRACTALS_PARAMS["rsi_long_max"] == 65
        assert FRACTALS_PARAMS["rsi_short_min"] == 35

    def test_adx_filter(self):
        assert FRACTALS_PARAMS["adx_min"] == 20

    def test_tp_greater_than_sl(self):
        """손익비 > 1 확인."""
        assert FRACTALS_PARAMS["tp_pct"] > FRACTALS_PARAMS["sl_pct"]


# ═══════════════════════════════════════════════════════════════
# TC-02: Indicator Computation
# ═══════════════════════════════════════════════════════════════

class TestIndicators:
    """TC-02: 지표 계산 검증."""

    def test_fractals_computed(self, sample_ohlcv):
        df = compute_fractals(sample_ohlcv, n=5)
        assert "last_fractal_high" in df.columns
        assert "last_fractal_low" in df.columns
        assert "ema_fast" in df.columns
        assert "ema_slow" in df.columns
        assert "rsi" in df.columns
        assert "adx" in df.columns

    def test_fractals_not_all_nan(self, sample_ohlcv):
        df = compute_fractals(sample_ohlcv, n=5)
        assert df["last_fractal_high"].notna().any()
        assert df["last_fractal_low"].notna().any()

    def test_fractal_high_is_local_max(self, sample_ohlcv):
        df = compute_fractals(sample_ohlcv, n=5)
        for i in df[df["fractal_high"].notna()].index:
            h = df.loc[i, "high"]
            for j in range(1, 6):
                if i - j >= 0:
                    assert h > df.loc[i - j, "high"]
                if i + j < len(df):
                    assert h > df.loc[i + j, "high"]

    def test_fractal_low_is_local_min(self, sample_ohlcv):
        df = compute_fractals(sample_ohlcv, n=5)
        for i in df[df["fractal_low"].notna()].index:
            l = df.loc[i, "low"]
            for j in range(1, 6):
                if i - j >= 0:
                    assert l < df.loc[i - j, "low"]
                if i + j < len(df):
                    assert l < df.loc[i + j, "low"]

    def test_rsi_range(self, sample_ohlcv):
        df = compute_fractals(sample_ohlcv, n=5)
        valid_rsi = df["rsi"].dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_adx_non_negative(self, sample_ohlcv):
        df = compute_fractals(sample_ohlcv, n=5)
        valid_adx = df["adx"].dropna()
        assert (valid_adx >= 0).all()

    def test_ema_fast_reacts_faster(self, sample_ohlcv):
        """EMA20이 EMA50보다 최근 가격에 더 가까운지."""
        df = compute_fractals(sample_ohlcv, n=5)
        last = df.iloc[-1]
        price = last["close"]
        assert abs(last["ema_fast"] - price) <= abs(last["ema_slow"] - price)


# ═══════════════════════════════════════════════════════════════
# TC-03: Entry Signal
# ═══════════════════════════════════════════════════════════════

class TestEntrySignal:
    """TC-03: 진입 시그널 검증."""

    def test_long_entry(self, long_entry_rows):
        prev, curr = long_entry_rows
        sig = get_fractals_entry_signal("TEST/USDT:USDT", curr, prev, None)
        assert sig is not None
        assert sig["side"] == "long"

    def test_short_entry(self, short_entry_rows):
        prev, curr = short_entry_rows
        sig = get_fractals_entry_signal("TEST/USDT:USDT", curr, prev, None)
        assert sig is not None
        assert sig["side"] == "short"

    def test_long_sl_below_entry(self, long_entry_rows):
        prev, curr = long_entry_rows
        sig = get_fractals_entry_signal("TEST/USDT:USDT", curr, prev, None)
        assert sig["stop_loss"] < sig["price"]

    def test_long_tp_above_entry(self, long_entry_rows):
        prev, curr = long_entry_rows
        sig = get_fractals_entry_signal("TEST/USDT:USDT", curr, prev, None)
        assert sig["take_profit"] > sig["price"]

    def test_short_sl_above_entry(self, short_entry_rows):
        prev, curr = short_entry_rows
        sig = get_fractals_entry_signal("TEST/USDT:USDT", curr, prev, None)
        assert sig["stop_loss"] > sig["price"]

    def test_short_tp_below_entry(self, short_entry_rows):
        prev, curr = short_entry_rows
        sig = get_fractals_entry_signal("TEST/USDT:USDT", curr, prev, None)
        assert sig["take_profit"] < sig["price"]

    def test_sl_distance_correct(self, long_entry_rows):
        prev, curr = long_entry_rows
        sig = get_fractals_entry_signal("TEST/USDT:USDT", curr, prev, None)
        sl_dist = abs(sig["price"] - sig["stop_loss"]) / sig["price"] * 100
        assert abs(sl_dist - FRACTALS_PARAMS["sl_pct"]) < 0.01

    def test_tp_distance_correct(self, short_entry_rows):
        prev, curr = short_entry_rows
        sig = get_fractals_entry_signal("TEST/USDT:USDT", curr, prev, None)
        tp_dist = abs(sig["price"] - sig["take_profit"]) / sig["price"] * 100
        assert abs(tp_dist - FRACTALS_PARAMS["tp_pct"]) < 0.01

    def test_symbol_preserved(self, long_entry_rows):
        prev, curr = long_entry_rows
        sig = get_fractals_entry_signal("BTC/USDT:USDT", curr, prev, None)
        assert sig["symbol"] == "BTC/USDT:USDT"


# ═══════════════════════════════════════════════════════════════
# TC-04: Filter Blocking
# ═══════════════════════════════════════════════════════════════

class TestFilters:
    """TC-04: 필터 검증 (use_filters 토글)."""

    FILTERED_PARAMS = {**FRACTALS_PARAMS, "use_filters": True}

    def test_nofilter_ignores_bad_adx(self, long_entry_rows):
        """노필터 모드: ADX 낮아도 진입."""
        prev, curr = long_entry_rows
        curr = curr.copy()
        curr["adx"] = 5
        sig = get_fractals_entry_signal("TEST", curr, prev, None)
        assert sig is not None

    def test_nofilter_ignores_ema_mismatch(self, long_entry_rows):
        """노필터 모드: EMA 역방향이어도 진입."""
        prev, curr = long_entry_rows
        curr = curr.copy()
        curr["ema_fast"] = 99
        curr["ema_slow"] = 100
        sig = get_fractals_entry_signal("TEST", curr, prev, None)
        assert sig is not None

    def test_filtered_adx_blocks(self, long_entry_rows):
        """필터 모드: ADX<20 차단."""
        prev, curr = long_entry_rows
        curr = curr.copy()
        curr["adx"] = 15
        sig = get_fractals_entry_signal("TEST", curr, prev, None, self.FILTERED_PARAMS)
        assert sig is None

    def test_filtered_ema_blocks_long(self, long_entry_rows):
        """필터 모드: EMA 역방향 차단."""
        prev, curr = long_entry_rows
        curr = curr.copy()
        curr["ema_fast"] = 99
        curr["ema_slow"] = 100
        sig = get_fractals_entry_signal("TEST", curr, prev, None, self.FILTERED_PARAMS)
        assert sig is None

    def test_filtered_rsi_blocks_long(self, long_entry_rows):
        """필터 모드: RSI>65 차단."""
        prev, curr = long_entry_rows
        curr = curr.copy()
        curr["rsi"] = 70
        sig = get_fractals_entry_signal("TEST", curr, prev, None, self.FILTERED_PARAMS)
        assert sig is None

    def test_filtered_rsi_blocks_short(self, short_entry_rows):
        """필터 모드: RSI<35 차단."""
        prev, curr = short_entry_rows
        curr = curr.copy()
        curr["rsi"] = 30
        sig = get_fractals_entry_signal("TEST", curr, prev, None, self.FILTERED_PARAMS)
        assert sig is None

    def test_default_is_nofilter(self):
        """기본 파라미터가 노필터인지 확인."""
        assert FRACTALS_PARAMS.get("use_filters") is False

    def test_adx_boundary_pass(self, long_entry_rows):
        """필터 모드: ADX=20 경계값 통과."""
        prev, curr = long_entry_rows
        curr = curr.copy()
        curr["adx"] = 20
        sig = get_fractals_entry_signal("TEST", curr, prev, None, self.FILTERED_PARAMS)
        assert sig is not None

    def test_nan_fractal_blocks(self):
        prev = pd.Series({"close": 100, "last_fractal_high": np.nan, "last_fractal_low": np.nan})
        curr = pd.Series({"close": 103, "last_fractal_high": np.nan, "last_fractal_low": np.nan,
                           "ema_fast": 101, "ema_slow": 100, "rsi": 50, "adx": 25})
        sig = get_fractals_entry_signal("TEST", curr, prev, None)
        assert sig is None

    def test_no_breakout_no_signal(self):
        """프랙탈 돌파/이탈 없으면 시그널 없음."""
        prev = pd.Series({"close": 101, "last_fractal_high": 105, "last_fractal_low": 95,
                           "ema_fast": 101, "ema_slow": 100, "rsi": 50, "adx": 25})
        curr = pd.Series({"close": 102, "high": 103, "low": 100,
                           "last_fractal_high": 105, "last_fractal_low": 95,
                           "ema_fast": 101, "ema_slow": 100, "rsi": 50, "adx": 25})
        sig = get_fractals_entry_signal("TEST", curr, prev, None)
        assert sig is None


# ═══════════════════════════════════════════════════════════════
# TC-05: Cooldown
# ═══════════════════════════════════════════════════════════════

class TestCooldown:
    """TC-05: 쿨다운 검증."""

    def test_cooldown_blocks(self, long_entry_rows):
        prev, curr = long_entry_rows
        sig = get_fractals_entry_signal("TEST", curr, prev, 1)
        assert sig is None, "1 candle < 2 cooldown should block"

    def test_cooldown_exact_blocks(self, long_entry_rows):
        prev, curr = long_entry_rows
        sig = get_fractals_entry_signal("TEST", curr, prev, 1)
        assert sig is None, "1 < 2 should block"

    def test_cooldown_expired_passes(self, long_entry_rows):
        prev, curr = long_entry_rows
        sig = get_fractals_entry_signal("TEST", curr, prev, 3)
        assert sig is not None, "3 >= 2 should pass"

    def test_cooldown_none_passes(self, long_entry_rows):
        prev, curr = long_entry_rows
        sig = get_fractals_entry_signal("TEST", curr, prev, None)
        assert sig is not None, "None cooldown should pass"


# ═══════════════════════════════════════════════════════════════
# TC-06: Exit — Stop Loss
# ═══════════════════════════════════════════════════════════════

class TestExitSL:
    """TC-06: 손절 청산 검증."""

    def test_long_sl_triggered(self):
        pos = {"side": "long", "entry_price": 100, "stop_loss": 97, "take_profit": 110, "best_pnl": 0, "trailing": False}
        row = pd.Series({"high": 101, "low": 96.5, "close": 97.5})
        r = check_fractals_exit(pos, row)
        assert r is not None
        assert r["reason"] == "SL"
        assert r["price"] == 97

    def test_short_sl_triggered(self):
        pos = {"side": "short", "entry_price": 100, "stop_loss": 103, "take_profit": 90, "best_pnl": 0, "trailing": False}
        row = pd.Series({"high": 103.5, "low": 99, "close": 102})
        r = check_fractals_exit(pos, row)
        assert r is not None
        assert r["reason"] == "SL"
        assert r["price"] == 103

    def test_long_sl_not_triggered(self):
        pos = {"side": "long", "entry_price": 100, "stop_loss": 97, "take_profit": 110, "best_pnl": 0, "trailing": False}
        row = pd.Series({"high": 102, "low": 97.5, "close": 101})
        r = check_fractals_exit(pos, row)
        assert r is None


# ═══════════════════════════════════════════════════════════════
# TC-07: Exit — Take Profit
# ═══════════════════════════════════════════════════════════════

class TestExitTP:
    """TC-07: 익절 청산 검증."""

    def test_long_tp_triggered(self):
        pos = {"side": "long", "entry_price": 100, "stop_loss": 97, "take_profit": 110, "best_pnl": 0, "trailing": False}
        row = pd.Series({"high": 111, "low": 108, "close": 109})
        r = check_fractals_exit(pos, row)
        assert r is not None
        assert r["reason"] == "TP"
        assert r["price"] == 110

    def test_short_tp_triggered(self):
        pos = {"side": "short", "entry_price": 100, "stop_loss": 103, "take_profit": 90, "best_pnl": 0, "trailing": False}
        row = pd.Series({"high": 95, "low": 89, "close": 91})
        r = check_fractals_exit(pos, row)
        assert r is not None
        assert r["reason"] == "TP"
        assert r["price"] == 90


# ═══════════════════════════════════════════════════════════════
# TC-08: Exit — Trailing Stop
# ═══════════════════════════════════════════════════════════════

class TestExitTrail:
    """TC-08: 트레일링 스탑 검증."""

    def test_trail_activates_at_threshold(self):
        pos = {"side": "long", "entry_price": 100, "stop_loss": 97, "take_profit": 110, "best_pnl": 0, "trailing": False}
        row = pd.Series({"high": 102.5, "low": 101.5, "close": 102})
        check_fractals_exit(pos, row)
        assert pos["best_pnl"] >= 2.0
        assert pos["trailing"] is True

    def test_trail_not_activated_below_threshold(self):
        pos = {"side": "long", "entry_price": 100, "stop_loss": 97, "take_profit": 110, "best_pnl": 0, "trailing": False}
        row = pd.Series({"high": 101.5, "low": 100.5, "close": 101})
        check_fractals_exit(pos, row)
        assert pos["trailing"] is False

    def test_trail_triggers_on_drawdown(self):
        pos = {"side": "long", "entry_price": 100, "stop_loss": 97, "take_profit": 110, "best_pnl": 4.0, "trailing": True}
        row = pd.Series({"high": 103, "low": 101.5, "close": 101.8})
        r = check_fractals_exit(pos, row)
        assert r is not None
        assert r["reason"] == "Trail"

    def test_trail_holds_if_drawdown_small(self):
        pos = {"side": "long", "entry_price": 100, "stop_loss": 97, "take_profit": 110, "best_pnl": 3.0, "trailing": True}
        row = pd.Series({"high": 103, "low": 101.5, "close": 102})
        r = check_fractals_exit(pos, row)
        assert r is None, "Drawdown 1% < trail 1.5% should hold"

    def test_short_trail_triggers(self):
        pos = {"side": "short", "entry_price": 100, "stop_loss": 103, "take_profit": 90, "best_pnl": 4.0, "trailing": True}
        row = pd.Series({"high": 98.5, "low": 97.5, "close": 98.2})
        r = check_fractals_exit(pos, row)
        assert r is not None
        assert r["reason"] == "Trail"

    def test_best_pnl_increases(self):
        pos = {"side": "long", "entry_price": 100, "stop_loss": 97, "take_profit": 110, "best_pnl": 1.0, "trailing": False}
        row = pd.Series({"high": 105, "low": 103, "close": 104})
        check_fractals_exit(pos, row)
        assert pos["best_pnl"] == pytest.approx(5.0, abs=0.1)


# ═══════════════════════════════════════════════════════════════
# TC-09: Exit Priority (SL before TP before Trail)
# ═══════════════════════════════════════════════════════════════

class TestExitPriority:
    """TC-09: 청산 우선순위 — SL > TP > Trail."""

    def test_sl_takes_priority_over_tp(self):
        """같은 캔들에서 SL과 TP 모두 터치하면 SL 우선."""
        pos = {"side": "long", "entry_price": 100, "stop_loss": 95, "take_profit": 110, "best_pnl": 0, "trailing": False}
        row = pd.Series({"high": 115, "low": 93, "close": 105})
        r = check_fractals_exit(pos, row)
        assert r["reason"] == "SL"

    def test_tp_takes_priority_over_trail(self):
        """TP와 Trail 동시 조건 시 TP 우선."""
        pos = {"side": "long", "entry_price": 100, "stop_loss": 97, "take_profit": 105, "best_pnl": 6.0, "trailing": True}
        row = pd.Series({"high": 106, "low": 103, "close": 103.5})
        r = check_fractals_exit(pos, row)
        assert r["reason"] == "TP"


# ═══════════════════════════════════════════════════════════════
# TC-10: State Persistence
# ═══════════════════════════════════════════════════════════════

class TestStatePersistence:
    """TC-10: 상태 저장/복원 검증."""

    def test_save_state_has_exit_candle_counts(self):
        import inspect
        from src.trader import IchimokuTrader
        src = inspect.getsource(IchimokuTrader._save_state)
        assert "exit_candle_counts" in src

    def test_load_state_has_exit_candle_counts(self):
        import inspect
        from src.trader import IchimokuTrader
        src = inspect.getsource(IchimokuTrader._load_state)
        assert "exit_candle_counts" in src

    def test_save_state_has_strategy_mode(self):
        import inspect
        from src.trader import IchimokuTrader
        src = inspect.getsource(IchimokuTrader._save_state)
        assert "strategy_mode" in src

    def test_position_has_best_pnl_field(self):
        """positions dict에 best_pnl 포함 (자동 저장됨)."""
        import inspect
        from src.trader import IchimokuTrader
        src = inspect.getsource(IchimokuTrader._open_position)
        assert "best_pnl" in src


# ═══════════════════════════════════════════════════════════════
# TC-11: Trader Integration
# ═══════════════════════════════════════════════════════════════

class TestTraderIntegration:
    """TC-11: Trader 클래스 통합 검증."""

    def test_default_strategy_mode(self):
        import inspect
        from src.trader import IchimokuTrader
        src = inspect.getsource(IchimokuTrader.__init__)
        assert 'self.strategy_mode = "fractals"' in src

    def test_run_once_dispatches(self):
        import inspect
        from src.trader import IchimokuTrader
        src = inspect.getsource(IchimokuTrader.run_once)
        assert "_run_once_fractals" in src

    def test_fractals_methods_exist(self):
        from src.trader import IchimokuTrader
        assert hasattr(IchimokuTrader, "_run_once_fractals")
        assert hasattr(IchimokuTrader, "_fetch_fractals_df")

    def test_ichimoku_preserved(self):
        from src.trader import IchimokuTrader
        assert hasattr(IchimokuTrader, "_run_once_ichimoku")
        assert hasattr(IchimokuTrader, "_fetch_ichimoku_df")

    def test_leverage_set_for_fractals(self):
        import inspect
        from src.trader import IchimokuTrader
        src = inspect.getsource(IchimokuTrader.__init__)
        assert "FRACTALS_LEVERAGE" in src


# ═══════════════════════════════════════════════════════════════
# TC-12: Telegram Naming
# ═══════════════════════════════════════════════════════════════

class TestTelegramNaming:
    """TC-12: 텔레그램 표시명 검증."""

    def test_no_ichimoku_korean(self):
        with open("src/telegram_bot.py", encoding="utf-8") as f:
            content = f.read()
        assert "이치모쿠" not in content

    def test_no_ilmok(self):
        with open("src/telegram_bot.py", encoding="utf-8") as f:
            content = f.read()
        assert "일목균형표" not in content

    def test_vertex_present(self):
        with open("src/telegram_bot.py", encoding="utf-8") as f:
            content = f.read()
        assert content.count("Vertex") >= 5

    def test_fractals_key_mapped(self):
        with open("src/telegram_bot.py", encoding="utf-8") as f:
            content = f.read()
        assert "'fractals'" in content
