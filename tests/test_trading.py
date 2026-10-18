import numpy as np
import pandas as pd
import pytest

from betai import s6_trading as s6


def make_market(n_days: int = 2_000, crash_on: int | None = None) -> pd.DataFrame:
    g = np.random.default_rng(0)
    dates = pd.bdate_range("1990-01-01", periods=n_days)
    ex = g.normal(0.0003, 0.01, n_days)
    if crash_on is not None:
        ex[crash_on] = -0.30
    return pd.DataFrame({"Mkt-RF": ex, "RF": 0.0001}, index=dates)


def make_industries(start: str = "1969-01-01", end: str = "1984-12-31", n: int = 12, seed: int = 0) -> pd.DataFrame:
    g = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    return pd.DataFrame(g.normal(0.0004, 0.012, (len(dates), n)), index=dates, columns=[f"I{i}" for i in range(n)])


def test_horizon_table_shape_and_range():
    rows = s6.horizon_table(make_market(n_days=6_000))  # longer than the 20-year window
    assert [r[0] for r in rows][:2] == ["1日", "1週"]
    for _, h, p_pos, _, n in rows:
        assert 0.0 <= p_pos <= 1.0
        assert n == 6_000 - h + 1


def test_leverage_wipes_out_when_a_single_day_exceeds_one_over_l():
    ff = make_market(crash_on=500)
    table = {row["L"]: row for row in s6.leverage_table(ff)}
    crash_day = ff.index[500].date().isoformat()
    assert table[3]["wiped_out"] == "—"  # 3 x 30% = 90%
    assert table[5]["wiped_out"] == crash_day  # 5 x 30% = 150%
    assert np.isinf(table[25]["cagr_excess_log"])


def test_day_trading_costs_lower_wealth():
    ff = make_market(n_days=1_500)
    cheap = s6.day_traders(ff, "1990-01-01", 500, 0.0, seed=1)
    costly = s6.day_traders(ff, "1990-01-01", 500, 0.001, seed=1)
    assert costly["median_multiple_vs_cash"] < cheap["median_multiple_vs_cash"]
    assert 0.0 <= costly["p_beat_bh"] <= 1.0


def test_industry_features_and_targets_have_no_look_ahead():
    ind = make_industries(end="1972-12-31")
    panel = s6.industry_panel(ind, 5)
    cut = 600  # change every return from this day on
    changed = ind.copy()
    changed.iloc[cut:] += 0.05
    panel2 = s6.industry_panel(changed, 5)
    reb = np.arange(260, len(ind) - 5, 5)
    features_known = reb < cut  # features use returns up to and including day t
    targets_known = reb + 5 < cut  # targets use days t+1 .. t+5
    assert np.array_equal(panel["X"][features_known], panel2["X"][features_known])
    assert np.array_equal(panel["fwd"][targets_known], panel2["fwd"][targets_known])
    assert not np.array_equal(panel["fwd"][~targets_known], panel2["fwd"][~targets_known])


def test_ml_signals_are_walk_forward():
    ind = make_industries()
    panel = s6.industry_panel(ind, 5)
    sig = s6.signals(panel, "ridge")
    years = panel["dates"].year
    assert np.isnan(sig[years < 1980]).all()
    assert not np.isnan(sig[years >= 1980]).any()
    changed = ind.copy()
    changed.loc["1983-01-01":] *= -3.0
    sig2 = s6.signals(s6.industry_panel(changed, 5), "ridge")
    known = (years >= 1980) & (years < 1983)
    assert np.array_equal(sig[known], sig2[known])


def test_backtest_turnover_and_costs():
    ind = make_industries(end="1972-12-31")
    panel = s6.industry_panel(ind, 5)
    constant = np.tile(np.arange(12, dtype=float), (len(panel["dates"]), 1))
    bt = s6.backtest(panel, constant, top_k=3)
    assert bt["turnover"][0] == pytest.approx(2.0)  # from cash into a full long-short book
    assert np.allclose(bt["turnover"][1:], 0.0)
    w = np.zeros(12)
    w[-3:], w[:3] = 1 / 3, -1 / 3
    assert np.allclose(bt["gross"], panel["fwd"] @ w)
    net = s6.net_of_costs(bt, 10)
    assert net[0] == pytest.approx(bt["gross"][0] - 2.0 * 10 / 1e4)
