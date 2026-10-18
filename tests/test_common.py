import json
import math

import numpy as np
import pytest

from betai import common


def test_mean_ci_contains_mean_and_shrinks_with_n():
    g = np.random.default_rng(0)
    small = g.normal(0.1, 1.0, 100)
    large = g.normal(0.1, 1.0, 10_000)
    m, lo, hi = common.mean_ci(small)
    assert lo < m < hi
    _, lo2, hi2 = common.mean_ci(large)
    assert (hi2 - lo2) < (hi - lo) / 5


def test_required_n_matches_the_even_odds_rule_of_thumb():
    # Even odds (o = 2) with a 1% edge: about 6.18 * (o - 1) / e^2 = 61,800 bets.
    p = 1.01 / 2.0
    sd = 2.0 * math.sqrt(p * (1 - p))
    assert common.required_n(0.01, sd) == pytest.approx(61_800, rel=0.01)
    assert math.isinf(common.required_n(0.0, sd))
    assert math.isinf(common.required_n(-0.01, sd))


def test_log_loss_and_brier_of_perfect_and_uniform_forecasts():
    y = np.array([0, 1, 2, 0])
    perfect = np.eye(3)[y]
    uniform = np.full((4, 3), 1 / 3)
    assert common.log_loss(perfect, y) == pytest.approx(0.0, abs=1e-9)
    assert common.log_loss(uniform, y) == pytest.approx(math.log(3))
    assert common.brier(perfect, y) == pytest.approx(0.0)
    assert common.brier(uniform, y) == pytest.approx(2 / 3)


def test_max_drawdown():
    assert common.max_drawdown([1, 2, 1, 3, 1.5]) == pytest.approx(-0.5)
    assert common.max_drawdown([1, 2, 3]) == 0.0


def test_bootstrap_ci_is_reproducible():
    x = np.random.default_rng(1).normal(size=500)
    assert common.bootstrap_ci(x, seed=3) == common.bootstrap_ci(x, seed=3)


def test_deflated_sharpe_penalises_many_trials():
    single = common.deflated_sharpe_prob(0.1, 500, 1, 0.01)
    many = common.deflated_sharpe_prob(0.1, 500, 100, 0.01)
    assert single > 0.95
    assert many < single
    # A zero Sharpe ratio tested once is a coin flip.
    assert common.deflated_sharpe_prob(0.0, 500, 1, 0.01) == pytest.approx(0.5)


def test_formatting_helpers():
    assert common.pct(0.0123) == "+1.23%"
    assert common.pct(0.5, 0, sign=False) == "50%"
    assert common.pct(float("nan")) == "—"
    assert common.num(float("inf")) == "∞"
    assert common.num(12345.6) == "12,346"


def test_report_writes_markdown_and_json(isolated_outputs):
    rep = common.Report("unit", "title")
    rep.h2("section")
    rep.table(["a", "b"], [[1, 2]])
    rep.metric("x", np.float64(1.5))
    rep.save()
    assert "| a | b |" in (isolated_outputs / "unit.md").read_text()
    assert json.loads((isolated_outputs / "unit.json").read_text()) == {"x": 1.5}
