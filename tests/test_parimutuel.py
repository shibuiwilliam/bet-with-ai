import math

import numpy as np
import pytest

from betai import s5_parimutuel as s5


@pytest.fixture(scope="module")
def races():
    return s5.make_races(3_000, seed=5)


def test_races_are_valid(races):
    p, mask, winner = races["p"], races["mask"], races["winner"]
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p[~mask] == 0).all()
    assert mask[np.arange(len(winner)), winner].all()
    assert np.allclose(races["r"].sum(axis=1), 1.0)


def test_growth_identity_holds_race_by_race(races):
    q = s5.model_probs(races, 0.4, seed=1)
    p, r, mask = races["p"], races["r"], races["mask"]
    t = s5.TAKEOUT
    o = s5.odds_from_shares(r, t, breakage=False)
    identity = math.log(1 - t) + s5.kl(p, r) - s5.kl(p, q)
    direct = np.where(mask, p * np.log(np.clip(q * o, 1e-300, None)), 0.0).sum(axis=1)
    assert np.allclose(identity, direct, atol=1e-10)


def test_betting_at_market_probabilities_loses_exactly_the_takeout(races):
    r, p, mask = races["r"], races["p"], races["mask"]
    o = s5.odds_from_shares(r, s5.TAKEOUT, breakage=False)
    w = np.where(mask, p * np.log(np.clip(r * o, 1e-300, None)), 0.0).sum(axis=1)
    assert np.allclose(w, math.log(1 - s5.TAKEOUT))


def test_breakage_rounds_down_to_ten_yen_with_a_floor_of_one():
    r = np.array([[0.9, 0.06, 0.04]])
    o = s5.odds_from_shares(r, 0.20)
    assert o[0, 0] == 1.0  # 0.8 / 0.9 < 1 is paid back at 1.0
    assert o[0, 1] == pytest.approx(13.3)  # 13.33 rounded down
    assert o[0, 2] == pytest.approx(20.0)


def test_kelly_bets_nothing_without_an_edge_and_never_more_than_the_bankroll():
    r = np.array([0.5, 0.3, 0.2])
    o = 0.8 / r
    assert s5.kelly_with_cash(r, o, 1.0).sum() == 0.0
    q = np.array([0.7, 0.2, 0.1])  # the favourite is underpriced
    b = s5.kelly_with_cash(q, o, 1.0)
    assert b[0] > 0 and b[1:].sum() == 0
    assert b.sum() <= 1.0
    assert s5.kelly_with_cash(q, o, 0.25)[0] == pytest.approx(b[0] / 4)


def test_capacity_is_positive_only_with_an_edge():
    stake, profit = s5.capacity(p=0.30, r=0.20, pool=10_000_000, t=0.20)  # p(1-t)/r = 1.2
    assert stake > 0 and profit > 0
    stake2, profit2 = s5.capacity(p=0.15, r=0.20, pool=10_000_000, t=0.20)  # 0.6
    assert profit2 == pytest.approx(0.0, abs=1.0)
    assert stake2 < 1_000
