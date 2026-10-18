import numpy as np
import pytest

from betai import s2_skill as s2


@pytest.fixture(scope="module")
def gto():
    strategy, _ = s2.cfr(3_000)
    return strategy


def test_payoffs_are_zero_sum_without_rake():
    for h in s2.TERMINAL:
        for c1, c2 in s2.DEALS:
            u1, u2 = s2.payoffs(h, c1, c2, 0.0)
            assert u1 + u2 == pytest.approx(0.0)


def test_rake_is_taken_from_the_pot():
    u1, u2 = s2.payoffs("bb", 2, 0, 0.10)  # pot of 4, P1 wins
    assert u1 == pytest.approx(4 * 0.9 - 2)
    assert u1 + u2 == pytest.approx(-0.4)


def test_cfr_converges_to_the_game_value(gto):
    v1, v2 = s2.expected_values(gto, gto)
    assert v1 == pytest.approx(-1 / 18, abs=0.005)
    assert v1 + v2 == pytest.approx(0.0, abs=1e-12)
    assert s2.exploitability(gto) < 0.01


def test_equilibrium_never_loses_on_average(gto):
    for opp in s2.OPPONENTS.values():
        ev, _ = s2.seat_average((gto, gto), opp)
        assert ev > -0.01


def test_best_response_beats_every_leaky_opponent(gto):
    for opp in s2.OPPONENTS.values():
        ev_br, _ = s2.seat_average(s2.exploiter(opp), opp)
        ev_gto, _ = s2.seat_average((gto, gto), opp)
        assert ev_br >= ev_gto - 1e-12
        assert ev_br > 0.1


def test_best_response_folds_jack_and_calls_king_against_an_always_betting_player():
    maniac = s2.OPPONENTS["マニアック（常にベット・コール）"]
    br = s2.best_response(maniac, br_player=1)
    assert br[("b", 0)] == 0.0  # fold J to a bet
    assert br[("b", 2)] == 1.0  # call with K


def test_shapley_values_add_up_to_the_total(gto):
    opp = s2.OPPONENTS["ロック（強い手以外は降りる）"]
    phi, total = s2.shapley_attribution(opp, gto)
    assert phi.sum() == pytest.approx(total, abs=1e-9)
    assert total > 0.2


def test_shapley_puts_all_profit_on_the_only_leak(gto):
    opp = dict(gto)
    leak = ("b", 1)  # always call a bet with Q
    opp[leak] = 1.0
    phi, total = s2.shapley_attribution(opp, gto)
    i = s2.INFOSETS.index(leak)
    assert phi[i] == pytest.approx(total)
    assert np.allclose(np.delete(phi, i), 0.0)


def test_placement_model_and_stable_dan():
    rates = s2.placement_rates(0.0, 1.0, n=200_000)
    assert np.allclose(rates, 0.25, atol=0.005)
    assert s2.stable_dan(np.full(4, 0.25)) == pytest.approx(5.0)
    assert s2.stable_dan(s2.SUPHX) == pytest.approx(8.78, abs=0.01)
    better = s2.placement_rates(0.3, 0.8, n=200_000)
    assert better[0] > 0.25 > better[3]
