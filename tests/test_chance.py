import numpy as np
import pytest

from betai import s1_chance as s1


@pytest.mark.parametrize("source", [s1.certified, s1.biased_wheel, s1.weak_prng])
def test_sources_produce_valid_pockets(source):
    seq = source(7, 5_000)
    assert seq.min() >= 0 and seq.max() < s1.N_POCKETS
    assert len(np.unique(seq)) == s1.N_POCKETS


def test_sources_are_reproducible():
    assert np.array_equal(s1.certified(3, 1000), s1.certified(3, 1000))
    assert np.array_equal(s1.weak_prng(3, 1000), s1.weak_prng(3, 1000))


def test_weak_prng_repeats_with_period_4096():
    seq = s1.weak_prng(11, 3 * 4096)
    assert np.array_equal(seq[:4096], seq[4096:8192])


def test_biased_wheel_favours_the_tilted_pockets():
    seq = s1.biased_wheel(0, 400_000)
    freq = np.bincount(seq, minlength=s1.N_POCKETS) / len(seq)
    tilted = freq[s1.WHEEL_NEIGHBOURS].mean()
    others = np.delete(freq, s1.WHEEL_NEIGHBOURS).mean()
    assert tilted / others == pytest.approx(1.10, rel=0.03)


def test_prob_above_rises_with_hits():
    low = s1.prob_above(hits=10, n=370, prior=0.5)
    high = s1.prob_above(hits=25, n=370, prior=0.5)
    assert 0 <= low < high <= 1


def test_mutual_information_separates_iid_from_predictable_sequences():
    # The weak PRNG's next pocket is almost determined by the previous two.
    iid, p_iid = s1.mutual_information(s1.certified(1, 20_000), 2, n_perm=20)
    lcg, p_lcg = s1.mutual_information(s1.weak_prng(1, 20_000), 2, n_perm=20)
    assert abs(iid) < 0.02 and p_iid > 0.05
    assert lcg > 1.0 and p_lcg <= 1 / 21 + 1e-12


def test_agent_learns_the_weak_prng_but_not_the_certified_rng():
    weak = s1.run_agent(s1.weak_prng(2, 30_000), context=2, confidence=0.99)
    fair = s1.run_agent(s1.certified(2, 30_000), context=0, confidence=0.99)
    assert weak["forced"][10_000:].mean() > 5  # wins far more than the 35:1 payout costs
    # A frequency model betting a certified RNG cannot expect more than the house edge.
    assert fair["forced"].mean() < 0.15


def test_martingale_wins_often_but_loses_per_unit_wagered():
    res = s1.roulette_sessions(3_000, seed=1)
    pnl, turnover = res["martingale (bank 255, target +20)"]
    assert (pnl > 0).mean() > 0.8
    assert pnl.sum() / turnover.sum() == pytest.approx(18 / 37 * 2 - 1, abs=0.02)


def test_kelly_policy_never_bets_on_a_negative_ev_slot():
    res = s1.slot_policies(200, seed=0, rounds=20)
    kelly = res["Kelly (EV<0, bet nothing)"]
    assert (kelly["final"] == 100).all() and kelly["turnover"].sum() == 0
    assert res["double after loss"]["ruined"].mean() > res["fixed $10"]["ruined"].mean()
