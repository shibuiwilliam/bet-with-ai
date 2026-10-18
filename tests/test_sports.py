import numpy as np
import pandas as pd
import pytest

from betai import s3_sports as s3


def make_matches(n_days: int = 20, seed: int = 0) -> pd.DataFrame:
    """A small league: 8 teams, 4 matches every week."""
    g = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(8)]
    rows = []
    for d in range(n_days):
        date = pd.Timestamp("2020-01-04") + pd.Timedelta(days=7 * d)
        order = g.permutation(8)
        for k in range(4):
            rows.append(
                {
                    "season": "1920",
                    "div": "E0",
                    "date": date,
                    "home": teams[order[2 * k]],
                    "away": teams[order[2 * k + 1]],
                    "hg": float(g.poisson(1.5)),
                    "ag": float(g.poisson(1.1)),
                    "hs": float(g.poisson(13)),
                    "as_": float(g.poisson(11)),
                    "hst": float(g.poisson(5)),
                    "ast": float(g.poisson(4)),
                }
            )
    return pd.DataFrame(rows)


def test_features_do_not_use_results_from_the_same_day():
    df = make_matches()
    base = s3.build_features(df)
    changed = df.copy()
    last = changed["date"] == changed["date"].max()
    changed.loc[last, ["hg", "ag"]] = [9.0, 0.0]
    after = s3.build_features(changed)
    pd.testing.assert_frame_equal(base, after)


def test_features_only_change_after_the_modified_match_day():
    df = make_matches()
    day = df["date"].unique()[10]
    changed = df.copy()
    changed.loc[changed["date"] == day, ["hg", "ag", "hs", "as_"]] = [7.0, 0.0, 30.0, 1.0]
    base, after = s3.build_features(df), s3.build_features(changed)
    before_or_on = (df["date"] <= day).to_numpy()
    pd.testing.assert_frame_equal(base[before_or_on], after[before_or_on])
    assert not np.allclose(base[~before_or_on].to_numpy(), after[~before_or_on].to_numpy())


def test_value_bets_pick_the_best_outcome_above_the_threshold():
    q = np.array([[0.5, 0.3, 0.2], [0.4, 0.3, 0.3], [0.6, 0.2, 0.2]])
    o = np.array([[2.1, 3.0, 6.0], [2.0, 3.0, np.nan], [1.5, 4.0, 4.0]])
    take, j = s3.value_bets(q, o, tau=0.05)
    assert take.tolist() == [True, False, False]
    assert j[0] == 2  # 0.2 * 6.0 = 1.2 is the largest value
    take0, _ = s3.value_bets(q, o, tau=0.0)
    assert take0[0]


def test_kaunitz_rule_compares_the_best_price_with_the_consensus():
    frame = {
        "avg_open": np.array([[2.0, 3.5, 4.0], [2.0, 3.5, 4.0]]),
        "max_open": np.array([[2.3, 3.6, 4.1], [2.05, 3.6, 4.1]]),
    }
    take, j = s3.kaunitz(frame, np.array([0, 1]), alpha=0.05)
    # Row 0: (1/2.0 - 0.05) * 2.3 = 1.035 > 1. Row 1: 0.45 * 2.05 < 1.
    assert take.tolist() == [True, False]
    assert j[0] == 0


def test_benter_blend_puts_no_weight_on_a_model_that_adds_nothing():
    g = np.random.default_rng(0)
    logits = g.normal(size=(20_000, 3))
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([g.choice(3, p=row) for row in p])
    noisy = np.exp(logits + g.normal(0, 1.0, logits.shape))
    noisy /= noisy.sum(axis=1, keepdims=True)
    a, b = s3.fit_blend(noisy, p, y)  # the market is the truth here
    assert abs(a) < 0.05
    assert b == pytest.approx(1.0, abs=0.08)
    blended = s3.apply_blend(noisy, p, (a, b))
    assert np.allclose(blended.sum(axis=1), 1.0)


def test_expected_value_decomposition_is_exact():
    frame = {
        "p_close": np.array([[0.5, 0.3, 0.2]] * 3),
        "avg_open": np.array([[1.9, 3.2, 4.8]] * 3),
        "avg_close": np.array([[1.85, 3.3, 4.9]] * 3),
    }
    rows, j = np.array([0, 1, 2]), np.array([0, 1, 2])
    odds_taken = np.array([2.0, 3.4, 5.2])
    y = np.array([0, 2, 2])
    st = s3.bet_stats(rows, j, odds_taken, y, frame)
    expected_log = np.log(odds_taken * frame["p_close"][rows, j]).mean()
    assert st["shopping"] + st["timing"] + st["margin"] == pytest.approx(expected_log)
    assert st["bets"] == 3
    assert st["hit_rate"] == pytest.approx(2 / 3)
    assert st["roi"] == pytest.approx(((2.0 - 1) - 1 + (5.2 - 1)) / 3)
