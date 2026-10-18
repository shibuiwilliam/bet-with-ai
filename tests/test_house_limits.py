import json

import numpy as np
import pandas as pd
import pytest

from betai import s4_house_limits as s4


@pytest.fixture
def small_world(isolated_outputs, monkeypatch):
    """A small population, synthetic odds and the per-bet EVs that study 3 would provide."""
    g = np.random.default_rng(0)
    odds = g.uniform(1.2, 12.0, (5_000, 3))
    fake = pd.DataFrame({"avg_open_h": odds[:, 0], "avg_open_d": odds[:, 1], "avg_open_a": odds[:, 2]})
    monkeypatch.setattr(s4.data, "load_football", lambda: fake)
    monkeypatch.setattr(s4, "N_ACCOUNTS", 4_000)
    monkeypatch.setattr(s4, "MAX_BETS", 400)
    monkeypatch.setattr(s4, "CHECKPOINTS", (50, 100, 200, 400))
    (isolated_outputs / "s3_sports.json").write_text(
        json.dumps(
            {
                "value_bets": {"GBM|b365_open|0.05": {"clv": -0.06}},
                "kaunitz": {"open|0.03": {"clv": 0.04}},
                "sharp_vs_soft": {"b365_open|0.0": {"clv": -0.005}},
            }
        )
    )


def test_bettor_types_use_study_3_values(small_world):
    types = s4.bettor_types()
    assert sum(t["share"] for t in types.values()) == pytest.approx(1.0)
    evs = [t["ev"] for t in types.values()]
    assert -0.06 in evs and 0.04 in evs
    assert max(evs) >= 0.02  # the sharp type keeps at least a 2% edge


def test_winners_are_restricted_early_and_restricted_accounts_are_profitable(small_world):
    res = s4.simulate(seed=1, restrict=True)
    names, kind = res["names"], res["kind"]
    positive = np.isin(kind, [k for k, n in enumerate(names) if res["types"][n]["ev"] > 0])
    assert res["restricted"][positive].mean() > 0.9
    assert np.median(res["n_bets"][positive & res["restricted"]]) < 200
    profit_rate_restricted = (res["profit"][res["restricted"]] > 0).mean()
    assert profit_rate_restricted > (res["profit"] > 0).mean()


def test_without_limits_every_record_runs_to_the_end(small_world):
    res = s4.simulate(seed=1, restrict=False)
    assert not res["restricted"].any()
    assert (res["n_bets"] == s4.MAX_BETS).all()
    assert np.allclose(res["at"][400], res["profit"])
