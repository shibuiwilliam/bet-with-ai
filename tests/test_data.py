import numpy as np
import pandas as pd
import pytest

from betai import data


@pytest.fixture
def odds():
    return np.array([[1.5, 4.0, 7.0], [2.1, 3.3, 3.6], [1.2, 6.5, 15.0]])


@pytest.mark.parametrize("method", ["power", "multiplicative"])
def test_devig_returns_probabilities(odds, method):
    p = data.devig(odds, method)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p > 0).all()


def test_power_devig_takes_more_margin_from_longshots(odds):
    mult = data.devig(odds, "multiplicative")
    power = data.devig(odds, "power")
    favourite, longshot = odds.argmin(axis=1), odds.argmax(axis=1)
    rows = np.arange(len(odds))
    assert (power[rows, longshot] < mult[rows, longshot]).all()
    assert (power[rows, favourite] > mult[rows, favourite]).all()


def test_devig_of_fair_odds_is_the_identity():
    p = np.array([[0.5, 0.3, 0.2]])
    for method in ("power", "multiplicative"):
        assert np.allclose(data.devig(1 / p, method), p, atol=1e-9)


def test_devig_propagates_missing_odds():
    p = data.devig(np.array([[2.0, np.nan, 3.0]]))
    assert np.isnan(p).all()


def test_parse_dates_accepts_both_year_formats():
    s = pd.Series(["11/08/2023", "05/08/17", "bad"])
    out = data._parse_dates(s)
    assert out.iloc[0] == pd.Timestamp("2023-08-11")
    assert out.iloc[1] == pd.Timestamp("2017-08-05")
    assert pd.isna(out.iloc[2])


def test_parse_french_block_reads_the_named_section_in_fractions():
    text = "\n".join(
        [
            "Some preamble",
            "  Average Equal Weighted Returns -- Daily",
            ",A,B",
            "19700102,   9.00,   9.00",
            "",
            "  Average Value Weighted Returns -- Daily",
            ",A,B",
            "19700102,   1.00,  -99.99",
            "19700105,  -0.50,   0.25",
            "",
            "Copyright",
        ]
    )
    df = data._parse_french_block(text, "Average Value Weighted Returns -- Daily")
    assert list(df.columns) == ["A", "B"]
    assert df.loc["1970-01-02", "A"] == pytest.approx(0.01)
    assert np.isnan(df.loc["1970-01-02", "B"])
    assert df.loc["1970-01-05", "B"] == pytest.approx(0.0025)
    assert len(df) == 2


def test_end_dates_match_the_published_run():
    assert pd.Timestamp("2026-10-14") == data.FOOTBALL_END
    assert pd.Timestamp("2026-08-31") == data.FRENCH_END
