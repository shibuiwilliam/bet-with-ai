import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytest

from betai import s7_llm as s7


def test_parse_json_list_tolerates_surrounding_text():
    text = 'Here you go:\n```json\n[{"id": "m001", "home": 0.5, "draw": 0.3, "away": 0.2}]\n```'
    assert s7.parse_json_list(text) == [{"id": "m001", "home": 0.5, "draw": 0.3, "away": 0.2}]
    assert s7.parse_json_list("no list here") == []
    assert s7.parse_json_list("[not json]") == []


def test_to_probs_normalises_and_skips_bad_answers():
    df = pd.DataFrame({"mid": ["a", "b", "c", "d"]})
    answers = {
        "a": {"home": 2, "draw": 1, "away": 1},
        "b": {"home": "x", "draw": 1, "away": 1},
        "d": {"home": 1.0, "draw": 0.0, "away": 0.0},
    }
    p = s7.to_probs(df, answers)
    assert np.allclose(p[0], [0.5, 0.25, 0.25])
    assert np.isnan(p[1]).all() and np.isnan(p[2]).all()
    assert np.allclose(p[3].sum(), 1.0) and p[3].min() > 0  # zeros are floored so log-loss stays finite


def test_match_line_shows_odds_only_when_asked():
    r = (
        pd.DataFrame(
            [
                {
                    "mid": "m000",
                    "date": pd.Timestamp("2025-08-16"),
                    "home": "Arsenal",
                    "away": "Chelsea",
                    "avg_open_h": 1.9,
                    "avg_open_d": 3.6,
                    "avg_open_a": 4.25,
                }
            ]
        )
        .itertuples()
        .__next__()
    )
    plain = json.loads(s7.match_line(r))
    assert plain == {"id": "m000", "date": "2025-08-16", "home": "Arsenal", "away": "Chelsea"}
    with_odds = json.loads(s7.match_line(r, with_odds=True))
    assert with_odds["avg_odds_away"] == 4.25


def test_ask_reads_the_cache_without_calling_the_api(tmp_path, monkeypatch):
    monkeypatch.setattr(s7, "CACHE", tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    prompt = "hello"
    key = hashlib.sha256(f"{s7.MODEL}|0|{prompt}".encode()).hexdigest()[:24]
    (tmp_path / f"probe_{key}.json").write_text(json.dumps({"text": "cached answer"}))
    before = s7.USAGE["calls"]
    assert s7.ask(prompt, "probe") == "cached answer"
    assert s7.USAGE["calls"] == before


def test_api_key_is_read_only_from_the_repository_root(tmp_path, monkeypatch):
    # setenv first so that monkeypatch restores the original state after the test.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    root = tmp_path / "repo"
    root.mkdir()
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=parent-key\n")
    monkeypatch.setattr(s7, "ROOT", root)
    s7.load_env_key()
    assert "ANTHROPIC_API_KEY" not in os.environ
    (root / ".env").write_text('ANTHROPIC_API_KEY="repo-key"\n')
    s7.load_env_key()
    assert os.environ["ANTHROPIC_API_KEY"] == "repo-key"


@pytest.mark.parametrize("size", [3, 10])
def test_batches_cover_every_row_once(size):
    df = pd.DataFrame({"x": range(23)})
    parts = list(s7.batches(df, size))
    assert sum(len(b) for b in parts) == 23
    assert pd.concat(parts)["x"].tolist() == list(range(23))
