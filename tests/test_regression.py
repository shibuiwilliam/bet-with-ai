"""Re-run each study into a temporary directory and compare the Markdown and
JSON with the published files in outputs/. Any change in the numbers fails.

These tests take about a minute in total. Run the quick tests only with
`uv run pytest -m "not slow"`.
"""

import pytest

from betai import (
    s1_chance,
    s2_skill,
    s3_sports,
    s4_house_limits,
    s5_parimutuel,
    s6_trading,
    s7_llm,
    s8_synthesis,
)

from conftest import assert_same_as_published, copy_published, needs_football, needs_french

pytestmark = pytest.mark.slow


def test_study_1_chance(isolated_outputs):
    s1_chance.main()
    assert_same_as_published("s1_chance", isolated_outputs)


def test_study_2_skill(isolated_outputs):
    s2_skill.main()
    assert_same_as_published("s2_skill", isolated_outputs)


@needs_football
def test_study_3_sports(isolated_outputs):
    s3_sports.main()
    assert_same_as_published("s3_sports", isolated_outputs)


@needs_football
def test_study_4_house_limits(isolated_outputs):
    copy_published(["s3_sports"], isolated_outputs)
    s4_house_limits.main()
    assert_same_as_published("s4_house_limits", isolated_outputs)


def test_study_5_parimutuel(isolated_outputs):
    s5_parimutuel.main()
    assert_same_as_published("s5_parimutuel", isolated_outputs)


@needs_french
def test_study_6_trading(isolated_outputs):
    s6_trading.main()
    assert_same_as_published("s6_trading", isolated_outputs)


@needs_football
def test_study_7_llm_rescores_the_cached_answers(isolated_outputs, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused")  # every prompt must come from data/llm_cache/
    s7_llm.main()
    assert s7_llm.USAGE["calls"] == 0
    assert_same_as_published("s7_llm", isolated_outputs)


def test_study_8_synthesis(isolated_outputs):
    copy_published(
        [
            f"s{i}_{n}"
            for i, n in [
                (1, "chance"),
                (2, "skill"),
                (3, "sports"),
                (4, "house_limits"),
                (5, "parimutuel"),
                (6, "trading"),
                (7, "llm"),
            ]
        ],
        isolated_outputs,
    )
    s8_synthesis.main()
    assert_same_as_published("s8_synthesis", isolated_outputs)
