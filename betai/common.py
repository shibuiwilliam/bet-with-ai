"""Shared helpers for every study: paths, statistics and report output.

Every study writes two files to outputs/: a human-readable Markdown file and
a JSON file with the headline metrics that the synthesis step (s8) reads.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import font_manager
from scipy import stats

# Figures are only saved to files, so use the non-interactive backend in every study.
matplotlib.use("Agg")
# Use a Japanese-capable font for figure labels when one is installed (macOS ships Hiragino).
_JP_FONTS = [
    f.name
    for f in font_manager.fontManager.ttflist
    if f.name in ("Hiragino Sans", "Hiragino Maru Gothic Pro", "Noto Sans CJK JP")
]
if _JP_FONTS:
    matplotlib.rcParams["font.family"] = [_JP_FONTS[0], "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
FIG = OUT / "figures"

Z_ALPHA = stats.norm.ppf(0.95)  # one-sided 5%
Z_BETA = stats.norm.ppf(0.80)  # 80% power


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def mean_ci(x, level: float = 0.95) -> tuple[float, float, float]:
    """Mean and normal-approximation confidence interval."""
    x = np.asarray(x, dtype=float)
    m = float(x.mean())
    if len(x) < 2:
        return m, float("nan"), float("nan")
    se = float(x.std(ddof=1) / math.sqrt(len(x)))
    z = stats.norm.ppf(0.5 + level / 2)
    return m, m - z * se, m + z * se


def t_stat(x) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def p_value_one_sided(x) -> float:
    """P-value for H0: mean <= 0 against H1: mean > 0."""
    t = t_stat(x)
    if math.isnan(t):
        return float("nan")
    return float(stats.norm.sf(t))


def bootstrap_ci(x, n_boot: int = 2000, level: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    g = np.random.default_rng(seed)
    idx = g.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def required_n(mean: float, sd: float) -> float:
    """Samples needed to detect a mean `mean` with per-sample SD `sd`
    (one-sided alpha 5%, power 80%). Returns inf when mean <= 0."""
    if mean <= 0 or sd <= 0:
        return float("inf")
    return ((Z_ALPHA + Z_BETA) * sd / mean) ** 2


def log_loss(prob: np.ndarray, outcome: np.ndarray) -> float:
    """Mean negative log-likelihood (nats) of the realised outcome index."""
    p = np.clip(prob[np.arange(len(outcome)), outcome], 1e-12, 1.0)
    return float(-np.log(p).mean())


def brier(prob: np.ndarray, outcome: np.ndarray) -> float:
    onehot = np.zeros_like(prob)
    onehot[np.arange(len(outcome)), outcome] = 1.0
    return float(((prob - onehot) ** 2).sum(axis=1).mean())


def sharpe(returns, periods_per_year: float) -> float:
    r = np.asarray(returns, dtype=float)
    if r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * math.sqrt(periods_per_year))


def max_drawdown(wealth) -> float:
    w = np.asarray(wealth, dtype=float)
    peak = np.maximum.accumulate(w)
    return float((w / peak - 1).min())


def deflated_sharpe_prob(
    sr_hat: float, n_obs: int, n_trials: int, sr_var: float, skew: float = 0.0, kurt: float = 3.0
) -> float:
    """Probability that the true Sharpe exceeds the best-of-N luck benchmark
    (Bailey & Lopez de Prado). Sharpe values are per observation."""
    emc = 0.5772156649
    if n_trials > 1:
        sr0 = math.sqrt(sr_var) * (
            (1 - emc) * stats.norm.ppf(1 - 1 / n_trials) + emc * stats.norm.ppf(1 - 1 / (n_trials * math.e))
        )
    else:
        sr0 = 0.0
    denom = math.sqrt(max(1e-12, 1 - skew * sr_hat + (kurt - 1) / 4 * sr_hat**2))
    return float(stats.norm.cdf((sr_hat - sr0) * math.sqrt(n_obs - 1) / denom))


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def pct(x: float, digits: int = 2, sign: bool = True) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x * 100:+.{digits}f}%" if sign else f"{x * 100:.{digits}f}%"


def num(x: float, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—" if (x is None or math.isnan(x)) else "∞"
    if abs(x) >= 1e4:
        return f"{x:,.0f}"
    return f"{x:.{digits}f}"


def ci_str(lo: float, hi: float, digits: int = 2) -> str:
    return f"[{pct(lo, digits)}, {pct(hi, digits)}]"


class Report:
    """Collects Markdown text and JSON metrics for one study."""

    def __init__(self, name: str, title: str):
        self.name = name
        self.lines: list[str] = [f"# {title}", ""]
        self.metrics: dict = {}

    def h2(self, text: str) -> None:
        self.lines += [f"## {text}", ""]

    def h3(self, text: str) -> None:
        self.lines += [f"### {text}", ""]

    def p(self, text: str) -> None:
        self.lines += [text, ""]

    def bullets(self, items: list[str]) -> None:
        self.lines += [f"- {i}" for i in items] + [""]

    def table(self, headers: list[str], rows: list[list]) -> None:
        self.lines.append("| " + " | ".join(headers) + " |")
        self.lines.append("|" + "|".join("---" for _ in headers) + "|")
        for r in rows:
            self.lines.append("| " + " | ".join(str(c) for c in r) + " |")
        self.lines.append("")

    def figure(self, path: Path, caption: str) -> None:
        rel = path.relative_to(OUT)
        self.lines += [f"![{caption}]({rel.as_posix()})", ""]

    def metric(self, key: str, value) -> None:
        self.metrics[key] = value

    def save(self) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{self.name}.md").write_text("\n".join(self.lines), encoding="utf-8")
        (OUT / f"{self.name}.json").write_text(
            json.dumps(self.metrics, ensure_ascii=False, indent=2, default=_jsonable), encoding="utf-8"
        )
        print(f"wrote outputs/{self.name}.md and .json")


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def savefig(fig, name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / f"{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    return path
