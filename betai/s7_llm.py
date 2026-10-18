"""Study 7 - a real LLM as the bettor (Claude Haiku 5.5 through the Anthropic API).

Premier League matches from 2022/23 to 2026/27 are given to the model in batches.
1. Recall: does the model remember final scores? (a direct test of look-ahead risk)
2. Blind forecast: home/draw/away probabilities from team names and date only.
3. Forecast with the market's opening odds shown, and value bets on disagreements.
4. Reproducibility: the blind forecasts are requested twice with identical prompts.

Responses are cached under data/llm_cache/, so re-running the study re-scores the
same answers without calling the API. Delete the cache to query the model again.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from . import data
from .common import DATA, ROOT, Report, brier, ci_str, log_loss, mean_ci, pct

MODEL = "claude-haiku-5-5"
SEASONS = {"2223": 100, "2324": 100, "2425": 100, "2526": 100, "2627": None}  # None = all matches so far
CACHE = DATA / "llm_cache"
LABEL = {"2223": "2022/23", "2324": "2023/24", "2425": "2024/25", "2526": "2025/26", "2627": "2026/27"}


def load_env_key() -> None:
    """Use ANTHROPIC_API_KEY from the environment, or from a .env file at the repository root."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"')
                return


def sample_matches() -> pd.DataFrame:
    df = data.load_football()
    df = df[(df["div"] == "E0") & df["season"].isin(SEASONS)].copy()
    parts = []
    for s, n in SEASONS.items():
        sub = df[df["season"] == s]
        if n is not None and len(sub) > n:
            sub = sub.sample(n, random_state=int(s))
        parts.append(sub)
    out = pd.concat(parts).sort_values("date").reset_index(drop=True)
    out["mid"] = [f"m{i:03d}" for i in range(len(out))]
    return out


# ---------------------------------------------------------------------------
# API with caching
# ---------------------------------------------------------------------------

USAGE = {"calls": 0, "cached": 0, "input_tokens": 0, "output_tokens": 0}


def ask(prompt: str, tag: str, run: int = 0) -> str:
    key = hashlib.sha256(f"{MODEL}|{run}|{prompt}".encode()).hexdigest()[:24]
    path = CACHE / f"{tag}_{key}.json"
    if path.exists():
        USAGE["cached"] += 1
        return json.loads(path.read_text())["text"]
    import anthropic

    client = anthropic.Anthropic()
    for attempt in range(4):
        try:
            r = client.messages.create(
                model=MODEL,
                max_tokens=4000,
                thinking={"type": "disabled"},
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except Exception as e:  # noqa: BLE001 - retry rate limits and transient errors
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
            print(f"  retry {tag}: {type(e).__name__}")
    text = "".join(b.text for b in r.content if b.type == "text")
    USAGE["calls"] += 1
    USAGE["input_tokens"] += r.usage.input_tokens
    USAGE["output_tokens"] += r.usage.output_tokens
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model": MODEL,
                "run": run,
                "prompt": prompt,
                "text": text,
                "usage": {"input": r.usage.input_tokens, "output": r.usage.output_tokens},
                "utc": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    return text


def parse_json_list(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return []


def batches(df: pd.DataFrame, size: int):
    for i in range(0, len(df), size):
        yield df.iloc[i : i + size]


def match_line(r, with_odds: bool = False) -> str:
    s = f'{{"id": "{r.mid}", "date": "{r.date.date()}", "home": "{r.home}", "away": "{r.away}"'
    if with_odds:
        s += f', "avg_odds_home": {r.avg_open_h:.2f}, "avg_odds_draw": {r.avg_open_d:.2f}, "avg_odds_away": {r.avg_open_a:.2f}'
    return s + "}"


RECALL_PROMPT = """Below are English Premier League matches (team names as written by football-data.co.uk).
For each match, give the final score only if you actually remember it from your training data.
If you are not sure, use null. Do not guess.

Matches:
{lines}

Answer with a JSON list only, one object per match:
[{{"id": "...", "home_goals": <int or null>, "away_goals": <int or null>}}, ...]"""

FORECAST_PROMPT = """You are a football forecaster. For each English Premier League match below, imagine it is the day before the match date.
Give your probabilities for a home win, a draw and an away win. Use only information that would have been available before kick-off;
do not use any knowledge of the actual result. The three probabilities must sum to 1.

Matches:
{lines}

Answer with a JSON list only, one object per match:
[{{"id": "...", "home": <prob>, "draw": <prob>, "away": <prob>}}, ...]"""

FORECAST_ODDS_PROMPT = """You are a football forecaster. For each English Premier League match below, imagine it is a few days before the match date.
The average bookmaker odds at that time are given (decimal odds, including the bookmakers' margin).
Give your own probabilities for a home win, a draw and an away win. You may use the odds as information, and you may disagree with them.
Use only information that would have been available before kick-off; do not use any knowledge of the actual result. The three probabilities must sum to 1.

Matches:
{lines}

Answer with a JSON list only, one object per match:
[{{"id": "...", "home": <prob>, "draw": <prob>, "away": <prob>}}, ...]"""


def run_probe(df: pd.DataFrame, template: str, tag: str, size: int, with_odds: bool, run: int = 0) -> dict:
    prompts = [
        template.format(lines="\n".join(match_line(r, with_odds) for r in b.itertuples())) for b in batches(df, size)
    ]
    with ThreadPoolExecutor(max_workers=4) as ex:
        texts = list(ex.map(lambda p: ask(p, tag, run), prompts))
    out = {}
    for t in texts:
        for item in parse_json_list(t):
            if isinstance(item, dict) and "id" in item:
                out[item["id"]] = item
    return out


def to_probs(df: pd.DataFrame, answers: dict) -> np.ndarray:
    p = np.full((len(df), 3), np.nan)
    for i, mid in enumerate(df["mid"]):
        a = answers.get(mid)
        if not a:
            continue
        try:
            v = np.array([float(a["home"]), float(a["draw"]), float(a["away"])])
        except (KeyError, TypeError, ValueError):
            continue
        if (v >= 0).all() and v.sum() > 0:
            p[i] = np.clip(v / v.sum(), 0.005, 1)
            p[i] /= p[i].sum()
    return p


# ---------------------------------------------------------------------------


def main() -> None:
    rep = Report("s7_llm", f"研究7　本物のLLMに賭けさせる：{MODEL} とプレミアリーグ")
    USAGE.update(calls=0, cached=0, input_tokens=0, output_tokens=0)  # count this run only
    load_env_key()
    have_cache = CACHE.exists() and any(CACHE.iterdir())
    if not os.environ.get("ANTHROPIC_API_KEY") and not have_cache:
        rep.p("ANTHROPIC_API_KEY が設定されておらず、キャッシュもないため、この研究は実行しなかった。")
        rep.save()
        return
    df = sample_matches()
    y = df["y"].to_numpy()
    rep.p(
        f"対象：イングランド・プレミアリーグの {len(df)} 試合（2022/23〜2025/26は各シーズン無作為に100試合、2026/27は10月12日までの全{(df['season'] == '2627').sum()}試合）。モデルは {MODEL}（思考なし、サンプリング設定はAPIの既定値。このSDKの版はtemperatureの指定を受け付けない）。試合はまとめて渡し（記憶の確認は20試合、予測は10試合ずつ）、回答はJSONで受け取る。モデル自身は学習データの終わりを「2025年初め頃」と答えた。"
    )

    recall = run_probe(df, RECALL_PROMPT, "recall", 20, False)
    blind = run_probe(df, FORECAST_PROMPT, "blind", 10, False)
    blind2 = run_probe(df, FORECAST_PROMPT, "blind", 10, False, run=1)
    with_odds = run_probe(df, FORECAST_ODDS_PROMPT, "odds", 10, True)

    # --- 1. recall
    rep.h2("1. 結果を覚えているか（先読みの直接テスト）")
    rows, rec = [], {}
    answered = np.zeros(len(df), dtype=bool)
    correct_res = np.zeros(len(df), dtype=bool)
    correct_score = np.zeros(len(df), dtype=bool)
    for i, r in enumerate(df.itertuples()):
        a = recall.get(r.mid) or {}
        hg, ag = a.get("home_goals"), a.get("away_goals")
        if isinstance(hg, int) and isinstance(ag, int):
            answered[i] = True
            res = 0 if hg > ag else 1 if hg == ag else 2
            correct_res[i] = res == r.y
            correct_score[i] = (hg == r.hg) and (ag == r.ag)
    for s in SEASONS:
        m = (df["season"] == s).to_numpy()
        n_ans = answered[m].sum()
        rows.append(
            [
                LABEL[s],
                m.sum(),
                pct(answered[m].mean(), 0, sign=False),
                pct(correct_score[m].sum() / max(n_ans, 1), 0, sign=False) if n_ans else "—",
                pct(correct_res[m].sum() / max(n_ans, 1), 0, sign=False) if n_ans else "—",
            ]
        )
        rec[s] = {
            "n": int(m.sum()),
            "answered": float(answered[m].mean()),
            "exact_given_answered": float(correct_score[m].sum() / max(n_ans, 1)),
            "result_given_answered": float(correct_res[m].sum() / max(n_ans, 1)),
        }
    rep.table(
        ["シーズン", "試合数", "スコアを答えた割合", "答えたうちスコアが正しい割合", "答えたうち勝敗が正しい割合"], rows
    )
    rep.p("勝敗を無作為に当てる確率は約3分の1、市場の本命を答え続けた場合は約5割。")
    rep.metric("recall", rec)

    # --- 2/3. forecasts
    rep.h2("2. 予測の精度：市場と比べる")
    p_close = data.devig(data.odds(df, "avg_close"))
    p_open = data.devig(data.odds(df, "avg_open"))
    pb, pb2, po = to_probs(df, blind), to_probs(df, blind2), to_probs(df, with_odds)
    rows, fc = [], {}
    for s in list(SEASONS) + ["all"]:
        m = np.ones(len(df), dtype=bool) if s == "all" else (df["season"] == s).to_numpy()
        m = m & ~np.isnan(pb).any(1) & ~np.isnan(po).any(1) & ~np.isnan(p_close).any(1)
        ll = {k: log_loss(v[m], y[m]) for k, v in (("blind", pb), ("odds", po), ("open", p_open), ("close", p_close))}
        diff = -np.log(pb[m][np.arange(m.sum()), y[m]]) + np.log(p_close[m][np.arange(m.sum()), y[m]])
        d, lo, hi = mean_ci(diff)
        rows.append(
            [
                LABEL.get(s, "全体"),
                m.sum(),
                f"{ll['blind']:.4f}",
                f"{ll['odds']:.4f}",
                f"{ll['open']:.4f}",
                f"{ll['close']:.4f}",
                f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]",
            ]
        )
        fc[s] = {
            "n": int(m.sum()),
            **ll,
            "blind_minus_close": [d, lo, hi],
            "brier_blind": brier(pb[m], y[m]),
            "brier_close": brier(p_close[m], y[m]),
        }
    rep.table(
        [
            "シーズン",
            "試合数",
            "LLM（オッズなし）",
            "LLM（オッズあり）",
            "市場（始値）",
            "市場（終値）",
            "LLM（オッズなし）− 終値 [95%CI]",
        ],
        rows,
    )
    rep.p(
        "log-lossが小さいほど良い。LLM（オッズなし）− 終値 がマイナスなら、LLMが市場より正確。知識の終わりより前のシーズンで、LLMが市場を大きく上回るなら、それは予測力ではなく記憶（先読み）を疑うべき兆候になる。"
    )
    rep.metric("forecast", fc)

    # Does recall leak into the forecast?
    m_rec = answered & correct_res & ~np.isnan(pb).any(1)
    m_not = ~answered & ~np.isnan(pb).any(1)
    leak_rows = []
    for label, m in (("勝敗を正しく思い出せた試合", m_rec), ("思い出せなかった試合", m_not)):
        if m.sum() == 0:
            continue
        leak_rows.append(
            [
                label,
                m.sum(),
                f"{pb[m][np.arange(m.sum()), y[m]].mean():.3f}",
                f"{p_close[m][np.arange(m.sum()), y[m]].mean():.3f}",
            ]
        )
    rep.h3("記憶が予測に漏れているか")
    rep.table(["", "試合数", "LLMが実際の結果につけた確率（平均）", "市場（終値）がつけた確率（平均）"], leak_rows)
    rep.p(
        "「結果を使わないで」と指示しても、思い出せた試合で実際の結果に市場より高い確率をつけていれば、記憶が予測に漏れている。"
    )

    # --- value bets with the odds-aware forecasts
    rep.h2("3. LLMの確率で賭ける")
    rep.p("オッズを見せた予測で、LLMの確率 × 最高値の始値 > 1.05 の結果に1単位ずつ賭ける（1試合1点まで）。")
    o_max = data.odds(df, "max_open")
    rows, vb = [], {}
    for s in list(SEASONS) + ["all"]:
        m = np.ones(len(df), dtype=bool) if s == "all" else (df["season"] == s).to_numpy()
        ev = np.where(np.isnan(po), -np.inf, po * o_max)
        j = ev.argmax(1)
        take = m & (ev[np.arange(len(df)), j] > 1.05)
        if take.sum() == 0:
            rows.append([LABEL.get(s, "全体"), 0, "—", "—", "—", "—"])
            continue
        o = o_max[take, j[take]]
        pnl = np.where(y[take] == j[take], o - 1, -1.0)
        clv = o * p_close[take, j[take]] - 1
        roi, lo, hi = mean_ci(pnl)
        rows.append(
            [
                LABEL.get(s, "全体"),
                int(take.sum()),
                pct((pnl > 0).mean(), 0, sign=False),
                pct(roi),
                ci_str(lo, hi),
                pct(clv.mean()),
            ]
        )
        vb[s] = {"bets": int(take.sum()), "roi": roi, "roi_ci": [lo, hi], "clv": float(clv.mean())}
    rep.table(["シーズン", "賭け数", "的中率", "ROI", "95%CI", "CLV（終値で評価した期待値）"], rows)
    rep.p(
        "数百試合では、ROIの信頼区間が広すぎて優位の有無は判定できない。CLVは、取った価格が試合直前の市場より有利だったかを示し、ROIよりはるかに少ない標本で傾向がわかる。"
    )
    rep.metric("value_bets", vb)

    # --- 4. reproducibility
    rep.h2("4. 同じ質問を2回した結果")
    m = ~np.isnan(pb).any(1) & ~np.isnan(pb2).any(1)
    mad = np.abs(pb[m] - pb2[m]).mean()
    max_d = np.abs(pb[m] - pb2[m]).max()
    flip = (pb[m].argmax(1) != pb2[m].argmax(1)).mean()
    identical = (np.abs(pb[m] - pb2[m]).max(axis=1) < 1e-9).mean()
    rep.table(
        ["指標", "値"],
        [
            ["比較できた試合", f"{m.sum()}"],
            ["確率が完全に一致した試合の割合", pct(identical, 1, sign=False)],
            ["確率の差の平均（絶対値）", f"{mad:.4f}"],
            ["確率の差の最大", f"{max_d:.3f}"],
            ["最も高い確率の結果が入れ替わった割合", pct(flip, 1, sign=False)],
            ["log-loss（1回目／2回目）", f"{log_loss(pb[m], y[m]):.4f}／{log_loss(pb2[m], y[m]):.4f}"],
        ],
    )
    rep.metric(
        "reproducibility",
        {"n": int(m.sum()), "identical_share": identical, "mean_abs_diff": mad, "max_diff": max_d, "argmax_flip": flip},
    )
    parsed = {k: len(v) for k, v in (("recall", recall), ("blind", blind), ("blind_run2", blind2), ("odds", with_odds))}
    files = [json.loads(f.read_text()) for f in CACHE.glob("*.json")]
    tok_in = sum(f["usage"]["input"] for f in files)
    tok_out = sum(f["usage"]["output"] for f in files)
    first = min(f["utc"] for f in files)
    rep.p(
        f"API呼び出しは計 {len(files)} 回（入力 {tok_in:,} トークン、出力 {tok_out:,} トークン、{first[:10]} に実行）。今回の実行では新規 {USAGE['calls']} 回、キャッシュから {USAGE['cached']} 回。回答を解析できた試合数：{parsed}（460試合中）。"
    )
    rep.metric(
        "usage",
        {
            "calls_total": len(files),
            "input_tokens": tok_in,
            "output_tokens": tok_out,
            "first_call_utc": first,
            "new_calls_this_run": USAGE["calls"],
            "parsed": parsed,
        },
    )
    rep.save()


if __name__ == "__main__":
    main()
