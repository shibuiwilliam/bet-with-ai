"""Study 3 - opponents: the house and the market, on real football data.

Data: about 60,000 matches from 18 European divisions, 2017/18-2026/27
(football-data.co.uk), with opening and closing odds from many bookmakers.

1. How good is the market's own forecast? (calibration, opening vs closing,
   favourite-longshot bias)
2. Machine-learning models (logistic regression and gradient boosting on Elo,
   goals and shots), trained walk-forward, plus a Benter-style blend with the market.
   Do they beat the market's forecast?
3. Value betting with each model at the best available price, a single bookmaker
   and the average price. ROI, closing line value, and a decomposition of every
   bet's expected value into margin, timing and line shopping.
4. A market-only rule in the style of Kaunitz et al. (2017).
"""

from __future__ import annotations

import math
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import data
from .common import Report, brier, ci_str, log_loss, mean_ci, num, p_value_one_sided, pct, required_n, savefig, t_stat

TEST_SEASONS = ["2122", "2223", "2324", "2425", "2526", "2627"]
OOS_SEASONS = ["2021"] + TEST_SEASONS  # seasons with out-of-sample model predictions
SEASON_LABEL = {s: f"20{s[:2]}/{s[2:]}" for s in data.WARMUP_SEASONS + data.ODDS_SEASONS}
OUTCOMES = ["H", "D", "A"]


# ---------------------------------------------------------------------------
# Features computed only from matches played before each kick-off
# ---------------------------------------------------------------------------


def country(div: str) -> str:
    return "".join(ch for ch in div if ch.isalpha())


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Elo ratings and exponentially weighted form, updated after each match day."""
    elo = defaultdict(lambda: 1500.0)
    stats_keys = ["gf", "ga", "sf", "sa", "tf", "ta", "pts"]
    prior = {"gf": 1.35, "ga": 1.35, "sf": 12.0, "sa": 12.0, "tf": 4.3, "ta": 4.3, "pts": 1.35}
    form = defaultdict(lambda: dict(prior))
    played = defaultdict(int)
    alpha, home_adv, k_base = 0.12, 60.0, 20.0

    feats = np.zeros((len(df), 2 + 2 * len(stats_keys) + 1))
    rows = df.reset_index(drop=True)
    # Group by date so that results from the same day never leak into each other.
    for _, idx in sorted(rows.groupby("date").groups.items()):
        updates = []
        for i in idx:
            r = rows.loc[i]
            c = country(r["div"])
            h, a = (c, r["home"]), (c, r["away"])
            diff = (elo[h] + home_adv - elo[a]) / 400.0
            fh, fa = form[h], form[a]
            feats[i, 0] = diff
            feats[i, 1] = min(played[h], played[a])
            for j, key in enumerate(stats_keys):
                feats[i, 2 + j] = fh[key]
                feats[i, 2 + len(stats_keys) + j] = fa[key]
            feats[i, -1] = 1.0 if r["div"] in ("E0", "D1", "I1", "SP1", "F1") else 0.0
            updates.append((i, h, a, diff))
        for i, h, a, diff in updates:
            r = rows.loc[i]
            hg, ag = r["hg"], r["ag"]
            if np.isnan(hg) or np.isnan(ag):
                continue
            exp_home = 1 / (1 + 10 ** (-diff))
            score = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
            k = k_base * (1 + math.log1p(abs(hg - ag)))
            elo[h] += k * (score - exp_home)
            elo[a] -= k * (score - exp_home)
            obs_h = {
                "gf": hg,
                "ga": ag,
                "sf": r["hs"],
                "sa": r["as_"],
                "tf": r["hst"],
                "ta": r["ast"],
                "pts": 3 if hg > ag else 1 if hg == ag else 0,
            }
            obs_a = {
                "gf": ag,
                "ga": hg,
                "sf": r["as_"],
                "sa": r["hs"],
                "tf": r["ast"],
                "ta": r["hst"],
                "pts": 3 if ag > hg else 1 if hg == ag else 0,
            }
            for team, obs in ((h, obs_h), (a, obs_a)):
                for key in stats_keys:
                    v = obs[key]
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        form[team][key] = (1 - alpha) * form[team][key] + alpha * v
                played[team] += 1
    cols = ["elo_diff", "min_played"] + [f"h_{k}" for k in stats_keys] + [f"a_{k}" for k in stats_keys] + ["top_div"]
    out = pd.DataFrame(feats, columns=cols)
    for k in stats_keys:
        out[f"d_{k}"] = out[f"h_{k}"] - out[f"a_{k}"]
    return out


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def fit_models(X: np.ndarray, y: np.ndarray, seed: int) -> dict:
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    lr.fit(X, y)
    gbm = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=15,
        min_samples_leaf=200,
        l2_regularization=1.0,
        random_state=seed,
    )
    gbm.fit(X, y)
    elo_only = LogisticRegression(max_iter=2000).fit(X[:, :1], y)
    return {"Elo (LR)": ("elo", elo_only), "LR": ("all", lr), "GBM": ("all", gbm)}


def predict(models: dict, X: np.ndarray) -> dict:
    out = {}
    for name, (cols, m) in models.items():
        out[name] = m.predict_proba(X[:, :1] if cols == "elo" else X)
    return out


def fit_blend(p_model: np.ndarray, p_market: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Benter's combination p ~ p_model^a * p_market^b, fitted by maximum likelihood."""
    lm, lk = np.log(np.clip(p_model, 1e-9, 1)), np.log(np.clip(p_market, 1e-9, 1))

    def nll(w):
        z = w[0] * lm + w[1] * lk
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return -np.log(p[np.arange(len(y)), y]).mean()

    res = optimize.minimize(nll, x0=[0.2, 0.8], method="Nelder-Mead")
    return float(res.x[0]), float(res.x[1])


def apply_blend(p_model, p_market, w):
    z = w[0] * np.log(np.clip(p_model, 1e-9, 1)) + w[1] * np.log(np.clip(p_market, 1e-9, 1))
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Betting
# ---------------------------------------------------------------------------


def value_bets(q: np.ndarray, o: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """At most one bet per match: the outcome with the highest q*o, if q*o > 1 + tau."""
    ev = q * o
    ev = np.where(np.isnan(ev), -np.inf, ev)
    j = ev.argmax(axis=1)
    take = ev[np.arange(len(ev)), j] > 1 + tau
    return take, j


def bet_stats(sel_rows: np.ndarray, j: np.ndarray, odds_taken: np.ndarray, y: np.ndarray, frame: dict) -> dict:
    """ROI, CLV and the decomposition for a set of flat 1-unit bets."""
    n = len(sel_rows)
    if n == 0:
        return {"bets": 0}
    o = odds_taken
    won = y[sel_rows] == j
    pnl = np.where(won, o - 1, -1.0)
    p_close = frame["p_close"][sel_rows, j]
    a_open = frame["avg_open"][sel_rows, j]
    a_close = frame["avg_close"][sel_rows, j]
    ev_close = o * p_close - 1  # expected value if the closing market is right
    roi, lo, hi = mean_ci(pnl)
    clv_mean, clv_lo, clv_hi = mean_ci(ev_close)
    return {
        "bets": n,
        "hit_rate": float(won.mean()),
        "avg_odds": float(o.mean()),
        "roi": roi,
        "roi_ci": [lo, hi],
        "roi_t": t_stat(pnl),
        "roi_p": p_value_one_sided(pnl),
        "clv": clv_mean,
        "clv_ci": [clv_lo, clv_hi],
        "clv_t": t_stat(ev_close),
        "luck": roi - clv_mean,
        # log decomposition of the expected value: ln(o * p_close)
        "shopping": float(np.mean(np.log(o / a_open))),
        "timing": float(np.mean(np.log(a_open / a_close))),
        "margin": float(np.mean(np.log(a_close * p_close))),
        "bets_needed": required_n(max(roi, 1e-12), pnl.std()) if roi > 0 else float("inf"),
        "pnl": pnl,
    }


def kaunitz(frame: dict, rows: np.ndarray, alpha: float, which: str = "open") -> tuple[np.ndarray, np.ndarray]:
    """Bet at the best price when it beats 1 / (1/average - alpha)."""
    avg, mx = frame[f"avg_{which}"][rows], frame[f"max_{which}"][rows]
    p_cons = 1 / avg - alpha
    with np.errstate(divide="ignore", invalid="ignore"):
        edge = np.where(p_cons > 0, mx * p_cons, -np.inf)
    edge = np.where(np.isnan(edge), -np.inf, edge)
    j = edge.argmax(axis=1)
    take = edge[np.arange(len(edge)), j] > 1
    return take, j


# ---------------------------------------------------------------------------


def main() -> None:
    rep = Report("s3_sports", "研究3　相手が「胴元」と「市場」：欧州サッカー約6万試合の実データ")
    df = data.load_football()
    feats = build_features(df)
    X_all = feats.to_numpy()
    y_all = df["y"].to_numpy()
    seasons = df["season"].to_numpy()

    frame = {name: data.odds(df, name) for name in data.ODDS_COLS}
    with np.errstate(invalid="ignore", divide="ignore"):
        frame["p_open"] = data.devig(frame["avg_open"])
        frame["p_close"] = data.devig(frame["avg_close"])
        frame["p_ps_close"] = data.devig(frame["ps_close"])
        frame["p_ps_open"] = data.devig(frame["ps_open"])
    has_market = (
        ~np.isnan(frame["avg_open"]).any(1) & ~np.isnan(frame["avg_close"]).any(1) & ~np.isnan(frame["max_open"]).any(1)
    )

    rep.p(
        f"データ：football-data.co.uk の欧州18リーグ（イングランド4部まで、スコットランド、ドイツ・イタリア・スペイン・フランスの1〜2部、オランダ、ベルギー、ポルトガル、トルコ、ギリシャ）、{len(df):,}試合。2017/18〜2018/19はレーティングの助走のみに使い、オッズの分析は2019/20〜2026/27（2026年10月14日まで）。"
    )
    rep.p(
        "市場の確率は、各社の平均オッズからマージンを除いた値。除き方は「べき乗法」（(1/オッズ)^k の合計が1になる k を求める）を使う。単純に合計1へ正規化する「乗法」は、下の表のとおり大穴の確率を過大に見積もるため使わない。「始値」はデータ提供元が試合の数日前（週末の試合は金曜午後）に記録したオッズ、「終値」はキックオフ直前のオッズ。"
    )

    # --- 1. market quality
    rep.h2("1. 市場の予測はどれほど正確か")
    m = has_market & (seasons != "1718") & (seasons != "1819")
    pc, po, y = frame["p_close"][m], frame["p_open"][m], y_all[m]
    bins = np.linspace(0, 1, 21)
    flat_p, flat_y = pc.ravel(), (np.eye(3)[y]).ravel()
    idx = np.digitize(flat_p, bins) - 1
    xs, ys, ws = [], [], []
    for b in range(20):
        sel = idx == b
        if sel.sum() < 200:
            continue
        xs.append(flat_p[sel].mean())
        ys.append(flat_y[sel].mean())
        ws.append(sel.sum())
    xs, ys, ws = map(np.array, (xs, ys, ws))
    r2 = 1 - (ws * (ys - xs) ** 2).sum() / (ws * (ys - np.average(ys, weights=ws)) ** 2).sum()
    ece = float((ws * np.abs(ys - xs)).sum() / ws.sum())
    ll_open, ll_close = log_loss(po, y), log_loss(pc, y)
    rep.table(
        ["指標", "値"],
        [
            ["対象試合数", f"{m.sum():,}"],
            ["終値の確率と実際の頻度の一致（重み付きR²、5%刻みのビン）", f"{r2:.4f}"],
            ["平均の較正誤差（ECE）", f"{ece:.4f}"],
            ["log-loss（始値）", f"{ll_open:.4f}"],
            ["log-loss（終値）", f"{ll_close:.4f}"],
            ["始値→終値の改善", f"{ll_open - ll_close:.4f} ナッツ/試合"],
        ],
    )
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], color="k", lw=0.8)
    ax.scatter(xs, ys, s=np.sqrt(ws) / 2)
    ax.set_xlabel("market probability (closing, de-vigged)")
    ax.set_ylabel("observed frequency")
    ax.set_title(f"Market calibration (R² = {r2:.4f})")
    rep.figure(savefig(fig, "s3_calibration"), "市場の確率と実際の頻度")
    plt.close(fig)

    # Favourite-longshot bias at average closing odds.
    oc = frame["avg_close"][m]
    p_mult = data.devig(oc, "multiplicative")
    flat_o = oc.ravel()
    flat_win = flat_y
    bands = [(1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 10.0), (10.0, 1000.0)]
    flb_rows, flb = [], {}
    for lo_o, hi_o in bands:
        sel = (flat_o >= lo_o) & (flat_o < hi_o)
        pnl = np.where(flat_win[sel] == 1, flat_o[sel] - 1, -1.0)
        mu, lo, hi = mean_ci(pnl)
        flb_rows.append(
            [
                f"{lo_o:.1f}〜{hi_o:.1f}" if hi_o < 1000 else f"{lo_o:.0f}以上",
                f"{sel.sum():,}",
                pct(flat_win[sel].mean(), 2, sign=False),
                pct(p_mult.ravel()[sel].mean(), 2, sign=False),
                pct(flat_p[sel].mean(), 2, sign=False),
                pct(mu),
                ci_str(lo, hi),
            ]
        )
        flb[f"{lo_o}-{hi_o}"] = {"n": int(sel.sum()), "roi": mu, "ci": [lo, hi]}
    rep.h3("本命–大穴バイアス：平均終値で全ての結果に賭けた場合の回収")
    rep.table(
        ["オッズ帯", "賭けの数", "実際の勝率", "乗法で除いた確率", "べき乗法で除いた確率", "ROI", "95%CI"], flb_rows
    )
    rep.p(
        "市場の確率は実際の頻度とよく一致し、試合直前に向けてさらに正確になる。オッズが長いほど回収は悪く、本命–大穴バイアスはサッカーにも存在する。ただし、どの帯でもROIはマイナスで、歪みはマージンの内側に収まる。"
    )
    rep.metric(
        "market", {"n": int(m.sum()), "r2": r2, "ece": ece, "ll_open": ll_open, "ll_close": ll_close, "flb": flb}
    )

    # --- 2. models, walk-forward
    rep.h2("2. 機械学習モデルは市場に勝てるか")
    rep.p(
        "特徴量は、各試合の前日までの結果だけから作る：Eloレーティング差（ホーム補正つき）、得点・失点・シュート・枠内シュート・勝点の指数平滑平均（ホームとアウェーそれぞれ）とその差、試合数、1部リーグかどうか。モデルは、Eloだけのロジスティック回帰、全特徴量のロジスティック回帰（LR）、勾配ブースティング（GBM）。各シーズンを、それより前の全シーズンで学習したモデルで予測する（ウォークフォワード）。"
    )
    rep.p(
        "ブレンドは、Benter（1994）と同じく p ∝ p_model^a × p_market^b の形でモデルと市場（始値）を組み合わせ、a と b を過去シーズンの標本外予測で最尤推定した。"
    )
    preds = {}
    for s in OOS_SEASONS:
        first = df.loc[seasons == s, "date"].min()
        train = (df["date"] < first).to_numpy() & (seasons != "1718")
        test = seasons == s
        models = fit_models(X_all[train], y_all[train], seed=int(s))
        p = predict(models, X_all[test])
        for name, v in p.items():
            preds.setdefault(name, np.full((len(df), 3), np.nan))[test] = v
        print(f"  trained models for {SEASON_LABEL[s]}")
    base = {}
    blend_w = {}
    preds["Blend (GBM+market)"] = np.full((len(df), 3), np.nan)
    for s in TEST_SEASONS:
        hist = np.isin(seasons, OOS_SEASONS[: OOS_SEASONS.index(s)]) & has_market
        w = fit_blend(preds["GBM"][hist], frame["p_open"][hist], y_all[hist])
        blend_w[s] = w
        test = (seasons == s) & has_market
        preds["Blend (GBM+market)"][test] = apply_blend(preds["GBM"][test], frame["p_open"][test], w)
        freq = np.bincount(y_all[(df["date"] < df.loc[seasons == s, "date"].min()).to_numpy()], minlength=3)
        base[s] = freq / freq.sum()

    ev_mask = np.isin(seasons, TEST_SEASONS) & has_market
    yt = y_all[ev_mask]
    cand = {"基準率（ホーム・引分・アウェーの頻度）": np.vstack([base[s] for s in seasons[ev_mask]])}
    for name in ["Elo (LR)", "LR", "GBM", "Blend (GBM+market)"]:
        cand[name] = preds[name][ev_mask]
    cand["市場（始値）"] = frame["p_open"][ev_mask]
    cand["市場（終値）"] = frame["p_close"][ev_mask]
    rows, model_metrics = [], {}
    ref = cand["市場（終値）"]
    for name, p in cand.items():
        ll = log_loss(p, yt)
        diff = -np.log(np.clip(p[np.arange(len(yt)), yt], 1e-12, 1)) + np.log(ref[np.arange(len(yt)), yt])
        d, dlo, dhi = mean_ci(diff)
        acc = (p.argmax(1) == yt).mean()
        rows.append(
            [name, f"{ll:.4f}", f"{brier(p, yt):.4f}", pct(acc, 1, sign=False), f"{d:+.4f} [{dlo:+.4f}, {dhi:+.4f}]"]
        )
        model_metrics[name] = {"log_loss": ll, "brier": brier(p, yt), "accuracy": acc, "vs_close": [d, dlo, dhi]}
    ps_mask = ev_mask & ~np.isnan(frame["p_ps_close"]).any(1)
    rep.table(["予測", "log-loss", "Brier", "的中率", "終値とのlog-loss差 [95%CI]"], rows)
    rep.p(
        f"評価は2021/22〜2026/27の {ev_mask.sum():,} 試合（全予測で同じ試合）。log-loss差がプラスなら市場（終値）より悪い。参考：Pinnacleの終値が揃う {ps_mask.sum():,} 試合では、Pinnacle終値のlog-lossは {log_loss(frame['p_ps_close'][ps_mask], y_all[ps_mask]):.4f}、平均終値は {log_loss(frame['p_close'][ps_mask], y_all[ps_mask]):.4f}。"
    )
    rep.p(
        "ブレンドの重み（a：モデル、b：市場）："
        + "、".join(f"{SEASON_LABEL[s]} a={w[0]:.2f}, b={w[1]:.2f}" for s, w in blend_w.items())
    )
    by_season = []
    for s in TEST_SEASONS:
        mk = (seasons == s) & has_market
        by_season.append(
            [SEASON_LABEL[s], f"{mk.sum():,}"]
            + [
                f"{log_loss(p_[mk], y_all[mk]):.4f}"
                for p_ in (preds["GBM"], preds["Blend (GBM+market)"], frame["p_open"], frame["p_close"])
            ]
        )
    rep.h3("シーズン別のlog-loss")
    rep.table(["シーズン", "試合数", "GBM", "ブレンド", "市場（始値）", "市場（終値）"], by_season)
    rep.metric("models", model_metrics)
    rep.metric("blend_weights", blend_w)

    # --- 3. value betting
    rep.h2("3. バリューベット：モデルの確率 × オッズ > 1 + τ のときだけ賭ける")
    rep.p(
        "1試合に最大1点、各1単位の均等賭け。賭ける価格は3種類：「最高値」は始値時点で全社の中の最も高いオッズ（全社に口座を持つ前提）、「Bet365」は1社だけ、「平均」は平均的な1社。CLVは、取ったオッズで終値の市場確率が正しい場合の期待値（odds × p_終値 − 1）。"
    )
    rep.p(
        "期待値の分解（対数）：ln(odds × p_終値) = 価格の比較（ln 取ったオッズ/平均始値）＋ 時機（ln 平均始値/平均終値：賭けた後にオッズが下がれば正）＋ マージン（ln 平均終値 × p_終値：胴元の取り分、常に負）。残差は実際のROIとCLVの差。"
    )
    rows, vb = [], {}
    rows_dec = []
    cum_curves = {}
    idx_all = np.where(ev_mask)[0]
    for model in ["GBM", "Blend (GBM+market)", "LR"]:
        q = preds[model][idx_all]
        for price in ["max_open", "b365_open", "avg_open"]:
            for tau in (0.0, 0.05, 0.10, 0.20):
                o_all = frame[price][idx_all]
                take, j = value_bets(q, o_all, tau)
                sel = idx_all[take]
                st = bet_stats(sel, j[take], o_all[take, j[take]] if take.any() else np.array([]), y_all, frame)
                key = f"{model}|{price}|{tau}"
                pnl = st.pop("pnl", None)
                vb[key] = st
                if st["bets"] == 0:
                    continue
                pred_ev = float((q[take, j[take]] * o_all[take, j[take]] - 1).mean())
                st["model_ev"] = pred_ev
                label = {"max_open": "最高値", "b365_open": "Bet365", "avg_open": "平均"}[price]
                rows.append(
                    [
                        model,
                        label,
                        f"{tau:.2f}",
                        f"{st['bets']:,}",
                        pct(st["hit_rate"], 1, sign=False),
                        num(st["avg_odds"], 2),
                        pct(pred_ev),
                        pct(st["roi"]),
                        ci_str(*st["roi_ci"]),
                        pct(st["clv"]),
                        num(st["roi_t"], 2),
                        num(st["clv_t"], 2),
                    ]
                )
                if tau == 0.05:
                    rows_dec.append(
                        [
                            model,
                            label,
                            f"{st['bets']:,}",
                            pct(st["shopping"]),
                            pct(st["timing"]),
                            pct(st["margin"]),
                            pct(st["clv"]),
                            pct(st["luck"]),
                            pct(st["roi"]),
                        ]
                    )
                if (model == "GBM" and tau == 0.05 and price in ("max_open", "avg_open")) or (
                    model == "Blend (GBM+market)" and tau == 0.0 and price == "max_open"
                ):
                    cum_curves[f"{model} @ {label}, τ={tau}"] = np.cumsum(pnl)
    rep.table(
        [
            "モデル",
            "価格",
            "τ",
            "賭け数",
            "的中率",
            "平均オッズ",
            "モデルが見込んだ期待値",
            "実際のROI",
            "ROI 95%CI",
            "CLV",
            "ROIのt値",
            "CLVのt値",
        ],
        rows,
    )
    rep.h3("期待値の分解（τ = 0.05）")
    rep.table(
        [
            "モデル",
            "価格",
            "賭け数",
            "価格の比較",
            "時機",
            "マージン",
            "期待値（CLV）",
            "残差（ROI − CLV）",
            "実際のROI",
        ],
        rows_dec,
    )
    # Adverse selection: outcomes whose best price is far above the average win less often than even the closing market says.
    prem = frame["max_open"] / frame["avg_open"]
    onehot = np.eye(3)[y_all] == 1
    prem_rows, prem_m = [], {}
    for lo_p, hi_p in [(1.0, 1.03), (1.03, 1.06), (1.06, 1.10), (1.10, 1.20), (1.20, 100.0)]:
        sel = has_market[:, None] & (prem >= lo_p) & (prem < hi_p)
        obs, pred = onehot[sel].mean(), frame["p_close"][sel].mean()
        prem_rows.append(
            [
                f"{lo_p:.2f}〜{hi_p:.2f}" if hi_p < 100 else f"{lo_p:.2f}以上",
                f"{sel.sum():,}",
                pct(obs, 2, sign=False),
                pct(pred, 2, sign=False),
                f"{obs / pred:.3f}",
            ]
        )
        prem_m[f"{lo_p}-{hi_p}"] = {"n": int(sel.sum()), "observed": obs, "closing_market": pred}
    rep.p(
        "残差が系統的にマイナスなのは、運だけでは説明できない。最高値が平均から大きく離れている結果ほど、終値の市場確率よりも実際に起きる頻度が低い（下表）。高すぎる価格は、それだけ起きにくい結果（多くは大穴）についている。最高値を機械的に拾う戦略は、この逆選択を受ける。"
    )
    rep.table(["最高値 ÷ 平均始値", "結果の数", "実際の頻度", "終値の市場確率", "比"], prem_rows)
    rep.metric("adverse_selection", prem_m)

    # --- 4. Kaunitz-style rule
    rep.h2("4. 予測モデルを使わない市場だけの戦略（Kaunitz et al. 2017 型）")
    rep.p(
        "平均オッズから「コンセンサス確率」1/平均 − α を作り、最高値のオッズがその逆数を上回る結果に賭ける。予測モデルは使わず、各社の価格のばらつきだけを使う。"
    )
    rows, kz = [], {}
    all_idx = np.where(has_market)[0]
    for which in ("open", "close"):
        for alpha in (0.03, 0.05, 0.07):
            take, j = kaunitz(frame, all_idx, alpha, which)
            sel = all_idx[take]
            o = frame[f"max_{which}"][sel, j[take]]
            st = bet_stats(sel, j[take], o, y_all, frame)
            pnl = st.pop("pnl")
            kz[f"{which}|{alpha}"] = st
            by_s = []
            for s in data.ODDS_SEASONS:
                ms = seasons[sel] == s
                if ms.sum():
                    by_s.append(f"{SEASON_LABEL[s][2:]}:{pnl[ms].mean() * 100:+.1f}%")
            rows.append(
                [
                    "始値" if which == "open" else "終値",
                    f"{alpha:.2f}",
                    f"{st['bets']:,}",
                    pct(st["hit_rate"], 1, sign=False),
                    num(st["avg_odds"], 2),
                    pct(st["roi"]),
                    ci_str(*st["roi_ci"]),
                    pct(st["clv"]),
                    num(st["roi_t"], 2),
                    " ".join(by_s),
                ]
            )
            if which == "open" and alpha == 0.03:
                cum_curves["Kaunitz rule (α 0.03, opening prices)"] = np.cumsum(pnl)
    rep.table(
        ["価格の時点", "α", "賭け数", "的中率", "平均オッズ", "ROI", "95%CI", "CLV", "t値", "シーズン別ROI"], rows
    )
    rep.h3("鋭い胴元の価格を「正解」とみなす戦略")
    rep.p(
        "Pinnacle（低マージンで勝ち客を制限しないとされる胴元）の始値からマージンを除いた確率を真の確率とみなし、他社の価格がそれを上回る結果に賭ける。これも予測モデルを使わない。"
    )
    rows, sharp = [], {}
    ps_idx = np.where(has_market & ~np.isnan(frame["p_ps_open"]).any(1))[0]
    for price in ("b365_open", "max_open"):
        for tau in (0.0, 0.02, 0.05):
            o_all = frame[price][ps_idx]
            take, j = value_bets(frame["p_ps_open"][ps_idx], o_all, tau)
            sel = ps_idx[take]
            st = bet_stats(sel, j[take], o_all[take, j[take]], y_all, frame)
            st.pop("pnl")
            sharp[f"{price}|{tau}"] = st
            rows.append(
                [
                    {"b365_open": "Bet365", "max_open": "最高値"}[price],
                    f"{tau:.2f}",
                    f"{st['bets']:,}",
                    pct(st["hit_rate"], 1, sign=False),
                    num(st["avg_odds"], 2),
                    pct(st["roi"]),
                    ci_str(*st["roi_ci"]),
                    pct(st["clv"]),
                    num(st["roi_t"], 2),
                ]
            )
    rep.table(["賭ける価格", "τ", "賭け数", "的中率", "平均オッズ", "ROI", "95%CI", "CLV", "t値"], rows)
    rep.metric("sharp_vs_soft", sharp)
    rep.p(
        "注意：「最高値」は集計サイトが記録した全社の最大値で、取引所（手数料前）や、すぐに修正される誤った価格を含みうる。Kaunitzらの紙上運用では、表示されたオッズの約30%が確認時点で変わっていた。ここでのROIは、記録された価格で必ず約定できたという上限の見積もりである。"
    )
    rep.metric("value_bets", vb)
    rep.metric("kaunitz", kz)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for name, c in cum_curves.items():
        ax.plot(c, label=name)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("bets (chronological)")
    ax.set_ylabel("cumulative P&L (units)")
    ax.set_title("Flat 1-unit bets, out-of-sample")
    ax.legend(fontsize=8)
    rep.figure(savefig(fig, "s3_cumulative_pnl"), "戦略別の累積損益")
    plt.close(fig)
    rep.save()


if __name__ == "__main__":
    main()
