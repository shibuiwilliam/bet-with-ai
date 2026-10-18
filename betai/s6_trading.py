"""Study 6 - financial markets: the risk premium, trading costs, leverage and an
AI signal, on 100 years of real US daily data (Kenneth R. French Data Library).

1. The risk premium: how often holding the market beats cash, by horizon.
2. Zero-skill day traders against buy-and-hold on the real return path.
3. Leverage on real daily returns: growth, drawdowns and wipe-outs.
4. Machine-learning signals on 49 industry portfolios, walk-forward, with costs.
5. Strategy search: in-sample winner vs out-of-sample, and the deflated Sharpe ratio.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from . import data
from .common import Report, deflated_sharpe_prob, max_drawdown, pct, savefig

DAYS = 252


# ---------------------------------------------------------------------------
# 1-3. Market, day traders, leverage
# ---------------------------------------------------------------------------


def horizon_table(ff: pd.DataFrame) -> list[list]:
    log_ex = np.log1p(ff["Mkt-RF"] + ff["RF"]) - np.log1p(ff["RF"])
    c = np.concatenate([[0.0], log_ex.cumsum().to_numpy()])
    rows = []
    for label, h in [
        ("1日", 1),
        ("1週", 5),
        ("1か月", 21),
        ("1年", 252),
        ("5年", 1260),
        ("10年", 2520),
        ("20年", 5040),
    ]:
        win = c[h:] - c[:-h]
        rows.append((label, h, float((win > 0).mean()), float(np.quantile(win, 0.05)), len(win)))
    return rows


def day_traders(ff: pd.DataFrame, start: str, n: int, cost: float, seed: int) -> dict:
    """Each day a zero-skill trader goes long or short the market at random and
    pays `cost` per round trip. Wealth is compared with buy-and-hold."""
    sub = ff.loc[start:]
    ex, rf = sub["Mkt-RF"].to_numpy(), sub["RF"].to_numpy()
    g = np.random.default_rng(seed)
    T = len(ex)
    log_bh = np.log1p(rf + ex).sum()
    finals, profit_1y, profit_5y = [], 0, 0
    for start_i in range(0, n, 500):
        m = min(500, n - start_i)
        side = np.where(g.random((m, T)) < 0.5, 1.0, -1.0)
        lr = np.log1p(rf[None, :] + side * ex[None, :] - cost)
        cum = lr.cumsum(axis=1)
        finals.append(cum[:, -1])
        profit_1y += (cum[:, DAYS - 1] > np.log1p(rf[:DAYS]).sum()).sum()
        profit_5y += (cum[:, 5 * DAYS - 1] > np.log1p(rf[: 5 * DAYS]).sum()).sum()
    finals = np.concatenate(finals)
    log_cash = np.log1p(rf).sum()
    return {
        "years": T / DAYS,
        "median_multiple_vs_cash": float(np.exp(np.median(finals) - log_cash)),
        "bh_multiple_vs_cash": float(np.exp(log_bh - log_cash)),
        "always_long_daily_round_trip": float(np.exp(np.log1p(rf + ex - cost).sum() - log_cash)),
        "p_beat_bh": float((finals > log_bh).mean()),
        "p_beat_cash_1y": profit_1y / n,
        "p_beat_cash_5y": profit_5y / n,
        "p_beat_cash_full": float((finals > log_cash).mean()),
    }


def leverage_table(ff: pd.DataFrame, borrow_spread: float = 0.005) -> list[dict]:
    ex, rf = ff["Mkt-RF"].to_numpy(), ff["RF"].to_numpy()
    mu, sigma = ex.mean() * DAYS, ex.std() * math.sqrt(DAYS)
    out = []
    for L in (1, 1.5, 2, 3, 5, 6, 10, 25):
        daily = rf + L * (ex - borrow_spread / DAYS)
        wiped = np.where(daily <= -1)[0]
        if len(wiped):
            first = ff.index[wiped[0]].date().isoformat()
            log_growth = -np.inf
            mdd = -1.0
        else:
            first = "—"
            lw = np.log1p(daily)
            log_growth = lw.sum() / (len(lw) / DAYS)
            mdd = max_drawdown(np.exp(np.cumsum(lw)))
        theory = L * (mu - borrow_spread) - L**2 * sigma**2 / 2
        out.append(
            {
                "L": L,
                "cagr_excess_log": log_growth - np.log1p(rf).sum() / (len(rf) / DAYS)
                if np.isfinite(log_growth)
                else -np.inf,
                "theory_g": theory,
                "max_dd": mdd,
                "wiped_out": first,
            }
        )
    return out


# ---------------------------------------------------------------------------
# 4. Industry signals
# ---------------------------------------------------------------------------

LOOKBACKS = [1, 5, 21, 63, 126, 252]


def industry_panel(ind: pd.DataFrame, horizon: int) -> dict:
    """Features at each rebalance day and the next-horizon returns."""
    ind = ind.loc["1969-01-01":].ffill(limit=1).fillna(0.0)
    logr = np.log1p(ind.to_numpy())
    cum = np.vstack([np.zeros(logr.shape[1]), logr.cumsum(axis=0)])
    T, N = logr.shape
    reb = np.arange(260, T - horizon, horizon)
    feats, fwd = [], []
    for t in reb:
        f = [cum[t + 1] - cum[t + 1 - lb] for lb in LOOKBACKS]
        f.append(cum[t + 1 - 21] - cum[t + 1 - 252])  # 12-1 month momentum
        f.append(logr[t - 62 : t + 1].std(axis=0))
        f = np.array(f).T  # (N, k)
        z = (f - f.mean(axis=0)) / (f.std(axis=0) + 1e-12)  # cross-sectional z-scores
        feats.append(z)
        fwd.append(np.expm1(cum[t + 1 + horizon] - cum[t + 1]))
    return {"dates": ind.index[reb], "X": np.array(feats), "fwd": np.array(fwd), "horizon": horizon}


FEATURE_NAMES = [f"r{lb}" for lb in LOOKBACKS] + ["mom12_1", "vol63"]
MODEL_LABEL = {
    "reversal": "短期リバーサル",
    "momentum": "モメンタム",
    "ridge": "リッジ回帰",
    "gbm": "勾配ブースティング",
}


def signals(panel: dict, model: str, seed: int = 0) -> np.ndarray:
    """Signal per rebalance and industry. ML models are refit every January on
    the previous 10 years and only predict later dates (walk-forward)."""
    X, fwd, dates = panel["X"], panel["fwd"], panel["dates"]
    R, N, K = X.shape
    if model == "reversal":
        return -X[:, :, FEATURE_NAMES.index("r5" if panel["horizon"] <= 5 else "r21")]
    if model == "momentum":
        return X[:, :, FEATURE_NAMES.index("mom12_1")]
    sig = np.full((R, N), np.nan)
    years = dates.year
    target = fwd - fwd.mean(axis=1, keepdims=True)
    for y in range(1980, years.max() + 1):
        test = np.where(years == y)[0]
        # Only use samples whose forward window ended before the test year starts.
        train = np.where((years >= y - 10) & (years < y))[0]
        train = train[train + 1 < test[0]] if len(test) else train
        if len(test) == 0 or len(train) < 50:
            continue
        Xtr, ytr = X[train].reshape(-1, K), target[train].reshape(-1)
        if model == "ridge":
            m = Ridge(alpha=10.0).fit(Xtr, ytr)
        else:
            m = HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=200,
                l2_regularization=1.0,
                random_state=seed,
            ).fit(Xtr, ytr)
        sig[test] = m.predict(X[test].reshape(-1, K)).reshape(len(test), N)
    return sig


def backtest(panel: dict, sig: np.ndarray, top_k: int) -> dict:
    """Equal-weight long top-k / short bottom-k at each rebalance. Returns gross
    returns, turnover per rebalance (sum of absolute weight changes) and the IC."""
    fwd, dates = panel["fwd"], panel["dates"]
    valid = np.where(~np.isnan(sig).any(axis=1))[0]
    gross, turns, ics = [], [], []
    w_old = np.zeros(sig.shape[1])
    for i in valid:
        order = np.argsort(sig[i])
        w = np.zeros(sig.shape[1])
        w[order[-top_k:]] = 1.0 / top_k
        w[order[:top_k]] = -1.0 / top_k
        turns.append(np.abs(w - w_old).sum())
        gross.append(float(w @ fwd[i]))
        ics.append(stats.spearmanr(sig[i], fwd[i])[0])
        w_old = w
    ics = np.array(ics)
    return {
        "dates": dates[valid],
        "gross": np.array(gross),
        "turnover": np.array(turns),
        "ic": float(np.nanmean(ics)),
        "ic_t": float(np.nanmean(ics) / (np.nanstd(ics) / math.sqrt(len(ics)))),
        "per_year": DAYS / panel["horizon"],
    }


def net_of_costs(bt: dict, cost_bps: float) -> np.ndarray:
    return bt["gross"] - bt["turnover"] * cost_bps / 1e4


def sr(x: np.ndarray, per_year: float) -> float:
    return float(x.mean() / x.std() * math.sqrt(per_year)) if len(x) > 2 and x.std() > 0 else float("nan")


# ---------------------------------------------------------------------------


def main() -> None:
    rep = Report(
        "s6_trading",
        "研究6　金融市場：リスクプレミアム、取引コスト、レバレッジ、AIの売買シグナル（米国100年の日次実データ）",
    )
    ff = data.load_ff_daily()
    rep.p(
        f"データ：Kenneth R. French Data Library の米国株の市場超過リターン（Mkt-RF）と短期金利（RF）の日次データ（{ff.index[0].date()}〜{ff.index[-1].date()}、{len(ff):,}営業日）、49業種ポートフォリオの日次リターン（時価総額加重）。"
    )

    # --- 1. risk premium
    rep.h2("1. リスクプレミアム：保有期間が長いほど、現金に勝つ確率が上がる")
    rows, hz = [], {}
    for label, _h, p_pos, q05, n in horizon_table(ff):
        rows.append([label, pct(p_pos, 1, sign=False), pct(math.expm1(q05), 1), f"{n:,}"])
        hz[label] = {"p_beat_cash": p_pos, "q05": math.expm1(q05)}
    ex = ff["Mkt-RF"]
    rep.table(["保有期間", "株式が現金を上回った割合", "下位5%の超過リターン", "期間の数（重なりあり）"], rows)
    mu, sig = ex.mean() * DAYS, ex.std() * math.sqrt(DAYS)
    rep.p(
        f"年率の超過リターン（算術平均）{pct(mu)}、変動率 {pct(sig, sign=False)}、シャープレシオ {mu / sig:.2f}。1日あたりのシャープレシオは {ex.mean() / ex.std():.3f} で、上昇した日の割合は {pct((ex > 0).mean(), 1, sign=False)}。1日単位では、プラスの期待値はほとんど見えない。"
    )
    rep.metric("horizons", hz)
    rep.metric(
        "market",
        {
            "mu": mu,
            "sigma": sig,
            "sharpe": mu / sig,
            "daily_sharpe": ex.mean() / ex.std(),
            "up_day_rate": (ex > 0).mean(),
        },
    )

    # --- 2. day traders
    rep.h2("2. 実力ゼロのデイトレーダー 対 買い持ち（1990〜2026年の実際の値動き）")
    rep.p(
        "各営業日に、市場全体を買うか売るかを無作為に決め、その日のうちに手仕舞う。1往復のコスト（手数料とスプレッド）は0.1%と0.02%の2通り。各5,000人。"
    )
    rows, dt = [], {}
    for cost in (0.001, 0.0002):
        r = day_traders(ff, "1990-01-01", 5_000, cost, seed=int(cost * 1e5))
        dt[cost] = r
        rows.append(
            [
                pct(cost, 2, sign=False),
                f"{r['years']:.1f}年",
                f"{r['median_multiple_vs_cash']:.4f}倍",
                f"{r['always_long_daily_round_trip']:.3f}倍",
                f"{r['bh_multiple_vs_cash']:.2f}倍",
                pct(r["p_beat_cash_1y"], 1, sign=False),
                pct(r["p_beat_cash_5y"], 1, sign=False),
                pct(r["p_beat_cash_full"], 2, sign=False),
                pct(r["p_beat_bh"], 2, sign=False),
            ]
        )
    rep.table(
        [
            "往復コスト",
            "期間",
            "デイトレーダーの資産（中央値、現金比）",
            "毎日買って手仕舞う人",
            "買い持ち",
            "1年目に現金を上回った割合",
            "5年目",
            "全期間",
            "買い持ちを上回った割合",
        ],
        rows,
    )
    rep.p(
        "買い持ちはリスクプレミアムを受け取るが、毎日売買すると、プレミアムのない賭けにコストを払い続けることになる。1年目だけ見れば、実力ゼロでも相当数が現金を上回る。"
    )
    rep.metric("day_traders", dt)

    # --- 3. leverage
    rep.h2("3. レバレッジ：期待値がプラスでも、倍率が高すぎると負ける")
    lv = leverage_table(ff)
    worst = ex.idxmin()
    rows = [
        [
            f"{d['L']:g}倍",
            pct(d["cagr_excess_log"]) if np.isfinite(d["cagr_excess_log"]) else "破綻",
            pct(d["theory_g"]),
            pct(d["max_dd"], 1),
            d["wiped_out"],
        ]
        for d in lv
    ]
    rep.p("市場超過リターンを毎日L倍で持ち直し、借入の上乗せ金利を年0.5%とする（1926〜2026年）。")
    rep.table(
        [
            "レバレッジ",
            "年率の対数成長率（現金比、実績）",
            "理論値 L(μ−k) − L²σ²/2",
            "最大ドローダウン",
            "1日で資金がゼロ以下になった日",
        ],
        rows,
    )
    rep.p(
        f"最悪の日は {worst.date()} の {pct(ex.min(), 1)}。この日を生き残れる上限は 1/{abs(ex.min()):.3f} ≈ {1 / abs(ex.min()):.1f}倍で、それを超える倍率は、その他の日の成績に関係なく一度で資金を失う。"
    )
    rep.metric("leverage", lv)

    # --- 4. AI signals on industries
    rep.h2("4. AIの売買シグナル：49業種のロング・ショート")
    rep.p(
        "各リバランス日に、過去1日・1週・1か月・3か月・6か月・1年のリターン、12−1か月モメンタム、3か月の変動率を業種間で標準化した特徴量を作り、次の期間のリターンを予測する。上位k業種を買い、下位k業種を売る（金額は同じ）。機械学習（リッジ回帰と勾配ブースティング）は、毎年1月に直前10年のデータで学び直し、その年だけを予測する。比較として、機械学習を使わない2つの古典的なルール（短期リバーサル、12−1モメンタム）も同じ条件で試す。評価は1980〜2026年。"
    )
    results = {}
    panels = {h: industry_panel(data.load_industries_daily(), h) for h in (5, 21)}
    variants = []
    for h, panel in panels.items():
        for model in ("reversal", "momentum", "ridge", "gbm"):
            s = signals(panel, model)
            for k in (5, 10):
                variants.append((h, model, k, s))
    rows = []
    for h, model, k, sgl in variants:
        bt = backtest(panels[h], sgl, k)
        mask = bt["dates"] >= "1980-01-01"
        per_year = bt["per_year"]
        g = bt["gross"][mask]
        nets = {c: net_of_costs(bt, c)[mask] for c in (5, 10, 20)}
        key = f"{model}|{'週次' if h == 5 else '月次'}|k={k}"
        results[key] = {
            "model": model,
            "horizon": h,
            "k": k,
            "gross_ret": g.mean() * per_year,
            "gross_sr": sr(g, per_year),
            "net_sr": {c: sr(v, per_year) for c, v in nets.items()},
            "net_ret_10": nets[10].mean() * per_year,
            "turnover": float(bt["turnover"][mask].mean()),
            "ic": bt["ic"],
            "ic_t": bt["ic_t"],
            "dates": bt["dates"][mask],
            "net10": nets[10],
            "gross": g,
            "per_year": per_year,
        }
        rows.append(
            [
                MODEL_LABEL[model],
                "週次" if h == 5 else "月次",
                k,
                pct(g.mean() * per_year, 1),
                f"{sr(g, per_year):.2f}",
                f"{sr(nets[5], per_year):.2f}",
                f"{sr(nets[10], per_year):.2f}",
                f"{sr(nets[20], per_year):.2f}",
                f"{bt['turnover'][mask].mean():.2f}",
                f"{bt['ic']:+.4f} (t={bt['ic_t']:.1f})",
            ]
        )
    mkt = ff.loc["1980-01-01":, "Mkt-RF"]
    rep.table(
        [
            "シグナル",
            "リバランス",
            "売買する業種数k",
            "年率リターン（コスト前）",
            "シャープ（コスト前）",
            "片道5bp",
            "片道10bp",
            "片道20bp",
            "1回の売買回転率",
            "IC（順位相関）",
        ],
        rows,
    )
    rep.p(
        f"参考：同じ期間の市場の買い持ち（超過リターン）のシャープレシオは {mkt.mean() / mkt.std() * math.sqrt(DAYS):.2f}。業種ETFの売買コストは片道数bp〜10bp程度。"
    )

    # by sub-period for the best ML and the rules (weekly, k=10)
    rep.h3("時代による変化（週次・k=10、片道10bpのシャープ）")
    periods = [("1980–1999", "1980", "1999"), ("2000–2012", "2000", "2012"), ("2013–2026", "2013", "2026")]
    rows = []
    for key in ("reversal|週次|k=10", "momentum|週次|k=10", "ridge|週次|k=10", "gbm|週次|k=10"):
        rr = results[key]
        cells = [MODEL_LABEL[key.split("|")[0]]]
        for _, a, b in periods:
            m = (rr["dates"] >= f"{a}-01-01") & (rr["dates"] <= f"{b}-12-31")
            cells.append(f"{sr(rr['gross'][m], rr['per_year']):.2f} → {sr(rr['net10'][m], rr['per_year']):.2f}")
        rows.append(cells)
    rep.table(["シグナル"] + [f"{p[0]}（コスト前 → 後）" for p in periods], rows)

    # --- 5. search and deflated Sharpe
    rep.h2("5. 戦略の探索：過去の最良は、未来の最良か")
    rep.p(
        f"上の{len(results)}通り（シグナル4種 × リバランス2種 × k 2種）から、1980〜1999年のコスト後シャープ（片道10bp）が最も高いものを選び、2000〜2026年の成績と比べる。"
    )
    ins, oos = {}, {}
    for key, rr in results.items():
        m_in = rr["dates"] < "2000-01-01"
        ins[key] = sr(rr["net10"][m_in], rr["per_year"])
        oos[key] = sr(rr["net10"][~m_in], rr["per_year"])
    best = max(ins, key=ins.get)
    rank_oos = sorted(oos, key=oos.get, reverse=True).index(best) + 1
    corr = stats.spearmanr([ins[k] for k in results], [oos[k] for k in results])[0]
    rr = results[best]
    x = rr["net10"]
    sr_obs = x.mean() / x.std()
    sr_var = np.var([v / math.sqrt(results[k]["per_year"]) for k, v in ins.items()])
    dsr = deflated_sharpe_prob(sr_obs, len(x), len(results), sr_var, stats.skew(x), stats.kurtosis(x, fisher=False))
    psr = deflated_sharpe_prob(sr_obs, len(x), 1, sr_var, stats.skew(x), stats.kurtosis(x, fisher=False))
    rep.table(
        ["指標", "値"],
        [
            ["1980〜1999年の最良", best],
            ["その標本内シャープ", f"{ins[best]:.2f}"],
            ["その標本外シャープ（2000〜2026年）", f"{oos[best]:.2f}"],
            ["標本外の順位", f"{rank_oos} / {len(results)}"],
            ["標本内と標本外のシャープの順位相関", f"{corr:.2f}"],
            ["全期間のシャープ（年率）", f"{sr(x, rr['per_year']):.2f}"],
            ["真のシャープが0より大きい確率（1回だけ試した場合、PSR）", pct(psr, 1, sign=False)],
            [
                f"Deflated Sharpe（{len(results)}通りの試行を考慮、真のシャープが運による最良値の期待値を上回る確率）",
                pct(dsr, 1, sign=False),
            ],
        ],
    )
    rep.p(
        "1つだけ試したなら有意に見える成績も、16通りから選んだという事実を考慮すると、運による最良値と区別できなくなる。標本外では、標本内の最良戦略のシャープは大きく下がった。"
    )
    rep.metric(
        "signals",
        {k: {kk: vv for kk, vv in v.items() if kk not in ("dates", "net10", "gross")} for k, v in results.items()},
    )
    rep.metric(
        "search",
        {
            "best_in_sample": best,
            "in_sample_sr": ins[best],
            "out_of_sample_sr": oos[best],
            "oos_rank": rank_oos,
            "rank_corr": corr,
            "deflated_sharpe_prob": dsr,
            "psr": psr,
        },
    )

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for key in ("reversal|週次|k=10", "momentum|月次|k=10", "ridge|週次|k=10", "gbm|週次|k=10"):
        rr = results[key]
        ax.plot(rr["dates"], np.cumsum(np.log1p(rr["net10"])), label=f"{key} (net 10bp)")
    m = ff.loc["1980-01-01":]
    ax.plot(m.index, np.cumsum(np.log1p(m["Mkt-RF"])), color="k", lw=1, label="market excess (buy & hold)")
    ax.set_ylabel("cumulative log return")
    ax.legend(fontsize=7)
    ax.set_title("Industry long-short signals after costs vs the market")
    rep.figure(savefig(fig, "s6_signals"), "業種シグナルのコスト後累積リターンと市場")
    plt.close(fig)
    rep.save()


if __name__ == "__main__":
    main()
