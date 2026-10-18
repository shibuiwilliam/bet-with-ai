"""Study 5 - opponent: the market (pari-mutuel horse racing at JRA's 20% takeout).

Races are simulated so that the true probabilities are known. The crowd's
pool shares are a noisy, flattened view of the truth (which creates a
favourite-longshot bias). AI models see the truth with independent noise.

1. The identity W = ln(1-t) + D(p||r) - D(p||q), checked race by race.
2. How large an information edge each model has, against the 0.223-nat bar.
3. Selective flat betting and Kelly betting: ROI, hit rate, and a decomposition
   of each bet's expected value into takeout, information edge and breakage.
4. The Benter blend of model and market.
5. Capacity: own bets lower the odds, which caps the profit per race.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from .common import Report, ci_str, mean_ci, num, pct, required_n, savefig

MAX_H = 16
TAKEOUT = 0.20


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.nanmax(z, axis=1, keepdims=True)
    e = np.where(np.isfinite(z), np.exp(z), 0.0)
    return e / e.sum(axis=1, keepdims=True)


def make_races(n: int, seed: int, beta_pub: float = 0.85, sd_pub: float = 0.35, gamma: float = 1.2) -> dict:
    g = np.random.default_rng(seed)
    sizes = g.integers(8, MAX_H + 1, n)
    mask = np.arange(MAX_H)[None, :] < sizes[:, None]
    true_logit = np.where(mask, gamma * g.standard_normal((n, MAX_H)), -np.inf)
    p = softmax(true_logit)
    pub = np.where(mask, beta_pub * (true_logit + sd_pub * g.standard_normal((n, MAX_H))), -np.inf)
    r = softmax(pub)
    u = g.random(n)
    winner = np.minimum((p.cumsum(axis=1) < u[:, None]).sum(axis=1), sizes - 1)
    return {"mask": mask, "logit": true_logit, "p": p, "r": r, "winner": winner, "g": g}


def odds_from_shares(r: np.ndarray, t: float, breakage: bool = True) -> np.ndarray:
    with np.errstate(divide="ignore"):
        o = (1 - t) / r
    if breakage:  # JRA pays per 100 yen in 10-yen steps, never below 100 yen
        o = np.maximum(1.0, np.floor(o * 10 + 1e-9) / 10)
    return np.where(r > 0, o, np.nan)


def model_probs(races: dict, sd: float, seed: int) -> np.ndarray:
    g = np.random.default_rng(seed)
    z = np.where(races["mask"], races["logit"] + sd * g.standard_normal(races["logit"].shape), -np.inf)
    return softmax(z)


def market_copy(races: dict, seed: int) -> np.ndarray:
    """An agent that mostly restates the market (as an LLM that reads the odds might)."""
    g = np.random.default_rng(seed)
    z = np.where(
        races["mask"], np.log(np.clip(races["r"], 1e-12, 1)) + 0.10 * g.standard_normal(races["r"].shape), -np.inf
    )
    return softmax(z)


def kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, p * np.log(p / np.clip(q, 1e-300, 1)), 0.0)
    return t.sum(axis=1)


def fit_blend(q, r, winner):
    lq, lr = np.log(np.clip(q, 1e-12, 1)), np.log(np.clip(r, 1e-12, 1))
    valid = q > 0

    def nll(w):
        z = np.where(valid, w[0] * lq + w[1] * lr, -np.inf)
        pr = softmax(z)
        return -np.log(np.clip(pr[np.arange(len(winner)), winner], 1e-300, 1)).mean()

    res = optimize.minimize(nll, x0=[0.5, 0.5], method="Nelder-Mead")
    return res.x


def apply_blend(q, r, w):
    z = np.where(q > 0, w[0] * np.log(np.clip(q, 1e-12, 1)) + w[1] * np.log(np.clip(r, 1e-12, 1)), -np.inf)
    return softmax(z)


def kelly_with_cash(q: np.ndarray, o: np.ndarray, fraction: float) -> np.ndarray:
    """Optimal Kelly fractions for one race when cash may be held back
    (the classic algorithm for mutually exclusive outcomes)."""
    order = np.argsort(-(q * o))
    s_q, s_inv, chosen = 0.0, 0.0, []
    for i in order:
        if not np.isfinite(o[i]) or q[i] <= 0:
            continue
        R = (1 - s_q) / (1 - s_inv) if s_inv < 1 else np.inf
        if q[i] * o[i] > R:
            chosen.append(i)
            s_q += q[i]
            s_inv += 1 / o[i]
        else:
            break
    b = np.zeros_like(q)
    if chosen:
        R = (1 - s_q) / (1 - s_inv)
        for i in chosen:
            b[i] = max(0.0, q[i] - R / o[i])
    return b * fraction


def capacity(p: float, r: float, pool: float, t: float) -> tuple[float, float]:
    """Stake that maximises expected profit on one horse when the stake itself
    enters the pool, and that maximum. `pool` is the win pool before our bet."""

    def neg_profit(x):
        o = (1 - t) * (pool + x) / (pool * r + x)
        return -(x * (p * o - 1))

    res = optimize.minimize_scalar(neg_profit, bounds=(0, pool * r * 5), method="bounded")
    return float(res.x), float(-res.fun)


# ---------------------------------------------------------------------------


def main() -> None:
    rep = Report("s5_parimutuel", "研究5　相手が「市場」：パリミュチュエル方式の競馬（JRAの単勝、控除率20%）")
    rep.p(
        "注意：既定の設定（群衆のノイズ0.35）は、AIが優位を見つけやすい比較的不正確な市場である。研究3の実データのように市場が正確なら、同じAIの優位は消える（2節の感度分析）。"
    )
    rep.p(
        "真の勝率がわかるように、レースを模擬する。1レース8〜16頭、各馬の強さは正規分布。群衆（オッズ）の確率は真の確率にノイズを加えて平らにしたもの（大穴が買われすぎる本命–大穴バイアスが生じる）。AIは真の確率に、群衆とは独立なノイズを加えて見る（現実より有利な仮定：現実のAIと群衆は同じ公開情報を使うので誤差が相関する）。単勝オッズ =（1 − 控除率）÷ 票数シェアを10円単位で切り捨て（最低1.0倍）。"
    )
    train = make_races(20_000, seed=1)
    test = make_races(60_000, seed=2)
    o = odds_from_shares(test["r"], TAKEOUT)
    o_raw = odds_from_shares(test["r"], TAKEOUT, breakage=False)
    p, r, w, mask = test["p"], test["r"], test["winner"], test["mask"]
    idx = np.arange(len(w))

    models = {
        "AI（強）ノイズ0.25": model_probs(test, 0.25, 10),
        "AI（中）ノイズ0.50": model_probs(test, 0.50, 11),
        "AI（弱）ノイズ0.80": model_probs(test, 0.80, 12),
        "市場を写すAI": market_copy(test, 13),
    }
    train_q = model_probs(train, 0.50, 20)
    bw = fit_blend(train_q, train["r"], train["winner"])
    models["ブレンド（AI中×市場、Benter型）"] = apply_blend(models["AI（中）ノイズ0.50"], r, bw)

    # --- 1. identity and information edge
    rep.h2("1. 成長率の恒等式と、必要な情報優位")
    rep.p(
        f"全頭に予測確率どおりの割合で全資金を賭けたとき、1レースの対数成長率は W = ln(1−t) + D(p‖r) − D(p‖q)。控除率20%なら ln(0.8) = {math.log(0.8):+.4f} ナッツ。市場に勝つには、市場より {-math.log(0.8):.3f} ナッツ以上、真の確率に近い予測が要る。"
    )
    rows, ident = [], {}
    market_ll = -np.log(r[idx, w]).mean()
    for name, q in models.items():
        realized = np.log(q[idx, w] * o_raw[idx, w])
        expected = math.log(1 - TAKEOUT) + kl(p, r) - kl(p, q)
        direct = np.nansum(np.where(mask, p * np.log(np.clip(q * o_raw, 1e-300, None)), 0.0), axis=1)
        gap = float(np.abs(expected - direct).max())
        edge = float((kl(p, r) - kl(p, q)).mean())
        ll = -np.log(q[idx, w]).mean()
        m, lo, hi = mean_ci(realized)
        rows.append(
            [
                name,
                f"{ll:.4f}",
                f"{market_ll - ll:+.4f}",
                f"{edge:+.4f}",
                f"{expected.mean():+.4f}",
                f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]",
                f"{gap:.1e}",
            ]
        )
        ident[name] = {
            "log_loss": ll,
            "edge_vs_market_nats": edge,
            "expected_W": float(expected.mean()),
            "realized_W": m,
            "identity_max_gap": gap,
        }
    rows.append(["市場（オッズ）", f"{market_ll:.4f}", "0", "0", f"{math.log(0.8):+.4f}", "—", "—"])
    rep.table(
        [
            "予測",
            "log-loss",
            "市場との差（実現）",
            "情報優位 D(p‖r)−D(p‖q)",
            "期待成長率 W",
            "実現した成長率 [95%CI]",
            "恒等式と直接計算の最大差",
        ],
        rows,
    )
    rep.p(
        "どのAIも、情報優位は 0.223 ナッツに遠く届かず、全頭への比例賭けでは資金が減る。恒等式は1レースごとに機械精度で成り立つ。"
    )
    rep.metric("identity", ident)

    # --- 2. selective betting
    rep.h2("2. 期待値がプラスに見える馬だけに賭ける")
    rep.p(
        "各馬について q × オッズ > 1 + τ なら1単位を賭ける。「見込んだ期待値」はAI自身の q で計算した値、「真の期待値」は真の確率 p で計算した値。期待値の分解は ln(p × オッズ) = ln(1−t) ＋ ln(p/r)（情報優位）＋ ln(切り捨て後÷切り捨て前)（端数切り捨て）で、賭けた馬の平均をとる。"
    )
    rows, sel_m = [], {}
    for name, q in models.items():
        for tau in (0.0, 0.1, 0.2, 0.3):
            take = mask & (q * o > 1 + tau)
            ri, ci = np.where(take)
            if len(ri) == 0:
                continue
            pnl = np.where(w[ri] == ci, o[ri, ci] - 1, -1.0)
            roi, lo, hi = mean_ci(pnl)
            true_ev = (p[ri, ci] * o[ri, ci] - 1).mean()
            model_ev = (q[ri, ci] * o[ri, ci] - 1).mean()
            info = np.log(p[ri, ci] / r[ri, ci]).mean()
            brk = np.log(o[ri, ci] / o_raw[ri, ci]).mean()
            rows.append(
                [
                    name,
                    f"{tau:.1f}",
                    f"{len(ri):,}",
                    pct((pnl > 0).mean(), 1, sign=False),
                    num(o[ri, ci].mean(), 1),
                    pct(model_ev),
                    pct(true_ev),
                    pct(roi),
                    ci_str(lo, hi),
                    pct(math.log(1 - TAKEOUT)),
                    pct(info),
                    pct(brk),
                    num(required_n(true_ev, pnl.std()), 0),
                ]
            )
            sel_m[f"{name}|{tau}"] = {
                "bets": len(ri),
                "hit_rate": (pnl > 0).mean(),
                "model_ev": model_ev,
                "true_ev": true_ev,
                "roi": roi,
                "roi_ci": [lo, hi],
                "takeout_log": math.log(1 - TAKEOUT),
                "info_log": info,
                "breakage_log": brk,
                "bets_needed": required_n(true_ev, pnl.std()),
            }
    rep.table(
        [
            "AI",
            "τ",
            "賭け数",
            "的中率",
            "平均オッズ",
            "見込んだ期待値",
            "真の期待値",
            "実際のROI",
            "95%CI",
            "控除",
            "情報優位",
            "切り捨て",
            "優位の証明に必要な賭け数",
        ],
        rows,
    )
    rep.p(
        "強いAIでも、期待値がプラスになるのは控除率を上回る情報優位（ln(p/r) > 0.223）を持つ一部の馬だけで、その割合は小さい。AIが見込んだ期待値は真の期待値より系統的に大きい：AIの誤差がたまたま有利に出た馬を選ぶからだ。"
    )
    rep.metric("selective", sel_m)

    # --- market accuracy sensitivity
    rep.h3("市場の精度を変えると")
    rep.p(
        "群衆の誤差（ノイズの標準偏差）を変えて、同じ強いAI（ノイズ0.25）が τ = 0.1 で賭けた結果を比べる。D(p‖r) は市場の確率が真の確率からどれだけ離れているか（ナッツ）。"
    )
    rows, sens = [], {}
    for sd_pub in (0.10, 0.20, 0.35, 0.50):
        tr = make_races(20_000, seed=31, sd_pub=sd_pub)
        te = make_races(60_000, seed=32, sd_pub=sd_pub)
        ot = odds_from_shares(te["r"], TAKEOUT)
        q_s = model_probs(te, 0.25, 33)
        q_m = model_probs(te, 0.50, 34)
        w_b = fit_blend(model_probs(tr, 0.50, 35), tr["r"], tr["winner"])
        q_b = apply_blend(q_m, te["r"], w_b)
        r_row = [f"{sd_pub:.2f}", f"{kl(te['p'], te['r']).mean():.4f}"]
        for qq in (q_s, q_b):
            take = te["mask"] & (qq * ot > 1.1)
            ri, ci = np.where(take)
            if len(ri) == 0:
                r_row += ["0", "—", "—"]
                continue
            pnl = np.where(te["winner"][ri] == ci, ot[ri, ci] - 1, -1.0)
            r_row += [f"{len(ri):,}", pct((te["p"][ri, ci] * ot[ri, ci]).mean() - 1), pct(pnl.mean())]
        rows.append(r_row)
        sens[sd_pub] = {"market_kl": float(kl(te["p"], te["r"]).mean())}
    rep.table(
        [
            "群衆のノイズ",
            "市場の誤差 D(p‖r)",
            "強いAI：賭け数",
            "真の期待値",
            "実際のROI",
            "ブレンド：賭け数",
            "真の期待値",
            "実際のROI",
        ],
        rows,
    )
    rep.p(
        "市場が正確になるほど（ノイズが小さいほど）、同じAIが見つけられる優位は小さくなる。期待値がプラスに見える結果の大きさは、AIの精度だけでなく、相手の市場がどれだけ不正確かで決まる。"
    )
    rep.metric("market_sensitivity", sens)

    # --- favourite-longshot bias
    rep.h3("本命–大穴バイアス：全馬に1単位ずつ賭けた場合")
    rows, flb = [], {}
    oo = np.where(mask, o, np.nan)
    for lo_o, hi_o in [(1, 3), (3, 6), (6, 12), (12, 30), (30, 100), (100, 1e9)]:
        sel = mask & (oo >= lo_o) & (oo < hi_o)
        ri, ci = np.where(sel)
        pnl = np.where(w[ri] == ci, o[ri, ci] - 1, -1.0)
        roi, lo, hi = mean_ci(pnl)
        rows.append(
            [
                f"{lo_o:g}〜{hi_o:g}倍" if hi_o < 1e8 else f"{lo_o:g}倍以上",
                f"{len(ri):,}",
                pct(roi),
                ci_str(lo, hi),
                pct((p[ri, ci] * o[ri, ci]).mean() - 1),
            ]
        )
        flb[f"{lo_o}-{hi_o}"] = {"n": len(ri), "roi": roi, "true_ev": (p[ri, ci] * o[ri, ci]).mean() - 1}
    rep.table(["オッズ帯", "賭け数", "実際のROI", "95%CI", "真の期待値"], rows)
    rep.metric("flb", flb)

    # --- 3. Kelly with cash
    rep.h2("3. ケリー基準で資金を配分する")
    rep.p(
        "各レースで、現金を残すケリー最適配分（相互排他の結果に対する古典的アルゴリズム）を、AIの確率で計算して賭ける。フル・1/2・1/4ケリーで、6万レース後の資金（対数）を比べる。"
    )
    rows, kel = [], {}
    for name in ["AI（強）ノイズ0.25", "AI（中）ノイズ0.50", "ブレンド（AI中×市場、Benter型）"]:
        q = models[name]
        bets = np.zeros_like(q)
        for i in range(len(w)):
            bets[i] = kelly_with_cash(q[i], o[i], 1.0)
        for frac in (1.0, 0.5, 0.25):
            b = bets * frac
            ret = 1 - b.sum(axis=1) + b[idx, w] * o[idx, w]
            logw = np.log(np.clip(ret, 1e-300, None))
            exp_logw = np.nansum(
                np.where(mask, p * np.log(np.clip(1 - b.sum(axis=1, keepdims=True) + b * o, 1e-300, None)), 0.0), axis=1
            )
            share = (b.sum(axis=1) > 0).mean()
            rows.append(
                [
                    name,
                    f"{frac:g}",
                    pct(share, 1, sign=False),
                    f"{logw.mean():+.5f}",
                    f"{exp_logw.mean():+.5f}",
                    f"{logw.sum():+.1f}",
                ]
            )
            kel[f"{name}|{frac}"] = {
                "race_share_bet": share,
                "realized_logw": logw.mean(),
                "expected_logw": exp_logw.mean(),
                "total_log": logw.sum(),
            }
    rep.table(
        [
            "AI",
            "ケリーの倍率",
            "賭けたレースの割合",
            "実現した対数成長率/レース",
            "真の期待対数成長率/レース",
            "6万レース後の対数資金",
        ],
        rows,
    )
    rep.metric("kelly", kel)
    rep.p(f"ブレンドの重み（学習用2万レースで推定）：AI {bw[0]:.2f}、市場 {bw[1]:.2f}。")

    # --- 4. capacity
    rep.h2("4. 容量：自分の賭けがオッズを下げる")
    rep.p(
        "単勝の総票数を3,000万円と仮定し、強いAIが τ = 0.1 で選んだ馬それぞれについて、自分の賭けで下がるオッズを考慮した最適な賭け金と期待利益を計算する（端数切り捨ては無視）。"
    )
    q = models["AI（強）ノイズ0.25"]
    take = mask & (q * o > 1.1)
    ri, ci = np.where(take)
    pool = 30_000_000
    sample = np.random.default_rng(3).choice(len(ri), min(3000, len(ri)), replace=False)
    stakes, profits = [], []
    for k in sample:
        x, prof = capacity(p[ri[k], ci[k]], r[ri[k], ci[k]], pool, TAKEOUT)
        stakes.append(x)
        profits.append(prof)
    stakes, profits = np.array(stakes), np.array(profits)
    races_with_bet = len(np.unique(ri)) / len(w)
    per_race = profits.mean() * (len(ri) / len(w))
    rep.table(
        ["指標", "値"],
        [
            ["賭ける馬がいるレースの割合", pct(races_with_bet, 1, sign=False)],
            ["1頭あたりの最適な賭け金（中央値）", f"{np.median(stakes):,.0f}円"],
            ["1頭あたりの最大期待利益（中央値）", f"{np.median(profits):,.0f}円"],
            ["1レースあたりの期待利益（全レース平均）", f"{per_race:,.0f}円（総票数の{per_race / pool * 100:.3f}%）"],
        ],
    )
    rep.p(
        "期待利益には上限がある。賭け金を増やすほど自分でオッズを下げ、ある額を超えると利益が減る。Benter（1994）が自身の期待利益の上限をレース売上の0.25〜0.5%と見積もったのと同じ構造である。"
    )
    rep.metric(
        "capacity",
        {
            "race_share": races_with_bet,
            "median_stake": float(np.median(stakes)),
            "median_profit": float(np.median(profits)),
            "profit_per_race": per_race,
            "share_of_pool": per_race / pool,
        },
    )

    # --- 5. takeout sensitivity
    rep.h2("5. 控除率を変えると")
    rows, tk = [], {}
    q = models["AI（強）ノイズ0.25"]
    for t in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        ot = odds_from_shares(r, t)
        take = mask & (q * ot > 1.1)
        ri, ci = np.where(take)
        true_ev = (p[ri, ci] * ot[ri, ci] - 1).mean() if len(ri) else float("nan")
        prop = np.log(q[idx, w] * odds_from_shares(r, t, False)[idx, w]).mean()
        rows.append([pct(t, 0, sign=False), f"{prop:+.4f}", f"{len(ri):,}", pct(true_ev)])
        tk[t] = {"proportional_W": prop, "selective_bets": len(ri), "selective_true_ev": true_ev}
    rep.table(["控除率", "全頭比例賭けの成長率（実現）", "τ=0.1で賭けた数", "その真の期待値"], rows)
    rep.p(
        "同じAIでも、控除率が低ければ比例賭けでも資金が増える。勝敗を分けるのはAIの精度そのものより、精度の差と控除率の大小関係である。"
    )
    rep.metric("takeout", tk)

    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(ident)
    ax.barh(names, [ident[n]["edge_vs_market_nats"] for n in names], color="tab:blue")
    ax.axvline(-math.log(0.8), color="tab:red", ls="--", label="needed: -ln(1-0.20) = 0.223")
    ax.set_xlabel("information edge over the market (nats per race)")
    ax.legend()
    ax.set_title("Edge vs the takeout bar")
    rep.figure(savefig(fig, "s5_edge_vs_takeout"), "AIの情報優位と控除率の壁")
    plt.close(fig)
    rep.save()


if __name__ == "__main__":
    main()
