"""Study 4 - opponent: the house. Account limits and the censoring of winners.

A bookmaker watches every account and restricts those that look sharp. This
study simulates a population of bettors whose per-bet expected values come from
study 3's real-data results, and asks:
1. How soon is a winning bettor restricted, compared with the number of bets the
   bettor needs to prove the edge to themselves?
2. Do restricted accounts look like the UK Gambling Commission's figures
   (restricted accounts far more often in profit)?
3. Which public track records survive, and how many of the "+10% ROI" records
   come from bettors with a real edge?
"""

from __future__ import annotations

import json

import numpy as np

from . import data
from .common import OUT, Report, num, pct, required_n

N_ACCOUNTS = 50_000
MAX_BETS = 1_000
CHECKPOINTS = (50, 100, 200, 500, 1_000)


def bettor_types() -> dict:
    """Per-bet expected value and odds range for each type. The EVs are the
    closing-line values measured in study 3 where available."""
    s3 = json.loads((OUT / "s3_sports.json").read_text())
    ai_ev = s3["value_bets"]["GBM|b365_open|0.05"]["clv"]
    shop_ev = s3["kaunitz"]["open|0.03"]["clv"]
    sharp_ev = s3["sharp_vs_soft"]["b365_open|0.0"]["clv"]
    return {
        "娯楽の客": {"share": 0.85, "ev": -0.07, "odds": (1.3, 8.0)},
        "AIモデルで賭ける客（研究3のGBM）": {"share": 0.10, "ev": ai_ev, "odds": (2.5, 10.0)},
        "価格を比べる客（研究3のKaunitz型）": {"share": 0.04, "ev": shop_ev, "odds": (1.5, 5.0)},
        "鋭い客（EV +2%）": {"share": 0.01, "ev": max(sharp_ev, 0.02), "odds": (1.5, 3.5)},
    }


def simulate(
    seed: int,
    restrict: bool,
    clv_noise: float = 0.10,
    min_bets: int = 30,
    t_limit: float = 2.5,
    profit_limit: float = 40.0,
) -> dict:
    """Each account places up to MAX_BETS flat bets. The bookmaker sees each bet's
    closing line value with noise (it knows the closing price, not the truth) and
    restricts an account when the CLV t-statistic exceeds `t_limit` after
    `min_bets` bets, or when profit exceeds `profit_limit` units."""
    g = np.random.default_rng(seed)
    real = data.odds(data.load_football(), "avg_open").ravel()
    real = real[~np.isnan(real)]
    types = bettor_types()
    names = list(types)
    shares = np.array([types[n]["share"] for n in names])
    kind = g.choice(len(names), N_ACCOUNTS, p=shares / shares.sum())

    n_bets = np.zeros(N_ACCOUNTS, dtype=int)
    profit = np.zeros(N_ACCOUNTS)
    restricted = np.zeros(N_ACCOUNTS, dtype=bool)
    sig_before = np.zeros(N_ACCOUNTS, dtype=bool)  # own profit significant before restriction
    at = {n: np.zeros(N_ACCOUNTS) for n in CHECKPOINTS}  # cumulative profit after n bets (no limits)
    steps = np.arange(1, MAX_BETS + 1)
    for k, name in enumerate(names):
        lo, hi = types[name]["odds"]
        pool = real[(real >= lo) & (real <= hi)]
        ev = types[name]["ev"]
        all_idx = np.where(kind == k)[0]
        for start in range(0, len(all_idx), 2_000):  # chunks keep memory small
            idx = all_idx[start : start + 2_000]
            odds = g.choice(pool, (len(idx), MAX_BETS))
            p_win = np.clip((1 + ev) / odds, 0, 1)
            pnl = np.where(g.random((len(idx), MAX_BETS)) < p_win, odds - 1, -1.0)
            cum_pnl = np.cumsum(pnl, axis=1)
            if restrict:
                clv = ev + clv_noise * g.standard_normal((len(idx), MAX_BETS))
                mean = np.cumsum(clv, axis=1) / steps
                var = np.maximum(np.cumsum(clv**2, axis=1) / steps - mean**2, 1e-12)
                t = mean / np.sqrt(var / steps)
                flag = ((t > t_limit) & (steps >= min_bets)) | (cum_pnl > profit_limit)
                first = np.where(flag.any(axis=1), flag.argmax(axis=1) + 1, MAX_BETS)
                restricted[idx] = flag.any(axis=1)
            else:
                first = np.full(len(idx), MAX_BETS)
            n_bets[idx] = first
            for n in CHECKPOINTS:
                at[n][idx] = cum_pnl[:, n - 1]
            profit[idx] = cum_pnl[np.arange(len(idx)), first - 1]
            # Did the bettor's own profit reach one-sided 5% significance before the record ended?
            mu = cum_pnl / steps
            sd = np.sqrt(np.maximum(np.cumsum(pnl**2, axis=1) / steps - mu**2, 1e-12))
            sig = (mu / (sd / np.sqrt(steps)) > 1.645) & (steps >= 30) & (steps[None, :] <= first[:, None])
            sig_before[idx] = sig.any(axis=1)
    return {
        "names": names,
        "types": types,
        "kind": kind,
        "n_bets": n_bets,
        "profit": profit,
        "restricted": restricted,
        "sig_before": sig_before,
        "at": at,
    }


def main() -> None:
    rep = Report("s4_house_limits", "研究4　相手が「胴元」：勝つ客の制限と、記録の打ち切り")
    rep.p(
        f"設定：{N_ACCOUNTS:,}口座が、それぞれ最大{MAX_BETS:,}回の均等賭けをする。客の種類と1回あたりの期待値は、研究3の実データの結果（終値に対する期待値、CLV）から取った。賭けるオッズは実データの始値から、種類ごとの範囲で抽出した。"
    )
    rep.p(
        "胴元は、各賭けの終値に対する価値（CLV）を誤差つきで観測し、30回以上賭けた口座のCLVのt値が2.5を超えるか、利益が40単位を超えたら、その口座を制限する（以後の記録は途切れる）。制限の基準は非公開なので、この規則は仮定である。"
    )
    with_limits = simulate(seed=21, restrict=True)
    no_limits = simulate(seed=21, restrict=False)
    names, types = with_limits["names"], with_limits["types"]

    rows, per_type = [], {}
    for k, name in enumerate(names):
        sel = with_limits["kind"] == k
        ev = types[name]["ev"]
        avg_odds = np.mean(types[name]["odds"])
        need = required_n(ev, np.sqrt(avg_odds - 1)) if ev > 0 else float("inf")
        restricted = with_limits["restricted"][sel]
        bets_before = with_limits["n_bets"][sel][restricted]
        rows.append(
            [
                name,
                pct(types[name]["share"], 0, sign=False),
                pct(ev),
                pct(restricted.mean(), 1, sign=False),
                num(np.median(bets_before), 0) if len(bets_before) else "—",
                num(need, 0),
                pct(with_limits["sig_before"][sel].mean(), 1, sign=False),
                pct(no_limits["sig_before"][sel].mean(), 1, sign=False),
            ]
        )
        per_type[name] = {
            "ev": ev,
            "restricted_rate": restricted.mean(),
            "median_bets_before_restriction": float(np.median(bets_before)) if len(bets_before) else None,
            "bets_needed_to_prove": need,
            "own_proof_with_limits": with_limits["sig_before"][sel].mean(),
            "own_proof_without_limits": no_limits["sig_before"][sel].mean(),
        }
    rep.h2("1. 勝つ客は、実力を証明する前に制限される")
    rep.table(
        [
            "客の種類",
            "割合",
            "1回の期待値",
            "制限された割合",
            "制限までの回数（中央値）",
            "自分で優位を証明するのに必要な回数",
            "制限前に自分の利益が有意になった割合",
            "（制限がない場合）",
        ],
        rows,
    )
    rep.p(
        "価格を比べる客と鋭い客の多くは、自分の利益が統計的に有意になる前に制限される。胴元は全口座の終値との比較を見られるので、本人より早く実力を見抜ける。なお、期待値がマイナスの客でも数%が「有意」になるのは、成績を毎回見直すと偶然の有意が積み重なるため（多重検定）。"
    )

    rep.h2("2. 制限された口座は黒字が多い")
    profit_all = (with_limits["profit"] > 0).mean()
    r = with_limits["restricted"]
    profit_restricted = (with_limits["profit"][r] > 0).mean()
    rep.table(
        ["", "黒字の割合"],
        [
            ["全口座", pct(profit_all, 1, sign=False)],
            ["制限された口座", pct(profit_restricted, 1, sign=False)],
            ["制限された口座の割合", pct(r.mean(), 2, sign=False)],
            ["参考：英国賭博委員会（2024年）全口座／制限口座", "25.42%／46.78%（制限は全口座の4.31%）"],
        ],
    )
    rep.p("制限された口座には、実力のある客と、たまたま利益が基準を超えた娯楽の客が混ざる。")

    rep.h2("3. 残る記録と消える記録")
    rep.p(
        "実際に賭けない「予想の記録」（紙上の記録）は制限されない。その記録でROI +10%以上を示した口座のうち、本当に期待値がプラスの客の割合を、記録の長さ別に見る（制限なしのシミュレーション）。"
    )
    pos_kinds = [k for k, nm in enumerate(names) if types[nm]["ev"] > 0]
    is_pos = np.isin(no_limits["kind"], pos_kinds)
    rows, ads = [], {}
    for n in CHECKPOINTS:
        good = no_limits["at"][n] / n >= 0.10
        share = is_pos[good].mean() if good.any() else float("nan")
        real_money = (with_limits["n_bets"][np.isin(with_limits["kind"], pos_kinds)] >= n).mean()
        rows.append(
            [
                f"{n:,}",
                f"{good.sum():,}",
                pct(share, 1, sign=False),
                pct(1 - share, 1, sign=False),
                pct(real_money, 1, sign=False),
            ]
        )
        ads[n] = {
            "records_roi_ge_10pct": int(good.sum()),
            "share_true_edge": share,
            "edge_bettors_reaching_n_with_limits": real_money,
        }
    rep.table(
        [
            "記録の長さ（回）",
            "ROI +10%以上の記録",
            "うち期待値がプラスの客",
            "うち期待値がマイナスの客（運）",
            "期待値がプラスの客が実資金でこの長さの記録を作れる割合",
        ],
        rows,
    )
    rep.p(
        "短い記録ほど、好成績の大半は運による。長い記録なら本物の勝者が残るが、実資金では、本物の勝者ほど制限されてその長さに届かない。宣伝に使われやすいのは短い記録と紙上の記録であり、どちらも実力の証拠として弱い。"
    )
    rep.metric("by_type", per_type)
    rep.metric(
        "restricted",
        {"share_restricted": r.mean(), "profit_rate_all": profit_all, "profit_rate_restricted": profit_restricted},
    )
    rep.metric("visible_records", ads)
    rep.save()


if __name__ == "__main__":
    main()
