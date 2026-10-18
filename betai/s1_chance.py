"""Study 1 - opponent: mathematics (roulette and slot machines).

1. Can a learning agent extract an edge from a certified random number generator?
   The same agent is also run against two flawed sources (a biased wheel and a
   weak PRNG) to show that any win comes from the flaw, not from the agent.
2. Why a high session win rate does not mean winning (flat betting vs martingale).
3. Bet-size freedom on a negative-EV slot (the design of Lee et al. 2025) and ruin.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import special

from .common import Report, bootstrap_ci, ci_str, num, pct, savefig

N_POCKETS = 37  # European roulette
PAYOUT = 36  # a winning straight-up bet returns 36x the stake (35:1 plus stake)
FAIR_P = 1 / 36  # break-even hit probability for a straight-up bet
SPINS = 200_000
WHEEL_NEIGHBOURS = [17, 34, 6]  # adjacent pockets on a European wheel


# ---------------------------------------------------------------------------
# Outcome sources
# ---------------------------------------------------------------------------


def certified(seed: int, n: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, N_POCKETS, n)


def biased_wheel(seed: int, n: int, tilt: float = 1.10) -> np.ndarray:
    """Three neighbouring pockets come up 10% more often (a worn or tilted wheel)."""
    w = np.ones(N_POCKETS)
    w[WHEEL_NEIGHBOURS] *= tilt
    return np.random.default_rng(seed).choice(N_POCKETS, n, p=w / w.sum())


def weak_prng(seed: int, n: int) -> np.ndarray:
    """A 12-bit linear congruential generator (period 4,096) mapped to 0..36."""
    m, a, c = 4096, 1665, 1013
    x = seed % m
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        x = (a * x + c) % m
        out[i] = (x * N_POCKETS) >> 12
    return out


SOURCES = {
    "certified RNG": certified,
    "biased wheel (3 pockets +10%)": biased_wheel,
    "weak PRNG (12-bit LCG)": weak_prng,
}


# ---------------------------------------------------------------------------
# Learning agents
# ---------------------------------------------------------------------------


def prob_above(hits: float, n: float, prior: float) -> float:
    """Posterior P(p > 1/36) for one pocket under a Beta prior."""
    a = hits + prior
    b = n - hits + prior * (N_POCKETS - 1)
    return float(1.0 - special.betainc(a, b, FAIR_P))


def run_agent(seq: np.ndarray, context: int, confidence: float) -> dict:
    """Online agent that predicts the next pocket from the previous `context` pockets.

    context=0 is a frequency model, context=2 a second-order Markov model. Each spin
    the agent takes the most frequent next pocket in its counts. A *forced* agent
    always bets 1 unit on it; a *selective* agent bets only when the posterior
    probability that the pocket beats the break-even rate exceeds `confidence`.
    """
    shape = (N_POCKETS,) * context + (N_POCKETS,)
    counts = np.zeros(shape, dtype=np.int64)
    prior = 0.5
    forced_pnl = np.zeros(len(seq))
    sel_pnl = np.zeros(len(seq))
    sel_bet = np.zeros(len(seq), dtype=bool)
    for t in range(len(seq)):
        if t >= context:
            key = tuple(seq[t - context : t])
            c = counts[key]
            k = int(c.argmax())
            win = PAYOUT - 1 if seq[t] == k else -1
            forced_pnl[t] = win
            n_ctx = c.sum()
            if n_ctx > 0 and prob_above(c[k], n_ctx, prior) > confidence:
                sel_bet[t] = True
                sel_pnl[t] = win
            counts[key + (seq[t],)] += 1
    return {"forced": forced_pnl, "selective": sel_pnl, "selective_mask": sel_bet}


def mutual_information(seq: np.ndarray, context: int, seed: int = 0, n_perm: int = 50) -> tuple[float, float]:
    """Plug-in MI (nats) between the context and the next outcome, minus its mean
    on shuffled data, and a permutation p-value."""
    n_ctx = N_POCKETS**context
    ctx = np.zeros(len(seq) - context, dtype=np.int64)
    for j in range(context):
        ctx = ctx * N_POCKETS + seq[j : len(seq) - context + j]
    nxt = seq[context:]

    def mi(c, y):
        joint = np.bincount(c * N_POCKETS + y, minlength=n_ctx * N_POCKETS).astype(float)
        n = joint.sum()
        pj = joint / n
        pc = pj.reshape(n_ctx, N_POCKETS).sum(1)
        py = pj.reshape(n_ctx, N_POCKETS).sum(0)
        nz = pj > 0
        outer = np.outer(pc, py).ravel()
        raw = float((pj[nz] * np.log(pj[nz] / outer[nz])).sum())
        # Miller-Madow: subtract (cells_joint - cells_c - cells_y + 1) / (2n)
        k_j, k_c, k_y = nz.sum(), (pc > 0).sum(), (py > 0).sum()
        return raw - (k_j - k_c - k_y + 1) / (2 * n)

    obs = mi(ctx, nxt)
    g = np.random.default_rng(seed)
    perm = [mi(ctx, g.permutation(nxt)) for _ in range(n_perm)]
    p = (1 + sum(v >= obs for v in perm)) / (n_perm + 1)
    # Report the excess over the shuffled-data mean: the plug-in estimate is biased
    # upward when there are many context cells, and shuffling measures that bias.
    return obs - float(np.mean(perm)), p


# ---------------------------------------------------------------------------
# Session win rate vs expected value
# ---------------------------------------------------------------------------


def roulette_sessions(n_sessions: int, seed: int) -> dict:
    """Even-money bets on red (18/37) in sessions of up to 200 spins."""
    g = np.random.default_rng(seed)
    p = 18 / 37
    res = {}
    # Flat: 1 unit per spin for 200 spins.
    wins = g.random((n_sessions, 200)) < p
    flat_pnl = np.where(wins, 1, -1).sum(axis=1)
    res["flat 1 unit x 200 spins"] = (flat_pnl, np.full(n_sessions, 200.0))
    # Martingale: double after each loss, stop at +20 units or when the next bet is unaffordable.
    pnl = np.zeros(n_sessions)
    turnover = np.zeros(n_sessions)
    for s in range(n_sessions):
        bank, bet, start = 255.0, 1.0, 255.0
        for _ in range(200):
            if bank - start >= 20 or bet > bank:
                break
            turnover[s] += bet
            if g.random() < p:
                bank += bet
                bet = 1.0
            else:
                bank -= bet
                bet *= 2
        pnl[s] = bank - start
    res["martingale (bank 255, target +20)"] = (pnl, turnover)
    return res


def slot_policies(n_sessions: int, seed: int, rounds: int) -> dict:
    """Slot with 30% hits paying 3x (EV -10%), $100 bankroll, up to `rounds` rounds."""
    g = np.random.default_rng(seed)
    out = {}
    policies = {
        "fixed $10": lambda bank, streak: 10.0,
        "double after loss": lambda bank, streak: 10.0 * 2**streak,
        "win back the deficit in one hit": lambda bank, streak: max(10.0, (100.0 - bank) / 2 + 10.0),
        "Kelly (EV<0, bet nothing)": lambda bank, streak: 0.0,
    }
    for name, policy in policies.items():
        final = np.zeros(n_sessions)
        ruined = np.zeros(n_sessions, dtype=bool)
        turnover = np.zeros(n_sessions)
        for s in range(n_sessions):
            bank, streak = 100.0, 0
            for _ in range(rounds):
                bet = min(bank, policy(bank, streak))
                if bet <= 0:
                    break
                turnover[s] += bet
                if g.random() < 0.30:
                    bank += 2 * bet
                    streak = 0
                else:
                    bank -= bet
                    streak += 1
                if bank < 1:
                    ruined[s] = True
                    break
            final[s] = bank
        out[name] = {"final": final, "ruined": ruined, "turnover": turnover}
    return out


# ---------------------------------------------------------------------------


def main() -> None:
    rep = Report("s1_chance", "研究1　相手が「数学」：ルーレットとスロット")
    rep.p(
        "目的：認証された乱数に対して、学習するAIが優位を得られるかを確かめる。同じAIを、欠陥のある乱数源（偏ったホイール、弱い疑似乱数）にも当て、勝ちの原因がAIの賢さか、相手の欠陥かを切り分ける。"
    )
    rep.p(
        f"設定：ヨーロピアン・ルーレット（37ポケット）の1点賭け（当たれば36倍、期待値 {pct(36 / 37 - 1)}）。各乱数源で {SPINS:,} 回転。AIは直前k回の出目から次の出目を数え上げで学習し、最も多い出目に賭ける（k=0：頻度モデル、k=2：2次マルコフモデル）。「強制」は毎回賭け、「選択的」は事後確率で損益分岐（1/36）を99%以上の確度で超えるときだけ賭ける。"
    )

    rows, curves, summary = [], {}, {}
    for si, (src_name, fn) in enumerate(SOURCES.items()):
        seq = fn(1000 + si, SPINS)
        mi0, p0 = mutual_information(seq, 1, seed=si)
        mi2, p2 = mutual_information(seq, 2, seed=si)
        freq_dev = np.bincount(seq, minlength=N_POCKETS) / len(seq) * N_POCKETS
        for ctx in (0, 2):
            r = run_agent(seq, ctx, confidence=0.99)
            for mode in ("forced", "selective"):
                pnl = r[mode] if mode == "forced" else r["selective"][r["selective_mask"]]
                if mode == "forced":
                    pnl = pnl[ctx:]
                n_bets = len(pnl)
                if n_bets == 0:
                    rows.append([src_name, f"k={ctx}", mode, 0, "—", "—", "—", "—"])
                    summary[f"{src_name}|k={ctx}|{mode}"] = {"bets": 0}
                    continue
                roi = pnl.mean()
                lo, hi = bootstrap_ci(pnl, seed=si)
                late = pnl[len(pnl) // 2 :]
                hit = (pnl > 0).mean()
                rows.append(
                    [
                        src_name,
                        f"k={ctx}",
                        mode,
                        f"{n_bets:,}",
                        pct(hit, sign=False),
                        pct(roi),
                        ci_str(lo, hi),
                        pct(late.mean()),
                    ]
                )
                summary[f"{src_name}|k={ctx}|{mode}"] = {
                    "bets": n_bets,
                    "hit_rate": hit,
                    "roi": roi,
                    "roi_ci": [lo, hi],
                    "roi_second_half": late.mean(),
                }
                if mode == "forced":
                    curves[(src_name, ctx)] = np.cumsum(r["forced"])
        summary[f"{src_name}|MI"] = {
            "mi_lag1_nats": mi0,
            "p_lag1": p0,
            "mi_ctx2_nats": mi2,
            "p_ctx2": p2,
            "max_pocket_freq_ratio": float(freq_dev.max()),
        }

    rep.h2("1. 学習するAIは乱数に勝てるか")
    rep.table(["乱数源", "文脈", "賭け方", "賭けた回数", "的中率", "ROI", "ROI 95%CI", "後半のROI"], rows)
    mi_rows = []
    for src_name in SOURCES:
        m = summary[f"{src_name}|MI"]
        mi_rows.append(
            [
                src_name,
                f"{m['mi_lag1_nats']:.5f}",
                num(m["p_lag1"], 3),
                f"{m['mi_ctx2_nats']:.5f}",
                num(m["p_ctx2"], 3),
                f"{m['max_pocket_freq_ratio']:.3f}",
            ]
        )
    rep.h3("過去の出目が次の出目について持つ情報量（相互情報量）")
    rep.p(
        "相互情報量はプラグイン推定（単位ナッツ）から、次の出目を並べ替えたデータでの平均値（推定の上方バイアス）を引いた値。p値は50回の置換検定（最小値は1/51≈0.020）。最後の列は最も多く出たポケットの出現率 ÷ 1/37。"
    )
    rep.table(["乱数源", "直前1回との情報量", "p値", "直前2回との情報量", "p値", "最頻ポケットの倍率"], mi_rows)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    styles = {
        ("certified RNG", 0): "tab:blue",
        ("biased wheel (3 pockets +10%)", 0): "tab:orange",
        ("biased wheel (3 pockets +10%)", 2): "tab:orange",
        ("weak PRNG (12-bit LCG)", 2): "tab:green",
    }
    for (name, ctx), color in styles.items():
        c = curves[(name, ctx)]
        ax.plot(
            np.arange(len(c)),
            c,
            color=color,
            ls="-" if ctx == 0 or "PRNG" in name else ":",
            label=f"{name}, {'frequency model (k=0)' if ctx == 0 else '2nd-order Markov (k=2)'}",
        )
    ax.axhline(0, color="k", lw=0.8)
    ax.set_yscale("symlog", linthresh=1000)
    ax.set_xlabel("spins")
    ax.set_ylabel("cumulative P&L (units, symlog)")
    ax.set_title("Learning agents against three random sources (forced to bet every spin)")
    ax.legend(fontsize=8)
    rep.figure(savefig(fig, "s1_learning_curves"), "同じ学習AIを3つの乱数源に当てた累積損益")
    plt.close(fig)

    cert_forced = summary["certified RNG|k=2|forced"]
    cert_sel = summary["certified RNG|k=2|selective"]
    rep.p("読み取り：")
    rep.bullets(
        [
            f"認証された乱数では、どの学習AIもROIの信頼区間が理論値 {pct(36 / 37 - 1)} を含み、優位は出ない（2次マルコフ・強制で {pct(cert_forced['roi'])}）。過去の出目と次の出目の情報量は統計的にゼロと区別できない。",
            f"選択的なAIのうち、頻度モデルは {summary['certified RNG|k=0|selective']['bets']:,} 回しか賭けず、ほぼ「賭けない」を選ぶ。一方、2次マルコフモデルは {cert_sel.get('bets', 0):,} 回（全体の約{cert_sel.get('bets', 0) / SPINS * 100:.0f}%）賭けた。1,369通りの文脈×37ポケットという大量の候補から最大値を選ぶため、偶然の偏りを「優位」と誤認する。複雑なモデルほど、存在しないパターンに賭けやすい。",
            "偏ったホイールでは、頻度モデルが偏ったポケットを見つけて勝つ。勝因はAIではなくホイールの物理的な欠陥で、偏りが小さいほど検出に回転数が要る。",
            "弱い疑似乱数では、直前2回の出目から次がほぼ決まり、2次マルコフモデルが大勝する。頻度モデル（文脈なし）は同じ乱数で勝てない。勝因は乱数の実装の欠陥と、それに合ったモデル構造の組み合わせ。",
        ]
    )
    rep.metric("roulette", summary)

    # --- 2. session win rate vs EV
    rep.h2("2. 勝率が高くても勝てない：均等賭けとマーチンゲール")
    rep.p(
        "ルーレットの赤黒（当たる確率18/37）。均等賭けは1単位×200回。マーチンゲールは負けるたびに賭け金を2倍にし、+20単位に達するか、次の賭け金を払えなくなったら終える（資金255単位、最大200回）。各10万セッション。"
    )
    sess = roulette_sessions(100_000, seed=7)
    rows = []
    mg = {}
    for name, (pnl, turnover) in sess.items():
        win_rate = (pnl > 0).mean()
        per_unit = pnl.sum() / turnover.sum()
        rows.append(
            [
                name,
                pct(win_rate, 1, sign=False),
                num(pnl.mean(), 2),
                num(np.median(pnl), 1),
                num(pnl.min(), 0),
                pct((pnl <= -100).mean(), 2, sign=False),
                pct(per_unit),
            ]
        )
        mg[name] = {
            "session_win_rate": win_rate,
            "mean_pnl": pnl.mean(),
            "loss_per_unit_wagered": per_unit,
            "p_loss_ge_100": (pnl <= -100).mean(),
        }
    rep.table(
        [
            "賭け方",
            "勝ったセッションの割合",
            "平均損益（単位）",
            "中央値",
            "最悪",
            "100単位以上負けた割合",
            "賭け金1単位あたりの損益",
        ],
        rows,
    )
    rep.p(
        f"マーチンゲールは約{pct(mg['martingale (bank 255, target +20)']['session_win_rate'], 0, sign=False)}のセッションで勝つが、賭け金1単位あたりの損益はどちらも理論値 {pct(18 / 37 * 2 - 1)} の近くにとどまる。賭け方は「勝つ頻度」と「負け方の形」を変えるだけで、期待値は変えない。"
    )
    rep.metric("martingale", mg)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    for name, (pnl, _) in sess.items():
        ax.hist(pnl, bins=120, alpha=0.6, label=name, density=True)
    ax.set_yscale("log")
    ax.set_xlabel("session P&L (units)")
    ax.set_ylabel("density (log)")
    ax.legend()
    ax.set_title("High win rate, same expected loss")
    rep.figure(savefig(fig, "s1_martingale"), "均等賭けとマーチンゲールのセッション損益分布")
    plt.close(fig)

    # --- 3. bet sizing freedom on a slot
    rep.h2("3. 賭け金の自由：期待値−10%のスロット")
    rep.p(
        "Lee et al.（2025）と同じ設計（当たる確率30%、当たれば賭け金の3倍が戻る、期待値−10%、初期資金100ドル）で、賭け金の決め方だけを変える。LLMそのものではなく、LLMが示した行動パターン（損失を取り返そうとする）をルールとして与えた。途中でやめる判断は入れず、20回で切り上げる短いセッションと100回続ける長いセッションを比べる。各2万セッション。"
    )
    rows, sp = [], {}
    for rounds in (20, 100):
        slots = slot_policies(20_000, seed=11 + rounds, rounds=rounds)
        for name, r in slots.items():
            ruin = r["ruined"].mean()
            up = (r["final"] > 100).mean()
            rows.append(
                [
                    f"{rounds}回",
                    name,
                    pct(ruin, 1, sign=False),
                    pct(up, 1, sign=False),
                    num(np.median(r["final"]), 1),
                    num(r["turnover"].mean(), 0),
                ]
            )
            sp[f"{rounds}|{name}"] = {
                "ruin_rate": ruin,
                "p_end_above_start": up,
                "median_final": float(np.median(r["final"])),
                "mean_turnover": float(r["turnover"].mean()),
            }
    rep.table(
        ["回数", "賭け金の決め方", "破産率", "100ドルより増えて終えた割合", "最終資金の中央値", "平均の賭け金総額"],
        rows,
    )
    rep.p(
        "期待値が負のゲームでは、ケリー基準の答えは「賭けない」。短いセッションでは、負けに応じて賭け金を増やす方針が破産率を大きく引き上げる。長く続ければ、固定額でも期待損失（1回1ドル×回数）が資金を食いつぶすので、破産を防ぐのは賭け方ではなく「やめること」になる。"
    )
    rep.metric("slot", sp)
    rep.save()


if __name__ == "__main__":
    main()
