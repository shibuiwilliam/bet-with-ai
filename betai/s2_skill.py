"""Study 2 - opponent: people (poker and mahjong).

Poker: Kuhn poker (3 cards, 1-chip ante, 1-chip bet) is small enough to solve
exactly, so every expected value below is computed by enumeration, not sampled.
- CFR learns the equilibrium (the "GTO AI").
- Four human-like opponent profiles with typical leaks.
- An oracle exploiter (best response) and an online learner that estimates the
  opponent's leaks from play and best-responds to them.
- Rake, variance and the hands needed to prove an edge.
- Attribution: how much of the AI's profit each opponent leak supplies.

Mahjong: a placement model fitted to the Suphx paper's rank rates, to measure how
many games are needed before a superhuman edge is visible.
"""

from __future__ import annotations

import itertools
import zlib

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from .common import Report, num, pct, required_n, savefig

CARDS = ["J", "Q", "K"]
DEALS = list(itertools.permutations(range(3), 2))  # (P1 card, P2 card)
INFOSETS = [(h, c) for h in ("", "pb", "p", "b") for c in range(3)]
OWNER = {"": 0, "pb": 0, "p": 1, "b": 1}
TERMINAL = ("pp", "bp", "bb", "pbp", "pbb")  # ordered, so sums are reproducible

LABELS = {
    ("", 0): "P1: J で先にベット（ブラフ）",
    ("", 1): "P1: Q で先にベット",
    ("", 2): "P1: K で先にベット",
    ("pb", 0): "P1: J でベットにコール",
    ("pb", 1): "P1: Q でベットにコール",
    ("pb", 2): "P1: K でベットにコール",
    ("p", 0): "P2: J でチェック後にベット（ブラフ）",
    ("p", 1): "P2: Q でチェック後にベット",
    ("p", 2): "P2: K でチェック後にベット",
    ("b", 0): "P2: J でベットにコール",
    ("b", 1): "P2: Q でベットにコール",
    ("b", 2): "P2: K でベットにコール",
}


def payoffs(h: str, c1: int, c2: int, rake: float) -> tuple[float, float]:
    """Net chips for (P1, P2) at a terminal history. The winner receives the pot minus rake."""
    contrib = [1.0, 1.0]
    for i, a in enumerate(h):
        if a == "b":
            contrib[i % 2] += 1.0
    pot = sum(contrib)
    if h == "bp":
        winner = 0
    elif h == "pbp":
        winner = 1
    else:
        winner = 0 if c1 > c2 else 1
    out = [-contrib[0], -contrib[1]]
    out[winner] += pot * (1 - rake)
    return out[0], out[1]


def action_prob(strategy: dict, h: str, card: int, a: str) -> float:
    p_bet = strategy[(h, card)]
    return p_bet if a == "b" else 1 - p_bet


def expected_values(s1: dict, s2: dict, rake: float = 0.0) -> tuple[float, float]:
    """Exact expected net chips for (P1, P2) when P1 plays s1 and P2 plays s2."""
    ev = np.zeros(2)
    for c1, c2 in DEALS:
        for h in TERMINAL:
            prob = 1.0
            for i, a in enumerate(h):
                hist = h[:i]
                card = c1 if OWNER[hist] == 0 else c2
                strat = s1 if OWNER[hist] == 0 else s2
                prob *= action_prob(strat, hist, card, a)
            if prob:
                ev += prob * np.array(payoffs(h, c1, c2, rake)) / len(DEALS)
    return float(ev[0]), float(ev[1])


def seat_average(ai: tuple[dict, dict], opp: dict, rake: float = 0.0) -> tuple[float, float]:
    """AI plays each seat half the time against one opponent strategy dict.
    Returns (AI EV, opponent EV) per hand."""
    a1, o2 = expected_values(ai[0], opp, rake)
    o1, a2 = expected_values(opp, ai[1], rake)
    return (a1 + a2) / 2, (o2 + o1) / 2


# ---------------------------------------------------------------------------
# CFR and best response
# ---------------------------------------------------------------------------


def cfr(iterations: int = 20_000) -> tuple[dict, list[float]]:
    """Vanilla counterfactual regret minimisation. Returns the average strategy
    as a single dict over all 12 information sets and the exploitability curve."""
    regret = {k: np.zeros(2) for k in INFOSETS}
    strat_sum = {k: np.zeros(2) for k in INFOSETS}

    def current(k):
        r = np.maximum(regret[k], 0)
        return r / r.sum() if r.sum() > 0 else np.array([0.5, 0.5])

    def walk(h, c1, c2, reach):
        if h in TERMINAL:
            return np.array(payoffs(h, c1, c2, 0.0))
        player = OWNER[h]
        k = (h, c1 if player == 0 else c2)
        sigma = current(k)
        values = np.zeros((2, 2))
        node = np.zeros(2)
        for ai, a in enumerate("pb"):
            new_reach = reach.copy()
            new_reach[player] *= sigma[ai]
            values[ai] = walk(h + a, c1, c2, new_reach)
            node += sigma[ai] * values[ai]
        regret[k] += reach[1 - player] * (values[:, player] - node[player])
        strat_sum[k] += reach[player] * sigma
        return node

    curve = []
    for it in range(1, iterations + 1):
        for c1, c2 in DEALS:
            walk("", c1, c2, np.ones(2))
        if it in (10, 30, 100, 300, 1000, 3000, 10000, iterations):
            avg = {k: strat_sum[k][1] / strat_sum[k].sum() for k in INFOSETS}
            curve.append((it, exploitability(avg)))
    avg = {k: strat_sum[k][1] / strat_sum[k].sum() for k in INFOSETS}
    return avg, curve


def best_response(opp: dict, br_player: int, rake: float = 0.0) -> dict:
    """Best response of `br_player` against the fixed strategy `opp` (12-infoset dict).
    Returns a full strategy dict where only `br_player`'s entries matter."""
    br = dict(opp)

    def u(h, c1, c2):
        return payoffs(h, c1, c2, rake)[br_player]

    if br_player == 0:
        for c1 in range(3):
            others = [c for c in range(3) if c != c1]
            # Decision after check-bet: call or fold, weighted by P2 betting with each card.
            w = [opp[("p", c2)] for c2 in others]
            call = sum(wi * u("pbb", c1, c2) for wi, c2 in zip(w, others, strict=True))
            fold = sum(wi * u("pbp", c1, c2) for wi, c2 in zip(w, others, strict=True))
            br[("pb", c1)] = 1.0 if call > fold else 0.0
            best_pb = max(call, fold)
            check = sum((1 - opp[("p", c2)]) * u("pp", c1, c2) for c2 in others) + best_pb
            bet = sum(opp[("b", c2)] * u("bb", c1, c2) + (1 - opp[("b", c2)]) * u("bp", c1, c2) for c2 in others)
            br[("", c1)] = 1.0 if bet > check else 0.0
    else:
        for c2 in range(3):
            others = [c for c in range(3) if c != c2]
            reach_p = [1 - opp[("", c1)] for c1 in others]
            check = sum(r * u("pp", c1, c2) for r, c1 in zip(reach_p, others, strict=True))
            bet = sum(
                r * (opp[("pb", c1)] * u("pbb", c1, c2) + (1 - opp[("pb", c1)]) * u("pbp", c1, c2))
                for r, c1 in zip(reach_p, others, strict=True)
            )
            br[("p", c2)] = 1.0 if bet > check else 0.0
            reach_b = [opp[("", c1)] for c1 in others]
            call = sum(r * u("bb", c1, c2) for r, c1 in zip(reach_b, others, strict=True))
            fold = sum(r * u("bp", c1, c2) for r, c1 in zip(reach_b, others, strict=True))
            br[("b", c2)] = 1.0 if call > fold else 0.0
    return br


def exploitability(s: dict) -> float:
    """Average gain per hand of a best responder against `s` playing both seats."""
    br1 = best_response(s, 0)
    br2 = best_response(s, 1)
    v1, _ = expected_values(br1, s)
    _, v2 = expected_values(s, br2)
    return (v1 + v2) / 2


def exploiter(opp: dict, rake: float = 0.0) -> tuple[dict, dict]:
    return best_response(opp, 0, rake), best_response(opp, 1, rake)


def shapley_attribution(opp: dict, gto: dict) -> tuple[np.ndarray, float]:
    """Shapley value of each opponent information set for the exploiter's EV.

    v(S) is the best-response EV when the opponent keeps its own strategy on the
    information sets in S and plays the equilibrium elsewhere."""
    n = len(INFOSETS)
    v = np.zeros(2**n)
    for mask in range(2**n):
        mixed = {k: (opp[k] if mask >> i & 1 else gto[k]) for i, k in enumerate(INFOSETS)}
        v[mask], _ = seat_average(exploiter(mixed), mixed)
    fact = [1.0]
    for i in range(1, n + 1):
        fact.append(fact[-1] * i)
    phi = np.zeros(n)
    for mask in range(2**n):
        size = bin(mask).count("1")
        for i in range(n):
            if not mask >> i & 1:
                w = fact[size] * fact[n - size - 1] / fact[n]
                phi[i] += w * (v[mask | 1 << i] - v[mask])
    return phi, float(v[-1] - v[0])


# ---------------------------------------------------------------------------
# Opponent profiles (one dict covers both seats)
# ---------------------------------------------------------------------------


def profile(p1_open, p1_call, p2_bet, p2_call) -> dict:
    s = {}
    for c in range(3):
        s[("", c)] = p1_open[c]
        s[("pb", c)] = p1_call[c]
        s[("p", c)] = p2_bet[c]
        s[("b", c)] = p2_call[c]
    return s


OPPONENTS = {
    "コーリングステーション（何でもコール）": profile([0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]),
    "ロック（強い手以外は降りる）": profile([0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]),
    "マニアック（常にベット・コール）": profile([1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]),
    "ブラフ好き（Jで頻繁にブラフ）": profile([0.6, 0, 1], [0, 0.5, 1], [0.6, 0, 1], [0, 0.6, 1]),
}


# ---------------------------------------------------------------------------
# Simulation (for win rate, variance and online learning)
# ---------------------------------------------------------------------------


def play_hand(g, s1: dict, s2: dict, rake: float) -> tuple[float, float, int, int, str]:
    c1, c2 = DEALS[g.integers(len(DEALS))]
    h = ""
    while h not in TERMINAL:
        player = OWNER[h]
        card = c1 if player == 0 else c2
        strat = s1 if player == 0 else s2
        h += "b" if g.random() < strat[(h, card)] else "p"
    u1, u2 = payoffs(h, c1, c2, rake)
    return u1, u2, c1, c2, h


def simulate(ai: tuple[dict, dict], opp: dict, hands: int, rake: float, seed: int) -> np.ndarray:
    g = np.random.default_rng(seed)
    res = np.empty(hands)
    for t in range(hands):
        if t % 2 == 0:
            u1, _, *_ = play_hand(g, ai[0], opp, rake)
            res[t] = u1
        else:
            _, u2, *_ = play_hand(g, opp, ai[1], rake)
            res[t] = u2
    return res


def online_learner(gto: dict, opp: dict, hands: int, rake: float, seed: int, refit_every: int = 100) -> np.ndarray:
    """Starts as GTO, counts the opponent's actions per information set (cards are
    shown after each hand, an assumption that favours the AI), and best-responds
    to the posterior mean, re-fitted every `refit_every` hands."""
    g = np.random.default_rng(seed)
    prior = 4.0
    bets = {k: prior * gto[k] for k in INFOSETS}
    seen = dict.fromkeys(INFOSETS, prior)
    ai = (gto, gto)
    res = np.empty(hands)
    for t in range(hands):
        ai_first = t % 2 == 0
        s1, s2 = (ai[0], opp) if ai_first else (opp, ai[1])
        c1, c2 = DEALS[g.integers(len(DEALS))]
        h = ""
        while h not in TERMINAL:
            player = OWNER[h]
            card = c1 if player == 0 else c2
            strat = s1 if player == 0 else s2
            a = "b" if g.random() < strat[(h, card)] else "p"
            opp_turn = (player == 1) if ai_first else (player == 0)
            if opp_turn:
                seen[(h, card)] += 1
                bets[(h, card)] += a == "b"
            h += a
        u1, u2 = payoffs(h, c1, c2, rake)
        res[t] = u1 if ai_first else u2
        if (t + 1) % refit_every == 0:
            model = {k: bets[k] / seen[k] for k in INFOSETS}
            ai = exploiter(model, rake)
    return res


# ---------------------------------------------------------------------------
# Mahjong placement model
# ---------------------------------------------------------------------------

SUPHX = np.array([0.293, 0.275, 0.244, 0.187])
TOP_HUMAN = np.array([0.280, 0.268, 0.247, 0.205])


def placement_rates(mu: float, sigma: float, n: int = 400_000, seed: int = 0) -> np.ndarray:
    """Rank distribution of a player scoring N(mu, sigma) against three N(0, 1) players."""
    g = np.random.default_rng(seed)
    me = mu + sigma * g.standard_normal(n)
    others = g.standard_normal((n, 3))
    rank = (others > me[:, None]).sum(axis=1)
    return np.bincount(rank, minlength=4) / n


def fit_placement(target: np.ndarray) -> tuple[float, float, np.ndarray]:
    def loss(x):
        r = placement_rates(x[0], np.exp(x[1]), n=200_000)
        return (r[0] - target[0]) ** 2 + (r[3] - target[3]) ** 2

    res = optimize.minimize(
        loss, x0=[0.1, 0.0], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-10, "maxiter": 400}
    )
    mu, sigma = res.x[0], float(np.exp(res.x[1]))
    return mu, sigma, placement_rates(mu, sigma)


def stable_dan(rates: np.ndarray) -> float:
    return (5 * rates[0] + 2 * rates[1]) / rates[3] - 2


# ---------------------------------------------------------------------------


def main() -> None:
    rep = Report("s2_skill", "研究2　相手が「人」：ポーカーと麻雀")

    # --- CFR
    gto, curve = cfr(20_000)
    v1, v2 = expected_values(gto, gto)
    rep.h2("1. ポーカーAIを作る：CFRで均衡戦略を学習する")
    rep.p(
        "クーン・ポーカー（J・Q・Kの3枚、参加費1チップ、ベット1チップ）は、全ての期待値を数え上げで厳密に計算できる最小のポーカーで、ゲーム理論の標準的な検証台である。Libratus・Pluribusと同じ系統のアルゴリズム（反実仮想後悔最小化、CFR）で均衡戦略を学習させた。"
    )
    rep.table(
        ["CFRの反復回数", "搾取可能性（最善の相手が1手あたり得るチップ）"], [[f"{it:,}", f"{e:.5f}"] for it, e in curve]
    )
    rep.p(
        f"学習後の戦略で先手の期待値は {v1:+.4f} チップ/手（理論値 −1/18 = {-1 / 18:+.4f}）。先手と後手を交互に務めれば、均衡戦略どうしの対戦の期待値はゼロになる。"
    )
    rep.metric("cfr", {"exploitability_curve": curve, "p1_value": v1, "theory": -1 / 18})

    # --- vs human-like opponents
    rep.h2("2. 人間らしい癖を持つ相手との対戦")
    rep.p(
        "AIは先手・後手を交互に務める。期待値は数え上げによる厳密値、的中率と標準偏差は各20万手のシミュレーション。「必要な手数」は、その期待値を片側5%・検出力80%で検出するのに要る手数。"
    )
    rows, vs = [], {}
    gto_pair = (gto, gto)
    for name, opp in OPPONENTS.items():
        ex = exploiter(opp)
        for ai_name, ai in (("均衡AI（GTO）", gto_pair), ("搾取AI（相手の癖を完全に知る）", ex)):
            ev, opp_ev = seat_average(ai, opp)
            sim = simulate(ai, opp, 200_000, 0.0, seed=zlib.crc32((name + ai_name).encode()))
            sd = sim.std()
            rows.append(
                [
                    name,
                    ai_name,
                    f"{ev:+.4f}",
                    pct((sim > 0).mean(), 1, sign=False),
                    f"{sd:.3f}",
                    num(required_n(ev, sd), 0),
                ]
            )
            vs[f"{name}|{ai_name}"] = {
                "ev": ev,
                "win_hand_rate": (sim > 0).mean(),
                "sd": sd,
                "hands_needed": required_n(ev, sd),
            }
    ev_self, _ = seat_average(gto_pair, gto)
    rows.append(["均衡AI自身", "均衡AI（GTO）", f"{ev_self:+.4f}", "—", "—", "∞"])
    rep.table(["相手", "AI", "期待値（チップ/手）", "勝った手の割合", "1手の標準偏差", "必要な手数"], rows)
    rep.metric("vs_opponents", vs)

    # --- online learning
    rep.h3("相手の癖をプレイ中に学ぶAI")
    rep.p(
        "均衡戦略から始め、相手の行動を情報集合ごとに数えて、100手ごとに推定した相手への最善応答に切り替える。各ハンドの後に相手の手札が見える（AIに有利な仮定）。各相手と2万手、10回の平均。"
    )
    rows, learn = [], {}
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, opp in OPPONENTS.items():
        runs = np.array([online_learner(gto, opp, 20_000, 0.0, seed=s) for s in range(10)])
        mean_curve = runs.mean(axis=0)
        oracle_ev, _ = seat_average(exploiter(opp), opp)
        gto_ev, _ = seat_average(gto_pair, opp)
        first, last = mean_curve[:1000].mean(), mean_curve[-5000:].mean()
        rows.append([name, f"{gto_ev:+.4f}", f"{first:+.4f}", f"{last:+.4f}", f"{oracle_ev:+.4f}"])
        learn[name] = {"gto_ev": gto_ev, "first_1000": first, "last_5000": last, "oracle": oracle_ev}
        ax.plot(np.convolve(mean_curve, np.ones(500) / 500, mode="valid"), label=name)
    rep.table(["相手", "均衡AIの期待値", "学習AI：最初の1,000手", "学習AI：最後の5,000手", "完全に知る搾取AI"], rows)
    ax.set_xlabel("hands")
    ax.set_ylabel("chips per hand (500-hand moving average)")
    ax.set_title("Online learner: from GTO to exploiting leaks")
    ax.legend(fontsize=7)
    rep.figure(savefig(fig, "s2_online_learner"), "相手の癖を学ぶAIの学習曲線")
    plt.close(fig)
    rep.metric("online_learner", learn)

    # --- attribution
    rep.h2("3. 勝因の分解：AIの利益は相手のどの癖から来るか")
    rep.p(
        "相手の12個の情報集合それぞれについて「癖のある戦略」か「均衡戦略」かを選ぶ全4,096通りで、搾取AIの期待値を計算し、シャープレイ値で各情報集合の寄与に分けた。寄与の合計は、搾取AIの期待値と均衡戦略に対する期待値の差に一致する。"
    )
    attr = {}
    for name, opp in OPPONENTS.items():
        phi, total = shapley_attribution(opp, gto)
        order = np.argsort(phi)[::-1]
        top = [(phi[i], INFOSETS[i]) for i in order if phi[i] > 1e-4][:4]
        neg = [(phi[i], INFOSETS[i]) for i in order if phi[i] < -1e-4]
        attr[name] = {
            "total": total,
            "top": [{"infoset": LABELS[k], "chips": d, "share": d / total} for d, k in top],
            "negative": [{"infoset": LABELS[k], "chips": d} for d, k in neg],
        }
        text = "、".join(
            f"{LABELS[k]}（相手 {opp[k]:.2f}／均衡 {gto[k]:.2f}）{d:+.4f}（{d / total * 100:.0f}%）" for d, k in top
        )
        rep.p(f"**{name}**（寄与の合計 {total:+.4f} チップ/手）：{text}")
    rep.p(
        "どの相手でも、利益の大半は少数の情報集合から来る。強い手でしか参加しない相手からは「Qで降りすぎる」ことが、何でもコールする相手からは「Jでコールする」ことが、AIの利益の源泉になる。AIの勝因は、AI自身の強さではなく、相手の癖の大きさで決まる。"
    )
    rep.metric("attribution", attr)

    # --- rake
    rep.h2("4. 場代（レーキ）の影響")
    rep.p("勝者がポットから一定割合を場に支払う。均衡AIどうしでは双方がレーキの分だけ負ける。")
    rows, rk = [], {}
    for rake in (0.0, 0.025, 0.05, 0.10):
        r = [pct(rake, 1, sign=False)]
        gg, _ = seat_average(gto_pair, gto, rake)
        r.append(f"{gg:+.4f}")
        rk[f"{rake}|gto_vs_gto"] = gg
        for name, opp in OPPONENTS.items():
            ex = exploiter(opp, rake)
            ev, _ = seat_average(ex, opp, rake)
            r.append(f"{ev:+.4f}")
            rk[f"{rake}|{name}"] = ev
        rows.append(r)
    rep.table(["レーキ", "均衡AI 対 均衡AI"] + [f"搾取AI 対 {n.split('（')[0]}" for n in OPPONENTS], rows)
    rep.metric("rake", rk)

    # --- mahjong
    rep.h2("5. 麻雀：超人AIでも1回の対局では7割負ける")
    mu_s, sg_s, r_s = fit_placement(SUPHX)
    mu_h, sg_h, r_h = fit_placement(TOP_HUMAN)
    rep.p(
        "各打ち手の1局の得点を正規分布とし、平均3人の打ち手（平均0・標準偏差1）と卓を囲む模型に、Suphx論文の順位率（1位と4位）を当てはめた。"
    )
    rep.table(
        ["打ち手", "得点の平均", "得点の標準偏差", "1位", "2位", "3位", "4位", "安定段位"],
        [
            ["Suphx（論文値）", "—", "—"] + [pct(x, 1, sign=False) for x in SUPHX] + [f"{stable_dan(SUPHX):.2f}"],
            ["Suphx（模型）", f"{mu_s:+.3f}", f"{sg_s:.3f}"]
            + [pct(x, 1, sign=False) for x in r_s]
            + [f"{stable_dan(r_s):.2f}"],
            ["トップ人間（論文値）", "—", "—"]
            + [pct(x, 1, sign=False) for x in TOP_HUMAN]
            + [f"{stable_dan(TOP_HUMAN):.2f}"],
            ["トップ人間（模型）", f"{mu_h:+.3f}", f"{sg_h:.3f}"]
            + [pct(x, 1, sign=False) for x in r_h]
            + [f"{stable_dan(r_h):.2f}"],
            ["平均的な打ち手", "0", "1"] + ["25.0%"] * 4 + [f"{stable_dan(np.full(4, 0.25)):.2f}"],
        ],
    )
    g = np.random.default_rng(5)
    rows, mj = [], {}
    for n_games in (100, 500, 1000, 2000, 5760, 10000, 20000):
        sims = 4000
        a = g.multinomial(n_games, SUPHX, size=sims)
        h = g.multinomial(n_games, TOP_HUMAN, size=sims)
        da = (5 * a[:, 0] + 2 * a[:, 1]) / np.maximum(a[:, 3], 1) - 2
        dh = (5 * h[:, 0] + 2 * h[:, 1]) / np.maximum(h[:, 3], 1) - 2
        lo, hi = np.quantile(da, [0.025, 0.975])
        p_ahead = (da > dh).mean()
        rows.append([f"{n_games:,}", f"[{lo:.2f}, {hi:.2f}]", pct(p_ahead, 1, sign=False)])
        mj[n_games] = {"dan_ci": [lo, hi], "p_ai_ahead": p_ahead}
    rep.p("同じ局数をSuphxとトップ人間が打ったときの、安定段位の推定のばらつき（各4,000回のシミュレーション）。")
    rep.table(["局数", "Suphxの安定段位の95%範囲", "Suphxの推定値が人間を上回る確率"], rows)
    rep.p(
        f"Suphxでも1局で1位になれるのは {pct(SUPHX[0], 1, sign=False)}、1位を逃す確率は {pct(1 - SUPHX[0], 1, sign=False)}。模型では、Suphxの優位は平均の高さ（{mu_s:+.3f}）と、得点のばらつきの小ささ（標準偏差 {sg_s:.3f}）の両方で表される。ばらつきが小さいほど4位を避けやすく、4位を重く罰する天鳳の段位ルールで有利になる。"
    )
    rep.metric("mahjong", {"suphx_fit": [mu_s, sg_s], "human_fit": [mu_h, sg_h], "by_games": mj})
    rep.save()


if __name__ == "__main__":
    main()
