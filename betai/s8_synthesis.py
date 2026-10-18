"""Study 8 - synthesis. Reads the JSON metrics of studies 1-7 and builds one
cross-game table: what the AI achieved against each opponent, and why."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt

from .common import OUT, Report, ci_str, pct, savefig


def load(name: str) -> dict:
    path = OUT / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> None:
    s1, s2, s3, s4, s5, s6, s7 = (
        load(f"s{i}_{n}")
        for i, n in [
            (1, "chance"),
            (2, "skill"),
            (3, "sports"),
            (4, "house_limits"),
            (5, "parimutuel"),
            (6, "trading"),
            (7, "llm"),
        ]
    )
    rep = Report("s8_synthesis", "研究8　まとめ：相手別に見た、AIの勝ち負けと勝因・敗因")
    rows, bars = [], []

    if s1:
        r = s1["roulette"]
        c = r["certified RNG|k=2|forced"]
        rows.append(
            [
                "数学",
                "ルーレット（認証された乱数）",
                "模擬",
                "出目を学習するAI（2次マルコフ）",
                f"{c['bets']:,}回",
                pct(c["hit_rate"], 1, sign=False),
                f"{pct(c['roi'])} {ci_str(*c['roi_ci'])}",
                "負け（理論値どおり）",
                "—",
                "過去と未来の出目の情報量がゼロ。控除率がそのまま期待値になる",
            ]
        )
        bars.append(("Roulette, certified RNG", c["roi"], c["roi_ci"]))
        b = r["biased wheel (3 pockets +10%)|k=0|forced"]
        w = r["weak PRNG (12-bit LCG)|k=2|forced"]
        rows.append(
            [
                "数学",
                "ルーレット（欠陥のある乱数源）",
                "模擬",
                "同じ学習AI",
                f"{b['bets']:,}回",
                pct(b["hit_rate"], 1, sign=False),
                f"偏ったホイール {pct(b['roi'])}／弱い疑似乱数 {pct(w['roi'], 0)}",
                "勝ち",
                "物理・実装の欠陥と、それに合ったモデル構造",
                "欠陥がなければ消える。モデル構造が合わないと欠陥があっても負ける",
            ]
        )
        bars.append(("Roulette, biased wheel", b["roi"], b["roi_ci"]))
    if s2:
        v = s2["vs_opponents"]
        e1 = v["コーリングステーション（何でもコール）|搾取AI（相手の癖を完全に知る）"]
        e2 = v["ロック（強い手以外は降りる）|均衡AI（GTO）"]
        rows.append(
            [
                "人",
                "ポーカー（クーン・ポーカー）",
                "厳密計算＋模擬",
                "CFRの均衡AI／相手の癖を突く搾取AI",
                "—",
                f"{pct(e1['win_hand_rate'], 1, sign=False)}（勝った手）",
                f"搾取AI {e1['ev']:+.3f}／均衡AI 対 ロック {e2['ev']:+.4f} チップ/手",
                "勝ち（相手に癖があれば）",
                "相手の癖（Jでのコール、Qでの降りすぎ）。利益の大半は少数の情報集合から",
                "均衡戦略は損をしないだけ。場代（レーキ）が小さな優位を消す",
            ]
        )
        mj = s2["mahjong"]["by_games"]
        n90 = min(int(n) for n, v in mj.items() if v["p_ai_ahead"] >= 0.9)
        rows.append(
            [
                "人",
                "麻雀（Suphxの順位率に当てはめた模型）",
                "模擬",
                "Suphx相当のAI",
                "—",
                "1位 29.3%",
                "安定段位 8.78（人間 7.44）",
                "長期では勝ち",
                "4位を避ける（得点のばらつきが小さい）",
                f"1局では7割が1位を逃す。人間を上回って見える確率が9割を超えるのは約{n90:,}局から",
            ]
        )
    if s3:
        vb = s3["value_bets"]
        g = vb["GBM|avg_open|0.05"]
        bl = vb["Blend (GBM+market)|max_open|0.0"]
        kz = s3["kaunitz"]["open|0.03"]
        rows.append(
            [
                "市場・胴元",
                "欧州サッカー（実データ、約3.5万試合で評価）",
                "実データ",
                "勾配ブースティング（Elo・得点・シュート）",
                f"{g['bets']:,}回",
                pct(g["hit_rate"], 1, sign=False),
                f"{pct(g['roi'])} {ci_str(*g['roi_ci'])}",
                "負け",
                "—",
                f"モデルは市場より不正確（log-loss +{s3['models']['GBM']['vs_close'][0]:.3f}）。マージン約{-g['margin'] * 100:.0f}%。見込んだ期待値 {pct(g['model_ev'])} は幻",
            ]
        )
        bars.append(("Football GBM @avg", g["roi"], g["roi_ci"]))
        rows.append(
            [
                "市場・胴元",
                "同上：モデル＋市場（Benter型）を最高値で",
                "実データ",
                "ブレンド",
                f"{bl['bets']:,}回",
                pct(bl["hit_rate"], 1, sign=False),
                f"{pct(bl['roi'])} {ci_str(*bl['roi_ci'])}",
                "ほぼ損益ゼロ",
                "全社の最高値を拾う価格比較",
                "モデルの重みはほぼゼロ。市場以上の情報がない",
            ]
        )
        bars.append(("Football blend @max", bl["roi"], bl["roi_ci"]))
        rows.append(
            [
                "胴元",
                "同上：予測なしの価格比較（Kaunitz型）",
                "実データ",
                "ルールのみ（AI不要）",
                f"{kz['bets']:,}回",
                pct(kz["hit_rate"], 1, sign=False),
                f"{pct(kz['roi'])} {ci_str(*kz['roi_ci'])}",
                "わずかに勝ち（有意性は境界）",
                "各社の価格のばらつき",
                "記録上の最高値で約定できる前提。勝つ客は制限される（研究4）",
            ]
        )
        bars.append(("Football Kaunitz rule", kz["roi"], kz["roi_ci"]))
    if s4:
        bt = s4["by_type"]["価格を比べる客（研究3のKaunitz型）"]
        rows.append(
            [
                "胴元",
                "勝つ客の制限（実データの期待値を使った模擬）",
                "模擬",
                "価格を比べる客",
                f"制限まで中央値{bt['median_bets_before_restriction']:.0f}回",
                "—",
                f"期待値 {pct(bt['ev'])}",
                "制限される（100%）",
                "—",
                f"優位の証明には約{bt['bets_needed_to_prove']:,.0f}回必要。胴元は終値との比較で先に見抜く",
            ]
        )
    if s5:
        sel = s5["selective"]
        a = sel["AI（強）ノイズ0.25|0.1"]
        m = sel["AI（中）ノイズ0.50|0.1"]
        b = sel["ブレンド（AI中×市場、Benter型）|0.1"]
        ident = s5["identity"]["AI（強）ノイズ0.25"]
        rows.append(
            [
                "市場",
                "競馬（パリミュチュエル、控除率20%）",
                "模擬",
                "強いAI／中程度のAI／中程度のAI＋市場",
                f"{a['bets']:,}回",
                pct(a["hit_rate"], 1, sign=False),
                f"強 {pct(a['roi'])}／中 {pct(m['roi'])}／中＋市場 {pct(b['roi'])}",
                "条件つきで勝ち",
                "控除率（0.223ナッツ）を超える情報優位を持つ馬だけを選ぶこと。市場との組み合わせ",
                f"全頭比例賭けでは負け（W={ident['realized_W']:+.3f}）。市場が正確なら強いAIでも負け。容量は総票数の約{s5['capacity']['share_of_pool'] * 100:.2f}%",
            ]
        )
        bars.append(("Horse racing, strong AI", a["roi"], a["roi_ci"]))
        bars.append(("Horse racing, medium AI", m["roi"], m["roi_ci"]))
    if s6:
        sg = s6["signals"]
        r_ = sg["ridge|週次|k=10"]
        rows.append(
            [
                "市場（リスクプレミアムあり）",
                "米国株：49業種のロング・ショート（実データ、1980〜2026年）",
                "実データ",
                "リッジ回帰／勾配ブースティング（週次）",
                "約2,400回のリバランス",
                "—",
                f"コスト前シャープ {r_['gross_sr']:.2f} → 片道10bp後 {r_['net_sr']['10']:.2f}",
                "コスト後は負け",
                "1980〜90年代には実在した短期の予測可能性",
                "回転率が高くコストに負ける。2000年以降は優位そのものが消えた",
            ]
        )
        rows.append(
            [
                "市場（リスクプレミアムあり）",
                "米国株：買い持ち",
                "実データ",
                "AIなし",
                "—",
                f"上昇日 {pct(s6['market']['up_day_rate'], 1, sign=False)}",
                f"シャープ {s6['market']['sharpe']:.2f}",
                "長期では勝ち",
                "リスクを負う対価（リスクプレミアム）",
                f"1年単位では{pct(1 - s6['horizons']['1年']['p_beat_cash'], 0, sign=False)}の期間で現金に負け、最大ドローダウンは{pct(s6['leverage'][0]['max_dd'], 0)}",
            ]
        )
    if s7:
        fc = s7["forecast"]["all"]
        vb = s7["value_bets"].get("all", {})
        rows.append(
            [
                "市場・胴元",
                "本物のLLM（claude-haiku-5-5）とプレミアリーグ460試合",
                "実データ＋実モデル",
                "LLMの確率で賭ける",
                f"{vb.get('bets', 0)}回",
                "—",
                f"{pct(vb.get('roi', float('nan')))} {ci_str(*vb.get('roi_ci', [float('nan')] * 2))}",
                "負け（統計的には未確定）",
                "—",
                f"予測は市場より不正確（log-loss +{fc['blind_minus_close'][0]:.3f}）。同じ質問への答えが{pct(1 - s7['reproducibility']['identical_share'], 0, sign=False)}の試合で変わる",
            ]
        )
        if vb:
            bars.append(("LLM (Claude Haiku 5.5)", vb["roi"], vb["roi_ci"]))

    rep.h2("相手別の結果")
    rep.table(["相手", "ゲーム", "データ", "AI", "標本", "勝率（的中率）", "成績", "判定", "勝因", "敗因・限界"], rows)

    if bars:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        names = [b[0] for b in bars][::-1]
        vals = [b[1] * 100 for b in bars][::-1]
        lo = [(b[1] - b[2][0]) * 100 for b in bars][::-1]
        hi = [(b[2][1] - b[1]) * 100 for b in bars][::-1]
        ax.barh(names, vals, xerr=[lo, hi], color=["tab:green" if v > 0 else "tab:red" for v in vals], alpha=0.8)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlim(-45, 45)
        ax.set_xlabel("ROI per unit staked (%), 95% CI")
        ax.set_title("What the AI earned against each opponent")
        rep.figure(savefig(fig, "s8_summary"), "相手別のROIと95%信頼区間")
        plt.close(fig)
    rep.save()


if __name__ == "__main__":
    main()
