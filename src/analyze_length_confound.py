# -*- coding: utf-8 -*-
"""长度混淆的排除检验（§4.2.4）。

要反驳的假设
------------
H_len：规模效应的符号由「大模型倾向给更长答案」驱动，而非由立场决定。
若 H_len 为真，则 Δ 的符号应当跟随「大小模型答案长度之差」的符号：
    长度变大 → Δ > 0 ；长度变小 → Δ < 0。

检验
----
逐立场算 (长度_大 − 长度_小) 与 Δ 的符号是否一致。任何一格「长度变大而 Δ<0」
或「长度变小而 Δ>0」都是 H_len 的反例 —— 长度无法解释符号。

另做 Kruskal-Wallis 检验，确认各立场的长度分布确实彼此不同（即立场真的
改变了长度，长度不是恒量），否则 H_len 无从谈起。

实现说明
--------
自带 Kruskal-Wallis，不依赖 scipy（本机 venv 无 scipy）。
H = 12/(N(N+1)) * Σ R_i²/n_i − 3(N+1)，含并列（tie）校正。
p 值用卡方分布近似，自由度 k−1，卡方尾概率由不完全 gamma 函数算。
"""
import argparse, io, json, math, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ORDER = ["ctx_only", "orig", "ctx_hedge", "neutral", "own_only", "len_ctrl", "prior"]

# 每个规模的立场文件给出**候选列表**（按优先级），远端 A800 重跑出来的优先。
# 此前这里写死本地文件名，于是 7B 于 A800 重跑之后本节仍读旧的本地产物 ——
# 结果是 §4.2.4 的长度差与 §4.2.3 的 Δ 出自不同批次的数据，
# 而它们在正文里被并排引用。候选顺序与 analyze_scale_ladder.LADDER 一致。
FILES = {
    "qwen3-vl-2b":   ("Qwen3-VL 2B",   "qwen3",  2, ["syc6_qwen3-vl-2b_text.json"]),
    "qwen3-vl-4b":   ("Qwen3-VL 4B",   "qwen3",  4, ["syc6_qwen3-vl-4b_text.json"]),
    "qwen3-vl-8b":   ("Qwen3-VL 8B",   "qwen3",  8, ["syc6_qwen3-vl-8b_text.json"]),
    "qwen2.5-vl-3b": ("Qwen2.5-VL 3B", "qwen25", 3,
                      ["syc6_qwen2.5-vl-3b_text.json", "syc6_3b_text.json"]),
    "qwen2.5-vl-7b": ("Qwen2.5-VL 7B", "qwen25", 7,
                      ["syc6_qwen2.5-vl-7b_text.json", "syc6_7b_text.json"]),
}
KFILES = {
    "qwen3-vl-2b":   "pkd_qwen3-vl-2b_text.json",
    "qwen3-vl-4b":   "pkd_qwen3-vl-4b_text.json",
    "qwen3-vl-8b":   "pkd_qwen3-vl-8b_text.json",
    "qwen2.5-vl-3b": "../results/pkd_3b_comb_text.json",
    "qwen2.5-vl-7b": "../results/pkd_7b_comb_text.json",
}


def key(r):
    return (r.get("question_named") or "", r.get("wikipedia_title") or "")


def load(p):
    if not os.path.exists(p):
        return None
    return json.load(io.open(p, encoding="utf-8")).get("records", [])


def load_k(tag, probe):
    fn = KFILES[tag]
    p = os.path.join(HERE, fn) if fn.startswith("..") else os.path.join(probe, fn)
    recs = load(p)
    if recs is None:
        return None
    return {key(r): bool(r.get("noc_follows_param")) for r in recs}


def _norm_sf(x):
    """标准正态上尾概率。"""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def _gammainc_upper_reg(a, x):
    """Q(a,x) = Γ(a,x)/Γ(a)，连分数法（Numerical Recipes gammq）。"""
    if x < 0 or a <= 0:
        raise ValueError
    if x == 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1.0:                      # 用 P(a,x) 的级数
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(1000):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-14:
                break
        return 1.0 - s * math.exp(-x + a * math.log(x) - gln)
    b, c = x + 1.0 - a, 1e300           # 连分数
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def chi2_sf(h, df):
    return _gammainc_upper_reg(df / 2.0, h / 2.0)


def kruskal(groups):
    """返回 (H, p, df)。groups = list[list[float]]。"""
    groups = [g for g in groups if len(g) > 1]
    k = len(groups)
    if k < 2:
        return 0.0, 1.0, 0
    allv = [(v, gi) for gi, g in enumerate(groups) for v in g]
    n = len(allv)
    allv.sort(key=lambda t: t[0])
    # 平均秩（含并列）
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and allv[j + 1][0] == allv[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[t] = avg
        i = j + 1
    rsum = [0.0] * k
    cnt = [0] * k
    for pos, (_, gi) in enumerate(allv):
        rsum[gi] += ranks[pos]
        cnt[gi] += 1
    h = 12.0 / (n * (n + 1)) * sum(rsum[i] ** 2 / cnt[i] for i in range(k)) - 3 * (n + 1)
    # 并列校正
    tie = 0.0
    i = 0
    vals = [v for v, _ in allv]
    while i < n:
        j = i
        while j + 1 < n and vals[j + 1] == vals[i]:
            j += 1
        t = j - i + 1
        if t > 1:
            tie += t ** 3 - t
        i = j + 1
    if tie:
        denom = 1.0 - tie / float(n ** 3 - n)
        # 全部观测同值（所有组所有样本都并列）时 denom = 0，
        # 此时 H 形如 0/0，按惯例取 H = 0（无可检出的组间差异）。
        # 长度数据是整数、并列极多，这个分支会被真实触发，不能只靠除零报错。
        h = h / denom if denom > 1e-12 else 0.0
    df = k - 1
    return h, chi2_sf(h, df), df


def _paired_dlen(loaded, shared, a, b, cd, lengths):
    """配对长度差（字符）：只在两侧都被判为可判定的样本上算。

    两侧的可判集由 analyze_scale_ladder.grade_eff 决定（含否定感知开关），
    从那里取判据实现，避免本节与 §4.2 主表判据分叉。
    """
    import analyze_scale_ladder as A
    tot = n = 0.0
    for k in shared:
        ra, rb = loaded[a].get(k), loaded[b].get(k)
        if ra is None or rb is None:
            continue
        ga = A.grade_eff(ra.get("out_" + cd), ra)
        gb = A.grade_eff(rb.get("out_" + cd), rb)
        if ga in (None, "BOTH", "NONE") or gb in (None, "BOTH", "NONE"):
            continue
        tot += len(rb.get("out_" + cd) or "") - len(ra.get("out_" + cd) or "")
        n += 1
    return (tot / n) if n else None


def _paired_delta(loaded, shared, a, b, cd):
    """配对 Δ（pp）与配对可判数。与 make_tables.delta_and_p 同一口径：
    两侧都必须在同一批样本上可判，PKD 分子也只在同一批里数。"""
    import analyze_scale_ladder as A
    n = pa = pb = 0
    for k in shared:
        ra, rb = loaded[a].get(k), loaded[b].get(k)
        if ra is None or rb is None:
            continue
        ga = A.grade_eff(ra.get("out_" + cd), ra)
        gb = A.grade_eff(rb.get("out_" + cd), rb)
        if ga in (None, "BOTH", "NONE") or gb in (None, "BOTH", "NONE"):
            continue
        n += 1
        pa += (ga == "PARAM")
        pb += (gb == "PARAM")
    if n == 0:
        return None, None, None
    return 100.0 * (pb - pa) / n, n, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default=os.path.join(HERE, "..", "results", "probe"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "length_confound.json"))
    ap.add_argument("--ladder", default="scale_ladder.json",
                    help="读哪个 PKD 结果文件取 Δ（默认 alias 判据；"
                         "配 --grader negation 时应传 scale_ladder_negation.json）")
    ap.add_argument("--grader", choices=["alias", "negation"], default="alias",
                    help="配对长度差用哪个判据决定『可判』。必须与 --ladder 对应，"
                         "否则本节会把 A 判据的样本集配上 B 判据的 Δ。")
    args = ap.parse_args()

    # 配对长度差的「可判」口径必须与 Δ 的来源一致。
    # analyze_scale_ladder 的 grade_eff 受该模块的全局开关控制，
    # 故这里显式设一次，让 _paired_dlen 与本进程读的 ladder 文件同源。
    import analyze_scale_ladder as _A
    if args.grader == "negation":
        from regrade_negation import is_explicit_rejection as _fn
        _A.USE_NEGATION, _A._NEG_FN = True, _fn
    else:
        _A.USE_NEGATION = False

    loaded, ks = {}, {}
    for tag, (disp, fam, nB, cands) in FILES.items():
        fn = next((c for c in cands if os.path.exists(os.path.join(args.probe, c))), None)
        if fn is None:
            print("⚠️  %s 的立场文件全缺（候选 %s）" % (tag, " | ".join(cands)))
            continue
        recs = load(os.path.join(args.probe, fn))
        if recs is None:
            continue
        loaded[tag] = {key(r): r for r in recs}
        k = load_k(tag, args.probe)
        if k:
            ks[tag] = k

    if len(loaded) < 2:
        sys.exit("可用规模点不足 2 个。")
    if len(ks) < len(loaded):
        print("⚠️  部分规模点缺锚定文件，D* 只能在其子集上算。")

    # D* 必须同时满足两个条件：
    #   ① 在所有规模点的立场文件里都**被评测过**（否则无从取答案）
    #   ② 在所有规模点上 K_s = 1（否则小模型无知识可跟随，压低其 PKD）
    # 只用 ② 会得到 1146 上的交集（893），但立场文件只有 927 条，会 KeyError；
    # 只用 ① 就等于没做锚定（927 条键相互全等）。
    covered = None
    for t in loaded:
        covered = set(loaded[t]) if covered is None else (covered & set(loaded[t]))
    covered = covered or set()
    kk = [t for t in ks if t in loaded]
    shared = sorted(i for i in covered if all(ks[t].get(i, False) for t in kk))
    print("共同被评测 n = %d；参与锚定的规模点：%s" % (len(covered), ", ".join(sorted(kk))))
    print("D* n = %d（占共同被评测 %.1f%%）"
          % (len(shared), 100.0 * len(shared) / max(len(covered), 1)))
    if not shared:
        sys.exit("D* 为空。")

    # 逐立场逐规模的答案长度
    lengths = {}
    for tag in loaded:
        lengths[tag] = {}
        for cd in ORDER:
            lengths[tag][cd] = [len((loaded[tag][i].get("out_" + cd) or "").strip())
                                for i in shared if i in loaded[tag]]
    print("\n逐立场答案长度（D* 内，字符）：均值 / 中位 / P90")
    tags = [t for t in FILES if t in loaded]
    hdr = "%-10s" % "立场" + "".join("%24s" % FILES[t][0] for t in tags)
    print(hdr)
    print("-" * len(hdr))
    for cd in ORDER:
        line = "%-10s" % cd
        for t in tags:
            L = lengths[t][cd]
            p90 = sorted(L)[min(int(0.9 * len(L)), len(L) - 1)]
            line += "%24s" % ("%.0f / %.0f / %.0f" % (st.mean(L), st.median(L), p90))
        print(line)

    # 长度是否随立场变化（H_len 的前提）
    print("\n各规模内，立场是否显著改变长度（Kruskal-Wallis）：")
    for t in tags:
        h, p, df = kruskal([lengths[t][cd] for cd in ORDER])
        # chi2 尾概率会下溢到 0.0；直接印 "p=0" 会被误读成「精确等于 0」
        ps = "p < 1e-300" if p == 0.0 else "p = %.3g" % p
        print("  %-14s H=%.1f  df=%d  %-14s %s"
              % (FILES[t][0], h, df, ps,
                 "长度随立场显著变化" if p < 0.05 else "长度不随立场变化"))

    # ★ 核心：长度差的符号 vs Δ 的符号
    print("\n" + "=" * 92)
    print("★ 核心检验：Δ 的符号是否由「长度差的符号」决定（H_len 的判决）")
    print("=" * 92)
    result = {"n_anchored": len(shared), "rows": []}
    verdict = []
    for fam in ("qwen3", "qwen25"):
        ft = sorted([t for t in tags if FILES[t][1] == fam], key=lambda x: FILES[x][2])
        if len(ft) < 2:
            continue
        a, b = ft[0], ft[-1]
        print("\n族 = %s（%s → %s）" % (fam, FILES[a][0], FILES[b][0]))
        print("  %-10s %12s %12s %10s %10s %10s   %s"
              % ("立场", "长度_小", "长度_大", "长度差", "Δ(pp)", "配对n", "一致性"))
        for cd in ORDER:
            # 长度差必须与 Δ 在同一批样本上算：取两侧都可判的那些样本。
            # 若各用各的可判子集，长度差与 Δ 就不是同一样本集上的量，
            # 判据的剔除偏向会被记进长度差里（实测可把符号翻掉），
            # 于是本节会用一个人为造出的反号去否证 H_len。
            # 这与 §4.2 主表的口径一致，两处必须由同一规则保证。
            La, Lb = st.mean(lengths[a][cd]), st.mean(lengths[b][cd])
            dlen = _paired_dlen(loaded, shared, a, b, cd, lengths)
            if dlen is None:
                print("  %-10s 该立场两侧无共同可判样本，跳过" % cd)
                continue
            # Δ 从共享的 analyze 结果读，避免判据实现分叉。
            # 文件名必须跟着 --grader 走：两个判据的 Δ 差别很大，
            # 读错文件会让本节的判决与 §4.2.3 的表对不上。
            #
            # 端点 Δ 必须取 pairs 里的**配对**记录，不能用 per_stance 的
            # 两个 pkd 相减：后者各用各的可判子集，差值不是配对差。
            # pairs 只含相邻规模对，端点（最小↔最大）不在其中，
            # 故按 family+stance 匹配 pairs 与端点对，无匹配则回退到
            # 逐格重算（下面 delta_and_p 的口径），并在日志里说明。
            d = n_paired = None
            sp = os.path.join(HERE, "..", "results", args.ladder)
            if os.path.exists(sp):
                sj = json.load(io.open(sp, encoding="utf-8"))
                for pr in sj.get("pairs", []):
                    if pr.get("family") == fam and pr.get("stance") == cd \
                       and pr.get("small") == FILES[a][0] and pr.get("large") == FILES[b][0]:
                        d, n_paired = pr.get("delta_pp"), pr.get("n_paired")
                        break
                if d is None:
                    # 端点对不在 pairs 里 —— 直接重算，口径与
                    # analyze_scale_ladder 的配对块一致。
                    d, n_paired, _ = _paired_delta(loaded, shared, a, b, cd)
            if d is None:
                continue
            # H_len 预测：长度差>0 ⇒ Δ>0 ；长度差<0 ⇒ Δ<0
            consistent = (dlen > 0) == (d > 0)
            tag_ok = "一致" if consistent else "**不一致**"
            if not consistent:
                verdict.append((fam, cd, dlen, d))
            print("  %-10s %12.1f %12.1f %+10.1f %+10.1f %9s   %s"
                  % (cd, La, Lb, dlen, d,
                     ("n=%d" % n_paired) if n_paired else "n/a", tag_ok))
            result["rows"].append(dict(family=fam, stance=cd, family_a=FILES[a][0],
                                       family_b=FILES[b][0], len_small=La, len_large=Lb,
                                       dlen=dlen, delta_pp=d, n_paired=n_paired,
                                       consistent=consistent))
    print("\n" + "-" * 92)
    if verdict:
        print("★ H_len 被否证：以下 %d 格中长度差与 Δ **反号**，长度无法解释其符号：" % len(verdict))
        for fam, cd, dlen, d in verdict:
            print("    %-8s %-10s 长度差 %+.0f 字符（变大） 而 Δ = %+.1f pp（负）" % (fam, cd, dlen, d) if dlen > 0 and d < 0
                  else "    %-8s %-10s 长度差 %+.0f 字符（变小） 而 Δ = %+.1f pp（正）" % (fam, cd, dlen, d))
        print("\n  → 这些立场上，大模型给出**更长**的答案，却**更少**跟随参数知识。")
        print("     长度混淆在此直接失效。")
    else:
        print("⚠️  未发现反例 —— H_len 未被否证，需更直接的干预实验（见下）。")
    result["n_inconsistent"] = len(verdict)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with io.open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print("\n已写入: %s" % os.path.abspath(args.out))


if __name__ == "__main__":
    main()
