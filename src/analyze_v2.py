# -*- coding: utf-8 -*-
"""修正分析：域 × 槽位受控的规模曲线（配合 rescore_v2 的判决口径）。

为什么必须做这个分析（首版分析的结构性缺陷）
==================================================================
原先只按"数据集域"（inaturalist vs landmarks）拆分，得出
"干净域无规模效应、脏域有"。但两个域的**槽位构成几乎不重叠**：

  landmarks/place  447 条，全部是 "In which **country** is X located?"
  landmarks/other  317 条，其中 **391** 条是 "In which **city** is X located?"
  inaturalist      是 place 69 / quantity 34 / diet 30 / other 26 的三分

也就是说"域"与"问题类型"完全混淆，原断言**无法区分**是域效应还是槽位效应。
本脚本把两者**交叉**拆开，并限定槽位做域比较，同时报告：

  1. 判决分布（PARAM / CTX / BOTH / NONE）—— 弃权率必须随结果一起报告
  2. 配对 McNemar 精确检验（同一样本集，功效最高）
  3. Wilson 区间（小样本比例）
  4. **最小可检出效应**（n 与不一致率决定）—— 防止把"未检出"说成"不存在"

用法：
  python analyze_v2.py --v2 ../results/v2 --anchors ../data/raw/evqa_anchors_combined.jsonl
"""
import argparse, collections, io, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def mcnemar_exact(b, c):
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(m, i) for i in range(k + 1)) / (2.0 ** m))


def stuart_maxwell(tab):
    """Stuart-Maxwell 检验：McNemar 在 k 类别上的推广。

    为什么需要它：v2 的判决有四个取值（PARAM/CTX/BOTH/NONE），其中
    BOTH/NONE 是**弃权**而非"翻到另一侧"。若按二元处理，弃权会被
    错误地计入不一致对（且弃权率随模型变化时偏倚不对称）。
    complete-case McNemar 会丢掉这些样本；Stuart-Maxwell 用**全部**样本，
    把弃权作为第三类合法结果一起检验。

    tab: k×k 列联表，tab[i][j] = (3B 判 i, 7B 判 j) 的条数。
    返回 (SM 统计量, df, p)。
    """
    k = len(tab)
    d = []
    for i in range(k):
        d.append(sum(tab[i][j] for j in range(k))
                 - sum(tab[j][i] for j in range(k)))
    # 协方差矩阵 V：V_ii = n_i. + n_.i - 2 n_ii ; V_ij = -(n_ij + n_ji)
    V = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            if i == j:
                V[i][i] = (sum(tab[i]) + sum(tab[r][i] for r in range(k))
                           - 2 * tab[i][i])
            else:
                V[i][j] = -(tab[i][j] + tab[j][i])
    # 只需前 k-1 行/列（d 各分量和为 0，V 奇异）
    m = k - 1
    A = [[V[i][j] for j in range(m)] for i in range(m)]
    b = d[:m]
    sm = _quadform(A, b)
    if sm is None:
        return None, m, None
    return sm, m, _chi2_sf(sm, m)


def _quadform(A, b):
    """b' A^{-1} b，用高斯-约当求逆；奇异返回 None。"""
    n = len(A)
    M = [row[:] + [1.0 if i == j else 0.0 for j in range(n)]
         for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-12:
            return None
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        M[c] = [x / pv for x in M[c]]
        for r in range(n):
            if r != c and abs(M[r][c]) > 1e-15:
                f = M[r][c]
                M[r] = [a - f * bb for a, bb in zip(M[r], M[c])]
    inv = [row[n:] for row in M]
    return sum(b[i] * inv[i][j] * b[j] for i in range(n) for j in range(n))


def _chi2_sf(x, df):
    """卡方尾概率（df=1/2/3 用闭式，避免引入 scipy）。"""
    if x <= 0:
        return 1.0
    if df == 1:
        return math.erfc(math.sqrt(x / 2.0))
    if df == 2:
        return math.exp(-x / 2.0)
    if df == 3:
        # P(X>x) = erfc(sqrt(x/2)) + sqrt(2x/pi) e^{-x/2}
        return (math.erfc(math.sqrt(x / 2.0))
                + math.sqrt(2 * x / math.pi) * math.exp(-x / 2.0))
    raise ValueError("df=%s 未实现" % df)


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mde(b, c, n):
    """最小可检出效应：达到 p<0.05 所需的不一致对之差（占 n 的比例）。

    回答"这个 n 到底能排除多大的效应"，是防止 absence-of-evidence 谬误的关键。
    """
    tot = b + c
    # 在给定不一致对总数下，找最小可显著的差
    for extra in range(0, tot + 1):
        m = tot + extra
        for k in range(0, m // 2 + 1):
            if mcnemar_exact(m - k, k) < 0.05:
                return (m - 2 * k) / n, m - 2 * k
    # 不一致对太少，需要更多总对
    for m in range(tot, 400):
        for k in range(0, m // 2 + 1):
            if mcnemar_exact(m - k, k) < 0.05:
                return (m - 2 * k) / n, m - 2 * k
    return None, None


def load(p):
    if not os.path.exists(p):
        return None
    return json.load(io.open(p, encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default="../results/v2")
    ap.add_argument("--anchors", default="../data/raw/evqa_anchors_combined.jsonl")
    ap.add_argument("--out", default="../results/v2/_analysis_v2.json")
    args = ap.parse_args()

    M = {}
    for l in io.open(args.anchors, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        M[(r.get("question_named") or "", r.get("wikipedia_title") or "")] \
            = (r.get("dataset_name"), r.get("slot"))

    def key(r):
        return (r.get("question_named") or "", r.get("wikipedia_title") or "")

    def grab(pref, arm):
        # rescore_v2 写出的文件名带 _v2 后缀
        d = load(os.path.join(args.v2, "%s_v2.json" % pref))
        if not d:
            return None
        return {key(r): r for r in d["records"]
                if r.get("usable") and (r.get("context_conflict") or "").strip()}

    out = {"per_arm": {}, "by_cell": {}, "domain_within_slot": {}}

    for arm in ("text", "vision"):
        L = grab("pkd_3b_comb_%s" % arm, arm)
        H = grab("pkd_7b_comb_%s" % arm, arm)
        if not L or not H:
            continue
        sh = [k for k in L if k in H and L[k].get("v2_verdict")]

        # 判决为 PARAM 且两侧不同时接地 —— v2 已经把 BOTH 单独归为弃权
        def is_param(r):
            return r.get("v2_verdict") == "PARAM"

        def decidable(r):
            return r.get("v2_verdict") in ("PARAM", "CTX")

        ORD = ["PARAM", "CTX", "BOTH", "NONE"]

        def pair_stats(ks):
            """两个口径 + 弃权列联表。

            `strict`：**完整案例** McNemar —— 只在两规模都可判定的样本上配对，
                      弃权样本整体排除。无偏，但丢样本，且若弃权与判决相关则有选择偏倚。
            `sm`    ：**Stuart-Maxwell**（4×4）—— 用全部样本，把弃权当合法第三类结果。
                      这是 v2 多类别判决下的正确检验，与严格口径互为稳健性检查。
            """
            ol = sum(1 for k in ks if is_param(L[k]) and is_param(H[k]) is False
                     and decidable(H[k]))
            oh = sum(1 for k in ks if is_param(H[k]) and is_param(L[k]) is False
                     and decidable(L[k]))
            tab = [[0] * 4 for _ in range(4)]
            for k in ks:
                a = L[k].get("v2_verdict")
                b = H[k].get("v2_verdict")
                if a in ORD and b in ORD:
                    tab[ORD.index(a)][ORD.index(b)] += 1
            sm, df, psm = stuart_maxwell(tab)
            return ol, oh, mcnemar_exact(ol, oh), tab, sm, df, psm

        print("=" * 92)
        print(" 臂 = %s   配对样本 = %d" % (arm, len(sh)))
        print("=" * 92)

        # --- 单元格：域 × 槽位 ---
        cells = collections.defaultdict(list)
        for k in sh:
            dn, sl = M.get(k, ("?", "?"))
            cells[(dn, sl)].append(k)

        print(" %-13s %-9s %5s %8s %8s %11s %9s %9s %9s"
              % ("域", "槽位", "n", "PKD(3B)", "PKD(7B)", "完整案例McNemar",
                 "p(McN)", "p(Stuart)", "最小可检出差"))
        print("-" * 104)
        rows = []
        for (dn, sl), ks in sorted(cells.items(), key=lambda x: -len(x[1])):
            if len(ks) < 15:
                continue
            a = [k for k in ks if decidable(L[k])]
            b = [k for k in ks if decidable(H[k])]
            pa = sum(1 for k in a if is_param(L[k])) / max(len(a), 1)
            pb = sum(1 for k in b if is_param(H[k])) / max(len(b), 1)
            ol, oh, p, tab, sm, df, psm = pair_stats(ks)
            d, dn_abs = mde(ol, oh, len(ks))
            ci = wilson(sum(1 for k in a if is_param(L[k])), len(a))
            print(" %-13s %-9s %5d %8.4f %8.4f %5d:%-5d %9.5f %9s %9s"
                  % (dn, sl, len(ks), pa, pb, ol, oh, p,
                     ("%.4f" % psm) if psm is not None else "退化*",
                     ("%.1f%%" % (100 * d)) if d is not None else "-"))
            rows.append(dict(domain=dn, slot=sl, n=len(ks),
                             pkd_low=pa, pkd_high=pb, only_low=ol, only_high=oh,
                             p=p, p_stuart=psm, stuart_sm=sm, stuart_df=df,
                             transition_table=tab, ord=ORD,
                             mde_pp=(100 * d) if d is not None else None,
                             pkd_low_ci=ci))
        out["by_cell"][arm] = rows

        # --- 域效应，限定槽位（控制混淆）---
        print("\n 域效应（**限定在同一槽位内**比较，控制槽位混淆）")
        print("-" * 92)
        ds = []
        for sl in sorted({M[k][1] for k in sh if k in M}):
            ins = [k for k in sh if M[k][1] == sl and M[k][0] == "inaturalist"]
            lms = [k for k in sh if M[k][1] == sl and M[k][0] == "landmarks"]
            if len(ins) < 10 or len(lms) < 10:
                continue
            def pk(ks, S):
                a = [k for k in ks if decidable(S[k])]
                return sum(1 for k in a if is_param(S[k])) / max(len(a), 1), len(a)
            il, nl = pk(ins, L); ih, nh = pk(ins, H)
            ll, ml = pk(lms, L); lh, mh = pk(lms, H)
            ol, oh, p, tab, sm, df, psm = pair_stats(lms)
            print(" 槽位=%-9s  inat n=%-4d: %.4f→%.4f (差%+.3f) | "
                  "landmarks n=%-4d: %.4f→%.4f (差%+.3f)  %d:%d p=%.5f p_SM=%s"
                  % (sl, nl, il, ih, ih - il, ml, ll, lh, lh - ll, ol, oh, p,
                     ("%.4f" % psm) if psm is not None else "退化*"))
            ds.append(dict(slot=sl, inat_n=nl, inat_low=il, inat_high=ih,
                           lm_n=ml, lm_low=ll, lm_high=lh,
                           only_low=ol, only_high=oh, p=p, p_stuart=psm))
        out["domain_within_slot"][arm] = ds

        # --- 全臂汇总 ---
        all_n = len(sh)
        ol, oh, p, tab, sm, df, psm = pair_stats(sh)
        d, _ = mde(ol, oh, all_n)
        print("\n 全臂合计：%d 条，完整案例不一致 %d:%d，p=%.6f；"
              "Stuart-Maxwell sm=%.3f df=%d p=%.4f，最小可检出差≈%s"
              % (all_n, ol, oh, p, sm, df, psm,
                 ("%.1f%%" % (100 * d)) if d else "-"))
        out["per_arm"][arm] = dict(n=all_n, only_low=ol, only_high=oh, p=p,
                                   p_stuart=psm, stuart_sm=sm, stuart_df=df,
                                   mde_pp=(100 * d) if d else None)
        print()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, io.open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("已写入:", args.out)


if __name__ == "__main__":
    main()
