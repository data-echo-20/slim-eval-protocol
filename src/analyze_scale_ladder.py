# -*- coding: utf-8 -*-
"""跨规模阶梯分析：把规模点从 2 个扩展到 5 个，检验 N1 命题的稳健性。

背景
----
本地既有结果只有 Qwen2.5-VL 3B/7B **两个**规模点，3B→7B 是仅有的相邻对。
对标工作 M4-RAG 用了 4 个规模点（Qwen2.5-VL 3B/7B/32B/72B、Gemma3 4B/12B/27B）。
2026-09-24 租 A800 补跑 Qwen3-VL 2B/4B/8B，得 2→4→8 倍增阶梯，
加上本地 2.5-VL 3B/7B，**总规模点达 5 个**（3B 与 4B 近似同档，下文按族分列）。

核心命题（N1）
--------------
同一族内，指令立场 ι 决定规模效应的**符号**：
    Δ_s(ι) = PKD_{s_max}(ι) − PKD_{s_min}(ι)
Δ 随"对参数知识的授权强度"单调，且跨零。

判据（`--grader`，默认 alias）
--------------------------------
**别名判据**（长度不变），与 regrade_aliases.py 完全一致：
锚定集自带 `parametric_aliases` / `context_answer_aliases`
配 `common.answers_match`（双向包含 + 内容词覆盖率）。
理由见 `04_论文/第八次自查_CVPR2026全文语料核查.md` §7：接地判据
`distinctive_ground` 的**不可判定率**与立场答案均长高度共变 ——
七个立场汇总点上 Pearson r=0.983、Spearman ρ=1.000，会制造假剂量-反应曲线。
（口径注意：这是**立场汇总点**的相关，不是逐样本相关；逐样本上弱得多，
`prior` 上 ρ 仅 0.42。引用时别把两个口径混说。）

**`--grader negation`**：在别名判据之上，把「显式驳回冲突上下文」的 BOTH
救回 PARAM。别名判据会把这类答案判为 BOTH 并在分母里剔除，而
「驳回」必须点名那个错误值，于是越是明确驳回越容易被剔除 ——
剔除量随规模上升，**系统性低估 PKD_大**（即低估本文的正效应）。
动机、模式表与两轮误判自查见 `regrade_negation.py` 与
`04_论文/判据缺陷_显式驳回被当作不可判定.md`。
**两个判据的数都要报**，不得只报修正后的。

锚定协议
--------
D* = ∩_s K_s(i)，K_s(i) 由锚定文件给出（`model_param_answer` 非空且与
`parametric_answer` 匹配）。**只用"所有规模都知道"的样本**，
否则规模间差异会混入"大模型多知道几个事实"这一平凡效应。

用法
----
  python analyze_scale_ladder.py --probe ../results/probe --out ../results/scale_ladder.json
"""
import argparse, collections, io, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import answers_match

# 立场按"对参数知识的授权强度"由弱到强预注册排序（非事后拟合）
ORDER = ["ctx_only", "orig", "ctx_hedge", "neutral", "own_only", "len_ctrl", "prior"]

# 规模阶梯：tag -> (显示名, 家族, 参数量B, 候选文件名列表)
# 候选顺序 = 优先顺序。远端跑出来的用远端命名（同一环境，可比性更强）；
# 找不到才回退到本地既有文件（syc6_3b_text.json 等，跑在不同环境上）。
# 远端命名规则见 remote/run_scale_ladder.sh:219
LADDER = [
    ("qwen3-vl-2b",   "Qwen3-VL 2B",   "qwen3", 2, ["syc6_qwen3-vl-2b_text.json"]),
    ("qwen3-vl-4b",   "Qwen3-VL 4B",   "qwen3", 4, ["syc6_qwen3-vl-4b_text.json"]),
    ("qwen3-vl-8b",   "Qwen3-VL 8B",   "qwen3", 8, ["syc6_qwen3-vl-8b_text.json"]),
    ("qwen2.5-vl-3b", "Qwen2.5-VL 3B", "qwen25", 3,
     ["syc6_qwen2.5-vl-3b_text.json", "syc6_3b_text.json"]),
    ("qwen2.5-vl-7b", "Qwen2.5-VL 7B", "qwen25", 7,
     ["syc6_qwen2.5-vl-7b_text.json", "syc6_7b_text.json"]),
]

# 知识文件（NO-CTX 锚定臂）。**必须**用它把 D* 限制为「所有规模都答得出」的样本。
# 只用立场文件的键交集是错的：那 927 条的键在各文件里都是 927（同一批样本），
# 交出来仍是 927，等于没有做锚定。
# 实测代价：927 条里 Qwen3-VL 2B 有 201 条 K=0（21.7%），
# 这些样本小模型「无知识可跟随」却被留在分母，会系统性压低 PKD_小，
# 从而**放大** Δ —— 即伪造出比真实更强的规模悖论。故锚定不可省。
KFILES = {
    "qwen3-vl-2b":   ["pkd_qwen3-vl-2b_text.json"],
    "qwen3-vl-4b":   ["pkd_qwen3-vl-4b_text.json"],
    "qwen3-vl-8b":   ["pkd_qwen3-vl-8b_text.json"],
    "qwen2.5-vl-3b": ["../results/pkd_3b_comb_text.json"],
    "qwen2.5-vl-7b": ["../results/pkd_7b_comb_text.json"],
}


def mcnemar_exact(b, c):
    """McNemar 精确检验（双尾）。b/c = 两个方向的翻转数。"""
    m = b + c
    if m == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(m, i) for i in range(min(b, c) + 1)) / (2.0 ** m))


def _spearman(x, y):
    """Spearman 秩相关。x 一般是 1..k 的序数，故其秩即自身。"""
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(s):
            r[i] = float(pos + 1)
        return r
    ra, rb = rank(x), rank(y)
    n = len(x)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    den = math.sqrt(sum((ra[i] - ma) ** 2 for i in range(n))
                    * sum((rb[i] - mb) ** 2 for i in range(n)))
    return num / den if den else 0.0


def _perm_test_spearman(x, y, seed=42, n=200000):
    """置换检验：ρ 的零分布由打乱 y 的次序得到。单尾 p(ρ ≥ 观测)。"""
    import random
    obs = _spearman(x, y)
    rnd = random.Random(seed)
    hit = 0
    for _ in range(n):
        p = y[:]
        rnd.shuffle(p)
        if _spearman(x, p) >= obs:
            hit += 1
    return hit / float(n)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def grade(ans, rec):
    """别名判据（长度不变）。返回 PARAM / CTX / BOTH / NONE。"""
    if not (ans or "").strip():
        return "NONE"
    pa = rec.get("parametric_aliases") or []
    ca = rec.get("context_answer_aliases") or []
    if isinstance(pa, str):
        pa = [pa]
    if isinstance(ca, str):
        ca = [ca]
    gp = answers_match(ans, pa)[0] if pa else False
    gc = answers_match(ans, ca)[0] if ca else False
    if gp and gc:
        return "BOTH"
    if gp:
        return "PARAM"
    if gc:
        return "CTX"
    return "NONE"


# 全局开关：是否用否定感知重判。由 --grader negation 打开。
USE_NEGATION = False
_NEG_FN = None


def grade_eff(ans, rec):
    """实际使用的判据。

    USE_NEGATION 打开时，把「显式驳回上下文」的 BOTH 救回 PARAM。
    理由见 regrade_negation.py：这类答案明确站在参数知识一边
    （"The context is incorrect. ... it is actually in Switzerland."），
    却因为**点名了错误值**而被双向包含判为 BOTH 并剔除。
    剔除量随规模上升，故该缺陷**系统性低估 PKD_大**。
    """
    g = grade(ans, rec)
    if USE_NEGATION and g == "BOTH" and _NEG_FN and _NEG_FN(ans, rec):
        return "PARAM"
    return g


def key_of(r):
    return (r.get("question_named") or "", r.get("wikipedia_title") or "")


def load(path):
    if not os.path.exists(path):
        return None
    d = json.load(io.open(path, encoding="utf-8"))
    recs = {key_of(r): r for r in d.get("records", [])}
    return dict(recs=recs, conds=d.get("meta", {}).get("conds") or list(ORDER))


def load_k(tag, probe):
    """读某规模点的 NO-CTX 锚定文件，返回 {key: K_s}。找不到返回 None。"""
    for fn in KFILES.get(tag, []):
        p = fn if os.path.isabs(fn) else os.path.join(
            HERE, fn) if fn.startswith("..") else os.path.join(probe, fn)
        if not os.path.exists(p):
            continue
        d = json.load(io.open(p, encoding="utf-8"))
        return {key_of(r): bool(r.get("noc_follows_param"))
                for r in d.get("records", [])}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default=os.path.join(HERE, "..", "results", "probe"))
    ap.add_argument("--out", default=None, help="默认按判据命名 scale_ladder[_negation].json")
    ap.add_argument("--min-points", type=int, default=3,
                    help="最少规模点数；<3 时不出跨规模结论，仅供脚本自检")
    ap.add_argument("--grader", choices=["alias", "negation"], default="alias",
                    help="alias=纯别名判据；negation=把显式驳回上下文的 BOTH 救回 PARAM")
    args = ap.parse_args()

    global USE_NEGATION, _NEG_FN
    if args.grader == "negation":
        try:
            from regrade_negation import is_explicit_rejection as _fn
        except ImportError as e:
            sys.exit("无法导入 regrade_negation：%s" % e)
        USE_NEGATION, _NEG_FN = True, _fn
        print("判据 = negation（否定感知；显式驳回上下文的 BOTH 计为 PARAM）")
    else:
        print("判据 = alias（纯别名匹配）")

    if args.out is None:
        args.out = os.path.join(HERE, "..", "results",
                                "scale_ladder.json" if args.grader == "alias"
                                else "scale_ladder_negation.json")

    # ---- 载入各规模点 ----
    loaded, conds, used_file = {}, None, {}
    for tag, disp, fam, nB, cands in LADDER:
        d, hit = None, None
        for fn in cands:
            d = load(os.path.join(args.probe, fn))
            if d is not None:
                hit = fn
                break
        if d is None:
            print("  [缺] %-16s 候选 %s" % (disp, " | ".join(cands)))
            continue
        conds = conds or d["conds"]
        loaded[tag] = d["recs"]
        used_file[tag] = hit
        print("  [有] %-16s %4d 条  ← %s" % (disp, len(d["recs"]), hit))

    if len(loaded) < args.min_points:
        sys.exit("\n可用规模点 %d 个，不足 %d 个，无法构成阶梯。" % (len(loaded), args.min_points))
    if len(loaded) < 3:
        print("\n⚠️  只有 %d 个规模点，跨规模结论不成立，仅供脚本自检。" % len(loaded))

    # ---- 锚定集 D* = ∩_s { i : K_s(i)=1 } ----
    # 注意：这里**不能**用「立场文件的键交集」。那 927 条的键在每个文件里
    # 都是同一批 927 条，交出来仍是 927，等于没做锚定。
    # 锚定必须读 NO-CTX 臂的 noc_follows_param。
    covered = None
    for tag in loaded:
        covered = set(loaded[tag]) if covered is None else (covered & set(loaded[tag]))
    covered = covered or set()

    kstate, missing_k = {}, []
    for tag in loaded:
        k = load_k(tag, args.probe)
        if k is None:
            missing_k.append(tag)
            continue
        kstate[tag] = k
    if missing_k:
        print("\n⚠️  缺锚定文件（NO-CTX）的规模点：%s" % ", ".join(missing_k))
        print("   缺锚定的点无法参与 D* —— 强行保留会让它的 PKD 分母掺入 K=0 样本。")
        print("   处理：从阶梯中剔除这些点，只对锚定齐全的点出结论。")
        for t in missing_k:
            loaded.pop(t, None)

    shared = sorted(i for i in covered
                    if all(kstate[t].get(i, False) for t in loaded))
    n = len(shared)
    print("\n共同覆盖样本 n = %d" % len(covered))
    print("锚定集 D*（所有规模 K=1）n = %d（占共同覆盖 %.1f%%）"
          % (n, 100.0 * n / max(len(covered), 1)))
    for t in loaded:
        miss = sum(1 for i in covered if not kstate[t].get(i, False))
        print("   %-16s 被 D* 剔除 %4d 条（%.1f%%）" % (t, miss, 100.0 * miss / max(len(covered), 1)))
    if n == 0:
        sys.exit("D* 为空 —— 各规模点的知识交集为空，无法做跨规模比较。")
    if n < 200:
        print("\n⚠️  D* 仅 %d 条，置信区间会被显著放大；"
              "本节结论须降级为「指示性」而非「确证性」。" % n)

    # ---- 逐立场 × 逐规模 ----
    tags = [t for t, *_ in LADDER if t in loaded]
    disp_of = {t: d for t, d, _, _, _ in LADDER}
    fam_of = {t: f for t, _, f, _, _ in LADDER}
    nB_of = {t: b for t, _, _, b, _ in LADDER}

    result = {"n_anchored": n, "criterion": ("alias(answers_match)+negation"
             if USE_NEGATION else "alias(answers_match)"),
              "tags": tags, "order": ORDER, "per_stance": {},
              "source_files": used_file}

    print("\n" + "=" * 100)
    # 标签必须跟着 --grader 走：此前写死"别名判据"，跑 negation 时表头会撒谎，
    # 让人以为在看原判据的数字（两个判据的数差别很大，足以影响结论）。
    print("逐立场 × 逐规模的 PKD（%s，锚定集 n=%d）"
          % ("否定感知判据" if USE_NEGATION else "别名判据", n))
    print("=" * 100)
    hdr = "%-10s" % "立场" + "".join("%14s" % disp_of[t] for t in tags)
    print(hdr)
    print("-" * len(hdr))

    for cd in ORDER:
        if cd not in conds:
            continue
        per = {}
        for t in tags:
            v = collections.Counter()
            for k in shared:
                v[grade_eff(loaded[t][k].get("out_" + cd), loaded[t][k])] += 1
            dec = v["PARAM"] + v["CTX"]
            per[t] = dict(pkd=(v["PARAM"] / dec) if dec else None,
                          param=v["PARAM"], ctx=v["CTX"], both=v["BOTH"],
                          none=v["NONE"], dec=dec, decided=dec / n,
                          ci=wilson(v["PARAM"], dec or 1))
        print("%-10s" % cd +
              "".join("%14s" % ("%.3f" % per[t]["pkd"] if per[t]["pkd"] is not None else "n/a")
                      for t in tags))
        result["per_stance"][cd] = per

    # ---- 逐立场：同族内部各相邻对的 Δ 与 McNemar ----
    print("\n" + "=" * 100)
    print("同族内相邻规模的 Δ(pp) = PKD_大 − PKD_小，及配对 McNemar 精确检验")
    print("=" * 100)
    pairs = []
    for fam in ("qwen3", "qwen25"):
        ft = sorted([t for t in tags if fam_of[t] == fam], key=lambda x: nB_of[x])
        for a, b in zip(ft, ft[1:]):
            pairs.append((fam, a, b))
    print("%-10s %-26s %9s %14s %9s %8s" %
          ("立场", "规模对", "Δ(pp)", "翻转 小→大/大→小", "p", "判出率"))
    print("-" * 88)
    for cd in ORDER:
        if cd not in conds:
            continue
        for fam, a, b in pairs:
            # 与下方端点块同一口径：Δ 与 McNemar 都在配对可判样本上算。
            n_pair = f_ab = f_ba = param_a = param_b = 0
            for k in shared:
                ga = grade_eff(loaded[a][k].get("out_" + cd), loaded[a][k])
                gb = grade_eff(loaded[b][k].get("out_" + cd), loaded[b][k])
                if ga in ("BOTH", "NONE") or gb in ("BOTH", "NONE"):
                    continue
                n_pair += 1
                param_a += (ga == "PARAM")
                param_b += (gb == "PARAM")
                if ga == "PARAM" and gb != "PARAM":
                    f_ab += 1
                elif gb == "PARAM" and ga != "PARAM":
                    f_ba += 1
            if n_pair == 0:
                continue
            p = mcnemar_exact(f_ab, f_ba)
            d = 100 * (param_b - param_a) / n_pair
            star = " *" if p < 0.05 else ""
            print("%-10s %-26s %+9.1f %8d/%-5d %9.2e %6.1f%%%s" %
                  (cd, "%s → %s" % (disp_of[a], disp_of[b]), d, f_ab, f_ba, p,
                   100.0 * n_pair / max(len(shared), 1), star))
            result.setdefault("pairs", []).append(dict(
                stance=cd, family=fam, small=disp_of[a], large=disp_of[b],
                delta_pp=d, flip_up=f_ab, flip_down=f_ba, p=p,
                n_paired=n_pair,
                decided_large=100.0 * n_pair / max(len(shared), 1)))

    # ---- 主命题：Δ 的符号是否随立场单调翻转 ----
    print("\n" + "=" * 100)
    print("主命题 N1：Δ 的符号是否随授权强度单调、且跨零")
    print("=" * 100)
    for fam in ("qwen3", "qwen25"):
        ft = sorted([t for t in tags if fam_of[t] == fam], key=lambda x: nB_of[x])
        if len(ft) < 2:
            continue
        a, b = ft[0], ft[-1]
        print("\n族 = %s（%s → %s，%d× 参数比）" %
              (fam, disp_of[a], disp_of[b], nB_of[b] // nB_of[a]))
        print("  %-10s %10s %10s %10s %8s" % ("立场", "PKD_小", "PKD_大", "Δ(pp)", "p"))
        signs = []
        for cd in ORDER:
            if cd not in conds:
                continue
            pa, pb = result["per_stance"][cd][a], result["per_stance"][cd][b]
            if pa["pkd"] is None or pb["pkd"] is None:
                continue
            # 配对口径：Δ 与 p 必须出自**同一批样本**。
            #
            # 此前这里从 per_stance 直接取 PKD 相减，两侧各自用自己可判的样本
            # （分母不同），而下面的 McNemar 又把 NONE/BOTH 也算成"不一致"，
            # 于是 Δ 与 p 来自两个不同的样本集。本文其余各处（cross_env_compare、
            # analyze_stratified、analyze_mde）都是配对口径，只有这里不是。
            #
            # 实测影响不大（两族各立场 Δ 变动 ≤ 2.7 pp，符号与显著性均不变），
            # 但这正是本文反复强调的那个错误类型，不因影响小而保留。
            n_pair = f_ab = f_ba = param_a = param_b = 0
            for k in shared:
                ga = grade_eff(loaded[a][k].get("out_" + cd), loaded[a][k])
                gb = grade_eff(loaded[b][k].get("out_" + cd), loaded[b][k])
                if ga in ("BOTH", "NONE") or gb in ("BOTH", "NONE"):
                    continue
                n_pair += 1
                param_a += (ga == "PARAM")
                param_b += (gb == "PARAM")
                if ga == "PARAM" and gb != "PARAM":
                    f_ab += 1
                elif gb == "PARAM" and ga != "PARAM":
                    f_ba += 1
            if n_pair == 0:
                continue
            pkd_a = param_a / n_pair
            pkd_b = param_b / n_pair
            p = mcnemar_exact(f_ab, f_ba)
            d = 100 * (pkd_b - pkd_a)
            signs.append((cd, d, p))
            print("  %-10s %10.3f %10.3f %+10.1f %8.2e  n=%d%s" %
                  (cd, pkd_a, pkd_b, d, p, n_pair, " *" if p < 0.05 else ""))
        sig = [x for x in signs if x[2] < 0.05]
        if sig:
            lo = min(sig, key=lambda x: x[1])
            hi = max(sig, key=lambda x: x[1])
            print("  → 显著条件 %d/%d，跨度 %.1f pp（%s %+.1f ↔ %s %+.1f）"
                  % (len(sig), len(signs), hi[1] - lo[1], lo[0], lo[1], hi[0], hi[1]))
            pos = [x for x in sig if x[1] > 0]
            neg = [x for x in sig if x[1] < 0]
            print("  → 符号：%d 正 / %d 负  %s"
                  % (len(pos), len(neg),
                     "✅ 跨零（支持 N1）" if pos and neg else "❌ 未跨零"))

            # ---- 单调性必须分开验：符号有序 ≠ 量值单调 ----
            # 这两件事经常被混为一谈。全负→全正只说明"跨零且符号有序"，
            # 它是本文真正的主张；"随授权强度单调上升"是更强的主张，
            # 需要量值本身有序。实测量值有逆序对，故必须分别报告、不得含糊。
            seq_all = [(cd, dd) for cd, dd, _ in signs]
            y = [dd for _, dd in seq_all]
            x = list(range(1, len(y) + 1))
            rho = _spearman(x, y)
            p_perm = _perm_test_spearman(x, y, seed=42, n=200000)
            strict = all(y[i] < y[i + 1] for i in range(len(y) - 1))
            neg_idx = [i for i, v in enumerate(y) if v < 0]
            pos_idx = [i for i, v in enumerate(y) if v > 0]
            ordered_signs = (not neg_idx or not pos_idx
                             or max(neg_idx) < min(pos_idx))
            print("  → 量值单调性：Spearman ρ=%.3f，置换检验 p=%.4f，严格单调=%s"
                  % (rho, p_perm, "是" if strict else "否"))
            if not strict:
                inv = [(seq_all[i][0], y[i], seq_all[i + 1][0], y[i + 1])
                       for i in range(len(y) - 1) if y[i] >= y[i + 1]]
                print("     逆序 %d 处：%s" % (len(inv), "；".join(
                    "%s(%+.1f)≥%s(%+.1f)" % t for t in inv)))
            print("  → 符号有序性（所有负号是否都排在正号之前）：%s"
                  % ("✅ 是" if ordered_signs else "❌ 否"))
            print("     ⚠️  正文只能主张「符号随授权强度有序、且跨零」；"
                  "「量值单调」%s。" % ("成立" if strict else "不成立，不得写成单调递增"))
            result.setdefault("symbol_check", {})[fam] = dict(
                n_sig=len(sig), n_tot=len(signs), span_pp=hi[1] - lo[1],
                # n_pos/n_neg/ordered_signs/span 必须与 seq 同口径（全格），
                # 否则 JSON 内部自相矛盾：早期版本用 sig（仅显著格）算
                # n_pos/n_neg，而 seq 存全格，导致 7 格序列配出 2+4=6 的计数。
                n_pos=len(pos_idx), n_neg=len(neg_idx),
                n_pos_sig=len(pos), n_neg_sig=len(neg),
                crosses_zero=bool(neg_idx and pos_idx),
                ordered_signs=bool(ordered_signs), strict_monotone=bool(strict),
                spearman_rho=rho, spearman_p_perm=p_perm,
                span_pp_all=max(y) - min(y),
                seq=[dict(stance=c, delta_pp=v) for c, v in seq_all])

    with io.open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print("\n已写入: %s" % os.path.abspath(args.out))


if __name__ == "__main__":
    main()
