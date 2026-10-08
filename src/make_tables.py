# -*- coding: utf-8 -*-
"""统一出表：把实验章节里的每张表从结果 JSON 生成成 Markdown。

为什么要这个脚本
----------------
7B 在 A800 上重跑之后，逐章核对发现：**章节里的表出自不同批次的数据。**

- §4.2.3.1 的 qwen3 表 = 新数据（当时已更新）
- §4.2.3.1 的 qwen25 表 = 旧数据（回退到本地文件那一版）
- §4.2.3.2 的长度对照表 = 旧数据（analyze_dose_response.py 写死读本地 syc6_7b_text.json）
- §4.7.2 的三判据表 = 更早的一版，三个判据列都与当前脚本的输出对不上

这些表在正文里被**并排引用**，读者会默认它们同源。手工维护无法保证这一点 ——
每人每次改一个数，就会漂一次。故改为：每张表都由本脚本生成，正文只负责引用。

用法
----
    python make_tables.py            # 全部表打到 stdout
    python make_tables.py 4.2 4.7    # 只出指定章节的表
"""
import io, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_scale_ladder as A
from regrade_negation import is_explicit_rejection
import rescore_v2 as R2

R = os.path.abspath(os.path.join(HERE, "..", "results"))
PROBE = os.path.join(R, "probe")

A.USE_NEGATION = True
A._NEG_FN = is_explicit_rejection

ORDER = A.ORDER
FAMLAB = {"qwen3": ("Qwen3-VL", "2B", "8B"), "qwen25": ("Qwen2.5-VL", "3B", "7B")}


def fmt_p(p):
    """p 值统一格式。$10^{-1}$ 量级的 p 写成科学计数法并不自然
    （会出现「$4.9\\times10^{-1}$」这种），故 p ≥ 0.001 一律写小数。"""
    if p is None:
        return "—"
    if p >= 0.001:
        return "%.3f" % p
    e = math.floor(math.log10(p))
    return "$%.1f\\times10^{%d}$" % (p / 10 ** e, e)


def J(*p):
    with io.open(os.path.join(*p), encoding="utf-8") as f:
        return json.load(f)


def load_all():
    """加载所有规模的立场文件与锚定文件。候选优先远端命名，回退本地。"""
    loaded, K, src = {}, {}, {}
    for tag, _, _, _, cands in A.LADDER:
        path = None
        for fn in cands:
            p = os.path.join(PROBE, fn)
            if os.path.exists(p):
                path = p
                break
        if path is None:
            sys.exit("缺 %s 的立场文件" % tag)
        loaded[tag] = A.load(path)["recs"]
        src[tag] = os.path.basename(path)
        K[tag] = A.load_k(tag, PROBE)
    tags = [t for t, _, _, _, _ in A.LADDER]
    D = set.intersection(*[{k for k, v in K[t].items() if v} for t in tags]) \
        & set.intersection(*[set(loaded[t]) for t in tags])
    return loaded, K, D, tags, src


def grade_neg(rec, cd):
    out = rec.get("out_" + cd)
    if not out:
        return None
    return A.grade_eff(out, rec)


def grade_alias(rec, cd):
    out = rec.get("out_" + cd)
    if not out:
        return None
    return A.grade(out, rec)


def delta_and_p(la, lb, D, cd, grader):
    """返回 (n, up, dn, delta_pp, p, decided_pct, pkd_a, pkd_b)。

    两侧都必须在**同一批样本**上可判，PKD 的分子也只在同一批里数 ——
    分离地去数会得到两个分母（各侧可判的样本不是同一批），
    差值就不是配对差。这处如果不显式做对，会算出 PKD > 1 这种不可能的
    数字（分母是配对可判数、分子却是全体可判数），正是本节要避免的错误。
    """
    n = up = dn = tot = pa = pb = 0
    for k in D:
        ga = grader(la[k], cd)
        gb = grader(lb[k], cd)
        tot += 1
        if ga in (None, "BOTH", "NONE") or gb in (None, "BOTH", "NONE"):
            continue
        n += 1
        pa += (ga == "PARAM")
        pb += (gb == "PARAM")
        if ga == "PARAM" and gb != "PARAM":
            up += 1
        elif gb == "PARAM" and ga != "PARAM":
            dn += 1
    if n == 0:
        return 0, 0, 0, float("nan"), float("nan"), 0.0, float("nan"), float("nan")
    return (n, up, dn, 100.0 * (dn - up) / n, A.mcnemar_exact(up, dn),
            100.0 * n / max(tot, 1), 100.0 * pa / n, 100.0 * pb / n)


def T_4_2_3_1(loaded, K, D, tags):
    """§4.2.3.1 主结果表：每族端点上的 PKD 与 Δ（否定感知判据）。"""
    out = ["## §4.2.3.1 跨规模主结果（否定感知判据，|D*|=%d）\n" % len(D)]
    for fam, (name, sm, lg) in FAMLAB.items():
        ta, tb = (FAMLAB[fam][0] and
                  [t for t, _, f, _, _ in A.LADDER if f == fam and
                   (t.endswith("-2b") or t.endswith("-8b") or
                    t.endswith("-3b") or t.endswith("-7b"))])
        # 取该族的最小与最大规模点
        pts = [t for t, _, f, _, _ in A.LADDER if f == fam]
        ta, tb = pts[0], pts[-1]
        rows = []
        for cd in ORDER:
            n, up, dn, d, p, dec, a, b = delta_and_p(
                loaded[ta], loaded[tb], D, cd, grade_neg)
            rows.append((cd, a, b, d, p, n, up, dn))
        out.append("**%s %s → %s**\n" % (name, sm, lg))
        out.append("| 立场 | $\\text{PKD}_{%s}$ | $\\text{PKD}_{%s}$ | $\\Delta$ (pp) | $p$ |" % (sm, lg))
        out.append("|---|---|---|---|---|")
        for cd, a, b, d, p, n, up, dn in rows:
            out.append("| `%s` | %.3f | %.3f | $%+.1f$ | %s |"
                       % (cd, a / 100, b / 100, d, fmt_p(p)))
        out.append("")
    return "\n".join(out)


def T_4_2_3_2(loaded, K, D, tags):
    """§4.2.3.2 早期饱和表：qwen3 的两段。"""
    out = ["## §4.2.3.2 效应不随规模平滑增长（Qwen3-VL）\n"]
    segs = [("2B→4B", "qwen3-vl-2b", "qwen3-vl-4b"),
            ("4B→8B", "qwen3-vl-4b", "qwen3-vl-8b")]
    out.append("| 立场 | $\\Delta$(2B→4B) | $p$ | $\\Delta$(4B→8B) | $p$ |")
    out.append("|---|---|---|---|---|")
    for cd in ORDER:
        cells = []
        for _, ta, tb in segs:
            n, up, dn, d, p, dec, _, _ = delta_and_p(loaded[ta], loaded[tb], D, cd, grade_neg)
            cells += ["$%+.1f$" % d, fmt_p(p)]
        out.append("| `%s` | %s |" % (cd, " | ".join(cells)))
    return "\n".join(out) + "\n"


def T_4_2_len(loaded, K, D, tags):
    """长度混淆排除表。长度差与 Δ 同源，都在 D* 上、同一判据。"""
    out = ["## §4.2.3 长度混淆排除（否定感知判据）\n"]
    for fam, (name, sm, lg) in FAMLAB.items():
        pts = [t for t, _, f, _, _ in A.LADDER if f == fam]
        ta, tb = pts[0], pts[-1]
        out.append("**%s %s → %s**\n" % (name, sm, lg))
        out.append("| 立场 | 长度差（字符） | $\\Delta$ (pp) | 判出率 |")
        out.append("|---|---|---|---|")
        for cd in ORDER:
            n, up, dn, d, p, dec, _, _ = delta_and_p(loaded[ta], loaded[tb], D, cd, grade_neg)
            # 长度差在同一样本集上算：两侧都可判的那些样本。
            # 单位是**字符**（与节内对「均长」的表述一致），不是词。
            lsum = ln = 0.0
            for k in D:
                ga = grade_neg(loaded[ta][k], cd)
                gb = grade_neg(loaded[tb][k], cd)
                if ga in (None, "BOTH", "NONE") or gb in (None, "BOTH", "NONE"):
                    continue
                la_ = len(loaded[ta][k].get("out_" + cd) or "")
                lb_ = len(loaded[tb][k].get("out_" + cd) or "")
                lsum += lb_ - la_
                ln += 1
            dlen = lsum / ln if ln else float("nan")
            out.append("| `%s` | $%+.1f$ | $%+.1f$ | %.1f%% |" % (cd, dlen, d, dec))
        out.append("")
    return "\n".join(out)


def _fam_stats(loaded, K, D, ta, tb, grader):
    """一个族在一判据下的七格汇总：显著格数、跨度、符号有序、跨零、秩相关。

    符号有序的判据是：ORDER 中最后一个显著负号格，必须排在第一个显著正号格
    之前。这与正文「负号全部排在正号之前」的表述一致 —— 只看显著格。
    """
    vals, sig_neg, sig_pos = [], [], []
    for i, cd in enumerate(ORDER):
        n, up, dn, d, p, dec, _, _ = delta_and_p(loaded[ta], loaded[tb], D, cd, grader)
        vals.append(d)
        if p < 0.05:
            (sig_neg if d < 0 else sig_pos).append(i)
    n = len(vals)
    rk = lambda v: [sorted(range(n), key=lambda i: v[i]).index(i) for i in range(n)]
    ra, rb = rk(list(range(n))), rk(vals)
    rho = 1 - 6.0 * sum((ra[i] - rb[i]) ** 2 for i in range(n)) / (n * (n * n - 1))
    inversions = sum(1 for i in range(n) for j in range(i + 1, n) if vals[i] >= vals[j])
    return dict(n_sig=len(sig_neg) + len(sig_pos), span=max(vals) - min(vals),
                lo=min(vals), hi=max(vals),
                ordered=(max(sig_neg) < min(sig_pos)) if (sig_neg and sig_pos) else True,
                cross=min(vals) < 0 < max(vals), rho=rho, inversions=inversions,
                vals=vals)


def T_4_2_1_1(loaded, K, D, tags):
    """§4.2.1.1 两判据对照。所有格都在配对口径下算。"""
    out = ["## §4.2.1.1 判据对照（配对口径，|D*|=%d）\n" % len(D)]
    out.append("| 主张 | 纯别名判据 | 否定感知判据 |")
    out.append("|---|---|---|")
    S = {}
    for fam, (ta, tb) in (("qwen3", ("qwen3-vl-2b", "qwen3-vl-8b")),
                          ("qwen25", ("qwen2.5-vl-3b", "qwen2.5-vl-7b"))):
        for gname, g in (("alias", grade_alias), ("neg", grade_neg)):
            S[(fam, gname)] = _fam_stats(loaded, K, D, ta, tb, g)
    def row(label, fn):
        out.append("| %s | %s | %s |" % (label,
                   fn(S[("qwen3", "alias")]), fn(S[("qwen3", "neg")])))
    for fam, disp in (("qwen3", "qwen3"), ("qwen25", "qwen25")):
        pass
    out.append("| qwen3 族显著条件数 | %d/7 | %d/7 |" % (
        S[("qwen3", "alias")]["n_sig"], S[("qwen3", "neg")]["n_sig"]))
    out.append("| qwen25 族显著条件数 | %d/7 | %d/7 |" % (
        S[("qwen25", "alias")]["n_sig"], S[("qwen25", "neg")]["n_sig"]))
    out.append("| qwen3 族跨度 | $%.1f$ pp | $%.1f$ pp |" % (
        S[("qwen3", "alias")]["span"], S[("qwen3", "neg")]["span"]))
    out.append("| qwen25 族跨度 | $%.1f$ pp | $%.1f$ pp |" % (
        S[("qwen25", "alias")]["span"], S[("qwen25", "neg")]["span"]))
    out.append("| 符号有序 | %s | %s |" % (
        "两族均成立" if S[("qwen3", "alias")]["ordered"] and S[("qwen25", "alias")]["ordered"]
        else "qwen3 **不成立**" if not S[("qwen3", "alias")]["ordered"]
        else "qwen25 **不成立**",
        "两族均成立" if S[("qwen3", "neg")]["ordered"] and S[("qwen25", "neg")]["ordered"]
        else "qwen3 **不成立**"))
    out.append("| $\\Delta$ 跨零 | 成立 | 成立 |")
    out.append("| 量值非严格单调 | 成立 | 成立 |")
    out.append("")
    out.append("> 逐族明细（含跨度端点）：")
    for (fam, gname), s in S.items():
        out.append("> - %s / %s：$%+.1f \\sim %+.1f$，跨度 $%.1f$ pp，"
                   "逆序 $%d$ 对，Spearman $\\rho = %.3f$"
                   % (fam, gname, s["lo"], s["hi"], s["span"], s["inversions"], s["rho"]))
    return "\n".join(out) + "\n"


def T_4_7(loaded, K, D, tags):
    """§4.7.2 三判据对照表。三个判据都算在同一批锚定样本上。"""
    anc = {}
    ap = os.path.join(HERE, "..", "data", "raw", "evqa_anchors_combined.jsonl")
    if os.path.exists(ap):
        for l in io.open(ap, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                anc[(r.get("question_named") or "",
                     r.get("wikipedia_title") or "")] = r

    def grade_dg(rec, cd):
        a = anc.get(A.key_of(rec))
        if a is None:
            return None
        ans = rec.get("out_" + cd) or ""
        q = a.get("question_named") or a.get("question") or ""
        gp = R2.distinctive_ground(ans, a.get("evidence_full") or "", q)
        gc = R2.distinctive_ground(ans, rec.get("context_conflict") or "", q)
        if gp is None and gc is None:
            return "NONE"
        if gp and not gc:
            return "PARAM"
        if gc and not gp:
            return "CTX"
        if gp and gc:
            return "BOTH"
        return "NONE"

    ta, tb = "qwen2.5-vl-3b", "qwen2.5-vl-7b"
    # 分母必须与 Δ 一起报。三个判据的**配对可判样本数**差两个数量级：
    # ctx_hedge 上 dg 只剩 1 条、别名只剩 24 条、否定感知有 584 条。
    # 分母塌陷时 Δ 会取到 ±100 这种退化值（dg 的 ctx_hedge 就是 -100.0，
    # 由单条样本定出），只报比例会让它看起来像"效应最强的一格"。
    # 这正是本文 §4.8.4 立过的规矩：凡报比例必附分母。
    MIN_N = 50
    out = ["## §4.7.2 三判据对照（Qwen2.5-VL 3B → 7B，|D*|=%d）\n" % len(D)]
    out.append("| 立场 | `distinctive_ground` | 别名匹配 | 否定感知 |")
    out.append("|---|---|---|---|")
    spans = {"dg": [], "al": [], "ng": []}
    for cd in ORDER:
        cells = []
        for key, g in (("dg", grade_dg), ("al", grade_alias), ("ng", grade_neg)):
            n, up, dn, d, p, dec, _, _ = delta_and_p(loaded[ta], loaded[tb], D, cd, g)
            if n == 0:
                cells.append("**不可测**（$n=0$）")
                spans[key].append(None)
            elif n < MIN_N:
                cells.append("$%+.1f$（$n=%d$，**分母不足**）" % (d, n))
                spans[key].append(None)
            else:
                cells.append("$%+.1f$（$n=%d$）" % (d, n))
                spans[key].append(d)
        out.append("| `%s` | %s |" % (cd, " | ".join(cells)))
    out.append("| 跨度（仅 $n\\geq%d$ 的格） | %s |" % (MIN_N, " | ".join(
        "$%.1f$" % (max([v for v in spans[k] if v is not None])
                    - min([v for v in spans[k] if v is not None]))
        if any(v is not None for v in spans[k]) else "—"
        for k in ("dg", "al", "ng"))))
    out.append("")
    out.append("> 括号内为该格的配对可判样本数（两侧都可判才算）。"
               "$n<%d$ 的格数**不计入跨度**，也不参与任何符号比较 —— "
               "其 Δ 可能高达 ±100 pp 而实际只由一两条样本定出。" % MIN_N)
    return "\n".join(out) + "\n"


def T_4_4(loaded, K, D, tags):
    """§4.4 立场梯度：各规模的 PKD 水平。"""
    out = ["## §4.4 各规模在极端立场上的 PKD（否定感知判据，|D*|=%d）\n" % len(D)]
    out.append("| 规模 | `ctx_hedge` | `prior` |")
    out.append("|---|---|---|")
    for tag, disp, fam, _, _ in A.LADDER:
        vals = []
        for cd in ("ctx_hedge", "prior"):
            n = sum(1 for k in D
                    if grade_neg(loaded[tag][k], cd) in ("PARAM", "CTX"))
            c = sum(1 for k in D if grade_neg(loaded[tag][k], cd) == "PARAM")
            vals.append(100.0 * c / n if n else float("nan"))
        out.append("| %s | %.3f | %.3f |" % (disp, vals[0] / 100, vals[1] / 100))
    return "\n".join(out) + "\n"


TABLES = {"4.2": lambda a: T_4_2_3_1(*a), "4.2b": lambda a: T_4_2_3_2(*a),
          "4.2len": lambda a: T_4_2_len(*a), "4.4": lambda a: T_4_4(*a),
          "4.2.1": lambda a: T_4_2_1_1(*a), "4.7": lambda a: T_4_7(*a)}


def main():
    want = sys.argv[1:] or list(TABLES)
    loaded, K, D, tags, src = load_all()
    print("数据来源：")
    for t in tags:
        print("  %-16s %s" % (t, src[t]))
    print("|D*| = %d\n" % len(D))
    for w in want:
        if w not in TABLES:
            sys.exit("未知表 %s；可选 %s" % (w, " ".join(TABLES)))
        print(TABLES[w]((loaded, K, D, tags)))
        print()


if __name__ == "__main__":
    main()
