# -*- coding: utf-8 -*-
"""评测框架复现的判分：选项选择率 + 三档抽取。

四个条件的选项映射一致（这是刻意的，见 eval_harness.py 的偏离说明）：
    A = 参数真值（抵抗冲突上下文）
    B = 冲突注入值（采纳被篡改的上下文）
    C = 无法确定

判据分三档，**分档必须并报**，因为原文用 logprob、本文用生成：
    letter : 输出里解析出明确的选项字母（"A" / "Answer: B" / "(C)" 等）
    text   : 未给字母，但输出文本命中参数别名（→A）或冲突值（→B）
    none   : 两者都判不出（记为不可判定，不计入分母）

主指标：P(A) = 选择参数真值的比例。规模效应 = P(A)_7B − P(A)_3B。
    >0 → 大模型更抵抗冲突上下文（与 Xie et al. 同向）
    <0 → 大模型更采纳冲突上下文（与 ConflictBank 同向）

用法：
  python rescore_harness.py --dir results/probe --prefix harness
"""
import argparse, collections, glob, io, json, math, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import answers_match, contains_any


def mcnemar_exact(b, c):
    m = b + c
    if m == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(m, i) for i in range(min(b, c) + 1))
               / (2.0 ** m))


_LET = re.compile(r"(?:^|[\s(\[=:*])([ABC])(?:[\s).,:\]]|$)")


def parse_letter(txt):
    """抽取选项字母。优先取 "Answer:"/"答案" 之后的第一个字母。"""
    t = (txt or "").strip()
    if not t:
        return None
    m = re.search(r"(?:answer|选项|答案)\s*[:：]?\s*\(?([ABC])\b", t, re.I)
    if m:
        return m.group(1).upper()
    # 输出本身极短（"A" / "B." / "(C)"）
    if len(t) <= 6:
        m = _LET.search(" " + t + " ")
        if m:
            return m.group(1).upper()
    # 退化：整段里只有一个选项字母
    hits = {x.upper() for x in _LET.findall(" " + t + " ")}
    if len(hits) == 1:
        return hits.pop()
    return None


def judge(txt, rec, cd="", swapped=False):
    """返回 (选择, 档位)，**已折算回语义**。选择 ∈ {'A','B','C',None}。

    这里返回的 'A' 恒表示"选了参数真值"、'B' 恒表示"选了冲突值"，
    与字母在 prompt 里的实际位置无关 —— swapped=True 时字母含义相反。
    折算在此处完成，调用方不必关心 --order 设置。
    """
    if not (txt or "").strip():
        return None, "none"
    let = parse_letter(txt)
    tier = "letter"
    if not let:
        # 无字母：按文本内容判
        param_al = rec.get("parametric_aliases") or \
            [rec.get("parametric_answer") or ""]
        ctx_al = rec.get("context_answer_aliases") or \
            [rec.get("context_answer") or ""]
        p = answers_match(txt, [a for a in param_al if a])[0]
        x = answers_match(txt, [a for a in ctx_al if a])[0]
        if p and not x:
            let, tier = "A", "text"
        elif x and not p:
            let, tier = "B", "text"
        elif re.search(r"\b(uncertain|cannot be determined|unknown|"
                       r"无法确定|不确定)\b", txt, re.I):
            let, tier = "C", "text"
        else:
            return None, "none"
    if swapped and let in ("A", "B"):
        let = "B" if let == "A" else "A"
    return let, tier


def load(p):
    d = json.load(io.open(p, encoding="utf-8"))
    return d["meta"], {r["question_named"]: r for r in d["records"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="../results/probe")
    ap.add_argument("--prefix", default="harness")
    ap.add_argument("--model-letters", default="3b,7b")
    args = ap.parse_args()

    models = args.model_letters.split(",")
    files = {}
    for m in models:
        g = glob.glob(os.path.join(args.dir, "%s_%s_*.json" % (args.prefix, m)))
        if not g:
            sys.exit("找不到 %s_%s_*.json" % (args.prefix, m))
        files[m] = g[0]

    data = {}
    conds = None
    for m, p in files.items():
        meta, recs = load(p)
        data[m] = recs
        conds = conds or meta["conds"]
        print("载入 %-46s model=%s n=%d" % (os.path.basename(p),
                                            meta["model"], len(recs)))

    # 共同样本
    ks = [k for k in data[models[0]]
          if all(k in data[m] for m in models)
          and all(data[m][k].get("usable") for m in models)]
    print("\n共同可用样本 n=%d\n" % len(ks))

    def J(m, k, cd):
        rec = data[m][k]
        return judge(rec.get("out_" + cd), rec, cd,
                     swapped=bool(rec.get("swap_" + cd)))

    # 顺序核对：若结果文件带 swap 标记，先确认两半样本的比例
    sw = [sum(1 for k in ks if data[models[0]][k].get("swap_" + c))
          for c in conds]
    if any(sw):
        print("选项顺序：交换样本数 = %s（共 %d）\n" % (sw, len(ks)))

    print("【1】各条件的选项选择率（分母为全部共同样本）")
    hdr = " %-13s" % "条件"
    for m in models:
        hdr += " | %-22s" % ("%s: P(A)/P(B)/P(C)" % m)
    hdr += " | %s" % "规模效应 ΔP(A)"
    print(hdr)
    print(" " + "-" * (len(hdr) + 6))
    out = {"n": len(ks), "conds": {}}
    for cd in conds:
        line = " %-13s" % cd
        pa = {}
        for m in models:
            cnt = collections.Counter()
            for k in ks:
                s, _ = J(m, k, cd)
                cnt[s] += 1
            n = len(ks)
            pa[m] = cnt["A"] / n
            line += " | %6.3f /%6.3f /%6.3f " % (
                cnt["A"] / n, cnt["B"] / n, cnt["C"] / n)
        d = pa[models[-1]] - pa[models[0]]
        line += " | %+8.3f" % d
        print(line)
        out["conds"][cd] = {"pA": pa, "delta_pA": d}

    print("\n【2】配对检验：3B 选 A 而 7B 不选 vs 反之（只在 A / 非A 二分上）")
    print(" %-13s %8s %8s %11s" % ("条件", "3B对7B错", "7B对3B错", "McNemar p"))
    ORD = models
    for cd in conds:
        a_ = [k for k in ks if J(ORD[0], k, cd)[0] == "A"]
        b_ = {k for k in ks if J(ORD[1], k, cd)[0] == "A"}
        ol = sum(1 for k in a_ if k not in b_)
        oh = sum(1 for k in b_ if k not in set(a_))
        p = mcnemar_exact(ol, oh)
        out["conds"][cd]["mcnemar"] = [ol, oh, p]
        print(" %-13s %8d %8d %11.1e" % (cd, ol, oh, p))

    print("\n【3】抽取档位分布（letter vs text vs none），用于说明判据可靠性")
    for m in models:
        cnt = collections.Counter()
        for cd in conds:
            for k in ks:
                _, tier = J(m, k, cd)
                cnt[tier] += 1
        tot = sum(cnt.values())
        print("  %-6s letter=%.3f  text=%.3f  none=%.3f"
              % (m, cnt["letter"] / tot, cnt["text"] / tot, cnt["none"] / tot))

    print("\n【4】判定：两篇工作的框架是否给出相反的规模效应符号")
    neg = [c for c in conds if out["conds"][c]["delta_pA"] < 0]
    pos = [c for c in conds if out["conds"][c]["delta_pA"] > 0]
    print("  ΔP(A) 为负（大模型更采纳冲突）= %s" % (neg or "无"))
    print("  ΔP(A) 为正（大模型更抵抗冲突）= %s" % (pos or "无"))
    if neg and pos:
        print("  ▶ 同一批样本、同一对模型，仅换评测框架即翻转符号 —— "
              "复现了两篇工作的方向分歧，且定位到框架本身为致因。")
    else:
        print("  ▶ 符号未随框架翻转：两篇工作的分歧不能由 prompt 框架单独解释。")

    op = os.path.join(args.dir, "_%s_summary.json" % args.prefix)
    json.dump(out, io.open(op, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n已写入:", op)


if __name__ == "__main__":
    main()
