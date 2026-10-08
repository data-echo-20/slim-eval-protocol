# -*- coding: utf-8 -*-
"""跨环境复现比对：同一模型、同一数据、同一协议，在两台机器上各跑一遍。

为什么单独成脚本
----------------
这个比对原本是临时算的，结果只留在日志里。两个后果：
  1. 论文里引的数字没有可复算的来源；
  2. 绘图脚本只能自己另算一遍，而它算的很可能是**另一个量**
     （比如把「逐立场的两环境差」画成了「逐立场的参数知识率」）。
故把比对固化成脚本，产物落盘，图与正文都读它。

比对两个量，必须分开报告
------------------------
  PKD 偏离      —— 本文报告的核心量，判据级复现
  逐字一致率    —— 字符串级复现，跨环境会漂移
两者不是一回事：贪心解码下同一输入仍会因浮点累加顺序、kernel 选择、
批次组成不同而产出不同但等价的表述（`Switzerland` vs `in Switzerland`）。
本文主张的是前者稳、后者不稳，所以必须同时给出。
"""
import argparse, io, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import answers_match

ORDER = ["ctx_only", "orig", "ctx_hedge", "neutral", "own_only", "len_ctrl", "prior"]


def key_of(r):
    return (r.get("question_named") or "", r.get("wikipedia_title") or "")


def load(p):
    with io.open(p, encoding="utf-8") as f:
        return {key_of(r): r for r in json.load(f).get("records", [])}


def grade(rec, cond):
    """返回该条件下模型给了哪个值。与 analyze_scale_ladder 的口径一致。"""
    out = rec.get("out_" + cond)
    if not out:
        return None
    pa = rec.get("parametric_aliases") or []
    ca = rec.get("context_answer_aliases") or []
    # answers_match 返回 (命中?, 分档)，分档只用于声明判定口径。
    # 直接 if answers_match(...) 恒为真 —— 非空元组永远 truthy，
    # 于是所有输出都被判成 PARAM，两个环境的 PKD 会双双变成 100%。
    m_p = answers_match(out, pa)[0]
    m_c = answers_match(out, ca)[0]
    if m_p and m_c:
        return "BOTH"
    if m_p:
        return "PARAM"
    if m_c:
        return "CTX"
    return "NONE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", required=True)
    ap.add_argument("--remote", required=True)
    ap.add_argument("--label", default="A800")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    A, B = load(args.local), load(args.remote)
    shared = sorted(set(A) & set(B))
    if not shared:
        sys.exit("两个文件没有共同样本 —— 键不一致，无法比对")

    rows = []
    for cd in ORDER:
        # 配对口径：两侧都必须在**同一批样本**上算 PKD。
        # 各算各的会得到两个分母（两侧的不可判定样本不是同一批），
        # 差就不是配对差，而本文的整个比较是配对的。
        # 实测代价：不配对时 ctx_hedge 会报 22.4% vs 20.3%（−2.1pp），
        # 配对后是同一批样本上的差，量值更小也更保守。
        n = dp_a = dp_b = 0
        same = tot = 0
        for k in shared:
            ra, rb = A[k], B[k]
            ga, gb = grade(ra, cd), grade(rb, cd)
            tot += 1
            if (ra.get("out_" + cd) or "").strip() == (rb.get("out_" + cd) or "").strip():
                same += 1
            if ga in (None, "BOTH", "NONE") or gb in (None, "BOTH", "NONE"):
                continue
            n += 1
            dp_a += (ga == "PARAM")
            dp_b += (gb == "PARAM")
        pkd_a = 100.0 * dp_a / n if n else float("nan")
        pkd_b = 100.0 * dp_b / n if n else float("nan")

        rows.append(dict(cond=cd, pkd_local=pkd_a, pkd_remote=pkd_b,
                         delta_pp=pkd_b - pkd_a, verbatim=100.0 * same / max(tot, 1),
                         n=n, n_local_only=dp_a, n_remote_only=dp_b))

    mx = max(abs(r["delta_pp"]) for r in rows)
    vv = sum(r["verbatim"] for r in rows) / len(rows)
    out = dict(n_shared=len(shared), label=args.label,
               local=os.path.basename(args.local),
               remote=os.path.basename(args.remote),
               max_abs_pkd_delta_pp=mx, mean_verbatim_pct=vv, rows=rows)
    with io.open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print("比对 %d 条共同样本" % len(shared))
    print("%-10s %10s %10s %9s %10s" % ("立场", "本地 PKD", "远端 PKD", "Δ(pp)", "逐字一致"))
    for r in rows:
        print("%-10s %9.1f%% %9.1f%% %+9.1f %9.1f%%"
              % (r["cond"], r["pkd_local"], r["pkd_remote"], r["delta_pp"],
                 r["verbatim"]))
    print("\n最大 PKD 偏离 %.1f pp；平均逐字一致率 %.1f%%" % (mx, vv))
    print("已写入 %s" % os.path.abspath(args.out))


if __name__ == "__main__":
    main()
