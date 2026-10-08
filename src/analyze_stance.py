# -*- coding: utf-8 -*-
"""立场梯度的槽位内泛化检验。

本脚本要排除的替代解释
======================
主表把四种立场连同"原论文指令式 prompt"放在一条梯度上。一个替代解释是：
**这个梯度是不是只由某一种编辑类型（比如国别替换）撑起来的？**
如果把编辑类型拆开，梯度还在不在？

做法：把 927 条按 edit_slot 分组，在每个组**内部**各自计算
（7B 准确率 − 3B 准确率），看四个立场是否仍按"对参数知识授权递减"单调下降。

判据：梯度在某个槽位内成立 = 该槽位的四个差值随授权递减而单调不增。
不成立的多是 n<50 的小槽位（地板效应 + 置信区间宽），须并报 n。

用法：
  python analyze_stance.py --probe results/probe --arm text
"""
import argparse, collections, io, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import answers_match

# 立场顺序：对参数知识的授权从大到小
STANCES = ["prior", "own_only", "neutral", "ctx_only"]


def mcnemar_exact(b, c):
    m = b + c
    if m == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(m, i) for i in range(min(b, c) + 1))
               / (2.0 ** m))


def load(path):
    d = json.load(io.open(path, encoding="utf-8"))
    return d["meta"], {r["question_named"]: r for r in d["records"]}


def hit(rec, stance):
    al = rec.get("parametric_aliases") or [rec.get("parametric_answer") or ""]
    return answers_match(rec.get("out_" + stance) or "", al)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="../results/probe")
    ap.add_argument("--arm", default="text", choices=("text", "vision"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    m3, A = load(os.path.join(args.probe, "syc_%s_%s.json" % ("3b", args.arm)))
    m7, B = load(os.path.join(args.probe, "syc_%s_%s.json" % ("7b", args.arm)))
    stances = [s for s in STANCES if all(("out_" + s) in r for r in A.values())]
    ks = [k for k in A if k in B and A[k].get("usable") and B[k].get("usable")]

    print("臂=%s  n=%d  立场=%s" % (args.arm, len(ks), stances))
    print()
    print("【A】全样本梯度（复现主表）")
    print(" %-10s %8s %8s %9s %7s %7s %11s"
          % ("立场", "3B", "7B", "差", "3B对7B错", "7B对3B错", "p"))
    rows = []
    for s in stances:
        c3 = sum(hit(A[k], s) for k in ks)
        c7 = sum(hit(B[k], s) for k in ks)
        ol = sum(1 for k in ks if hit(A[k], s) and not hit(B[k], s))
        oh = sum(1 for k in ks if hit(B[k], s) and not hit(A[k], s))
        n = len(ks)
        p = mcnemar_exact(ol, oh)
        rows.append((s, c3 / n, c7 / n, c7 / n - c3 / n, ol, oh, p))
        print(" %-10s %8.4f %8.4f %+9.4f %7d %7d %11.1e"
              % (s, c3 / n, c7 / n, c7 / n - c3 / n, ol, oh, p))
    mono_all = all(rows[i][3] >= rows[i + 1][3] for i in range(len(rows) - 1))
    print(" 全样本梯度单调: %s" % ("是" if mono_all else "否"))

    print()
    print("【B】槽位内梯度（泛化检验：拆开编辑类型后梯度是否仍在）")
    groups = collections.defaultdict(list)
    for k in ks:
        groups[A[k].get("edit_slot")].append(k)
    print(" %-10s %5s | %-34s | %-6s | %s"
          % ("编辑槽位", "n", " ".join("%-8s" % s for s in stances),
             "单调", "最小 n 提示"))
    out = {"arm": args.arm, "all": [], "by_slot": {}}
    for slot, ks_s in sorted(groups.items(), key=lambda x: -len(x[1])):
        cells, ok = [], True
        for s in stances:
            c3 = sum(hit(A[k], s) for k in ks_s)
            c7 = sum(hit(B[k], s) for k in ks_s)
            cells.append(c7 / len(ks_s) - c3 / len(ks_s))
        ok = all(cells[i] >= cells[i + 1] for i in range(len(cells) - 1))
        out["by_slot"][slot] = dict(n=len(ks_s), deltas=cells, monotone=ok)
        print(" %-10s %5d | %-34s | %-6s | %s"
              % (slot, len(ks_s), " ".join("%+8.3f" % c for c in cells),
                 "是" if ok else "否",
                 "" if len(ks_s) >= 50 else "样本量小，地板效应"))

    n_big = [s for s, v in out["by_slot"].items() if v["n"] >= 30]
    n_ok = [s for s in n_big if out["by_slot"][s]["monotone"]]
    print()
    print(" n>=30 的槽位 %d 个，其中梯度单调 %d 个（%.0f%%）"
          % (len(n_big), len(n_ok), 100.0 * len(n_ok) / max(len(n_big), 1)))
    print(" 不单调的槽位: %s"
          % (", ".join(s for s in n_big if s not in n_ok) or "无"))

    if args.out:
        json.dump(out, io.open(args.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("\n已写入:", args.out)


if __name__ == "__main__":
    main()
