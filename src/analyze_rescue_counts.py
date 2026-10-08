#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐格统计否定感知判据的救回量（别名 BOTH → 否定感知 PARAM）。

为什么单独写一个脚本
--------------------
正文与图 6 都需要「每格救回多少条」。此前的做法是从两个 ladder 文件的
`per_stance` 里取 param 数相减：

    rescue = LAD["per_stance"][st][t]["param"] - ALIAS["per_stance"][st][t]["param"]

这条式子是错的。两个 `per_stance` 条目各用**各的可判子集**做分母
（别名判据剔掉的 BOTH 更多），相减得到的差值同时掺进了分母变动，
不是「同一批样本里有多少条被改判」。实测与配对口径相差约 14%，
且在「一侧判 NONE」的格上会漏记。

本文其余各处（analyze_scale_ladder 的端点块与相邻对块、analyze_length_confound、
analyze_stratified、analyze_mde）全部走配对口径。此处对齐。

口径
----
- 样本集：D* = ∩_s {i : K_s(i)=1}，即所有规模点都答得出的 649 条。
  K_s 取自 NO-CTX 臂的 noc_follows_param（与 analyze_scale_ladder 同源）。
- 对每个 (模型, 立场)：
    rescue = #{ i ∈ D* : grade_alias(out) == BOTH
                         且 grade_negation(out) == PARAM }
  否定感知只在 BOTH 上生效，故这两个条件等价于
  「别名判 BOTH 且 is_explicit_rejection 为真」。

输出
----
../results/negation_rescue_counts.json
  {"n_dstar": 649, "counts": {tag: {stance: int}}, "totals": {...}}
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analyze_scale_ladder as A                       # noqa: E402
from regrade_negation import is_explicit_rejection     # noqa: E402

RES = os.path.join(HERE, "..", "results")
PROBE = os.path.join(RES, "probe")


def pathof(fn):
    """复刻 analyze_scale_ladder 的路径解析：绝对路径 / ../results / probe。"""
    if os.path.isabs(fn):
        return fn
    if fn.startswith(".."):
        return os.path.join(HERE, fn)
    return os.path.join(PROBE, fn)


def load_records(fn):
    with io.open(pathof(fn), encoding="utf-8") as f:
        d = json.load(f)
    return {A.key_of(r): r for r in d.get("records", [])}


def main():
    stance_recs, kstate, order = {}, {}, []
    for tag, disp, fam, nB, cands in A.LADDER:
        lp = next((c for c in cands if os.path.exists(pathof(c))), None)
        kp = next((c for c in A.KFILES.get(tag, []) if os.path.exists(pathof(c))), None)
        if not lp or not kp:
            print("  [缺] %s（立场 %s / 锚定 %s）" % (disp, lp, kp))
            continue
        stance_recs[tag] = load_records(lp)
        with io.open(pathof(kp), encoding="utf-8") as f:
            kstate[tag] = {A.key_of(r): bool(r.get("noc_follows_param"))
                           for r in json.load(f).get("records", [])}
        order.append((tag, disp))

    if not order:
        sys.exit("没有可用的规模点。")

    covered = set(stance_recs[order[0][0]])
    for tag, _ in order[1:]:
        covered &= set(stance_recs[tag])
    dstar = sorted(i for i in covered
                   if all(kstate[t].get(i, False) for t, _ in order))
    print("共同覆盖 n=%d，|D*|=%d" % (len(covered), len(dstar)))

    counts, totals = {}, {}
    for tag, disp in order:
        counts[tag] = {}
        for st in A.ORDER:
            cd = "out_" + st
            n = 0
            for i in dstar:
                r = stance_recs[tag].get(i)
                if r is None:
                    continue
                ans = r.get(cd)
                if ans is None:
                    continue
                if A.grade(ans, r) == "BOTH" and is_explicit_rejection(ans, r):
                    n += 1
            counts[tag][st] = n
            totals[st] = totals.get(st, 0) + n
        print("  %-14s %s  合计 %d" % (
            disp, {s: counts[tag][s] for s in A.ORDER}, sum(counts[tag].values())))

    print("\n逐立场（五规模点合计）：")
    for st in A.ORDER:
        print("  %-10s %5d" % (st, totals[st]))
    print("  合计     %5d" % sum(totals.values()))

    out = {"n_dstar": len(dstar), "criterion": "alias=BOTH & negation=PARAM",
           "counts": counts, "totals": totals}
    with io.open(os.path.join(RES, "negation_rescue_counts.json"), "w",
                 encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n已写入: %s" % os.path.join(RES, "negation_rescue_counts.json"))


if __name__ == "__main__":
    main()
