# -*- coding: utf-8 -*-
"""分域检验：在每个槽位**内部**重算规模效应。

为什么要单独成脚本
------------------
§4.8 的数值原本是临时算的，没有落盘产物，图与正文都无从复算。
这与跨环境比对是同一类问题（见 cross_env_compare.py 的说明），故一并固化。

口径
----
- 样本限在锚定集 D*（各规模都能答出的样本）内，与主结果一致；
- 槽位取自记录的 `slot_coarse`（place / other / quantity / diet / time）；
- 对每个槽位内部的每条样本做配对翻转统计，Δ = PKD_大 − PKD_小；
- **强制同时报告翻转对数**：小样本上比例会到 ±100pp，分母只有一两对时
  这个比例没有意义。只报比例不报分母是分域分析最容易犯的错。
"""
import argparse, collections, io, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import answers_match
from analyze_scale_ladder import (ORDER, LADDER, KFILES, mcnemar_exact,
                                  key_of, load, load_k)

FAM = {"qwen3": ["qwen3-vl-2b", "qwen3-vl-8b"],
       "qwen25": ["qwen2.5-vl-3b", "qwen2.5-vl-7b"]}


USE_NEGATION = False
_NEG_FN = None


def grade(rec, cd):
    out = rec.get("out_" + cd)
    if not out:
        return None
    mp = answers_match(out, rec.get("parametric_aliases") or [])[0]
    mc = answers_match(out, rec.get("context_answer_aliases") or [])[0]
    if mp and mc:
        # BOTH 的默认处置是剔除。但显式驳回上下文的答案应当计 PARAM ——
        # 方向性明确，且两判据的结论必须能对照，故做成全局开关而非硬编码。
        if USE_NEGATION and _NEG_FN and _NEG_FN(out, rec):
            return "PARAM"
        return "BOTH"
    if mp:
        return "PARAM"
    if mc:
        return "CTX"
    return "NONE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default=os.path.join(HERE, "..", "results", "probe"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results",
                                                 "stratified.json"))
    ap.add_argument("--min-n", type=int, default=100,
                    help="低于此样本量的槽位只报数、不出结论")
    ap.add_argument("--no-anchor", action="store_true",
                    help="不做 D* 锚定，直接在全部冲突样本上算（仅供与旧结果核对）")
    ap.add_argument("--grader", choices=["alias", "negation"], default="alias",
                    help="alias=纯别名；negation=把显式驳回上下文的 BOTH 救回 PARAM。"
                         "本节表须与 §4.2 主表同判据，否则两节的分域/总体对照不成立。")
    args = ap.parse_args()

    global USE_NEGATION, _NEG_FN
    if args.grader == "negation":
        from regrade_negation import is_explicit_rejection as _fn
        USE_NEGATION, _NEG_FN = True, _fn
    if USE_NEGATION:
        print("判据 = negation（否定感知）")

    tags = [t for t, _, _, _, _ in LADDER]
    loaded, K = {}, {}
    for tag, disp, _, _, cands in LADDER:
        # 候选按优先级试，与 analyze_scale_ladder 一致：远端命名优先，
        # 找不到才回退本地既有文件。只取 cands[0] 会在只有本地文件时误报「缺」。
        d = None
        for fn in cands:
            d = load(os.path.join(args.probe, fn))
            if d is not None:
                break
        if d is None:
            sys.exit("缺 %s 的立场文件（候选 %s）" % (tag, " | ".join(cands)))
        loaded[tag] = d["recs"]
        K[tag] = load_k(tag, args.probe)

    common = set.intersection(*[set(loaded[t]) for t in tags])
    if args.no_anchor:
        D = set(common)
        print("⚠️  --no-anchor：未做 D* 锚定，样本 = 全部冲突样本 %d 条" % len(D))
    else:
        D = set.intersection(*[{k for k, v in K[t].items() if v} for t in tags]) & common
        print("锚定集 |D*| = %d" % len(D))
    if not D:
        sys.exit("D* 为空")

    # 槽位分布
    slot = {}
    for k in D:
        slot[k] = loaded[tags[0]][k].get("slot_coarse") or "?"
    dist = collections.Counter(slot.values())
    print("\n槽位分布（D* 内）：")
    for s, n in dist.most_common():
        flag = "" if n >= args.min_n else "  ← 样本量不足，不作结论"
        print("  %-10s %4d (%.1f%%)%s" % (s, n, 100.0 * n / len(D), flag))

    out = dict(n_dstar=len(D), min_n=args.min_n, anchored=not args.no_anchor,
               slot_dist={s: n for s, n in dist.items()}, families={})
    for fam, (ta, tb) in FAM.items():
        print("\n族 = %s（%s → %s）" % (fam, ta, tb))
        print("  %-10s %-10s %6s %9s %12s %10s" %
              ("槽位", "立场", "n", "Δ(pp)", "翻转 小→大/大→小", "p"))
        famrows = []
        for s in dist:
            ks = [k for k in D if slot[k] == s]
            if not ks:
                continue
            for cd in ORDER:
                n = up = dn = 0
                for k in ks:
                    ga = grade(loaded[ta][k], cd)
                    gb = grade(loaded[tb][k], cd)
                    if ga in (None, "BOTH", "NONE") or gb in (None, "BOTH", "NONE"):
                        continue
                    n += 1
                    if ga == "PARAM" and gb != "PARAM":
                        dn += 1        # 小跟随、大不跟随
                    elif gb == "PARAM" and ga != "PARAM":
                        up += 1        # 大跟随
                if n == 0:
                    continue
                # Δ 必须与全文同口径：PKD(大) − PKD(小)，分母是**可判样本数 n**。
                d = 100.0 * (up - dn) / n
                # 另一个估计量：只用不一致对。它忽略全部一致对，量值通常大 3~6 倍，
                # 与 §4.2 的 Δ 不可比。§4.8 早期版本误用了它，故一并输出以便核对，
                # 但**不作为报告值**。
                d_disc = 100.0 * (up - dn) / (up + dn) if (up + dn) else float("nan")
                p = mcnemar_exact(dn, up)
                usable = len(ks) >= args.min_n
                famrows.append(dict(slot=s, stance=cd, n=n, delta_pp=d,
                                    delta_discordant_pp=d_disc,
                                    flip_small_to_large=up, flip_large_to_small=dn,
                                    p=p, usable=usable))
                print("  %-10s %-10s %6d %+9.1f %8d/%-5d %10.2e%s" %
                      (s, cd, n, d, up, dn, p, "" if usable else "  (不可用)"))
        out["families"][fam] = famrows

    with io.open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n已写入 %s" % os.path.abspath(args.out))


if __name__ == "__main__":
    main()
