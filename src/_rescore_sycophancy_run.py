# -*- coding: utf-8 -*-
"""中立项实验的打分与判定：跟随上下文 vs 跟随参数知识，按 prompt 立场与规模分解。

输出每个 (立场, 模型) 的 PKD(可判定) 及配对比较，并给出本实验的判定结论：

  R1 若 neutral 下 7B 的 PKD 仍显著低于 3B（配对 McNemar p<0.05 且方向一致）
     -> H_instr（"只是跟指令"）被排除，主发现成立。
  R2 若 neutral 下两者都塌到 ~0 且无差异
     -> H_instr 成立，本文主发现作废。

用法：
  python rescore_sycophancy.py --dir ../results/probe \
      --anchors ../data/raw/evqa_anchors_combined.jsonl
"""
import argparse, collections, io, json, math, os, sys, glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rescore_v2 as R2


def mcnemar_exact(b, c):
    m = b + c
    if m == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(m, i) for i in range(min(b, c) + 1))
               / (2.0 ** m))


def grade(rec, cd, anchor):
    """对某个立场的输出判 PARAM/CTX/BOTH/NONE。复用 v2 的排他接地判据。"""
    ans = rec.get("out_" + cd) or ""
    q = anchor.get("question_named") or anchor.get("question") or ""
    gp = R2.distinctive_ground(ans, anchor.get("evidence_full") or "", q)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="../results/probe")
    ap.add_argument("--anchors", default="../data/raw/evqa_anchors_combined.jsonl")
    ap.add_argument("--out", default="../results/probe/_syc_summary.json")
    args = ap.parse_args()

    anc = {}
    for l in io.open(args.anchors, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            anc[(r.get("question_named") or "", r.get("wikipedia_title") or "")] = r

    def key(r):
        return (r.get("question_named") or "", r.get("wikipedia_title") or "")

    files = sorted(glob.glob(os.path.join(args.dir, "syc_*.json")))
    if not files:
        sys.exit("在 %s 下没找到 syc_*.json" % args.dir)

    by_model = collections.defaultdict(dict)
    conds = None
    for p in files:
        d = json.load(io.open(p, encoding="utf-8"))
        cs = d["meta"]["conds"]
        conds = conds or cs
        model = d["meta"]["model"]
        recs = {key(r): r for r in d["records"] if r.get("usable")}
        for cd in cs:
            v = collections.Counter()
            for k, r in recs.items():
                v[grade(r, cd, anc.get(k, {}))] += 1
            dec = v["PARAM"] + v["CTX"]
            by_model[model][cd] = dict(
                n=len(recs), param=v["PARAM"], ctx=v["CTX"],
                both=v["BOTH"], none=v["NONE"], decidable=dec,
                abstain=(v["BOTH"] + v["NONE"]) / max(len(recs), 1),
                pkd=(v["PARAM"] / dec) if dec else None,
                recs=recs)

    out = {"conds": conds, "by_model": {}}
    for model, per in sorted(by_model.items()):
        print("=" * 96)
        print(" 模型 = %s" % model)
        print("=" * 96)
        print(" %-10s %6s %7s %7s %7s %8s %10s" %
              ("立场", "n", "PARAM", "CTX", "弃权", "弃权率", "PKD(可判定)"))
        out["by_model"][model] = {}
        for cd in conds:
            if cd not in per:
                continue
            s = per[cd]
            print(" %-10s %6d %7d %7d %7d %8.1f%% %10s" %
                  (cd, s["n"], s["param"], s["ctx"], s["both"] + s["none"],
                   100 * s["abstain"],
                   ("%.4f" % s["pkd"]) if s["pkd"] is not None else "n/a"))
            out["by_model"][model][cd] = {k: v for k, v in s.items()
                                          if k != "recs"}

        # 同模型内跨立场的配对比较 —— 回答"PKD 有多少是被 prompt 逼出来的"
        if all(cd in per for cd in conds):
            base = "neutral" if "neutral" in per else conds[0]
            print("\n %s 相对各立场的配对变化：" % base)
            for cd in conds:
                if cd == base:
                    continue
                a, b = per[base]["recs"], per[cd]["recs"]
                sh = [k for k in a if k in b]
                pa = sum(1 for k in sh if grade(a[k], base, anc.get(k, {})) == "PARAM")
                pb = sum(1 for k in sh if grade(b[k], cd, anc.get(k, {})) == "PARAM")
                ol = sum(1 for k in sh
                         if grade(a[k], base, anc.get(k, {})) == "PARAM"
                         and grade(b[k], cd, anc.get(k, {})) == "CTX")
                oh = sum(1 for k in sh
                         if grade(b[k], cd, anc.get(k, {})) == "PARAM"
                         and grade(a[k], base, anc.get(k, {})) == "CTX")
                print("   %-10s PARAM %d → %d   （%s→%s %d 对，%s→%s %d 对，p=%.4f）"
                      % (cd, pa, pb, base, cd, ol, cd, base, oh,
                         mcnemar_exact(ol, oh)))

    # 跨模型：本实验的核心判定
    ms = sorted(by_model)
    if len(ms) >= 2:
        lo, hi = ms[0], ms[-1]
        print("\n" + "=" * 96)
        print(" 判定：%s vs %s 在 **中立项 (neutral)** 下的配对比较" % (lo, hi))
        print("=" * 96)
        if "neutral" in by_model[lo] and "neutral" in by_model[hi]:
            A = by_model[lo]["neutral"]["recs"]
            B = by_model[hi]["neutral"]["recs"]
            sh = [k for k in A if k in B]
            ol = sum(1 for k in sh
                     if grade(A[k], "neutral", anc.get(k, {})) == "PARAM"
                     and grade(B[k], "neutral", anc.get(k, {})) == "CTX")
            oh = sum(1 for k in sh
                     if grade(B[k], "neutral", anc.get(k, {})) == "PARAM"
                     and grade(A[k], "neutral", anc.get(k, {})) == "CTX")
            p = mcnemar_exact(ol, oh)
            pa = by_model[lo]["neutral"]["pkd"]
            pb = by_model[hi]["neutral"]["pkd"]
            print(" n=%d  PKD %s=%.4f → %s=%.4f  （仅%s判PARAM %d / 仅%s判PARAM %d，p=%.4f）"
                  % (len(sh), lo, pa or -1, hi, pb or -1, lo, ol, hi, oh, p))
            if pa is not None and pb is not None and pb < pa and p < 0.05:
                print("\n ▶ 结论 R1：中立项下规模效应**依然成立** —— "
                      "H_instr（'只是遵从指令'）被排除。")
            elif (pa is not None and pb is not None
                  and abs(pb - pa) < 0.02 and p > 0.05):
                print("\n ▶ 结论 R2：中立项下规模效应**消失** —— "
                      "H_instr 成立，主发现须作废。")
            else:
                print("\n ▶ 结论 R3：既非清晰成立也非清晰消失，须逐格复核。")
            out["judgement"] = dict(n=len(sh), low=lo, high=hi, pkd_low=pa,
                                    pkd_high=pb, only_low=ol, only_high=oh, p=p)

    json.dump(out, io.open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n已写入:", args.out)


if __name__ == "__main__":
    main()
