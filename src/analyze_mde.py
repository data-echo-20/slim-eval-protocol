# -*- coding: utf-8 -*-
"""功效核算：配对设计下的最小可检出效应（MDE）。

为什么单独成脚本
----------------
§4.6.4 此前报的是

    MDE ≈ (z_{1-α/2} + z_{1-β}) · sqrt(2/n) / 2

并把它称作「对配对设计」的近似。这个式子实际是**非配对**、两组等样本量、
且在 p = 0.5 处取最大方差的经典两比例公式 —— 它的 sqrt(2/n)/2 正是
sqrt(0.5/n)。而本文所有主比较都是同一批样本的配对比较（§4.6.5 明确
声明不使用独立样本检验，理由是「会低估功效并高估 p」）。

于是 §4.6.4 与 §4.6.5 自相矛盾：正文一边说不用独立检验，一边用独立检验的
检出限。报出的 7.8 pp 因此**高估**了检出限，方向上属保守，但它不是本文
设计下的检出限。

配对设计的 MDE
--------------
设 n 对样本，不一致率 π_d = P(两模型给出不同判定)。检验
H0: p_{up} = p_{down} 时，δ̂ = p̂_up − p̂_down 在 H0 下方的方差为 π_d / n，
故（Connor 1987；本文按设计期不知道 δ，用 √π_d 作两个 z 项的近似）

    MDE ≈ (z_{1-α/2} + z_{1-β}) · sqrt(π_d / n)

π_d 是**实测**的，故本核算严格说是事后（post-hoc）的：它回答的是
「在每个立场上，按它实际出现的不一致率，多大的 Δ 才能被检出」，
而不是「设计前该招多少样本」。论文中必须连这一点一起写明。

产物供 §4.6.4 正文与 fig14 共同读取，避免两处各算一遍。
"""
import argparse, io, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import answers_match
import analyze_scale_ladder as A

Z_ALPHA2 = 1.959964      # z_{0.975}
Z_BETA = 0.8416212       # z_{0.80}

FAM = {"qwen3": ("qwen3-vl-2b", "qwen3-vl-8b"),
       "qwen25": ("qwen2.5-vl-3b", "qwen2.5-vl-7b")}
ENDLAB = {"qwen3": "Qwen3-VL 8B", "qwen25": "Qwen2.5-VL 7B"}


def grade_neg(rec, cd):
    """否定感知判据：与主结果一致，不另立口径。"""
    out = rec.get("out_" + cd)
    if not out:
        return None
    return A.grade_eff(out, rec)


def mde_paired(pi_d, n):
    """配对 MDE（百分点）。π_d 需先用小数传入。"""
    if n <= 0:
        return float("nan")
    return 100.0 * (Z_ALPHA2 + Z_BETA) * math.sqrt(pi_d / n)


def mde_unpaired(n):
    """非配对、p=0.5 处的 MDE（百分点）。只为与旧报告值对照。"""
    return 100.0 * (Z_ALPHA2 + Z_BETA) * math.sqrt(0.5 / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default=os.path.join(HERE, "..", "results", "probe"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "mde.json"))
    args = ap.parse_args()

    # 否定感知判据：与 analyze_scale_ladder.py --grader negation 走同一个函数。
    # 不这么做就会用别名判据重算一遍 Δ，得到的数与 §4.2 对不上 ——
    # 而 §4.6 引的正是 §4.2 的 Δ。
    from regrade_negation import is_explicit_rejection
    A.USE_NEGATION = True
    A._NEG_FN = is_explicit_rejection

    tags = [t for t, _, _, _, _ in A.LADDER]
    loaded, K = {}, {}
    for tag, _, _, _, cands in A.LADDER:
        d = None
        for fn in cands:
            d = A.load(os.path.join(args.probe, fn))
            if d is not None:
                break
        if d is None:
            sys.exit("缺 %s 的立场文件" % tag)
        loaded[tag] = d["recs"]
        K[tag] = A.load_k(tag, args.probe)

    D = set.intersection(*[{k for k, v in K[t].items() if v} for t in tags]) \
        & set.intersection(*[set(loaded[t]) for t in tags])
    if not D:
        sys.exit("D* 为空")

    print("|D*| = %d；判据 = negation" % len(D))
    print("对照：旧报告值（非配对公式）= %.2f pp" % mde_unpaired(len(D)))

    out = dict(n_anchored=len(D), criterion="negation",
               z_alpha2=Z_ALPHA2, z_beta=Z_BETA,
               mde_unpaired_pp=mde_unpaired(len(D)),
               formula_paired="100*(z_a2+z_b)*sqrt(pi_d/n)",
               note="π_d 为实测，故本核算为事后（post-hoc）",
               families={})
    all_rows = []
    for fam, (ta, tb) in FAM.items():
        rows = []
        for cd in A.ORDER:
            n = up = dn = 0
            for k in D:
                ga = grade_neg(loaded[ta][k], cd)
                gb = grade_neg(loaded[tb][k], cd)
                if ga in (None, "BOTH", "NONE") or gb in (None, "BOTH", "NONE"):
                    continue
                n += 1
                if ga == "PARAM" and gb != "PARAM":
                    up += 1
                elif gb == "PARAM" and ga != "PARAM":
                    dn += 1
            if n == 0:
                continue
            d_pp = 100.0 * (dn - up) / n
            pi_d = (up + dn) / float(n)
            m = mde_paired(pi_d, n)
            rows.append(dict(stance=cd, n=n, n_discordant=up + dn, pi_d=pi_d,
                             mde_pp=m, delta_pp=d_pp,
                             exceeds_mde=bool(abs(d_pp) > m)))
            all_rows.append((fam, cd, n, up + dn, pi_d, m, d_pp))
        out["families"][fam] = rows

    print("\n%-8s %-10s %6s %7s %8s %10s %10s %7s" %
          ("族", "立场", "n", "不一致", "π_d", "配对MDE", "Δ", "Δ超限"))
    for fam, cd, n, nd, pid, m, d in all_rows:
        print("%-8s %-10s %6d %7d %8.3f %10.2f %+10.1f %7s"
              % (fam, cd, n, nd, pid, m, d, "是" if abs(d) > m else "否"))

    meds = {}
    for fam in FAM:
        ms = [r["mde_pp"] for r in out["families"][fam]]
        lo = min(ms)
        hi = max(ms)
        med = sorted(ms)[len(ms) // 2]
        meds[fam] = dict(min=lo, max=hi, median=med)
        print("\n%s：配对 MDE %.2f ~ %.2f pp，中位 %.2f pp（逐立场）"
              % (FAM[fam][0].split("-vl-")[0].upper(), lo, hi, med))
    out["mde_summary"] = meds
    out["mde_range_pp"] = [min(v["min"] for v in meds.values()),
                           max(v["max"] for v in meds.values())]
    out["mde_median_range_pp"] = [min(v["median"] for v in meds.values()),
                                  max(v["median"] for v in meds.values())]
    print("\n总体：逐立场配对 MDE 落在 %.2f ~ %.2f pp；两族中位 %.2f / %.2f pp"
          % (out["mde_range_pp"][0], out["mde_range_pp"][1],
             meds["qwen3"]["median"], meds["qwen25"]["median"]))
    print("（旧报告值 %.2f pp 为非配对公式，高估了检出限）" % out["mde_unpaired_pp"])

    with io.open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n已写入 %s" % os.path.abspath(args.out))


if __name__ == "__main__":
    main()
