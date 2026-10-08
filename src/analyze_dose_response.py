# -*- coding: utf-8 -*-
"""立场阶梯的规模效应符号分析（本文主打贡献 N1）。

对每个 system-prompt 立场 cd，计算同族两规模 (3B, 7B) 的 PKD：
    PKD_s(cd) = |PARAM| / (|PARAM| + |CTX|)
    Delta(cd) = PKD_7B(cd) - PKD_3B(cd)
并做逐样本配对 McNemar 精确检验（判定为 PARAM 与否）。

同时给出弃权率（BOTH+NONE 占比）作为第二个因变量，检验"口径杠杆"是否够翻符号。

用法：
  python analyze_dose_response.py --anchors ../data/raw/evqa_anchors_combined.jsonl
"""
import argparse, collections, io, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rescore_v2 as R2


def mcnemar_exact(b, c):
    m = b + c
    if m == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(m, i) for i in range(min(b, c) + 1)) / (2.0 ** m))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def grade(rec, cd, anchor):
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


# 立场按"对参数知识的授权强度"由弱到强排序（人为预注册顺序，非事后拟合）
ORDER = ["ctx_only", "orig", "ctx_hedge", "neutral", "own_only", "len_ctrl", "prior"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(HERE, "..", "results", "probe"))
    ap.add_argument("--anchors", default=os.path.join(HERE, "..", "data", "raw",
                                                      "evqa_anchors_combined.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results",
                                                  "stance_dose_response.json"))
    args = ap.parse_args()

    anc = {}
    for l in io.open(args.anchors, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            anc[(r.get("question_named") or "", r.get("wikipedia_title") or "")] = r

    def key(r):
        return (r.get("question_named") or "", r.get("wikipedia_title") or "")

    models = {}
    conds = None
    # 候选按优先级试，与 analyze_scale_ladder 的 LADDER 一致：远端 A800 重跑
    # 出来的优先，找不到才回退本地既有文件。此前这里写死本地文件名，
    # 于是在 7B 于 A800 重跑之后，本节仍读旧的本地产物 ——
    # 结果是本文两张关系最近的表（§4.2 主表与 §4.6 长度对照）出自**不同批次的数据**，
    # 而它们在正文里被并排引用。这类「同一批样本」的假设必须由代码保证。
    for tag, cands in [("3B", ["syc6_qwen2.5-vl-3b_text.json", "syc6_3b_text.json"]),
                       ("7B", ["syc6_qwen2.5-vl-7b_text.json", "syc6_7b_text.json"])]:
        path = None
        for fn in cands:
            p = os.path.join(args.dir, fn)
            if os.path.exists(p):
                path = p
                break
        if path is None:
            sys.exit("缺 %s 的立场文件（候选 %s）" % (tag, " | ".join(cands)))
        print("  %s 读 %s" % (tag, os.path.basename(path)))
        d = json.load(io.open(path, encoding="utf-8"))
        conds = conds or d["meta"]["conds"]
        models[tag] = {key(r): r for r in d["records"] if r.get("usable")}

    k3, k7 = "3B", "7B"
    shared = [k for k in models[k3] if k in models[k7]]
    n = len(shared)
    print("同族两规模配对样本 n = %d\n" % n)

    rows = []
    print("%-11s %8s %8s %9s %-22s %8s %8s" %
          ("立场", "PKD_3B", "PKD_7B", "Δ(pp)", "配对翻转 3B→7B / 7B→3B", "p", "弃权 3B/7B"))
    print("-" * 96)
    for cd in ORDER:
        if cd not in conds:
            continue
        st = {}
        for tag in (k3, k7):
            v = collections.Counter()
            for k in shared:
                v[grade(models[tag][k], cd, anc.get(k, {}))] += 1
            dec = v["PARAM"] + v["CTX"]
            st[tag] = dict(
                pkd=(v["PARAM"] / dec) if dec else None,
                param=v["PARAM"], ctx=v["CTX"], dec=dec,
                abstain=(v["BOTH"] + v["NONE"]) / n)
        a = sum(1 for k in shared
                if grade(models[k3][k], cd, anc.get(k, {})) == "PARAM"
                and grade(models[k7][k], cd, anc.get(k, {})) != "PARAM")
        b = sum(1 for k in shared
                if grade(models[k7][k], cd, anc.get(k, {})) == "PARAM"
                and grade(models[k3][k], cd, anc.get(k, {})) != "PARAM")
        p = mcnemar_exact(a, b)
        d3, d7 = st[k3]["pkd"], st[k7]["pkd"]
        delta = (d7 - d3) if (d3 is not None and d7 is not None) else None
        lo3, hi3 = wilson(st[k3]["param"], st[k3]["dec"] or 1)
        lo7, hi7 = wilson(st[k7]["param"], st[k7]["dec"] or 1)
        print("%-11s %8s %8s %9s %-22s %8s %6.1f%%/%.1f%%" % (
            cd,
            "%.4f" % d3 if d3 is not None else "n/a",
            "%.4f" % d7 if d7 is not None else "n/a",
            "%+.1f" % (100 * delta) if delta is not None else "n/a",
            "%d / %d" % (a, b),
            "%.2e" % p if p < 0.001 else "%.4f" % p,
            100 * st[k3]["abstain"], 100 * st[k7]["abstain"]))
        rows.append(dict(cond=cd, pkd_3b=d3, pkd_7b=d7, delta_pp=100 * delta if delta is not None else None,
                         flip_3to7=a, flip_7to3=b, p=p,
                         abstain_3b=st[k3]["abstain"], abstain_7b=st[k7]["abstain"],
                         ci_3b=[lo3, hi3], ci_7b=[lo7, hi7],
                         n_param_3b=st[k3]["param"], n_param_7b=st[k7]["param"],
                         dec_3b=st[k3]["dec"], dec_7b=st[k7]["dec"]))

    ds = [r["delta_pp"] for r in rows if r["delta_pp"] is not None]
    print("\nΔ 跨度 = %.1f pp  （%.1f … %.1f）" % (max(ds) - min(ds), min(ds), max(ds)))
    sign = sorted(set("正" if x > 0 else "负" for x in ds))
    print("出现的符号：%s  →  %d 个立场为负，%d 个为正"
          % ("/".join(sign), sum(1 for x in ds if x < 0), sum(1 for x in ds if x > 0)))

    # 长度对照分解：len_ctrl 是多少词的安慰剂，prior 是同等结构的授权句
    pm = {r["cond"]: r for r in rows}
    if "len_ctrl" in pm and "prior" in pm:
        print("\n长度对照分解（同为 17 词、同结构，仅授权语义不同）：")
        print("  len_ctrl（无授权）Δ = %+.1f pp" % pm["len_ctrl"]["delta_pp"])
        print("  prior  （有授权）Δ = %+.1f pp" % pm["prior"]["delta_pp"])
        print("  ⇒ 授权语义单独贡献 %+.1f pp，长度本身贡献 %+.1f pp"
              % (pm["prior"]["delta_pp"] - pm["len_ctrl"]["delta_pp"],
                 pm["len_ctrl"]["delta_pp"]))

    with io.open(args.out, "w", encoding="utf-8") as f:
        json.dump({"n_shared": n, "order": ORDER, "rows": rows}, f,
                  ensure_ascii=False, indent=1)
    print("\n已写入: %s" % os.path.abspath(args.out))


if __name__ == "__main__":
    main()
