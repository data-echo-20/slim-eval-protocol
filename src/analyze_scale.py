# -*- coding: utf-8 -*-
"""规模曲线分析：PKD 是否随模型规模上升？

对 eval_pkd.py 的输出做配对统计（纯后处理，免费，可反复迭代）。

为什么必须做**配对**检验
==================================================================
两个规模是在**同一批样本**上测的（同一份锚定集、同一份冲突上下文），
所以"3B 跟随参数、7B 跟随上下文"的样本本身就携带了最直接的证据。
把两个比例各自做独立 z 检验会严重低估功效 —— 它丢掉了"同一道题"这个
配对信息。用 McNemar 精确检验只在**不一致对**（3B跟随/7B不跟随 与 反之）
上做二项检验，是同设计下功效最高的做法。

关键教训（本项目踩到）
==================================================================
聚合层面上一度看到"PKD 随规模显著下降"（927 条上 3B 0.282 → 7B 0.178,
McNemar p<1e-4）。但**按数据集域拆开后**，这个下降完全来自 landmarks：
  inaturalist（知识匹配干净、两规模 ctx_follow 完全相同 0.8684）:
      3B vs 7B = 5 : 6, p=0.50 —— **完全平局**
  landmarks: 137 : 40, p<1e-4 —— 下降全部来自这里
于是在少数类上的"聚合显著"其实是一个 Simpon 型陷阱的候选。
区分这两种解释的办法只有一个：**按域拆开分别检验**。

用法：
  python analyze_scale.py --results ../results --out ../results/scale_analysis.json
"""
import argparse, collections, io, json, math, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def key_of(r):
    """样本身份：锚定问题 + 实体名。跨规模的记录靠它配对。"""
    return (r.get("question_named") or r.get("question") or "",
            r.get("wikipedia_title") or "")


def mcnemar_exact(b, c):
    """McNemar 精确检验（双侧）。b/c 是两个不一致格的计数。

    返回 p 值。用精确二项而非卡方：不一致对常常只有个位数，
    卡方近似在此不可靠（实测 inaturalist 上 b+c=11，卡方给不出可信 p）。
    """
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    # 双侧 = 2 * P(X <= min(b,c))，截断到 1
    tail = sum(math.comb(m, i) for i in range(k + 1)) / (2.0 ** m)
    return min(1.0, 2.0 * tail)


def wilson(k, n, z=1.96):
    """Wilson 区间：小样本比例比正态近似稳（inaturalist 只有 152 条）。"""
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def load_records(results_dir, model, arm):
    p = os.path.join(results_dir, "pkd_%s_comb_%s.json" % (model, arm))
    if not os.path.exists(p):
        return None
    return json.load(io.open(p, encoding="utf-8"))["records"]


def conflict_records(recs, domain=None, dom_map=None):
    """取有效的冲突样本（PKD 的分母）。"""
    out = []
    for r in recs:
        if not r.get("usable"):
            continue
        if not (r.get("context_conflict") or "").strip():
            continue
        if domain and dom_map is not None:
            if dom_map.get(key_of(r)) != domain:
                continue
        out.append(r)
    return out


def stats(recs):
    n = len(recs)
    if n == 0:
        return dict(n=0)
    f = sum(1 for r in recs if r["cfl_follows_param"])
    x = sum(1 for r in recs if r["cfl_follows_ctx"])
    return dict(n=n, pkd=f / n, ctx=x / n,
                pkd_ci=wilson(f, n), ctx_ci=wilson(x, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    ap.add_argument("--anchors", default="../data/raw/evqa_anchors_combined.jsonl",
                    help="用于取 dataset_name（域）")
    ap.add_argument("--out", default="../results/scale_analysis.json")
    ap.add_argument("--models", default="3b,7b")
    ap.add_argument("--arms", default="text,vision")
    args = ap.parse_args()

    # 域查找表
    dom_map = {}
    if os.path.exists(args.anchors):
        for l in io.open(args.anchors, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            dom_map[(r.get("question_named") or "", r.get("wikipedia_title") or "")] \
                = r.get("dataset_name", "?")

    models = args.models.split(",")
    arms = args.arms.split(",")
    out = {"per_arm": {}, "paired": {}}

    for arm in arms:
        recs = {m: load_records(args.results, m, arm) for m in models}
        if any(v is None for v in recs.values()):
            continue
        print("=" * 78)
        print(" 臂 = %-8s（同锚定集、同量化状态）" % arm)
        print("=" * 78)

        # --- 各模型总体 ---
        print("%-8s %6s %9s %10s %10s" % ("模型", "n", "PKD", "ctx_follow", "know"))
        for m in models:
            u = [r for r in recs[m] if r.get("usable")]
            conf = conflict_records(recs[m])
            s = stats(conf)
            know = sum(1 for r in u if r.get("noc_follows_param")) / max(len(u), 1)
            print("%-8s %6d %9.4f %10.4f %10.4f"
                  % (m.upper(), s["n"], s.get("pkd", 0), s.get("ctx", 0), know))
            out["per_arm"].setdefault(arm, {})[m] = dict(
                n=s["n"], pkd=s.get("pkd"), ctx=s.get("ctx"),
                pkd_ci=s.get("pkd_ci"), ctx_ci=s.get("ctx_ci"), know=know)

        # --- 按域拆分 + 配对检验 ---
        domains = sorted({v for v in dom_map.values() if v and v != "?"})
        print()
        print(" 按域拆分（配对 McNemar，仅在两个规模都有效的冲突样本上）")
        print("-" * 78)
        print("%-14s %6s %9s %9s %9s %8s %7s"
              % ("域", "n", "PKD(low)", "PKD(high)", "b:a-only", "c:b-only", "p"))
        for dn in domains:
            a, b = [], []
            ka, kb = {}, {}
            for m in models:
                for r in conflict_records(recs[m], dn, dom_map):
                    (ka if m == models[0] else kb)[key_of(r)] = r
            shared = [k for k in ka if k in kb]
            only_low = sum(1 for k in shared
                           if ka[k]["cfl_follows_param"] and not kb[k]["cfl_follows_param"])
            only_high = sum(1 for k in shared
                            if kb[k]["cfl_follows_param"] and not ka[k]["cfl_follows_param"])
            s_low = stats(list(ka.values()))
            s_high = stats(list(kb.values()))
            p = mcnemar_exact(only_low, only_high)
            print("%-14s %6d %9.4f %9.4f %9d %8d %7.4f%s"
                  % (dn, len(shared), s_low.get("pkd", 0), s_high.get("pkd", 0),
                     only_low, only_high, p, "  ← 显著" if p < 0.05 else ""))
            out["paired"].setdefault(arm, {})[dn] = dict(
                n_paired=len(shared), pkd_low=s_low.get("pkd"),
                pkd_high=s_high.get("pkd"), only_low=only_low,
                only_high=only_high, p=p)

        # --- 全部合并（不拆域）---
        ka = {key_of(r): r for r in conflict_records(recs[models[0]])}
        kb = {key_of(r): r for r in conflict_records(recs[models[1]])}
        shared = [k for k in ka if k in kb]
        b = sum(1 for k in shared
                if ka[k]["cfl_follows_param"] and not kb[k]["cfl_follows_param"])
        c = sum(1 for k in shared
                if kb[k]["cfl_follows_param"] and not ka[k]["cfl_follows_param"])
        p = mcnemar_exact(b, c)
        print("-" * 78)
        print("%-14s %6d %9.4f %9.4f %9d %8d %7.4f%s"
              % ("全部", len(shared), stats(list(ka.values())).get("pkd", 0),
                 stats(list(kb.values())).get("pkd", 0), b, c, p,
                 "  ← 显著" if p < 0.05 else ""))
        out["paired"].setdefault(arm, {})["ALL"] = dict(
            n_paired=len(shared), only_low=b, only_high=c, p=p)
        print()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, io.open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("已写入:", args.out)


if __name__ == "__main__":
    main()
