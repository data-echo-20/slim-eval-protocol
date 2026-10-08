# -*- coding: utf-8 -*-
"""把所有 PKD 评测结果汇总成论文要用的表。

为什么要有这一步：不同臂（规模 × 视觉/纯文本 × 量化）是分次跑出来的，
文件夹里散着十几个 json。论文里的规模曲线必须从**同一份口径**生成，
手工抄数字迟早抄错，而且"哪个数该进表"这件事本身需要被固化下来。

关键裁决口径（见 研究方案.md §6.1.1）：
  knowledge rate 只在**纯文本臂**上判读。带图臂是有倾向的测量，
  它测的是"视觉证据介入后参数知识还剩多少"，不是"模型知不知道"。

用法：
  python summarize.py --dir ../results --out ../results/scale_curve.json
"""
import argparse, collections, glob, io, json, os, re


def load(path):
    try:
        d = json.load(io.open(path, encoding="utf-8"))
    except Exception:
        return None
    s = d.get("summary")
    if not s:
        return None
    recs = d.get("records") or []
    s = dict(s)
    s["_path"] = os.path.basename(path)
    s["_n_vision_ok"] = sum(1 for r in recs if r.get("noc_follows_param"))
    s["_n_trunc"] = sum(1 for r in recs if r.get("truncated"))

    # ★ 从 records 重算 PKD，而不是直接信 summary 里的数。
    #   理由：summary 是**评测当时**的口径算出来的。早期版本的 eval_pkd 把
    #   rewrite 失败（context_conflict 为空）的样本也算进了 CONFLICT 分母——
    #   而 eval_pkd 在这类样本上喂的是 `context_conflict or context_original`，
    #   也就是**与参数知识一致的**上下文，它们永远不可能表现出冲突，
    #   等于往 PKD 里掺 0。实测 236 条里有 12 条（5.1%）如此，会把 PKD
    #   系统性压低约 0.02。
    #   这里重算一次，既修好了旧结果，也让新旧结果在同一个口径下可比——
    #   不需要为了改口径把已经花掉的 GPU 时间重跑一遍。
    u = [r for r in recs if r.get("usable")]
    conf = [r for r in u if (r.get("context_conflict") or "").strip()]
    nc = len(conf)
    if nc:
        f = float(nc)
        s["n_conflict"] = nc
        s["PKD_rate"] = round(sum(1 for r in conf if r.get("cfl_follows_param")) / f, 4)
        s["ctx_follow_rate"] = round(sum(1 for r in conf if r.get("cfl_follows_ctx")) / f, 4)
        s["both_rate"] = round(sum(1 for r in conf
                                  if r.get("cfl_follows_param")
                                  and r.get("cfl_follows_ctx")) / f, 4)
        s["neither_rate"] = round(sum(1 for r in conf
                                      if not r.get("cfl_follows_param")
                                      and not r.get("cfl_follows_ctx")) / f, 4)
        s["_recomputed"] = True
    # 知识基线仍用全部 usable：NO-CTX 根本不读上下文，
    # 那 12 条的知识基线测量是有效的。
    if u:
        s["noctx_param_rate"] = round(
            sum(1 for r in u if r.get("noc_follows_param")) / float(len(u)), 4)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="../results")
    ap.add_argument("--out", default="../results/scale_curve.json")
    args = ap.parse_args()

    runs = []
    for p in sorted(glob.glob(os.path.join(args.dir, "pkd_*.json"))):
        s = load(p)
        if s:
            runs.append(s)

    # 归组：模型 -> {vision, text}
    groups = collections.defaultdict(dict)
    for s in runs:
        arm = "vision" if s.get("vision") else "text"
        groups[s["model"]][arm] = s

    rows = []
    for model in sorted(groups):
        g = groups[model]
        v, t = g.get("vision"), g.get("text")
        rows.append(dict(
            model=model,
            n=(v or t or {}).get("n_usable"),
            n_conflict=(v or t or {}).get("n_conflict"),
            pkd_vision=(v or {}).get("PKD_rate"),
            pkd_text=(t or {}).get("PKD_rate"),
            ctx_follow_vision=(v or {}).get("ctx_follow_rate"),
            ctx_follow_text=(t or {}).get("ctx_follow_rate"),
            # 只认纯文本臂
            knowledge_rate=(t or {}).get("noctx_param_rate"),
            knowledge_rate_vision=(v or {}).get("noctx_param_rate"),
            interference=((v or {}).get("noctx_param_rate") -
                          (t or {}).get("noctx_param_rate")
                          if v and t else None),
            drop_rate=(v or t or {}).get("drop_rate"),
        ))

    def fmt(x, p=4):
        return "-" if x is None else ("%%.%df" % p) % x

    print("=" * 104)
    print("  规模曲线汇总（knowledge_rate 只取纯文本臂，见 研究方案.md §6.1.1）")
    print("=" * 104)
    print("  %-22s %5s %5s %9s %9s %9s %9s %11s" %
          ("model", "n", "n_con", "PKD(vis)", "PKD(txt)", "know(txt)", "know(vis)",
           "干扰(vis-txt)"))
    print("  " + "-" * 104)
    for r in rows:
        print("  %-22s %5s %5s %9s %9s %9s %9s %11s" %
              (r["model"], r["n"], r["n_conflict"],
               fmt(r["pkd_vision"]), fmt(r["pkd_text"]),
               fmt(r["knowledge_rate"]), fmt(r["knowledge_rate_vision"]),
               fmt(r["interference"], 3)))
    print()
    print("  读法：")
    print("    n_con           CONFLICT 条件的**有效分母**（有真正冲突上下文的样本）。")
    print("                    < n 的部分是 rewrite 失败、退化成 CONSIST 的样本，已剔除。")
    print("    PKD(vis)/(txt)  冲突条件下跟随参数知识的比例；两臂都报，不得混为一谈")
    print("    know(txt)       锚定集成立性的自检，应接近 1；明显偏低说明该规模上筛选前提被破坏")
    print("    干扰            负数 = 图像推翻了参数知识（见 §6.1.2）")
    print()
    warn = [r for r in rows if r["knowledge_rate"] is not None
            and r["knowledge_rate"] < 0.9]
    if warn:
        print("  ⚠ 以下规模的纯文本 knowledge_rate < 0.9，其 PKD 在解读前必须先解释知识基线为何下降：")
        for r in warn:
            print("     %s  know(txt)=%s" % (r["model"], fmt(r["knowledge_rate"])))
        print("     可能原因：量化损伤 / prompt 与模型偏好不合 / 筛选前提确实被破坏")
        print()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"rows": rows, "runs": runs},
              io.open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("已写入:", args.out)


if __name__ == "__main__":
    main()
