# -*- coding: utf-8 -*-
"""双臂 PKD 评测的差异分析：视觉条件 vs 纯文本条件。

为什么必须跑这个
==================================================================
锚定集的定义是"**最小规模模型在没有上下文时也答得出**"（3B 的知识集）。
所以 NO-CTX 条件的答对率在纯文本下必须接近 1，否则筛选前提就被破坏了。

实测（3B, 236 条）：
    --no-vision   noctx_param_rate = 0.9661   <- 符合预期，锚定集成立
    默认（带图）  noctx_param_rate = 0.5381   <- 缺了 0.43

同一个模型、同一批问题、同样的 system prompt，**唯一差别是有没有图**。
这不是噪声，是图像系统性地推翻了模型的参数知识。

要判定它是"视觉证据压过参数知识"还是"无关图片分散注意力"，
唯一的判据是：**模型能不能从图里认出这个物种**。
  - 认得出来（识别对）而仍答错 -> 图像给了正确实体，模型却没用好  -> 分散注意
  - 认不出来（识别错）而仍答错 -> 图像断言了**错误的实体**，
    模型进而按错误实体的属性作答 -> 视觉证据支配

故本脚本同时读入物种识别探针 results/inat_species_id_probe.json。

用法：
  python analyze_interference.py --vision ../results/pkd_3B_named.json \
      --text ../results/pkd_3B_named_novision.json \
      --id-probe ../results/inat_species_id_probe.json \
      --out ../results/interference_3B.json
"""
import argparse, io, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import _content_tokens, answers_match


def norm_yesno(t):
    return bool(re.fullmatch(r"(yes|no)[ ,.]*", t.strip().lower()))


def classify(vrec, nrec):
    """有图答案相对无图答案的退化类型。"""
    v, n = vrec["out_noc"].strip(), nrec["out_noc"].strip()
    if norm_yesno(v):
        return "degenerate"
    vt, nt = _content_tokens(v), _content_tokens(n)
    if vt and nt and set(vt) < set(nt):
        return "generic"
    if answers_match(v, [n])[0]:
        return "compatible"
    return "contradict"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--id-probe", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    def load(p):
        return {r["image"]: r for r in
                json.load(io.open(p, encoding="utf-8"))["records"]}

    V, N = load(args.vision), load(args.text)
    ks = [k for k in V if k in N]
    n = len(ks)

    idok = {}
    if args.id_probe and os.path.exists(args.id_probe):
        for o in json.load(io.open(args.id_probe, encoding="utf-8")):
            idok[o["title"].lower()] = o["guess"].lower().replace("*", "")

    def identified(r):
        """该样本的图像是否被正确识别（按学名或属名匹配）。"""
        g = idok.get(r["wikipedia_title"].lower()) if idok else None
        if g is None:
            return None
        t = r["wikipedia_title"].lower()
        if t in g:
            return True
        return t.split()[0] in g if len(t.split()) > 1 else False

    rows = []
    for k in ks:
        v, nr = V[k], N[k]
        knew = nr["noc_follows_param"]          # 纯文本下知道（锚定集的定义）
        vis_ok = v["noc_follows_param"]         # 带图下仍答对
        rows.append(dict(
            image=k, title=v.get("wikipedia_title", ""), question=v["question"],
            param=v["parametric_answer"], out_text=nr["out_noc"].strip(),
            out_vis=v["out_noc"].strip(),
            knew=knew, vis_ok=vis_ok,
            identified=identified(v),
            kind=(classify(v, nr) if (knew and not vis_ok) else ""),
        ))

    knew = [r for r in rows if r["knew"]]
    broke = [r for r in knew if not r["vis_ok"]]
    kind = {}
    for r in broke:
        kind[r["kind"]] = kind.get(r["kind"], 0) + 1

    ident_known = [r for r in knew if r["identified"] is not None]
    id_rate = (sum(1 for r in ident_known if r["identified"]) / len(ident_known)
               if ident_known else None)
    broke_id = [r for r in broke if r["identified"]]
    broke_noid = [r for r in broke if r["identified"] is False]

    out = {
        "n": n,
        "noctx_text": round(sum(1 for r in rows if r["knew"]) / n, 4),
        "noctx_vision": round(sum(1 for r in rows if r["vis_ok"]) / n, 4),
        "n_knew": len(knew),
        "n_broke": len(broke),
        "breakdown": kind,
        "species_id_rate": None if id_rate is None else round(id_rate, 4),
        "n_identified_ok": sum(1 for r in ident_known if r["identified"]),
        # 关键对照：认得出来 vs 认不出来，各自的"知识被推翻"比例
        "broke_when_identified": (round(len(broke_id) / max(len(ident_known), 1), 4)
                                  if ident_known else None),
        "n_broke_identified": len(broke_id),
        "n_broke_unidentified": len(broke_noid),
        "rows": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, io.open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print("=" * 66)
    print("  视觉条件对参数知识回忆的影响（3B, %d 条锚定样本）" % n)
    print("=" * 66)
    print("  NO-CTX 答对率   纯文本 %.4f   带图像 %.4f   差 %+.4f"
          % (out["noctx_text"], out["noctx_vision"],
             out["noctx_vision"] - out["noctx_text"]))
    print("  纯文本下知道答案 %d 条，其中带图后答错 %d 条 (%.1f%%)"
          % (len(knew), len(broke), 100.0 * len(broke) / max(len(knew), 1)))
    print()
    print("  退化类型：")
    for k in ("degenerate", "generic", "contradict", "compatible"):
        if k in kind:
            print("    %-12s %3d  (%.0f%%)" % (k, kind[k], 100.0 * kind[k] / len(broke)))
    if id_rate is not None:
        print()
        print("  物种识别率（独立探针，问'图里是什么物种'）  %.4f" % id_rate)
        print("    认对且仍答错  %d 条  -> 图像给了正确实体却没用好" % len(broke_id))
        print("    认错且仍答错  %d 条  -> 图像断言了错误实体，模型照错实体作答" % len(broke_noid))
    print()
    print("已写入:", args.out)


if __name__ == "__main__":
    main()
