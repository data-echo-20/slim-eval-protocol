# -*- coding: utf-8 -*-
"""锚定协议 D* = ∩_s { i : K_s(i)=1 } —— 跨规模阶梯的知识交集。

为什么必须有这一步
------------------
927 条主实验集本身是**用 Qwen2.5-VL 3B ∩ 7B 的知识交集**筛出来的
（`RESULTS.md` §变更表第 1 行：1146 条，3B∩7B 多规模交集筛选）。
新加入的规模点（Qwen3-VL 2B/4B/8B）是**另一个模型族**，
它们不一定知道这 927 条所锚定的全部事实。

若某条事实新模型 Z 根本不知道（K_Z(i)=0），那么在冲突条件下
它"跟随参数知识"这件事**没有定义** —— 不是"选择不跟随"，而是"无知识可跟随"。
把它算进 PKD 分母等于往指标里掺噪声，且这会**系统性偏向小模型**：
小模型知识少 → 大量不可判定 → 分母缩小、结论失真。

锚定协议把评测限制在所有规模点**都答得出**的样本上，
使 Δ_s(ι) 的跨规模比较只在"知识持有量相同"的前提下进行。

K_s(i) 怎么来
-------------
`eval_pkd.py` 在 NO-CTX 条件下（中性 system prompt，见其 SYS_NOCTX 注释）
逐条记录 `noc_follows_param = match(out_noc, parametric_aliases)`。
本脚本读该字段。

用法
----
  python anchor_ladder.py --probe ../results/probe --out ../results/ladder_anchors.json
"""
import argparse, collections, io, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def key_of(r):
    return (r.get("question_named") or "", r.get("wikipedia_title") or "")


def load_k(path):
    """读一个 pkd_*.json，返回 {key: bool}。缺文件返回 None。"""
    if not os.path.exists(path):
        return None
    d = json.load(io.open(path, encoding="utf-8"))
    out = {}
    for r in d.get("records", []):
        # usable=False 的记录也算：NO-CTX 是独立于冲突条件的一次调用，
        # 空/截断只是说明答不出，正是我们要测的 K_s=0。
        out[key_of(r)] = bool(r.get("noc_follows_param"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default=os.path.join(HERE, "..", "results", "probe"))
    ap.add_argument("--results", default=os.path.join(HERE, "..", "results"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "ladder_anchors.json"))
    # tag -> 候选文件（优先远端，回退本地）
    ap.add_argument("--models", default=(
        "qwen3-vl-2b=qwen3vl_qwen3-vl-2b_text.json,"
        "qwen3-vl-4b=qwen3vl_qwen3-vl-4b_text.json,"
        "qwen3-vl-8b=qwen3vl_qwen3-vl-8b_text.json,"
        "qwen2.5-vl-3b=../results/pkd_3b_comb_text.json,"
        "qwen2.5-vl-7b=../results/pkd_7b_comb_text.json"),
        help="逗号分隔 tag=相对probe的文件名")
    args = ap.parse_args()

    ks, order = {}, []
    for spec in args.models.split(","):
        tag, fn = spec.split("=", 1)
        p = fn if os.path.isabs(fn) else os.path.join(
            args.results if fn.startswith("..") else args.probe, os.path.basename(fn))
        if fn.startswith(".."):
            p = os.path.join(HERE, fn)
        k = load_k(p)
        if k is None:
            print("  [缺] %-16s %s" % (tag, p))
            continue
        ks[tag] = k
        order.append(tag)
        hit = sum(1 for v in k.values() if v)
        print("  [有] %-16s %4d 条  K=1 占 %5.1f%%  ← %s"
              % (tag, len(k), 100 * hit / max(len(k), 1), os.path.basename(p)))

    if len(ks) < 2:
        sys.exit("\n可用模型不足 2 个。")

    print("\n" + "=" * 84)
    print("知识率 K_s 与锚定集 D* 的逐规模收敛")
    print("=" * 84)
    common = None
    for tag in order:
        common = set(ks[tag]) if common is None else (common & set(ks[tag]))
    # 只在共同覆盖的样本上算 K
    n_common = len(common)
    print("所有模型共同覆盖的样本 n = %d\n" % n_common)
    print("%-16s %10s %10s" % ("模型", "K=1 条数", "知识率"))
    print("-" * 40)
    kk = {}
    for tag in order:
        hit = sum(1 for i in common if ks[tag].get(i))
        kk[tag] = hit
        print("%-16s %10d %9.1f%%" % (tag, hit, 100 * hit / max(n_common, 1)))

    # D* 随规模加入逐步收缩
    print("\n锚定集 D* 随规模加入的收缩（先小后大，与 ORDER 一致）：")
    ordered = sorted(order, key=lambda t: kk[t])  # 按知识量从小到大加
    acc, prev = None, n_common
    seq = []
    print("  %-34s %8s %8s" % ("加入后", "|D*|", "收缩"))
    for tag in ordered:
        s = {i for i in common if ks[tag].get(i)}
        acc = s if acc is None else (acc & s)
        print("  + %-32s %8d %8d" % (tag, len(acc), prev - len(acc)))
        prev = len(acc)
        seq.append(dict(model=tag, n_anchored=len(acc)))

    # 全交集
    allk = {i for i in common if all(ks[t].get(i) for t in order)}
    print("\n★ D*（所有规模 K=1）= %d 条（占共同覆盖 %.1f%%）"
          % (len(allk), 100 * len(allk) / max(n_common, 1)))

    # 逐模型：它自己 K=0 的条数（即被锚定协议剔除的量）
    print("\n各模型被剔除的条数（在全交集口径下）：")
    for tag in order:
        miss = sum(1 for i in common if not ks[tag].get(i))
        print("  %-16s 缺 %4d 条（%.1f%%）" % (tag, miss, 100 * miss / max(n_common, 1)))

    with io.open(args.out, "w", encoding="utf-8") as f:
        json.dump(dict(n_common=n_common, k_rate={t: kk[t] / max(n_common, 1) for t in order},
                       n_anchored=len(allk), sequence=seq,
                       anchored_keys=sorted("%s||%s" % k for k in allk)),
                  f, ensure_ascii=False, indent=1)
    print("\n已写入: %s" % os.path.abspath(args.out))


if __name__ == "__main__":
    main()
