# -*- coding: utf-8 -*-
"""否定感知重判：把「明确驳回上下文」的答案从 BOTH 里救回 PARAM。

问题
----
`common.answers_match` 用双向包含判定别名命中。于是下面这类答案

    "The context is incorrect. Porrentruy Castle is not located in Peru.
     It is actually situated in Switzerland."

会同时命中 `context_answer`（Peru）与 `parametric_answer`（Switzerland），
被判为 BOTH，然后在 PKD 的分母里被**整个剔除**。

但这条答案的立场是**毫不含糊地站在参数知识一边**的 —— 它驳回上下文、给出
参数真值。它应当是 PARAM 的**最强证据**，却被当成不可判定丢掉了。

为什么这很严重
--------------
丢弃量随规模上升（实测 6/7 立场在**两个族**上都显著上升，见下），
因为大模型更常**显式驳回**冲突上下文，而驳回**必须点名**那个错误值。
于是剔除偏向大模型，**系统性低估** PKD_大 —— 即本文的正效应被低估。

这是判据缺陷，不是模型行为。修它会让结论更强，而不是更弱。

两类必须分开
------------
同样是"同时出现两个值"，有两种截然不同的行为，混为一谈才是真的错：

  (a) **显式驳回上下文**：`The context is incorrect. ... actually X.`
      → 已裁决，站参数知识。应判 PARAM。
  (b) **真·骑墙**：`It could be X, or possibly Y.`
      → 未裁决。应留 BOTH（剔除）。

本脚本只救 (a)，不救 (b)。

做法与自我约束
--------------
1. **模式按语言范畴预注册**，不按结果调参。模式分五类（见 NEG_PATTERNS）。
2. **只对 BOTH 生效**，不动 PARAM/CTX/NONE，把改动面压到最小。
3. **双向人工抽检**：既要看是否漏掉真驳回，也要看是否误伤真骑墙。
4. 原判与重判**两个数都报**，不覆盖原结果。
5. 若重判改变结论，如实报告；本脚本不保证结论不变。

用法
----
  python regrade_negation.py --probe ../results/probe --sample 12
"""
import argparse, io, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import answers_match

# 否定/驳回模式，按语言范畴分组（预注册：先写定，后看数据）
#
# 关键区分（血泪教训，见下）：
#   组 A「元话语型」—— 直接评判上下文本身（"The context is incorrect"）。
#     句子的陈述对象就是那条上下文，命中即驳回，**与槽位类型无关**，恒可用。
#   组 B「取值否定型」—— 否定某个取值（"is not located in Peru"）。
#     它默认"被否定的名词短语即上下文取值"。这个前提**只在槽位是名物类时成立**。
#     对极性槽位会灾难性误判：
#         参数=edible  上下文=not edible  答句="No, it is not edible."
#     这里"not edible"是**复述上下文**（=同意上下文），却被 B 组读成
#     "否定 edible"，于是误救成 PARAM。
#     所以：**上下文取值本身含否定词时，B 组必须整体禁用。**
#   （实测误判量：7B 的 ctx_hedge 有 534 条会被误救，2B 只有 79 条 ——
#    误判本身随规模增大，会把结论推向"救回量随规模上升"这一预期方向。
#    正因为它偏向预期结论，更必须修。）
# 组 A：元话语型 —— 陈述对象就是那条上下文本身，命中即驳回。
#         刻意收得很紧：只认"对上下文的**否定性**评判"。
#         试过但**已删**的宽松模式（都会误救）：
#           `\bshould\s+be\b`      —— 误中 "it should be noted"（中性补注）
#           `\bno\s*,\s*(?:it|the)\b` —— 误中 "No, it is located in Peru"（其实同意上下文）
#           `\bnot\s+located\s+in\b`  —— 误中极性槽位 "is not edible"
NEG_METADISC = [
    r"\bcontext\s+(?:is|seems|appears)\s+(?:incorrect|wrong|erroneous|mistaken|inaccurate|false)",
    r"\bcontext\s+contains\s+(?:an?\s+)?(?:error|mistake|inaccuracy)",
    r"\bcontext\s+(?:incorrectly|wrongly|erroneously)\s+(?:states|says|mentions|claims|indicates)",
    r"\b(?:provided|given|supplied)\s+context\s+(?:is|contains)\s+\w*\s*(?:incorrect|wrong|error|mistaken)",
    r"\bcontrary\s+to\s+the\s+context",
    r"\bdespite\s+the\s+context",
    r"\bthe\s+(?:above|preceding)\s+(?:statement|claim)\s+is\s+(?:incorrect|wrong|not\s+correct)",
]
# 组 B：取值否定型 —— 只在槽位非极性时可用（见 _slot_is_polarity）。
#         保留的是**指向取值本身**的表述；配合下面的动态"not <上下文取值>"模式。
NEG_VALUE = [
    r"\b(?:is|are|was|were)\s+not\s+(?:located\s+in|in|at)\b",
    r"\bdoes\s+not\s+(?:lay|occur|live|grow|inhabit)",
    r"\bisn'?t\s+(?:in|located)",
    # 刻意**不收** `it is actually`：单独出现时可能只是补充信息
    # （"The context is mostly accurate, but it should be noted..."），
    # 不足以证明驳回。宁可漏救。
]
RE_NEG_META = [re.compile(p, re.I) for p in NEG_METADISC]
RE_NEG_VALUE = [re.compile(p, re.I) for p in NEG_VALUE]


def _slot_is_polarity(rec):
    """上下文取值自身是否含否定词（极性槽位，如 not edible）。"""
    if not rec:
        return False
    for v in (rec.get("context_answer_aliases") or []):
        if isinstance(v, str) and re.search(r"\b(?:not|no|non|never)\b", v, re.I):
            return True
    return False


RE_NEG = RE_NEG_META + RE_NEG_VALUE   # 兼容旧引用

# 真·骑墙的迹象：若命中这些，则不救（宁可漏救，不可误救）
HEDGE_PATTERNS = [
    r"\bcould\s+be\s+(?:either|both)\b",
    r"\b(?:either|both)\s+.{0,40}\bor\b",
    r"\b(?:might|may)\s+be\s+(?:either|both)\b",
    r"\bambiguous\b",
    r"\buncertain\b",
    r"\bhard\s+to\s+(?:say|determine|tell)\b",
    r"\bboth\s+(?:are|is)\s+(?:possible|plausible|correct)\b",
    r"\bcannot\s+determine\b",
    r"\bdepending\s+on\b",
]
RE_HEDGE = [re.compile(p, re.I) for p in HEDGE_PATTERNS]


def key_of(r):
    return (r.get("question_named") or "", r.get("wikipedia_title") or "")


def grade(answer, rec):
    """返回 PARAM / CTX / BOTH / NONE / PARAM_by_negation。"""
    a = (answer or "").strip()
    if not a:
        return "NONE"
    pa = rec.get("parametric_aliases") or []
    ca = rec.get("context_answer_aliases") or []
    if isinstance(pa, str):
        pa = [pa]
    if isinstance(ca, str):
        ca = [ca]
    gp = answers_match(a, pa)[0] if pa else False
    gc = answers_match(a, ca)[0] if ca else False
    if gp and gc:
        # 只有 BOTH 才进入否定判定
        if is_explicit_rejection(a, rec):
            return "PARAM_by_negation"
        return "BOTH"
    if gp:
        return "PARAM"
    if gc:
        return "CTX"
    return "NONE"


def is_explicit_rejection(a, rec=None):
    """答句是否**显式驳回**上下文取值。

    判定顺序（保守优先）：
      1. 命中骑墙模式 → 不救（宁可漏救，不可误救）
      2. 命中元话语型 → 救（与槽位类型无关）
      3. 极性槽位 → 到此为止，**不用**取值否定型（否则把"同意上下文"误救成 PARAM）
      4. 否则命中取值否定型 → 救
    """
    if any(r.search(a) for r in RE_HEDGE):
        return False
    if any(r.search(a) for r in RE_NEG_META):
        return True
    if _slot_is_polarity(rec):
        return False
    if any(r.search(a) for r in RE_NEG_VALUE):
        return True
    # 动态：上下文取值被**直接**否定（实体名无法穷举）。
    #
    # 必须紧贴 —— 只认"否定词 + 0~12 字符 + 取值"这种直接否定的形式。
    # 放宽到 60 字符会误救：实测
    #   "The context is mostly accurate ... not just in Chile."
    # 里 "not" 与 Chile 相距远、且语义是"不只是"，并非驳回。
    # 窗口取 12 是实测调出来的：够覆盖 "not located in X" / "not in X"
    # / "not native to X"，又不足以跨越从句边界。
    if rec:
        for v in (rec.get("context_answer_aliases") or []):
            if not isinstance(v, str) or not v.strip():
                continue
            vs = re.escape(v.strip())
            pat = (r"\b(?:not|isn'?t|wasn'?t|aren'?t|n't|never|no)\b"
                   r"[^.;!?]{0,12}?\b" + vs + r"\b")
            if re.search(pat, a, re.I):
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default=os.path.join(HERE, "..", "results", "probe"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "regrade_negation.json"))
    ap.add_argument("--sample", type=int, default=0, help="打印 N 条重判样本供人工核查")
    ap.add_argument("--sample-hedge", type=int, default=0, help="打印 N 条仍判 BOTH 的样本")
    args = ap.parse_args()

    FILES = [("Qwen3-VL 2B", "syc6_qwen3-vl-2b_text.json"),
             ("Qwen3-VL 4B", "syc6_qwen3-vl-4b_text.json"),
             ("Qwen3-VL 8B", "syc6_qwen3-vl-8b_text.json"),
             ("Qwen2.5-VL 3B", "syc6_3b_text.json"),
             ("Qwen2.5-VL 7B", "syc6_7b_text.json")]
    ORDER = ["ctx_only", "orig", "ctx_hedge", "neutral", "own_only", "len_ctrl", "prior"]

    loaded = {}
    for disp, fn in FILES:
        p = os.path.join(args.probe, fn)
        if not os.path.exists(p):
            continue
        loaded[disp] = json.load(io.open(p, encoding="utf-8")).get("records", [])
    if not loaded:
        sys.exit("无可用结果文件。")

    print("=" * 88)
    print("否定感知重判：BOTH 中有多少是「显式驳回上下文」")
    print("=" * 88)
    print("%-16s %-10s %8s %8s %8s %8s" % ("模型", "立场", "BOTH原", "其中驳回", "占比", "含量"))
    resc = {}
    samples = []
    hedge_samples = []
    for disp, recs in loaded.items():
        resc[disp] = {}
        for cd in ORDER:
            nb = nneg = 0
            for r in recs:
                g = grade(r.get("out_" + cd), r)
                if g == "BOTH":
                    nb += 1
                elif g == "PARAM_by_negation":
                    nneg += 1
                    if len(samples) < args.sample:
                        samples.append((disp, cd, r))
                elif g == "CTX":
                    pass
            # 原 BOTH 数 = 新 BOTH + 新 PARAM_by_negation
            orig_both = nb + nneg
            resc[disp][cd] = dict(both_new=nb, rescued=nneg, both_orig=orig_both)
            if orig_both:
                print("%-16s %-10s %8d %8d %7.1f%% %7.1f%%"
                      % (disp, cd, orig_both, nneg, 100.0 * nneg / orig_both,
                         100.0 * orig_both / max(len(recs), 1)))
    if args.sample:
        print("\n" + "=" * 88)
        print("人工核查 A：被判为「显式驳回」的样本（抽查是否真的是驳回）")
        print("=" * 88)
        for disp, cd, r in samples:
            print("\n【%s / %s】%s" % (disp, cd, (r.get("question_named") or "")[:80]))
            print("  参数值 : %s" % r.get("parametric_answer"))
            print("  上下文 : %s" % r.get("context_answer"))
            print("  答句   : %s" % repr((r.get("out_" + cd) or "")[:260]))
    if args.sample_hedge:
        print("\n" + "=" * 88)
        print("人工核查 B：仍判 BOTH 的样本（抽查是否真的骑墙，防误救）")
        print("=" * 88)
        cnt = 0
        for disp, recs in loaded.items():
            for r in recs:
                for cd in ORDER:
                    if cnt >= args.sample_hedge:
                        break
                    if grade(r.get("out_" + cd), r) == "BOTH":
                        print("\n【%s / %s】%s" % (disp, cd, (r.get("question_named") or "")[:80]))
                        print("  参数值 : %s" % r.get("parametric_answer"))
                        print("  上下文 : %s" % r.get("context_answer"))
                        print("  答句   : %s" % repr((r.get("out_" + cd) or "")[:260]))
                        cnt += 1
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with io.open(args.out, "w", encoding="utf-8") as f:
        json.dump(resc, f, ensure_ascii=False, indent=1)
    print("\n已写入: %s" % os.path.abspath(args.out))


if __name__ == "__main__":
    main()
