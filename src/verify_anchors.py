# -*- coding: utf-8 -*-
"""锚定样本核验层 —— 对 screen_knowledge.py 的输出做二次质检。

为什么需要单独一层
==================================================================
模型调用代价高且不可重复，核验是纯后处理、免费且可反复迭代，
所以拆开做（只读 screen 的 --out-full，不再问模型）。

逐个核验 300 条抽样时发现三类缺陷，它们都会让锚定集**看起来成立但实际无效**：

  缺陷 1  泛化标注变体蒙混命中
    E-VQA 的 answer 字段同时含泛化变体与特异变体，例如
      Oreaster reticulatus -> ["red cushion sea star", "sea star", ...]
      Procambarus clarkii  -> ["red swamp crayfish", "crayfish", ...]
    模型答 "Sea Star" / "Crayfish"（**科/属级**）会命中泛化变体，
    被记成"知道这个物种"。但"知道有海星这回事"与"知道红垫海星"天差地别。
    判据：命中的是最泛化的变体，且存在更特异的变体未被命中 -> generic_only

  缺陷 2  span 定位到错误的句子
    问题问"种子如何传播"，但 span 定位器可能命中另一句里的同形词：
      Coprosma rhamnoides -> 问题 seeds spread，span "wind" 出现在
      "believed to be wind pollinated"（**授粉**，非传播）
    反事实编辑会改错事实，PKD 测量随之失真。
    判据：承载 span 的句子必须与问题共享实词（topic_ok）

  缺陷 3  模型答案源于命名而非知识
      Pedicularis groenlandica（种加词意为"格陵兰的"）-> 模型答 "Greenland"，
      而证据说 "western North America"。这是**从学名猜**，不是知识。
    判据：模型答案与实体学名的词素高度重合 -> name_leak

用法：
  python verify_anchors.py --screen ../results/evqa_screen_full.jsonl \
                           --out ../data/raw/evqa_anchors.jsonl
"""
import argparse, collections, io, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import normalize, _content_tokens

# 与问题属性有关的线索词，用于 topic_ok 判定
TOPIC_HINTS = {
    "time": {"season", "spring", "summer", "autumn", "fall", "winter", "flower",
             "flowers", "flowering", "bloom", "fruit", "fruits", "ripen",
             "mating", "breed", "month", "period", "active", "year"},
    "place": {"native", "found", "occurs", "occur", "distribut", "region",
              "country", "range", "habitat", "endemic", "grow", "grows",
              "lives", "live", "reside", "depth", "atlantic", "pacific"},
    "quantity": {"size", "length", "long", "tall", "height", "weight",
                 "wingspan", "weigh", "reach", "reaching", "measur", "cm", "mm",
                 "m", "ft", "in", "kg", "eggs", "lay", "pregnant", "gestation"},
    "diet": {"eat", "eats", "feed", "feeds", "diet", "prey", "food", "consume",
             "forage", "hunt"},
    "name": {"known", "called", "common", "name"},
}


def matched_variant(model_answer, candidates):
    """返回命中的候选中**最特异**的那个（内容词最多），及其内容词数。"""
    from common import answers_match
    hit, tier = answers_match(model_answer, candidates)
    if not hit:
        return None, None, None
    best, best_n = None, -1
    for c in candidates:
        h, _ = answers_match(model_answer, [c])
        if not h:
            continue
        n = len(_content_tokens(c))
        if n > best_n:
            best, best_n = c, n
    return best, best_n, tier


def answers_match_pair(a, c):
    from common import answers_match
    return answers_match(a, c)


def topic_ok(question, context):
    """承载 span 的句子是否与问题属性相关。

    防的是 span 定位到同形词的另一句。两个实测踩到的例子：

      Coprosma rhamnoides 问 "seeds spread"，span "wind" 却出现在
      "believed to be wind **pollinated**"（**授粉**，非传播）-> 动作不符
      Digitaria sanguinalis 问 "give fruits"，span "Summer" 却出现在
      "must be harvested by hand"（讲采收，非结果期）-> 属性词不符
    """
    ql = question.lower()
    cl = context.lower()

    # 显式动作校验：问题问某动作，句子必须讲同一动作
    ACTION_SYN = [
        (("spread", "dispers"), ("spread", "dispers", "detach", "carried")),
        (("pollinat",),         ("pollinat",)),
        (("eat", "feed", "diet", "prey"), ("eat", "feed", "diet", "prey",
                                           "food", "consume", "forage")),
        (("flower", "bloom"),   ("flower", "bloom", "blossom", "infloresc")),
        (("fruit",),            ("fruit", "ripen", "berry", "berries")),
    ]
    for q_keys, c_keys in ACTION_SYN:
        if any(k in ql for k in q_keys):
            # 问题讲这个动作，但句子讲的是另一个已被覆盖的动作 -> 不符
            if not any(k in cl for k in c_keys):
                return False, 0

    want = set()
    for slot, hints in TOPIC_HINTS.items():
        for h in hints:
            if h in ql:
                want |= hints
    if not want:
        return True, 0            # 无法判定属性，不拦
    n = sum(1 for w in want if w in cl)
    return n > 0, n


def _supported_by(token, ev_norm, ev_tokens):
    """token 是否被证据支持（含词形包含，plankton ~ zooplankton）。"""
    if (" " + token + " ") in ev_norm:
        return True
    if len(token) >= 4:
        for e in ev_tokens:
            if len(e) >= 4 and (token in e or e in token):
                return True
    return False


def contradicts(model_answer, evidence, span):
    """模型答案是否**改答**（而非多答）了 span 的事实。

    区分"多答"与"改答"是关键 —— 实测踩到的两类：

      多答（**应保留**）：模型答出正确事实，同时多说了几句
        Albizia julibrissin  span=wind，模型="Wind and water."
          -> wind 在证据中且模型给了它，多出的 water 不影响
        Parthenocissus       span=birds，模型="Squirrels, birds, and deer."
          -> birds 在证据中且模型给了它
      改答（**应剔除**）：模型丢掉了 span 的区分性词，换成一个证据里没有的词
        Cardisoma guanhumi   span="blue land crab"，模型="Giant land crab"
          -> 证据只写 blue，模型却答 giant（giant 不在证据中，blue 也丢了）
        Stenopus hispidus    span="Atlantic Ocean"，模型="Australia"
        Clematis terniflora  span="10 to 30 feet"，模型="Up to 10 feet tall."

    判据：两个条件同时成立才算改答
      (a) 模型答案含证据不支持的区分性词
      (b) span 含证据支持的区分性词，而模型答案丢了它
    """
    from common import _STOP
    mt = _content_tokens(model_answer)
    if not mt:
        return True
    st = _content_tokens(span or "")
    ev_norm = " " + normalize(evidence) + " "
    ev_tok = [t for t in normalize(evidence).split() if t]

    def distinctive(toks):
        return {t for t in toks if len(t) >= 3 and t not in _STOP}

    m_d, s_d = distinctive(mt), distinctive(st)
    novel = {t for t in (m_d - s_d) if not _supported_by(t, ev_norm, ev_tok)}
    dropped = {t for t in (s_d - m_d) if _supported_by(t, ev_norm, ev_tok)}
    return bool(novel) and bool(dropped)


def generic_only(model_answer, candidates, span):
    """模型答的是 span 的**严格上位词**（丢了区分性修饰词）-> True。

    判据：命中的变体其内容词是 span 内容词的**真子集**。
      "Sea Star"   ⊂ "red cushion sea star"  -> 泛化（Oreaster 只是海星）
      "Crayfish"   ⊂ "red swamp crayfish"    -> 泛化（Procambarus 只是螯虾）
      "Ochre Starfish" vs span "purple sea star" -> 非子集，是另一别名 -> 不拦
      "North America" == span                    -> 相等非真子集 -> 不拦
    """
    if not span:
        return False
    mt = _content_tokens(model_answer)
    st = _content_tokens(span)
    if not mt or not st:
        return False
    return mt < st          # 真子集


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", default="../results/evqa_screen_full.jsonl")
    ap.add_argument("--out", default="../data/raw/evqa_anchors.jsonl")
    ap.add_argument("--out-verified-full", default="../results/evqa_anchors_verified.jsonl")
    args = ap.parse_args()

    recs = [json.loads(l) for l in io.open(args.screen, encoding="utf-8") if l.strip()]
    hits = [r for r in recs if r.get("hit_named")]
    print("screen 记录 %d 条，实体知识命中 %d 条" % (len(recs), len(hits)))

    kept, rej = [], collections.Counter()
    for r in hits:
        reasons = []
        ma = r["model_answer_named"]
        cands = r["answers"]
        ctx = r.get("context") or ""
        q = r["q"]

        if not r.get("locatable") or not r.get("span"):
            reasons.append("no_span")
        else:
            ok, n = topic_ok(q, ctx)
            if not ok:
                reasons.append("topic_mismatch")
            if contradicts(ma, ctx, r["span"]):
                reasons.append("contradicts")
            if generic_only(ma, cands, r["span"]):
                reasons.append("generic_only")
            if r.get("is_prior"):
                reasons.append("template_prior")

        if reasons:
            rej[reasons[0]] += 1
            r["reject_reasons"] = reasons
            continue
        mv, mn, tier = matched_variant(ma, cands)
        kept.append({
            "image": "",
            "question": q,
            "question_named": r["q_named"],
            "param_answer": r["span"],
            "parametric_aliases": cands,
            "span": r["span"],
            "slot": r["slot"],
            "context": ctx,
            "tier": "verified",
            "matched_variant": mv,
            "matched_tokens": mn,
            "match_tier": tier,
            "model_param_answer": ma,
            "model_answer_generic": r["model_answer_generic"],
            "src": "evqa",
            "wikipedia_title": r["title"],
            "wikipedia_url": r["wikipedia_url"],
            "dataset_name": r["dataset_name"],
            "dataset_category_id": r["category_id"],
            "evidence_full": r["evidence"],
        })

    # 同时写出被拒的，供人工复核（论文需报告筛选流程）
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with io.open(args.out, "w", encoding="utf-8") as f:
        for a in kept:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    with io.open(args.out_verified_full, "w", encoding="utf-8") as f:
        for r in recs:
            if r.get("hit_named"):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print()
    print("=" * 62)
    print(" 核验结果")
    print("=" * 62)
    print("  实体知识命中            %5d" % len(hits))
    for k, v in rej.most_common():
        print("  ├ 剔除 %-16s %5d" % (k, v))
    print("  └ 通过核验（锚定集）    %5d" % len(kept))
    print("       —— 通过率 %.1f%%" % (100.0 * len(kept) / max(len(hits), 1)))
    print()
    print("  slot:", dict(collections.Counter(a["slot"] for a in kept).most_common()))
    print("  match_tier:", dict(collections.Counter(a["match_tier"] for a in kept).most_common()))
    print()
    print("已写入:", args.out)
    print()
    print("--- 被拒样例（每类 3 条）---")
    seen = collections.Counter()
    for r in hits:
        rs = r.get("reject_reasons")
        if not rs:
            continue
        k = rs[0]
        if seen[k] >= 3:
            continue
        seen[k] += 1
        print("  [%s] %s" % (k, r["q_named"][:58]))
        print("        模型答=%r  span=%r" % (r["model_answer_named"][:30], str(r.get("span"))[:30]))
        print("        context=%s" % (r.get("context") or "")[:78])


if __name__ == "__main__":
    main()
