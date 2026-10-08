# -*- coding: utf-8 -*-
"""E-VQA 知识筛选 —— 构造 MedusaBench 的锚定样本集。

为什么不能直接用原始问法筛选（本脚本存在的理由）
==================================================================
E-VQA 的 8133 条候选里，**100% 的问题不含实体名**，一律用
"this plant / this animal / this insect" 指代 —— 实体身份**只能从图像识别**。

于是在**无图**条件下问 "In which season does this plant give flowers?"，
模型根本不知道在问哪个物种，它答 "Spring" 靠的是"花大多春天开"的语言先验。
实测（3B, n=120）：

    泛化问法 "this plant"   旧判定器 0.8%  →  新判定器 5.8%   ← 模板先验
    实体名问法 "Coprosma"   新判定器 7.5%                    ← 实体知识

5.8% 与 7.5% 接近，说明**绝大多数"命中"是模板先验而非实体知识**。
若拿 5.8% 去筛锚定集，筛出来的是"语言先验强的样本"，不是"模型知道这个实体"。
由此得到的 knowledge rate 不具可解释性。

正确做法（本脚本）：
  1. 把实体名写进问题 -> 测"模型是否**拥有**该事实"（知识持有，与图无关）
  2. 用 answers_match 做长答案双向匹配（E-VQA 标注中位数 4 词、最长 116 词，
     旧 contains_any 方向相反，会把 "Spring" vs "late winter to spring" 判错）
  3. 与模板先验众数比对，分 strong / weak 两档
     strong = 实体特异知识；weak = 与模板先验混淆（留作对照，不废弃）

PKD 实验阶段再配图像（图像提供实体身份）；筛选阶段不配图，
避免把"识别能力"混进"知识持有"的测量。

用法：
  python screen_knowledge.py --in ../data/raw/evqa_candidates.jsonl --workers 16
"""
import argparse, collections, io, json, math, os, random, re, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Client, answers_match, normalize

SYS = "You are a knowledge assistant. Answer with a short phrase."

# 问题里的实体指代词 —— 替换成实体名（物种名 / 地标名）。
#
# ★ landmarks 子集的指代词是**实体类别词**（this museum / this square ...），
#   不是 "this place"。早先只加了 this place/area/site，导致 75% 的 landmarks
#   样本走了 entity_question 的兜底分支，问句变成
#     "In which city is this museum located? (species: Wolverhampton Art Gallery)"
#   —— 把博物馆称作 "species"，随 benchmark 发布极不体面，且 "this museum"
#   仍是纯指代。这里按 landmarks 实际出现的类别词逐个列出（来源：对
#   evqa 候选做 "this \w+" 频次统计，覆盖全部 ≥5 次的类别词）。
ANAPHORA = ("this plant", "this animal", "this insect", "this bird",
            "this reptile", "this fish", "this fungi", "this fungus",
            "this mammal", "this amphibian", "this species",
            # --- landmarks 类别词（按频次降序）---
            "this square", "this museum", "this park", "this stadium",
            "this palace", "this mosque", "this bridge", "this island",
            "this hotel", "this reservoir", "this castle", "this lake",
            "this house", "this mountain", "this lighthouse", "this church",
            "this dam", "this monastery", "this cathedral", "this temple",
            "this monument", "this building", "this theatre", "this theater",
            "this library", "this tower", "this fort", "this ruin",
            "this arch", "this garden", "this cemetery", "this station",
            "this place", "this area", "this site", "this landmark",
            "this structure", "this settlement")

SLOT_PATTERNS = [
    (r"\b(season|when|what year|which year|period|month|mating)\b", "time"),
    (r"\b(country|region|part of the world|where|native|habitat|reside|depth)\b", "place"),
    (r"\bhow (big|large|tall|high|long|heavy|old|many|much|wide)\b", "quantity"),
    (r"\b(size|weight|wingspan|length|height|pregnant|lay)\b", "quantity"),
    (r"\b(eat|feed|diet|prey)\b", "diet"),
    (r"\b(common name|called)\b", "name"),
    (r"\b(confused|protect|predator)\b", "other"),
]


def classify_slot(q):
    ql = q.lower()
    for pat, slot in SLOT_PATTERNS:
        if re.search(pat, ql):
            return slot
    return "other"


def entity_question(q, title):
    """把指代词换成物种名。若问题本就不含指代词，则把物种名附在末尾。"""
    low = q.lower()
    for w in ANAPHORA:
        i = low.find(w)
        if i >= 0:
            return q[:i] + title + q[i + len(w):]
    return "%s (species: %s)" % (q, title)


def parse_answers(field):
    out = []
    for chunk in (field or "").split("&&"):
        for a in chunk.split("|"):
            a = a.strip()
            if a:
                out.append(a)
    return list(dict.fromkeys(out))


def split_evidence(ev):
    return [x.strip() for x in (ev or "").split("|") if x.strip()]


def locate_span(evidence_parts, answers, model_answer=None):
    """在证据中定位承载答案的 span，返回 (span, 承载它的整句)。

    **必须优先取模型实际命中的那个标注变体**，而不是最短变体。
    反例（实测踩到）：
      Aurelia aurita 标注 = ['jellyfish, moon jellyfish, ...', 'common jellyfish', ...]
      模型答 "Moon jellyfish"（**正确**），但按"取最短变体"会选中
      'common jellyfish'，于是 span 与模型答案对不上，
      后续反事实编辑改的是另一个事实，PKD 测量整体失真。

    取用顺序：
      1. 模型命中的变体里，能在证据中找到的、最特异的那个
      2. 退化为：任一能在证据中找到的变体，取最短（最可能逐字出现）
    要求 |span| >= 2，避开 "m"、"1" 这类单字符的偶然匹配。
    """
    from common import answers_match

    def anchored(a):
        """该变体在证据中的定位，返回 (a, sent) 或 None。

        **必须卡词边界**。朴素 find() 会让短答案命中长单词内部，
        产生语义颠倒的锚定样本（实测踩到的最危险的一类错误）：
          span='edible' 命中 "considered **inedible** by mushroomers"
          -> 记录的"参数答案"与证据事实**完全相反**，
             后续 PKD 测量会把模型答对记成答错。
        """
        if not a or len(a) < 2:
            return None
        pat = r"(?<![A-Za-z0-9])" + re.escape(a) + r"(?![A-Za-z0-9])"
        for ev in evidence_parts:
            m = re.search(pat, ev, flags=re.IGNORECASE)
            if not m:
                continue
            idx = m.start()
            start = 0
            for sep in (". ", "; ", "|"):
                p = ev.rfind(sep, 0, idx)
                if p > start - 1:
                    start = p + len(sep)
            end = len(ev)
            for sep in (".", ";", "|"):
                p = ev.find(sep, idx + len(a))
                if p >= 0:
                    end = min(end, p + 1)
            sent = ev[start:end].strip().strip("|").strip()
            if len(sent) < 15:
                sent = ev
            return a, sent
        return None

    if model_answer:
        matched = [a for a in answers if answers_match(model_answer, [a])[0]]
        # 最特异优先（内容词最多）
        for a in sorted(set(matched), key=lambda s: -len(_ct(s))):
            r = anchored(a)
            if r:
                return r
    for a in sorted(set(answers), key=len):
        r = anchored(a)
        if r:
            return r
    return None, None


def _ct(s):
    from common import _content_tokens
    return _content_tokens(s)


def template_prior(rows):
    """每个模板 + 槽位下的答案分布，用于算"盲猜先验"。

    按 (question, slot) 分组，因为同一问法在不同类群上答案分布不同。
    """
    d = collections.defaultdict(collections.Counter)
    for r in rows:
        for a in parse_answers(r["answer"]):
            d[(r["question"], classify_slot(r["question"]))][normalize(a)] += 1
    return d


def prior_prob(question, slot, model_answer, dist):
    """模型答案在模板先验分布下的概率（越低越说明来自实体知识）。

    用 leave-one-out 之外的口径：直接取该模板下这个答案的出现频率。
    返回 (是否等于众数, 众数, 该答案先验概率, 众数概率)。
    """
    c = dist.get((question, slot))
    if not c:
        return False, None, 0.0, 0.0
    tot = float(sum(c.values()))
    top, topn = max(c.items(), key=lambda kv: kv[1])
    h = normalize(model_answer)
    # 模型可能给多词答案，取其中命中候选的部分计频
    p = c.get(h, 0) / tot
    return (h == top), top, p, topn / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="../data/raw/evqa_anchors.jsonl")
    ap.add_argument("--out-full", default="../results/evqa_screen_full.jsonl")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen2.5-vl-3b")
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()

    rows = [json.loads(l) for l in io.open(args.inp, encoding="utf-8") if l.strip()]
    if args.n:
        rows = rows[:args.n]
    print("候选 %d 条 | 模型 %s | 并发 %d" % (len(rows), args.model, args.workers))

    dist = template_prior(rows)
    c = Client(base_url=args.base_url, api_key="EMPTY",
               model=args.model, support_vision=False)

    def work(irow):
        i, r = irow
        answers = parse_answers(r["answer"])
        slot = classify_slot(r["question"])
        q_named = entity_question(r["question"], r["wikipedia_title"])

        # 两个问法都问，用于分离"模板先验"与"实体知识"
        outs = {}
        for tag, qq in (("generic", r["question"]), ("named", q_named)):
            try:
                a, _ = c.chat("Question: %s\nAnswer briefly:" % qq,
                              system=SYS, max_tokens=args.max_tokens)
            except Exception as e:
                a = "[ERR] " + str(e)[:60]
            hit, tier = answers_match(a, answers)
            outs[tag] = (a, hit, tier)

        a_named, hit_named, tier_named = outs["named"]
        a_gen, hit_gen, _ = outs["generic"]

        span, sent = locate_span(split_evidence(r["evidence"]), answers,
                                 model_answer=a_named) if hit_named else (None, None)
        is_prior, top, p_ans, p_top = prior_prob(
            r["question"], slot, a_named, dist) if hit_named else (False, None, 0.0, 0.0)

        return i, dict(
            idx=i, title=r["wikipedia_title"], q=r["question"], q_named=q_named,
            slot=slot,
            model_answer_named=a_named, hit_named=hit_named, tier=tier_named,
            model_answer_generic=a_gen, hit_generic=hit_gen,
            answers=answers,
            locatable=bool(span), span=span, context=sent,
            is_prior=is_prior, prior_top=top,
            prior_p=round(p_ans, 4), prior_top_p=round(p_top, 4),
            category_id=r.get("dataset_category_id"),
            dataset_name=r.get("dataset_name"),
            wikipedia_url=r.get("wikipedia_url"),
            evidence=r["evidence"],
        )

    t0 = time.time()
    full = [None] * len(rows)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, rec in ex.map(work, list(enumerate(rows))):
            full[i] = rec
            done += 1
            if done % 500 == 0 or done == len(rows):
                nh = sum(1 for x in full if x and x["hit_named"])
                print("  %d/%d  实体知识命中 %d (%.2f%%)  用时 %.1f 分"
                      % (done, len(rows), nh, 100.0 * nh / done,
                         (time.time() - t0) / 60))

    hits = [x for x in full if x["hit_named"]]
    loc = [x for x in hits if x["locatable"]]
    strong = [x for x in loc if not x["is_prior"]]
    weak = [x for x in loc if x["is_prior"]]
    noloc = [x for x in hits if not x["locatable"]]

    anchors = [{
        "image": "",
        "question": x["q"],
        "question_named": x["q_named"],
        "param_answer": x["span"],
        "parametric_aliases": x["answers"],
        "span": x["span"],
        "slot": x["slot"],
        "context": x["context"],
        "tier": "strong",
        "model_param_answer": x["model_answer_named"],
        "match_tier": x["tier"],
        "prior_top": x["prior_top"],
        "prior_top_p": x["prior_top_p"],
        "src": "evqa",
        "wikipedia_title": x["title"],
        "wikipedia_url": x["wikipedia_url"],
        "dataset_name": x["dataset_name"],
        "dataset_category_id": x["category_id"],
        "evidence_full": x["evidence"],
    } for x in strong]
    random.Random(0).shuffle(anchors)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with io.open(args.out, "w", encoding="utf-8") as f:
        for a in anchors:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_full)), exist_ok=True)
    with io.open(args.out_full, "w", encoding="utf-8") as f:
        for x in full:
            if x:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    n = float(len(full) or 1)
    print()
    print("=" * 62)
    print(" 筛选漏斗（模型 %s）" % args.model)
    print("=" * 62)
    print("  候选总数                        %5d" % len(full))
    print("  ├ 泛化问法命中(模板先验上界)    %5d  (%.2f%%)"
          % (sum(1 for x in full if x["hit_generic"]),
             100.0 * sum(1 for x in full if x["hit_generic"]) / n))
    print("  └ 实体名问法命中(实体知识)      %5d  (%.2f%%)"
          % (len(hits), 100.0 * len(hits) / n))
    print("     ├ 证据中无法定位 span（剔除）%4d" % len(noloc))
    print("     └ 可定位                     %4d" % len(loc))
    print("        ├ 与模板先验众数重合(weak)%4d" % len(weak))
    print("        └ 实体特异        (strong)%4d   <= 锚定样本" % len(strong))
    print()
    print("  锚定集 %d 条。所有规模都在这一套固定集合上测。" % len(anchors))
    print("  slot:", dict(collections.Counter(a["slot"] for a in anchors).most_common()))
    print("  判定档:", dict(collections.Counter(a["match_tier"] for a in anchors).most_common()))
    print()
    print("  另存 weak %d 条作对照（见 --out-full 里 is_prior=true）。" % len(weak))
    print("已写入:", args.out)


if __name__ == "__main__":
    main()
