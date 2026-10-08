# -*- coding: utf-8 -*-
"""离线重打分：把"别名子串匹配"换成"证据接地（grounding）匹配"。

为什么必须做（这是本项目第二个测量偏倚）
==================================================================
eval_pkd 用 answers_match（别名表子串匹配）判定模型是否跟随参数知识/上下文。
别名表是**粗粒度**的，而模型越大给出的答案**越细**，于是：

  Q: In which country is Poinsett Bridge located?   参数=United States  冲突=Madagascar
  7B 答: "South Carolina"      ← 比"美国"更细（州），语义上仍是美国，却匹配不上
  Q: In which city is Aston Hall located?            参数=Birmingham      冲突=Pokhara
  7B 答: "Aston, England"      ← 在英国，却匹配不上 Birmingham

实测 landmarks 的 "neither" 回答里 70%(3B)/81%(7B) 被证据支持 —— 是**真事实，
只是粒度不同**。这个偏倚**系统性惩罚答得更细的大模型**。

修复思路（不依赖模型、可复现）
==================================================================
用 E-VQA 自带的 evidence 段落做**接地判定**：
  - 模型答案若被"参数答案所在的那句/那段证据"支持 -> grounded in param side
  - 模型答案若被"冲突后被编辑过的那句上下文"支持 -> grounded in ctx side
  - 支持判据：答案的内容词在证据里出现，且**不受否定词反转**
    （"not venomous" 不应判成 venomous）

为什么**不用**"答案是否在冲突上下文里"作唯一判据：那样会退化成子串匹配。
关键是把两边都拿**原始证据**（未被编辑的 Wikipedia 段落）来接地 ——
原始证据代表**真实世界的实体事实**，冲突后的上下文代表**被改动的事实**。

用法：
  python rescore_grounded.py --pkd ../results/pkd_7b_comb_text.json \
                             --anchors ../data/raw/evqa_anchors_combined.jsonl \
                             --out ../results/rescore_7b_comb_text.json
"""
import argparse, io, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import normalize, _content_tokens, _STOP

# 否定词：出现在答案词前若干词内 -> 该词的极性可能被反转
#
# ★ 只收**独立的否定副词**，绝不收 in/un/non 这类前缀。
#   踩过的坑：把 "in" 放进 \b(...)\b 会匹配介词 "in" ——
#     "ripening in the summer" 里的 "in" 被判成否定语境，
#     于是答案词 "summer" 被跳过，70% 的样本误判为 undecided。
#   前缀否定（nonvenomous / inedible / unthreatened）不需要在这里处理：
#   它们与肯定式是**不同的词**，词边界匹配天然不会让 "venomous" 命中
#   "nonvenomous"，所以前缀已经自动排除了。
_NEG = re.compile(r"\b(not|n't|never|without|no longer|isn't|aren't|wasn't)\b", re.I)


def _neg_window(text, tok, win=24):
    """tok 前面 win 个字符内是否有独立否定词（只在句子范围内找）。

    窗口取 24 字符而非 40：否定词通常紧贴被否定的词（"not venomous"、
    "never found in"），窗口过大容易把**上一个从句**的否定误套到本词上。
    """
    tl = text.lower()
    for m in re.finditer(r"(?<![a-z0-9])" + re.escape(tok) + r"(?![a-z0-9])", tl):
        head = tl[max(0, m.start() - win):m.start()]
        # 遇到句读就截断：否定不跨句
        for sep in (". ", "; ", ", "):
            p = head.rfind(sep)
            if p >= 0:
                head = head[p + len(sep):]
        if _NEG.search(head):
            return True
    return False


def _tokens(s):
    return {t for t in normalize(s or "").split() if len(t) >= 3 and t not in _STOP}


def _negated(text, tok):
    """tok 在 text 中是否处于否定语境（见 _neg_window）。"""
    return _neg_window(normalize(text), tok)


def supported(answer, evidence):
    """answer 是否被 evidence 支持（词级接地，卡词边界，避否定反转）。

    返回 (是否支持, 命中的词数, 答案的词数)。

    判据：答案的**内容词**多数能在证据中找到，且不是靠否定式命中。
      "South Carolina" 在 "…bridge in South Carolina, United States…" -> 2/2 ✓
      "venomous"        在 "…is a nonvenomous snake…"                -> 否定命中 ✗
    """
    at = _tokens(answer)
    if not at:
        return False, 0, 0
    ev = normalize(evidence or "")
    hit = 0
    for t in at:
        pat = r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])"
        if re.search(pat, ev):
            if _negated(evidence, t):
                continue
            hit += 1
        elif len(t) >= 4:
            # 词形包含（plankton ~ zooplankton / Carolina ~ Carolinas）
            if any(len(e) >= 4 and (t in e or e in t)
                   for e in ev.split()):
                hit += 1
    return hit >= max(1, (len(at) + 1) // 2), hit, len(at)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkd", required=True, nargs="+",
                    help="一个或多个 eval_pkd 输出（可一次重打分多个规模）")
    ap.add_argument("--anchors", default="../data/raw/evqa_anchors_combined.jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    anc = {}
    for l in io.open(args.anchors, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            anc[(r.get("question_named") or "", r.get("wikipedia_title") or "")] = r

    for path in args.pkd:
        d = json.load(io.open(path, encoding="utf-8"))
        recs = d["records"]
        for r in recs:
            a = anc.get((r.get("question_named") or "",
                         r.get("wikipedia_title") or ""), {})
            ev = a.get("evidence_full") or ""
            out_cfl = r.get("out_cfl") or ""

            # 接地判定：答案被原始证据支持 -> 站在"参数知识一侧"
            g_param, hp, tp = supported(out_cfl, ev)
            # 被冲突后的上下文支持 -> 站在"检索一侧"
            g_ctx, hc, tc = supported(out_cfl, r.get("context_conflict") or "")

            r["ground_param"] = g_param
            r["ground_ctx"] = g_ctx
            r["ground_hit_param"] = hp
            r["ground_n_tok"] = tp

        # 只在有效冲突样本上统计
        u = [r for r in recs if r.get("usable")]
        c = [r for r in u if (r.get("context_conflict") or "").strip()]
        n = float(len(c) or 1)
        n_usable = float(len(u) or 1)

        pkd_old = sum(1 for r in c if r["cfl_follows_param"]) / n
        pkd_new = sum(1 for r in c if r.get("ground_param")) / n
        ctx_new = sum(1 for r in c if r.get("ground_ctx")) / n
        # 救回：旧口径判 neither、新口径判定到某一边
        neither_old = [r for r in c
                       if not r["cfl_follows_param"] and not r["cfl_follows_ctx"]]
        rescue_p = sum(1 for r in neither_old if r.get("ground_param"))
        rescue_c = sum(1 for r in neither_old if r.get("ground_ctx"))
        # 救回后仍无法判定
        undecided = sum(1 for r in c
                        if not r.get("ground_param") and not r.get("ground_ctx"))

        print("=" * 74)
        print(" %s" % os.path.basename(path))
        print("=" * 74)
        print("  n_conflict=%d  n_usable=%d" % (len(c), len(u)))
        print("  PKD  旧(别名匹配) %.4f  ->  新(证据接地) %.4f   (%+.4f)"
              % (pkd_old, pkd_new, pkd_new - pkd_old))
        print("  ctx  新(证据接地) %.4f" % ctx_new)
        print("  旧 neither %d 条中: 救回参数侧 %d, 救回上下文侧 %d, 仍无法判定 %d"
              % (len(neither_old), rescue_p, rescue_c, undecided))
        print("  know 基线 %.4f" % (sum(1 for r in u if r["noc_follows_param"]) / n_usable))
        print()

        if args.out:
            outp = args.out if len(args.pkd) == 1 else \
                os.path.join(os.path.dirname(args.out),
                             os.path.basename(path).replace(".json", "_grounded.json"))
            d["summary"]["PKD_rate_alias"] = round(pkd_old, 4)
            d["summary"]["PKD_rate_grounded"] = round(pkd_new, 4)
            d["summary"]["ctx_follow_grounded"] = round(ctx_new, 4)
            d["summary"]["n_undecided"] = undecided
            json.dump(d, io.open(outp, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            print("已写入:", outp)


if __name__ == "__main__":
    main()
