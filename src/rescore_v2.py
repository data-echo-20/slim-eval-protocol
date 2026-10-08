# -*- coding: utf-8 -*-
"""证据接地打分器 v2：修正 v1 的**通用词部分命中**缺陷。

v1 为什么必须废弃（首版判据的方向性缺陷）
==================================================================
v1 的 supported() 用"内容词过半命中"打分，于是答案里的**通用词**会在
两侧证据里同时命中：

  Q: In which city is Amon Carter Museum... located?
  param = Fort Worth ; 冲突上下文改成 "Da Nang, Texas"
  7B 在冲突条件下答 = "Da Nang, Texas"      ← 教科书式跟随上下文
  v1 判定：_tokens → {dang, texas}；证据原句含 "Fort Worth, Texas"
           → texas 命中 → 1/2 → ground_param=True   ← **误判**

实测后果（927 条文本臂样本）：
  ground_param=True 的样本里，**同时** ground_ctx=True 的比例
      3B 45.8%  /  7B 70.7%
  打分器无法区分两侧，且失效程度**随模型增大而恶化** —— 恰好制造出
  "大模型更跟随参数知识"的假象。
  landmarks/other 格 111 条"被救回参数侧"的样本，**100% 同时被判跟随上下文**。

v2 的修正判据（三重收紧）
==================================================================
1. **剔除通用词**：city/river/country/… 在两侧证据里都出现的词不具区分度。
2. **剔除问句词**：问句里已出现的词（"city"、"country"）不具区分度。
3. **全命中而非过半**：剩余实词必须**全部**接地。
4. **双向互斥**：若同一答案在两侧都接地 -> 判为 **unresolved（弃权）**，
   而不是记给任何一侧。这是关键 —— v1 的错误正来自允许两侧同时为真。

弃权率会上升，这是**正确的代价**：判不了就不判，好过判错。
统计时以"可判定集"为分母，并**同时报告弃权率**。

用法：
  python rescore_v2.py --prefix pkd_3b_comb_text --anchors ../data/raw/evqa_anchors_combined.jsonl
  python rescore_v2.py --all --anchors ../data/raw/evqa_anchors_combined.jsonl --out ../results/v2
"""
import argparse, io, json, os, re, sys, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 通用词：在实体描述里高频出现，两侧证据都可能有 -> 不具区分度
GENERIC = set("""
city town village river lake island country state province region county district
park museum bridge dam fort tower castle church cathedral temple mosque palace
garden library station square stadium hotel house mountain lighthouse monastery
cemetery ruins ruin arch theatre theater building structure settlement area site
place landmark north south east west northern southern eastern western
united states kingdom republic nation island island
""".split())

# 否定词：只收独立否定副词（v1 的教训：绝不可收 in/un/non 这类前缀，
#   \bin\b 会匹配介词 "in"，曾导致 70% 样本误判 undecided）
_NEG = re.compile(r"\b(not|n't|never|without|no longer|isn't|aren't|wasn't)\b", re.I)


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def toks(s):
    return [t for t in norm(s).split() if len(t) >= 3]


def _neg_window(text, tok, win=24):
    tl = norm(text)
    for m in re.finditer(r"(?<![a-z0-9])" + re.escape(tok) + r"(?![a-z0-9])", tl):
        head = tl[max(0, m.start() - win):m.start()]
        for sep in (". ", "; ", ", "):
            p = head.rfind(sep)
            if p >= 0:
                head = head[p + len(sep):]
        if _NEG.search(head):
            return True
    return False


def _present(tok, ev_tokens_set, ev_norm):
    """tok 是否在证据里以**非否定**方式出现（含词形包含）。"""
    if re.search(r"(?<![a-z0-9])" + re.escape(tok) + r"(?![a-z0-9])", ev_norm):
        return not _neg_window(ev_norm, tok)
    if len(tok) >= 4:
        for e in ev_tokens_set:
            if len(e) >= 4 and (tok in e or e in tok):
                if not _neg_window(ev_norm, e):
                    return True
    return False


def distinctive_ground(answer, evidence, question):
    """答案是否**排他地**接地于该证据。

    return True  / False  / None(弃权：答案无区分性实词)
    """
    at = toks(answer)
    if not at:
        return None
    qt = set(toks(question))
    keep = [t for t in at if t not in GENERIC and t not in qt]
    if not keep:
        # 全是通用词/问句词 -> 没有可用的判别信号，弃权
        return None
    ev = norm(evidence or "")
    evs = set(ev.split())
    if not evs:
        return False
    return all(_present(t, evs, ev) for t in keep)


def ground_fullset(answer, evidence):
    """**全词**接地判据（不剔通用词），用于评估判别器覆盖，不作主判据。

    动机：v2 的 `distinctive_ground` 通过剔除通用词获得区分力，代价是
    对"River"/"City"这类通用答案弃权。需量化**弃权里有多少本可判定**，
    否则"弃权率高"这一代价是否可接受就无从判断。

    与 v1（部分命中）的关键差别：这里要求**全部内容词**接地，因此
    不会产生 v1 那种"两侧同时命中"的假象。
    """
    at = toks(answer)
    if not at:
        return False
    ev = norm(evidence or "")
    evs = set(ev.split())
    if not evs:
        return False
    return all(_present(t, evs, ev) for t in at)


def grade(rec, anchor):
    """返回 (verdict, params)，verdict ∈ {PARAM, CTX, BOTH, NONE}。"""
    ans = rec.get("out_cfl") or ""
    q = anchor.get("question_named") or anchor.get("question") or ""
    gp = distinctive_ground(ans, anchor.get("evidence_full") or "", q)
    gc = distinctive_ground(ans, rec.get("context_conflict") or "", q)
    if gp is None and gc is None:
        return "NONE", (gp, gc)
    if gp and not gc:
        return "PARAM", (gp, gc)
    if gc and not gp:
        return "CTX", (gp, gc)
    if gp and gc:
        return "BOTH", (gp, gc)      # 两侧都接地 -> 无法归属，弃权
    return "NONE", (gp, gc)          # 两侧都不接地


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", nargs="+", default=None,
                    help="结果文件前缀，如 pkd_3b_comb_text（可多个）")
    ap.add_argument("--all", action="store_true", help="处理全部 4 个 comb 文件")
    ap.add_argument("--results", default="../results")
    ap.add_argument("--anchors", default="../data/raw/evqa_anchors_combined.jsonl")
    ap.add_argument("--out", default="../results/v2")
    args = ap.parse_args()

    prefixes = args.prefix or []
    if args.all:
        prefixes = ["pkd_3b_comb_text", "pkd_7b_comb_text",
                    "pkd_3b_comb_vision", "pkd_7b_comb_vision"]

    anc = {}
    for l in io.open(args.anchors, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            anc[(r.get("question_named") or "", r.get("wikipedia_title") or "")] = r

    out = {"meta": {"grader": "v2-distinctive-exclusive", "anchors": args.anchors},
           "per_file": {}}

    for pref in prefixes:
        p = os.path.join(args.results, pref + ".json")
        if not os.path.exists(p):
            print("MISSING", p); continue
        d = json.load(io.open(p, encoding="utf-8"))
        recs = d["records"]

        c = [r for r in recs if r.get("usable")
             and (r.get("context_conflict") or "").strip()]
        v = collections.Counter()
        for r in c:
            a = anc.get((r.get("question_named") or "",
                         r.get("wikipedia_title") or ""), {})
            verdict, (gp, gc) = grade(r, a)
            r["v2_verdict"] = verdict
            r["v2_ground_param"], r["v2_ground_ctx"] = gp, gc
            # 覆盖评估：若放宽到"全词接地"本可判定多少（仅用于量化弃权代价）
            if verdict == "NONE":
                fp = ground_fullset(r.get("out_cfl") or "",
                                    a.get("evidence_full") or "")
                fc = ground_fullset(r.get("out_cfl") or "",
                                    r.get("context_conflict") or "")
                r["v2_cover_param"], r["v2_cover_ctx"] = fp, fc
                if fp and not fc:
                    r["v2_cover_verdict"] = "PARAM"
                elif fc and not fp:
                    r["v2_cover_verdict"] = "CTX"
                elif fp and fc:
                    r["v2_cover_verdict"] = "BOTH"
                else:
                    r["v2_cover_verdict"] = "NONE"
            v[verdict] += 1

        # 弃权代价：NONE 中有多少在"全词接地"口径下本可判定
        cov = collections.Counter()
        for r in c:
            cv = r.get("v2_cover_verdict")
            if r.get("v2_verdict") == "NONE" and cv is not None:
                cov[cv] += 1

        n = len(c)
        dec = n - v["NONE"] - v["BOTH"]
        line = ("%-24s n=%-5d  PARAM=%-4d CTX=%-4d BOTH=%-4d NONE=%-4d "
                "| 弃权率=%.1f%%  可判定=%-4d  PKD(dec)=%s"
                % (pref, n, v["PARAM"], v["CTX"], v["BOTH"], v["NONE"],
                   100.0 * (v["NONE"] + v["BOTH"]) / max(n, 1), dec,
                   ("%.4f" % (v["PARAM"] / dec)) if dec else "n/a"))
        print(line)

        out["per_file"][pref] = dict(
            n=n, param=v["PARAM"], ctx=v["CTX"], both=v["BOTH"],
            none=v["NONE"], decidable=dec,
            abstain_rate=(v["NONE"] + v["BOTH"]) / max(n, 1),
            pkd_decidable=(v["PARAM"] / dec) if dec else None,
            pkd_all=(v["PARAM"] / n) if n else None,
            cover_from_none=dict(cov),
            pkd_widen=(v["PARAM"] + cov["PARAM"]) /
                      max(v["PARAM"] + v["CTX"] + cov["PARAM"] + cov["CTX"], 1))
        print("    └ 弃权覆盖评估：NONE 中本可判定 %d 条（PARAM %d / CTX %d / BOTH %d）"
              % (sum(cov.values()), cov["PARAM"], cov["CTX"], cov["BOTH"]))

        os.makedirs(args.out, exist_ok=True)
        json.dump(d, io.open(os.path.join(args.out, pref + "_v2.json"), "w",
                             encoding="utf-8"), ensure_ascii=False, indent=1)

    os.makedirs(args.out, exist_ok=True)
    json.dump(out, io.open(os.path.join(args.out, "_summary_v2.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n已写入:", args.out)


if __name__ == "__main__":
    main()
