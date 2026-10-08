# -*- coding: utf-8 -*-
"""Phase 0-B：PKD-rate 评测。

在三种条件下测量模型输出，计算参数知识支配率。

  NO-CTX    : 无检索上下文           -> 基线，测参数知识
  CONSIST   : 检索与参数知识一致      -> 对照，无法区分是否用了检索
  CONFLICT  : 检索被反事实编辑        -> 关键条件

  PKD-rate = |CONFLICT下输出跟随参数知识| / |CONFLICT有效样本|

同时记录：
  - acc_param  : 输出等于参数知识的比例
  - acc_ctx    : 输出等于编辑后事实的比例（= 遵从检索）
  - 两者应近似互补

用法：
  python eval_pkd.py --data ../data/medusa_demo.jsonl --out ../results/pkd_demo.json
"""
import argparse, io, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (Client, read_jsonl, image_to_b64, contains_any,
                    answers_match, write_jsonl)


SYS = ("You are a visual question answering system. "
       "Answer with a short phrase, using the provided context when relevant.")

# NO-CTX 条件**不能**用上面的 SYS。
# 那份 system prompt 明确要求"使用提供的上下文"，而 NO-CTX 根本没有上下文，
# 等于在暗示模型不要依靠自身知识，会系统性压低知识基线。
# 实测（3B, 236 条锚定集）：
#   用 CTX 版 SYS 时 noctx_param_rate = 0.496
#   改成中性 SYS 后             = 0.992（锚定集本应按最小规模筛选得到 ~1）
# 前者会让 PKD 的分母里混进"被 prompt 劝退"的样本，指标失去意义。
SYS_NOCTX = "You are a knowledge assistant. Answer with a short phrase."

PROMPT_NOCTX = "Question: {q}\nAnswer briefly:"
PROMPT_CTX = ("Context:\n{ctx}\n\nQuestion: {q}\n"
              "Answer briefly based on the context and the image:")


def pick_question(r, field):
    """选问法。

    named   : 把物种名写进问题（锚定集的 question_named）。**PKD 实验该用这个。**
              原因：锚定集是用"实体名问法"筛的（模型确实拥有该事实），
              但 E-VQA 的原始问法用 "this plant" 指代，实体只能靠图像识别。
              3B 对长尾 iNat 物种的识别率很低，若用泛化问法，
              NO-CTX 条件答错的原因是**没认出来**，而不是"没有参数知识" ——
              于是 PKD 会被"识别失败"混淆，测出的下降与假设无关。
              写进物种名可把"感知"和"知识支配"分离开。
    generic : 原始问法（"this plant"）。用于测量感知缺口，
              即"看见却认不出/认出了却不用"的三角冲突分析。
    """
    if field == "named":
        return r.get("question_named") or r.get("question")
    return r.get("question")


def build_client(args):
    return Client(base_url=args.base_url, api_key=args.api_key,
                  model=args.model, support_vision=not args.no_vision)


def ask(c, prompt, img=None, max_tokens=2048, system=None):
    """返回 (答案, 是否被截断)。

    注意：推理型模型（deepseek-flash 等）的 reasoning_tokens 会先消耗预算，
    max_tokens 给小了会导致正文为空且 finish_reason=length。故这里给足余量，
    并把截断显式暴露出来，避免把"没答完"误判成"没跟随参数知识"。
    """
    try:
        txt, raw = c.chat(prompt, image_b64=img, system=system or SYS,
                          max_tokens=max_tokens)
        fr = raw["choices"][0].get("finish_reason")
        return txt, (fr == "length")
    except Exception as e:
        return "[ERROR] " + str(e)[:120], False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default=os.environ.get("VLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("VLM_API_KEY", "EMPTY"))
    ap.add_argument("--model", default=os.environ.get("VLM_MODEL", "qwen2.5-vl"))
    ap.add_argument("--no-vision", action="store_true", help="纯文本模式（无视觉模型）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="单次生成上限；推理型模型需给足（reasoning_tokens 会先占预算）")
    ap.add_argument("--question-field", choices=("named", "generic"), default="named",
                    help="named=把物种名写进问题（PKD 实验用）；generic=原始泛化问法")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--match", choices=("auto", "strict"), default="auto",
                    help="auto=长答案用 answers_match；strict=只用 contains_any")
    args = ap.parse_args()

    rows = read_jsonl(args.data)
    if args.limit:
        rows = rows[:args.limit]

    # 判定器：E-VQA 的标注是中位数 4 词、最长 116 词的长描述短语，
    # contains_any 的方向（"候选必须是模型答案的子串"）在此会系统性判错。
    def match(answer, cands):
        if args.match == "strict":
            return contains_any(answer, cands)
        return answers_match(answer, cands)[0]

    c = build_client(args)

    def work(i_r):
        i, r = i_r
        img = None
        if not args.no_vision and os.path.exists(r.get("image", "")):
            img = image_to_b64(r["image"])

        q = pick_question(r, args.question_field)
        # NO-CTX 用中性 system：见 SYS_NOCTX 注释（否则知识基线被系统性压低）
        o_noc, t_noc = ask(c, PROMPT_NOCTX.format(q=q), img, args.max_tokens,
                           system=SYS_NOCTX)
        o_cns, t_cns = ask(c, PROMPT_CTX.format(ctx=r["context_original"], q=q), img, args.max_tokens)
        o_cfl, t_cfl = ask(c, PROMPT_CTX.format(ctx=r["context_conflict"] or r["context_original"], q=q),
                           img, args.max_tokens)

        pa = r["parametric_aliases"]
        ca = r["context_answer_aliases"]

        # 截断或空响应的样本不计入统计——否则会把"没答完"错算成"没跟随"
        bad = (t_noc or t_cns or t_cfl
               or not o_noc.strip() or not o_cns.strip() or not o_cfl.strip())

        rec = dict(r)
        rec.update({
            "out_noc": o_noc, "out_cns": o_cns, "out_cfl": o_cfl,
            "truncated": bool(t_noc or t_cns or t_cfl),
            "empty": bool(not o_noc.strip() or not o_cns.strip() or not o_cfl.strip()),
            "usable": not bad,
            "noc_follows_param": match(o_noc, pa),
            "cns_follows_param": match(o_cns, pa),
            "cfl_follows_param": match(o_cfl, pa),
            "cfl_follows_ctx":   match(o_cfl, ca),
            # 知识持有基线：无上下文时是否真的答得出参数答案。
            # 这个数在各规模上应接近 1（锚定集就是按最小规模的知识筛的），
            # 若某规模上明显下降，说明筛选前提被破坏，必须先查这个数。
            "knows_param": match(o_noc, pa),
        })
        return i, rec

    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    recs = [None] * len(rows)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, rec in ex.map(work, list(enumerate(rows))):
            recs[i] = rec
            done += 1
            if done % 20 == 0 or done == len(rows):
                print("  %d/%d  用时 %.1f 分"
                      % (done, len(rows), (time.time() - t0) / 60), flush=True)

    usable = [r for r in recs if r["usable"]]
    nu = len(usable)

    # ★ CONFLICT 类指标的分母必须是**真的有冲突上下文**的样本。
    #   rewrite 失败的样本 context_conflict 为空，上面喂下去的是
    #   `context_conflict or context_original` —— 也就是**原样的、与参数知识一致的**
    #   上下文。这类样本在 CONFLICT 条件下永远不可能表现出"冲突",
    #   把它们算进分母等于往 PKD 里掺 0，会系统性压低指标。
    #   实测 236 条里有 12 条（5.1%）属于这种，且全部被算进了 n_usable。
    #   注意 noctx_param_rate **不受影响**：NO-CTX 根本不读上下文，
    #   这 12 条的知识基线测量依然是有效的，故仍用全部 usable 样本。
    conflict = [r for r in usable if (r.get("context_conflict") or "").strip()]
    nc = len(conflict)
    f = float(nc or 1)
    pkd = sum(1 for r in conflict if r["cfl_follows_param"]) / f
    ctx = sum(1 for r in conflict if r["cfl_follows_ctx"]) / f
    both = sum(1 for r in conflict if r["cfl_follows_param"] and r["cfl_follows_ctx"]) / f
    neither = sum(1 for r in conflict if not r["cfl_follows_param"]
                  and not r["cfl_follows_ctx"]) / f
    # 知识基线：分母用全部 usable（见上）
    noc_acc = sum(1 for r in usable if r["noc_follows_param"]) / float(nu or 1)

    summary = {
        "model": args.model,
        "vision": not args.no_vision,
        "n_total": len(recs),
        "n_usable": nu,
        "n_conflict": nc,
        "drop_rate": round(1 - nu / float(len(recs) or 1), 4),
        "PKD_rate": round(pkd, 4),
        "ctx_follow_rate": round(ctx, 4),
        "noctx_param_rate": round(noc_acc, 4),
        "both_rate": round(both, 4),
        "neither_rate": round(neither, 4),
        "question_field": args.question_field,
        "match": args.match,
        "calls": c.calls,
        "minutes": round((time.time() - t0) / 60, 2),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"summary": summary, "records": recs},
              io.open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("\n=== 结果 ===")
    for k, v in summary.items():
        print("  %-18s %s" % (k, v))
    print("\nPKD-rate 越高说明参数知识支配越强。")
    print("both_rate 高说明判定过宽（参数与上下文答案同时命中），需检查别名表。")
    print("已写入:", args.out)


if __name__ == "__main__":
    main()
