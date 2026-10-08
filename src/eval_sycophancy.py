# -*- coding: utf-8 -*-
"""中立项实验：用来说明"跟随上下文"到底是被 prompt 指令逼出来的，还是自发的。

为什么必须做（这是 v2 头号结论的最大竞争解释）
================================================
原评测的 prompt 是**明确的指令式**：

  system : "...using the provided context when relevant."
  user   : "Context:\n{ctx}\n\nQuestion: {q}\n
            Answer briefly based on the context and the image:"

在这套 prompt 下，3B 在 landmarks/other（城市/河流识别）上对**跨洲替换**的
荒谬上下文采纳率高达 **0.959**（人工抽检逐字输出 "Ganges"/"Volga River"/"Arequipa"，
见 主发现_v2_修正版.md §3.6）。7B 更高。

**这留下一个比"参数知识支配"更平凡的解释**：
  H_instr：模型只是**在遵从"按上下文回答"的指令**；大模型更会跟指令，
          所以"PKD 随规模下降"测的是 instruction-following，不是知识冲突。
若 H_instr 成立，本文的主发现不成立。

本实验用**四种 system prompt 立场**测 CONFLICT 条件，其余全部不变
（同一批样本、同一冲突上下文、同一模型、同一 temperature=0）。
四种立场按"对参数的授权"从小到大排列：

  ctx_only   : 只准看上下文          —— 上限：与 H_instr 同族，测纯遵从
  neutral    : 两个来源并列、不分主次  —— **中立项**
  prior      : 允许已有知识优先       —— 中间档
  own_only   : 只准用自己的知识       —— 下限：测是否能压到 ~0

README 中的关键预测（本实验的判定规则）：
  * 若 neutral 下 PKD 仍显著随规模下降 → H_instr 被排除，主发现成立。
  * 若 neutral 下 PKD 塌到 ~0 且规模效应消失 → H_instr 成立，主发现作废。

用法：
  python eval_sycophancy.py --data ../data/medusa_combined.jsonl \
      --out ../results/probe/syc_3b_text.json --no-vision \
      --cond all --workers 12

  # 只跑中立项（快，用于快速判定）
  python eval_sycophancy.py --data ../data/medusa_combined.jsonl \
      --out ../results/probe/syc_neutral_3b.json --no-vision --cond neutral
"""
import argparse, io, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Client, read_jsonl, image_to_b64


# ---- 四种立场。变量只有 system 与 user 模板里的措辞，样本/上下文完全不动。----
CONDITIONS = {
    # 现行论文（eval_pkd.py）所用的原始 prompt，**逐字复制**，作为第五个立场。
    # 加它的原因：主表里"原论文指令式 prompt"一行原先取自 v2 的另一次运行
    # （results/v2/pkd_*_v2.json），再按 question_named 与 927 条求交集 ——
    # 同一个数字却来自不同批次，可比性无法保证（解码批次、服务端状态）。
    # 把原始 prompt 作为本实验的一个立场重跑，主表七行即出自同一次运行。
    # 注意 system 里的 "and the image" 在 --no-vision 下同样保留：
    # 原论文那次文本臂运行正是这个措辞，改动即不构成复现。
    "orig": dict(
        system="You are a visual question answering system. "
               "Answer with a short phrase, using the provided context "
               "when relevant.",
        user=("Context:\n{ctx}\n\nQuestion: {q}\n"
              "Answer briefly based on the context and the image:")),

    # 以下两个立场是稳健性设计（第五、第六个条件），不是核心梯度的一部分。
    #
    # ctx_hedge —— 真实 RAG 系统实际在用的措辞：给上下文，但留一个逃生口。
    #   前四个立场都不是这个。若只看前四个，梯度会被读成是在
    #   "命令用上下文"与"允许用自有知识"两个极端之间拉出来的，
    #   而真实系统用的是中间态。本条件直接测该中间态，并定位梯度过零点。
    #
    # len_ctrl —— **长度控制的安慰剂**。前四个立场的 system prompt 词数不同
    #   （ctx_only 13 词 / neutral 7 词 / prior 17 词 / own_only 14 词），
    #   梯度因此与长度共变。本条件与 prior **等长且句式同构**（17 词，
    #   "Use X when Y"），但语义中立（"as appropriate" 不授予任何优先权）。
    #   若它落在 neutral 附近而非 prior 附近 → 梯度来自语义而非长度。
    "ctx_hedge": dict(
        system="You are a question answering system. Answer using the provided "
               "context, but if you believe the context is inaccurate, answer "
               "from your own knowledge.",
        user=("Context:\n{ctx}\n\nQuestion: {q}\n"
              "Answer briefly, correcting the context if it is wrong:")),

    "len_ctrl": dict(
        system="You are a question answering system. Use the provided context "
               "and your own knowledge together, and answer as appropriate.",
        user=("Context:\n{ctx}\n\nQuestion: {q}\n"
              "Answer briefly using the context and your own knowledge "
              "as appropriate:")),

    "ctx_only": dict(
        system="You are a question answering system that relies solely on the "
               "provided context.",
        user=("Context:\n{ctx}\n\nQuestion: {q}\n"
              "Answer briefly using only the context:")),

    "neutral": dict(
        system="You are a question answering system.",
        user=("Context:\n{ctx}\n\nQuestion: {q}\nAnswer briefly:")),

    "prior": dict(
        system="You are a question answering system. Use your own knowledge "
               "when it is more reliable than the provided context.",
        user=("Context:\n{ctx}\n\nQuestion: {q}\n"
              "Answer briefly, preferring your own knowledge if it is more "
              "reliable:")),

    "own_only": dict(
        system="You are a question answering system that relies solely on "
               "your own knowledge.",
        user=("Context:\n{ctx}\n\nQuestion: {q}\n"
              "Answer briefly using only your own knowledge:")),

    # ================= 第 8 档：adjudicate（预测检验，2026-09-24 预注册）=================
    # 预注册文件：04_论文/第八档立场_预测检验_预注册.md
    # 预测：两族符号均为正，量值落在 own_only 与 prior 之间。
    #
    # 设计约束（逐条对应一个已知失败模式）：
    #   1. **不复用 ctx_hedge 的「命令驳回」路线** —— 那一格是本文量值最不可信、
    #      Qwen2.5 族符号不可主张的一格（别名判据 n=24）。走同一条路线会继承同一缺陷。
    #   2. **授权语义是新的** —— 前七档都是「直接指定」优先权；本档不指定任何一方，
    #      而是把来源优先权的判断**委托给模型**。这是「授权」与「委托判断」的差别。
    #   3. **长度作为混淆被显式控制** —— 本档 37 词，比 prior（31 词）长 6 词，
    #      故同批加跑 adjudicate_len（42 词）作长度对照。
    #
    # 逐字文本以预注册文件为准，**不得在见到数据后修改**。
    "adjudicate": dict(
        system="You are a question answering system. Decide which source is "
               "more reliable for this question, the provided context or your "
               "own knowledge, then answer from that source.",
        user=("Context:\n{ctx}\n\nQuestion: {q}\n"
              "Decide which source is more reliable, then answer from it:")),

    "adjudicate_len": dict(
        system="You are a question answering system. Decide which source is "
               "more reliable for this question, the provided context or your "
               "own knowledge, then answer from that source. Take your time "
               "to think.",
        user=("Context:\n{ctx}\n\nQuestion: {q}\n"
              "Decide which source is more reliable, then answer from it:")),
}


def ask(c, prompt, img, system, max_tokens):
    try:
        txt, raw = c.chat(prompt, image_b64=img, system=system,
                          max_tokens=max_tokens)
        return txt, (raw["choices"][0].get("finish_reason") == "length")
    except Exception as e:
        return "[ERROR] " + str(e)[:120], False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url",
                    default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("VLM_API_KEY", "EMPTY"))
    ap.add_argument("--model", default=os.environ.get("VLM_MODEL", "qwen2.5-vl-3b"))
    ap.add_argument("--cond", nargs="+", default=["neutral"],
                    help="要跑的立场，可多个；all = 全部四种")
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--question-field", choices=("named", "generic"),
                    default="named")
    ap.add_argument("--ctx-field", choices=("conflict", "original"),
                    default="conflict",
                    help="conflict=注入冲突值（主实验）；"
                         "original=上下文与参数知识一致（CONSIST 对照，"
                         "用于判定模型是'无视上下文'还是'有选择地信任上下文'）")
    args = ap.parse_args()

    conds = list(CONDITIONS) if "all" in args.cond else args.cond
    for cd in conds:
        if cd not in CONDITIONS:
            sys.exit("未知立场: %s（可选 %s）" % (cd, list(CONDITIONS)))

    rows = read_jsonl(args.data)
    if args.limit:
        rows = rows[:args.limit]
    c = Client(base_url=args.base_url, api_key=args.api_key, model=args.model,
               support_vision=not args.no_vision)

    # 预检：服务端模型名与 --model 是否一致。
    # 不检的代价是实测踩过的 —— 服务端着 7B、命令忘了传 --model，
    # 于是每次调用都 404，而 404 被 ask() 当作普通异常吞成 "[ERROR] ..." 文本，
    # 脚本一路跑到最后写出一个**全是错误串**的结果文件，静默且不报错。
    import urllib.request
    try:
        with urllib.request.urlopen(
                args.base_url.rstrip("/") + "/models", timeout=15) as r:
            served = [m["id"] for m in json.load(r).get("data", [])]
    except Exception as e:
        sys.exit("预检失败：无法访问 %s（%s）" % (args.base_url, e))
    if args.model not in served:
        sys.exit("预检失败：--model=%s 不在服务端 %s 中。"
                 "\n  vLLM 一次只服务一个模型，请在启动脚本里确认模型名，"
                 "\n  或按当前服务改用 --model %s。"
                 % (args.model, served, served[0] if served else "?"))
    print("预检通过：服务端模型 %s，调用 %d 条 × %d 立场"
          % (served, len(rows), len(conds)), flush=True)

    def work(i_r):
        i, r = i_r
        img = None
        if not args.no_vision and os.path.exists(r.get("image", "")):
            img = image_to_b64(r["image"])
        q = (r.get("question_named") if args.question_field == "named"
             else r.get("question")) or r.get("question")
        if not (r.get("context_conflict") or "").strip():
            return None            # 无冲突上下文 -> 不在本实验范围
        # CONSIST 对照时喂与参数知识一致的原文；否则喂注入后的冲突文本
        ctx = (r.get("context_original") if args.ctx_field == "original"
               else r.get("context_conflict")) or ""
        out = {}
        for cd in conds:
            spec = CONDITIONS[cd]
            txt, trunc = ask(c, spec["user"].format(
                ctx=ctx, q=q), img, spec["system"],
                args.max_tokens)
            out["out_" + cd] = txt
            out["trunc_" + cd] = bool(trunc)
        rec = dict(r)
        rec.update(out)
        rec["usable"] = all((rec.get("out_" + cd) or "").strip()
                            and not rec.get("trunc_" + cd) for cd in conds)
        return rec

    t0 = time.time()
    recs = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for n, rec in enumerate(ex.map(work, enumerate(rows)), 1):
            if rec:
                recs.append(rec)
            if n % 100 == 0:
                print("  %d/%d  %.1fs" % (n, len(rows), time.time() - t0),
                      flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"meta": {"conds": conds, "data": args.data,
                        "model": args.model, "no_vision": args.no_vision,
                        "question_field": args.question_field,
                        "ctx_field": args.ctx_field,
                        "calls": c.calls},
               "records": recs},
              io.open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
    print("已写入 %s（%d 条，%d 次调用，%.1fs）"
          % (args.out, len(recs), c.calls, time.time() - t0))


if __name__ == "__main__":
    main()
