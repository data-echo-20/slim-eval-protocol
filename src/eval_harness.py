# -*- coding: utf-8 -*-
"""**评测框架复现**：用两篇已发表工作的 prompt 跑本文同一批样本。

要解决的问题
============
知识冲突文献在"规模效应的方向"上自相矛盾，且两个方向都已有主：

  * Xie et al. (ICLR 2024, arXiv:2305.13300)  报告**大模型更坚持参数知识**
    （更抵抗冲突上下文）—— 即规模效应为正。
  * ConflictBank (arXiv:2408.12076) §3.2    报告**大模型更易被冲突知识说服**
    （MR 随规模下降）—— 即规模效应为负。

两者用的是**同族的 Qwen/Llama 规模序列**、**同一类反事实冲突样本**，
却得到相反的坐标符号。通常的解释是"任务/数据/指标不同，不可比" ——
这个解释无法证伪，因为两篇工作没有共享任何输入。

本脚本把它们的 **prompt 模板逐字复刻**（含 system prompt 与选项格式），
喂**同一批 927 条样本**、**同一对模型**、**同一解码参数**。
于是数据侧全部条件相同，唯一差异是评测框架本身。
若规模效应的符号随之翻转，则"框架不同不可比"这一说法的反面被直接演示：
**框架本身就是那个决定符号的变量。**

四个条件（全部为原文逐字模板，出处见每条的注释）：

  cb_default   ConflictBank 无证据基线
               prompt.py: "According to your knowledge, choose the best choice..."
  cb_conflict  ConflictBank 冲突证据条件
               prompt.py: "According to the evidence provided and your knowledge..."
  xie_implicit Xie et al. 隐式模式（有证据 + "and your knowledge"）
               prompt_preparation.py:355，dynamic_prompt = " and your knowledge"
  xie_explicit Xie et al. 显式模式（同模板，去掉 "and your knowledge"）
               prompt_preparation.py，dynamic_prompt = ""

三处刻意的偏离，必须写进论文，否则不构成复现：

  (1) **选项构造**。原文是四选一（ConflictBank）／三选一（Xie et al.，
      固定含 "Uncertain."）。本文样本的候选为
      A=参数真值、B=冲突注入值、C=无法确定的占位。
      ConflictBank 的 D 选项填入原文样本里的干扰项；本文数据无此字段，
      故省略 D —— 即本文跑的是**三选一**版本的 ConflictBank 模板。
      Xie et al. 的选项行**逐字保留**（含 "Uncertain." 的句点）。
  (2) **证据字段**。ConflictBank 的 Evidence 填入**冲突证据**（本文的
      context_conflict），因为该条件即为冲突条件；其 correct_evidence
      对应本文的 context_original，此处不使用。
  (3) **答案抽取**。ConflictBank 原文按 first-token logprob 在
      [" A"," B"," C"," D"] 上取 argmax。本文用生成式接口，
      改为生成后字符串匹配（见 rescore_harness.py 的三档判据），
      并在论文中明示该差异为**抽取方式差异**，非模板差异。

用法：
  python eval_harness.py --data data/medusa_combined.jsonl \
      --out results/probe/harness_3b_text.json --no-vision \
      --cond all --model qwen2.5-vl-3b --workers 16
"""
import argparse, io, json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Client, read_jsonl, image_to_b64


def fmt_options(cands, letters=("A", "B", "C")):
    return "\n".join("%s: %s." % (l, c) for l, c in zip(letters, cands))


def cb_prompt(q, ctx, param, conflict, no_evidence=False, swap=False):
    """ConflictBank 模板（逐字）。

    原文（prompt.py 第 15 行，无证据）：
        "According to your knowledge, choose the best choice from the "
        "following options."
    拼接：prompt_text + '\n\nQuestion: {q}\nA. {o0}\nB. {o1}\nC. {o2}\n
          D. {o3}\nAnswer:'
    原文（第 23 行，有证据）：
        "According to the evidence provided and your knowledge, choose the "
        "best choice from the following options."
    拼接：prompt_text + '\n\nEvidence: {ev}\nQuestion: {q}\nA. ...\nAnswer:'
    原文无 system prompt（inference.py 直接把整串喂 vLLM）。
    """
    # swap=True 时把参数真值放 B、冲突值放 A —— 用于分离"位置偏好"与"内容偏好"。
    # 不分离的后果：四个条件里 A 恒为参数真值，若模型有选 A 的位置偏好，
    # 该偏好会与"抵抗冲突上下文"混为一谈（实测 P(A)≈0.49 逼近随机，须排除）。
    a_val, b_val = (conflict, param) if swap else (param, conflict)
    opts = "\n".join("%s. %s" % (l, c)
                     for l, c in zip("ABC", (a_val, b_val, "Uncertain")))
    if no_evidence:
        head = ("According to your knowledge, choose the best choice from "
                "the following options.")
        return head + "\n\nQuestion: %s\n%s\nAnswer:" % (q, opts)
    head = ("According to the evidence provided and your knowledge, choose "
            "the best choice from the following options.")
    return (head + "\n\nEvidence: %s\nQuestion: %s\n%s\nAnswer:"
            % (ctx, q, opts))


def xie_prompt(q, ctx, param, conflict, implicit=True, swap=False):
    """Xie et al. 模板（逐字）。

    原文（prompt_preparation.py 第 355 行）：
        prompt_text = ('''According to the given information''' + dynamic_prompt
                       + ''', choose the best choice from the following options.
        \\n\\nInformation: ''' + evidence + "\\n\\nQuestion: " + question
                       + "\\n\\n" + option + "\\n\\nAnswer: ")
      其中第 257/303/326/337/350 行：dynamic_prompt = " and your knowledge"
      （implicit 模式）；explicit 模式下 dynamic_prompt = ""。
      option = "Options:\\nA: " + first_pred + "\\nB: " + contrary_claim
               + "\\nC: Uncertain."
    system prompt（run.py 第 26 行）："You are a helpful assistant."
    """
    dyn = " and your knowledge" if implicit else ""
    a_val, b_val = (conflict, param) if swap else (param, conflict)
    option = ("Options:\nA: %s\nB: %s\nC: Uncertain."
              % (a_val, b_val))
    return ("According to the given information" + dyn
            + ", choose the best choice from the following options."
            + "\n\nInformation: " + ctx
            + "\n\nQuestion: " + q
            + "\n\n" + option + "\n\nAnswer: ")


CONDITIONS = {
    "cb_default": dict(system="", kind="cb", no_evidence=True),
    "cb_conflict": dict(system="", kind="cb", no_evidence=False),
    "xie_implicit": dict(system="You are a helpful assistant.",
                         kind="xie", implicit=True),
    "xie_explicit": dict(system="You are a helpful assistant.",
                         kind="xie", implicit=False),
}


def build(cd, rec, swap=False):
    """按条件构造 prompt，返回 (user, system)。

    候选一律为 参数真值 / 冲突值 / 无法确定。
    ConflictBank 原文无 system prompt，此处传空串（Client 端不发送 system 字段）。
    swap 见 --randomize-order：交换 A/B 的内容以分离位置与内容偏好。
    """
    spec = CONDITIONS[cd]
    q = rec.get("question_named") or rec.get("question") or ""
    ctx = rec.get("context_conflict") or ""
    param = rec.get("parametric_answer") or ""
    conflict = rec.get("context_answer") or ""
    if spec["kind"] == "cb":
        return (cb_prompt(q, ctx, param, conflict,
                          no_evidence=spec["no_evidence"], swap=swap),
                spec["system"])
    if spec["kind"] == "xie":
        return (xie_prompt(q, ctx, param, conflict,
                           implicit=spec["implicit"], swap=swap),
                spec["system"])
    raise ValueError(cd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url",
                    default=os.environ.get("VLLM_BASE_URL",
                                           "http://127.0.0.1:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("VLM_API_KEY", "EMPTY"))
    ap.add_argument("--model", default=os.environ.get("VLM_MODEL", "qwen2.5-vl-3b"))
    ap.add_argument("--cond", nargs="+", default=["all"])
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--question-field", choices=("named", "generic"),
                    default="named")
    ap.add_argument("--order", choices=("ab", "random", "ba"), default="ab",
                    help="选项顺序：ab=参数真值恒为A（原复现跑法）；"
                         "random=按样本确定性随机；ba=全部交换。"
                         "用于分离位置偏好与内容偏好。")
    args = ap.parse_args()

    conds = list(CONDITIONS) if "all" in args.cond else args.cond
    for cd in conds:
        if cd not in CONDITIONS:
            sys.exit("未知条件: %s（可选 %s）" % (cd, list(CONDITIONS)))

    rows = read_jsonl(args.data)
    if args.limit:
        rows = rows[:args.limit]
    c = Client(base_url=args.base_url, api_key=args.api_key, model=args.model,
               support_vision=not args.no_vision)
    try:
        with urllib.request.urlopen(
                args.base_url.rstrip("/") + "/models", timeout=15) as r:
            served = [m["id"] for m in json.load(r).get("data", [])]
    except Exception as e:
        sys.exit("预检失败：%s" % e)
    if args.model not in served:
        sys.exit("预检失败：--model=%s 不在服务端 %s 中。" % (args.model, served))
    print("预检通过：服务端 %s，%d 条 × %d 条件" % (served, len(rows), len(conds)),
          flush=True)

    def work(i_r):
        i, r = i_r
        if not (r.get("context_conflict") or "").strip():
            return None
        img = None
        if not args.no_vision and os.path.exists(r.get("image", "")):
            img = image_to_b64(r["image"])
        # 顺序：按样本索引确定性决定，可复现；random 时奇偶样本交替
        if args.order == "random":
            swap = (i % 2 == 1)
        else:
            swap = (args.order == "ba")
        out = {}
        for cd in conds:
            user, system = build(cd, r, swap=swap)
            out["swap_" + cd] = bool(swap)
            txt, _ = c.chat(user, image_b64=img, system=system,
                            max_tokens=args.max_tokens)
            out["out_" + cd] = txt
        rec = dict(r)
        rec.update(out)
        rec["usable"] = all((rec.get("out_" + cd) or "").strip()
                            for cd in conds)
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
    json.dump({"meta": {"conds": conds, "data": args.data, "model": args.model,
                        "no_vision": args.no_vision,
                        "question_field": args.question_field,
                        "order": args.order,
                        "note": "两篇已发表工作的逐字 prompt 模板复现",
                        "deviations": ["D 选项省略（本文样本无干扰项字段）",
                                       "证据字段用 context_conflict",
                                       "答案抽取改为字符串匹配"],
                        "calls": c.calls},
               "records": recs},
              io.open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
    print("已写入 %s（%d 条，%d 次调用，%.1fs）"
          % (args.out, len(recs), c.calls, time.time() - t0))


if __name__ == "__main__":
    main()
