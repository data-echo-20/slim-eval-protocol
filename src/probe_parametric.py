# -*- coding: utf-8 -*-
"""判定性验证：A-OKVQA 上模型的参数知识是否存在。

整个课题测的是「参数知识支配」。若模型对某类问题**本来就没有参数先验**，
就不存在被"支配"的东西，样本对该研究无效。

本脚本不看图、不给检索上下文，纯靠参数知识回答，对照 direct_answers 计分：

  准确率高 -> 该问题可由参数知识回答，可用于 PKD 实验
  准确率低 -> 纯视觉问题，参数知识不存在，须剔除

用法：
  python probe_parametric.py --n 200 --out ../results/probe_param.json
  python probe_parametric.py --n 200 --with-image   # 加视觉对照
"""
import argparse, io, json, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Client, image_to_b64, contains_any, write_jsonl

import pyarrow.parquet as pq

SYS = ("You are a visual question answering system. "
       "Answer with a short phrase.")

PROMPT = "Question: {q}\nAnswer briefly:"
PROMPT_VIS = "Question: {q}\nAnswer briefly based on the image:"


def parse_direct_answers(v):
    """A-OKVQA 的 direct_answers 是**字符串**，内容是 Python 列表的字面量。

    HF 数据集卡标注 dtype: string 是准确的——它是被字符串化的列表，
    形如 "['cigarette', 'cigarette', ...]"（10 个人工标注答案）。
    直接当列表用会被逐字符迭代，候选退化成 ['[', "'"]，
    命中率完全失真。必须 literal_eval 还原。
    """
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    if not v:
        return []
    try:
        import ast
        p = ast.literal_eval(v)
        return [str(x) for x in p] if isinstance(p, (list, tuple)) else [str(p)]
    except Exception:
        return [str(v)]


def build_candidates(row):
    """合并人工答案与正确选项，作为可接受的答案集合。

    兼容两种 schema：
      A-OKVQA (HF): direct_answers 是**字符串化**的列表 + choices/correct_choice_idx
      OK-VQA   (HF): answers 是**正规列表**，无选项
    """
    if "answers" in row and isinstance(row["answers"], (list, tuple)):
        cands = [str(x) for x in row["answers"]]
    else:
        cands = parse_direct_answers(row.get("direct_answers"))
    choices = row.get("choices")
    idx = row.get("correct_choice_idx")
    if choices and isinstance(idx, int) and 0 <= idx < len(choices):
        cands.append(choices[idx])
    # 去重但保序
    seen, out = set(), []
    for c in cands:
        k = c.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(c.strip())
    return out


def score(answer, candidates):
    """答案是否命中任一候选。"""
    return contains_any(answer, candidates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="../data/raw/aokvqa_val.parquet")
    ap.add_argument("--n", type=int, default=200, help="抽样条数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen2.5-vl-3b")
    ap.add_argument("--out", default="../results/probe_param.json")
    ap.add_argument("--with-image", action="store_true",
                    help="加视觉对照（会慢很多）")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--limit-mm", type=int, default=1)
    args = ap.parse_args()

    t = pq.read_table(args.parquet)
    n = t.num_rows
    rng = random.Random(args.seed)
    idx = sorted(rng.sample(range(n), min(args.n, n)))
    print("数据集 %d 条，抽样 %d 条" % (n, len(idx)))

    c = Client(base_url=args.base_url, api_key="EMPTY", model=args.model,
               support_vision=args.with_image)

    recs = []
    t0 = time.time()
    for k, i in enumerate(idx):
        row = {col: t.column(col)[i].as_py() for col in t.column_names}
        q = row["question"]
        # 候选：人工答案 + 正确选项（两者都是可接受的表述）
        cands = build_candidates(row)

        img = None
        if args.with_image:
            b = row["image"]["bytes"]
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                f.write(b)
                img = image_to_b64(f.name)
                os.unlink(f.name)
            prompt = PROMPT_VIS.format(q=q)
        else:
            prompt = PROMPT.format(q=q)

        try:
            ans, raw = c.chat(prompt, image_b64=img, system=SYS,
                              max_tokens=args.max_tokens)
            fr = raw["choices"][0].get("finish_reason")
        except Exception as e:
            ans, fr = "[ERROR] " + str(e)[:100], "error"

        hit = score(ans, cands)
        recs.append({
            "question_id": str(row.get("question_id")),
            "question": q,
            "answer": ans,
            "hit": hit,
            "truncated": fr == "length",
            "candidates": cands[:6],
            "rationales": row.get("rationales"),
            "question_type": row.get("question_type"),
            "answer_type": row.get("answer_type"),
        })
        if (k + 1) % 20 == 0 or k + 1 == len(idx):
            nh = sum(1 for r in recs if r["hit"])
            print("  %d/%d  命中 %d (%.1f%%)  用时 %.1f 分"
                  % (k + 1, len(idx), nh, 100.0 * nh / (k + 1),
                     (time.time() - t0) / 60))

    usable = [r for r in recs if not r["truncated"]]
    nu = len(usable) or 1
    hits = sum(1 for r in usable if r["hit"])
    summary = {
        "model": args.model,
        "with_image": args.with_image,
        "parquet": args.parquet,
        "n_sampled": len(recs),
        "n_usable": len(usable),
        "drop_rate": round(1 - len(usable) / float(len(recs) or 1), 4),
        "param_hit_rate": round(hits / float(nu), 4),
        "minutes": round((time.time() - t0) / 60, 2),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"summary": summary, "records": recs},
              io.open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print("\n=== 结果 ===")
    for k, v in summary.items():
        print("  %-16s %s" % (k, v))
    print()
    print("判读：")
    print("  param_hit_rate 高 -> 模型确有参数知识，样本可用于 PKD 实验")
    print("  param_hit_rate 低 -> 纯视觉题，参数知识不存在，须剔除或换数据")
    print("已写入:", args.out)


if __name__ == "__main__":
    main()
