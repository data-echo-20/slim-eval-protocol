# -*- coding: utf-8 -*-
"""第 8 档立场 `adjudicate` 的预测检验判定（见 04_论文/第八档立场_预测检验_预注册.md）。

为什么是独立脚本而不是给 analyze_scale_ladder.py 加参数
--------------------------------------------------------
预测检验的成败取决于 `adjudicate` 的 Δ 与已有七档的 Δ **同口径**。
若另写一套打分/McNemar 代码，即使数值对，也等于本文自己踩了
本轮刚修掉的那个坑（记账与判定不同源）。故本脚本**只做数据合并与判定**，
所有统计量一律调用 analyze_scale_ladder 里的函数，不复制实现。

口径一致性断言（每次运行都查，失败即退出）
--------------------------------------------
  1. 新旧文件覆盖同一批 927 条，键集合必须完全一致
  2. 锚定集 |D*| 必须等于 649（沿用主实验；第 8 档不改变锚定协议）
  3. 已有七档的 Δ 与 scale_ladder[_negation].json **逐位相等**
     —— 这条是核心：它证明本脚本没有偷偷换口径。
     任一不成立则退出，不出结论。

用法
----
  python3 analyze_adjudicate.py --grader alias
  python3 analyze_adjudicate.py --grader negation
"""
import argparse, collections, io, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analyze_scale_ladder as L

# 预注册的预测（写定于见到任何 adjudicate 输出之前，见预注册 §三）
PRED_SIGN = "pos"                     # 两族均预测为正
PRED_RANGE = {"qwen3": (12.0, 28.0), "qwen25": (22.0, 34.0)}
# 定位参照：adjudicate 应落在 own_only 与 prior 之间
ANCHOR = {"own_only": None, "prior": None}
N_ANCHORED_EXPECTED = 649


def load_new(probe_new, tag):
    """载入远端跑出的新档文件 → {key: {out_adjudicate, out_adjudicate_len}}。"""
    fn = os.path.join(probe_new, "syc8_%s_text.json" % tag)
    if not os.path.exists(fn):
        return None, fn
    d = json.load(io.open(fn, encoding="utf-8"))
    recs = d.get("records", d)
    out = {}
    for r in recs:
        k = L.key_of(r)
        out[k] = dict(adjudicate=r.get("out_adjudicate"),
                      adjudicate_len=r.get("out_adjudicate_len"))
    return out, fn


def delta_paired(recs_a, recs_b, shared, stance):
    """同族端点对的配对 Δ 与 McNemar。**与 analyze_scale_ladder 端点块同码**。"""
    n_pair = f_ab = f_ba = pa = pb = 0
    for k in shared:
        ga = L.grade_eff(recs_a[k].get("out_" + stance), recs_a[k])
        gb = L.grade_eff(recs_b[k].get("out_" + stance), recs_b[k])
        if ga in ("BOTH", "NONE") or gb in ("BOTH", "NONE"):
            continue
        n_pair += 1
        pa += (ga == "PARAM")
        pb += (gb == "PARAM")
        if ga == "PARAM" and gb != "PARAM":
            f_ab += 1
        elif gb == "PARAM" and ga != "PARAM":
            f_ba += 1
    if n_pair == 0:
        return None
    pkd_a, pkd_b = pa / n_pair, pb / n_pair
    return dict(delta_pp=100.0 * (pkd_b - pkd_a), p=L.mcnemar_exact(f_ab, f_ba),
                n_paired=n_pair, pkd_small=pkd_a, pkd_large=pkd_b,
                flip_up=f_ab, flip_down=f_ba)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grader", choices=["alias", "negation"], default="negation",
                    help="主判据默认 negation（与正文一致）")
    ap.add_argument("--probe", default=os.path.join(HERE, "..", "results", "probe"))
    ap.add_argument("--probe-new", default=os.path.join(HERE, "..", "results_adjudicate"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    L.USE_NEGATION, L._NEG_FN = False, None
    if args.grader == "negation":
        from regrade_negation import is_explicit_rejection as fn
        L.USE_NEGATION, L._NEG_FN = True, fn
    print("判据 = %s\n" % ("negation（否定感知；显式驳回计 PARAM）" if L.USE_NEGATION
                          else "alias（纯别名）"))

    # ---- 载入已有七档 ----
    loaded, used = {}, {}
    for tag, disp, fam, nB, cands in L.LADDER:
        for fn in cands:
            p = os.path.join(args.probe, fn)
            if os.path.exists(p):
                d = L.load(p)
                if d:
                    loaded[tag] = d["recs"]
                    used[tag] = fn
                    break
        else:
            sys.exit("❌ 缺已有七档结果：%s（候选 %s）" % (disp, cands))
    print("已有七档：%d 个规模点，各 %d 条" %
          (len(loaded), len(next(iter(loaded.values())))))

    # ---- 载入新两档 ----
    new, newfiles = {}, {}
    for tag, disp, fam, nB, cands in L.LADDER:
        d, fn = load_new(args.probe_new, tag)
        if d is None:
            print("  [缺] %-16s %s" % (disp, fn))
            continue
        new[tag] = d
        newfiles[tag] = fn
    if not new:
        sys.exit("❌ 未找到任何新档结果（%s/syc8_*_text.json）。"
                 "先把 run_adjudicate.sh 的输出下载到这里。" % args.probe_new)

    # ---- 断言 1：键集合完全一致 ----
    for tag in sorted(new):
        a, b = set(loaded[tag]), set(new[tag])
        if a != b:
            sys.exit("❌ %s 的新旧键集合不一致：仅旧有 %d / 仅新有 %d。"
                     "不可并表。" % (tag, len(a - b), len(b - a)))
    print("✓ 断言1 通过：新旧覆盖同一批样本，键集合完全一致")

    # ---- 断言 2/3：复用主实验的 D*，并逐位核对七档 Δ ----
    ref_path = os.path.join(HERE, "..", "results",
                            "scale_ladder.json" if args.grader == "alias"
                            else "scale_ladder_negation.json")
    if not os.path.exists(ref_path):
        sys.exit("❌ 缺参照文件 %s —— 无参照则无法证明同口径。" % ref_path)
    ref = json.load(io.open(ref_path, encoding="utf-8"))

    # D* 从参照 JSON 取主实验的 n_anchored（不重算：本脚本不改锚定协议）
    n_star = ref["n_anchored"]
    if n_star != N_ANCHORED_EXPECTED:
        sys.exit("❌ 参照 |D*| = %d，与预期 %d 不符。中止。"
                 % (n_star, N_ANCHORED_EXPECTED))
    print("✓ 断言2 通过：沿用主实验 |D*| = %d（未重算）" % n_star)

    # 从主实验文件恢复 shared 的键集合
    kstate = {}
    for tag in loaded:
        k = L.load_k(tag, args.probe)
        if k is None:
            sys.exit("❌ 缺锚定文件（NO-CTX）：%s" % tag)
        kstate[tag] = k
    shared = sorted(i for i in set.intersection(*[set(loaded[t]) for t in loaded])
                    if all(kstate[t].get(i, False) for t in loaded))
    if len(shared) != n_star:
        sys.exit("❌ 重算 |D*| = %d，与参照 %d 不符。中止。" % (len(shared), n_star))
    print("✓ 断言3a 通过：重算 |D*| 与参照一致")

    # 把新档并入 records，构造「九档」的完整记录
    for tag in new:
        for k in new[tag]:
            loaded[tag][k]["out_adjudicate"] = new[tag][k]["adjudicate"]
            loaded[tag][k]["out_adjudicate_len"] = new[tag][k]["adjudicate_len"]

    # 断言 3b：七档 Δ 必须与主实验逐位相等
    fam_of = {t: f for t, _, f, _, _ in L.LADDER}
    nB_of = {t: b for t, _, _, b, _ in L.LADDER}
    diffs, checked = [], 0
    for fam in ("qwen3", "qwen25"):
        ft = sorted([t for t in loaded if fam_of[t] == fam], key=lambda x: nB_of[x])
        a, b = ft[0], ft[-1]
        for cd in L.ORDER:
            r = delta_paired(loaded[a], loaded[b], shared, cd)
            if r is None:
                continue
            old = None
            for sc in ref.get("symbol_check", {}).get(fam, {}).get("seq", []):
                if sc["stance"] == cd:
                    old = sc["delta_pp"]
            if old is None:
                continue
            checked += 1
            if abs(old - r["delta_pp"]) > 1e-9:
                diffs.append((fam, cd, old, r["delta_pp"]))
    if diffs:
        for fam, cd, old, new_v in diffs:
            print("  ❌ %s/%s：参照 %+.6f vs 本脚本 %+.6f" % (fam, cd, old, new_v))
        sys.exit("❌ 断言3b 失败：七档 Δ 与主实验不一致，本脚本口径有别，不出结论。")
    print("✓ 断言3b 通过：七档 %d 个 Δ 与主实验逐位相等 → 同口径" % checked)

    # ---- 出结论 ----
    disp_of = {t: d for t, d, _, _, _ in L.LADDER}

    res = {"criterion": ref.get("criterion"), "n_anchored": n_star,
           "grader": args.grader, "source_new": newfiles,
           "predictions": {"sign": PRED_SIGN, "range_pp": PRED_RANGE},
           "per_family": {}}

    print("\n" + "=" * 104)
    print("预测检验：adjudicate 的 Δ（%s，|D*|=%d）"
          % ("否定感知" if L.USE_NEGATION else "别名", n_star))
    print("=" * 104)
    print("%-8s %-20s %-24s %9s %9s %6s" %
          ("族", "臂", "规模对", "Δ(pp)", "p", "n"))
    print("-" * 70)

    signs = {}
    for fam in ("qwen3", "qwen25"):
        fams = sorted([t for t in loaded if fam_of[t] == fam], key=lambda x: nB_of[x])
        if len(fams) < 2:
            continue
        a, b = fams[0], fams[-1]
        entry = {"small": disp_of[a], "large": disp_of[b], "arms": {},
                 "anchors": {}}
        for arm in ("adjudicate", "adjudicate_len", "own_only", "prior"):
            r = delta_paired(loaded[a], loaded[b], shared, arm)
            if r is None:
                continue
            entry["arms"][arm] = r
            if arm in ("own_only", "prior"):
                entry["anchors"][arm] = r["delta_pp"]
            else:
                print("%-8s %-20s %-24s %+9.1f %9.2e %6d" %
                      (fam, arm, "%s → %s" % (disp_of[a], disp_of[b]),
                       r["delta_pp"], r["p"], r["n_paired"]))
        r = entry["arms"].get("adjudicate")
        if r:
            d = r["delta_pp"]
            signs[fam] = "pos" if d > 0 else ("neg" if d < 0 else "zero")
            lo, hi = PRED_RANGE[fam]
            entry["sign_pred_correct"] = (signs[fam] == PRED_SIGN)
            entry["in_predicted_range"] = (lo <= d <= hi)
            entry["between_anchors"] = (
                entry["anchors"].get("own_only") is not None
                and entry["anchors"].get("prior") is not None
                and min(entry["anchors"]["own_only"],
                        entry["anchors"]["prior"]) <= d
                <= max(entry["anchors"]["own_only"], entry["anchors"]["prior"]))
            # 长度臂
            rl = entry["arms"].get("adjudicate_len")
            if rl:
                same_sign = (rl["delta_pp"] > 0) == (d > 0)
                dl = rl["delta_pp"] - d
                entry["len_arm"] = dict(
                    delta_pp=rl["delta_pp"], p=rl["p"], n_paired=rl["n_paired"],
                    same_sign=bool(same_sign), shift_pp=dl)
        res["per_family"][fam] = entry

    # ---- 判定（严格按预注册 §四，不做事后调整）----
    print("\n" + "=" * 104)
    print("判定（预注册 §四）")
    print("=" * 104)
    if not signs:
        sys.exit("❌ 无有效 Δ，无法判定。")
    all_pos = all(v == "pos" for v in signs.values())
    any_neg = any(v != "pos" for v in signs.values())

    if all_pos:
        verdict = "预测成功"
    elif any_neg and len(set(signs.values())) == 1:
        verdict = "预测失败（两族均为负）"
    elif any_neg:
        verdict = "部分失败（一正一负）"
    else:
        verdict = "无法判定"

    all_in = all(e.get("in_predicted_range") for e in res["per_family"].values())
    print("  符号预测（两族均为正）：%s   实测 %s" %
          ("✓" if all_pos else "✗", signs))
    print("  量值预测（落在预测区间）：%s" % ("✓" if all_in else "✗"))
    for fam, e in res["per_family"].items():
        la = e.get("len_arm")
        if la:
            print("  长度臂 %-8s Δ=%+.1f p=%.2e n=%d，与主臂同号=%s，位移 %+.1f pp" %
                  (fam, la["delta_pp"], la["p"], la["n_paired"],
                   "是" if la["same_sign"] else "否", la["shift_pp"]))
    print("\n  >>> 判定：%s" % verdict)
    if all_pos and not all_in:
        print("  >>> 符号主张成立，量值主张失败 —— 两者分别报告（预注册 §四末行）。")
    res["verdict"] = verdict
    res["sign_all_pos"] = all_pos
    res["all_in_predicted_range"] = all_in

    out = args.out or os.path.join(HERE, "..", "results",
                                   "adjudicate_test%s.json"
                                   % ("" if args.grader == "alias" else "_negation"))
    if os.path.abspath(args.probe_new).startswith("/tmp/"):
        sys.exit("\n❌ --probe-new 指向 /tmp（试跑用的伪造数据），拒绝写出正式结果文件。\n"
                 "   真实结果须下载到 %s" % os.path.join(HERE, "..", "results_adjudicate"))
    with io.open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("\n已写入: %s" % os.path.abspath(out))


if __name__ == "__main__":
    main()
