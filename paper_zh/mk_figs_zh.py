#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成论文用图：图题用中文，坐标轴标签用中文，字号按 A4 单栏排版设定。

为什么要单独做一套
------------------
`03_实验/src/fig/make_figures.py` 画的是宽版图（figsize 宽 10~13 英寸）。
宽版在 Markdown 预览里好看，但塞进 A4 单栏（正文宽 455 pt ≈ 6.3 英寸）时
要被缩到约 50%，11 pt 的标签落到 5 pt 上下，印出来看不清。

本脚本按**最终尺寸**作图：figsize 宽 6.2 英寸（略小于正文宽），
配 9 pt 基础字号，`\includegraphics[width=\linewidth]` 时几乎不缩放，
屏幕与纸面字号一致。

数据全部读自 03_实验/results，不重新计算；图与正文数字同源。
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve():
    """定位工程根、results/ 与出图目录，两种目录布局都能跑。

    本脚本有两份副本，各自的相对位置不同：
      · 写作工程：slim/04_论文/latex/mk_figs_zh.py → 根 slim/，数据在
        03_实验/results 或 results/，出图到 04_论文/figures/。
      · GitHub 仓库：paper_zh/mk_figs_zh.py → 根 repo/，数据在 results/，
        出图到 paper_zh/figures/（仓库里 figures/ 放的是英文图，中文图
        在 paper_zh/figures/，与 paper_en/ 并列）。

    早期版本把根写死成 HERE/../..，仓库副本里会指到 repo 的上一级，
    出图落到仓库外面（甚至覆盖掉顶层的英文图）。这里按"哪份数据在旁边"
    往上找根，再按布局决定输出目录。
    """
    for up in ("..", os.path.join("..", "..")):
        proj = os.path.abspath(os.path.join(HERE, up))
        for cand in (os.path.join(proj, "03_实验", "results"),
                     os.path.join(proj, "results")):
            if not os.path.isdir(cand):
                continue
            if os.path.isdir(os.path.join(proj, "04_论文")):
                # 写作工程：04_论文/figures 与 04_论文/figures_en 并列
                return (proj, cand,
                        os.path.join(proj, "04_论文", "figures"),
                        os.path.join(proj, "04_论文", "figures_en"))
            # 仓库布局：中文图在 paper_zh/figures（脚本旁边），
            # 英文图在仓库根的 figures/（README 里 documented 的位置）。
            return (proj, cand, os.path.join(HERE, "figures"),
                    os.path.join(proj, "figures"))
    raise SystemExit("找不到 results/ 目录（本脚本需要与数据同工程）")


PROJ, RES, OUT, OUT_EN = _resolve()

# ---------------------------------------------------------------- 字体
# 跨平台中文字体：先找已装字体的文件名（Linux 路径 + Windows 路径），
# 都没有就退回按字体名找一个可用的 CJK 字体。本机（Windows）装的是
# Noto Sans SC / SimHei / 微软雅黑。
_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    r"C:\Windows\Fonts\NotoSansSC-VF.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
]
# 按名字兜底（Windows 上 addfont 对 .ttc 偶有失败，直接从已注册表里取）
_FONT_NAMES = ["Noto Sans SC", "Microsoft YaHei", "SimHei", "SimSun"]


def install_font():
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                fm.fontManager.addfont(p)
                return fm.FontProperties(fname=p).get_name()
            except Exception:
                pass
    installed = {f.name for f in fm.fontManager.ttflist}
    for n in _FONT_NAMES:
        if n in installed:
            return n
    return "DejaVu Sans"


CJK = install_font()

# ---------------------------------------------------------------- 样式
C_QWEN3 = "#1f5fa8"      # 蓝，Qwen3 族
C_QWEN25 = "#c0392b"     # 红，Qwen2.5 族
C_POS = "#2e8b57"        # 正号
C_NEG = "#b03a3a"        # 负号
C_GREY = "#8a8a8a"
C_ACC = "#e08a1e"        # 强调橙
STANCE_C = ["#3b6ea5", "#6b93c4", "#8f7a5c", "#9a9a9a",
            "#d9a05b", "#c4614e", "#7a6aa8"]

plt.rcParams.update({
    "font.sans-serif": [CJK, "DejaVu Sans"],
    "font.family": "sans-serif",
    # 字体必须按 TrueType 子集（Type 0/Identity-H）嵌入，不用 matplotlib 默认的
    # Type 3。CJK 字体（NotoSansSC）在 Type 3 下部分字形会错位：图里画出的是
    # 一串汉字，文字层却是正确的原文（抽文本查不出来，只有渲成位图才看得见）。
    # 中文图里正文是汉字，错位恰好落在 ASCII 那一部分；英文图更明显，整条
    # 轴标签会变成汉字。设成 42 后字形正确，且文字可选可搜。
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "font.size": 8.6,
    "axes.titlesize": 9.2,
    "axes.labelsize": 8.8,
    "legend.fontsize": 7.4,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linewidth": 0.5,
    "axes.linewidth": 0.7,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "lines.linewidth": 1.5,
    "lines.markersize": 3.4,
})
W = 6.2            # 单栏图宽（英寸），对应 A4 正文宽


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"))
    plt.close(fig)
    print("  ✓", name)


# ---------------------------------------------------------------- 立场定义
# 预注册的授权序，由弱到强。与正文 §4.3.2 一致。
STANCES = ["ctx_only", "orig", "ctx_hedge", "neutral",
           "own_only", "len_ctrl", "prior"]
CN = {
    "ctx_only": "仅上下文",
    "orig": "原始",
    "ctx_hedge": "条件让步",
    "neutral": "中立",
    "own_only": "仅自有",
    "len_ctrl": "长度对照",
    "prior": "自有优先",
}
FAMS = ["qwen3", "qwen25"]
# 数据里的规模 tag 形如 qwen3-vl-2b / qwen2.5-vl-3b。用 startswith(FAMS 里的
# 键) 匹配会漏掉 Qwen2.5 —— "qwen2.5-vl-3b".startswith("qwen25") 是 False
# （族名里有点号）。这里显式给出前缀，两族都能命中。
FAMPREFIX = {"qwen3": ("qwen3-",), "qwen25": ("qwen2.5-",)}
FAMLAB = {"qwen3": "Qwen3-VL（2B / 4B / 8B）",
          "qwen25": "Qwen2.5-VL（3B / 7B）"}
FAMC = {"qwen3": C_QWEN3, "qwen25": C_QWEN25}


def fam_tags(tags, fam):
    """取某一族的全部规模 tag。"""
    pre = FAMPREFIX[fam]
    return [t for t in tags if t.startswith(pre)]


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        raise SystemExit("缺数据文件：%s" % p)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _first(*cands):
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def recompute_dstar():
    """从 probe/ 原始输出重算五规模锚定集 D* 与逐规模知识率。

    为什么不直接用 `results/ladder_anchors.json` 的 k_rate：那份是早期给
    「跨代可比性」做检查时算的，**只覆盖 Qwen2.5 两个规模点**（k_rate 里
    只有 qwen2.5-vl-3b/7b 两个键）。拿它画图会漏掉 Qwen3 三个规模点，
    且图题里的 |D*| 会停在 1134（两模型交集），与正文的 649（五模型交集）
    对不上 —— 这正是 A4 版图 1 一度画错的原因。

    缺原始文件时返回 None，由调用方回退到 ladder_anchors 并显式告警，
    避免悄悄画出一张少三个规模点的图。
    """
    def key_of(r):
        return (r.get("question_named") or "", r.get("wikipedia_title") or "")

    probe_cands = [os.path.join(PROJ, "03_实验", "results", "probe"),
                   os.path.join(PROJ, "results", "probe")]
    res_cands = [os.path.join(PROJ, "03_实验", "results"),
                 os.path.join(PROJ, "results")]
    ladder = [("qwen3-vl-2b", "pkd_qwen3-vl-2b_text.json"),
              ("qwen3-vl-4b", "pkd_qwen3-vl-4b_text.json"),
              ("qwen3-vl-8b", "pkd_qwen3-vl-8b_text.json"),
              ("qwen2.5-vl-3b", "pkd_3b_comb_text.json"),
              ("qwen2.5-vl-7b", "pkd_7b_comb_text.json")]
    k = {}
    for tag, fn in ladder:
        p = _first(*[os.path.join(d, fn) for d in probe_cands + res_cands])
        if p is None:
            return None
        with open(p, encoding="utf-8") as f:
            recs = json.load(f).get("records", [])
        k[tag] = {key_of(r): bool(r.get("noc_follows_param")) for r in recs}

    # 「共同覆盖」= 任一规模点跑过的那批冲突样本，用 Qwen3-VL 2B 的冲突条件
    # 输出做基准（它就是全量跑的那一份）。
    cp = _first(*[os.path.join(d, "syc6_qwen3-vl-2b_text.json")
                  for d in probe_cands])
    if cp is None:
        return None
    with open(cp, encoding="utf-8") as f:
        conf = {key_of(r) for r in json.load(f).get("records", [])}

    kc = {t: {kk: v for kk, v in k[t].items() if kk in conf} for t in k}
    D = set.intersection(*[{kk for kk, v in kc[t].items() if v} for t in kc])
    excl = {t: len([kk for kk in conf if not kc[t].get(kk, False)])
            for t in kc}
    krate = {t: sum(kc[t].values()) / float(len(kc[t])) for t in kc}
    return dict(n_conf=len(conf), n_dstar=len(D), excl=excl, krate=krate,
                order=[t for t, _ in ladder])


# ================================================================ 数据
LAD = load("scale_ladder_negation.json")          # 主结果（否定感知判据）
ALIAS = load("scale_ladder.json")                 # 纯别名判据对照
LEN = load("length_confound_negation.json")
MDE = load("mde.json")
STRAT = load("stratified_negation.json")
ADJ = load("adjudicate_test_negation.json")
# D* 内逐格救回量（别名 BOTH → 否定感知 PARAM）。键为模型 tag、立场。
# 由 analyze_rescue_counts.py 从 probe/ 原始输出重算，口径与正文配对一致。
RESCUE = load("negation_rescue_counts.json")
MODELS = ["qwen3-vl-2b", "qwen3-vl-4b", "qwen3-vl-8b",
          "qwen2.5-vl-3b", "qwen2.5-vl-7b"]
XENV = load("cross_env_3b.json")
ANCH = load("ladder_anchors.json")

SMALL = {"qwen3": "qwen3-vl-2b", "qwen25": "qwen2.5-vl-3b"}
LARGE = {"qwen3": "qwen3-vl-8b", "qwen25": "qwen2.5-vl-7b"}
SMALLN = {"qwen3": "Qwen3-VL 2B", "qwen25": "Qwen2.5-VL 3B"}
LARGEN = {"qwen3": "Qwen3-VL 8B", "qwen25": "Qwen2.5-VL 7B"}
NSTAR = LAD["n_anchored"]


def delta_of(fam, stance, data=None):
    d = data or LAD
    seq = d["symbol_check"][fam]["seq"]
    for s in seq:
        if s["stance"] == stance:
            return s["delta_pp"]
    return None


def pkd_of(stance, tag, data=None):
    d = data or LAD
    return d["per_stance"][stance][tag]["pkd"]


def p_of(stance, fam, data=None):
    d = data or LAD
    for pr in d["pairs"]:
        if pr["stance"] == stance and pr["family"] == fam:
            return pr["p"], pr["n_paired"]
    return None, None


# ================================================================ 图
def fig01():
    """立场 × 规模的 PKD 折线：主图。"""
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.75), sharey=True)
    for ax, fam in zip(axes, FAMS):
        tags = fam_tags(LAD["tags"], fam)
        xs = np.arange(len(tags))
        for i, st in enumerate(STANCES):
            ys = [100 * pkd_of(st, t) for t in tags]
            ax.plot(xs, ys, "o-", color=STANCE_C[i], lw=1.5, ms=3.2,
                    label=CN[st], zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels([t.split("-")[-1].upper() for t in tags])
        ax.set_xlabel("模型规模")
        ax.set_title(FAMLAB[fam], pad=4)
    axes[0].set_ylabel("PKD 率（%）")
    axes[0].legend(ncol=2, loc="upper left", handlelength=1.3,
                   columnspacing=0.7, labelspacing=0.28)
    fig.suptitle("各指令立场下 PKD 率随规模的变化（否定感知判据，"
                 r"$|\mathcal{D}^{*}|=%d$）" % NSTAR,
                 fontsize=9.2, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig01_stance_scale_lines")


def fig02():
    """Δ 的符号由授权强度决定。"""
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.9), sharex=True)
    for ax, fam in zip(axes, FAMS):
        sc = LAD["symbol_check"][fam]
        seq = sc["seq"]
        xs = np.arange(len(seq))
        ys = [s["delta_pp"] for s in seq]
        cols = [C_NEG if v < 0 else C_POS for v in ys]
        ax.bar(xs, ys, color=cols, width=0.66, zorder=3,
               edgecolor="white", linewidth=0.5)
        for x, v in zip(xs, ys):
            ax.text(x, v + (2.2 if v >= 0 else -2.2), "%+.1f" % v,
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=6.8, color="#333333")
        ax.axhline(0, color="#333333", lw=0.9, zorder=4)
        ax.set_xticks(xs)
        ax.set_xticklabels([CN[s["stance"]] for s in seq],
                           rotation=32, ha="right")
        ax.set_title("%s\n显著 %d/%d，跨度 %.1f pp，符号%s序"
                     % (FAMLAB[fam], sc["n_sig"], sc["n_tot"], sc["span_pp"],
                        "有" if sc["ordered_signs"] else "无"),
                     fontsize=8.2, pad=4)
        ax.set_ylim(min(ys) - 11, max(ys) + 11)
    axes[0].set_ylabel("规模效应 $\\Delta$（pp）")
    fig.suptitle("规模效应 $\\Delta$ 的符号由授权强度决定", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig02_delta_by_stance")


def fig03():
    """两判据的 Δ 对照。"""
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.85), sharey=True)
    for ax, fam in zip(axes, FAMS):
        for data, col, lab, mk, dz in [
                (ALIAS, C_GREY, "纯别名判据", "o", 0),
                (LAD, C_QWEN3 if fam == "qwen3" else C_QWEN25,
                 "否定感知判据", "D", 0.07)]:
            seq = LAD["symbol_check"][fam]["seq"] if data is LAD \
                else ALIAS["symbol_check"][fam]["seq"]
            xs = np.arange(len(seq)) + dz
            ys = [s["delta_pp"] for s in seq]
            ax.plot(xs, ys, mk, color=col, ms=3.4, lw=1.3, label=lab, zorder=3)
        ax.axhline(0, color="#333333", lw=0.8, zorder=2)
        ax.axvline(3.5, color=C_ACC, lw=0.9, ls="--", zorder=2)
        ax.set_xticks(np.arange(7))
        ax.set_xticklabels([CN[s] for s in STANCES], rotation=32, ha="right")
        ax.set_title(FAMLAB[fam], fontsize=8.6, pad=4)
        ax.text(6.3, ax.get_ylim()[1] * 0.82, "强授权", fontsize=6.6,
                color=C_ACC, ha="right")
    axes[0].set_ylabel("规模效应 $\\Delta$（pp）")
    axes[0].legend(loc="lower left", handlelength=1.2)
    fig.suptitle("两判据下的 $\\Delta$ 对照（虚线右侧为强授权立场）", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save(fig, "fig03_criterion_compare")


def fig04():
    """锚定协议：五规模知识率与 D* 收缩。

    数据优先从 probe/ 原始输出重算（五个规模点、|D*|=649）；
    原始文件不在时回退到 ladder_anchors.json，但那只有两个规模点，
    与正文口径不一致，故打印显式告警。
    """
    d5 = recompute_dstar()
    if d5 is None:
        print("  ! fig04：缺 probe/ 原始锚定输出，回退到 ladder_anchors.json"
              "（仅覆盖 Qwen2.5 两个规模点，与正文 |D*|=649 不一致）")
        n_com = ANCH["n_common"]
        krate = ANCH["k_rate"]
        excl = {t: round(n_com * (1 - v)) for t, v in krate.items()}
        n_star = ANCH["n_anchored"]
        tags = list(krate)
    else:
        n_com = d5["n_conf"]
        krate = d5["krate"]
        excl = d5["excl"]
        n_star = d5["n_dstar"]
        tags = d5["order"]

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5))
    ax = axes[0]
    ys = [100 * krate[t] for t in tags]
    # 两族配色：Qwen3 蓝、Qwen2.5 红，与全文一致。
    cols = [C_QWEN3 if t.startswith("qwen3") else C_QWEN25 for t in tags]
    ax.barh(np.arange(len(tags)), ys, color=cols, height=0.62, zorder=3)
    for y, v in enumerate(ys):
        ax.text(v + 1.2, y, "%.1f%%" % v, va="center", fontsize=7)
    ax.set_yticks(np.arange(len(tags)))
    ax.set_yticklabels([t.replace("qwen", "Qwen") for t in tags], fontsize=7)
    ax.set_xlim(0, 112)
    ax.set_xlabel("判为已知的比例 $K_s$（%）")
    ax.set_title("各规模在 %d 条冲突样本上的知识率" % n_com, fontsize=8.4)
    ax.grid(axis="y", alpha=0)

    ax = axes[1]
    cut = [excl[t] for t in tags]
    ax.barh(np.arange(len(tags)), cut, color=C_NEG, height=0.62, zorder=3,
            alpha=0.85)
    for y, v in enumerate(cut):
        ax.text(v + n_com * 0.008, y,
                "%d (%.1f%%)" % (v, 100.0 * v / n_com),
                va="center", fontsize=6.8)
    ax.set_yticks(np.arange(len(tags)))
    ax.set_yticklabels([""] * len(tags))
    ax.set_xlim(0, n_com * 0.3)
    ax.set_xlabel("被锚定集剔除的样本数")
    ax.set_title("锚定剔除量（交集 $|\\mathcal{D}^{*}|=%d$，占 %.1f%%）"
                 % (n_star, 100 * n_star / n_com), fontsize=8.4)
    ax.grid(axis="y", alpha=0)
    # 图题只写内容，编号交给 LaTeX 的 \caption —— 图内再写一遍「图 N」会在
    # 分章重排后对不上号（原「图 4」就是被排成了正文的图 1）。
    fig.suptitle("锚定协议 $\\mathcal{D}^{*}=\\bigcap_s\\{i:K_s(i)=1\\}$ "
                 "的构造与代价", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig04_anchoring")


def fig05():
    """立场梯度。"""
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.6))
    for ax, fam in zip(axes, FAMS):
        sc = LAD["symbol_check"][fam]
        seq = sc["seq"]
        xs = np.arange(len(seq))
        ys = [100 * pkd_of(s["stance"], LARGE[fam]) for s in seq]
        ax.plot(xs, ys, "o-", color=FAMC[fam], lw=1.5, ms=3.6, zorder=3)
        for x, s, y in zip(xs, seq, ys):
            hot = s["stance"] == "ctx_hedge"
            ax.plot(x, y, "o", ms=6.4 if hot else 0, mfc="none",
                    mec=C_ACC, mew=1.5, zorder=4)
            ax.annotate("%.1f" % y, (x, y), textcoords="offset points",
                        xytext=(0, 7 if not hot else -12), ha="center",
                        fontsize=6.8, color="#333")
        ax.axvline(1.5, color=C_GREY, lw=0.7, ls=":")
        ax.set_xticks(xs)
        ax.set_xticklabels([CN[s["stance"]] for s in seq], rotation=32,
                           ha="right")
        ax.set_title("%s　$\\rho=%.3f$（$p=%.3f$）"
                     % (FAMLAB[fam], sc["spearman_rho"], sc["spearman_p_perm"]),
                     fontsize=8.2, pad=4)
    axes[0].set_ylabel("PKD 率（%）")
    fig.suptitle("立场梯度：PKD 随授权强度上升，条件让步立场逸出阶梯",
                 y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig05_stance_gradient")


def fig06():
    """否定感知判据的救回量。

    口径：D*（各规模 K=1 的 649 条）内，某立场下「别名判据判 BOTH、
    否定感知判据改判 PARAM」的样本数，五个规模点求和。
    **不能**用 per_stance 的 param 数相减：那条式子两侧各用各的可判子集，
    差值同时掺进了分母变动，与本文贯穿全文的配对口径不一致（实测差 ~14%，
    且会漏掉「一侧判 NONE」的格）。
    """
    fig, ax = plt.subplots(figsize=(W, 2.35))
    stances = STANCES
    rescue = []
    for st in stances:
        r = 0
        for t in MODELS:
            r += RESCUE["counts"].get(t, {}).get(st, 0)
        rescue.append(r)
    xs = np.arange(len(stances))
    cols = [C_ACC if v == max(rescue) else C_QWEN3 for v in rescue]
    ax.bar(xs, rescue, color=cols, width=0.62, zorder=3,
           edgecolor="white", linewidth=0.5)
    for x, v in zip(xs, rescue):
        ax.text(x, v + 3, "%+d" % v, ha="center", fontsize=7.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([CN[s] for s in stances], rotation=28, ha="right")
    ax.set_ylabel("救回的 PARAM 判回数")
    if not any(rescue):
        raise SystemExit("fig06：救回量全为 0，数据未取到，拒绝画图。")
    ax.set_ylim(0, max(rescue) * 1.18)
    fig.suptitle("否定感知判据相对纯别名判据的救回量：集中在一格",
                 y=0.99, fontsize=9.2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig06_rescue_counts")


def fig07():
    """长度混淆的排除。"""
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.6))
    for ax, fam in zip(axes, FAMS):
        rows = [r for r in LEN["rows"] if r["family"] == fam]
        rows.sort(key=lambda r: STANCES.index(r["stance"]))
        xs = np.arange(len(rows))
        dl = [r["dlen"] for r in rows]
        dd = [r["delta_pp"] for r in rows]
        ok = [r["consistent"] for r in rows]
        ax.axhline(0, color="#333", lw=0.8, zorder=2)
        ax.axvline(0, color="#333", lw=0.8, zorder=2)
        for x, a, b, good in zip(xs, dl, dd, ok):
            ax.plot(x, a, "o", color=FAMC[fam], ms=4.0, zorder=3,
                    mfc="none" if good else C_NEG, mew=1.3)
            ax.plot(x, b, "s", color=C_ACC if good else C_NEG, ms=3.4,
                    zorder=3)
        # Pearson
        r = np.corrcoef(dl, dd)[0, 1]
        ax.set_title("%s　$r=%.2f$" % (FAMLAB[fam], r), fontsize=8.4, pad=4)
        ax.set_xticks(xs)
        ax.set_xticklabels([CN[s] for s in [r0["stance"] for r0 in rows]],
                           rotation=32, ha="right")
    axes[0].set_ylabel("长度差（字符）／$\\Delta$（pp）")
    axes[0].plot([], [], "o", color=C_GREY, mfc="none", ms=4, mew=1.2,
                 label="长度差")
    axes[0].plot([], [], "s", color=C_ACC, ms=3.4, label="$\\Delta$")
    axes[0].legend(loc="upper left", handlelength=1.1, ncol=2)
    fig.suptitle("长度混淆的排除：$\\Delta$ 与长度差无系统共变", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig07_length_confound")


def fig08():
    """立场 × 规模 判出率热力图。"""
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.0))
    for ax, fam in zip(axes, FAMS):
        tags = fam_tags(LAD["tags"], fam)
        M = np.array([[100 * LAD["per_stance"][s][t]["decided"]
                       for t in tags] for s in STANCES])
        im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(np.arange(len(tags)))
        ax.set_xticklabels([t.split("-")[-1].upper() for t in tags])
        ax.set_yticks(np.arange(len(STANCES)))
        ax.set_yticklabels([CN[s] for s in STANCES], fontsize=7.4)
        for i in range(len(STANCES)):
            for j in range(len(tags)):
                v = M[i, j]
                ax.text(j, i, "%.0f" % v, ha="center", va="center",
                        fontsize=6.9, color="#1a1a1a")
        bad = [(i, j) for i in range(len(STANCES))
               for j in range(len(tags)) if M[i, j] < 60]
        for i, j in bad:
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                       ec=C_NEG, ls="--", lw=1.2))
        ax.grid(False)
        ax.set_title(FAMLAB[fam], fontsize=8.4, pad=4)
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("判出率（%）", fontsize=7.6)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle("立场 $\\times$ 规模全表：虚线框为分母塌陷格", y=0.995,
                 fontsize=9.2)
    save(fig, "fig08_heatmap")


def fig09():
    """符号跨零。"""
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.75), sharey=True)
    for ax, fam in zip(axes, FAMS):
        for data, col, lab, mk, dz in [
                (ALIAS, C_GREY, "纯别名判据", "o", 0),
                (LAD, FAMC[fam], "否定感知判据", "D", 0.08)]:
            seq = data["symbol_check"][fam]["seq"]
            xs = np.arange(len(seq)) + dz
            ys = [s["delta_pp"] for s in seq]
            ax.plot(xs, ys, mk + "-", color=col, ms=3.4, lw=1.2,
                    label=lab, zorder=3, alpha=0.95)
            for x, v in zip(xs, ys):
                ax.plot(x, v, "o" if v < 0 else "o", ms=2.4, color=col)
        ax.axhline(0, color="#333", lw=1.0, zorder=4)
        ax.set_xticks(np.arange(7))
        ax.set_xticklabels([CN[s] for s in STANCES], rotation=32, ha="right")
        ax.set_title(FAMLAB[fam], fontsize=8.6, pad=4)
        ax.set_ylim(-38, 42)
    axes[0].set_ylabel("规模效应 $\\Delta$（pp）")
    axes[0].legend(loc="lower left", handlelength=1.2)
    fig.suptitle("符号跨零：Qwen3 族在纯别名判据下的长度对照立场逸出负区",
                 y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig09_symbol_crossing")


def fig10():
    """跨环境复现。"""
    rows = XENV["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5))
    xs = np.arange(len(rows))
    st = [r["cond"] for r in rows]

    ax = axes[0]
    lp = [r["pkd_local"] for r in rows]
    rp = [r["pkd_remote"] for r in rows]
    ax.plot(xs, lp, "o-", color=C_QWEN3, ms=3.6, lw=1.4, label="本地", zorder=3)
    ax.plot(xs, rp, "s--", color=C_QWEN25, ms=3.4, lw=1.4, label="远程",
            zorder=3)
    for x, r in zip(xs, rows):
        if abs(r["delta_pp"]) > 1.0:
            ax.annotate("%.1f" % r["delta_pp"], (x, r["pkd_remote"]),
                        textcoords="offset points", xytext=(0, -11),
                        ha="center", fontsize=6.4, color=C_NEG)
    ax.set_xticks(xs)
    ax.set_xticklabels([CN.get(s, s) for s in st], rotation=32, ha="right")
    ax.set_ylabel("PKD 率（%）")
    ax.set_title("配对口径，最大偏离 %.1f pp" % XENV["max_abs_pkd_delta_pp"],
                 fontsize=8.4)
    ax.legend(loc="best", handlelength=1.2)

    ax = axes[1]
    vb = [r["verbatim"] for r in rows]
    ax.bar(xs, vb, color=C_GREY, width=0.6, zorder=3)
    mv = XENV["mean_verbatim_pct"]
    ax.axhline(mv, color=C_NEG, lw=1.0, ls="--", zorder=4)
    ax.set_ylim(0, 108)
    ax.text(0.97, mv + 3, "均值 %.1f%%" % mv, transform=ax.get_yaxis_transform(),
            fontsize=7.4, ha="right", color=C_NEG)
    ax.set_xticks(xs)
    ax.set_xticklabels([CN.get(s, s) for s in st], rotation=32, ha="right")
    ax.set_ylabel("逐字一致率（%）")
    ax.set_title("字符串级复现差得多", fontsize=8.4)
    fig.suptitle("跨环境复现（Qwen2.5-VL 3B，%d 条共同样本）"
                 % XENV["n_shared"], y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig10_cross_env")


def fig11():
    """位置偏好自证否。

    关键：pA 必须经 rescore_harness.judge() 折算，即「选了参数真值」而非
    「选了字母 A」。直接数原始输出的首字母会得到 ~0.50，与论文数字对不上 ——
    因为 judge 用 swap_* 标记把字母含义翻转，字母位置与语义已解耦。
    """
    HS = load("probe/_harness_summary.json")
    conds = ["cb_default", "cb_conflict", "xie_implicit", "xie_explicit"]
    cn = {"cb_default": "默认值优先", "cb_conflict": "冲突值优先",
          "xie_implicit": "含蓄提示", "xie_explicit": "明确提示"}
    fig, ax = plt.subplots(figsize=(W, 2.5))
    xs = np.arange(len(conds))
    for i, (mm, col, lab) in enumerate([("3b", C_QWEN25, "Qwen2.5-VL 3B"),
                                        ("7b", C_QWEN3, "Qwen2.5-VL 7B")]):
        ys = [100 * HS["conds"][c]["pA"][mm] for c in conds]
        ax.plot(xs + (i - 0.5) * 0.06, ys, "o-", color=col, ms=3.8, lw=1.4,
                label=lab, zorder=3)
        for x, v in zip(xs + (i - 0.5) * 0.06, ys):
            ax.annotate("%.3f" % v, (x, v), textcoords="offset points",
                        xytext=(0, 6 if i == 0 else -12), ha="center",
                        fontsize=6.4, color=col)
    ax.axhline(50, color=C_GREY, lw=0.9, ls="--", zorder=2)
    ax.text(len(conds) - 0.5, 51.5, "随机水平 50%", fontsize=6.6,
            color=C_GREY, ha="right")
    ax.axvspan(-0.5, 0.5, color=C_NEG, alpha=0.10, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([cn[c] for c in conds])
    ax.set_ylim(30, 108)
    ax.set_ylabel("选中参数真值的比例")
    ax.legend(loc="lower left", ncol=2, handlelength=1.2)
    ax.set_title("默认顺序条件下高达 %.3f，其数字不可用于内容推断"
                 % HS["conds"]["cb_default"]["pA"]["3b"], fontsize=8.2, pad=4)
    fig.suptitle("位置偏好自证否", y=0.99, fontsize=9.2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig11_position_bias")


def fig12():
    """分域检验。families[fam] 是逐 (slot, stance) 的行，只画 place/other
    两个大槽位（slot_dist 里其余槽位 n<min_n，不具备功效，不作结论）。"""
    SLOTCN = {"place": "地点", "other": "其他"}
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.95), sharey=True)
    for ax, fam in zip(axes, FAMS):
        rows = [r for r in STRAT["families"][fam]
                if r["slot"] in SLOTCN and r.get("usable")]
        rows.sort(key=lambda r: (list(SLOTCN).index(r["slot"]),
                                 STANCES.index(r["stance"])))
        xs = np.arange(len(rows))
        ys = [r["delta_pp"] for r in rows]
        ns = [r["n"] for r in rows]
        ps = [r.get("p", 1.0) for r in rows]
        cols = [C_GREY if p >= 0.05 else (C_NEG if v < 0 else C_POS)
                for v, p in zip(ys, ps)]
        ax.bar(xs, ys, color=cols, width=0.66, zorder=3,
               edgecolor="white", linewidth=0.5)
        for x, v, p in zip(xs, ys, ps):
            star = "" if p >= 0.05 else ("$^{*}$" if p >= 0.01 else "$^{**}$")
            ax.text(x, v + (1.8 if v >= 0 else -1.8), "%+.1f%s" % (v, star),
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=6.3)
        ax.axhline(0, color="#333", lw=0.9, zorder=4)
        ax.set_xticks(xs)
        ax.set_xticklabels([CN[r["stance"]] for r in rows], rotation=42,
                           ha="right", fontsize=6.8)
        for x, r in zip(xs, rows):
            ax.text(x, 0.4, "%s\nn=%d" % (SLOTCN[r["slot"]], r["n"]),
                    ha="center", va="bottom", fontsize=5.6, color="#555")
        ax.set_ylim(min(ys) - 9, max(ys) + 9)
        ax.set_title(FAMLAB[fam], fontsize=8.4, pad=4)
    axes[0].set_ylabel("槽位内 $\\Delta$（pp）")
    fig.suptitle("分域检验：灰色为槽位内不显著（$p\\geq0.05$）", y=0.995,
                 fontsize=9.2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig12_stratified")


def fig13():
    """判据对照：跨度与显著格数。"""
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5))
    for ax, (metric, lab, fmt) in zip(axes, [
            ("span_pp", "$\\Delta$ 跨度（pp）", "%.1f"),
            ("n_sig", "显著格数（共 7）", "%d")]):
        xs = np.arange(2)
        w = 0.34
        for k, (data, col, name) in enumerate([
                (LAD, C_QWEN3, "否定感知"), (ALIAS, C_GREY, "纯别名")]):
            vals = [data["symbol_check"][f][metric] for f in FAMS]
            ax.bar(xs + (k - 0.5) * w, vals, width=w, color=col, zorder=3,
                   label=name, edgecolor="white", linewidth=0.5)
            for x, v in zip(xs + (k - 0.5) * w, vals):
                ax.text(x, v + max(vals) * 0.02, fmt % v, ha="center",
                        fontsize=7.2)
        ax.set_xticks(xs)
        ax.set_xticklabels(["Qwen3 族", "Qwen2.5 族"])
        ax.set_ylabel(lab)
        ax.set_ylim(0, max([LAD["symbol_check"][f][metric] for f in FAMS]
                           + [ALIAS["symbol_check"][f][metric] for f in FAMS]) * 1.25)
        if k == 0:
            ax.legend(loc="upper right", handlelength=1.1)
    fig.suptitle("判据对照：跨度相近，但显著格数不同", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig13_criterion_span")


def fig14():
    """实测效应量 vs 最小可检出效应。"""
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.7))
    for ax, fam in zip(axes, FAMS):
        ms = MDE["mde_summary"][fam]
        lo, hi = ms["min"], ms["max"]
        for i, st in enumerate(STANCES):
            v = delta_of(fam, st)
            pv, _ = p_of(st, fam)
            sig = pv is not None and pv < 0.05
            y = v
            ax.plot([i], [y], "o", ms=4.0,
                    color=(C_NEG if v < 0 else C_POS) if sig else C_GREY,
                    zorder=3)
        ax.axhspan(lo, hi, color=C_ACC, alpha=0.18, zorder=1,
                   label="最小可检出效应 %.1f–%.1f pp" % (lo, hi))
        ax.axhline(0, color="#333", lw=0.8, zorder=2)
        ax.axhline(lo, color=C_ACC, lw=0.8, ls="--", zorder=2)
        ax.axhline(hi, color=C_ACC, lw=0.8, ls="--", zorder=2)
        ax.set_xticks(np.arange(len(STANCES)))
        ax.set_xticklabels([CN[s] for s in STANCES], rotation=32, ha="right")
        ax.set_title("%s　中位检出限 %.1f pp" % (FAMLAB[fam], ms["median"]),
                     fontsize=8.2, pad=4)
        ax.legend(loc="upper left", handlelength=1.1)
    axes[0].set_ylabel("实测 $\\Delta$（pp）")
    fig.suptitle("实测效应量与逐立场最小可检出效应（灰点为不显著格）",
                 y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig14_mde")


def fig15():
    """别名判据的 BOTH 丢弃缺陷。"""
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.6))
    ax = axes[0]
    for fam, col in [("qwen3", C_QWEN3), ("qwen25", C_QWEN25)]:
        tags = fam_tags(LAD["tags"], fam)
        for st, ls in [("ctx_hedge", "-"), ("prior", "--")]:
            ys = [100 * LAD["per_stance"][st][t]["decided"] for t in tags]
            ax.plot(np.arange(len(tags)) + (0.05 if st == "prior" else -0.05),
                    ys, ls, marker="o", ms=3.0, lw=1.3, color=col,
                    label="%s · %s" % (FAMLAB[fam].split("（")[0], CN[st]))
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(["2B", "4B", "8B", "3B", "7B"])
    ax.set_xlabel("模型规模")
    ax.set_ylabel("判出率（%）")
    ax.set_title("判出率：两立场随规模的变化", fontsize=8.4)
    ax.legend(loc="lower left", ncol=1, fontsize=6.4, handlelength=1.2)

    ax = axes[1]
    xs = np.arange(len(STANCES))
    both = []
    for st in STANCES:
        v = sum(LAD["per_stance"][st][t]["both"] for t in LAD["tags"])
        both.append(v)
    ax.bar(xs, both, color=[C_ACC if v == max(both) else C_QWEN3 for v in both],
           width=0.62, zorder=3, edgecolor="white", linewidth=0.5)
    for x, v in zip(xs, both):
        ax.text(x, v + max(both) * 0.03, "%d" % v, ha="center", fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([CN[s] for s in STANCES], rotation=32, ha="right")
    ax.set_ylabel("BOTH 判回数（对数轴）")
    ax.set_title("BOTH 占比：一格高出两个数量级", fontsize=8.4)
    fig.suptitle("别名判据的 BOTH 丢弃缺陷在两个维度上的表现", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig15_both_defect")


FIGS = [fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09, fig10,
        fig11, fig12, fig13, fig14, fig15]


def main():
    want = [int(a) for a in sys.argv[1:] if a.isdigit()]
    for i, fn in enumerate(FIGS, 1):
        if want and i not in want:
            continue
        try:
            fn()
        except Exception:
            import traceback
            print("  ✗ 图 %d %s 失败" % (i, fn.__name__))
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
