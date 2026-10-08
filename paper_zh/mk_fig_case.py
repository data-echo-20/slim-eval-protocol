#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图 16–17：否定感知判据的救援机制。

为什么需要这两张图
------------------
正文 §4.2.1.1 用文字描述了一个反直觉的现象：模型越是明确地驳回错误的
上下文，越容易被纯别名判据判为「不可判定」而剔除。这个论证在纸面上不易
看懂——读者要同时记住四种判定（PARAM / CTX / BOTH / NONE）和两条判据的
差别。这两张图分别把「同一批答案在两条判据下的去向」和「为什么会被误判」
画出来。

数据来源（不重新计算，全部读 results/，与正文同源）：
  · results/scale_ladder.json           纯别名判据逐格判定计数
  · results/scale_ladder_negation.json  否定感知判据逐格判定计数
  · results/negation_rescue_counts.json 逐格救回条数
  · results/negation_rescue_rates.json  显式驳回占 BOTH 的比例
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

from mk_figs_zh import (install_font, load, _resolve, C_QWEN3, C_QWEN25,
                        C_POS, C_NEG, C_ACC, C_GREY)

# 与 mk_figs_zh 共用同一套路径解析（写作工程 / GitHub 仓库两种布局都能跑）。
PROJ, RES, OUT, _OUT_EN = _resolve()

CJK = install_font()
plt.rcParams.update({
    "font.sans-serif": [CJK, "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 8.6,
    "axes.linewidth": 0.7,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

W = 6.2

ALIAS = load("scale_ladder.json")
LAD = load("scale_ladder_negation.json")
RATE = load("negation_rescue_rates.json")

# 四条判定的配色：PARAM 是本文要测的量，用主色；BOTH 是要解释的缺陷，
# 用暖橙；CTX/NONE 是陪衬，压成灰阶，不进视觉竞争。
C_PARAM = "#1f5fa8"
C_CTX = "#b8c4d0"
C_BOTH = "#e08a1e"
C_NONE = "#dcdcdc"
LBL = {"param": "PARAM", "ctx": "CTX", "both": "BOTH", "none": "NONE"}
ORD = ["param", "ctx", "both", "none"]
COLS = [C_PARAM, C_CTX, C_BOTH, C_NONE]


def fig16_case_rescue():
    """左：同一格答案在两条判据下的判定组成；右：显式驳回比例随规模。"""
    fig = plt.figure(figsize=(W, 2.95))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.06], wspace=0.30,
                          bottom=0.235, top=0.895, left=0.13, right=0.985)

    # ------------------------------------------------ 左：判定组成对照
    # 取 Qwen3-VL 8B · ctx_hedge 这一格（正文 §4.2.1.1 的例子格）。
    # 两条判据面对的是**同一批 649 条**答案，差别全在四类判定的分配上。
    #
    # 用分组柱而非堆叠条：堆叠条虽然更直接地表现「总数守恒」，但这一格里
    # 别名判据的 PARAM 段只有 73 条、否定感知判据的 BOTH 段只有 22 条，
    # 段内写不下标签、标签写到段外又会被 bbox_inches="tight" 裁掉。
    # 分组柱把四个类别摊在横轴上，每个柱都自带上方标注空间，不会溢出。
    TAG, ST = "qwen3-vl-8b", "ctx_hedge"
    a = ALIAS["per_stance"][ST][TAG]
    n = LAD["per_stance"][ST][TAG]
    cats = ["param", "ctx", "both", "none"]
    ax = fig.add_subplot(gs[0, 0])
    xs = np.arange(len(cats))
    va = [a[k] for k in cats]
    vn = [n[k] for k in cats]
    # 两条判据用同族两色，与论文其余各图的配色一致
    ax.bar(xs - 0.19, va, width=0.34, color=C_GREY, zorder=3,
           edgecolor="white", linewidth=0.6, label="纯别名判据")
    ax.bar(xs + 0.19, vn, width=0.34, color=C_QWEN3, zorder=3,
           edgecolor="white", linewidth=0.6, label="否定感知判据")
    for x, v in zip(xs - 0.19, va):
        ax.text(x, v + 14, "%d" % v, ha="center", va="bottom", fontsize=6.8,
                color="#4a4a4a")
    for x, v in zip(xs + 0.19, vn):
        ax.text(x, v + 14, "%d" % v, ha="center", va="bottom", fontsize=6.8,
                color=C_QWEN3)
    ax.set_xticks(xs)
    ax.set_xticklabels([LBL[k] for k in cats], fontsize=7.4)
    ax.set_xlim(-0.62, 3.62)
    ax.set_ylim(0, 790)
    ax.set_yticks([0, 200, 400, 600])
    ax.tick_params(axis="y", labelsize=7.0)
    ax.set_ylabel("答案条数", fontsize=8.0)
    ax.grid(axis="y", color="#e6e6e6", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # 救援方向：BOTH 那一组的第 2 个柱（22）指向 PARAM 组的第 2 个柱（613）
    ax.annotate("", xy=(0.19, 700), xytext=(2.19, 700),
                arrowprops=dict(arrowstyle="-|>", color=C_POS, lw=1.5,
                                connectionstyle="arc3,rad=0.16"), zorder=5)
    ax.text(1.19, 712, "显式驳回 540 条改判 PARAM", ha="center", va="bottom",
            fontsize=6.5, color=C_POS, fontweight="bold")
    # 判据名用**轴内相对坐标**（transAxes）钉在左上角。数据坐标会随
    # xlim/ylim 的改动而漂移，相对坐标不会：PARAM 柱组右侧（x>0.30）
    # 的纵向带 0.60–0.72 是空的（CTX/BOTH/NONE 三组都远低于此高度）。
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(ax.transAxes, ax.transAxes)
    for yy, col, lab in [(0.585, C_GREY, "纯别名判据"),
                         (0.470, C_QWEN3, "否定感知判据")]:
        ax.add_patch(plt.Rectangle((0.62, yy - 0.032), 0.045, 0.062,
                                   fc=col, ec="white", lw=0.4, zorder=6,
                                   transform=tr, clip_on=False))
        ax.text(0.685, yy, lab, fontsize=6.6,
                color="#4a4a4a" if col == C_GREY else col,
                va="center", ha="left", zorder=6, transform=tr)
    ax.set_title("(a) 同一批 649 条答案的判定分配", fontsize=8.2, pad=5)

    # ------------------------------------------------ 右：显式驳回比例
    ax2 = fig.add_subplot(gs[0, 1])
    # prior 立场，BOTH 中属于显式驳回的比例（D* 上实测）。
    # 五个规模点全画，不抽稀——抽掉 8B 会让 Qwen3 族看起来单调上升，
    # 而 4B→8B 实为下降 13.1 pp，正文 §4.2.1.1 专门解释了这个非单调点。
    fam = ["Qwen3-VL\n2B", "Qwen3-VL\n4B", "Qwen3-VL\n8B",
           "Qwen2.5-VL\n3B", "Qwen2.5-VL\n7B"]
    # 键是显示名而非 tag（"Qwen3-VL 2B"），取值时按这里列出的顺序对齐
    MODELKEY = ["Qwen3-VL 2B", "Qwen3-VL 4B", "Qwen3-VL 8B",
                "Qwen2.5-VL 3B", "Qwen2.5-VL 7B"]
    pct = [RATE["per_model"][k]["pct"] for k in MODELKEY]
    cols = [C_QWEN3, C_QWEN3, C_QWEN3, C_QWEN25, C_QWEN25]
    xs = np.array([0, 1, 2, 3.2, 4.2])
    ax2.bar(xs, pct, width=0.72, color=cols, zorder=3,
            edgecolor="white", linewidth=0.6)
    for x, y in zip(xs, pct):
        ax2.text(x, y + 2.4, "%.1f" % y, ha="center", va="bottom",
                 fontsize=7.2, color="#2b2b2b")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(fam, fontsize=6.4)
    ax2.set_ylabel("BOTH 中属于显式驳回的比例（%）", fontsize=7.6)
    ax2.grid(axis="y", color="#e4e4e4", lw=0.6, zorder=0)
    ax2.set_axisbelow(True)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.set_xlim(-0.65, 4.85)
    ax2.set_ylim(0, 123)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.tick_params(axis="y", labelsize=7.0)

    # 三处标注各占一条**互不重叠的水平带**，全部落在柱条与柱顶数字之上
    # （最高柱 88.8，其数字顶到约 95）：
    #   带1 y≈98   同族端点跨度（2B→8B、3B→7B），与正文 §4.2.1.1 口径一致
    #   带2 y≈107  4B→8B 的非单调落差
    # 早先把跨度箭头画在 101、非单调标注画在 92.5，正好压住 88.8/85.8
    # 两个柱顶数字，才有「文字重叠」的观感。
    ax2.annotate("", xy=(2.0, 98), xytext=(0.0, 98),
                 arrowprops=dict(arrowstyle="-|>", color=C_ACC, lw=1.4))
    ax2.text(1.0, 99.5, "同族端点 +56.7 pp", ha="center", va="bottom",
             fontsize=6.6, color=C_ACC)
    ax2.annotate("", xy=(4.2, 98), xytext=(3.2, 98),
                 arrowprops=dict(arrowstyle="-|>", color=C_ACC, lw=1.4))
    ax2.text(3.7, 99.5, "+32.7 pp", ha="center", va="bottom",
             fontsize=6.6, color=C_ACC)
    ax2.annotate("", xy=(2.0, 108), xytext=(1.0, 108),
                 arrowprops=dict(arrowstyle="-|>", color=C_NEG, lw=1.2))
    ax2.text(1.5, 109.5, "4B→8B −13.1 pp（非单调）", ha="center", va="bottom",
             fontsize=6.3, color=C_NEG)
    ax2.set_title("(b) 显式驳回比例随规模上升", fontsize=8.2, pad=5)

    fig.savefig(os.path.join(OUT, "fig16_case_rescue.pdf"))
    fig.savefig(os.path.join(OUT, "fig16_case_rescue.png"))
    plt.close(fig)
    print("  ✓ fig16_case_rescue")


def fig17_case_examples():
    """逐字样本：被纯别名判据丢弃、被否定感知判据救回的答案。

    排版要点：每个样本卡片内，「模型输出」与两条「命中」占**不同的行**。
    早先把两条命中并排写在 x=0.22 与 x=4.9 上，长文（如
    「命中 参数真值 Scotland」）会越过中线撞上右边那条，故改为竖排。
    """
    cases = [
        ("参数真值 Scotland", "Nepal（上下文）",
         "The context is wrong. Princes Street Gardens is located in\n"
         "Edinburgh, Scotland, not Nepal."),
        ("参数真值 insects", "crustaceans（上下文）",
         "Chamaeleo chamaeleon is insectivorous and primarily eats\n"
         "insects, not crustaceans. The context incorrectly states..."),
        ("参数真值 Bristol", "Valparaiso（上下文）",
         "The context is incorrect. Portland Square is not in Valparaiso.\n"
         "Portland Square is located in Bristol, England."),
        ("参数真值 Ramla", "Mombasa, Israel（上下文）",
         "The context is incorrect. The White Mosque is not located in\n"
         "Mombasa, Israel — Mombasa is a city in Kenya."),
    ]
    fig, axes = plt.subplots(len(cases), 1, figsize=(W, 3.9))
    fig.subplots_adjust(hspace=0.30, bottom=0.10)
    for k, (ax, (param, ctx, txt)) in enumerate(zip(axes, cases)):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0.0, 0.0), 9.95, 9.95,
                                   facecolor="#f6f7f8",
                                   edgecolor="#ccd2d8", lw=0.6))
        ax.text(0.28, 9.30, "模型输出", fontsize=6.8, color="#7b8794",
                va="top")
        ax.text(0.28, 7.55, txt, fontsize=6.7, color="#1a1a1a",
                va="top", linespacing=1.55, family="monospace")
        # 两条命中竖排，各占一行，左端对齐 —— 不再有横向碰撞
        ax.text(0.28, 2.95, "命中" + param, fontsize=7.0, color=C_POS,
                va="center", fontweight="bold")
        ax.text(0.28, 1.35, "命中" + ctx, fontsize=7.0, color=C_QWEN25,
                va="center", fontweight="bold")
    # 说明放图底一行共享，而不是塞进每张卡片——避免被卡片右边界截断
    fig.text(0.5, 0.015,
             "两判据均命中两个值 → 纯别名判据判为 BOTH；"
             "否定感知判据识别驳回词 → 改判 PARAM",
             ha="center", va="bottom", fontsize=6.9, color="#4a4a4a")
    fig.savefig(os.path.join(OUT, "fig17_case_examples.pdf"))
    fig.savefig(os.path.join(OUT, "fig17_case_examples.png"))
    plt.close(fig)
    print("  ✓ fig17_case_examples")


if __name__ == "__main__":
    fig16_case_rescue()
    fig17_case_examples()
