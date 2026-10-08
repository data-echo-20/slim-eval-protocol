# -*- coding: utf-8 -*-
"""论文绘图统一样式。所有 Phase 6 图共用，保证字号/配色/线宽一致。

要点
----
1. 中文字体走 Noto Sans CJK SC（系统已装，见 fonts-noto-cjk）。
   显式指定字体文件路径而不是靠 fontManager 扫描，避免在无缓存环境下静默
   退化成方框 —— 退化了图还能出，但中文会变豆腐块，且人眼在缩略图上
   不一定立刻发现。
2. 只存 PDF + PNG 两份：PDF 供 LaTeX，PNG 供 Markdown 预览。
3. 关闭数学文本的斜体偏差，保证中文与公式混排时基线一致。
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

_HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "04_论文", "figures"))

# 中文字体：优先用预抽出的 SC 面（见 fonts/README 说明抽取理由）
# NotoSansCJK-*.ttc 是多面集合，face 0 是 **JP**。matplotlib 的 addfont 只注册
# face 0，于是「门」「骨」「直」这类共用汉字会拿到日文字形变体 —— 中文论文里
# 是实打实的错字，且在缩略图上不容易一眼看出。故预抽 face 2（SC）。
_FONT_DIR = os.path.join(_HERE, "fonts")
_CJK_CANDIDATES = [
    os.path.join(_FONT_DIR, "NotoSansSC-Regular.ttf"),
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]


def _install_fonts():
    chosen = None
    for p in _CJK_CANDIDATES:
        if os.path.exists(p):
            try:
                fm.fontManager.addfont(p)
                if chosen is None:
                    chosen = fm.FontProperties(fname=p).get_name()
            except Exception:
                continue
    return chosen


_CJK = _install_fonts()

# 配色：两个族用两条固定色，七个立场用同一渐变色序（弱授权 -> 强授权）
C_QWEN3 = "#1f77b4"     # 蓝
C_QWEN25 = "#d62728"    # 红
C_POS = "#2ca02c"       # 正号
C_NEG = "#c44e52"       # 负号
C_GREY = "#7f7f7f"
STANCE_COLORS = ["#4c72b0", "#6b8ebf", "#937860", "#8c8c8c",
                 "#dd8452", "#c44e52", "#8172b3"]


def apply():
    plt.rcParams.update({
        "font.sans-serif": ([_CJK] if _CJK else []) + ["DejaVu Sans"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,        # 负号用 ASCII，避免方块
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.autolayout": False,
    })
    return _CJK


def save(fig, name):
    """存 PDF + PNG 到 figures/。返回两个路径。"""
    os.makedirs(FIGDIR, exist_ok=True)
    pdf = os.path.join(FIGDIR, name + ".pdf")
    png = os.path.join(FIGDIR, name + ".png")
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    return pdf, png


def cn(stance):
    """立场名 -> 中文短标签（图中用）。"""
    return {
        "ctx_only": "仅上下文",
        "orig": "原始",
        "ctx_hedge": "驳回上下文",
        "neutral": "中性",
        "own_only": "仅自身",
        "len_ctrl": "长度对照",
        "prior": "优先自身",
    }.get(stance, stance)
