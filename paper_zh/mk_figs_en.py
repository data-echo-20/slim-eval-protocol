#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成英文版图：与 mk_figs_zh.py 共用同一套绘图代码与同一份数据，
只在最外层把送进 matplotlib 的字符串换成英文，输出到 04_论文/figures_en/。

为什么要这样做
--------------
图的骨架、配色、字号、注释位置全部与中文版逐像素一致 —— 只有文字不同。
复制一份 draw 代码会立刻产生两套需要分别维护的图，改一处漏一处；
这里用一层翻译包住，数据与几何只有一份。

用法：
    python mk_figs_en.py            # 全部 15 张
    python mk_figs_en.py 1 2        # 只画第 1、2 张
"""
import importlib.util
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


zh = _load("mk_figs_zh", os.path.join(HERE, "mk_figs_zh.py"))
# 输出目录改成英文版。解析逻辑在 mk_figs_zh._resolve()，两种布局都覆盖：
# 写作工程 → 04_论文/figures_en/，仓库 → 仓库根的 figures/。
zh.OUT = zh.OUT_EN

# 字体必须以 TrueType 子集（Type 0/Identity-H）嵌入，不能用 matplotlib 的
# 默认 Type 3。原因（20261008 定位）：正文用的是 CJK 字体 NotoSansSC-Thin，
# 它的**拉丁字形**在 Type 3 嵌入下会渲染错位 —— 英文图里 "PKD rate (%)" 会
# 被画成一串汉字，而文字层仍是正确的 ASCII，用 pdftotext/PyMuPDF 抽文本查
# 不出来，只有把 PDF 渲成位图才看得见。中文图不受影响（正文就是汉字）。
# 顺带把 pdf.fonttype/ps.fonttype 设成 42 后文字变成可选可搜。
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# ---------------------------------------------------------------- 翻译表
# 1) 完整字符串（标题、轴标签、图例项）。这些在运行时已拼好，直接整句替换。
FULL = {
    # ---- 图题（suptitle）----
    "各指令立场下 PKD 率随规模的变化（否定感知判据，"
    r"$|\mathcal{D}^{*}|=%d$" % zh.NSTAR:
        "PKD rate vs. model scale under each instruction stance "
        r"(negation-aware criterion, $|\mathcal{D}^{*}|=%d$)" % zh.NSTAR,
    r"规模效应 $\Delta$ 的符号由授权强度决定":
        r"The sign of the scale effect $\Delta$ is set by authorisation strength",
    r"两判据下的 $\Delta$ 对照（虚线右侧为强授权立场）":
        r"$\Delta$ under the two criteria (dashed line marks the strong-authorisation stances)",
    "锚定协议 $\\mathcal{D}^{*}=\\bigcap_s\\{i:K_s(i)=1\\}$ 的构造与代价":
        "Construction and cost of the anchoring protocol "
        r"$\mathcal{D}^{*}=\bigcap_s\{i:K_s(i)=1\}$",
    "立场梯度：PKD 随授权强度上升，条件让步立场逸出阶梯":
        "Stance gradient: PKD rises with authorisation strength; "
        "the hedged stance escapes the ladder",
    "否定感知判据相对纯别名判据的救回量：集中在一格":
        "Cases rescued by the negation-aware criterion over the "
        "alias-only one: concentrated in a single cell",
    r"长度混淆的排除：$\Delta$ 与长度差无系统共变":
        r"Length confound ruled out: $\Delta$ does not co-vary systematically with length",
    r"立场 $\times$ 规模全表：虚线框为分母塌陷格":
        r"Stance $\times$ scale decision-rate table; dashed boxes mark denominator collapse",
    "符号跨零：Qwen3 族在纯别名判据下的长度对照立场逸出负区":
        "Sign crossing: under the alias-only criterion the length-control "
        "stance escapes the negative region in Qwen3",
    "跨环境复现（Qwen2.5-VL 3B，%d 条共同样本）" % zh.XENV["n_shared"]:
        "Cross-environment replication (Qwen2.5-VL 3B, %d shared samples)"
        % zh.XENV["n_shared"],
    "位置偏好自证否": "Position preference refutes itself",
    r"分域检验：灰色为槽位内不显著（$p\geq0.05$）":
        r"Stratified test: grey marks non-significant within-slot effects ($p\geq0.05$)",
    "判据对照：跨度相近，但显著格数不同":
        "Criterion comparison: similar spans, different numbers of "
        "significant cells",
    "实测效应量与逐立场最小可检出效应（灰点为不显著格）":
        "Observed effect sizes against per-stance minimum detectable "
        "effects (grey = non-significant)",
    "别名判据的 BOTH 丢弃缺陷在两个维度上的表现":
        "The alias-only BOTH-discard defect along two dimensions",
    # ---- 轴标签 ----
    "模型规模": "Model scale",
    "PKD 率（%）": "PKD rate (%)",
    r"规模效应 $\Delta$（pp）": r"Scale effect $\Delta$ (pp)",
    "判出率（%）": "Decision rate (%)",
    r"长度差（字符）／$\Delta$（pp）": r"Length diff (chars) / $\Delta$ (pp)",
    r"槽位内 $\Delta$（pp）": r"Within-slot $\Delta$ (pp)",
    r"实测 $\Delta$（pp）": r"Observed $\Delta$ (pp)",
    r"$\Delta$ 跨度（pp）": r"$\Delta$ span (pp)",
    "显著格数（共 7）": "Significant cells (of 7)",
    "逐字一致率（%）": "Verbatim-agreement rate (%)",
    "选中参数真值的比例": "Rate of selecting the parametric truth",
    "BOTH 判回数（对数轴）": "BOTH decisions (log scale)",
    "救回的 PARAM 判回数": "PARAM decisions rescued",
    r"判为已知的比例 $K_s$（%）": r"Known rate $K_s$ (%)",
    # ---- 图例 / 注释 ----
    "本地": "local",
    "远程": "remote",
    "长度差": "Length diff",
    r"$\Delta$": r"$\Delta$",
    "随机水平 50%": "Chance level 50%",
    "纯别名判据": "Alias-only criterion",
    "否定感知判据": "Negation-aware criterion",
    "否": "No",
    "有": "Yes",
    "强授权": "strong auth.",
    "弱授权": "weak auth.",
    # ---- 子图标题 ----
    "字符串级复现差得多": "verbatim agreement is far lower",
    "判出率：两立场随规模的变化": "Decision rate vs. scale for two stances",
    "BOTH 占比：一格高出两个数量级": "BOTH share: one cell is two orders higher",
    # ---- 立场名（CN 字典）----
    "仅上下文": "ctx only",
    "原始": "original",
    "条件让步": "hedged",
    "中立": "neutral",
    "仅自有": "own only",
    "长度对照": "len ctrl",
    "自有优先": "prior",
    # ---- 槽位 / 条件名 ----
    "地点": "place",
    "其他": "other",
    "默认值优先": "default-first",
    "冲突值优先": "conflict-first",
    "含蓄提示": "implicit hint",
    "明确提示": "explicit hint",
    # ---- 族名 ----
    "Qwen3 族": "Qwen3",
    "Qwen2.5 族": "Qwen2.5",
    "Qwen3-VL（2B / 4B / 8B）": "Qwen3-VL (2B / 4B / 8B)",
    "Qwen2.5-VL（3B / 7B）": "Qwen2.5-VL (3B / 7B)",
    # ---- mk_fig_case ----
    "(a) 同一批 649 条答案的判定分配":
        "(a) Decision breakdown over the same 649 answers",
    "(b) 显式驳回比例随规模上升":
        "(b) Explicit-rejection share rises with scale",
    "BOTH 中属于显式驳回的比例（%）":
        "Share of BOTH that is explicit rejection (%)",
    "左：同一格答案在两条判据下的判定组成；右：显式驳回比例随规模。":
        "Left: decision composition of one cell under both criteria. "
        "Right: explicit-rejection share vs. scale.",
    "两判据均命中两个值 → 纯别名判据判为 BOTH；":
        "both criteria match two values -> alias-only marks BOTH;",
    "否定感知判据识别驳回词 → 改判 PARAM":
        "negation-aware detects the rejection cue -> PARAM",
    "显式驳回 540 条改判 PARAM": "540 explicit rejections relabelled PARAM",
    "参数真值 Bristol": "parametric truth Bristol",
    "参数真值 Ramla": "parametric truth Ramla",
    "参数真值 Scotland": "parametric truth Scotland",
    "参数真值 insects": "parametric truth insects",
    "Mombasa, Israel（上下文）": "Mombasa, Israel (context)",
    "Nepal（上下文）": "Nepal (context)",
    "Valparaiso（上下文）": "Valparaiso (context)",
    "crustaceans（上下文）": "crustaceans (context)",
    "模型输出": "model output",
    "答案条数": "answers",
    "命中": "match",
    "4B→8B −13.1 pp（非单调）": "4B->8B -13.1 pp (non-monotone)",
    "同族端点 +56.7 pp": "family endpoint +56.7 pp",
}

# 2) 带格式参数的动态串，用正则替换。
PAT = [
    # 图 1 图题（|D*| 数字会变，用正则兜住）
    (re.compile(r"^图 1　各指令立场下 PKD 率随规模的变化（否定感知判据，(.+)）$"),
     r"PKD rate vs. model scale under each instruction stance "
     r"(negation-aware criterion, \1)"),
    # 锚定剔除量注释
    (re.compile(r"^锚定剔除量（交集 (.+)，占 (.+)）$"),
     r"Anchoring removals (intersection \1, \2)"),
    # fig02 子图标题。"有/无序" 直接映射成 ordered / not ordered，
    # 不能把中文那个字带进英文串。
    (re.compile(r"^(Qwen[\d.]+-VL)（(.+)）\n显著 (\d+/\d+)，跨度 (.+) pp，符号有序$"),
     r"\1 (\2)\n\3 significant, span \4 pp, signs ordered"),
    (re.compile(r"^(Qwen[\d.]+-VL)（(.+)）\n显著 (\d+/\d+)，跨度 (.+) pp，符号无序$"),
     r"\1 (\2)\n\3 significant, span \4 pp, signs NOT ordered"),
    # 图例："Qwen2.5-VL · 条件让步"
    (re.compile(r"^(Qwen[\d.]+-VL) · (仅上下文|原始|条件让步|中立|仅自有|长度对照|自有优先)$"),
     r"\1 · \2"),
    # fig12 的 "地点\nn=347"
    (re.compile(r"^(地点|其他)\nn=(\d+)$"), r"\1\nn=\2"),
    # 通用尾缀
    (re.compile(r"^(.+)　\\rho=(.+)$"), r"\1  rho=\2"),
    (re.compile(r"^(.+)　\$r=(.+)\$$"), r"\1  r=\2"),
    (re.compile(r"^(.+)　中位检出限 (.+) pp$"), r"\1  median MDE \2 pp"),
    (re.compile(r"^配对口径，最大偏离 (.+) pp$"), r"paired scale, max deviation \1 pp"),
    (re.compile(r"^默认顺序条件下高达 (.+)，其数字不可用于内容推断$"),
     r"reaches \1 under the default order; unusable for content inference"),
    (re.compile(r"^被锚定集剔除的样本数$"), r"Samples removed by anchoring"),
    (re.compile(r"^最小可检出效应 (.+) pp$"), r"Minimum detectable effect \1 pp"),
    (re.compile(r"^均值 (.+)%$"), r"mean \1%"),
    (re.compile(r"^各规模在 (\d+) 条冲突样本上的知识率$"),
     r"Known rate on \1 conflict samples"),
]

CJK = re.compile(u"[一-鿿]")
_untranslated = []


# FULL 里没有 CJK 的值（如 "$\Delta$"）不适合做子串替换的键，先剔掉。
_TOKENS = sorted([(k, v) for k, v in FULL.items() if CJK.search(k)],
                 key=lambda kv: -len(kv[0]))


def tr(s):
    # set_xticklabels / set_yticklabels 传进来的是**列表**，早先的实现直接原样返回，
    # 于是 fig10/11/13/14 的刻度标签全留着中文。逐项翻译后再返回。
    if isinstance(s, (list, tuple)):
        return type(s)(tr(x) for x in s)
    if not isinstance(s, str) or not CJK.search(s):
        return s
    if s in FULL:
        return FULL[s]
    for pat, rep in PAT:
        if pat.match(s):
            out = pat.sub(rep, s)
            # 正则里可能带出中文（族人名、立场名），再过一遍词替换。
            for k, v in _TOKENS:
                if k in out:
                    out = out.replace(k, v)
            if not CJK.search(out):
                return out
    # 兜底：按最长优先做子串替换（覆盖 "n=347" 这类拼装串）。
    out = s
    for k, v in _TOKENS:
        if k in out:
            out = out.replace(k, v)
    if not CJK.search(out):
        return out
    _untranslated.append(s)
    return s


# ---------------------------------------------------------------- 打补丁
def _wrap(name, cls, *pos):
    orig = getattr(cls, name)

    def fn(self, *a, **k):
        a = list(a)
        for i in pos:
            if i < len(a):
                a[i] = tr(a[i])
        return orig(self, *a, **k)
    setattr(cls, name, fn)


_wrap("set_title", Axes, 0)
_wrap("set_xlabel", Axes, 0)
_wrap("set_ylabel", Axes, 0)
_wrap("set_xticklabels", Axes, 0)
# 热图（fig08）用 set_yticklabels 写立场名，早先漏了这一个，整列立场名留在中文。
_wrap("set_yticklabels", Axes, 0)
_wrap("text", Axes, 2)
# Figure.text（fig.text(x, y, s)）是另一个方法，不在 Axes 上：fig17 底部那行
# 说明就是用它写的，只包 Axes.text 会漏掉。
_wrap("text", Figure, 2)
_wrap("annotate", Axes, 0)
_wrap("suptitle", Figure, 0)


def _legend(self, *a, **k):
    orig = _legend_orig(self, *a, **k)
    for t in orig.get_texts():
        t.set_text(tr(t.get_text()))
    return orig


_legend_orig = Axes.legend
Axes.legend = _legend


def main():
    want = [int(a) for a in sys.argv[1:] if a.isdigit()]
    os.makedirs(zh.OUT, exist_ok=True)
    skipped = []
    for i, fn in enumerate(zh.FIGS, 1):
        if want and i not in want:
            continue
        try:
            fn()
        except SystemExit as e:
            # 数据缺失（本副本没带 results/probe/）。跳过并记录，不中断其余图。
            skipped.append((i, fn.__name__, str(e)))
            print("  - 跳过 图 %d %s：%s" % (i, fn.__name__, e))
    if skipped:
        print("\n跳过 %d 张（数据缺失）：" % len(skipped))
        for i, n, why in skipped:
            print("   图 %d %s" % (i, n))
    if _untranslated:
        print("\n!! 未翻译 %d 条：" % len(_untranslated))
        for u in sorted(set(_untranslated)):
            print("   ", repr(u))
        sys.exit(1)
    print("\n英文版全部完成 ->", zh.OUT)


if __name__ == "__main__":
    main()
