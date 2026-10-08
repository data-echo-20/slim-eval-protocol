# -*- coding: utf-8 -*-
"""把两张单栏图纵向拼成一张组合图（英文版）。

用法：
    python _combine_en.py          # 生成全部三张

坑（20261008 定位并修复）
------------------------
早先这版用 pypdf 的 `merge_transformed_page` 把两页并进同一个 mediabox。结果
**下图里所有字都渲染成错位的汉字**，而文字层却老老实实是英文 —— 用 pdftotext
或 PyMuPDF 抽文本会得出"这张图没问题"的错误结论，只有把页面渲成位图才看得见。

根因：matplotlib 出图时正文都是同一个 CJK 字体（NotoSansSC-Thin）的 **Type 3**
子集，两个源文件里的子集**同名前缀**（`GTXPTE+NotoSansSC-Thin`），但各自的
CharProcs 字形程序只对应该图自己用到的那些字。pypdf 合并时把资源改名成 `F1` /
`F1-0`，可**字体的 BaseFont 名字仍然相同**，阅读器按名字复用第一张图的字形缓存，
于是第二张图的每个字都被换成第一张图的字——恰好是汉字。

改法：改用 PyMuPDF 的 `show_pdf_page`，把每张源页作为**独立的 Form XObject**
嵌进来；XObject 的字体资源各自命名空间隔离，不再撞名。仍然是矢量合并，
不渲染成位图，字号位置与两张单图逐像素一致。
"""
import os

import fitz  # PyMuPDF

HERE = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.insert(0, HERE)
from mk_figs_zh import _resolve          # 与出图脚本共用同一套布局判断

# 别再自己猜目录：仓库布局里 ../figures 是**英文**图，猜错会把英文图当
# 中文图合并。_resolve() 返回 (工程根, results, 中文图目录, 英文图目录)。
_P, _R, _ZH, _EN = _resolve()
SRC = _ZH if "en" == "zh" else _EN
OUT = SRC                       # 组合图与单图同目录，出图构建按同目录取

GAP = 6.0                       # 两张子图之间留的空隙（pt）
# (输出名, 上图, 下图)
JOBS = [
    ("_c1_lines_delta", "fig01_stance_scale_lines", "fig02_delta_by_stance"),
    ("_c2_cross_len",   "fig09_symbol_crossing",    "fig07_length_confound"),
    ("_c3_crit_both",   "fig03_criterion_compare",  "fig15_both_defect"),
]


def combine(out, top, bot):
    dt = fitz.open(os.path.join(SRC, top + ".pdf"))
    db = fitz.open(os.path.join(SRC, bot + ".pdf"))
    wt, ht = dt[0].rect.width, dt[0].rect.height
    wb, hb = db[0].rect.width, db[0].rect.height
    W = max(wt, wb)
    H = ht + GAP + hb

    doc = fitz.open()
    page = doc.new_page(width=W, height=H)
    # 上图贴顶
    page.show_pdf_page(fitz.Rect((W - wt) / 2.0, 0,
                                 (W - wt) / 2.0 + wt, ht), dt, 0)
    # 下图贴底
    page.show_pdf_page(fitz.Rect((W - wb) / 2.0, H - hb,
                                 (W - wb) / 2.0 + wb, H), db, 0)

    dst = os.path.join(OUT, out + ".pdf")
    doc.save(dst, garbage=4, deflate=True)

    # 自查：抽文本确认两张子图的文字都在，且字体资源确已隔离。
    fonts = {f[3] for f in fitz.open(dst)[0].get_fonts()}
    txt = fitz.open(dst)[0].get_text()
    miss = [k for k in ("PKD rate (%)", "Model scale", "Scale effect (pp)")
            if k not in txt and k in (dt[0].get_text() + db[0].get_text())]
    print("  %-18s < %s + %s   %.1f x %.1f pt  字体 %d 种%s"
          % (out, top, bot, W, H, len(fonts),
             "  缺字：" + ",".join(miss) if miss else ""))


if __name__ == "__main__":
    for o, a, b in JOBS:
        combine(o, a, b)
