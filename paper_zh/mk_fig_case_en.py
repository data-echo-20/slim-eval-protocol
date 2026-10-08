#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图 16–17 的英文版。

与 mk_figs_en.py 同思路：复用 mk_fig_case.py 的几何与数据，只在最外层把
送进 matplotlib 的中文串换掉，输出到 04_论文/figures_en/。
翻译表直接取自 mk_figs_en（那里已覆盖这两张图用到的全部串）。
"""
import importlib.util
import os
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


EN = _load("mk_figs_en", os.path.join(HERE, "mk_figs_en.py"))
tr = EN.tr

C = _load("mk_fig_case", os.path.join(HERE, "mk_fig_case.py"))
C.OUT = EN.zh.OUT          # 与 mk_figs_en 同一个输出目录


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
_wrap("text", Axes, 2)
_wrap("annotate", Axes, 0)
_wrap("suptitle", Figure, 0)

_orig_legend = Axes.legend


def _legend(self, *a, **k):
    o = _orig_legend(self, *a, **k)
    for t in o.get_texts():
        t.set_text(tr(t.get_text()))
    return o


Axes.legend = _legend


def main():
    os.makedirs(C.OUT, exist_ok=True)
    C.fig16_case_rescue()
    C.fig17_case_examples()
    if EN._untranslated:
        print("\n!! 未翻译 %d 条：" % len(EN._untranslated))
        for u in sorted(set(EN._untranslated)):
            print("   ", repr(u))
        sys.exit(1)
    print("->", C.OUT)


if __name__ == "__main__":
    main()
