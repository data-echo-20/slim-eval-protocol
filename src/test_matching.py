# -*- coding: utf-8 -*-
"""contains_any 回归测试。

覆盖三类答案形态：纯英文、纯中文、中英混合。
重点是中文——旧实现对中文恒返回 False，会把中文作答的样本
全部误判进 neither_rate，从而系统性压低 PKD-rate。

运行：python test_matching.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import contains_any

# (text, candidates, expected, 说明)
CASES = [
    # ---- 纯英文：基本正负例 ----
    ("Mongolia", ["Italy"], False, "英文不同答案"),
    ("Italy", ["Italy"], True, "英文精确命中"),
    ("The answer is Italy.", ["Italy"], True, "英文嵌在句中"),
    ("italy", ["Italy"], True, "大小写归一"),

    # ---- 纯英文：子串误判防护 ----
    ("The context says Mongolia, not Italy.", ["Italy"], True,
     "提到即命中——这是设计上的宽松，非 bug；此处断言其行为已知"),
    ("Italian cuisine", ["Italy"], False, "Italy 不得命中 Italian 的子串"),
    ("Ital", ["Italy"], False, "候选须完整出现"),
    ("I cannot tell", ["can"], False, "can 不得命中 cannot"),

    # ---- 纯中文：旧实现全部失败的用例 ----
    ("这是意大利的菜", ["意大利"], True, "中文命中（旧实现返回 False）"),
    ("这是蒙古的菜", ["意大利"], False, "中文不同答案"),
    ("答案是意大利。", ["意大利"], True, "中文嵌在句中"),
    ("这是意大利", ["意大利"], True, "中文位于句尾"),

    # ---- 纯中文：子串防护 ----
    ("意大利面很好吃", ["意大利"], True, "宽松匹配：意面含意大利，可接受"),
    ("abc意大利", ["意大利"], False, "中文候选两侧不得紧邻字母数字"),

    # ---- 中英混合 ----
    ("用 Samsung 手机拍摄", ["Samsung 手机"], True, "中英混合命中"),
    ("Samsung phone", ["Samsung 手机"], False, "中英混合不命中"),

    # ---- 边界 ----
    ("", ["Italy"], False, "空文本"),
    ("Italy", [], False, "空候选列表"),
    ("Italy", [""], False, "空候选"),
    ("蒙古", ["Mongolia"], False, "中英互不命中"),
]

# 多词英文答案
CASES += [
    ("We make jollof rice here", ["jollof rice"], True, "多词英文命中"),
    ("jollof", ["jollof rice"], False, "多词英文不完整"),
]


def main():
    ok = bad = 0
    for text, cands, exp, note in CASES:
        got = contains_any(text, cands)
        mark = "✓" if got == exp else "✗"
        if got == exp:
            ok += 1
        else:
            bad += 1
        print("%s %-34s %-22r exp=%-5s got=%-5s  %s"
              % (mark, repr(text)[:34], cands, exp, got, note))
    print("\n通过 %d / %d" % (ok, ok + bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
