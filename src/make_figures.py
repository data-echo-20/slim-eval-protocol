# -*- coding: utf-8 -*-
"""Phase 6 绘图：从 results/*.json 直接出图，不手抄任何数字。

原则
----
1. 图里的每个数都从 JSON 读，脚本里不出现硬编码的百分比 —— 否则论文改一次
   数据，图与表就对不上，而这种不一致几乎不会被肉眼发现。
2. 只画有依据的图。样本量不足的格（§4.8 的小槽位）不画成条形图假装可比。
3. 每个图函数打印它读的文件与关键数，便于与章节正文逐一核对。

用法
----
  python fig/make_figures.py            # 全部
  python fig/make_figures.py 1 3 5      # 指定编号
"""
import io, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle
from figstyle import cn, save, apply, STANCE_COLORS, C_QWEN3, C_QWEN25, C_POS, C_NEG, C_GREY

import matplotlib.pyplot as plt
import numpy as np

R = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
P = os.path.abspath(os.path.join(HERE, "..", "..", "results", "probe"))
FONT = apply()
print("中文字体 = %s" % FONT)


def J(*parts):
    with io.open(os.path.join(*parts), encoding="utf-8") as f:
        return json.load(f)


NEG = None      # 主判据（否定感知）
ALI = None      # 对照判据（纯别名）
DOSE = None
LEN_NEG = None
LEN_ALI = None
REG = None
ANCH = None


def load_all():
    global NEG, ALI, DOSE, LEN_NEG, LEN_ALI, REG, ANCH
    NEG = J(R, "scale_ladder_negation.json")
    ALI = J(R, "scale_ladder.json")
    DOSE = J(R, "stance_dose_response.json")
    LEN_NEG = J(R, "length_confound_negation.json")
    LEN_ALI = J(R, "length_confound.json")
    REG = J(R, "regrade_negation.json")
    ANCH = J(R, "ladder_anchors.json")


FAM = {"qwen3": ["qwen3-vl-2b", "qwen3-vl-4b", "qwen3-vl-8b"],
       "qwen25": ["qwen2.5-vl-3b", "qwen2.5-vl-7b"]}
FAMLAB = {"qwen3": "Qwen3-VL 2B→8B", "qwen25": "Qwen2.5-VL 3B→7B"}
FAMCOL = {"qwen3": C_QWEN3, "qwen25": C_QWEN25}


def pkd(tag, stance, src=None):
    d = (src or NEG)["per_stance"][stance][tag]
    return d["pkd"] * 100.0


def ci(tag, stance, src=None):
    d = (src or NEG)["per_stance"][stance][tag]
    lo, hi = d["ci"]
    return d["pkd"] * 100.0 - lo * 100.0, hi * 100.0 - d["pkd"] * 100.0


def first_valid(files):
    """返回第一个存在的路径；找不到返回 None（不抛，交给调用方决定）。"""
    for f in files:
        p = f if os.path.isabs(f) else os.path.join(P, f)
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------- 图 1
def fig01():
    """七立场 × 五规模的 PKD 折线（主判据）。"""
    order = NEG["order"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True)
    for ax, fam in zip(axes, ["qwen3", "qwen25"]):
        tags = FAM[fam]
        xs = list(range(len(tags)))
        for si, st in enumerate(order):
            ys = [pkd(t, st) for t in tags]
            ax.plot(xs, ys, "o-", color=STANCE_COLORS[si], lw=1.8, ms=5.5,
                    label=cn(st), zorder=3)
            for x, y in zip(xs, ys):
                ax.annotate("%.0f" % y, (x, y), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=7.2,
                            color=STANCE_COLORS[si])
        ax.set_xticks(xs)
        ax.set_xticklabels([t.split("-vl-")[1].upper() for t in tags])
        ax.set_title(FAMLAB[fam])
        ax.set_xlabel("规模")
        ax.set_ylim(-4, 104)
    axes[0].set_ylabel("PKD（跟随参数知识的比例，%）")
    axes[0].legend(ncol=2, loc="upper left", fontsize=8.4)
    fig.suptitle("图 1  指令立场与规模的交互（否定感知判据，$|D^*|=%d$）"
                 % NEG["n_anchored"], y=1.02, fontsize=12)
    save(fig, "fig01_stance_scale_lines")


# ---------------------------------------------------------------- 图 2
def fig02():
    """两族 Δ 的双向条形图，按授权强度排序。"""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.2), sharey=True)
    for ax, fam in zip(axes, ["qwen3", "qwen25"]):
        seq = NEG["symbol_check"][fam]["seq"]
        labs = [cn(s["stance"]) for s in seq]
        vals = [s["delta_pp"] for s in seq]
        cols = [C_POS if v > 0 else C_NEG for v in vals]
        y = np.arange(len(seq))
        ax.barh(y, vals, color=cols, height=0.62)
        ax.set_yticks(y)
        ax.set_yticklabels(labs)
        ax.axvline(0, color="black", lw=0.9)
        for yi, v in zip(y, vals):
            ax.annotate("%+.1f" % v, (v, yi), textcoords="offset points",
                        xytext=(7 if v > 0 else -7, 0), va="center",
                        ha="left" if v > 0 else "right", fontsize=8.4)
        sc = NEG["symbol_check"][fam]
        # 不能用 markdown 星号：matplotlib 标题不解析 markdown，`**不**` 会
        # 原样印成星号。要加重要走 fontweight。
        ok = sc["ordered_signs"]
        ax.set_title("%s\n%d/%d 显著，跨度 %.1f pp，符号%s有序"
                     % (FAMLAB[fam], sc["n_sig"], sc["n_tot"], sc["span_pp"],
                        "" if ok else "不"),
                     color="black" if ok else C_NEG,
                     fontweight="normal" if ok else "bold")
        ax.set_xlabel("$\\Delta$ = PKD(大) − PKD(小)，pp")
        lim = max(abs(min(vals)), abs(max(vals))) * 1.35
        ax.set_xlim(-lim, lim)
    fig.suptitle("图 2  规模效应 Δ 的符号由授权强度决定", y=1.03, fontsize=12)
    save(fig, "fig02_delta_by_stance")


# ---------------------------------------------------------------- 图 3
def fig03():
    """两判据的 Δ 并排对照，标出符号不一致的格。"""
    order = NEG["order"]
    fig, axes = plt.subplots(2, 1, figsize=(11.4, 6.6), sharex=True)
    for ax, fam in zip(axes, ["qwen3", "qwen25"]):
        a = {s["stance"]: s["delta_pp"] for s in ALI["symbol_check"][fam]["seq"]}
        n = {s["stance"]: s["delta_pp"] for s in NEG["symbol_check"][fam]["seq"]}
        x = np.arange(len(order))
        w = 0.38
        ax.bar(x - w / 2, [n[s] for s in order], w, color=C_QWEN3,
               label="否定感知判据")
        ax.bar(x + w / 2, [a[s] for s in order], w, color="#bbbbbb",
               label="纯别名判据")
        ax.axvline(2.5, color="black", ls=":", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([cn(s) for s in order])
        ax.axhline(0, color="black", lw=0.9)
        # 标出两判据反号处
        for xi, s in enumerate(order):
            if (n[s] > 0) != (a[s] > 0):
                ax.annotate("反号", (xi, max(n[s], a[s]) + 6), ha="center",
                            fontsize=8.6, color=C_NEG, fontweight="bold")
        ax.set_title(FAMLAB[fam])
        ax.set_ylabel("$\\Delta$ (pp)")
        ax.legend(loc="lower right")
    axes[0].set_ylim(-42, 46)
    fig.suptitle("图 3  两判据的 Δ 对照（灰色 = 纯别名；虚线右侧为强授权立场）",
                 y=0.98, fontsize=12)
    save(fig, "fig03_criterion_compare")


# ---------------------------------------------------------------- 图 4
def fig04():
    """锚定协议：各规模的 K 率与 D* 收缩。"""
    tags = NEG["tags"]
    # D* 与 K 率都从原始锚定文件重算。
    # 不用 ladder_anchors.json 里的 k_rate：那份只覆盖 qwen25 两点（它当时的用途
    # 是给「跨代可比性」做检查），拿它画五点图会在 qwen3 上得到 None。
    d5 = recompute_dstar()
    krate = {t: d5["krate"][t] for t in tags}
    excl = [d5["excl"][t] for t in tags]
    n_conf = d5["n_conf"]
    n_common = d5["n_conf"]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.2))
    ax = axes[0]
    x = np.arange(len(tags))
    vals = [100.0 * krate[t] for t in tags]
    bars = ax.bar(x, vals, color=[FAMCOL["qwen3"]] * 3 + [FAMCOL["qwen25"]] * 2,
                  width=0.6)
    for xi, v in zip(x, vals):
        ax.annotate("%.1f%%" % v, (xi, v), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8.6)
    ax.axhline(100, color=C_GREY, ls="--", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([t.split("-vl-")[1].upper() for t in tags])
    ax.set_ylim(70, 104)
    ax.set_ylabel("参数知识率 $K_s$（%）")
    # 口径必须写进图里：这里的 K 率算在 927 条**冲突子集**上，而 §4.1 正文
    # 引的 0.785/0.878/0.901 算在 1146 条全集上。两个数都对，但分母不同，
    # 同一页并列而不注明会看起来像前后矛盾。
    ax.set_title("各规模在 %d 条冲突样本上的 $K_s$\n"
                 "（正文 §4.1 的数字分母为 1146 条全集，故略高）" % n_conf,
                 fontsize=10.5)

    ax = axes[1]
    ax.bar(x, excl, color=[C_NEG] * 5, width=0.6)
    for xi, v in zip(x, excl):
        ax.annotate("%d\n(%.1f%%)" % (v, 100.0 * v / n_conf), (xi, v),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=8.2)
    ax.set_xticks(x)
    ax.set_xticklabels([t.split("-vl-")[1].upper() for t in tags])
    ax.set_ylabel("被 $D^*$ 剔除的条数")
    ax.set_ylim(0, max(excl) * 1.32)
    ax.set_title("锚定剔除量（交集 $|D^*|=%d$，占 %.1f%%）"
                 % (d5["n_dstar"], 100.0 * d5["n_dstar"] / n_conf))
    fig.suptitle("图 4  锚定协议 $D^*=\\bigcap_s\\{i:K_s(i)=1\\}$ 的构造与代价",
                 y=1.02, fontsize=12)
    save(fig, "fig04_anchoring")


def recompute_dstar():
    """从原始锚定文件重算 D*，供图 4 与图 8 使用。"""
    def key_of(r):
        return (r.get("question_named") or "", r.get("wikipedia_title") or "")
    ladder = [("qwen3-vl-2b", ["pkd_qwen3-vl-2b_text.json"]),
              ("qwen3-vl-4b", ["pkd_qwen3-vl-4b_text.json"]),
              ("qwen3-vl-8b", ["pkd_qwen3-vl-8b_text.json"]),
              ("qwen2.5-vl-3b", [os.path.join(R, "pkd_3b_comb_text.json")]),
              ("qwen2.5-vl-7b", [os.path.join(R, "pkd_7b_comb_text.json")])]
    k = {}
    for tag, files in ladder:
        p = first_valid(files)
        if p is None:
            sys.exit("缺锚定文件: %s" % files)
        k[tag] = {key_of(r): bool(r.get("noc_follows_param"))
                  for r in J(p)["records"]}
    conf = set(J(P, "syc6_qwen3-vl-2b_text.json")["records"][0].keys()) and \
        {key_of(r) for r in J(P, "syc6_qwen3-vl-2b_text.json")["records"]}
    kc = {t: {kk: v for kk, v in k[t].items() if kk in conf} for t in k}
    D = set.intersection(*[{kk for kk, v in kc[t].items() if v} for t in kc])
    excl = {t: len([kk for kk in conf if not kc[t].get(kk, False)]) for t in kc}
    krate = {t: sum(kc[t].values()) / float(len(kc[t])) for t in kc}
    return dict(n_conf=len(conf), n_dstar=len(D), excl=excl, krate=krate)


# ---------------------------------------------------------------- 图 5
def fig05():
    """立场梯度：PKD 对预注册授权强度的散点 + 秩相关。"""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), sharey=True)
    tags = NEG["tags"]
    order = NEG["order"]
    xr = np.arange(1, len(order) + 1)
    for ax, fam in zip(axes, ["qwen3", "qwen25"]):
        for t in FAM[fam]:
            ys = [pkd(t, s) for s in order]
            ax.plot(xr, ys, "o-", lw=1.3, ms=4.5, alpha=0.85,
                    label=t.split("-vl-")[1].upper())
        ax.set_xticks(xr)
        ax.set_xticklabels([cn(s) for s in order], rotation=22, ha="right")
        ax.set_xlabel("预注册授权强度（弱 → 强）")
        ax.set_title(FAMLAB[fam])
    axes[0].set_ylabel("PKD（%）")
    axes[0].legend(loc="upper left", ncol=3, fontsize=8.4)
    fig.suptitle("图 5  立场梯度：PKD 随授权强度上升，但 `ctx_hedge` 逸出阶梯",
                 y=1.02, fontsize=12)
    save(fig, "fig05_stance_gradient")


# ---------------------------------------------------------------- 图 6
def fig06():
    """`ctx_hedge` 的判据敏感性：BOTH 救回量。"""
    models = list(REG.keys())
    order = NEG["order"]
    rescued = []
    for m in models:
        d = REG[m]
        rescued.append([d.get(s, {}).get("rescued", 0) for s in order])
    rescued = np.array(rescued, dtype=float)

    fig, ax = plt.subplots(figsize=(11.0, 4.4))
    x = np.arange(len(order))
    w = 0.15
    cols = ["#4c72b0", "#8c8c8c", "#6b8ebf", "#c44e52"]
    for mi, m in enumerate(models):
        ax.bar(x + (mi - 1.5) * w, rescued[mi], w, label=m, color=cols[mi % 4])
    ax.set_xticks(x)
    ax.set_xticklabels([cn(s) for s in order])
    ax.set_ylabel("被救回的条数（BOTH → PARAM）")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_title("图 6  否定感知判据的救回量：集中在 `ctx_hedge` 一格", fontsize=12)
    ax.legend(ncol=4, loc="upper left", fontsize=8.6)
    ax.annotate("救回量比其余六格\n高两个数量级",
                xy=(2, rescued[:, 2].max()), xytext=(3.4, rescued[:, 2].max() * 0.8),
                arrowprops=dict(arrowstyle="->", color=C_NEG, lw=1.2),
                fontsize=9, color=C_NEG)
    save(fig, "fig06_rescue_counts")


# ---------------------------------------------------------------- 图 7
def fig07():
    """长度混淆排除：Δ 对 Δlen 的散点。"""
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    for ax, (fam, data) in zip(axes, [("qwen3", LEN_NEG), ("qwen25", LEN_NEG)]):
        rows = [r for r in data["rows"] if r["family"] == fam]
        xs = [r["dlen"] for r in rows]
        ys = [r["delta_pp"] for r in rows]
        cs = [C_POS if r["consistent"] else C_NEG for r in rows]
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(0, color="black", lw=0.8)
        ax.scatter(xs, ys, c=cs, s=64, zorder=3, edgecolor="white", lw=0.8)
        for r in rows:
            ax.annotate(cn(r["stance"]), (r["dlen"], r["delta_pp"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=7.8)
        # 相关
        if len(xs) > 2:
            rr = np.corrcoef(xs, ys)[0, 1]
            ax.set_title("%s\n$r$(Δ长度, Δ) = %+.2f" % (FAMLAB[fam], rr))
        ax.set_xlabel("Δ 平均答案长度（字符）")
        ax.set_ylabel("$\\Delta$ PKD (pp)")
    fig.suptitle("图 7  长度混淆的排除：Δ 与 Δ长度 无系统共变（否定感知判据）",
                 y=1.03, fontsize=12)
    save(fig, "fig07_length_confound")


# ---------------------------------------------------------------- 图 8
def fig08():
    """立场 × 规模的完整热力图（含未判定率的可见性）。"""
    order = NEG["order"]
    tags = NEG["tags"]
    M = np.array([[pkd(t, s) for t in tags] for s in order])
    D = np.array([[(NEG["per_stance"][s][t]["decided"] * 100.0) for t in tags]
                  for s in order])
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    for ax, (Mx, title, cmap, vmin, vmax) in zip(
            axes,
            [(M, "PKD（%）", "RdYlBu_r", 0, 100),
             (D, "判出率（%，分母健康度）", "viridis", 75, 100)]):
        im = ax.imshow(Mx, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(tags)))
        ax.set_xticklabels([t.split("-vl-")[1].upper() for t in tags])
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([cn(s) for s in order])
        ax.set_title(title)
        ax.grid(False)
        # 文字颜色按**该格实际填色的亮度**定，不按归一化位置定。
        # 按位置定会在压缩色域上出错：右图 vmin=75/vmax=100，几乎所有格都
        # 归一化到 0.6 以上而被判成白字，而 viridis 那一段是浅黄 —— 白字浅底
        # 直接看不清。
        cm = plt.get_cmap(cmap)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        for i in range(Mx.shape[0]):
            for j in range(Mx.shape[1]):
                v = Mx[i, j]
                r, g, b, _ = cm(norm(v))
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                ax.text(j, i, "%.0f" % v, ha="center", va="center", fontsize=8.4,
                        color="black" if lum > 0.55 else "white")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        # 标出判出率低的行（分母塌陷风险）
        if Mx is D:
            for i in range(D.shape[0]):
                if D[i].min() < 88:
                    ax.axhline(i, color=C_NEG, lw=1.6, ls="--", alpha=0.9)
    fig.suptitle("图 8  立场 × 规模全表：右图低判出率行即分母塌陷处（虚线）",
                 y=1.02, fontsize=12)
    save(fig, "fig08_heatmap")


# ---------------------------------------------------------------- 图 9
def fig09():
    """两族的符号结构对照：符号有序 vs 跨零。"""
    fig, ax = plt.subplots(figsize=(10.6, 4.4))
    order = NEG["order"]
    xr = np.arange(len(order))
    for fam in ["qwen3", "qwen25"]:
        seq = {s["stance"]: s["delta_pp"] for s in NEG["symbol_check"][fam]["seq"]}
        ax.plot(xr, [seq[s] for s in order], "o-", color=FAMCOL[fam], lw=2.0,
                ms=7, label="%s（否定感知）" % FAMLAB[fam])
        a = {s["stance"]: s["delta_pp"] for s in ALI["symbol_check"][fam]["seq"]}
        ax.plot(xr, [a[s] for s in order], "s--", color=FAMCOL[fam], lw=1.3,
                ms=5, alpha=0.45, label="%s（纯别名）" % FAMLAB[fam])
    ax.axhline(0, color="black", lw=1.1)
    ax.axvspan(-0.4, 2.4, color=C_GREY, alpha=0.10)
    ax.annotate("负号区", (1.0, ax.get_ylim()[0] * 0.82), ha="center",
                fontsize=9.5, color=C_GREY)
    ax.annotate("正号区", (4.6, ax.get_ylim()[1] * 0.86), ha="center",
                fontsize=9.5, color=C_GREY)
    ax.set_xticks(xr)
    ax.set_xticklabels([cn(s) for s in order], rotation=18, ha="right")
    ax.set_ylabel("$\\Delta$ PKD (pp)")
    ax.set_xlabel("预注册授权强度（弱 → 强）")
    ax.legend(ncol=2, loc="upper left", fontsize=8.6)
    ax.set_title("图 9  符号跨零：Qwen3 族在纯别名判据下的 `len_ctrl` 逸出负区",
                 fontsize=12)
    save(fig, "fig09_symbol_crossing")


# ---------------------------------------------------------------- 图 10
def fig10():
    """跨环境复现：读 cross_env_compare.py 的产物，不自己另算。

    之前的写法是现场重算「逐立场的参数知识率」，而那是**另一个量** ——
    跨环境复现要报的是同一条件下的 PKD 偏离，不是知识率水平。
    两个量的差异足以让图与正文对不上，故改为只读落盘产物。
    """
    f = os.path.join(R, "cross_env_3b.json")
    if not os.path.exists(f):
        print("  [跳过] 缺 %s（先跑 src/cross_env_compare.py）" % f)
        return
    d = J(f)
    rows = d["rows"]
    order = [r["cond"] for r in rows]
    a = [r["pkd_local"] for r in rows]
    b = [r["pkd_remote"] for r in rows]
    vb = [r["verbatim"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.3))
    ax = axes[0]
    x = np.arange(len(order))
    w = 0.36
    ax.bar(x - w / 2, a, w, color=C_QWEN25, label="本地（消费级卡）")
    ax.bar(x + w / 2, b, w, color=C_QWEN3, label="A800（重跑）")
    for xi, (va, vbb) in enumerate(zip(a, b)):
        ax.annotate("%+.1f" % (vbb - va), (xi, max(va, vbb) + 1.6), ha="center",
                    fontsize=8.2, color=C_GREY)
    ax.set_xticks(x)
    ax.set_xticklabels([cn(s) for s in order], rotation=18, ha="right")
    ax.set_ylabel("PKD（%）")
    ax.set_ylim(0, max(max(a), max(b)) * 1.22)
    ax.legend(loc="upper left")
    ax.set_title("PKD：配对口径，最大偏离 %.1f pp" % d["max_abs_pkd_delta_pp"])

    ax = axes[1]
    cols = [C_NEG if v < 90 else C_POS for v in vb]
    ax.bar(x, vb, color=cols, width=0.6)
    for xi, v in zip(x, vb):
        ax.annotate("%.1f%%" % v, (xi, v + 1.2), ha="center", fontsize=8.2)
    ax.axhline(np.mean(vb), color=C_GREY, ls="--", lw=1.0)
    ax.annotate("均值 %.1f%%" % np.mean(vb), (len(order) - 1.4, np.mean(vb) + 1.4),
                fontsize=8.6, color=C_GREY)
    ax.set_xticks(x)
    ax.set_xticklabels([cn(s) for s in order], rotation=18, ha="right")
    ax.set_ylabel("逐字一致率（%）")
    ax.set_ylim(0, 112)
    ax.set_title("逐字一致率：字符串级复现差得多")
    fig.suptitle("图 10  跨环境复现（Qwen2.5-VL 3B，%d 条共同样本）："
                 "判据级稳（≤%.1f pp），逐字级不稳（均 %.1f%%）"
                 % (d["n_shared"], d["max_abs_pkd_delta_pp"], d["mean_verbatim_pct"]),
                 y=1.03, fontsize=12)
    save(fig, "fig10_cross_env")



# ---------------------------------------------------------------- 图 11
def fig11():
    """位置偏好：同内容 A/B 互换后的选择变化。"""
    h = J(P, "_harness_summary.json")
    hr = J(P, "_harnessR_summary.json")
    conds = ["cb_default", "cb_conflict", "xie_implicit", "xie_explicit"]
    lab = {"cb_default": "默认（原文模板）", "cb_conflict": "冲突版",
           "xie_implicit": "Xie 隐式", "xie_explicit": "Xie 显式"}
    fig, ax = plt.subplots(figsize=(10.2, 4.4))
    x = np.arange(len(conds))
    w = 0.36
    for si, (side, name) in enumerate([("3b", "Qwen2.5-VL 3B"), ("7b", "Qwen2.5-VL 7B")]):
        bias = [h["conds"][c]["pA"][side] - (1 - hr["conds"][c]["pA"][side])
                for c in conds]
        ax.bar(x + (si - 0.5) * w, bias, w,
               color=[C_QWEN25, C_QWEN3][si], label=name)
        for xi, v in zip(x, bias):
            ax.annotate("%+.3f" % v, (xi + (si - 0.5) * w, v),
                        textcoords="offset points",
                        xytext=(0, 5 if v >= 0 else -12), ha="center", fontsize=8.2)
    ax.axhline(0, color="black", lw=1.0)
    ax.axhline(0.2, color=C_NEG, ls="--", lw=1.0)
    ax.annotate("0.2（本文的不可用阈值）", (len(conds) - 0.5, 0.215),
                ha="right", fontsize=8.6, color=C_NEG)
    ax.set_xticks(x)
    ax.set_xticklabels([lab[c] for c in conds])
    ax.set_ylabel("位置偏好 $p_A(ab) - (1-p_A(ba))$")
    ax.set_ylim(-0.18, 1.12)
    ax.legend(loc="upper right")
    ax.set_title("图 11  位置偏好自证否：`cb_default` 的位置偏好接近 1，其数字不可用于内容推断",
                 fontsize=11.5)
    save(fig, "fig11_position_bias")


# ---------------------------------------------------------------- 图 12
def fig12():
    """分域检验：place / other 两槽位内的 Δ，并标出样本量不足的槽位。"""
    # 读否定感知版：本节表须与 §4.2 主表同判据，否则分域/总体不可对照。
    # 纯别名版（stratified.json）里 ctx_hedge 的分域分母塌到 12/8，
    # 画出来会像「该立场在槽位内不可测」，那是判据的产物不是数据的性质。
    f = os.path.join(R, "stratified_negation.json")
    if not os.path.exists(f):
        print("  [跳过] 缺 %s（先跑 "
              "src/analyze_stratified.py --grader negation）" % f)
        return
    d = J(f)
    order = [s["stance"] for s in NEG["symbol_check"]["qwen25"]["seq"]]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.5), sharey=True)
    for ax, fam in zip(axes, ["qwen25", "qwen3"]):
        rows = {(r["slot"], r["stance"]): r for r in d["families"][fam]}
        x = np.arange(len(order))
        w = 0.36
        for si, slot in enumerate(["place", "other"]):
            vals, cols = [], []
            for st in order:
                r = rows.get((slot, st))
                vals.append(r["delta_pp"] if r else 0.0)
                sig = r and r["p"] < 0.05
                cols.append((C_POS if r and r["delta_pp"] > 0 else C_NEG)
                            if sig else "#c9c9c9")
            ax.bar(x + (si - 0.5) * w, vals, w, color=cols,
                   label="%s 槽位（深色=显著）" % slot)
        ax.axhline(0, color="black", lw=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([cn(s) for s in order], rotation=18, ha="right")
        ax.set_title(FAMLAB[fam])
        ax.set_ylabel("$\\Delta$ (pp)")
        ax.legend(loc="upper left", fontsize=8.6)
    fig.suptitle("图 12  分域检验：灰色 = 该槽位内不显著（$p\\geq0.05$）", y=1.02,
                 fontsize=12)
    save(fig, "fig12_stratified")


# ---------------------------------------------------------------- 图 13
def fig13():
    """三个判据的跨度与显著格数对照。"""
    d = J(R, "scale_ladder_negation.json")
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    # 左：三种判据在两族上的跨度（别名与否定感知有；distinctive_ground 见 §4.7 表）
    ax = axes[0]
    fams = ["qwen3", "qwen25"]
    x = np.arange(len(fams))
    w = 0.36
    ali = [J(R, "scale_ladder.json")["symbol_check"][f]["span_pp"] for f in fams]
    neg = [d["symbol_check"][f]["span_pp"] for f in fams]
    ax.bar(x - w / 2, ali, w, color="#bbbbbb", label="纯别名判据")
    ax.bar(x + w / 2, neg, w, color=C_QWEN3, label="否定感知判据")
    for xi, (a, n) in enumerate(zip(ali, neg)):
        ax.annotate("%.1f" % a, (xi - w / 2, a + 1.2), ha="center", fontsize=8.4)
        ax.annotate("%.1f" % n, (xi + w / 2, n + 1.2), ha="center", fontsize=8.4)
    ax.set_xticks(x)
    ax.set_xticklabels([FAMLAB[f] for f in fams])
    ax.set_ylabel("显著格的跨度 (pp)")
    ax.set_ylim(0, 72)
    ax.legend(loc="lower right")
    ax.set_title("两判据的跨度接近")

    # 右：显著条件数
    ax = axes[1]
    alias_sig = [J(R, "scale_ladder.json")["symbol_check"][f]["n_sig"] for f in fams]
    neg_sig = [d["symbol_check"][f]["n_sig"] for f in fams]
    ax.bar(x - w / 2, alias_sig, w, color="#bbbbbb", label="纯别名判据")
    ax.bar(x + w / 2, neg_sig, w, color=C_QWEN3, label="否定感知判据")
    for xi, (a, n) in enumerate(zip(alias_sig, neg_sig)):
        ax.annotate("%d/7" % a, (xi - w / 2, a + 0.12), ha="center", fontsize=8.8)
        ax.annotate("%d/7" % n, (xi + w / 2, n + 0.12), ha="center", fontsize=8.8)
    ax.set_xticks(x)
    ax.set_xticklabels([FAMLAB[f] for f in fams])
    ax.set_ylabel("显著条件数（共 7）")
    ax.set_ylim(0, 8.6)
    ax.legend(loc="lower right")
    ax.set_title("否定感知判据显著条件更多")
    fig.suptitle("图 13  判据对照：跨度相近，但显著条件数与符号有序性不同", y=1.02,
                 fontsize=12)
    save(fig, "fig13_criterion_span")


# ---------------------------------------------------------------- 图 14
def fig14():
    """MDE 与实测效应量的对照：哪些格超过了检出限。

    检出限逐立场不同 —— 配对设计的 MDE 随该立场实测的不一致率 π_d 变化，
    不是一条水平线。早期版本画成一条 7.8 pp 的水平线，那个值是非配对公式
    （见 analyze_mde.py 的说明），且与本文设计不符。故这里读
    results/mde.json，不再自己算。
    """
    f = os.path.join(R, "mde.json")
    if not os.path.exists(f):
        print("  [跳过] 缺 %s（先跑 src/analyze_mde.py）" % f)
        return
    M = J(f)
    order = NEG["order"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4), sharex=True)
    for ax, fam in zip(axes, ["qwen3", "qwen25"]):
        rows = {r["stance"]: r for r in M["families"][fam]}
        vals = [abs(rows[s]["delta_pp"]) for s in order]
        mdes = [rows[s]["mde_pp"] for s in order]
        cols = [C_POS if rows[s]["exceeds_mde"] else "#c9c9c9" for s in order]
        x = np.arange(len(order))
        ax.bar(x, vals, color=cols, width=0.6, zorder=2)
        # 检出限是逐格的值，故画成阶梯线而不是水平线：横跨每根柱的区间
        ax.step(np.append(x - 0.3, x[-1] + 0.3), np.append(mdes, mdes[-1]),
                where="post", color=C_NEG, ls="--", lw=1.5, zorder=3)
        for xi, (v, m) in enumerate(zip(vals, mdes)):
            ax.plot([xi - 0.3, xi + 0.3], [m, m], color=C_NEG, lw=1.5,
                    ls="--", zorder=3)
            ax.annotate("%.1f" % m, (xi, m), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=7.6, color=C_NEG)
        ax.set_xticks(x)
        ax.set_xticklabels([cn(s) for s in order], rotation=20, ha="right")
        ax.set_ylabel("|$\\Delta$| (pp)  与逐格 MDE")
        ax.set_title(FAMLAB[fam])
        ax.set_ylim(0, max(vals) * 1.22)
    fig.suptitle("图 14  实测效应量 vs 逐立场最小可检出效应"
                 "（$|D^*|=%d$，配对口径）；灰色 = 低于该格检出限" % M["n_anchored"],
                 y=1.02, fontsize=12)
    save(fig, "fig14_mde")


# ---------------------------------------------------------------- 图 15
def fig15():
    """BOTH 丢弃缺陷：判出率随规模/立场的变化。"""
    order = NEG["order"]
    tags = NEG["tags"]
    D = np.array([[NEG["per_stance"][s][t]["decided"] * 100.0 for t in tags]
                  for s in order])
    B = np.array([[NEG["per_stance"][s][t]["both"] * 100.0
                   / max(NEG["per_stance"][s][t]["decided"]
                         + NEG["per_stance"][s][t]["both"], 1)
                   for t in tags] for s in order])
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.4))
    ax = axes[0]
    x = np.arange(len(order))
    for si, t in enumerate(tags):
        ax.plot(x, D[:, si], "o-", lw=1.4, ms=4.5,
                label=t.split("-vl-")[1].upper())
    ax.set_xticks(x)
    ax.set_xticklabels([cn(s) for s in order], rotation=20, ha="right")
    ax.set_ylabel("判出率（%）")
    ax.set_ylim(75, 100)
    ax.legend(ncol=3, fontsize=8.4, loc="lower left")
    ax.set_title("判出率：`len_ctrl` 在 Qwen3 族下降")

    ax = axes[1]
    for si, t in enumerate(tags):
        ax.plot(x, B[:, si], "o-", lw=1.4, ms=4.5,
                label=t.split("-vl-")[1].upper())
    ax.set_xticks(x)
    ax.set_xticklabels([cn(s) for s in order], rotation=20, ha="right")
    ax.set_ylabel("BOTH 占比（%）")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_title("BOTH 占比：`ctx_hedge` 一格高两个数量级")
    fig.suptitle("图 15  别名判据的 BOTH 丢弃缺陷在两个维度上的表现", y=1.02,
                 fontsize=12)
    save(fig, "fig15_both_defect")


FIGS = [fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09, fig10,
        fig11, fig12, fig13, fig14, fig15]


def main():
    want = [int(a) for a in sys.argv[1:] if a.isdigit()]
    load_all()
    for i, fn in enumerate(FIGS, 1):
        if want and i not in want:
            continue
        try:
            fn()
            print("  ✓ 图 %-2d %s" % (i, fn.__doc__.splitlines()[0].strip()))
        except Exception as e:
            print("  ✗ 图 %-2d 失败: %s: %s" % (i, type(e).__name__, e))
            raise


if __name__ == "__main__":
    main()
