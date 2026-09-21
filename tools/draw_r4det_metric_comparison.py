#!/usr/bin/env python3
"""R4Det 指标对比图：GRU 时序基线 vs 主模型（N4 RSSM + head-v2）。

对比 6 项指标 x 2 个模型 + 6 个差值 = 18 个数：
  - 四类 AP：Car(strict) / Truck(strict) / Pedestrian(loose) / Cyclist(loose)
  - Overall 3D moderate / Overall BEV moderate

四类不是随便选的：它们正是 Overall_3D_moderate 的构成项
（kitti_utils/eval.py: Overall = mean(Ped_loose, Cyc_loose, Car_strict, Truck_strict)），
所以前四行求平均 == Overall 3D 行，表内可自校验。

数据来源：
  - baseline_temporal (GRU TemporalDeformableFusion, N=2, 18e) BEST ep12：
    docs/training_runs_full.md §3 / §7「逐类别 BEST 对比 (loose/strict)」行 baseline_temporal。
  - 主模型 (pretrained backbone + N=4 Motion-Aligned RSSM + head-v2, 24e) 已存最优 ep14：
    work_dirs/rssm_N4_2x4_24e_pretrained_v2_head/*.log.json 的 mode=val 记录。

渲染后自带几何自检（字形覆盖 / 文本出画布 / 文本互相压盖 / 表格列不越界），
因为本环境无法人工看图，布局正确性只能靠断言保证。

输出 assets/ 下 svg + png + pdf。
"""
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle

# ---------------------------------------------------------------- fonts
CJK = ["/home/lurui/.local/share/fonts/NotoSansCJKsc-Regular.otf",
       "/home/lurui/.local/share/fonts/NotoSansCJKsc-Bold.otf"]
for f in CJK:
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------- palette (matches draw_r4det_two_architectures.py)
BASE_C, BASE_E = "#6C8EBF", "#3D6491"   # baseline blue
MAIN_C, MAIN_E = "#E8A33D", "#B87A18"   # main model orange
TXT, SUB = "#1A1A1A", "#666666"
GRID = "#DDDDDD"
BAND = "#F4F6F9"
POS, NEG = "#2E7D32", "#C62828"

# ---------------------------------------------------------------- data
# (显示名, 口径, baseline_temporal ep12, 主模型最高单点 seed_2 ep14, 是否 Overall 行)
ROWS = [
    ("Car", "strict", 32.82, 51.35, False),
    ("Truck", "strict", 29.91, 30.76, False),
    ("Pedestrian", "loose", 26.64, 31.32, False),
    ("Cyclist", "loose", 48.63, 50.09, False),
    ("Overall 3D moderate", "", 34.50, 40.88, True),
    ("Overall BEV moderate", "", 41.68, 49.85, True),
]

# 自校验：前四行（Car_s / Truck_s / Ped_l / Cyc_l）均值必须等于 Overall 3D 行
for _mi, _lbl in ((2, "baseline_temporal"), (3, "主模型")):
    _c = sum(r[_mi] for r in ROWS[:4]) / 4
    assert abs(_c - ROWS[4][_mi]) < 0.02, f"{_lbl} compose {_c} != {ROWS[4][_mi]}"

N_NUMBERS = len(ROWS) * 3  # 6 指标 × (2 模型 + 1 差值)
assert N_NUMBERS == 18, N_NUMBERS

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(15.5, 9.0), dpi=200)
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.24,
                      left=0.235, right=0.965, top=0.845, bottom=0.128)

XMAX = 88.0
Y_TOP = len(ROWS) - 0.10          # 顶部留出 Δ 列标题的位置

# ================================================================ panel A: bars
ax = fig.add_subplot(gs[0])
ax.set_facecolor("white")
n = len(ROWS)
ys = list(range(n))[::-1]          # 第 0 行画在最上方
h = 0.30
GAPY = 0.215                       # 同组两根柱的中心偏移：Truck 两侧数值接近，
                                   # 偏移必须 > 半个字高，否则两个数值标签会叠在一起
bar_texts = []

for i, (name, thr, b, m, is_ov) in enumerate(ROWS):
    y = ys[i]
    if is_ov:                       # Overall 行整条底色带（跨满整个 x 范围）
        ax.add_patch(Rectangle((-0.8, y - 0.5), XMAX + 1.6, 1.0,
                               facecolor=BAND, edgecolor="none", zorder=0))
    ax.barh(y + GAPY, b, height=h, color=BASE_C, edgecolor=BASE_E,
            linewidth=1.1, zorder=3)
    ax.barh(y - GAPY, m, height=h, color=MAIN_C, edgecolor=MAIN_E,
            linewidth=1.1, zorder=3)

    bar_texts.append(ax.text(b + 1.0, y + GAPY, f"{b:.2f}", va="center",
                             ha="left", fontsize=11, color=BASE_E,
                             fontweight="bold", zorder=4))
    bar_texts.append(ax.text(m + 1.0, y - GAPY, f"{m:.2f}", va="center",
                             ha="left", fontsize=11, color=MAIN_E,
                             fontweight="bold", zorder=4))

    d = m - b
    col = POS if d >= 0 else NEG
    sign = "+" if d >= 0 else "−"
    bar_texts.append(ax.text(XMAX - 0.5, y, f"{sign}{abs(d):.2f}", va="center",
                             ha="right", fontsize=12.5, color=col,
                             fontweight="bold", zorder=4))

# Overall 与四类之间的分隔线
ax.axhline((ys[3] + ys[4]) / 2, color="#BBBBBB", lw=1.2, ls=(0, (5, 4)), zorder=2)

ax.set_yticks(ys)
ax.set_yticklabels([f"{nm}  ({th})" if th else nm for nm, th, *_ in ROWS],
                   fontsize=13)
for tick, row in zip(ax.get_yticklabels(), ROWS):
    if row[4]:
        tick.set_fontweight("bold")
    tick.set_color(TXT)
ax.set_xlim(0, XMAX)
ax.set_ylim(-0.60, Y_TOP)
ax.set_xlabel("AP40 / %  （moderate 难度）", fontsize=12, color=SUB, labelpad=8)
ax.xaxis.grid(True, color=GRID, lw=0.9, zorder=1)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="both", length=0, colors=SUB, labelsize=11.5)

ax.text(XMAX - 0.5, Y_TOP - 0.22, "Δ（主 − 基线）", ha="right", va="center",
        fontsize=11.5, color=SUB, fontweight="bold")

handles = [Rectangle((0, 0), 1, 1, facecolor=BASE_C, edgecolor=BASE_E),
           Rectangle((0, 0), 1, 1, facecolor=MAIN_C, edgecolor=MAIN_E)]
legend = fig.legend(handles,
                    ["baseline_temporal（GRU 时序基线）",
                     "主模型（N4 RSSM + head-v2）"],
                    loc="upper right", bbox_to_anchor=(0.965, 0.908),
                    ncol=2, frameon=False, fontsize=12,
                    handlelength=1.5, handleheight=1.0, borderpad=0.1,
                    columnspacing=2.4)

# ================================================================ panel B: table
ax2 = fig.add_subplot(gs[1])
ax2.axis("off")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

# 列布局：所有列右边界必须 <= 1.0，否则 Δ 列会捅出表格右边框
COLX = [0.008, 0.430, 0.640, 0.855]
COLW = [0.415, 0.200, 0.205, 0.140]
for x, w in zip(COLX, COLW):
    assert x + w <= 1.0 + 1e-9, f"列越界: x={x} w={w}"
HEAD = ["指标", "baseline_temporal\n(GRU, ep12)",
        "主模型 最高单点\n(seed_2 ep14, 已存)", "Δ 主 − 基线"]
HDR_BOX, HDR_TXT, CELL_TXT = [], [], []

row_h, head_h, y0 = 0.106, 0.180, 0.905
TABLE_R = COLX[0] + sum(COLW)

# 表头
for x, w, txt in zip(COLX, COLW, HEAD):
    box = FancyBboxPatch((x + 0.004, y0 - head_h + 0.010), w - 0.008,
                         head_h - 0.020,
                         boxstyle="round,pad=0.004,rounding_size=0.012",
                         facecolor="#EBEFF5", edgecolor="#CBD5E1", lw=1.0)
    ax2.add_patch(box)
    HDR_BOX.append((box, x, x + w))
    HDR_TXT.append(ax2.text(x + w / 2, y0 - head_h / 2 + 0.004, txt,
                            ha="center", va="center", fontsize=11.5, color=TXT,
                            fontweight="bold", linespacing=1.30))

for i, (name, thr, b, m, is_ov) in enumerate(ROWS):
    yt = y0 - head_h - (i + 1) * row_h
    yc = yt + row_h / 2
    bg = "#F0F4FA" if is_ov else ("#FCFCFC" if i % 2 else "white")
    ax2.add_patch(Rectangle((COLX[0], yt + 0.006), TABLE_R - COLX[0],
                            row_h - 0.012, facecolor=bg, edgecolor="#E5E7EB",
                            lw=0.8))
    lw = "bold" if is_ov else "normal"
    label = f"{name}（{thr}）" if thr else name
    CELL_TXT.append((ax2.text(COLX[0] + 0.016, yc, ("★  " if is_ov else "     ") + label,
                              ha="left", va="center", fontsize=12.5, color=TXT,
                              fontweight=lw), 0))
    CELL_TXT.append((ax2.text(COLX[1] + COLW[1] - 0.016, yc, f"{b:.2f}", ha="right",
                              va="center", fontsize=13, color=BASE_E,
                              fontweight=lw, family="DejaVu Sans"), 1))
    CELL_TXT.append((ax2.text(COLX[2] + COLW[2] - 0.016, yc, f"{m:.2f}", ha="right",
                              va="center", fontsize=13, color=MAIN_E,
                              fontweight=lw, family="DejaVu Sans"), 2))
    d = m - b
    col = POS if d >= 0 else NEG
    sign = "+" if d >= 0 else "−"
    CELL_TXT.append((ax2.text(COLX[3] + COLW[3] - 0.016, yc, f"{sign}{abs(d):.2f}",
                              ha="right", va="center", fontsize=13, color=col,
                              fontweight="bold", family="DejaVu Sans"), 3))
    if abs(d) >= 5:                     # 显著差值加底色（严格限制在 Δ 列内）
        ax2.add_patch(Rectangle((COLX[3] + 0.012, yt + 0.019),
                                COLW[3] - 0.030, row_h - 0.038, facecolor=col,
                                alpha=0.11, edgecolor="none"))

# ================================================================ title / footnote
TITLE = fig.text(0.038, 0.972, "R4Det：GRU 时序基线 vs 主模型 —— Overall 与四类 AP 对比",
                 ha="left", va="top", fontsize=19, color=TXT, fontweight="bold")
SUBTITLE = fig.text(
    0.038, 0.936,
    "主模型 = 预训练 backbone + N=4 Motion-Aligned RSSM + head-v2；两行各取自身 BEST 单点（baseline ep12 / 主模型 seed_2 ep14，均为已存权重）",
    ha="left", va="top", fontsize=11.5, color=SUB)
FOOT = [
    fig.text(0.038, 0.100,
             "注：四类即 Overall_3D_moderate 的构成项 —— Overall_3D = mean(Ped_loose, Cyc_loose, Car_strict, Truck_strict)，"
             "故前四行均值恰好等于 Overall 3D 行（基线 34.50 / 主模型 40.88），可自行校验。",
             ha="left", va="bottom", fontsize=10.2, color="#555555"),
    fig.text(0.038, 0.074,
             "主模型口径：此处取已存最高单点 40.88（seed_2 ep14）；三 seed BEST 均值 ± std = 40.42 ± 0.51；"
             "原 Run 10 峰值 40.60（ep11）因 interval=2 未落盘，其已存最优为 39.65（ep14）。",
             ha="left", va="bottom", fontsize=10.2, color="#888888"),
    fig.text(0.038, 0.048,
             "口径：Car / Truck 取 strict，Pedestrian / Cyclist 取 loose（与 Overall 混合口径一致）。",
             ha="left", va="bottom", fontsize=10.2, color="#888888"),
    fig.text(0.038, 0.022,
             "数据：baseline_temporal = docs/training_runs_full.md §3（原始日志已删）；"
             "主模型 = work_dirs/run10_headv2_multiseed/seed_2 日志 ep14。",
             ha="left", va="bottom", fontsize=10.2, color="#888888"),
]

# ================================================================ 几何自检
# 缺字形警告必须变成硬错误：DejaVu 不含 CJK，一旦字体回退就会静默画成豆腐块。
warnings.filterwarnings("error", message=".*[Gg]lyph.*")
fig.canvas.draw()
R = fig.canvas.get_renderer()
W, H = fig.canvas.get_width_height()


def bbox(t):
    return t.get_window_extent(renderer=R)


def overlap(a, b, pad=1.0):
    return (a.x0 < b.x1 - pad and b.x0 < a.x1 - pad and
            a.y0 < b.y1 - pad and b.y0 < a.y1 - pad)


# 1) 所有文本必须在画布内
for t in fig.findobj(matplotlib.text.Text):
    if not t.get_text().strip():
        continue
    bb = bbox(t)
    assert -1 <= bb.x0 and bb.x1 <= W + 1 and -1 <= bb.y0 and bb.y1 <= H + 1, \
        f"文本出画布: {t.get_text()[:40]!r} bbox={bb}"

# 2) 柱条数值标签之间、以及与 Δ 列之间不得压盖
for i in range(len(bar_texts)):
    for j in range(i + 1, len(bar_texts)):
        assert not overlap(bbox(bar_texts[i]), bbox(bar_texts[j])), \
            f"A 面板文本压盖: {bar_texts[i].get_text()!r} vs {bar_texts[j].get_text()!r}"

# 3) 图例不得压到柱条或 A 面板文字或副标题
lg = legend.get_window_extent(renderer=R)
for t in bar_texts + [SUBTITLE, TITLE] + FOOT:
    assert not overlap(lg, bbox(t)), f"图例压盖文本: {t.get_text()[:30]!r}"
for p in ax.patches:
    assert not overlap(lg, p.get_window_extent(renderer=R)), "图例压盖柱条"

# 4) 表头文字必须装进各自表头框
for (box, x0, x1), t in zip(HDR_BOX, HDR_TXT):
    bb, pb = bbox(t), box.get_window_extent(renderer=R)
    assert pb.x0 - 3 <= bb.x0 and bb.x1 <= pb.x1 + 3, \
        f"表头文字溢出框: {t.get_text()[:30]!r} text={bb} box={pb}"
    assert pb.y0 - 3 <= bb.y0 and bb.y1 <= pb.y1 + 3, \
        f"表头文字溢出框(纵向): {t.get_text()[:30]!r}"

# 5) 表格单元格文字不得越出自己的列
for t, ci in CELL_TXT:
    bb = bbox(t)
    a0 = ax2.transData.transform((COLX[ci], 0))[0]
    a1 = ax2.transData.transform((COLX[ci] + COLW[ci], 0))[0]
    assert a0 - 2 <= bb.x0 and bb.x1 <= a1 + 2, \
        f"单元格越列: col={ci} {t.get_text()!r} text=[{bb.x0:.0f},{bb.x1:.0f}] col=[{a0:.0f},{a1:.0f}]"

# 6) 表格整体不得越出坐标轴框
tb_left = ax2.transData.transform((COLX[0], 0))[0]
tb_right = ax2.transData.transform((TABLE_R, 0))[0]
ab = ax2.get_window_extent(renderer=R)
assert tb_left >= ab.x0 - 1 and tb_right <= ab.x1 + 1, \
    f"表格越出轴框: table=[{tb_left:.0f},{tb_right:.0f}] axes=[{ab.x0:.0f},{ab.x1:.0f}]"

print("几何自检通过：字形覆盖 / 画布边界 / 文本压盖 / 表头与单元格边界")

# ================================================================ save
out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
os.makedirs(out_dir, exist_ok=True)
stem = os.path.join(out_dir, "r4det_baseline_vs_main_metrics")
for ext in ("svg", "png", "pdf"):
    fig.savefig(f"{stem}.{ext}", facecolor="white", bbox_inches=None)
    print("wrote", f"{stem}.{ext}")

print(f"\nrows={len(ROWS)}  numbers={N_NUMBERS}")
for name, thr, b, m, _ in ROWS:
    print(f"  {name:22s} {thr:6s} base={b:6.2f}  main={m:6.2f}  d={m - b:+6.2f}")
