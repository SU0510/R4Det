#!/usr/bin/env python3
"""Render the no_temporal-vs-N4 (main model) metric tables as a PNG/PDF.

Server-side rendering: no headless browser is available, so the tables are
drawn directly with matplotlib.  Chinese text needs Noto Sans CJK SC, which is
registered below; a missing glyph is promoted to a hard error.
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------- fonts
CJK = ["/home/lurui/.local/share/fonts/NotoSansCJKsc-Regular.otf",
       "/home/lurui/.local/share/fonts/NotoSansCJKsc-Bold.otf"]
for f in CJK:
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"

# --- font size bump: every text size goes up by FONT_DELTA points -------------
FONT_DELTA = 2


def fs(x):
    return x + FONT_DELTA


HDR_BG = "#2f5597"
HDR_FG = "white"
OV_BG = "#eef5ff"
ALT_BG = "#f7f9fc"
EDGE = "#b8c4d6"
TXT = "#1f2430"
DIM = "#6b7280"
HL = "#b05a00"

# ---------------------------------------------------------------- data
# (label, no_temporal(seed0), N4 seed0, Δ0, N4 seed2, Δ2)
BEST = [
    ("Overall 3D moderate",          35.59, 39.88, "+4.29", 40.88, "+5.29"),
    ("Overall BEV moderate",         42.23, 47.70, "+5.47", 49.85, "+7.62"),
    ("Car 3D moderate strict",       47.86, 49.98, "+2.11", 51.35, "+3.49"),
    ("Truck 3D moderate strict",     25.53, 30.43, "+4.90", 30.76, "+5.23"),
    ("Pedestrian 3D moderate loose", 27.07, 28.64, "+1.57", 31.32, "+4.25"),
    ("Cyclist 3D moderate loose",    41.89, 50.47, "+8.58", 50.09, "+8.20"),
]
WIN = [
    ("Overall 3D moderate",          34.65, 38.39, "+3.74", 39.40, "+4.75"),
    ("Overall BEV moderate",         41.62, 46.52, "+4.90", 48.33, "+6.71"),
    ("Car 3D moderate strict",       46.16, 47.80, "+1.64", 48.98, "+2.81"),
    ("Truck 3D moderate strict",     23.48, 28.21, "+4.73", 30.04, "+6.56"),
    ("Pedestrian 3D moderate loose", 26.67, 28.92, "+2.24", 30.58, "+3.91"),
    ("Cyclist 3D moderate loose",    42.28, 48.63, "+6.34", 47.99, "+5.71"),
]

# column widths as fractions of the usable width, and left edges
WFRAC = [0.300, 0.155, 0.150, 0.095, 0.150, 0.095]  # sums to 0.945
COLX = [0.0]
for w in WFRAC:
    COLX.append(COLX[-1] + w)

HEAD_TOP = ["指标", "No-Temporal\n(seed0)", "", "", "", ""]
HEAD_TOP_SPAN = [(0, 1), (1, 1), (2, 2), (4, 2)]
HEAD_TOP_TXT = ["指标", "No-Temporal\n(seed0)", "主模型 seed0\n（种子对齐）",
                "主模型 seed2\n（最高单点）"]
HEAD_SUB = ["", "", "N4 RSSM", "Δ", "N4 RSSM", "Δ"]


def cell(ax, x0, x1, y0, y1, text, *, bg=None, fg=TXT, bold=False,
         size=10.5, align="center", boxes=None):
    if bg:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=bg,
                               edgecolor=EDGE, linewidth=0.8, zorder=1))
    else:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none",
                               edgecolor=EDGE, linewidth=0.8, zorder=1))
    if not text:
        return None
    pad = 0.008
    if align == "center":
        x, ha = (x0 + x1) / 2, "center"
    else:
        x, ha = x0 + pad, "left"
    t = ax.text(x, (y0 + y1) / 2, text, ha=ha, va="center", fontsize=fs(size),
                color=fg, fontweight="bold" if bold else "normal", zorder=3,
                linespacing=1.25)
    if boxes is not None:
        boxes.append((t, (x0, x1, y0, y1)))
    return t


def header(ax, y_top, rh, boxes):
    y1, y0 = y_top, y_top - rh                      # top row
    for (cs, span), txt in zip(HEAD_TOP_SPAN, HEAD_TOP_TXT):
        cell(ax, COLX[cs], COLX[cs + span], y0, y1, txt,
             bg=HDR_BG, fg=HDR_FG, bold=True, size=10.0, boxes=boxes)
    y1b, y0b = y0, y0 - rh                          # sub row
    for i, txt in enumerate(HEAD_SUB):
        cell(ax, COLX[i], COLX[i + 1], y0b, y1b, txt,
             bg=HDR_BG, fg=HDR_FG, bold=True, size=9.5, boxes=boxes)
    return y0b


def body(ax, y_top, rh, rows, boxes):
    y = y_top
    for i, (lab, nt, s0, d0, s2, d2) in enumerate(rows):
        y1, y0 = y, y - rh
        overall = lab.startswith("Overall")
        bg = OV_BG if overall else (ALT_BG if i % 2 else None)
        cell(ax, COLX[0], COLX[1], y0, y1, lab, bg=bg, align="left", bold=overall,
             size=10.5, boxes=boxes)
        cell(ax, COLX[1], COLX[2], y0, y1, f"{nt:.2f}", bg=bg, size=10.5, boxes=boxes)
        cell(ax, COLX[2], COLX[3], y0, y1, f"{s0:.2f}", bg=bg, size=10.5,
             bold=overall, boxes=boxes)
        cell(ax, COLX[3], COLX[4], y0, y1, d0, bg=bg, fg=HL, bold=True,
             size=10.5, boxes=boxes)
        cell(ax, COLX[4], COLX[5], y0, y1, f"{s2:.2f}", bg=bg, size=10.5,
             bold=overall, boxes=boxes)
        cell(ax, COLX[5], COLX[6], y0, y1, d2, bg=bg, fg=HL, bold=True,
             size=10.5, boxes=boxes)
        y = y0
    return y


def seed_table(ax, y_top, rh, boxes):
    head = ["seed", "BEST Overall 3D", "@ep", "BEST Overall BEV", "窗口 3D", "窗口 BEV"]
    wf = [0.12, 0.20, 0.10, 0.20, 0.16, 0.165]
    cx = [0.0]
    for w in wf:
        cx.append(cx[-1] + w)
    y1, y0 = y_top, y_top - rh
    for i, txt in enumerate(head):
        cell(ax, cx[i], cx[i + 1], y0, y1, txt, bg=HDR_BG, fg=HDR_FG, bold=True,
             size=10.0, boxes=boxes)
    rows = [("seed0", "39.88", "ep16", "47.70", "38.39", "46.52"),
            ("seed1", "40.51", "ep15", "48.13", "39.20", "46.85"),
            ("seed2（最高）", "40.88", "ep14", "49.85", "39.40", "48.33"),
            ("均值", "40.42 ± 0.51", "ep14–16", "48.56", "39.00", "47.23")]
    y = y0
    for i, row in enumerate(rows):
        y1r, y0r = y, y - rh
        last = i == len(rows) - 1
        bg = OV_BG if last else (ALT_BG if i % 2 else None)
        for j, txt in enumerate(row):
            cell(ax, cx[j], cx[j + 1], y0r, y1r, txt, bg=bg,
                 bold=last or j == 0, size=10.0, boxes=boxes)
        y = y0r
    return y


# ---------------------------------------------------------------- build
warnings.filterwarnings("error", message=".*[Gg]lyph.*")

fig = plt.figure(figsize=(13.6, 10.2), dpi=200)
fig.patch.set_facecolor("white")
ax = fig.add_axes([0.045, 0.03, 0.925, 0.95])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

rh = 0.0400
GAP = 0.038
boxes = []

labels = []
labels.append(ax.text(0, 0.994, "No-Temporal vs 主模型（N4 Motion-Aligned RSSM）", ha="left",
        va="bottom", fontsize=fs(14.5), fontweight="bold", color=TXT))
labels.append(ax.text(0, 0.975, "同栈单变量：两份 config 差异仅 temporal_fusion（None ↔ MotionAlignedRSSMFusion）",
        ha="left", va="bottom", fontsize=fs(10.0), color=DIM))

y = 0.935
labels.append(ax.text(0, y + 0.004, "表 1　BEST 口径（论文 / 汇报用）", ha="left", va="bottom",
        fontsize=fs(11.5), fontweight="bold", color=TXT))
y = header(ax, y, rh, boxes)
y = body(ax, y, rh, BEST, boxes)

y -= GAP
labels.append(ax.text(0, y + 0.004, "表 2　窗口口径（ep12-16 均值，实验室选型用）", ha="left",
        va="bottom", fontsize=fs(11.5), fontweight="bold", color=TXT))
y = header(ax, y, rh, boxes)
y = body(ax, y, rh, WIN, boxes)

y -= GAP
labels.append(ax.text(0, y + 0.004, "主模型三 seed（同一份 config）", ha="left", va="bottom",
        fontsize=fs(11.5), fontweight="bold", color=TXT))
y = seed_table(ax, y, rh, boxes)

fig.canvas.draw()
ren = fig.canvas.get_renderer()
fb = fig.bbox

# --- assertions (substitute for eyeballing the PNG)
for t, (x0, x1, y0, y1) in boxes:
    bb = t.get_window_extent(renderer=ren)
    assert bb.x0 >= -1 and bb.y0 >= -1, f"text outside canvas (lo): {t.get_text()!r}"
    assert bb.x1 <= fb.x1 + 1 and bb.y1 <= fb.y1 + 1, f"text outside canvas (hi): {t.get_text()!r}"
    # horizontal containment within its own cell
    inv = ax.transData.inverted()
    (cx0, _), (cx1, _) = inv.transform((bb.x0, 0)), inv.transform((bb.x1, 0))
    assert cx0 >= x0 - 0.004, f"text left of cell: {t.get_text()!r}"
    assert cx1 <= x1 + 0.004, f"text right of cell: {t.get_text()!r}"

for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        a = boxes[i][0].get_window_extent(renderer=ren)
        b = boxes[j][0].get_window_extent(renderer=ren)
        if a.overlaps(b):
            ov = min(a.x1, b.x1) - max(a.x0, b.x0)
            assert ov <= 1.0, (f"text overlap: {boxes[i][0].get_text()!r} vs "
                               f"{boxes[j][0].get_text()!r}")

# --- title / sub-title / table labels: inside canvas, clear of every cell
for lb in labels:
    lbb = lb.get_window_extent(renderer=ren)
    tag = lb.get_text()[:24]
    assert lbb.x0 >= -1 and lbb.y0 >= -1, f"label outside canvas (lo): {tag!r}"
    assert lbb.x1 <= fb.x1 + 1 and lbb.y1 <= fb.y1 + 1, f"label outside canvas (hi): {tag!r}"
    for t, _ in boxes:
        tb = t.get_window_extent(renderer=ren)
        if lbb.overlaps(tb):
            ov = min(lbb.x1, tb.x1) - max(lbb.x0, tb.x0)
            assert ov <= 1.0, f"label overlaps cell {t.get_text()[:16]!r}: {tag!r}"
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        a = labels[i].get_window_extent(renderer=ren)
        b = labels[j].get_window_extent(renderer=ren)
        if a.overlaps(b):
            ov = min(a.x1, b.x1) - max(a.x0, b.x0)
            assert ov <= 1.0, (f"label overlap: {labels[i].get_text()[:20]!r} vs "
                               f"{labels[j].get_text()[:20]!r}")

out = "/home/lurui/workspace/R4Det/assets/r4det_notemporal_vs_n4_table"
fig.savefig(out + ".png", dpi=200)
fig.savefig(out + ".pdf")
print(f"OK  {len(boxes)} cells, assertions passed")
print(f"    {out}.png")
print(f"    {out}.pdf")
