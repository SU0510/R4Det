#!/usr/bin/env python3
"""Dedicated table: No-Temporal vs the main model's best seed (seed2, 40.88).

Server-side rendering (no headless browser available), so the table is drawn
with matplotlib.  Missing CJK glyphs are promoted to a hard error, and the
layout is checked programmatically before saving.
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

for _f in ["/home/lurui/.local/share/fonts/NotoSansCJKsc-Regular.otf",
           "/home/lurui/.local/share/fonts/NotoSansCJKsc-Bold.otf"]:
    if os.path.exists(_f):
        font_manager.fontManager.addfont(_f)
plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"

# --- font size bump: every text size goes up by FONT_DELTA points -------------
FONT_DELTA = 2


def fs(x):
    return x + FONT_DELTA


HDR_BG, HDR_FG = "#2f5597", "white"
OV_BG, ALT_BG = "#eef5ff", "#f7f9fc"
EDGE, TXT, DIM, HL = "#b8c4d6", "#1f2430", "#6b7280", "#b05a00"

# Δ is taken from the *rounded* displayed values so the reader can verify it
# by subtraction (Car: exact +3.48, displayed +3.49).
BEST_ROWS = [
    ("Overall 3D moderate",          "35.59", "40.88", "+5.29"),
    ("Overall BEV moderate",         "42.23", "49.85", "+7.62"),
    ("Car 3D moderate strict",       "47.86", "51.35", "+3.49"),
    ("Truck 3D moderate strict",     "25.53", "30.76", "+5.23"),
    ("Pedestrian 3D moderate loose", "27.07", "31.32", "+4.25"),
    ("Cyclist 3D moderate loose",    "41.89", "50.09", "+8.20"),
]
WIN_ROWS = [
    ("Overall 3D moderate",          "34.65", "39.40", "+4.75"),
    ("Overall BEV moderate",         "41.62", "48.33", "+6.71"),
    ("Car 3D moderate strict",       "46.16", "48.98", "+2.82"),
    ("Truck 3D moderate strict",     "23.48", "30.04", "+6.56"),
    ("Pedestrian 3D moderate loose", "26.67", "30.58", "+3.91"),
    ("Cyclist 3D moderate loose",    "42.28", "47.99", "+5.71"),
]

COLW = [0.360, 0.205, 0.205, 0.105]        # sums to 0.875
COLX = [0.0]
for _w in COLW:
    COLX.append(COLX[-1] + _w)
HEAD = ["指标", "No-Temporal", "N4 RSSM", "Δ"]


def cell(ax, j, y0, y1, text, *, bg=None, fg=TXT, bold=False, size=10.5,
         align="center", boxes=None):
    x0, x1 = COLX[j], COLX[j + 1]
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                           facecolor=bg if bg else "none",
                           edgecolor=EDGE, linewidth=0.8, zorder=1))
    if not text:
        return
    if align == "center":
        x, ha = (x0 + x1) / 2, "center"
    else:
        x, ha = x0 + 0.010, "left"
    t = ax.text(x, (y0 + y1) / 2, text, ha=ha, va="center", fontsize=fs(size),
                color=fg, fontweight="bold" if bold else "normal", zorder=3)
    if boxes is not None:
        boxes.append((t, (x0, x1, y0, y1)))


def table(ax, y_top, rh, rows, boxes):
    y1, y0 = y_top, y_top - rh
    for j, txt in enumerate(HEAD):
        cell(ax, j, y0, y1, txt, bg=HDR_BG, fg=HDR_FG, bold=True,
             size=10.0, boxes=boxes)
    y = y0
    for i, (lab, nt, s2, d) in enumerate(rows):
        y1, y0 = y, y - rh
        overall = lab.startswith("Overall")
        bg = OV_BG if overall else (ALT_BG if i % 2 else None)
        cell(ax, 0, y0, y1, lab, bg=bg, align="left", bold=overall, boxes=boxes)
        cell(ax, 1, y0, y1, nt, bg=bg, boxes=boxes)
        cell(ax, 2, y0, y1, s2, bg=bg, bold=overall, boxes=boxes)
        cell(ax, 3, y0, y1, d, bg=bg, fg=HL, bold=True, boxes=boxes)
        y = y0
    return y


warnings.filterwarnings("error", message=".*[Gg]lyph.*")

fig = plt.figure(figsize=(12.8, 7.2), dpi=200)
fig.patch.set_facecolor("white")
ax = fig.add_axes([0.055, 0.03, 0.90, 0.93])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

rh = 0.055
boxes = []

ax.text(0, 1.005, "No-Temporal vs 主模型最高单点（N4 Motion-Aligned RSSM, seed2）",
        ha="left", va="bottom", fontsize=fs(14.0), fontweight="bold", color=TXT)
ax.text(0, 0.972, "同栈单变量：两份 config 差异仅 temporal_fusion（None ↔ MotionAlignedRSSMFusion）",
        ha="left", va="bottom", fontsize=fs(10.0), color=DIM)

y = 0.930
ax.text(0, y + 0.006, "BEST 口径（论文 / 汇报用）　　Δ = 主模型 seed2 − No-Temporal",
        ha="left", va="bottom", fontsize=fs(11.5), fontweight="bold", color=TXT)
y = table(ax, y, rh, BEST_ROWS, boxes)

y -= 0.055
ax.text(0, y + 0.006, "窗口口径（ep12-16 均值，实验室选型用）",
        ha="left", va="bottom", fontsize=fs(11.5), fontweight="bold", color=TXT)
y = table(ax, y, rh, WIN_ROWS, boxes)

ax.text(0, 0.058,
        "No-Temporal BEST 落在 ep20，主模型 seed2 BEST 落在 ep14；No-Temporal 仅有 seed0 一次，"
        "故本表是「单帧 seed0 vs 主模型最优 seed」。",
        ha="left", va="top", fontsize=fs(9.5), color=DIM)
ax.text(0, 0.026,
        "四类均值已验证等于 Overall 3D 行；数据来源：原始 *.log.json 重算。",
        ha="left", va="top", fontsize=fs(9.5), color=DIM)

fig.canvas.draw()
ren = fig.canvas.get_renderer()
fb = fig.bbox

for t, (x0, x1, y0, y1) in boxes:
    bb = t.get_window_extent(renderer=ren)
    assert bb.x0 >= -1 and bb.y0 >= -1, f"text below/left of canvas: {t.get_text()!r}"
    assert bb.x1 <= fb.x1 + 1 and bb.y1 <= fb.y1 + 1, f"text above/right of canvas: {t.get_text()!r}"
    inv = ax.transData.inverted()
    cx0 = inv.transform((bb.x0, 0))[0]
    cx1 = inv.transform((bb.x1, 0))[0]
    assert cx0 >= x0 - 0.004, f"text left of its cell: {t.get_text()!r}"
    assert cx1 <= x1 + 0.004, f"text right of its cell: {t.get_text()!r}"
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        a = boxes[i][0].get_window_extent(renderer=ren)
        b = boxes[j][0].get_window_extent(renderer=ren)
        if a.overlaps(b):
            ov = min(a.x1, b.x1) - max(a.x0, b.x0)
            assert ov <= 1.0, (f"overlap {boxes[i][0].get_text()!r} / "
                               f"{boxes[j][0].get_text()!r}")
# Δ column must be consistent with the two value columns
for rows in (BEST_ROWS, WIN_ROWS):
    for lab, nt, s2, d in rows:
        assert abs((float(s2) - float(nt)) - float(d)) < 0.005, (lab, nt, s2, d)

out = "/home/lurui/workspace/R4Det/assets/r4det_notemporal_vs_seed2_table"
fig.savefig(out + ".png", dpi=200)
fig.savefig(out + ".pdf")
print(f"OK  {len(boxes)} cells, all assertions passed (incl. Δ arithmetic)")
print(f"    {out}.png\n    {out}.pdf")
