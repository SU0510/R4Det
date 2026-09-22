#!/usr/bin/env python3
"""Paper-style wide table: 3 rows (2 methods + delta) x 10 metric columns.

Column order (left to right) follows the request:
  Car 3D strict, Ped 3D loose, Cyc 3D loose, Truck 3D strict, Overall 3D,
  Car BEV strict, Ped BEV loose, Cyc BEV loose, Truck BEV strict, Overall BEV
(moderate difficulty throughout).  The third row is N4 RSSM - No-Temporal.

Server-side rendering (no headless browser), so the table is drawn with
matplotlib.  Missing CJK glyphs are hard errors; layout and the arithmetic
identities are asserted before saving.
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
BAND_BG = "#4a6fa5"
OURS_BG, ALT_BG, DELTA_BG = "#eef5ff", "#f7f9fc", "#fff4e6"
EDGE, TXT, DIM, HL = "#b8c4d6", "#1f2430", "#6b7280", "#b05a00"

CLASSES = ["Car", "Pedestrian", "Cyclist", "Truck", "Overall"]
QUAL = ["strict", "loose", "loose", "strict", ""]
BAND = ["3D detection (moderate)", "BEV detection (moderate)"]

# No-Temporal BEST @ep20, N4 RSSM (seed2) BEST @ep14
NOTEMPORAL = [47.86, 27.07, 41.89, 25.53, 35.59, 63.32, 28.81, 44.25, 32.54, 42.23]
N4RSSM = [51.35, 31.32, 50.09, 30.76, 40.88, 70.61, 33.54, 51.81, 43.45, 49.85]
# delta taken from the rounded displayed values, so a reader can verify by
# subtraction (Car 3D: exact +3.48, displayed +3.49)
DELTA = [round(b - a, 2) for a, b in zip(NOTEMPORAL, N4RSSM)]

COLW = [0.135] + [0.077] * 10            # sums to 0.905
COLX = [0.0]
for _w in COLW:
    COLX.append(COLX[-1] + _w)

ROWH = {"band": 0.100, "class": 0.120, "qual": 0.090, "data": 0.120}
NOTE_Y0, NOTE_DY = 0.270, 0.075


def cell(ax, x0, x1, y0, y1, text, *, bg=None, fg=TXT, bold=False, size=10.0,
         align="center", boxes=None):
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


def span_cell(ax, i0, i1, y0, y1, text, boxes, **kw):
    cell(ax, COLX[i0], COLX[i1], y0, y1, text, boxes=boxes, **kw)


warnings.filterwarnings("error", message=".*[Gg]lyph.*")

fig = plt.figure(figsize=(15.6, 4.2), dpi=200)
fig.patch.set_facecolor("white")
ax = fig.add_axes([0.025, 0.03, 0.95, 0.88])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

fig.text(0.025, 0.978, "No-Temporal 与 N4 RSSM 在 TJ4D 验证集上的对比（AP40 / %）",
         ha="left", va="top", fontsize=fs(12.5), fontweight="bold", color=TXT)

boxes = []
y = 1.000

# --- band row (grouping)
h = ROWH["band"]
cell(ax, COLX[0], COLX[1], y - h, y, "", boxes=boxes)
span_cell(ax, 1, 6, y - h, y, BAND[0], boxes, bg=BAND_BG, fg=HDR_FG, bold=True, size=10.0)
span_cell(ax, 6, 11, y - h, y, BAND[1], boxes, bg=BAND_BG, fg=HDR_FG, bold=True, size=10.0)
y -= h

# --- class row
h = ROWH["class"]
cell(ax, COLX[0], COLX[1], y - h, y, "Method", boxes=boxes,
     bg=HDR_BG, fg=HDR_FG, bold=True, size=10.0)
for k, cls in enumerate(CLASSES):
    for off in (0, 5):
        i = 1 + off + k
        cell(ax, COLX[i], COLX[i + 1], y - h, y, cls, boxes=boxes,
             bg=HDR_BG, fg=HDR_FG, bold=True, size=10.0)
y -= h

# --- qualifier row
h = ROWH["qual"]
cell(ax, COLX[0], COLX[1], y - h, y, "", boxes=boxes, bg=ALT_BG)
for k, q in enumerate(QUAL):
    for off in (0, 5):
        i = 1 + off + k
        cell(ax, COLX[i], COLX[i + 1], y - h, y, q, boxes=boxes,
             bg=ALT_BG, fg=DIM, size=9.0)
y -= h

# --- data rows
for name, vals, ours in (("No-Temporal", NOTEMPORAL, False),
                         ("N4 RSSM", N4RSSM, True)):
    h = ROWH["data"]
    cell(ax, COLX[0], COLX[1], y - h, y, name, boxes=boxes,
         bg=OURS_BG if ours else None, bold=ours, size=10.5, align="left")
    for i, v in enumerate(vals):
        cell(ax, COLX[i + 1], COLX[i + 2], y - h, y, f"{v:.2f}", boxes=boxes,
             bg=OURS_BG if ours else None, bold=ours, size=10.0)
    y -= h

# --- delta row
h = ROWH["data"]
cell(ax, COLX[0], COLX[1], y - h, y, "Δ (N4 − No-Temporal)", boxes=boxes,
     bg=DELTA_BG, fg=HL, bold=True, size=9.5, align="left")
for i, v in enumerate(DELTA):
    cell(ax, COLX[i + 1], COLX[i + 2], y - h, y, f"{v:+.2f}", boxes=boxes,
         bg=DELTA_BG, fg=HL, bold=True, size=10.0)
y -= h

NOTES = [
    "列依次为：Car 3D moderate strict · Pedestrian 3D moderate loose · Cyclist 3D moderate loose · "
    "Truck 3D moderate strict · Overall 3D moderate ·",
    "　　　　　Car BEV moderate strict · Pedestrian BEV moderate loose · Cyclist BEV moderate loose · "
    "Truck BEV moderate strict · Overall BEV moderate",
    "Overall = 四类均值（已验证逐位相等）。No-Temporal BEST @ep20；N4 RSSM BEST @ep14（主模型最高 seed）。",
    "同栈单变量：两份 config 差异仅 temporal_fusion（None ↔ MotionAlignedRSSMFusion）。数据来源：原始 *.log.json 重算。",
]
note_ts = [ax.text(0, NOTE_Y0 - NOTE_DY * k, txt, ha="left", va="top",
                   fontsize=fs(8.5), color=DIM) for k, txt in enumerate(NOTES)]

fig.canvas.draw()
ren = fig.canvas.get_renderer()
fb = fig.bbox

for t, (x0, x1, y0, y1) in boxes:
    bb = t.get_window_extent(renderer=ren)
    assert bb.x0 >= -1 and bb.y0 >= -1, f"outside canvas (lo): {t.get_text()!r}"
    assert bb.x1 <= fb.x1 + 1 and bb.y1 <= fb.y1 + 1, f"outside canvas (hi): {t.get_text()!r}"
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

# notes: inside canvas, no self-overlap, no overlap with table cells
for n in note_ts:
    nb = n.get_window_extent(renderer=ren)
    assert nb.x0 >= -1 and nb.y0 >= -1, f"note outside canvas: {n.get_text()[:24]!r}"
    assert nb.x1 <= fb.x1 + 1 and nb.y1 <= fb.y1 + 1, f"note outside canvas: {n.get_text()[:24]!r}"
for i in range(len(note_ts)):
    for j in range(i + 1, len(note_ts)):
        a = note_ts[i].get_window_extent(renderer=ren)
        b = note_ts[j].get_window_extent(renderer=ren)
        assert not a.overlaps(b), f"note overlap: {note_ts[i].get_text()[:24]!r}"
for t, _ in boxes:
    bb = t.get_window_extent(renderer=ren)
    for n in note_ts:
        assert not bb.overlaps(n.get_window_extent(renderer=ren)), \
            f"note overlaps table cell: {n.get_text()[:24]!r}"

# arithmetic: four classes average to Overall (both modalities, both rows),
# the delta row equals the difference of the two rows, and N4 wins every column
for label, v in (("No-Temporal", NOTEMPORAL), ("N4 RSSM", N4RSSM)):
    assert abs(sum(v[0:4]) / 4 - v[4]) < 0.005, (label, "3D")
    assert abs(sum(v[5:9]) / 4 - v[9]) < 0.005, (label, "BEV")
for a, b, d in zip(NOTEMPORAL, N4RSSM, DELTA):
    assert abs(round(b, 2) - round(a, 2) - d) < 0.005, (a, b, d)
assert all(b > a for a, b in zip(NOTEMPORAL, N4RSSM)), "N4 must win every column"

out = "/home/lurui/workspace/R4Det/assets/r4det_paper_table_notemporal_vs_n4"
fig.savefig(out + ".png", dpi=200)
fig.savefig(out + ".pdf")
print(f"OK  {len(boxes)} cells, assertions passed "
      f"(layout + Overall identity + Δ row + 10/10 win)")
print(f"    {out}.png\n    {out}.pdf")
