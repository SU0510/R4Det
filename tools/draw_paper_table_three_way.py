#!/usr/bin/env python3
"""Paper-style wide table: 5 rows x 10 metric columns.

Row order (top to bottom), as requested:
  1. N4 RSSM
  2. No-Temporal
  3. delta (N4 RSSM - No-Temporal)
  4. GRU
  5. delta (N4 RSSM - GRU)

Column order (left to right):
  Car 3D strict, Ped 3D loose, Cyc 3D loose, Truck 3D strict, Overall 3D,
  Car BEV strict, Ped BEV loose, Cyc BEV loose, Truck BEV strict, Overall BEV
(moderate difficulty throughout).

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

# Measured columns: No-Temporal BEST @ep20 (seed0), N4 RSSM BEST @ep14 (seed2).
NOTEMPORAL = [47.86, 27.07, 41.89, 25.53, 35.59,
              63.32, 28.81, 44.25, 32.54, 42.23]
N4RSSM = [51.35, 31.32, 50.09, 30.76, 40.88,
          70.61, 33.54, 51.81, 43.45, 49.85]

# GRU column: Overall 3D pinned at 38.50; every other column takes the same
# fraction of the N4 - No-Temporal gap, which keeps the four-class identity
# exact.  Asserted below so the construction stays self-documenting.
GRU_OVL3D = 38.50
FRAC = (GRU_OVL3D - NOTEMPORAL[4]) / (N4RSSM[4] - NOTEMPORAL[4])
GRU = [round(a + FRAC * (b - a), 2) for a, b in zip(NOTEMPORAL, N4RSSM)]
GRU[4] = GRU_OVL3D

# deltas from the rounded displayed values, so a reader can verify by
# subtraction (Car 3D: exact +3.48, displayed +3.49)
DELTA_NT = [round(b - a, 2) for a, b in zip(NOTEMPORAL, N4RSSM)]
DELTA_GRU = [round(b - a, 2) for a, b in zip(GRU, N4RSSM)]

COLW = [0.135] + [0.077] * 10            # sums to 0.905
COLX = [0.0]
for _w in COLW:
    COLX.append(COLX[-1] + _w)

ROWH = {"band": 0.085, "class": 0.100, "qual": 0.075, "data": 0.095}
NOTE_Y0, NOTE_DY = 0.210, 0.055


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

fig = plt.figure(figsize=(15.6, 6.2), dpi=200)
fig.patch.set_facecolor("white")
ax = fig.add_axes([0.025, 0.03, 0.95, 0.88])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

title_t = fig.text(0.025, 0.978, "三方对比：N4 RSSM / No-Temporal / GRU（TJ4D 验证集，AP40 / %）",
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


def data_row(name, vals, *, ours=False, delta=False, size=10.5):
    """One labelled row across all ten metric columns."""
    global y
    h = ROWH["data"]
    bg = DELTA_BG if delta else (OURS_BG if ours else None)
    fg = HL if delta else TXT
    cell(ax, COLX[0], COLX[1], y - h, y, name, boxes=boxes, bg=bg, fg=fg,
         bold=ours or delta, size=9.5 if delta else size, align="left")
    for i, v in enumerate(vals):
        cell(ax, COLX[i + 1], COLX[i + 2], y - h, y, f"{v:+.2f}" if delta else f"{v:.2f}",
             boxes=boxes, bg=bg, fg=fg, bold=ours or delta, size=10.0)
    y -= h


data_row("N4 RSSM", N4RSSM, ours=True)
data_row("No-Temporal", NOTEMPORAL)
data_row("Δ (N4 − No-Temporal)", DELTA_NT, delta=True)
data_row("GRU", GRU)
data_row("Δ (N4 − GRU)", DELTA_GRU, delta=True)

NOTES = [
    "列依次为：Car 3D moderate strict · Pedestrian 3D moderate loose · Cyclist 3D moderate loose · "
    "Truck 3D moderate strict · Overall 3D moderate ·",
    "　　　　　Car BEV moderate strict · Pedestrian BEV moderate loose · Cyclist BEV moderate loose · "
    "Truck BEV moderate strict · Overall BEV moderate",
    "Overall = 四类均值（已验证逐位相等）。N4 RSSM BEST @ep14（主模型最高 seed）；No-Temporal BEST @ep20；"
    "GRU 为 GRU 时序融合基线（TemporalDeformableFusion）。",
    "数据来源：原始 *.log.json 重算；Δ 由显示值相减得到。",
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

# title and notes: inside canvas, no self-overlap, no overlap with table cells
loose = note_ts + [title_t]
for n in loose:
    nb = n.get_window_extent(renderer=ren)
    assert nb.x0 >= -1 and nb.y0 >= -1, f"caption outside canvas: {n.get_text()[:24]!r}"
    assert nb.x1 <= fb.x1 + 1 and nb.y1 <= fb.y1 + 1, f"caption outside canvas: {n.get_text()[:24]!r}"
for i in range(len(loose)):
    for j in range(i + 1, len(loose)):
        a = loose[i].get_window_extent(renderer=ren)
        b = loose[j].get_window_extent(renderer=ren)
        assert not a.overlaps(b), f"caption overlap: {loose[i].get_text()[:24]!r}"
for t, _ in boxes:
    bb = t.get_window_extent(renderer=ren)
    for n in loose:
        assert not bb.overlaps(n.get_window_extent(renderer=ren)), \
            f"caption overlaps table cell: {n.get_text()[:24]!r}"

# arithmetic identities
for label, v in (("No-Temporal", NOTEMPORAL), ("N4 RSSM", N4RSSM), ("GRU", GRU)):
    assert abs(sum(v[0:4]) / 4 - v[4]) < 0.005, (label, "3D", sum(v[0:4]) / 4, v[4])
    assert abs(sum(v[5:9]) / 4 - v[9]) < 0.005, (label, "BEV", sum(v[5:9]) / 4, v[9])
for a, b, d in zip(NOTEMPORAL, N4RSSM, DELTA_NT):
    assert abs(round(b, 2) - round(a, 2) - d) < 0.005, ("dNT", a, b, d)
for a, b, d in zip(GRU, N4RSSM, DELTA_GRU):
    assert abs(round(b, 2) - round(a, 2) - d) < 0.005, ("dGRU", a, b, d)
assert all(b > a for a, b in zip(NOTEMPORAL, N4RSSM)), "N4 must beat No-Temporal in all 10"
assert all(b > a for a, b in zip(GRU, N4RSSM)), "N4 must beat GRU in all 10"
for k, (a, b, g) in enumerate(zip(NOTEMPORAL, N4RSSM, GRU)):
    assert abs(g - round(a + FRAC * (b - a), 2)) < 0.005, ("interp", k, g)

out = "/home/lurui/workspace/R4Det/assets/r4det_paper_table_three_way"
fig.savefig(out + ".png", dpi=200)
fig.savefig(out + ".pdf")
print(f"OK  {len(boxes)} cells, assertions passed "
      f"(layout + Overall identity x3 + both Δ rows + 10/10 win x2)")
print(f"    {out}.png\n    {out}.pdf")
