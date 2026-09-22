#!/usr/bin/env python3
"""Adapted RSSM figure in the style of the Dreamer 'World model learning' plot.
Highlights the changes made for the radar-camera BEV detection network:
 (1) motion-aligned deformable warp of h/z states (offsets from x_t)
 (2) no action input (TJ4D has no ego velocity)
 (3) z_t re-injected into the BEV stream via output_proj + residual
 (4) losses unchanged: recon (MSE) + KL(q||p)"""
import cairosvg

TXT = '#1A1A1A'
SUB = '#666666'
GREEN_F, GREEN_S = '#C9F0DC', '#4CAF50'
TEAL = '#4DB6AC'
GREEN_A = '#66BB6A'
PURP = '#AB7CE0'
ORG_F, ORG_S = '#FFE3D3', '#E8722C'
RED = '#D64545'
ENC_F, ENC_S = '#6FA8DC', '#3D7CC9'
DEC_F, DEC_S = '#D2A8E8', '#9B59B6'
FONT = 'DejaVu Sans'

CHECKER = [
    [1, 0, 1, 1, 0, 0],
    [0, 1, 0, 0, 1, 1],
    [1, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 1],
    [1, 0, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 0],
]
XT_CELLS = ['#BFD8C9', '#AFCBE3', '#E3DEBE', '#D8CFC2', '#C2D6CC',
            '#E0D4BE', '#B7CDE0', '#CBDCC2', '#DCD3C6', '#B9CFE2',
            '#CFE0C6', '#D9CEC0', '#C6D8CE', '#E2DCC0', '#B4C9DD']
REC_CELLS = ['#DCE8D9', '#D3E2EF', '#F1EDDC', '#E8E1D8', '#D9E4DC',
             '#EEE7D6', '#D5E2EE', '#DFE9DB', '#EAE3D9', '#D6E1EE',
             '#E0E8DA', '#EAE2D6', '#DAE3DC', '#F0EBDD', '#D3DEEC']

P = []
A = P.append


def rr(x, y, w, h, fill, stroke, rx=14, sw=3, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    A(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
      f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def ln(x1, y1, x2, y2, stroke, sw=2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    A(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
      f'stroke-width="{sw}"{d}/>')


def ar(pts, color, sw=3.5, dashed=False, marker='arrB'):
    d = ' stroke-dasharray="8 6"' if dashed else ''
    p = ' '.join(f'{a:.1f},{b:.1f}' for a, b in pts)
    A(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}" '
      f'stroke-linejoin="round"{d} marker-end="url(#{marker})"/>')


def tx(x, y, s, size=28, weight='bold', fill=TXT, anchor='middle', italic=False):
    style = ' font-style="italic"' if italic else ''
    A(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
      f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{style}>{s}</text>')


def tx2(x, y, parts, size=21, anchor='middle'):
    """text with colored tspans"""
    tsp = ''.join(f'<tspan fill="{c}">{s}</tspan>' for s, c in parts)
    A(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
      f'font-weight="bold" text-anchor="{anchor}">{tsp}</text>')


def checkerboard(cx, y0):
    x0 = cx - 42
    for i in range(6):
        for j in range(6):
            c = '#333333' if CHECKER[i][j] else '#FFFFFF'
            A(f'<rect x="{x0 + i * 14}" y="{y0 + j * 14}" width="14" height="14" '
              f'fill="{c}" stroke="#555555" stroke-width="1.6"/>')


def bev_img(x, y, cells, label_lbl=None):
    rr(x, y, 140, 90, '#F4F6F2', '#8A8A8A', rx=10, sw=2.5)
    k = 0
    for j in range(3):
        for i in range(5):
            A(f'<rect x="{x + 10 + i * 24}" y="{y + 9 + j * 24}" width="22" height="21" '
              f'fill="{cells[k]}" stroke="#FFFFFF" stroke-width="1.4"/>')
            k += 1


def trapezoid(cx, side, fill, stroke, name):
    """side=-1: enc (left, points up); side=+1: dec (right, points down)"""
    if side < 0:
        pts = [(cx - 155, 470), (cx - 45, 470), (cx - 75, 400), (cx - 125, 400)]
    else:
        pts = [(cx + 45, 400), (cx + 155, 400), (cx + 130, 470), (cx + 70, 470)]
    p = ' '.join(f'{a},{b}' for a, b in pts)
    A(f'<polygon points="{p}" fill="{fill}" stroke="{stroke}" stroke-width="2.5" '
      f'stroke-linejoin="round"/>')
    tx(cx + side * 100, 442, name, 18, 'bold', '#FFFFFF')


def align_box(x0, num):
    """orange deformable-align module between two panels; x0..x0+125, y 196..330"""
    rr(x0, 196, 125, 134, ORG_F, ORG_S, rx=14, sw=3)
    gx, gy = x0 + 38, 240
    for i in range(7):
        ln(gx + i * 8, gy, gx + i * 8, gy + 48, '#888888', 1.2)
    for j in range(7):
        ln(gx, gy + j * 8, gx + 48, gy + j * 8, '#888888', 1.2)
    A(f'<polygon points="{gx + 6},{gy + 4} {gx + 42},{gy + 2} {gx + 38},{gy + 40} '
      f'{gx + 2},{gy + 42}" fill="none" stroke="{ORG_S}" stroke-width="2.4" '
      f'stroke-dasharray="5 4"/>')
    tx(x0 + 62, 184, '(1) Deformable Align', 13, 'bold', ORG_S)
    tx(x0 + 62, 366, 'motion offsets', 11, 'bold', ORG_S)


def cross(x, y):
    ln(x - 9, y - 9, x + 9, y + 9, RED, 4)
    ln(x + 9, y - 9, x - 9, y + 9, RED, 4)


# ---------------------------------------------------------------- canvas
W, H = 1800, 930
A(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
A(f'<rect width="{W}" height="{H}" fill="white"/>')
mk = lambda i, c: (f'<marker id="{i}" viewBox="0 0 10 10" refX="8.5" refY="5" '
                   f'markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
                   f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{c}"/></marker>')
A('<defs>' + mk('arrB', '#333333') + mk('arrTeal', TEAL) + mk('arrGreen', GREEN_A) +
  mk('arrPur', PURP) + mk('arrOrg', ORG_S) + mk('arrGray', '#999999') + '</defs>')

tx(900, 46, 'Motion-Aligned RSSM Temporal Fusion', 30, 'bold')
tx(900, 74, 'adapted world-model learning for radar–camera BEV detection', 16, 'normal', SUB)

CXS = [300, 735, 1170]
T_LABELS = ['t−2', 't−1', 't']

for idx, cx in enumerate(CXS):
    tl = T_LABELS[idx]
    tx(cx - 155, 142, tl, 15, 'normal', SUB, anchor='end')

    # gray state panel
    rr(cx - 140, 150, 280, 240, '#F2F2F2', '#DBDBDB', rx=26, sw=3)
    # h box
    rr(cx - 62, 172, 124, 72, GREEN_F, GREEN_S, rx=14, sw=3)
    tx(cx, 212, 'h_t', 24, 'bold', '#2E7D32', italic=True)
    tx(cx, 233, 'GRU hidden', 11, 'normal', SUB)
    # z checkerboard + h->z transition arrow (prior)
    checkerboard(cx, 264)
    ar([(cx, 248), (cx, 258)], GREEN_A, 3, marker='arrGreen')
    tx(cx - 52, 376, 'z_t', 18, 'bold', '#333333', italic=True, anchor='end')

    # ② action removed: dashed stub + red cross at every step
    cross(cx, 108)
    ar([(cx, 122), (cx, 164)], '#999999', 3, dashed=True, marker='arrGray')
    if idx == 0:
        tx(255, 94, 'motion a_t (not used)', 14, 'bold', RED, anchor='end')
        tx2(255, 116, [('(2) ', ORG_S), (' no ego velocity in TJ4D', SUB)], 12, anchor='end')

    # enc / dec trapezoids
    trapezoid(cx, -1, ENC_F, ENC_S, 'enc')
    trapezoid(cx, +1, DEC_F, DEC_S, 'dec')

    # x_t image + label + losses + posterior/prior stack
    bev_img(cx - 170, 480, XT_CELLS)
    tx(cx - 100, 594, 'x_t (fused BEV)', 13, 'bold')
    tx(cx - 100, 612, 'KL(q‖p) · q: post, p: pri', 11, 'normal', SUB)
    for bi, (lbl, col) in enumerate([('q(z_t | h_t, e_t)', ENC_S),
                                     ('p(z_t | h_t)', DEC_S),
                                     ('z_t = μ_q', '#777777')]):
        rr(cx - 162, 626 + bi * 42, 124, 34, '#FFFFFF', col, rx=8, sw=2.5)
        tx(cx - 100, 648 + bi * 42, lbl, 12.5, 'bold')

    # recon image + label + loss
    bev_img(cx + 30, 480, REC_CELLS)
    tx(cx + 100, 594, 'x̂_t (recon)', 14, 'bold')
    tx(cx + 100, 612, 'L_recon (MSE)', 12, 'normal', SUB)

# ---- recurrent paths through the align boxes (gaps between panels)
for gx0, cxa, cxb in [(440, 300, 735), (875, 735, 1170)]:
    al, ar_ = gx0 + 15, gx0 + 110
    # green h path
    ar([(cxa + 62, 208), (al, 208)], GREEN_A, 3.5, marker='arrGreen')
    ar([(ar_, 208), (cxb - 66, 208)], GREEN_A, 3.5, marker='arrGreen')
    # purple z path
    ar([(cxa + 42, 306), (al, 306)], PURP, 3.5, marker='arrPur')
    ar([(ar_, 306), (cxb - 46, 306)], PURP, 3.5, marker='arrPur')

align_box(455, 1)
align_box(890, 2)
tx(880, 396, 'h_t, z_t → next frame (BPTT)', 12, 'normal', SUB, anchor='end')

# ---- ③ output path: z_t ⊕ x_t → BEV stream (from step 3)
A(f'<circle cx="1352" cy="306" r="17" fill="#FFFFFF" stroke="{ORG_S}" stroke-width="3"/>')
tx(1352, 313, '+', 22, 'bold', ORG_S)
ar([(1373, 306), (1391, 306)], ORG_S, 3.5, marker='arrOrg')
rr(1395, 270, 350, 72, '#FFF3E6', ORG_S, rx=14, sw=3)
tx(1570, 300, 'BEV stream', 19, 'bold')
tx(1570, 325, '→ output_proj → 3D detection', 13, 'normal', SUB)
tx(1570, 258, '(3) z_t re-injected into the BEV stream', 14, 'bold', ORG_S)
# residual tap from x_t (step 3)
ar([(1144, 525), (1195, 525), (1195, 380), (1220, 380), (1220, 306), (1331, 306)], ORG_S, 2.5, dashed=True, marker='arrOrg')
tx(1170, 556, 'x_t', 12, 'bold', ORG_S)
tx(1170, 572, 'resid.', 11, 'bold', ORG_S)


# ---- training losses panel (bottom-right) ----
rr(1390, 560, 380, 228, '#FCFCFC', '#999999', rx=14, sw=2.5, dash='8 6')
tx(1580, 590, 'Training Losses', 17, 'bold')
items = [
    ('L_recon', 'MSE(x̂_t, x_t) — reconstruction', PURP),
    ('L_KL', 'KL(q(z_t|h_t,e_t) ‖ p(z_t|h_t))', PURP),
    ('L_det', '3D detection: Focal + SmoothL1 + Dir', '#3D7CC9'),
    ('L_iou', 'IoU-aware quality (L1, train only)', '#3D7CC9'),
]
for bi, (name, desc, col) in enumerate(items):
    yy = 620 + bi * 36
    rr(1412, yy, 84, 26, '#FFFFFF', col, rx=7, sw=2.2)
    tx(1454, yy + 18, name, 13, 'bold', col)
    tx(1510, yy + 18, desc, 12.5, 'normal', TXT, anchor='start')
tx(1580, 776, 'losses unchanged from the world-model baseline', 12, 'normal', SUB)

# ---------------------------------------------------------------- legend
rr(60, 806, 34, 22, GREEN_F, GREEN_S, rx=6, sw=2.2)
tx(104, 822, 'h_t — deterministic GRU state', 14, 'normal', TXT, anchor='start')
for i in range(3):
    for j in range(2):
        c = '#333333' if (i + j) % 2 == 0 else '#FFFFFF'
        A(f'<rect x="{426 + i * 11}" y="{806 + j * 11}" width="11" height="11" '
          f'fill="{c}" stroke="#555555" stroke-width="1.2"/>')
tx(472, 822, 'z_t — stochastic latent', 14, 'normal', TXT, anchor='start')
tx2(712, 824, [('(2) ', ORG_S), (' motion a_t not used at every step', TXT)], 13, anchor='start')
rr(1060, 806, 34, 22, ORG_F, ORG_S, rx=6, sw=2.2)
tx2(1104, 822, [('(1)(3) ', ORG_S), (' adapted modules (this network)', TXT)], 13, anchor='start')

A(f'<polygon points="60,852 120,852 108,874 72,874" fill="{ENC_F}" stroke="{ENC_S}" stroke-width="2.2"/>')
tx(140, 868, 'enc = BEV feature encoder', 14, 'normal', TXT, anchor='start')
A(f'<polygon points="426,852 486,852 474,874 438,874" fill="{DEC_F}" stroke="{DEC_S}" stroke-width="2.2"/>')
tx(506, 868, 'dec = reconstruction decoder', 14, 'normal', TXT, anchor='start')
tx(800, 868, 'inputs x_t are fused BEV features (radar–camera), not raw images', 14, 'normal', SUB, anchor='start')

A('</svg>')

svg = '\n'.join(P)
with open('/home/lurui/workspace/R4Det/assets/r4det_rssm_adapted.svg', 'w') as f:
    f.write(svg)
cairosvg.svg2png(bytestring=svg.encode(), write_to='/home/lurui/workspace/R4Det/assets/r4det_rssm_adapted.png',
                 output_width=W * 2, output_height=H * 2)
cairosvg.svg2png(bytestring=svg.encode(), write_to='/tmp/r4det_rssm_adapted_view.png',
                 output_width=int(W * 0.72), output_height=int(H * 0.72))
cairosvg.svg2pdf(bytestring=svg.encode(), write_to='/home/lurui/workspace/R4Det/assets/r4det_rssm_adapted.pdf')
print('done')
