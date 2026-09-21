#!/usr/bin/env python3
"""Generate the R4Det paper architecture figure (SVG -> PNG + PDF)."""
import cairosvg

W, H = 2850, 900
parts = []

ORANGE_F, ORANGE_S = '#FDEBC8', '#E8A33D'
BLUE_F, BLUE_S = '#DAE8FC', '#6C8EBF'
PINK_F, PINK_S = '#F8CECC', '#B85450'
GRAY_F, GRAY_S = '#F5F5F5', '#999999'
TXT = '#1A1A1A'
SUB = '#666666'
FONT = 'DejaVu Sans, Helvetica, Arial, sans-serif'


def add(s):
    parts.append(s)


def rrect(x, y, w, h, fill, stroke, rx=16, sw=3.5, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def poly(pts, fill, stroke, sw=3):
    p = ' '.join(f'{a:.1f},{b:.1f}' for a, b in pts)
    add(f'<polygon points="{p}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
        f'stroke-linejoin="round"/>')


def line(x1, y1, x2, y2, stroke, sw=2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def arrow(pts, color='#1A1A1A', sw=4.5, dashed=False, marker='arrB'):
    d = ' stroke-dasharray="9 7"' if dashed else ''
    p = ' '.join(f'{a:.1f},{b:.1f}' for a, b in pts)
    add(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}" '
        f'stroke-linejoin="round"{d} marker-end="url(#{marker})"/>')


def text(x, y, s, size=22, weight='bold', fill=TXT, anchor='middle'):
    add(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{s}</text>')


def defs():
    add(f'''<defs>
<marker id="arrB" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse">
  <path d="M 0 0 L 10 5 L 0 10 z" fill="#1A1A1A"/>
</marker>
<marker id="arrBlu" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse">
  <path d="M 0 0 L 10 5 L 0 10 z" fill="{BLUE_S}"/>
</marker>
<marker id="arrOrg" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse">
  <path d="M 0 0 L 10 5 L 0 10 z" fill="{ORANGE_S}"/>
</marker>
<marker id="arrBlk" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
  <path d="M 0 0 L 10 5 L 0 10 z" fill="#333333"/>
</marker>
</defs>''')


def iso_cube(x, y, w, h, dep, fill, stroke, grid=True, sw=3):
    """x,y = front-bottom-left corner of the front face."""
    sk, sk2 = dep * 0.62, dep * 0.40
    front = [(x, y), (x + w, y), (x + w, y - h), (x, y - h)]
    top = [(x, y - h), (x + w, y - h), (x + w + sk, y - h - sk2), (x + sk, y - h - sk2)]
    side = [(x + w, y - h), (x + w + sk, y - h - sk2), (x + w + sk, y - sk2), (x + w, y)]
    poly(top, fill, stroke, sw)
    poly(side, fill, stroke, sw)
    poly(front, fill, stroke, sw)
    if grid:
        nx, ny = 3, 2
        for i in range(1, nx):
            gx = x + w * i / nx
            line(gx, y, gx, y - h, stroke, 1.6)
            line(gx, y - h, gx + sk * i / nx, y - h - sk2 * i / nx, stroke, 1.6)
        for j in range(1, ny):
            gy = y - h * j / ny
            line(x, gy, x + w, gy, stroke, 1.6)
        for j in range(1, ny):
            gy = y - h * j / ny
            line(x + w, gy, x + w + sk, gy - sk2, stroke, 1.6)
        for i in range(1, nx):
            gx = x + w + sk * i / nx
            line(gx, y - h - sk2 * i / nx, gx, y - sk2 * i / nx, stroke, 1.6)


def sheet_stack(cx, cy, w, h, n, fill, stroke):
    """n stacked feature sheets (parallelograms), centered around (cx, cy)."""
    x0 = cx - w / 2 - (n - 1) * 8
    y0 = cy + h / 2 + (n - 1) * 7
    for i in range(n - 1, -1, -1):
        x, y = x0 + i * 16, y0 - i * 14
        poly([(x, y), (x + w, y - 20), (x + w, y - 20 - h), (x, y - h)], fill, stroke, 2.6)


# =====================================================================
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}">')
add(f'<rect width="{W}" height="{H}" fill="white"/>')
defs()

# ------------------------------------------------ camera branch (row 1)
# input icon: two overlapping frustums
poly([(55, 232), (115, 212), (115, 142), (55, 162)], ORANGE_F, ORANGE_S, 2.6)
poly([(125, 168), (168, 178), (168, 212), (125, 222)], ORANGE_F, ORANGE_S, 2.6)
text(105, 278, 'Multi-view')
text(105, 305, 'Images')
arrow([(180, 190), (243, 190)])

# camera encoder
rrect(250, 140, 180, 100, ORANGE_F, ORANGE_S)
for i in range(2, -1, -1):
    add(f'<rect x="{288 + i * 13}" y="{172 - i * 10}" width="62" height="52" rx="8" '
        f'fill="white" stroke="{ORANGE_S}" stroke-width="2.4"/>')
text(340, 278, 'Camera Encoder', 20)
text(340, 305, 'Backbone + FPN', 16, 'normal', SUB)
arrow([(430, 190), (483, 190)])

# multi-scale camera feature sheets
sheet_stack(540, 190, 96, 82, 3, ORANGE_F, ORANGE_S)
text(545, 278, 'Multi-scale')
text(545, 305, 'Camera Feat.')

# PDF module (ours) -- label above so the radar-depth arrow can enter below
arrow([(612, 188), (648, 190)])
rrect(655, 140, 190, 100, PINK_F, PINK_S)
for i in range(3):
    poly([(706, 168 + i * 24), (794, 168 + i * 24), (784, 182 + i * 24), (696, 182 + i * 24)],
         '#F5B5B0', PINK_S, 2.2)
text(750, 122, 'Panoramic Depth Fusion (PDF)')
arrow([(845, 190), (913, 190)])

# view transform (frustum pyramid)
rrect(920, 140, 190, 100, ORANGE_F, ORANGE_S)
add(f'''<g stroke="#333333" stroke-width="2.2" fill="none">
<polygon points="1046,160 1082,173 1082,213 1046,226" fill="#E8E8E8"/>
<line x1="955" y1="193" x2="1046" y2="160"/>
<line x1="955" y1="193" x2="1082" y2="173"/>
<line x1="955" y1="193" x2="1082" y2="213"/>
<line x1="955" y1="193" x2="1046" y2="226"/>
</g>
<rect x="962" y="185" width="17" height="17" fill="#333333"/>''')
text(1015, 278, 'Camera-to-BEV')
text(1015, 305, 'View Transform')
text(1015, 330, 'LSS depth lift', 16, 'normal', SUB)
arrow([(1110, 190), (1180, 214)])

# camera BEV cube
iso_cube(1190, 275, 105, 75, 48, ORANGE_F, ORANGE_S)
text(1258, 318, 'Camera Feat.')
text(1258, 345, '(in BEV)')

# ------------------------------------------------ radar branch (row 2)
# radar icon: arcs + dot
for r in (14, 27, 40):
    add(f'<path d="M {68 + r} {660 - r} A {r} {r} 0 0 1 {68 + r} {660 + r}" '
        f'fill="none" stroke="{BLUE_S}" stroke-width="3.4"/>')
add(f'<circle cx="68" cy="660" r="6" fill="{BLUE_S}"/>')
text(120, 738, '4D Radar')
text(120, 765, 'Points')
arrow([(180, 650), (243, 650)])

# radar encoder
rrect(250, 600, 180, 100, BLUE_F, BLUE_S)
for i in range(3):
    for j in range(3):
        add(f'<rect x="{305 + i * 22}" y="{628 + j * 22}" width="17" height="17" rx="3" '
            f'fill="white" stroke="{BLUE_S}" stroke-width="2.2"/>')
text(330, 738, 'Radar Encoder', 20)
text(330, 763, 'Voxel Encoder +', 14, 'normal', SUB)
text(330, 782, 'SECONDFPN', 14, 'normal', SUB)
arrow([(430, 650), (482, 655)])

# radar voxel feature cube
iso_cube(496, 700, 80, 56, 40, BLUE_F, BLUE_S)
text(560, 738, 'Radar Voxel', 20)
text(560, 763, 'Feat.', 20)
arrow([(592, 655), (648, 650)])

# BEV collapse
rrect(660, 600, 150, 100, BLUE_F, BLUE_S)
poly([(683, 650), (733, 634), (785, 650), (735, 666)], '#333333', '#333333', 1.5)
arrow([(709, 614), (709, 638)], '#333333', 5)
arrow([(760, 688), (760, 664)], '#333333', 5)
text(735, 738, 'BEV Collapse')
text(735, 765, 'height → BEV', 16, 'normal', SUB)
arrow([(810, 650), (1000, 632)])

# radar BEV cube
iso_cube(1010, 655, 105, 75, 48, BLUE_F, BLUE_S)
text(1078, 738, 'Radar Feat.')
text(1078, 765, '(in BEV)')

# ------------------------------------------------ fusion (middle row)
arrow([(1330, 222), (1443, 398)])
arrow([(1152, 638), (1443, 445)])
rrect(1450, 375, 180, 90, GRAY_F, GRAY_S)
rrect(1482, 398, 38, 44, ORANGE_F, ORANGE_S, rx=8, sw=2.6)
rrect(1560, 398, 38, 44, BLUE_F, BLUE_S, rx=8, sw=2.6)
text(1540, 430, '+', 26)
text(1540, 505, 'RC-BEV Fusion')
text(1540, 532, 'concat + conv', 16, 'normal', SUB)
arrow([(1630, 420), (1683, 420)])

# multi-frame fused BEV cubes
for i in range(2, -1, -1):
    iso_cube(1700 + i * 20, 462 - i * 14, 78, 42, 34, '#EDE3F8' if i else '#DCD0F0',
             '#8E7CC3', grid=(i == 0), sw=2.6)
text(1790, 505, 'Fused BEV Feat.')
text(1790, 532, '(t−N … t)', 16, 'normal', SUB)
arrow([(1848, 420), (1908, 420)])

# DGTF (ours)
rrect(1915, 375, 200, 90, PINK_F, PINK_S)
poly([(1960, 405), (2070, 405), (2058, 420), (1948, 420)], '#F5B5B0', PINK_S, 2.2)
poly([(1960, 438), (2070, 438), (2058, 453), (1948, 453)], '#F5B5B0', PINK_S, 2.2)
add(f'''<path d="M 1985 372 C 1985 340, 2045 340, 2045 368" fill="none"
 stroke="#333333" stroke-width="3" marker-end="url(#arrBlk)"/>''')
text(2015, 330, 'h, z state (BPTT)', 15, 'normal', SUB)
text(2015, 505, 'Deformable Gated')
text(2015, 532, 'Temporal Fusion')
text(2015, 557, 'pose-free align · gated agg. (DGTF)', 15, 'normal', SUB)
arrow([(2115, 420), (2173, 420)])

# IGDR (ours)
rrect(2180, 375, 200, 90, PINK_F, PINK_S)
iso_cube(2212, 452, 55, 34, 26, '#F5B5B0', PINK_S, grid=True, sw=2)
for i in range(3):
    add(f'<rect x="{2300}" y="{395 + i * 22}" width="52" height="14" rx="4" '
        f'fill="#F5B5B0" stroke="{PINK_S}" stroke-width="2"/>')
text(2280, 505, 'Instance-Guided')
text(2280, 532, 'Dynamic Refinement')
text(2280, 557, 'foreground-gated fusion (IGDR)', 15, 'normal', SUB)
arrow([(2380, 420), (2438, 420)])

# detection head
rrect(2445, 375, 170, 90, GRAY_F, GRAY_S)
poly([(2480, 448), (2540, 432), (2596, 448), (2536, 464)], '#E8E8E8', '#888888', 2)
add(f'''<g stroke="#B85450" stroke-width="2.4" fill="none">
<polygon points="2500,440 2522,432 2536,438 2514,446"/>
<line x1="2500" y1="440" x2="2500" y2="428"/>
<line x1="2522" y1="432" x2="2522" y2="420"/>
<line x1="2536" y1="438" x2="2536" y2="426"/>
<line x1="2514" y1="446" x2="2514" y2="434"/>
</g>''')
text(2530, 505, '3D Detection Head')
text(2530, 532, 'Dynamic Anchor Head', 16, 'normal', SUB)
arrow([(2615, 420), (2673, 420)])

# output boxes icon
iso_cube(2690, 462, 62, 42, 30, '#EDEDED', '#999999', grid=True, sw=2.4)
add(f'''<g stroke="#B85450" stroke-width="2.6" fill="none">
<polygon points="2702,436 2738,424 2762,432 2726,444"/>
<line x1="2702" y1="436" x2="2702" y2="418"/>
<line x1="2738" y1="424" x2="2738" y2="406"/>
<line x1="2762" y1="432" x2="2762" y2="414"/>
<line x1="2726" y1="444" x2="2726" y2="426"/>
<polygon points="2702,418 2738,406 2762,414 2726,426"/>
</g>''')
text(2745, 505, '3D Boxes')

# ------------------------------------------------ 2D instance branch (dashed)
rrect(2180, 25, 200, 85, 'white', ORANGE_S, dash='8 6', sw=2.8)
text(2280, 62, '2D Instance Branch')
text(2280, 90, 'RPN + Mask Head', 15, 'normal', SUB)
arrow([(540, 100), (540, 70), (2172, 70)], ORANGE_S, 3, dashed=True, marker='arrOrg')
text(1250, 60, '2D features', 15, 'normal', SUB)
arrow([(2280, 110), (2280, 368)], ORANGE_S, 3, dashed=True, marker='arrOrg')
text(2292, 200, 'E: instance features', 15, 'normal', SUB, anchor='start')
text(2292, 226, 'S: BEV instance map', 15, 'normal', SUB, anchor='start')

# ------------------------------------------------ radar-depth dashed arrow (PDF)
arrow([(540, 606), (540, 490), (755, 490), (755, 248)], BLUE_S, 3, dashed=True, marker='arrBlu')
text(647, 478, 'Radar Depth', 15, 'normal', BLUE_S)

# ------------------------------------------------ legend
rrect(40, 845, 34, 24, ORANGE_F, ORANGE_S, rx=6, sw=2.4)
text(84, 863, 'Camera branch', 17, 'normal', TXT, anchor='start')
rrect(260, 845, 34, 24, BLUE_F, BLUE_S, rx=6, sw=2.4)
text(304, 863, 'Radar branch', 17, 'normal', TXT, anchor='start')
rrect(480, 845, 34, 24, PINK_F, PINK_S, rx=6, sw=2.4)
text(524, 863, 'Proposed modules (Ours)', 17, 'normal', TXT, anchor='start')
line(820, 857, 880, 857, BLUE_S, 3, dash='8 6')
text(890, 863, 'cross-modal input (radar depth)', 17, 'normal', TXT, anchor='start')
line(1230, 857, 1290, 857, ORANGE_S, 3, dash='8 6')
text(1300, 863, 'auxiliary path (2D instance branch)', 17, 'normal', TXT, anchor='start')

add('</svg>')

svg = '\n'.join(parts)
with open('/home/lurui/workspace/R4Det/assets/r4det_architecture.svg', 'w') as f:
    f.write(svg)
cairosvg.svg2png(bytestring=svg.encode(), write_to='/home/lurui/workspace/R4Det/assets/r4det_architecture.png',
                 output_width=W * 2, output_height=H * 2)
cairosvg.svg2png(bytestring=svg.encode(), write_to='/tmp/r4det_arch_view.png',
                 output_width=int(W * 0.75), output_height=int(H * 0.75))
cairosvg.svg2pdf(bytestring=svg.encode(), write_to='/home/lurui/workspace/R4Det/assets/r4det_architecture.pdf')
print('done')
