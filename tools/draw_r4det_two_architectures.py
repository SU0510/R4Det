#!/usr/bin/env python3
"""R4Det architecture figures: baseline_temporal + N4 Run-10 mainline.
N4 includes an expanded Motion-Aligned RSSM panel; no IGDR / fg supervision.
All arrows anchored to shape edges; automatic text-overflow checks."""
import cairosvg

ORANGE_F, ORANGE_S = '#FDEBC8', '#E8A33D'
BLUE_F, BLUE_S = '#DAE8FC', '#6C8EBF'
PURP_F, PURP_S = '#DCD0F0', '#8E7CC3'
PURP_L = '#EDE6F7'
GRAY_F, GRAY_S = '#F5F5F5', '#999999'
TXT = '#1A1A1A'
SUB = '#666666'
FONT = 'DejaVu Sans'

warnings = []


def rr(x, y, w, h, fill, stroke, rx=16, sw=3.5, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def poly(pts, fill, stroke, sw=3, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    p = ' '.join(f'{a:.1f},{b:.1f}' for a, b in pts)
    return (f'<polygon points="{p}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
            f'stroke-linejoin="round"{d}/>')


def ln(x1, y1, x2, y2, stroke, sw=2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{sw}"{d}/>')


def ar(pts, color='#1A1A1A', sw=4.5, dashed=False, marker='arrB'):
    d = ' stroke-dasharray="9 7"' if dashed else ''
    p = ' '.join(f'{a:.1f},{b:.1f}' for a, b in pts)
    return (f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}" '
            f'stroke-linejoin="round"{d} marker-end="url(#{marker})"/>')


def tx(x, y, s, size=22, weight='bold', fill=TXT, anchor='middle'):
    fw = 'bold' if weight == 'bold' else 'normal'
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{fw}" fill="{fill}" text-anchor="{anchor}">{s}</text>')


def defs():
    m = lambda i, c: (f'<marker id="{i}" viewBox="0 0 10 10" refX="8.5" refY="5" '
                      f'markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse">'
                      f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{c}"/></marker>')
    return ('<defs>' + m('arrB', '#1A1A1A') + m('arrBlu', BLUE_S) + m('arrOrg', ORANGE_S) +
            m('arrBlk', '#333333') + m('arrPur', PURP_S) + m('arrGry', GRAY_S) + '</defs>')


def cube(x, y, w, h, dep, fill, stroke, grid=True, sw=3, dash=None):
    sk, sk2 = dep * 0.62, dep * 0.40
    out = []
    top = [(x, y - h), (x + w, y - h), (x + w + sk, y - h - sk2), (x + sk, y - h - sk2)]
    side = [(x + w, y - h), (x + w + sk, y - h - sk2), (x + w + sk, y - sk2), (x + w, y)]
    front = [(x, y), (x + w, y), (x + w, y - h), (x, y - h)]
    for p in (top, side, front):
        out.append(poly(p, fill, stroke, sw, dash))
    if grid:
        nx, ny = 3, 2
        for i in range(1, nx):
            gx = x + w * i / nx
            out.append(ln(gx, y, gx, y - h, stroke, 1.6))
            out.append(ln(gx, y - h, gx + sk * i / nx, y - h - sk2 * i / nx, stroke, 1.6))
        for j in range(1, ny):
            gy = y - h * j / ny
            out.append(ln(x, gy, x + w, gy, stroke, 1.6))
            out.append(ln(x + w, gy, x + w + sk, gy - sk2, stroke, 1.6))
        for i in range(1, nx):
            gx = x + w + sk * i / nx
            out.append(ln(gx, y - h - sk2 * i / nx, gx, y - sk2 * i / nx, stroke, 1.6))
    return '\n'.join(out)


def sheets(cx, cy, w, h, n, fill, stroke):
    out = []
    x0 = cx - w / 2 - (n - 1) * 8
    y0 = cy + h / 2 + (n - 1) * 7
    for i in range(n - 1, -1, -1):
        x, y = x0 + i * 16, y0 - i * 14
        out.append(poly([(x, y), (x + w, y - 20), (x + w, y - 20 - h), (x, y - h)], fill, stroke, 2.6))
    return '\n'.join(out)


# ======================================================================
# FIG A — baseline_temporal
# ======================================================================
def fig_baseline():
    W, H = 2570, 880
    P = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
         f'<rect width="{W}" height="{H}" fill="white"/>', defs()]
    A = P.append

    # camera row
    A(poly([(55, 252), (115, 232), (115, 162), (55, 182)], ORANGE_F, ORANGE_S, 2.6))
    A(poly([(125, 188), (168, 198), (168, 232), (125, 242)], ORANGE_F, ORANGE_S, 2.6))
    A(tx(105, 298, 'Multi-view', 20)); A(tx(105, 324, 'Images', 20))
    A(ar([(178, 210), (242, 210)]))
    A(rr(248, 160, 200, 100, ORANGE_F, ORANGE_S))
    for i in range(2, -1, -1):
        A(rr(294 + i * 13, 192 - i * 10, 62, 52, 'white', ORANGE_S, rx=8, sw=2.4))
    A(tx(348, 298, 'Camera Encoder', 20))
    A(tx(348, 324, 'ResNet-50 + FPN', 15, 'normal', SUB))
    A(ar([(452, 210), (484, 210)]))
    A(sheets(560, 210, 96, 82, 3, ORANGE_F, ORANGE_S))
    A(tx(566, 298, 'Multi-scale', 20)); A(tx(566, 324, 'Camera Feat.', 20))
    A(ar([(650, 210), (676, 210)]))
    A(rr(685, 160, 190, 100, ORANGE_F, ORANGE_S))
    A('''<g stroke="#333333" stroke-width="2.2" fill="none">
<polygon points="812,180 848,193 848,233 812,246" fill="#E8E8E8"/>
<line x1="720" y1="213" x2="812" y2="180"/><line x1="720" y1="213" x2="848" y2="193"/>
<line x1="720" y1="213" x2="848" y2="233"/><line x1="720" y1="213" x2="812" y2="246"/>
</g><rect x="727" y="205" width="17" height="17" fill="#333333"/>''')
    A(tx(780, 298, 'Camera-to-BEV', 20)); A(tx(780, 324, 'View Transform', 20))
    A(tx(780, 349, 'LSS depth lift', 15, 'normal', SUB))
    A(ar([(879, 250), (956, 250)]))
    A(cube(960, 292, 105, 72, 46, ORANGE_F, ORANGE_S))
    A(tx(1028, 335, 'Camera Feat.', 20)); A(tx(1028, 361, '(in BEV)', 20))

    # radar row
    for r in (14, 27, 40):
        A(f'<path d="M {68 + r} {650 - r} A {r} {r} 0 0 1 {68 + r} {650 + r}" '
          f'fill="none" stroke="{BLUE_S}" stroke-width="3.4"/>')
    A(f'<circle cx="68" cy="650" r="6" fill="{BLUE_S}"/>')
    A(tx(120, 738, '4D Radar', 20)); A(tx(120, 764, 'Points', 20))
    A(ar([(178, 650), (242, 650)]))
    A(rr(248, 600, 200, 100, BLUE_F, BLUE_S))
    for i in range(3):
        for j in range(3):
            A(rr(318 + i * 22, 628 + j * 22, 17, 17, 'white', BLUE_S, rx=3, sw=2.2))
    A(tx(348, 738, 'Radar Encoder', 20))
    A(tx(348, 763, 'PillarFeatureNet +', 14, 'normal', SUB))
    A(tx(348, 782, 'Scatter + SECOND + FPN', 14, 'normal', SUB))
    A(ar([(452, 673), (503, 673)]))
    A(cube(507, 700, 80, 54, 38, BLUE_F, BLUE_S))
    A(tx(566, 738, 'Radar Voxel', 20)); A(tx(566, 764, 'Feat.', 20))
    A(ar([(620, 673), (676, 673)]))
    A(rr(680, 600, 170, 100, BLUE_F, BLUE_S))
    A(poly([(703, 650), (762, 634), (818, 650), (759, 666)], '#333333', '#333333', 1.5))
    A(ar([(731, 614), (731, 638)], '#333333', 5)); A(ar([(790, 688), (790, 664)], '#333333', 5))
    A(tx(765, 738, 'BEV Collapse', 20))
    A(tx(765, 763, 'height → BEV', 15, 'normal', SUB))
    A(ar([(854, 640), (1006, 640)]))
    A(cube(1010, 655, 105, 72, 46, BLUE_F, BLUE_S))
    A(tx(1080, 738, 'Radar Feat.', 20)); A(tx(1080, 764, '(in BEV)', 20))

    # fusion row
    A(ar([(1097, 256), (1420, 256), (1420, 402), (1451, 402)]))
    A(ar([(1147, 619), (1420, 619), (1420, 446), (1451, 446)]))
    A(rr(1455, 378, 185, 92, GRAY_F, GRAY_S))
    A(rr(1488, 401, 40, 46, ORANGE_F, ORANGE_S, rx=8, sw=2.6))
    A(rr(1568, 401, 40, 46, BLUE_F, BLUE_S, rx=8, sw=2.6))
    A(tx(1548, 434, '+', 26))
    A(tx(1547, 508, 'RC-BEV Fusion', 20))
    A(tx(1547, 534, 'concat + conv', 15, 'normal', SUB))
    A(ar([(1644, 448), (1706, 448)]))
    A(cube(1732, 454, 80, 44, 36, PURP_F, PURP_S, grid=False, sw=2.6))
    A(cube(1710, 470, 80, 44, 36, PURP_F, PURP_S, grid=True, sw=2.6))
    A(tx(1800, 508, 'Fused BEV Feat.', 20)); A(tx(1800, 534, '(t−1, t)', 16, 'normal', SUB))
    A(ar([(1838, 438), (1921, 438)]))
    A(ar([(1838, 462), (1921, 462)], PURP_S, 3, dashed=True, marker='arrPur'))
    A(tx(1878, 484, 'prev BEV', 14, 'normal', SUB))
    A(rr(1927, 378, 210, 92, PURP_F, PURP_S))
    A(poly([(1958, 410), (2072, 410), (2060, 424), (1946, 424)], '#C4B5E8', PURP_S, 2.2))
    A(poly([(1958, 440), (2072, 440), (2060, 454), (1946, 454)], '#C4B5E8', PURP_S, 2.2))
    A(f'''<path d="M 1990 402 C 1990 372, 2044 372, 2044 400" fill="none"
 stroke="#333333" stroke-width="3" marker-end="url(#arrBlk)"/>''')
    A(tx(2017, 362, 'motion offsets', 14, 'normal', SUB))
    A(tx(2032, 508, 'Temporal Fusion', 20))
    A(tx(2032, 534, 'deformable align + gated agg.', 15, 'normal', SUB))
    A(ar([(2141, 424), (2196, 424)]))
    A(rr(2202, 378, 175, 92, GRAY_F, GRAY_S))
    A(poly([(2238, 452), (2298, 436), (2354, 452), (2294, 468)], '#E8E8E8', '#888888', 2))
    A('''<g stroke="#B85450" stroke-width="2.4" fill="none">
<polygon points="2258,444 2280,436 2294,442 2272,450"/>
<line x1="2258" y1="444" x2="2258" y2="432"/><line x1="2280" y1="436" x2="2280" y2="424"/>
<line x1="2294" y1="442" x2="2294" y2="430"/><line x1="2272" y1="450" x2="2272" y2="438"/>
</g>''')
    A(tx(2289, 508, '3D Detection', 20))
    A(tx(2289, 534, 'Anchor3DHead', 15, 'normal', SUB))
    A(ar([(2381, 444), (2446, 444)]))
    A(cube(2450, 464, 58, 40, 28, '#EDEDED', '#999999', grid=True, sw=2.4))
    A('''<g stroke="#B85450" stroke-width="2.6" fill="none">
<polygon points="2460,438 2494,426 2517,434 2483,446"/>
<line x1="2460" y1="438" x2="2460" y2="421"/><line x1="2494" y1="426" x2="2494" y2="409"/>
<line x1="2517" y1="434" x2="2517" y2="417"/><line x1="2483" y1="446" x2="2483" y2="429"/>
<polygon points="2460,421 2494,409 2517,417 2483,429"/>
</g>''')
    A(tx(2505, 508, '3D Boxes', 20))

    # legend
    A(rr(40, 822, 34, 24, ORANGE_F, ORANGE_S, rx=6, sw=2.4))
    A(tx(84, 840, 'Camera branch', 17, 'normal', TXT, anchor='start'))
    A(rr(260, 822, 34, 24, BLUE_F, BLUE_S, rx=6, sw=2.4))
    A(tx(304, 840, 'Radar branch', 17, 'normal', TXT, anchor='start'))
    A(rr(480, 822, 34, 24, PURP_F, PURP_S, rx=6, sw=2.4))
    A(tx(524, 840, 'Temporal fusion module', 17, 'normal', TXT, anchor='start'))
    A(ln(810, 834, 870, 834, PURP_S, 3, dash='8 6'))
    A(tx(880, 840, 'recurrent prev-frame path', 17, 'normal', TXT, anchor='start'))
    A(tx(2530, 840, 'Baseline (temporal, no proposed modules)', 17, 'normal', SUB, anchor='end'))
    P.append('</svg>')
    return '\n'.join(P), W, H, 'r4det_baseline_temporal'


# ======================================================================
# FIG B — N4 (fgfull) with expanded Motion-Aligned RSSM panel
# ======================================================================
def fig_n4():
    W, H = 2900, 1180
    P = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
         f'<rect width="{W}" height="{H}" fill="white"/>', defs()]
    A = P.append

    # ---- camera row (y 250)
    A(poly([(55, 292), (115, 272), (115, 202), (55, 222)], ORANGE_F, ORANGE_S, 2.6))
    A(poly([(125, 228), (168, 238), (168, 272), (125, 282)], ORANGE_F, ORANGE_S, 2.6))
    A(tx(105, 338, 'Multi-view', 20)); A(tx(105, 364, 'Images', 20))
    A(ar([(178, 250), (242, 250)]))
    A(rr(248, 200, 200, 100, ORANGE_F, ORANGE_S))
    for i in range(2, -1, -1):
        A(rr(294 + i * 13, 232 - i * 10, 62, 52, 'white', ORANGE_S, rx=8, sw=2.4))
    A(tx(348, 338, 'Camera Encoder', 20))
    A(tx(348, 364, 'ResNet-50 + FPN', 15, 'normal', SUB))
    A(ar([(452, 250), (484, 250)]))
    A(sheets(560, 250, 96, 82, 3, ORANGE_F, ORANGE_S))
    A(tx(566, 338, 'Multi-scale', 20)); A(tx(566, 364, 'Camera Feat.', 20))
    A(ar([(650, 250), (676, 250)]))

    # depth net
    A(rr(685, 200, 190, 100, ORANGE_F, ORANGE_S))
    for i in range(3):
        A(poly([(736, 228 + i * 24), (824, 228 + i * 24), (814, 242 + i * 24), (726, 242 + i * 24)],
               '#F5B5B0', ORANGE_S, 2.2))
    A(tx(780, 338, 'Depth Net', 20))
    A(tx(780, 363, 'GeometryDepth_Net', 14, 'normal', SUB))
    A(ar([(879, 250), (952, 250)]))

    # view transform
    A(rr(950, 200, 190, 100, ORANGE_F, ORANGE_S))
    A('''<g stroke="#333333" stroke-width="2.2" fill="none">
<polygon points="1078,220 1114,233 1114,273 1078,286" fill="#E8E8E8"/>
<line x1="985" y1="253" x2="1078" y2="220"/><line x1="985" y1="253" x2="1114" y2="233"/>
<line x1="985" y1="253" x2="1114" y2="273"/><line x1="985" y1="253" x2="1078" y2="286"/>
</g><rect x="992" y="245" width="17" height="17" fill="#333333"/>''')
    A(tx(1045, 338, 'Camera-to-BEV', 20)); A(tx(1045, 364, 'View Transform', 20))
    A(tx(1045, 389, 'LSS depth lift', 15, 'normal', SUB))
    A(ar([(1144, 280), (1216, 280)]))
    A(cube(1220, 335, 105, 72, 46, ORANGE_F, ORANGE_S))
    A(tx(1288, 378, 'Camera Feat.', 20)); A(tx(1288, 404, '(in BEV)', 20))

    # ---- radar row (y 710)
    for r in (14, 27, 40):
        A(f'<path d="M {68 + r} {710 - r} A {r} {r} 0 0 1 {68 + r} {710 + r}" '
          f'fill="none" stroke="{BLUE_S}" stroke-width="3.4"/>')
    A(f'<circle cx="68" cy="710" r="6" fill="{BLUE_S}"/>')
    A(tx(120, 798, '4D Radar', 20)); A(tx(120, 824, 'Points', 20))
    A(ar([(178, 710), (242, 710)]))
    A(rr(248, 660, 200, 100, BLUE_F, BLUE_S))
    for i in range(3):
        for j in range(3):
            A(rr(318 + i * 22, 688 + j * 22, 17, 17, 'white', BLUE_S, rx=3, sw=2.2))
    A(tx(348, 798, 'Radar Encoder', 20))
    A(tx(348, 823, 'PillarFeatureNet +', 14, 'normal', SUB))
    A(tx(348, 842, 'Scatter + SECOND + FPN', 14, 'normal', SUB))
    A(ar([(452, 733), (503, 733)]))
    A(cube(507, 760, 80, 54, 38, BLUE_F, BLUE_S))
    A(tx(566, 798, 'Radar Voxel', 20)); A(tx(566, 824, 'Feat.', 20))
    A(ar([(620, 733), (676, 733)]))
    A(rr(680, 660, 170, 100, BLUE_F, BLUE_S))
    A(poly([(703, 710), (762, 694), (818, 710), (759, 726)], '#333333', '#333333', 1.5))
    A(ar([(731, 674), (731, 698)], '#333333', 5)); A(ar([(790, 748), (790, 724)], '#333333', 5))
    A(tx(765, 798, 'BEV Collapse', 20))
    A(tx(765, 823, 'height → BEV', 15, 'normal', SUB))
    A(ar([(854, 690), (1066, 690)]))
    A(cube(1070, 715, 105, 72, 46, BLUE_F, BLUE_S))
    A(tx(1138, 798, 'Radar Feat.', 20)); A(tx(1138, 824, '(in BEV)', 20))

    # ---- fusion row (y 484)
    A(ar([(1357, 299), (1420, 299), (1420, 462), (1446, 462)]))
    A(ar([(1207, 679), (1420, 679), (1420, 506), (1446, 506)]))
    A(rr(1450, 438, 185, 92, GRAY_F, GRAY_S))
    A(rr(1483, 461, 40, 46, ORANGE_F, ORANGE_S, rx=8, sw=2.6))
    A(rr(1563, 461, 40, 46, BLUE_F, BLUE_S, rx=8, sw=2.6))
    A(tx(1543, 494, '+', 26))
    A(tx(1542, 568, 'RC-BEV Fusion', 20))
    A(tx(1542, 594, 'concat + conv', 15, 'normal', SUB))
    A(ar([(1639, 490), (1701, 490)]))
    A(cube(1745, 494, 78, 42, 34, PURP_F, PURP_S, grid=False, sw=2.6))
    A(cube(1725, 508, 78, 42, 34, PURP_F, PURP_S, grid=False, sw=2.6))
    A(cube(1705, 522, 78, 42, 34, PURP_F, PURP_S, grid=True, sw=2.6))
    A(tx(1790, 568, 'Fused BEV Feat.', 20)); A(tx(1790, 594, '(t−N … t)', 16, 'normal', SUB))

    # DGTF (pipeline box)
    A(ar([(1850, 484), (1911, 484)]))
    A(rr(1915, 428, 230, 112, PURP_F, PURP_S))
    A(poly([(1945, 485), (2055, 485), (2043, 499), (1933, 499)], '#C4B5E8', PURP_S, 2.2))
    A(poly([(1945, 513), (2055, 513), (2043, 527), (1933, 527)], '#C4B5E8', PURP_S, 2.2))
    A(f'<circle cx="2108" cy="480" r="16" fill="white" stroke="{PURP_S}" stroke-width="2.6"/>')
    A(tx(2108, 486, 'z', 15))
    A(f'''<path d="M 1975 424 C 1975 394, 2090 394, 2090 422" fill="none"
 stroke="#333333" stroke-width="3" marker-end="url(#arrBlk)"/>''')
    A(tx(2032, 384, 'h, z state (BPTT)', 14, 'normal', SUB))
    A(tx(2030, 568, 'Motion-Aligned', 20))
    A(tx(2030, 594, 'RSSM Temporal Fusion', 15, 'normal', SUB))
    A(tx(2030, 619, 'deformable align · gated agg.', 15, 'normal', SUB))

    A(ar([(2149, 484), (2451, 484)]))

    A(ln(2152, 540, 2152, 640, PURP_S, 2, dash='6 6'))
    A(tx(2164, 637, 'zoom in', 13, 'normal', PURP_S, anchor='start'))

    # detection head
    A(rr(2457, 438, 175, 92, GRAY_F, GRAY_S))
    A(poly([(2493, 512), (2553, 496), (2609, 512), (2549, 528)], '#E8E8E8', '#888888', 2))
    A('''<g stroke="#B85450" stroke-width="2.4" fill="none">
<polygon points="2513,504 2535,496 2549,502 2527,510"/>
<line x1="2513" y1="504" x2="2513" y2="492"/><line x1="2535" y1="496" x2="2535" y2="484"/>
<line x1="2549" y1="502" x2="2549" y2="490"/><line x1="2527" y1="510" x2="2527" y2="498"/>
</g>''')
    A(tx(2544, 568, '3D Detection', 20))
    A(tx(2544, 594, 'Anchor3DHead (head-v2)', 15, 'normal', SUB))
    A(ar([(2636, 500), (2699, 500)]))
    A(cube(2703, 524, 58, 40, 28, '#EDEDED', '#999999', grid=True, sw=2.4))
    A('''<g stroke="#B85450" stroke-width="2.6" fill="none">
<polygon points="2715,498 2749,486 2772,494 2738,506"/>
<line x1="2715" y1="498" x2="2715" y2="481"/><line x1="2749" y1="486" x2="2749" y2="469"/>
<line x1="2772" y1="494" x2="2772" y2="477"/><line x1="2738" y1="506" x2="2738" y2="489"/>
<polygon points="2715,481 2749,469 2772,477 2738,489"/>
</g>''')
    A(tx(2757, 568, '3D Boxes', 20))

    # ================= expanded Motion-Aligned RSSM panel ================
    A(rr(1560, 640, 1300, 400, '#FCFBFE', PURP_S, rx=18, sw=3))
    A(tx(2210, 676, 'Motion-Aligned RSSM Temporal Fusion', 22))
    # inputs
    A(cube(1610, 795, 70, 40, 26, PURP_F, PURP_S, grid=True, sw=2.2, dash='7 5'))
    A(tx(1655, 724, 'h(t−1), z(t−1)', 13, 'normal', SUB))
    A(cube(1700, 950, 70, 40, 26, PURP_F, PURP_S, grid=True, sw=2.2))
    A(tx(1745, 1002, 'x_t (fused BEV)', 13, 'normal', SUB))
    # row 1
    A(ar([(1700, 775), (1761, 775)]))
    A(rr(1765, 735, 185, 80, PURP_L, PURP_S))
    A(tx(1857, 772, 'Deformable', 17)); A(tx(1857, 796, 'Align', 17))
    A(ar([(1954, 772), (2009, 772)]))
    A(rr(2015, 735, 175, 80, PURP_L, PURP_S))
    A(tx(2102, 772, 'Transition', 17)); A(tx(2102, 796, 'GRU', 17))
    A(ar([(2194, 772), (2349, 772)]))
    A(rr(2355, 735, 190, 80, PURP_L, PURP_S))
    A(tx(2450, 768, 'Gated', 17)); A(tx(2450, 792, 'Aggregation', 17))
    A(ar([(2549, 805), (2601, 805)]))
    A(cube(2605, 830, 64, 42, 26, PURP_F, PURP_S, grid=True, sw=2.2))
    A(ar([(2692, 809), (2856, 809)]))
    A(tx(2770, 795, 'to head', 13, 'normal', SUB))
    # row 2
    A(ar([(1790, 920), (1801, 920)]))
    A(rr(1805, 865, 170, 80, PURP_L, PURP_S))
    A(tx(1890, 902, 'Encoder', 17)); A(tx(1890, 926, '(e_t)', 15, 'normal', SUB))
    A(ar([(1979, 905), (2214, 905)]))
    A(tx(2000, 893, 'e_t', 13, 'normal', SUB))
    A(rr(2220, 865, 175, 80, PURP_L, PURP_S))
    A(tx(2307, 891, 'Posterior q', 16))
    A(tx(2307, 913, 'Prior p', 14, 'normal', SUB))
    A(tx(2307, 936, 'KL(q‖p)', 12, 'normal', SUB))
    A(ar([(2399, 905), (2444, 905)]))
    A(f'<circle cx="2472" cy="905" r="24" fill="white" stroke="{PURP_S}" stroke-width="3"/>')
    A(tx(2472, 912, 'z_t', 16))
    A(ar([(2472, 877), (2472, 819)]))
    # offsets dashed: x_t -> Deformable Align
    A(ar([(1775, 896), (1775, 819)], PURP_S, 2.5, dashed=True, marker='arrPur'))
    A(tx(1788, 860, 'motion offsets', 12, 'normal', SUB, anchor='start'))
    # BPTT loop
    A(ar([(2637, 838), (2637, 1020), (1655, 1020), (1655, 801)], PURP_S, 3, dashed=True, marker='arrPur'))
    A(tx(2146, 1010, 'h_t, z_t → next frame (BPTT)', 13, 'normal', SUB))
    A(tx(2845, 665, 'expanded', 13, 'normal', SUB, anchor='end'))

    # ---- legend (y 1120)
    A(rr(40, 1112, 34, 24, ORANGE_F, ORANGE_S, rx=6, sw=2.4))
    A(tx(84, 1130, 'Camera branch', 17, 'normal', TXT, anchor='start'))
    A(rr(260, 1112, 34, 24, BLUE_F, BLUE_S, rx=6, sw=2.4))
    A(tx(304, 1130, 'Radar branch', 17, 'normal', TXT, anchor='start'))
    A(rr(480, 1112, 34, 24, PURP_F, PURP_S, rx=6, sw=2.4))
    A(tx(524, 1130, 'Proposed temporal/refinement modules', 17, 'normal', TXT, anchor='start'))
    A(tx(2860, 1130, 'Run 10 mainline (pretrained RSSM + head-v2)', 17, 'normal', SUB, anchor='end'))
    P.append('</svg>')
    return '\n'.join(P), W, H, 'r4det_n4_run10'


def export(svg, W, H, name):
    with open(f'/home/lurui/workspace/R4Det/assets/{name}.svg', 'w') as f:
        f.write(svg)
    cairosvg.svg2png(bytestring=svg.encode(),
                     write_to=f'/home/lurui/workspace/R4Det/assets/{name}.png',
                     output_width=W * 2, output_height=H * 2)
    cairosvg.svg2png(bytestring=svg.encode(),
                     write_to=f'/tmp/{name}_view.png',
                     output_width=int(W * 0.7), output_height=int(H * 0.7))
    cairosvg.svg2pdf(bytestring=svg.encode(),
                     write_to=f'/home/lurui/workspace/R4Det/assets/{name}.pdf')


if __name__ == '__main__':
    for fn in (fig_baseline, fig_n4):
        svg, W, H, name = fn()
        export(svg, W, H, name)
        print(f'--- {name} exported')
