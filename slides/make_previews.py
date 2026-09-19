#!/usr/bin/env python
"""Emit a faithful HTML mirror of each composed slide for visual QA (v2).

Mirrors compose_deck_local.py v2 geometry exactly (which adapts the skill's
compose_evidence_deck.mjs), so screenshots reflect the real pptx layout.
"""
import json
import math
import re
from html import escape
from pathlib import Path

W, H, M = 1280, 720, 72
C = {"bg": "#F7F9F6", "ink": "#17372F", "body": "#263732", "muted": "#62736C",
     "green": "#2D6B5B", "green2": "#3E8571", "light": "#E5EFE9", "line": "#C9D8D0",
     "amber": "#F6D676", "amberText": "#705717", "amberBg": "#FFF9E5",
     "white": "#FFFFFF", "card": "#FFFFFF", "cardLine": "#D8E4DD"}
FONT_URL = "file:///home/lurui/.local/share/fonts/NotoSansCJKsc-Regular.otf"
FONT_URL_B = "file:///home/lurui/.local/share/fonts/NotoSansCJKsc-Bold.otf"

CSS = f"""
@page {{ size: 1280px 720px; margin: 0; }}
@font-face {{ font-family: DeckCJK; src: url('{FONT_URL}'); font-weight: normal; }}
@font-face {{ font-family: DeckCJK; src: url('{FONT_URL_B}'); font-weight: bold; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ width:{W}px; height:{H}px; background:{C['bg']};
       font-family:'DeckCJK','Microsoft YaHei',sans-serif; overflow:hidden; position:relative; }}
.t {{ position:absolute; white-space:pre-wrap; word-break:break-word; line-height:1.25; }}
.rule {{ position:absolute; height:1px; background:{C['line']}; }}
.badge {{ position:absolute; background:{C['light']}; border-radius:13px; }}
.dot {{ position:absolute; width:7px; height:7px; border-radius:50%; background:{C['green']}; }}
.fbox {{ position:absolute; border:1px solid {C['line']}; background:{C['white']};
         display:flex; align-items:center; justify-content:center; overflow:hidden; }}
.fbox img {{ max-width:100%; max-height:100%; }}
.amberbox {{ position:absolute; background:{C['amberBg']}; border:1px solid {C['amber']}; }}
.card {{ position:absolute; background:{C['card']}; border:1px solid {C['cardLine']}; border-radius:10px; }}
.accent {{ position:absolute; background:{C['green']}; }}
.chipnum {{ position:absolute; background:{C['green']}; border-radius:50%; color:#fff;
            text-align:center; line-height:56px; font-weight:bold; }}
table.deck {{ position:absolute; border-collapse:collapse; }}
table.deck td, table.deck th {{ border:none; padding:0 8px; }}
"""


def clean(v, fallback=""):
    out = re.sub(r"\s+", " ", str(v if v is not None else fallback)).strip()
    return out or fallback


def compact(value, base, minimum, comfortable):
    n = len(clean(value))
    return base if n <= comfortable else max(minimum, base - math.ceil((n - comfortable) / 8))


def t(content, left, top, width, height, size, color, bold=False, align="left"):
    weight = "bold" if bold else "normal"
    return (f'<div class="t" style="left:{left}px;top:{top}px;width:{width}px;height:{height}px;'
            f'font-size:{size}px;color:{color};font-weight:{weight};text-align:{align};">{escape(clean(content))}</div>')


def header(page, total, section, title):
    chip_w = max(96, 28 + 16 * len(clean(section or "网络组会")))
    return (t(f"{page:02d}", M, 36, 40, 18, 12, C["green"], True)
            + f'<div class="badge" style="left:124px;top:30px;width:{chip_w}px;height:28px;"></div>'
            + t(section or "网络组会", 124, 36, chip_w, 18, 12, C["green"], True, "center")
            + t(f"{page} / {total}", 1080, 36, 128, 18, 12, C["muted"], False, "right")
            + f'<div class="accent" style="left:{M}px;top:64px;width:6px;height:34px;"></div>'
            + t(title, M + 18, 64, 1080, 36, compact(title, 28, 21, 30), C["ink"], True)
            + f'<div class="rule" style="left:{M}px;top:112px;width:{W-2*M}px;"></div>')


def footer(locator, citation):
    rule = f'<div class="rule" style="left:{M}px;top:672px;width:{W-2*M}px;"></div>'
    return rule + t(f"{citation} · {clean(locator)}", M, 686, W - 2 * M, 16, 10, C["muted"])


def bullet_html(content, left, top, width, height, size):
    return (f'<div class="dot" style="left:{left}px;top:{top+10}px;"></div>'
            + t(content, left + 22, top, width - 22, height,
                compact(content, size, max(14, size - 4), 32), C["body"]))


def kpi_html(items, left=M, top=136, card_w=272, gap=24, height=96):
    out = ""
    for i, k in enumerate(items[:4]):
        x = left + i * (card_w + gap)
        out += f'<div class="card" style="left:{x}px;top:{top}px;width:{card_w}px;height:{height}px;"></div>'
        out += f'<div class="accent" style="left:{x}px;top:{top}px;width:5px;height:{height}px;"></div>'
        out += t(k.get("value", ""), x + 20, top + 14, card_w - 36, 40,
                 compact(k.get("value", ""), 26, 18, 12), C["ink"], True)
        out += t(k.get("label", ""), x + 20, top + 58, card_w - 36, 30, 12, C["muted"])
    return out


def render(item, page, total, citation):
    kind = item["type"]
    inner = ""
    if kind == "title":
        inner += f'<div class="accent" style="left:0;top:0;width:14px;height:{H}px;"></div>'
        inner += t("网络进展组会 · R4Det + RSSM", 86, 84, 640, 22, 14, C["green"], True)
        title = item.get("title", "")
        inner += t(title, 86, 124, 1080, 60, compact(title, 38, 28, 26), C["ink"], True)
        inner += f'<div style="position:absolute;left:88px;top:208px;width:120px;height:8px;background:{C["amber"]};"></div>'
        if item.get("body"):
            inner += t(item["body"], 86, 240, 1000, 44, 16, C["muted"])
        kpis = item.get("kpi") or []
        card_w, gap, height = 352, 28, 132
        for i, k in enumerate(kpis[:3]):
            x = 86 + i * (card_w + gap)
            inner += f'<div class="card" style="left:{x}px;top:316px;width:{card_w}px;height:{height}px;"></div>'
            inner += f'<div class="accent" style="left:{x}px;top:316px;width:5px;height:{height}px;"></div>'
            inner += t(k.get("value", ""), x + 24, 338, card_w - 44, 46,
                       compact(k.get("value", ""), 28, 20, 12), C["ink"], True)
            inner += t(k.get("label", ""), x + 24, 396, card_w - 44, 34, 12.5, C["muted"])
        if item.get("bullets"):
            inner += t(item["bullets"][0], 86, 496, 1080, 30, 15, C["body"])
        inner += t("汇报约 15 分钟 · 19 页 · 每页页脚标注证据出处", 86, 548, 1080, 26, 12.5, C["muted"])
        inner += footer(item.get("source", ""), citation)
    elif kind == "figure":
        inner += header(page, total, item.get("section"), item["title"])
        aspect = float(item.get("image_aspect") or 0)
        if aspect >= 1.45:
            inner += f'<div class="fbox" style="left:72px;top:136px;width:1136px;height:330px;"><img src="{item["asset"]}"></div>'
            inner += t(f'{item["figure_label"]} · 报告 §{item["pdf_page"]}', 72, 476, 1136, 16, 10, C["muted"], align="center")
            inner += f'<div class="badge" style="left:72px;top:502px;width:112px;height:26px;"></div>'
            inner += t("本文证据", 84, 507, 88, 16, 11, C["green"], True)
            for i, b in enumerate(item.get("bullets", [])[:3]):
                inner += bullet_html(b, 72, 536 + i * 29, 1128, 28, 17)
            inner += f'<div class="amberbox" style="left:72px;top:625px;width:1136px;height:39px;"></div>'
            inner += t(item.get("caveat", ""), 90, 635, 1098, 20, 14, C["amberText"], True)
        else:
            inner += f'<div class="fbox" style="left:72px;top:136px;width:628px;height:436px;"><img src="{item["asset"]}"></div>'
            inner += t(f'{item["figure_label"]} · 报告 §{item["pdf_page"]}', 72, 586, 628, 16, 10, C["muted"], align="center")
            inner += f'<div class="badge" style="left:770px;top:150px;width:112px;height:26px;"></div>'
            inner += t("本文证据", 782, 155, 88, 16, 11, C["green"], True)
            for i, b in enumerate(item.get("bullets", [])[:3]):
                inner += bullet_html(b, 770, 210 + i * 86, 430, 56, 18)
            inner += f'<div class="amberbox" style="left:770px;top:498px;width:438px;height:78px;"></div>'
            inner += t(item.get("caveat", ""), 790, 514, 398, 46, 16, C["amberText"], True)
        inner += footer(item.get("source", ""), citation)
    elif item.get("table"):
        inner += header(page, total, item.get("section"), item["title"])
        tb = item["table"]
        cols, rows = tb["cols"], tb["rows"]
        widths = tb["col_widths"]
        left, top = 72, 142
        row_h = 34 if len(rows) > 8 else 38
        inner += (f'<div class="accent" style="left:{left}px;top:{top}px;width:1136px;height:{row_h}px;"></div>')
        x = left
        for j, name in enumerate(cols):
            inner += t(name, x + 8, top + 8, widths[j] - 16, 20, 14, "#FFFFFF", True,
                       "left" if j == 0 else "center")
            x += widths[j]
        y = top + row_h
        for i, row in enumerate(rows, start=1):
            if i % 2 == 0:
                inner += f'<div style="position:absolute;left:{left}px;top:{y}px;width:1136px;height:{row_h}px;background:{C["light"]};"></div>'
            if i == len(rows):
                inner += f'<div class="rule" style="left:{left}px;top:{y}px;width:1136px;"></div>'
            x = left
            for j, v in enumerate(row):
                bold = (j == 0) or (i == len(rows))
                inner += t(v, x + 8, y + 7, widths[j] - 16, 22, 13, C["ink"] if bold else C["body"],
                           bold, "left" if j == 0 else "center")
                x += widths[j]
            y += row_h
        below = y + 18
        for i, b in enumerate(item.get("bullets", [])[:3]):
            inner += bullet_html(b, 72, below + i * 30, 1128, 28, 16)
        caveat_y = below + 30 * len(item.get("bullets", [])[:3]) + 8
        if item.get("caveat"):
            inner += f'<div class="amberbox" style="left:72px;top:{caveat_y}px;width:1136px;height:36px;"></div>'
            inner += t(item["caveat"], 90, caveat_y + 8, 1098, 20, 14, C["amberText"], True)
        inner += footer(item.get("source", ""), citation)
    elif item.get("agenda"):
        inner += header(page, total, item.get("section"), item["title"])
        entries = item.get("agenda", [])
        col_w, row_h = 546, 108
        for i, e in enumerate(entries[:6]):
            col, row = i % 2, i // 2
            x = M + col * (col_w + 44)
            y = 160 + row * (row_h + 26)
            inner += f'<div class="card" style="left:{x}px;top:{y}px;width:{col_w}px;height:{row_h}px;"></div>'
            inner += f'<div class="chipnum" style="left:{x+18}px;top:{y+26}px;width:56px;height:56px;font-size:18px;">{i+1:02d}</div>'
            inner += t(e.get("name", ""), x + 92, y + 20, col_w - 110, 30, 18, C["ink"], True)
            inner += t(e.get("desc", ""), x + 92, y + 56, col_w - 110, 40, 12.5, C["muted"])
        inner += footer(item.get("source", ""), citation)
    elif item.get("closing"):
        inner += f'<div class="accent" style="left:0;top:0;width:14px;height:{H}px;"></div>'
        inner += t(item.get("title", "谢谢 · 欢迎讨论"), 86, 200, 1080, 70, 36, C["ink"], True, "center")
        if item.get("body"):
            inner += t(item["body"], 86, 292, 1080, 36, 17, C["muted"], False, "center")
        chips = item.get("bullets") or []
        chip_w = 340
        total_w = len(chips[:3]) * chip_w + (len(chips[:3]) - 1) * 28
        x0 = (W - total_w) // 2
        for i, chip in enumerate(chips[:3]):
            x = x0 + i * (chip_w + 28)
            inner += f'<div class="card" style="left:{x}px;top:366px;width:{chip_w}px;height:110px;"></div>'
            inner += f'<div class="accent" style="left:{x}px;top:366px;width:5px;height:110px;"></div>'
            inner += t(chip, x + 20, 382, chip_w - 40, 80, 14, C["body"])
        inner += footer(item.get("source", ""), citation)
    else:
        inner += header(page, total, item.get("section"), item["title"])
        body = item.get("body")
        kpis = item.get("kpi") or []
        if kpis:
            inner += kpi_html(kpis)
        body_top = 260 if kpis else 168
        if body:
            inner += t(body, M, body_top, 1080, 110, compact(body, 24, 18, 52), C["ink"], True)
        start = (body_top + 128) if body else (body_top + 24)
        step = 96 if not body else 92
        for i, b in enumerate(item.get("bullets", [])[:3]):
            inner += bullet_html(b, M, start + i * step, 1080, 64, 20)
        inner += footer(item.get("source", ""), citation)
    return f'<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>{inner}</body></html>'


def main():
    spec_path = Path(__file__).resolve().parent / "deck-spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    out_dir = Path(__file__).resolve().parent / "preview"
    out_dir.mkdir(exist_ok=True)
    citation = clean(spec["paper"].get("citation", spec["paper"]["title"]))
    total = len(spec["slides"])
    for i, item in enumerate(spec["slides"], start=1):
        html = render(item, i, total, citation)
        html = html.replace('src="assets/', 'src="../assets/')
        (out_dir / f"slide_{i:02d}.html").write_text(html, encoding="utf-8")
    print(f"wrote {total} preview pages to {out_dir}")


if __name__ == "__main__":
    main()
