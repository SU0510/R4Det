#!/usr/bin/env python
"""Group-meeting deck composer (plain style, mirrors the user's example decks).

10x7.5in 4:3, white background, Microsoft YaHei, black text, thin-border tables,
12pt source line on content pages, page number bottom-right. Emits the pptx AND
a same-geometry HTML mirror per slide for visual QA.
"""
import html
import json
import re
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
from pptx.oxml.ns import qn

W_IN, H_IN = 10.0, 7.5
PX = 96  # css px per inch
FONT = "Microsoft YaHei"
BLACK = RGBColor(0x20, 0x20, 0x20)
RED = RGBColor(0xC0, 0x30, 0x30)
GRAY = RGBColor(0x59, 0x59, 0x59)

A = "gm_assets"
SRC_LINE = "实验记录：docs/training_runs_full.md · work_dirs 训练日志（2026.7 – 2026.9）"


def esc(s):
    return html.escape(str(s))


class Slide:
    """Collects draw ops, renders to pptx shapes and to an html fragment."""

    def __init__(self, prs, mirror_list, page):
        self.prs = prs
        self.slide = prs.slides.add_slide(prs.slide_layouts[6])
        self.mirror = []
        self.page = page
        mirror_list.append(self.mirror)

    # -- primitives ------------------------------------------------------
    def _mirror_text(self, l, t, w, h, runs, size, align, line_h=1.32):
        span = "".join(
            f'<b style="color:{c}">{esc(x)}</b>' if b else
            f'<span style="color:{c}">{esc(x)}</span>'
            for x, b, c in runs)
        color = runs[0][2] if runs else "#202020"
        self.mirror.append(
            f'<div class="t" style="left:{l*PX:.0f}px;top:{t*PX:.0f}px;width:{w*PX:.0f}px;'
            f'height:{h*PX:.0f}px;font-size:{size}pt;color:{color};'
            f'text-align:{align};line-height:{line_h};">{span}</div>')

    def text(self, l, t, w, h, runs, size=14, align="left", color="202020", line_h=None):
        """runs: str or list of (text, bold) or list of (text, bold, hexcolor)."""
        if isinstance(runs, str):
            runs = [(runs, False)]
        runs = [(x, b, c if len(rl) > 2 else "202020") for rl in [(r if isinstance(r, tuple) else (r, False)) for r in runs]
                for x, b, c in [(rl[0], rl[1], rl[2] if len(rl) > 2 else "202020")]]
        box = self.slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER,
                       "right": PP_ALIGN.RIGHT}[align]
        for x, b, c in runs:
            r = p.add_run()
            r.text = x
            r.font.name = FONT
            r.font.size = Pt(size)
            r.font.bold = b
            r.font.color.rgb = RGBColor.from_string(c)
            rPr = r._r.get_or_add_rPr()
            ea = rPr.find(qn("a:ea"))
            if ea is None:
                ea = rPr.makeelement(qn("a:ea"), {})
                rPr.append(ea)
            ea.set("typeface", FONT)
        hexc = runs[0][2] if runs else color
        self._mirror_text(l, t, w, h, [(x, b, ("#" + c)) for x, b, c in runs], size, align,
                          line_h or 1.32)

    def para_list(self, l, t, w, h, lines, size=14, gap_pt=6):
        """Multiple paragraphs; each line: str or (text,bold) or (text,bold,color)."""
        box = self.slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        mirror_rows = []
        for i, line in enumerate(lines):
            if isinstance(line, str):
                line = [(line, False)]
            elif isinstance(line, tuple):
                line = [line]
            norm = []
            for r in line:
                if isinstance(r, str):
                    norm.append((r, False, "202020"))
                else:
                    norm.append((r[0], r[1], r[2] if len(r) > 2 else "202020"))
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            if i > 0:
                p.space_before = Pt(gap_pt)
            for x, b, c in norm:
                r = p.add_run()
                r.text = x
                r.font.name = FONT
                r.font.size = Pt(size)
                r.font.bold = b
                r.font.color.rgb = RGBColor.from_string(c)
                rPr = r._r.get_or_add_rPr()
                ea = rPr.find(qn("a:ea"))
                if ea is None:
                    ea = rPr.makeelement(qn("a:ea"), {})
                    rPr.append(ea)
                ea.set("typeface", FONT)
            mirror_rows.append("".join(
                f'<b style="color:#{c}">{esc(x)}</b>' if b else f'<span style="color:#{c}">{esc(x)}</span>'
                for x, b, c in norm))
        self.mirror.append(
            f'<div class="t" style="left:{l*PX:.0f}px;top:{t*PX:.0f}px;width:{w*PX:.0f}px;'
            f'height:{h*PX:.0f}px;font-size:{size}pt;color:#202020;line-height:1.3;">'
            + "".join(f'<div style="margin-bottom:{gap_pt}pt;">{row}</div>' for row in mirror_rows)
            + "</div>")

    def pic(self, l, t, w, path, caption=None):
        from PIL import Image
        with Image.open(path) as im:
            iw, ih = im.size
        h = w * ih / iw
        self.slide.shapes.add_picture(str(path), Inches(l), Inches(t), Inches(w), Inches(h))
        self.mirror.append(
            f'<img src="../gm_assets/{Path(path).name}" style="position:absolute;'
            f'left:{l*PX:.0f}px;top:{t*PX:.0f}px;width:{w*PX:.0f}px;height:{h*PX:.0f}px;">')
        if caption:
            self.text(l, t + h + 0.04, w, 0.25, caption, size=11, align="center", color="595959")
        return h

    def table(self, l, t, w, rows, col_w, row_h=0.30, header=True, size=12):
        n_r, n_c = len(rows), len(rows[0])
        gfx = self.slide.shapes.add_table(n_r, n_c, Inches(l), Inches(t), Inches(w),
                                          Inches(row_h * n_r))
        tbl = gfx.table
        tbl.first_row = False
        tbl.horz_banding = False
        for j, cw in enumerate(col_w):
            tbl.columns[j].width = Inches(cw)
        for i, row in enumerate(rows):
            for j, v in enumerate(row):
                cell = tbl.cell(i, j)
                cell.margin_left = cell.margin_right = Inches(0.06)
                cell.margin_top = cell.margin_bottom = Inches(0.02)
                cell.vertical_anchor = 3  # middle? MSO_ANCHOR.MIDDLE
                p = cell.text_frame.paragraphs[0]
                p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
                r = p.add_run()
                r.text = str(v)
                bold = header and i == 0
                r.font.name = FONT
                r.font.size = Pt(size)
                r.font.bold = bold
                r.font.color.rgb = BLACK
                rPr = r._r.get_or_add_rPr()
                ea = rPr.find(qn("a:ea"))
                if ea is None:
                    ea = rPr.makeelement(qn("a:ea"), {})
                    rPr.append(ea)
                ea.set("typeface", FONT)
                self._set_cell_border(cell)
                if header and i == 0:
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = RGBColor.from_string("EFEFEF")
        html_rows = []
        for i, row in enumerate(rows):
            tag = "th" if (header and i == 0) else "td"
            style_extra = ' style="background:#EFEFEF;font-weight:bold;"' if (header and i == 0) else ""
            tds = "".join(
                f'<{tag} class="{"l" if j==0 else "c"}" style="width:{col_w[j]*PX:.0f}px;{style_extra}">{esc(v)}</{tag}>'
                for j, v in enumerate(row))
            html_rows.append(f"<tr>{tds}</tr>")
        self.mirror.append(
            f'<table class="deck" style="left:{l*PX:.0f}px;top:{t*PX:.0f}px;">{"".join(html_rows)}</table>')
        return t + row_h * n_r

    @staticmethod
    def _set_cell_border(cell):
        tcPr = cell._tc.get_or_add_tcPr()
        for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
            old = tcPr.find(qn(tag))
            if old is not None:
                tcPr.remove(old)
            ln = tcPr.makeelement(qn(tag), {"w": "6350", "cap": "flat"})
            fill = ln.makeelement(qn("a:solidFill"), {})
            clr = fill.makeelement(qn("a:srgbClr"), {"val": "808080"})
            fill.append(clr)
            ln.append(fill)
            # order matters: lnL lnR lnT lnB must precede fill element
            tcPr.append(ln)


def page_number(slide, page):
    box = slide.slide.shapes.add_textbox(Inches(9.21), Inches(6.90), Inches(0.59), Inches(0.47))
    p = box.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    r = p.add_run()
    r.text = str(page)
    r.font.name = FONT
    r.font.size = Pt(12)
    r.font.color.rgb = GRAY
    slide.mirror.append(
        f'<div class="t" style="left:{9.21*PX:.0f}px;top:{6.90*PX:.0f}px;width:{0.59*PX:.0f}px;'
        f'height:{0.47*PX:.0f}px;font-size:12pt;color:#595959;text-align:right;">{page}</div>')


def source_line(slide):
    slide.text(0.55, 0.22, 8.9, 0.28, SRC_LINE, size=11, color="595959")


def heading(slide, text_, size=18):
    slide.text(0.55, 0.52, 8.9, 0.42, [(text_, True)], size=size)


def build(spec_dir: Path, out_pptx: Path, out_dir: Path):
    prs = Presentation()
    prs.slide_width = Inches(W_IN)
    prs.slide_height = Inches(H_IN)
    mirrors = []

    S = Slide(prs, mirrors, 1)  # S1 agenda
    S.para_list(0.55, 1.55, 8.9, 4.0, [
        "近期工作内容：",
        "1. 时序融合模块实验：GRU 基线 → RSSM（v1–v3 → 现在的版本）",
        "2. 训练结果与消融实验（三 seed 复现）",
        "3. 逐类别分析：Truck / Pedestrian / Cyclist",
        "4. 问题与下一步计划",
        ("实验均在 R4Det + TJ4D 上进行，训练与评估记录见 docs/training_runs_full.md", False, "595959"),
    ], size=16, gap_pt=8)

    page_number(S, 1)

    # S2 baseline
    S = Slide(prs, mirrors, 2)
    source_line(S)
    heading(S, "1. 基线：R4Det（radar + camera 双路 BEV 检测）")
    h = S.pic(0.55, 1.05, 5.35, f"{spec_dir}/{A}/fig_arch.png")
    S.para_list(6.15, 1.10, 3.30, 4.6, [
        [("雷达支路", True)],
        "0.16m pillar → Scatter → SECOND+FPN，384ch BEV",
        [("相机支路", True)],
        "ResNet50 + 深度 + LSS，256ch BEV",
        [("融合与检测", True)],
        "拼接卷积融合（640→256）→ 12 anchor × 4 类",
        [("数据集 TJ4D", True)],
        "train 5706 / val 2040 帧；Car 48.4%、Ped 13.5%",
        [("为什么做时序", True)],
        "雷达单帧点云稀疏，BEV 特征帧间不稳 → 把历史帧信息聚合进当前帧",
    ], size=13, gap_pt=4)

    page_number(S, 2)

    # S3 GRU -> RSSM v1-v3
    S = Slide(prs, mirrors, 3)
    source_line(S)
    heading(S, "2. 时序融合：GRU 基线 → RSSM v1–v3")
    y = S.table(0.55, 1.05, 8.9, [
        ["版本", "改动", "BEST (18e)"],
        ["GRU 基线", "上一帧 BEV 特征 + 门控融合（无内部状态）", "34.50"],
        ["RSSM v1", "加 prior/posterior/KL；kl_scale=0.1、free_nats=0", "30.89"],
        ["RSSM v2", "KL 超参放开（1.0 / 1.0），从 v1 续训", "34.01"],
        ["RSSM v3", "logstd 加上界防方差爆炸，从头训", "33.77"],
    ], [1.5, 5.9, 1.5], size=12)
    S.para_list(0.55, y + 0.25, 8.9, 3.4, [
        [("日志诊断出的三个问题：", True)],
        "① KL 项收敛后恒等于 free_nats 常数、梯度为 0（推理不走 prior，对 AP 中性，见第 8 节）",
        "② velocity 分支输入恒为零（TJ4D 没有 ego 位姿标注）→ 直接删掉",
        "③ 输出层零初始化使 z 起步零贡献、起步慢 → 改 Xavier，残差保留",
        [("结论：", True), "v2/v3 与 GRU 同档，差距不在 KL 超参，在模块细节和训练安排。"],
    ], size=13.5, gap_pt=5)

    page_number(S, 3)

    # S4 RSSM module
    S = Slide(prs, mirrors, 4)
    source_line(S)
    heading(S, "3. 现在的 RSSM 模块（MotionAlignedRSSMFusion）")
    S.pic(1.55, 1.02, 6.9, f"{spec_dir}/{A}/gm_rssm.png")
    S.para_list(0.55, 5.62, 8.9, 1.5, [
        "h：确定性状态（ConvGRU + deform 对齐到当前帧）；z：随机状态（prior / posterior）",
        "训练：z 从后验采样，加 KL 与重建损失；推理：z 取后验均值，prior 不参与",
        "输出 = output_proj(z) + feat 残差；截断 BPTT=1（只回传最近一帧历史）",
    ], size=13.5, gap_pt=4)
    page_number(S, 4)

    # S5 training setup + stages
    S = Slide(prs, mirrors, 5)
    source_line(S)
    heading(S, "4. 训练设置与两次关键提升")
    S.para_list(0.55, 1.02, 8.9, 1.15, [
        "AdamW 1.5e-4，Cosine 退火；3 卡 × batch 2 × 累积 2 = 有效 batch 12；24 epoch",
        "KL warmup：ep0–10 从 0 线性升到 1.0；每 epoch 验证一次；三 seed 均为 deterministic 推理",
        "两次提升：① 官方 TJ4D 预训练权重 ② 检测头 v2（Ped 3 组 anchor、关方向分类、IoU 分支）",
    ], size=13.5, gap_pt=4)
    S.pic(1.30, 2.45, 7.4, f"{spec_dir}/{A}/gm_stages.png",
          caption="各阶段 BEST Overall 3D moderate（同一主干，逐级叠加改动）")

    page_number(S, 5)

    # S6 three-seed curves
    S = Slide(prs, mirrors, 6)
    source_line(S)
    heading(S, "5. 训练结果：三 seed 复现")
    S.pic(0.85, 1.02, 8.3, f"{spec_dir}/{A}/gm_val_curves.png",
          caption="验证集 Overall 3D moderate 逐 epoch 曲线（work_dirs 日志）")
    S.para_list(0.55, 5.30, 8.9, 1.6, [
        [("seed0 39.88 / seed1 40.51 / seed2 40.88 → 均值 ", False),
         ("40.42 ± 0.51", True, "C03030"), ("（deterministic，24e）", False)],
        "峰值集中在 ep12–16；取 seed2 ep14 = 40.88 作为 checkpoint",
        [("No-Temporal 35.59", True), ("：同配置去掉时序融合，净收益 ", False), ("+4.83", True)],
    ], size=14, gap_pt=5)

    page_number(S, 6)

    # S7 per-class table
    S = Slide(prs, mirrors, 7)
    source_line(S)
    heading(S, "6. 逐类别结果（3D moderate，各 seed BEST epoch）")
    y = S.table(0.55, 1.02, 8.9, [
        ["类别 · 口径", "seed0 (ep16)", "seed1 (ep15)", "seed2 (ep14)", "均值"],
        ["Pedestrian · strict", "0.10", "0.15", "0.12", "0.12"],
        ["Pedestrian · loose", "28.64", "30.39", "31.32", "30.12"],
        ["Cyclist · strict", "23.17", "26.64", "28.07", "25.96"],
        ["Cyclist · loose", "50.47", "49.36", "50.09", "49.97"],
        ["Car · strict", "49.98", "47.49", "51.35", "49.60"],
        ["Car · loose", "69.20", "72.57", "75.77", "72.52"],
        ["Truck · strict", "30.43", "34.79", "30.76", "31.99"],
        ["Truck · loose", "49.44", "51.58", "52.01", "51.01"],
        ["Overall", "39.88", "40.51", "40.88", "40.42"],
    ], [2.1, 1.7, 1.7, 1.7, 1.7], row_h=0.29, size=12)
    S.para_list(0.55, y + 0.18, 8.9, 1.4, [
        "Pedestrian strict 三个 seed 都是 0.1 上下：瓶颈在 anchor 几何（λ-recall@0.5 只有 4.5%），不是训练没练好",
        "Car loose 波动最大（69.2–75.8）；Overall 口径 = Car/Truck strict + Ped/Cyc loose 取均值",
    ], size=13.5, gap_pt=4)

    page_number(S, 7)

    # S8 ablation table
    S = Slide(prs, mirrors, 8)
    source_line(S)
    heading(S, "7. 消融：RSSM 的组件是不是都有用")
    y = S.table(0.55, 1.02, 8.9, [
        ["配置", "改动", "窗口均值 (ep12–16)"],
        ["No-Temporal", "去掉时序融合，其余不变", "34.65"],
        ["Deterministic", "删随机性，z 取后验均值", "37.08"],
        ["FixedNoise", "z = μ + 0.1·N(0, I)", "36.35"],
        ["Learnable-std", "删 prior，保留可学习 std", "37.04"],
        ["KL = 0（三 seed）", "关掉 KL 梯度", "37.94"],
        ["完整 RSSM（主线）", "prior/posterior + KL + 采样", "39.00"],
    ], [2.2, 4.9, 1.8], size=12)
    S.para_list(0.55, y + 0.25, 8.9, 2.2, [
        "确定性传播（对齐 + GRU + 重建 + 残差）贡献约 +2.4，随机建模再贡献约 +1.9",
        [("KL=0 单 seed 曾到 40.35，但三 seed 只有 37.94（seed2 −4.03）→ KL 不能删", True)],
        "固定噪声、可学习 std 都不如完整采样；随机建模的两部分要一起用",
    ], size=13.5, gap_pt=5)

    page_number(S, 8)

    # S9 KL stats
    S = Slide(prs, mirrors, 9)
    source_line(S)
    heading(S, "8. KL 项的现象：posterior collapse")
    S.pic(0.85, 1.02, 8.3, f"{spec_dir}/{A}/gm_kl_stats.png",
          caption="主线 run 训练日志 stat_* 字段按 epoch 取均值（左：对数轴）")
    S.para_list(0.55, 4.95, 8.9, 1.9, [
        "kl_raw 从 15 收敛到 ≈0.12，clamped_ratio = 100%：KL 全被 free_nats 钳住，梯度为 0",
        "prior / posterior 的 std、μ 差距收敛到 0.01–0.06 量级（不同 run 略有差异）→ prior 退化成后验的复制品",
        "推理不走 prior，所以对 AP 中性；但删掉 KL 多 seed 会掉（上一节）→ 训练期仍然必要",
    ], size=13.5, gap_pt=5)

    page_number(S, 9)

    # S10 truck
    S = Slide(prs, mirrors, 10)
    source_line(S)
    heading(S, "9. 类别专项：Truck 回归三轮都没通过")
    y = S.table(0.55, 1.02, 8.9, [
        ["方案", "Truck Δ", "Car Δ", "Cyclist Δ", "结论"],
        ["残差 refine（dx,dy,dl）", "+2.85", "+1.38", "−4.94", "未过"],
        ["refine + 输入 detach", "−2.79", "−1.38", "−5.11", "未过，否证梯度干扰假设"],
        ["独立回归 tower", "+1.46", "+3.03", "−5.65", "未过"],
    ], [2.6, 1.3, 1.3, 1.3, 2.4], size=12)
    S.para_list(0.55, y + 0.25, 8.9, 2.4, [
        "三轮都撞同一条线：Truck 涨点必伴随 Cyclist 大跌 —— 共享 BEV 特征容量是零和的",
        "前置诊断：Truck 首要问题是定位（中心 1.41m / 长轴 3.02m），不是分类",
        [("处理：Truck 回归路线停止；类别级改动需要从结构上分离（类专属头），后续讨论是否立项", True)],
    ], size=13.5, gap_pt=5)

    page_number(S, 10)

    # S11 cyclist branch
    S = Slide(prs, mirrors, 11)
    source_line(S)
    heading(S, "10. Cyclist 分类残差分支（进行中）")
    S.para_list(0.55, 1.10, 8.9, 2.6, [
        "做法：给 Cyclist 的 12 个 logit 通道加一个 3×3 残差分支（零初始化，等效恒等起步），其余不动",
        "seed0：40.64（+1.18），Cyclist 只 −0.72 —— 相比共享 stem 的 −4.68 修复了大半",
        "seed1：四项全 fail（Ped −3.78 / Truck −3.03），复现失败",
        "seed2：9.19 已启动，等出分",
    ], size=14, gap_pt=6)
    S.para_list(0.55, 4.05, 8.9, 1.6, [
        [("目前的判断：", True)],
        "跨 seed 唯一稳健的信号是 Car strict +2.9；Overall 提升没有被第二个 seed 支持",
        "诊断显示 Cyclist 的瓶颈在分数排序（oracle recall 只差 0.013，AP 却差 5+），不在召回",
    ], size=13.5, gap_pt=4)

    page_number(S, 11)

    # S12 problems & next
    S = Slide(prs, mirrors, 12)
    source_line(S)
    heading(S, "11. 问题与下一步计划")
    S.para_list(0.55, 1.05, 8.9, 2.9, [
        [("还没解决的问题：", True)],
        "① Pedestrian strict ≈ 0：0.32m BEV + anchor 几何的上限，换头 / 加 anchor / 高分辨率支线七轮都没用，需要换思路（point/center 头 + 输入稠密化）",
        "② Car–Truck 混淆与 Truck 定位差（中心 1.41m / 长轴 3.02m）",
        "③ Cyclist 掉点来自分数排序质量，不是召回",
        "④ posterior collapse：随机性实际没有兑现，z 接近确定性",
    ], size=13.5, gap_pt=4)
    S.para_list(0.55, 4.15, 8.9, 2.6, [
        [("下一步：", True)],
        "1. 等 CycCls seed2 出分，三 seed 配对后正式定性该分支",
        "2. 仓库清理：主配置回退 clean（去掉 shared_stem），修复 GRU 基线模块缺失的 return",
        "3. 论文口径定稿：主表 40.42±0.51，消融与失败分析进附录",
        "4. 待讨论：Truck 类分离 tower、Pedestrian 换范式是否立项",
    ], size=13.5, gap_pt=4)

    page_number(S, 12)

    out_pptx.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out_pptx)

    # mirror html
    css = f"""
@page {{ size: 960px 720px; margin: 0; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ width:960px; height:720px; background:#FFFFFF; font-family:'DeckCJK','Microsoft YaHei',sans-serif;
        position:relative; overflow:hidden; }}
.t {{ position:absolute; white-space:pre-wrap; word-break:break-word; line-height:1.3; }}
img {{ position:absolute; }}
table.deck {{ position:absolute; border-collapse:collapse; font-family:inherit; }}
table.deck td, table.deck th {{ border:0.5px solid #808080; padding:2px 6px; font-size:12pt;
        font-weight:normal; color:#202020; }}
table.deck td.l, table.deck th.l {{ text-align:left; }}
table.deck td.c, table.deck th.c {{ text-align:center; }}
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    for page, ops in enumerate(mirrors, start=1):
        body = "".join(ops)
        (out_dir / f"gm_{page:02d}.html").write_text(
            f'<!doctype html><html><head><meta charset="utf-8"><style>{css}'
            f"@font-face {{ font-family: DeckCJK; src: url('file:///home/lurui/.local/share/fonts/NotoSansCJKsc-Regular.otf'); }}"
            f"@font-face {{ font-family: DeckCJK; src: url('file:///home/lurui/.local/share/fonts/NotoSansCJKsc-Bold.otf'); font-weight:bold; }}"
            f"</style></head><body>{body}</body></html>", encoding="utf-8")
    print(f"wrote {len(mirrors)}-slide plain deck to {out_pptx} and mirror to {out_dir}")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    build(base, base / "01-9.19.pptx", base / "gm_preview")
