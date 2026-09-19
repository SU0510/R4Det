#!/usr/bin/env python
"""Generate evidence figures for the R4Det+RSSM progress deck.

All numbers come from docs/R4Det_RSSM_full_report.md (independently verified
against work_dirs logs). Charts use the deck's palette.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

for p in ["/home/lurui/.local/share/fonts/NotoSansCJKsc-Regular.otf",
          "/home/lurui/.local/share/fonts/NotoSansCJKsc-Bold.otf"]:
    fm.fontManager.addfont(p)
plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False

INK = "#17372F"; BODY = "#263732"; MUTED = "#62736C"; GREEN = "#2D6B5B"
LIGHT = "#E5EFE9"; LINE = "#C9D8D0"; AMBER = "#C99A1E"; RED = "#B4453A"
OUT = Path(__file__).resolve().parent

def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=11)
    ax.set_facecolor("white")

def save(fig, name):
    fig.savefig(OUT / name, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("wrote", name)

# ---------- 1. dataset class distribution (half) ----------
fig, ax = plt.subplots(figsize=(7.0, 4.9))
classes = ["Car", "Cyclist", "Truck", "Pedestrian"]
train_n = [11649, 5212, 3963, 3257]
val_n = [4361, 2153, 1404, 999]
x = np.arange(len(classes)); w = 0.38
b1 = ax.bar(x - w/2, train_n, w, color=GREEN, label="train 5706 帧")
b2 = ax.bar(x + w/2, val_n, w, color="#8FB8AC", label="val 2040 帧")
for b in list(b1) + list(b2):
    ax.annotate(f"{int(b.get_height())}", (b.get_x() + b.get_width()/2, b.get_height()),
                ha="center", va="bottom", fontsize=10, color=BODY)
ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=12, color=INK)
ax.set_ylabel("3D 框数量", fontsize=11, color=BODY)
ax.legend(frameon=False, fontsize=11, loc="upper right")
ax.set_ylim(0, 13500)
style_ax(ax)
ax.set_title("TJ4D 类别分布：Car 占近半，Pedestrian 最稀少", fontsize=14, color=INK, fontweight="bold", pad=12)
save(fig, "fig_dataset.png")

# ---------- 2. evolution chain (wide) ----------
fig, ax = plt.subplots(figsize=(9.6, 3.2))
runs = ["GRU 基线\n(Run1 18e)", "RSSM v3\n(18e)", "N4/hdim128\n(Run7)", "30e 对照",
        "+预训练\n(Run9 30e)", "+head-v2\n(Run10 24e)", "三 seed 均值\n(最终主线)"]
vals = [34.50, 33.77, 34.05, 34.71, 37.94, 39.65, 40.42]
colors = [MUTED, MUTED, MUTED, MUTED, GREEN, GREEN, INK]
bars = ax.bar(range(len(runs)), vals, color=colors, width=0.62)
for i, b in enumerate(bars):
    label = f"{vals[i]:.2f}" if i < 6 else "40.42±0.51"
    ax.annotate(label, (b.get_x() + b.get_width()/2, b.get_height()),
                ha="center", va="bottom", fontsize=10.5, color=INK, fontweight="bold")
ax.axhline(35.59, color=AMBER, lw=1.4, ls="--")
ax.annotate("No-Temporal 35.59（同主线配置）", (0.02, 35.59 + 0.25), fontsize=10, color=AMBER)
ax.axvline(3.5, color=LINE, lw=1.0)
ax.annotate("预训练 + head-v2 前", (0.9, 42.3), fontsize=10, color=MUTED)
ax.annotate("预训练 + head-v2 时代", (4.6, 42.3), fontsize=10, color=MUTED)
ax.set_xticks(range(len(runs))); ax.set_xticklabels(runs, fontsize=9.5, color=BODY)
ax.set_ylim(30, 43.5); ax.set_ylabel("BEST Overall 3D moderate", fontsize=10.5, color=BODY)
style_ax(ax)
save(fig, "fig_evolution.png")

# ---------- 3. mechanism ablation (half) ----------
fig, ax = plt.subplots(figsize=(7.0, 4.9))
items = [("完整 RSSM（主模型）", 39.00, GREEN), ("KL=0（三 seed 均值）", 37.94, "#8FB8AC"),
         ("Deterministic（删随机性）", 37.08, "#8FB8AC"), ("Learnable-std", 37.04, "#8FB8AC"),
         ("FixedNoise", 36.35, "#8FB8AC"), ("No-Temporal", 34.65, MUTED)]
names = [i[0] for i in items]; v = [i[1] for i in items]; c = [i[2] for i in items]
y = np.arange(len(items))
bars = ax.barh(y, v, color=c, height=0.58)
for b, val in zip(bars, v):
    ax.annotate(f"{val:.2f}", (b.get_width() + 0.06, b.get_y() + b.get_height()/2),
                va="center", fontsize=11, color=INK, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=11.5, color=BODY)
ax.invert_yaxis()
ax.set_xlim(33, 40.2)
ax.set_xlabel("窗口均值 Overall 3D moderate（ep12–16, seed0）", fontsize=10.5, color=BODY)
style_ax(ax)
ax.set_title("RSSM 机制拆解：确定性传播与随机建模共同必需", fontsize=14, color=INK, fontweight="bold", pad=12)
save(fig, "fig_ablation.png")

# ---------- 4. three-seed result (half) ----------
fig, ax = plt.subplots(figsize=(7.0, 4.9))
seeds = ["seed 0", "seed 1", "seed 2"]
best = [39.88, 40.51, 40.88]
bars = ax.bar(seeds, best, color=[GREEN, GREEN, GREEN], width=0.5)
for b, val in zip(bars, best):
    ax.annotate(f"{val:.2f}", (b.get_x() + b.get_width()/2, b.get_height()),
                ha="center", va="bottom", fontsize=12, color=INK, fontweight="bold")
ax.axhline(40.42, color=INK, lw=1.6, ls="-", alpha=0.75)
ax.text(-0.42, 40.58, "均值 40.42", fontsize=11, color=INK, fontweight="bold", ha="left")
ax.axhline(35.59, color=AMBER, lw=1.4, ls="--")
ax.annotate("No-Temporal 35.59", (2.28, 35.72), fontsize=10.5, color=AMBER, ha="right")
ax.set_ylim(33, 42.2); ax.set_ylabel("BEST Overall 3D moderate", fontsize=11, color=BODY)
ax.tick_params(axis="x", labelsize=12)
style_ax(ax)
ax.set_title("主线三 seed 复现：40.42 ± 0.51（24e, deterministic）", fontsize=14, color=INK, fontweight="bold", pad=12)
save(fig, "fig_seed.png")

# ---------- 5. truck zero-sum (half) ----------
fig, ax = plt.subplots(figsize=(7.0, 4.9))
exps = ["残差 refine\n(§26)", "refine+detach\n(§27)", "独立 tower\n(§28)"]
delta = {"Truck": [2.85, -2.79, 1.46], "Car": [1.38, -1.38, 3.03], "Cyclist": [-4.94, -5.11, -5.65]}
x = np.arange(len(exps)); w = 0.24
cols = {"Truck": GREEN, "Car": "#8FB8AC", "Cyclist": RED}
for i, (cls, vals) in enumerate(delta.items()):
    bars = ax.bar(x + (i - 1) * w, vals, w, color=cols[cls], label=cls)
    for b, val in zip(bars, vals):
        ax.annotate(f"{val:+.2f}", (b.get_x() + b.get_width()/2, b.get_height() + (0.18 if val >= 0 else -0.55)),
                    ha="center", fontsize=9.5, color=BODY)
ax.axhline(0, color=LINE, lw=1.0)
ax.axhline(-4.0, color=RED, lw=1.2, ls="--", alpha=0.7)
ax.set_xlim(-0.55, 3.35)
ax.annotate("Cyclist 门控红线 −4.0", (2.52, -3.72), fontsize=10, color=RED, ha="left")
ax.set_xticks(x); ax.set_xticklabels(exps, fontsize=11, color=BODY)
ax.set_ylabel("相对 clean 基线 Δ (AP)", fontsize=11, color=BODY)
ax.set_ylim(-6.6, 4.2)
ax.legend(frameon=False, fontsize=11, ncol=3, loc="upper left")
style_ax(ax)
ax.set_title("Truck 专项三轮：Truck 涨点必伴随 Cyclist 大跌（零和）", fontsize=14, color=INK, fontweight="bold", pad=12)
save(fig, "fig_truck.png")

# ---------- 6. architecture block diagram (half) ----------
fig, ax = plt.subplots(figsize=(7.0, 4.9))
ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")

def box(x, y, w, h, text, fc=LIGHT, ec=GREEN, fs=10.5, tc=INK, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal")

def arrow(x1, y1, x2, y2, color=GREEN):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, color=color, lw=1.4))

# radar branch (top)
box(0.2, 5.6, 2.0, 0.85, "雷达点云 (N,5)\nv / SNR", fc="white")
box(2.55, 5.6, 2.05, 0.85, "Pillar 0.16m\nPFN 64ch")
box(4.95, 5.6, 2.05, 0.85, "Scatter 496×432\nSECOND+FPN")
box(7.35, 5.6, 2.45, 0.85, "雷达 BEV\n(B,384,248,216)")
arrow(2.2, 6.02, 2.55, 6.02); arrow(4.6, 6.02, 4.95, 6.02); arrow(7.0, 6.02, 7.35, 6.02)
# camera branch (bottom)
box(0.2, 0.45, 2.0, 0.85, "相机 480×640", fc="white")
box(2.55, 0.45, 2.05, 0.85, "ResNet50\n+FPN 256ch")
box(4.95, 0.45, 2.05, 0.85, "深度 72 bins\nLSS splat")
box(7.35, 0.45, 2.45, 0.85, "相机 BEV\n(B,256,248,216)")
arrow(2.2, 0.88, 2.55, 0.88); arrow(4.6, 0.88, 4.95, 0.88); arrow(7.0, 0.88, 7.35, 0.88)
# fusion + rssm + head (middle)
box(2.6, 3.0, 2.2, 0.9, "ConcatConvFusion\n640→256")
box(5.35, 3.0, 2.9, 0.9, "RSSM 时序融合 (h, z)\n输出 = proj(z) + feat 残差", fc="#DCEAE2", ec=INK, bold=True, fs=9.5)
box(8.8, 3.0, 1.0, 0.9, "检测头\nv2", fc="#DCEAE2", ec=INK, bold=True)
arrow(8.35, 5.6, 3.7, 3.9, color=GREEN)
arrow(8.35, 1.3, 3.7, 3.0, color=GREEN)
arrow(4.8, 3.45, 5.35, 3.45); arrow(8.25, 3.45, 8.8, 3.45)
ax.text(5.0, 6.9, "N=4 帧序列（old→new），每帧独立编码；仅当前帧出检测",
        fontsize=10.5, color=MUTED, ha="center")
save(fig, "fig_arch.png")

# ---------- 7. RSSM mechanism diagram (half) ----------
fig, ax = plt.subplots(figsize=(7.0, 4.9))
ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")

box(0.2, 4.9, 2.3, 0.95, "上一帧状态\nh$_{t-1}$ / z$_{t-1}$", fc="white")
box(3.0, 4.9, 2.6, 0.95, "Deform 对齐\n(零初始化=恒等起步)")
box(6.2, 4.9, 2.2, 0.95, "ConvGRU\n→ h$_t$ (确定性)", fc="#DCEAE2", ec=INK, bold=True)
box(8.8, 5.6, 1.0, 0.8, "当前帧\nBEV 特征", fc="white", fs=9.5)
arrow(2.5, 5.37, 3.0, 5.37); arrow(5.6, 5.37, 6.2, 5.37)
arrow(8.8, 5.95, 7.3, 5.6)
# prior / posterior
box(5.5, 3.1, 2.0, 0.85, "先验 p(z$_t$|h$_t$)\nμ$_p$, σ$_p$", fc="white", fs=10)
box(8.0, 3.1, 1.8, 0.85, "后验 q(z$_t$|h$_t$,e$_t$)\nμ$_q$, σ$_q$", fc="#DCEAE2", ec=INK, bold=True, fs=10)
arrow(7.3, 4.9, 6.5, 3.95)
arrow(8.9, 5.6, 8.9, 3.95)
ax.text(7.05, 4.15, "KL\n(free-bits\n+warmup)", fontsize=9, color=AMBER, ha="center", fontweight="bold")
# z sample -> decoder -> residual
box(4.6, 1.6, 2.4, 0.85, "z$_t$：训练采样\n推理取 μ$_q$", fc="#DCEAE2", ec=INK, bold=True, fs=10)
box(7.8, 1.6, 2.0, 0.85, "decoder 重建\n(MSE loss)", fc="white", fs=10)
box(1.6, 1.6, 2.2, 0.85, "output_proj(z$_t$)\n+ feat 残差输出", fc="white", fs=10)
arrow(8.9, 3.1, 5.8, 2.45); arrow(7.0, 2.02, 7.8, 2.02); arrow(4.6, 2.02, 3.8, 2.02)
# state store
box(0.2, 3.1, 2.4, 0.85, "状态存储\nh$_t$, z$_t$ → 下一帧", fc="white", fs=10)
arrow(4.55, 2.3, 2.62, 3.25)
ax.text(5.0, 6.55, "训练：KL + 重建损失；推理：prior 不参与，历史帧 no_grad",
        fontsize=10.5, color=MUTED, ha="center")
ax.text(5.0, 0.5, "posterior collapse 实测稳态：clamped_ratio→100%，μ_diff²→0.04（KL 梯度归零但推理不受影响）",
        fontsize=9.5, color=AMBER, ha="center")
save(fig, "fig_rssm.png")

print("all charts done")
