#!/usr/bin/env python
"""Plain group-meeting style charts for the 01-9.19 deck (all data from logs)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import json
from pathlib import Path

for p in ["/home/lurui/.local/share/fonts/NotoSansCJKsc-Regular.otf",
          "/home/lurui/.local/share/fonts/NotoSansCJKsc-Bold.otf"]:
    fm.fontManager.addfont(p)
plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent
CURVES = json.load(open(OUT / "val_curves.json"))
STATS = json.load(open(OUT / "kl_stats.json"))

INK = "#222222"; GRAY = "#888888"
C1 = "#2D6B5B"; C2 = "#4C78A8"; C3 = "#C9793A"; C4 = "#999999"

def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10, colors=INK)
    ax.grid(True, axis="y", lw=0.5, color="#DDDDDD")
    ax.set_axisbelow(True)

# ---- 1. three-seed val curves ----
fig, ax = plt.subplots(figsize=(8.6, 3.9))
for name, label, color in [("seed0", "seed0", C1), ("seed1", "seed1", C2), ("seed2", "seed2", C3)]:
    xs = sorted(int(k) for k in CURVES[name])
    ax.plot(xs, [CURVES[name][str(x)] for x in xs], lw=1.6, color=color, label=label)
xs = sorted(int(k) for k in CURVES["no_temporal"])
ax.plot(xs, [CURVES["no_temporal"][str(x)] for x in xs], lw=1.6, ls="--", color=C4,
        label="No-Temporal")
ax.set_xlabel("epoch", fontsize=11)
ax.set_ylabel("Overall 3D moderate (AP40)", fontsize=11)
ax.set_xlim(1, 24)
ax.legend(fontsize=10, frameon=False, loc="lower right", ncol=4)
style(ax)
fig.tight_layout()
fig.savefig(OUT / "gm_val_curves.png", dpi=200, facecolor="white")
plt.close(fig)

# ---- 2. KL stats (two panels) ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
xs = sorted(int(k) for k in STATS["stat_kl_raw_mean"])
a1.plot(xs, [STATS["stat_kl_raw_mean"][str(x)] for x in xs], lw=1.6, color=C1)
a1.set_yscale("log")
a1.set_xlabel("epoch", fontsize=11)
a1.set_ylabel("stat_kl_raw_mean（对数轴）", fontsize=11)
a1.annotate("收敛后 ≈ 0.12", (14, 0.16), fontsize=10, color=GRAY)
style(a1)
xs = sorted(int(k) for k in STATS["stat_clamped_ratio"])
a2.plot(xs, [STATS["stat_clamped_ratio"][str(x)] for x in xs], lw=1.6, color=C3,
        label="clamped_ratio")
a2.set_ylim(0, 1.05)
a2.set_xlabel("epoch", fontsize=11)
a2.set_ylabel("clamped_ratio", fontsize=11)
ax2b = a2.twinx()
xs = sorted(int(k) for k in STATS["stat_prior_std"])
ax2b.plot(xs, [STATS["stat_prior_std"][str(x)] for x in xs], lw=1.4, ls="--", color=C2,
          label="prior_std")
ax2b.plot(xs, [STATS["stat_posterior_std"][str(x)] for x in xs], lw=1.4, ls=":", color=INK,
          label="posterior_std")
ax2b.set_ylabel("prior/posterior std", fontsize=10, color=C2)
ax2b.tick_params(labelsize=10, colors=C2)
ax2b.spines["top"].set_visible(False)
h1, l1 = a2.get_legend_handles_labels()
h2, l2 = ax2b.get_legend_handles_labels()
a2.legend(h1 + h2, l1 + l2, fontsize=9, frameon=False, loc="center right")
style(a2)
fig.tight_layout()
fig.savefig(OUT / "gm_kl_stats.png", dpi=200, facecolor="white")
plt.close(fig)

# ---- 3. stage gains bar ----
fig, ax = plt.subplots(figsize=(8.6, 3.2))
labels = ["N4 30e\n无预训练", "+预训练\n(Run9)", "+检测头v2\n(Run10)", "三 seed 均值\n(最终)"]
vals = [34.71, 37.94, 39.65, 40.42]
bars = ax.bar(range(4), vals, width=0.55, color=[GRAY, C1, C1, "#1F4A3E"])
for i, b in enumerate(bars):
    ax.annotate(f"{vals[i]:.2f}" if i < 3 else "40.42±0.51",
                (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center", va="bottom", fontsize=11)
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=10.5)
ax.set_ylim(30, 43)
ax.set_ylabel("BEST Overall 3D moderate", fontsize=11)
style(ax)
fig.tight_layout()
fig.savefig(OUT / "gm_stages.png", dpi=200, facecolor="white")
plt.close(fig)

print("gm charts done")

# ---- 4. plain RSSM diagram (no amber, no extra notes) ----
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig, ax = plt.subplots(figsize=(8.6, 4.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 6.4); ax.axis("off")

def gbox(x, y, w, h, text_, fc="#FFFFFF", ec="#333333", fs=10.5, bold=False, tc="#222222"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text_, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal")

def garrow(x1, y1, x2, y2, color="#333333"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, color=color, lw=1.3))

gbox(0.2, 4.6, 2.3, 0.9, "上一帧状态\nh$_{t-1}$ / z$_{t-1}$")
gbox(3.0, 4.6, 2.6, 0.9, "Deform 对齐\n（零初始化 = 恒等起步）")
gbox(6.2, 4.6, 2.2, 0.9, "ConvGRU\n→ h$_t$（确定性）", fc="#E3EEE8", ec="#2D6B5B", bold=True)
gbox(8.8, 5.3, 1.0, 0.75, "当前帧\nBEV 特征", fs=9.5)
garrow(2.5, 5.05, 3.0, 5.05); garrow(5.6, 5.05, 6.2, 5.05); garrow(8.8, 5.6, 7.3, 5.3)
gbox(5.5, 2.9, 2.0, 0.8, "先验 p(z$_t$|h$_t$)\nμ$_p$, σ$_p$", fs=10)
gbox(8.0, 2.9, 1.8, 0.8, "后验 q(z$_t$|h$_t$,e$_t$)\nμ$_q$, σ$_q$", fc="#E3EEE8", ec="#2D6B5B", bold=True, fs=10)
garrow(7.3, 4.6, 6.5, 3.7); garrow(8.9, 5.3, 8.9, 3.7)
ax.text(7.15, 3.95, "KL\nfree-bits\n+warmup", fontsize=9, color="#222222", ha="center")
gbox(4.6, 1.4, 2.4, 0.8, "z$_t$：训练采样\n推理取 μ$_q$", fc="#E3EEE8", ec="#2D6B5B", bold=True, fs=10)
gbox(7.8, 1.4, 2.0, 0.8, "decoder 重建\n（MSE loss）", fs=10)
gbox(1.6, 1.4, 2.2, 0.8, "output_proj(z$_t$)\n+ feat 残差输出", fs=10)
garrow(8.9, 2.9, 5.8, 2.2); garrow(7.0, 1.8, 7.8, 1.8); garrow(4.6, 1.8, 3.8, 1.8)
gbox(0.2, 2.9, 2.4, 0.8, "状态存储\nh$_t$, z$_t$ → 下一帧", fs=10)
garrow(4.55, 2.1, 2.62, 3.05)
fig.tight_layout()
fig.savefig(OUT / "gm_rssm.png", dpi=200, facecolor="white")
plt.close(fig)
print("gm_rssm done")
