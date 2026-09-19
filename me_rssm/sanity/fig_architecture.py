"""Draft architecture figure for the ME-RSSM paper (fig2 candidate).

Zero-training; pure matplotlib. Two-band layout:
  top band    = PREDICT:  (h,z) -> motion-conditioned align -> dyn-gated
                GRU -> h_t -> prior p(z_t|h_t)
  bottom band = CORRECT:  appearance/Doppler/camera observations ->
                confidence-weighted obs fusion -> posterior q(z_t|h_t,e)
                -> learned Kalman gain -> z_t = mu_p + g (z_sel - mu_p)
Grey = inherited baseline components; green = Doppler evidence pathway;
blue = camera observation pathway; orange = gating/gain decisions;
pink = zero-init mixing points (identity start).

Output: me_rssm/figures/me_rssm_architecture_draft.png
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

_OUT = os.path.join(os.path.dirname(__file__), '..', 'figures',
                    'me_rssm_architecture_draft.png')

C_BASE, C_DOP, C_CAM, C_GATE, C_MIX = ('#ececec', '#c8e6c9', '#bbdefb',
                                       '#ffe0b2', '#ffcdd2')

fig, ax = plt.subplots(figsize=(13.6, 7.8))
ax.set_xlim(0, 100)
ax.set_ylim(0, 66)
ax.axis('off')


def box(x, y, w, h, text, color, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4',
                                fc=color, ec='#555555', lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, linespacing=1.25)


def arrow(x1, y1, x2, y2, color='#444444', lw=1.4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                 mutation_scale=13, color=color, lw=lw))


ax.text(50, 64.2, 'ME-RSSM: Doppler-informed predict-correct filtering '
        '(fig2 draft)', ha='center', fontsize=12)

# ============================ PREDICT band =================================
ax.text(1, 58.2, 'PREDICT', fontsize=10, color='#2ca02c', weight='bold')

box(2, 47, 13, 8, 'state\n$(h_{t-1}, z_{t-1})$', C_BASE, fs=8.5)
box(22, 47, 17, 8, 'align:\n$conv(o_t, h) + conv(e^{mot})$\n(zero-init term)',
    C_DOP, fs=8)
box(46, 47, 18, 8, 'GRU:\n$u\' = u + \\beta\\, d\\, u(1-u)$\n'
    '$h_t = (1-u\')h + u\'\\tilde h$', C_GATE, fs=8)
box(71, 47, 13, 8, 'prior\n$p(z_t|h_t)$', C_BASE, fs=8.5)
box(88, 47, 10, 8, '$h_t$', C_BASE, fs=9)

arrow(15, 51, 22, 51)
arrow(39, 51, 46, 51)
arrow(64, 51, 71, 51)
arrow(84, 51, 88, 51)

# state loop: h_t back to state box (outer right loop)
arrow(98, 55, 99, 60.5, color='#888888', lw=1.1)
arrow(99, 60.5, 8.5, 60.5, color='#888888', lw=1.1)
arrow(8.5, 60.5, 8.5, 55.4, color='#888888', lw=1.1)

# Doppler canvas into align + GRU gate
box(22, 34, 17, 8, 'Doppler canvas $C$ (7ch)\n-> motion encoder $e^{mot}$,'
    ' $c^{mot}$', C_DOP, fs=7.8)
arrow(39, 53.5, 46, 53.5, color='#2ca02c')          # e_mot -> GRU gate? (d)
ax.text(41.5, 54.6, '$d$', fontsize=8, color='#2ca02c')
arrow(39, 40, 24, 47, color='#2ca02c')              # e_mot -> align offsets

# ============================ CORRECT band =================================
ax.text(1, 28.6, 'CORRECT', fontsize=10, color='#1565c0', weight='bold')

box(2, 17, 13, 8, 'BEV feat $o_t$\n(fused cam+radar)', C_BASE, fs=8.5)
box(22, 17, 17, 8, 'appearance encoder\n$e^{app}$', C_BASE, fs=8.5)
box(2, 2, 13, 8, 'camera BEV $I_t$', C_CAM, fs=8.5)
box(22, 2, 17, 8, 'camera encoder\n$e^{cam}$, $c^{cam}$', C_CAM, fs=8)

arrow(15, 21, 22, 21)
arrow(15, 6, 22, 6)
arrow(39, 6, 44, 12, color='#1565c0')
arrow(39, 38, 44, 22.5, color='#2ca02c')

box(44, 11, 20, 12, 'obs fusion (zero-init proj):\n$e^{fused} = e^{app} + '
    'proj([c^{cam} e^{cam}, c^{mot} e^{mot}])$', C_MIX, fs=7.8)

box(71, 11, 12, 12, 'posterior\n$q(z_t|h_t, e^{fused})$', C_BASE, fs=8)
arrow(64, 17, 71, 17)

box(88, 2, 10, 12, 'Kalman gain\n$z_t = \\mu_p + g(z_{sel}-\\mu_p)$\n'
    '$g = \\sigma(conv([c^{cam}, c^{mot}]))$', C_GATE, fs=7)
arrow(83, 17, 88, 12)
arrow(90, 47, 90, 14, color='#c62828', lw=1.3)      # prior mean into gain
arrow(44, 40, 92, 8.5, color='#2ca02c', lw=1.0)     # c_mot into gain

# outputs
box(44, 0.5, 20, 7, 'output: $proj(z_t) + o_t$ · recon: $dec(h_t, z_t)$',
    C_BASE, fs=8)
arrow(71, 13, 64, 6)
ax.text(93, 1, '$z_t$', fontsize=8)
arrow(88, 6, 64, 4.5, color='#c62828', lw=0.9)

# legend
legend = [('inherited baseline component', C_BASE),
          ('Doppler evidence pathway (new)', C_DOP),
          ('camera observation pathway (new)', C_CAM),
          ('gating / gain decision (new)', C_GATE),
          ('zero-init mixing (identity start)', C_MIX)]
for i, (txt, c) in enumerate(legend):
    x = 2 + i * 19.6
    ax.add_patch(FancyBboxPatch((x, -5.4), 1.8, 2.6,
                                boxstyle='round,pad=0.2', fc=c,
                                ec='#555555', lw=0.8))
    ax.text(x + 2.4, -4.1, txt, fontsize=7.2, va='center')

fig.subplots_adjust(bottom=0.1)
fig.savefig(_OUT, dpi=170, bbox_inches='tight')
print(f'saved -> {os.path.abspath(_OUT)}')
