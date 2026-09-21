"""Flowchart depictions for the friend to redraw in PowerPoint."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

RED, BLU, GRY, GRN, ORG = "#c0143c", "#2166ac", "#6b7280", "#2ca25f", "#e08214"

def box(ax, x, y, w, h, text, fc="white", ec=GRY, tc="#111", fs=10.2, bold=False, lw=1.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=tc, weight="bold" if bold else "normal", linespacing=1.4, zorder=3)

def arrow(ax, p0, p1, col=GRY, lw=2.0, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=17,
                                 lw=lw, color=col, zorder=1, linestyle=ls,
                                 shrinkA=3, shrinkB=3))

# ============================ FLOW A — the loop ==============================
fig, ax = plt.subplots(figsize=(9.6, 3.5)); ax.set_xlim(0, 10); ax.set_ylim(0, 3.5); ax.axis("off")
ax.text(0.05, 3.30, "TODAY  —  open loop", fontsize=11.5, weight="bold", color=GRY)
box(ax, 0.1, 2.35, 2.3, 0.72, "Pre-mission\nschedule", fc="#f3f4f6")
box(ax, 3.0, 2.35, 2.3, 0.72, "Sweep the band\nlist in order", fc="#f3f4f6")
box(ax, 5.9, 2.35, 2.3, 0.72, "Same 50 ms on a\nweather radar as\non a fire-control", fc="#f3f4f6", fs=9.2)
arrow(ax, (2.45, 2.71), (2.95, 2.71)); arrow(ax, (5.35, 2.71), (5.85, 2.71))
ax.text(8.4, 2.71, "✕", fontsize=22, color=RED, ha="center", va="center", weight="bold")

ax.text(0.05, 1.72, "NARADA  —  closed loop", fontsize=11.5, weight="bold", color=RED)
box(ax, 0.1, 0.55, 2.0, 0.80, "Tune one band\nfor one dwell", fc="white", ec=RED, bold=True)
box(ax, 2.7, 0.55, 2.0, 0.80, "Receiver declares\nhit / no hit", fc="white", ec=RED, bold=True)
box(ax, 5.3, 0.55, 2.0, 0.80, "Update belief over\nall 36 bands", fc="white", ec=RED, bold=True)
box(ax, 7.9, 0.55, 2.0, 0.80, "Policy picks the\nnext band", fc="white", ec=RED, bold=True)
for x0, x1 in ((2.15, 2.65), (4.75, 5.25), (7.35, 7.85)):
    arrow(ax, (x0, 0.95), (x1, 0.95), col=RED)
ax.add_patch(FancyArrowPatch((8.9, 0.53), (1.1, 0.53), arrowstyle="-|>", mutation_scale=17,
        lw=2.0, color=RED, connectionstyle="arc3,rad=0.30", zorder=1))
ax.text(5.0, -0.50, "repeat every dwell  —  600 decisions in 30 s", fontsize=9.6,
        color=RED, ha="center", va="bottom", style="italic")
fig.tight_layout(); fig.savefig("/private/tmp/claude-501/-Users-aamirhashmi-Code-projects-SIH26055/90581c6c-2a45-4658-9929-b806de01bddc/scratchpad/fig/flowA_loop.png", dpi=200, facecolor="white")

# ====================== FLOW B — system architecture =========================
fig, ax = plt.subplots(figsize=(9.6, 5.4)); ax.set_xlim(0, 10); ax.set_ylim(0, 6.0); ax.axis("off")
ax.text(5.0, 5.80, "What we built  —  four layers, each validated before the next",
        fontsize=12, weight="bold", ha="center")
rows = [
    ("L0", "SCENARIO", "2.2 GB of real Turing radar recordings\n→ 3,443 reusable emitter contributions", "#eef2f7", BLU),
    ("L1", "TRUTH",    "36 bands × 600 slots of 50 ms\nwho is transmitting, whether or not we look", "#eef2f7", BLU),
    ("L2", "RECEIVER", "declares a hit iff  signal + noise ≥ threshold\nPd 0.84 · Pfa 0.0013 — frozen, not trained", "#eef2f7", BLU),
    ("L3", "SCHEDULER","Recurrent PPO picks the next band\n183-value observation · its own history only", "#fdeaee", RED),
]
y = 4.55
for tag, name, body, fc, ec in rows:
    box(ax, 1.55, y, 6.9, 0.95, "", fc=fc, ec=ec, lw=2.0)
    ax.text(1.85, y + 0.475, tag, fontsize=13, weight="bold", color=ec, va="center")
    ax.text(2.75, y + 0.65, name, fontsize=11.2, weight="bold", color="#111", va="center")
    ax.text(2.75, y + 0.29, body, fontsize=9.3, color="#333", va="center", linespacing=1.35)
    if y > 1.0: arrow(ax, (5.0, y - 0.02), (5.0, y - 0.32), col=ec)
    y -= 1.27
box(ax, 0.05, 1.05, 1.35, 0.95, "Operator\nor threat\nlibrary", fc="white", ec=GRN, tc=GRN, fs=9.4, bold=True)
arrow(ax, (1.42, 1.52), (1.50, 1.52), col=GRN)
ax.text(0.72, 0.62, "priority p\n(default: uniform)", fontsize=8.6, color=GRN, ha="center", va="top", linespacing=1.3)
box(ax, 8.6, 1.05, 1.35, 0.95, "Chosen\nband", fc="white", ec=RED, tc=RED, fs=9.6, bold=True)
arrow(ax, (8.48, 1.52), (8.56, 1.52), col=RED)
ax.add_patch(FancyArrowPatch((9.25, 2.05), (9.25, 4.40), arrowstyle="-|>", mutation_scale=15,
        lw=1.7, color=GRY, linestyle=(0,(4,3)), connectionstyle="arc3,rad=-0.3", zorder=1))
ax.text(9.85, 3.2, "acts on\nthe world", fontsize=8.6, color=GRY, rotation=90,
        ha="center", va="center", linespacing=1.3)
ax.text(5.0, 0.28, "Frozen before any result was quoted  ·  3 validation gates PASS, 1 measured  ·  433 tests",
        fontsize=9.4, color=GRY, ha="center", style="italic")
fig.tight_layout(); fig.savefig("/private/tmp/claude-501/-Users-aamirhashmi-Code-projects-SIH26055/90581c6c-2a45-4658-9929-b806de01bddc/scratchpad/fig/flowB_arch.png", dpi=200, facecolor="white")
print("ok")
