"""Flow B and Flow C as decision flows, in the style of the reference decks.

Plain boxes, short action text, diamonds where something is actually decided, and
Pass/Fail labels on the branches. No coloured section headers, no pictures inside
the flow — the pictures live in docs/ppt/assets/ and get placed on the slide.

Flow C deliberately does not repeat Flow A. Flow A is the operational loop the
receiver runs. Flow C is the machine-learning story: how the reward was screened
before anything trained on it, how training actually runs, and the one decision
taken at inference time.

    python docs/ppt/figures/mk_flows_v3.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

HERE = Path(__file__).resolve().parent
INK, LINE, RED, GRN, BLU = "#15181d", "#3f4550", "#c0143c", "#1f8a4c", "#2166ac"
plt.rcParams.update({"font.family": "DejaVu Sans"})


def fig_ax(w, h):
    fig = plt.figure(figsize=(w, h), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def box(ax, cx, cy, w, h, text, fs=7.6, ec=LINE, fc="white", lw=1.15, bold=False, tc=INK):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.004,rounding_size=0.045",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(cx, cy, text, fontsize=fs, color=tc, ha="center", va="center",
            linespacing=1.28, zorder=4, weight="bold" if bold else "normal")


def diamond(ax, cx, cy, w, h, text, fs=7.4, ec=LINE, fc="#f7f9fc"):
    ax.add_patch(Polygon([[cx, cy + h / 2], [cx + w / 2, cy], [cx, cy - h / 2], [cx - w / 2, cy]],
                         closed=True, fc=fc, ec=ec, lw=1.3, zorder=3))
    ax.text(cx, cy, text, fontsize=fs, color=INK, ha="center", va="center",
            linespacing=1.25, zorder=4)


def arr(ax, p0, p1, col=LINE, lw=1.15, ls="-", rad=0.0, ms=11):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw,
                                 color=col, linestyle=ls, zorder=2, shrinkA=1.5, shrinkB=1.5,
                                 connectionstyle=f"arc3,rad={rad}"))


def lab(ax, x, y, s, fs=6.9, col=INK, ha="center", va="center", bold=False, st="normal"):
    ax.text(x, y, s, fontsize=fs, color=col, ha=ha, va=va, zorder=6,
            weight="bold" if bold else "normal", style=st, linespacing=1.25)


def elbow(ax, x0, y0, x1, y1, col=LINE, lw=1.15):
    """Right-angle connector: out horizontally, then vertically, then in."""
    ax.plot([x0, x1], [y0, y0], lw=lw, color=col, zorder=2, solid_capstyle="round")
    arr(ax, (x1, y0), (x1, y1), col=col, lw=lw)


# ============================== FLOW B ======================================
def flow_b():
    W, H = 7.30, 3.05
    fig, ax = fig_ax(W, H)

    # the build chain
    bh, bw, cy = 0.52, 1.26, 2.60
    xs = [0.72, 2.10, 3.48]
    texts = ["Turing radar\nrecordings", "Extract every\npulse", "Group pulses\nby emitter"]
    for x, t in zip(xs, texts):
        box(ax, x, cy, bw, bh, t)
    gx, gw = 5.45, 1.44
    box(ax, gx, cy, gw, bh, "Fill the 36 × 600\nband–slot grid")
    for i in range(2):
        arr(ax, (xs[i] + bw / 2, cy), (xs[i + 1] - bw / 2, cy))
    arr(ax, (xs[2] + bw / 2, cy), (gx - gw / 2, cy))

    # the environment boundary
    ax.add_patch(FancyBboxPatch((4.58, 1.02), 2.66, 1.92,
                                boxstyle="round,pad=0.01,rounding_size=0.05",
                                fc="none", ec=BLU, lw=1.1, ls=(0, (5, 3)), zorder=1))
    lab(ax, 4.66, 1.12, "simulated RF environment", 6.5, BLU, ha="left", st="italic")

    # the receiver decision
    dcx, dcy, dw, dh = 5.45, 1.72, 1.40, 0.58
    ax.plot([dcx, dcx], [cy - bh / 2, 2.16], lw=1.15, color=LINE, zorder=2)
    arr(ax, (dcx, 2.16), (dcx, dcy + dh / 2))
    diamond(ax, dcx, dcy, dw, dh, "signal + noise\nover threshold ?", 7.0)
    box(ax, 6.64, 2.06, 0.88, 0.34, "Hit", 7.4, GRN, "#f3fbf6", bold=True, tc=GRN)
    box(ax, 6.64, 1.38, 0.88, 0.34, "No hit", 7.4, RED, "#fdf4f6", bold=True, tc=RED)
    arr(ax, (dcx + dw / 2, dcy), (6.20, 2.06), rad=-0.24)
    arr(ax, (dcx + dw / 2, dcy), (6.20, 1.38), rad=0.24)
    lab(ax, 6.00, 2.08, "yes", 6.5, GRN, bold=True, ha="right")
    lab(ax, 6.00, 1.36, "no", 6.5, RED, bold=True, ha="right")

    # what the scheduler is told
    box(ax, 5.45, 0.42, 2.90, 0.40,
        "one bit — all the scheduler is ever told", 7.2, LINE, "white", bold=True)
    arr(ax, (dcx, dcy - dh / 2), (dcx, 0.62))

    # truth leaves the environment before the receiver ever runs
    tx, tw = 2.52, 2.50
    box(ax, tx, dcy - 0.06, tw, 0.52, "ground truth — scoring only,\nnever shown to the scheduler", 7.2)
    ax.plot([dcx, tx], [2.16, 2.16], lw=1.15, color=LINE, zorder=2)
    arr(ax, (tx, 2.16), (tx, dcy + 0.20))

    fig.savefig(HERE / "flowB_env.png", dpi=240, facecolor="white")
    plt.close(fig)


# ============================== FLOW C ======================================
def flow_c():
    W, H = 7.30, 4.35
    fig, ax = fig_ax(W, H)

    def lane(y, text):
        lab(ax, 0.10, y, text, 7.2, BLU, ha="left", bold=True)

    # ---- lane 1: the reward is screened before anything trains on it --------
    y1 = 3.62
    lane(4.22, "BEFORE TRAINING")
    box(ax, 0.98, y1, 1.24, 0.46, "Reward\ncandidate")
    diamond(ax, 2.86, y1, 1.90, 0.70, "ranks good policies\nabove known bad ones ?", 7.0)
    box(ax, 5.46, y1 + 0.28, 1.62, 0.38, "Rejected — 2 of our 6 were", 6.8, RED, "#fdf4f6", tc=RED)
    box(ax, 5.46, y1 - 0.30, 1.62, 0.38, "Train on it", 7.2, GRN, "#f3fbf6", bold=True, tc=GRN)
    arr(ax, (1.60, y1), (1.91, y1))
    arr(ax, (3.81, y1), (4.65, y1 + 0.28), rad=-0.16)
    arr(ax, (3.81, y1), (4.65, y1 - 0.30), rad=0.16)
    lab(ax, 4.40, y1 + 0.34, "fail", 6.5, RED, bold=True)
    lab(ax, 4.40, y1 - 0.38, "pass", 6.5, GRN, bold=True)

    # ---- lane 2: the training loop -----------------------------------------
    y2 = 2.42
    lane(2.96, "TRAINING")
    bw, bh = 1.20, 0.50
    xs = [0.84, 2.24, 3.64, 5.04, 6.44]
    texts = ["New scenario\nevery episode", "Pick a band", "Hit / no hit",
             "Score the\nmove", "Update the\nweights"]
    for x, t in zip(xs, texts):
        box(ax, x, y2, bw, bh, t)
    for i in range(4):
        arr(ax, (xs[i] + bw / 2, y2), (xs[i + 1] - bw / 2, y2))

    # pass -> training, as an elbow rather than a swoop
    ax.plot([5.46, 5.46], [y1 - 0.49, 3.02], lw=1.15, color=LINE, zorder=2)
    ax.plot([5.46, xs[0]], [3.02, 3.02], lw=1.15, color=LINE, zorder=2)
    arr(ax, (xs[0], 3.02), (xs[0], y2 + bh / 2))

    # the loop back
    ax.plot([xs[4], xs[4]], [y2 - bh / 2, y2 - 0.56], lw=1.15, color=LINE, zorder=2)
    ax.plot([xs[4], xs[1]], [y2 - 0.56, y2 - 0.56], lw=1.15, color=LINE, zorder=2)
    arr(ax, (xs[1], y2 - 0.56), (xs[1], y2 - bh / 2))
    lab(ax, 4.20, y2 - 0.70, "400,000 times", 6.8, INK, bold=True)
    lab(ax, 5.04, y2 + 0.44, "the score may read the truth — training is offline",
        6.3, LINE, st="italic")

    # ---- lane 3: what happens once it is deployed --------------------------
    y3 = 0.92
    lane(1.46, "DEPLOYED — weights frozen")
    box(ax, 0.94, y3, 1.32, 0.50, "183 numbers from\nits own scan history")
    box(ax, 2.66, y3, 1.24, 0.50, "A weight on\nall 36 bands")
    diamond(ax, 4.48, y3, 1.62, 0.68, "draw one, or\nalways take the best ?", 7.0)
    box(ax, 6.42, y3 + 0.34, 1.44, 0.38, "Visits 31 of 36 bands", 6.8, GRN, "#f3fbf6", tc=GRN)
    box(ax, 6.42, y3 - 0.36, 1.44, 0.38, "Collapses onto 1 band", 6.8, RED, "#fdf4f6", tc=RED)
    arr(ax, (1.60, y3), (2.04, y3))
    arr(ax, (3.28, y3), (3.67, y3))
    arr(ax, (5.29, y3), (5.70, y3 + 0.34), rad=-0.16)
    arr(ax, (5.29, y3), (5.70, y3 - 0.36), rad=0.16)
    lab(ax, 5.62, y3 + 0.40, "draw", 6.5, GRN, ha="right", bold=True)
    lab(ax, 5.62, y3 - 0.44, "best", 6.5, RED, ha="right", bold=True)

    for y in (3.22, 1.68):
        ax.plot([0.10, W - 0.10], [y, y], lw=0.7, color="#e6e9ed", zorder=0)

    fig.savefig(HERE / "flowC_rl.png", dpi=240, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    flow_b(); flow_c()
    print("wrote flowB_env.png, flowC_rl.png")
