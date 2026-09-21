"""Two pieces: the periodic-intercept panel, and Flow C rebuilt as a method diagram.

`periodic_intercept.png` is the answer to the problem statement's sixth ask
("approaches to intercept a periodic scan receiver optimally should be outlined").
Every number in it is recomputed from `rfenv.validate`'s gate-3 case.

`flowC_method.png` replaces the training-chronology version. The reference decks
label named techniques, not the order things happened in, so this one does too --
and it is where the reward function lives now that Flow B ends at the receiver.

    python docs/ppt/figures/mk_periodic_and_method.py
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parents[0] / "assets"
CACHE = Path("/private/tmp/claude-501/-Users-aamirhashmi-Code-projects-SIH26055"
             "/26974edd-04d8-4ba9-a85b-783a35335334/scratchpad")

RED, BLU, GRY, INK, LINE = "#c0143c", "#2166ac", "#8a9099", "#15181d", "#3f4550"
plt.rcParams.update({"font.family": "DejaVu Sans"})


# ==================== the periodic-intercept panel ==========================
def periodic_panel():
    d = np.load(CACHE / "periodic.npz")
    first = d["first"]
    tau_r, T_r = float(d["tau_r"]), float(d["T_r"])
    tau_e, T_e = float(d["tau_e"]), float(d["T_e"])
    worst = int(np.argmax(first))

    fig = plt.figure(figsize=(5.4, 3.35), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.62,
                          left=0.115, right=0.975, top=0.86, bottom=0.135)

    # -- top: the two periodic trains, at the worst starting phase -----------
    ax = fig.add_subplot(gs[0])
    span = 13.4
    ph = worst * (T_e / 60.0)
    t = 0.0
    while t < span:                                  # receiver dwells
        ax.add_patch(Rectangle((t, 0.58), tau_r, 0.30, fc=BLU, ec="none"))
        t += T_r
    t = ph % T_e
    while t < span:                                  # emitter illuminations
        ax.add_patch(Rectangle((t, 0.12), tau_e, 0.30, fc=GRY, ec="none"))
        t += T_e
    ax.axvline(first[worst], color=RED, lw=1.6, ls="--", zorder=4)
    ax.annotate(f"first intercept only here — {first[worst]:.2f} s in",
                (first[worst], 0.96), xytext=(first[worst] - 0.55, 1.34),
                fontsize=7.4, color=RED, weight="bold", ha="right", va="center",
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.9))
    ax.text(-0.22, 0.73, "receiver", fontsize=7.6, color=BLU, ha="right", va="center", weight="bold")
    ax.text(-0.22, 0.27, "emitter", fontsize=7.6, color=GRY, ha="right", va="center", weight="bold")
    ax.text(span + 0.25, -0.26, f"dwell {tau_r:.2f} s every {T_r:.2f} s   ·   "
                         f"on {tau_e:.2f} s every {T_e:.2f} s",
            fontsize=6.9, color=GRY, ha="right", va="top", style="italic")
    ax.set_xlim(-0.05, span + 0.25); ax.set_ylim(0, 1.60)
    ax.set_yticks([]); ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


    # -- bottom: how much the starting phase alone is worth ------------------
    ax2 = fig.add_subplot(gs[1])
    phases = np.arange(len(first)) * (T_e / len(first))
    ax2.fill_between(phases, 0, first, color=BLU, alpha=0.16, step="post")
    ax2.step(phases, first, where="post", lw=1.8, color=BLU)
    ax2.axhline(first.max(), color=RED, lw=1.1, ls="--")
    ax2.text(T_e, first.max() - 0.35, f"worst case {first.max():.2f} s",
             fontsize=7.4, color=RED, weight="bold", ha="right", va="top")
    ax2.text(0.06, 0.55, f"best case {first.min():.2f} s", fontsize=7.4,
             color=INK, ha="left", va="bottom")
    ax2.set_xlabel("where in the emitter's cycle we happen to start  (s)", fontsize=8.0)
    ax2.set_ylabel("time to first\nintercept  (s)", fontsize=8.0, linespacing=1.25)
    ax2.set_xlim(0, T_e); ax2.set_ylim(0, first.max() * 1.17)
    ax2.tick_params(labelsize=7.4, length=3)
    ax2.grid(True, lw=0.5, color="#eceff3"); ax2.set_axisbelow(True)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax2.spines[sp].set_color("#c9ced6")

    fig.text(0.115, 0.925, "A fixed sweep is at the mercy of the emitter's phase",
             fontsize=9.4, color=INK, weight="bold", ha="left", va="bottom")
    fig.savefig(ASSETS / "periodic_intercept.png", dpi=240, facecolor="white")
    plt.close(fig)
    print("   periodic_intercept.png  "
          f"(60 phases, {first.min():.2f}–{first.max():.2f} s)")


# ==================== Flow C, as a method diagram ==========================
def flow_c_method():
    W, H = 6.40, 2.52
    fig = plt.figure(figsize=(W, H), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    def box(cx, cy, w, h, head, body, ec=LINE, fc="white", hc=BLU):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                    boxstyle="round,pad=0.004,rounding_size=0.04",
                                    fc=fc, ec=ec, lw=1.2, zorder=3))
        ax.text(cx, cy + h / 2 - 0.13, head, fontsize=7.5, color=hc, ha="center",
                va="center", weight="bold", zorder=4)
        ax.text(cx, cy - 0.09, body, fontsize=6.8, color=GRY, ha="center",
                va="center", linespacing=1.28, zorder=4)

    def arr(p0, p1, col=LINE, ls="-", rad=0.0):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11,
                                     lw=1.15, color=col, linestyle=ls, zorder=2,
                                     shrinkA=1.5, shrinkB=1.5,
                                     connectionstyle=f"arc3,rad={rad}"))

    cy, bh = 1.82, 0.78
    xs, bw = [0.82, 2.38, 3.94, 5.44], 1.42
    box(xs[0], cy, bw, bh, "SELF-DERIVED\nOBSERVATION", "183 values, built only\nfrom its own scan history")
    box(xs[1], cy, bw, bh, "RECURRENT\nPOLICY (LSTM)", "carries memory across\ndwells, not just the last")
    box(xs[2], cy, bw, bh, "ACTION\nDISTRIBUTION", "a weight on every one\nof the 36 bands")
    box(xs[3], cy, 1.34, bh, "STOCHASTIC\nSELECTION", "draws the next band —\nnever the argmax", ec=RED, hc=RED)
    for i in range(3):
        arr((xs[i] + bw / 2, cy), (xs[i + 1] - (1.34 if i == 2 else bw) / 2, cy))

    # the reward: trains the policy, never seen at run time
    rx, ry, rw, rh = 3.16, 0.62, 5.86, 0.98
    ax.add_patch(FancyBboxPatch((rx - rw / 2, ry - rh / 2), rw, rh,
                                boxstyle="round,pad=0.004,rounding_size=0.04",
                                fc="#f7f9fc", ec=BLU, lw=1.2, ls=(0, (4, 2.5)), zorder=3))
    ax.text(rx, ry + 0.36, "SHAPED REWARD  —  training only", fontsize=7.5,
            color=BLU, ha="center", va="center", weight="bold", zorder=4)
    ax.text(rx, ry + 0.09,
            r"$R \;=\; 0.5\,h_a \;+\; (1.5 - d_a)\,s_a \;+\; 0.5\sum Z \;-\; 3.0\,d_a\,n$",
            fontsize=10.0, color=INK, ha="center", va="center", zorder=4)
    ax.text(rx, ry - 0.17,
            "exploit what pays   ·   explore what is overdue   ·   cost of hogging airtime",
            fontsize=6.4, color=GRY, ha="center", va="center", zorder=4)
    ax.text(rx, ry - 0.36,
            "$h$ hit rate   $d$ airtime share   $s$ staleness   $n$ slots in the dwell   "
            "— all read from its own observation",
            fontsize=6.0, color=GRY, ha="center", va="center", style="italic", zorder=4)
    arr((xs[1], ry + rh / 2), (xs[1], cy - bh / 2), col=BLU, ls="--")

    fig.savefig(HERE / "flowC_method.png", dpi=240, facecolor="white")
    plt.close(fig)
    print("   flowC_method.png")


# ==================== the reward screen (D62) ===============================
def reward_screen():
    d = np.load(CACHE / "screen.npz")
    keys = [str(k) for k in d["keys"]]
    mean, sd = d["mean"], d["sd"]
    pretty = {"recency": "Recency heuristic\n(the bar)",
              "apfeld_active_rfs": "Apfeld\n(literature)",
              "round_robin": "Round-robin\n(the floor)",
              "camper": "Camper\n(the failure mode)"}
    cols = [BLU, GRY, GRY, RED]

    fig, ax = plt.subplots(figsize=(5.30, 2.85))
    y = np.arange(len(keys))[::-1]
    ax.barh(y, mean, xerr=sd, height=0.62, color=cols, alpha=0.92,
            error_kw=dict(ecolor="#9aa4b2", lw=1.1, capsize=3))
    ax.axvline(0, lw=1.0, color="#6b7280")
    ax.set_yticks(y)
    ax.set_yticklabels([pretty[k] for k in keys], fontsize=7.4, linespacing=1.25)
    ax.set_xlabel("reward the candidate gives each policy   (8 seeds)", fontsize=8.0)
    ax.tick_params(labelsize=7.4, length=3)
    ax.grid(True, axis="x", lw=0.5, color="#eceff3"); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#c9ced6")
    ax.set_xlim(-620, 470)

    ax.text(-414, y[-1] + 0.38, f"{float(d['exp']):.1f}σ below both sweeps",
            fontsize=7.2, color=RED, ha="center", va="bottom", weight="bold")
    fig.text(0.02, 0.945, "A reward is screened before anything trains on it",
             fontsize=9.2, color=INK, weight="bold", ha="left", va="bottom")
    fig.text(0.02, 0.868,
             "the bar must sit clearly above the floor, the failure mode clearly below both\n"
             f"+59 reward  ·  {float(d['sep']):.1f}σ apart  ·  8 of 8 seeds",
             fontsize=6.9, color=GRY, ha="left", va="bottom", linespacing=1.35)
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.835))
    fig.savefig(ASSETS / "reward_screen.png", dpi=240, facecolor="white")
    plt.close(fig)
    print("   reward_screen.png  (%.2f sigma, bar %.1f)" % (float(d["sep"]), float(d["minsep"])))


if __name__ == "__main__":
    periodic_panel(); flow_c_method(); reward_screen()
