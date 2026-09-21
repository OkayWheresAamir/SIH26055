"""The three "research-worthy" panels: an ROC, a scenario-diversity strip, equations.

These fill the slot Falkon used for its weight-generator maths and Cattle Race used
for its robust-image-performance photo grid.

    python docs/ppt/figures/mk_research.py

All three are measured, not illustrative. The ROC is swept over the 35 training
configs at the frozen sigma; the diversity strip is four real truth grids spanning
the development set from 1 detectable emitter to 76.
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[0] / "assets"
CACHE = Path("/private/tmp/claude-501/-Users-aamirhashmi-Code-projects-SIH26055"
             "/26974edd-04d8-4ba9-a85b-783a35335334/scratchpad")

RED, BLU, GRY, INK = "#c0143c", "#2166ac", "#8a9099", "#15181d"
plt.rcParams.update({"font.family": "DejaVu Sans"})


# ---------------------------------------------------------------- ROC --------
def roc_curve():
    d = np.load(CACHE / "roc.npz")
    pd, pfa = d["pd"], d["pfa"]
    opd, opfa = float(d["op_pd"]), float(d["op_pfa"])

    fig, ax = plt.subplots(figsize=(3.55, 3.05))
    ax.plot(pfa, pd, lw=2.4, color=BLU, zorder=3)
    ax.plot([0, 1], [0, 1], lw=1.0, color="#c9ced6", ls=(0, (4, 3)), zorder=1)
    ax.scatter([opfa], [opd], s=120, c=RED, edgecolors="white", linewidths=1.8, zorder=5)
    ax.annotate("operating point\n$P_d$ 0.84 · $P_{fa}$ 0.0013",
                (opfa, opd), xytext=(0.20, 0.62), fontsize=7.6, color=RED, weight="bold",
                linespacing=1.3,
                arrowprops=dict(arrowstyle="-", color=RED, lw=1.0, shrinkA=0, shrinkB=4))
    ax.text(0.97, 0.10, "sensitivity −107.2 dBm\nthreshold frozen before any\nscheduler was scored",
            fontsize=6.9, color=GRY, ha="right", va="bottom", linespacing=1.35, style="italic")

    ax.set_xlabel("Probability of false alarm  $P_{fa}$", fontsize=8.4)
    ax.set_ylabel("Probability of detection  $P_d$", fontsize=8.4)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.tick_params(labelsize=7.6, length=3)
    ax.grid(True, lw=0.5, color="#eceff3", zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c9ced6")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "roc_curve.png", dpi=240, facecolor="white")
    plt.close(fig)
    print("   roc_curve.png")


# ------------------------------------------------------- scenario diversity --
def diversity_strip():
    d = np.load(CACHE / "diversity.npz", allow_pickle=True)
    meta = {str(r[0]): (int(r[1]), int(r[2]), float(r[3])) for r in d["meta"]}
    order = sorted(meta, key=lambda c: meta[c][1])

    fig, axes = plt.subplots(1, 4, figsize=(7.1, 1.85))
    for ax, cfg in zip(axes, order):
        Z = d[f"Z_{cfg}"]
        ax.imshow(Z, aspect="auto", cmap="Greys", vmin=0, vmax=1.35, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#c9ced6"); sp.set_linewidth(1.0)
        n_em, n_det, occ = meta[cfg]
        ax.set_title(f"{n_det} emitter" + ("s" if n_det != 1 else ""),
                     fontsize=8.6, color=INK, weight="bold", pad=4)
        pct = f"{occ * 100:.1f}" if occ < 0.05 else f"{occ * 100:.0f}"
        ax.set_xlabel(f"{pct}% of the grid busy", fontsize=7.4, color=GRY, labelpad=3)
    fig.tight_layout(pad=0.45, w_pad=1.1)
    fig.savefig(OUT / "scenario_diversity.png", dpi=240, facecolor="white")
    plt.close(fig)
    print("   scenario_diversity.png  (" +
          ", ".join(f"{meta[c][1]}" for c in order) + " detectable emitters)")


# -------------------------------------------------------------- equations ----
def equation_block():
    fig = plt.figure(figsize=(4.05, 2.42), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 4.05); ax.set_ylim(0, 2.42); ax.axis("off")

    items = [
        ("The receiver declares a hit",
         r"$Y_{b,t} = \mathbb{1}\,[\, S_{b,t} + n \geq \gamma \,],\;\; n \sim \mathcal{N}(0,\sigma^2)$"),
        ("so detection is a curve, not a constant",
         r"$P_d(S) = \Phi\!\left(\dfrac{S-\gamma}{\sigma}\right)$"),
        ("and a missed emitter costs a full episode",
         r"$\bar{T} = \dfrac{1}{|E|}\sum_{e \in E}\left(\mathrm{first}_e - \mathrm{on}_e\right),\;\;"
         r"\mathrm{first}_e \equiv T$"),
    ]
    y = 2.30
    for cap, eq in items:
        ax.text(0.12, y, cap, fontsize=7.4, color=GRY, ha="left", va="top", style="italic")
        ax.text(0.20, y - 0.31, eq, fontsize=10.2, color=INK, ha="left", va="center")
        y -= 0.72
    ax.text(0.12, 0.09, "an emitter we never found is charged the whole 30 s —\n"
                        "the convention that stops a camper winning",
            fontsize=6.7, color=RED, ha="left", va="bottom", linespacing=1.3)
    fig.savefig(OUT / "equations.png", dpi=240, facecolor="white")
    plt.close(fig)
    print("   equations.png")


if __name__ == "__main__":
    print("writing", OUT)
    roc_curve(); diversity_strip(); equation_block()
