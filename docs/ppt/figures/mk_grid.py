"""The 36 x 600 grid, on its own.

The picture behind flowB_env.png's "fill the 36 x 600 band-slot grid" box: one
real stare replay, every band, every slot, whether or not anyone looked. No
annotation -- everything else about it gets said out loud.

Numbers are computed here at draw time from rfenv. Self-contained on purpose:
the older figure scripts read a scratch cache that no longer exists.

    python docs/ppt/figures/mk_grid.py
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from rfenv.constants import BAND_CENTRES_MHZ, EPISODE_S, GAMMA_DBM, N_BANDS, N_SLOTS
from rfenv.scenario import Scenario
from rfenv.truth import TruthGrid

HERE = Path(__file__).resolve().parent
CONFIG = "config_1366"          # a train stare replay, mid-range occupancy
INK, GRY = "#15181d", "#9aa1ab"
HEAT = LinearSegmentedColormap.from_list("heat", ["#f4f7fb", "#bfdbfe", "#2166ac", "#0b2e4f"])
plt.rcParams.update({"font.family": "DejaVu Sans"})


def main():
    grid = TruthGrid.from_scenario(Scenario.replay(CONFIG, "stare"))
    occ_S = grid.S[grid.Z]

    W, H = 6.40, 2.70
    fig = plt.figure(figsize=(W, H), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    ax.text(0.62, 2.48, "36 bands × 600 slots  —  one 30 second episode",
            fontsize=9.0, color=INK, ha="left", va="center", weight="bold")

    gx0, gy0, gw, gh = 0.62, 0.52, 5.10, 1.78
    g = fig.add_axes([gx0 / W, gy0 / H, gw / W, gh / H])
    im = g.imshow(grid.S, cmap=HEAT, aspect="auto", origin="lower",
                  vmin=GAMMA_DBM - 12.0, vmax=float(np.percentile(occ_S, 97)),
                  interpolation="nearest", extent=[0, EPISODE_S, -0.5, N_BANDS - 0.5])
    g.set_xlim(0, EPISODE_S); g.set_ylim(-0.5, N_BANDS - 0.5)
    g.set_xticks([0, 10, 20, 30]); g.set_xticklabels(["0", "10 s", "20 s", "30 s"])
    g.set_yticks([0, 8, 17, 26, 35])
    g.set_yticklabels([f"{BAND_CENTRES_MHZ[i] / 1000:.1f}" for i in (0, 8, 17, 26, 35)])
    g.tick_params(labelsize=6.6, length=2, color=GRY, labelcolor="#4b5563")
    for s in g.spines.values():
        s.set_color("#cbd2da"); s.set_linewidth(0.8)

    ax.text(0.20, gy0 + gh / 2, "band centre, GHz", fontsize=7.0, color="#4b5563",
            ha="center", va="center", rotation=90)
    ax.text(gx0 + gw / 2, 0.20, "600 slots of 50 ms", fontsize=7.0, color="#4b5563",
            ha="center", va="center")

    cax = fig.add_axes([5.86 / W, gy0 / H, 0.11 / W, gh / H])
    cb = fig.colorbar(im, cax=cax)
    cax.tick_params(labelsize=6.0, length=2, color=GRY, labelcolor="#4b5563")
    cax.set_title("dBm", fontsize=6.2, color="#4b5563", pad=4)
    cb.outline.set_edgecolor("#cbd2da"); cb.outline.set_linewidth(0.8)

    out = HERE / "flowB_grid.png"
    fig.savefig(out, dpi=240, facecolor="white")
    plt.close(fig)
    n_cells = N_BANDS * N_SLOTS
    print(f"wrote {out}")
    print(f"  {CONFIG}: {grid.n_emitters} emitters, {grid.total_pulses:,} pulses, "
          f"{int(grid.Z.sum()):,} of {n_cells:,} cells live "
          f"({100 * grid.Z.mean():.1f}%), {100 * float((occ_S < GAMMA_DBM).mean()):.1f}% "
          f"of those below gamma")


if __name__ == "__main__":
    main()
