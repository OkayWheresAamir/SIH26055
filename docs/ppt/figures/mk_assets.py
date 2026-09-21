"""Standalone diagram assets for the deck — drop straight into PowerPoint.

These are the pictures that came out of the project's own implementation, each
cropped tight with no title, no frame and no caption, so the slide can place and
label them however it likes. The flowcharts are built separately; this is the
raw material for them.

    python docs/ppt/figures/mk_assets.py

Writes docs/ppt/assets/*.png. Everything here is measured, not illustrative:
see assets/README.md for what each one is and where it came from.
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[0] / "assets"
OUT.mkdir(exist_ok=True)
CACHE = Path("/private/tmp/claude-501/-Users-aamirhashmi-Code-projects-SIH26055"
             "/26974edd-04d8-4ba9-a85b-783a35335334/scratchpad")

RED, BLU, ORG, GRY = "#c0143c", "#2166ac", "#e08214", "#8a9099"
HEAT = LinearSegmentedColormap.from_list("heat", ["#f8fafc", "#bfdbfe", "#2166ac", "#0b2e4f"])

R = np.load(CACHE / "real.npz")
P = np.load(CACHE / "pdw.npz")
Z, OBS, PROBS, RL_BAND = R["Z"], R["obs"], R["probs"], R["rl_band"]
PDW, LAB = P["d"], P["lab"]
DPI = 200


def bare(w, h, facecolor="white"):
    fig = plt.figure(figsize=(w, h), facecolor=facecolor)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  ", name)


# 1 — the spectrum we must watch, with the receiver's window on one band
def spectrum_bars():
    fig, ax = bare(3.6, 0.85)
    occ = Z.mean(axis=1)
    ax.set_xlim(-0.5, 35.5); ax.set_ylim(0, 1.12)
    for b in range(36):
        ax.plot([b, b], [0.04, 0.10 + 0.82 * (occ[b] / occ.max()) ** 0.6],
                lw=4.0, color="#9aa4b2", solid_capstyle="round")
    ax.add_patch(plt.Rectangle((10.5, 0.0), 1.0, 1.06, fc="none", ec=RED, lw=2.2))
    save(fig, "spectrum_bars.png")


# 2 — every recorded pulse: time vs frequency, coloured by power
def pulse_cloud():
    fig, ax = bare(3.4, 1.9, facecolor="#0b1220")
    ax.set_facecolor("#0b1220")
    idx = np.random.default_rng(0).choice(len(PDW), 14000, replace=False)
    ax.scatter(PDW[idx, 0] / 1e6, PDW[idx, 1] / 1000, c=PDW[idx, 4], s=1.4,
               cmap="magma", vmin=-140, vmax=-40, linewidths=0)
    save(fig, "pulse_cloud.png")


# 3 — one emitter's beam sweeping past: peak power every 0.1 s
def beam_lobes():
    fig, ax = bare(3.4, 1.45)
    swing, who = -1e9, 0
    for lb in np.unique(LAB):
        m = LAB == lb
        if m.sum() < 500:
            continue
        d = np.percentile(PDW[m, 4], 99) - np.percentile(PDW[m, 4], 20)
        if d > swing:
            swing, who = d, lb
    m = LAB == who
    nb = 300
    b_i = np.clip((PDW[m, 0] / 30e6 * nb).astype(int), 0, nb - 1)
    env = np.full(nb, -np.inf)
    np.maximum.at(env, b_i, PDW[m, 4])
    env[np.isneginf(env)] = np.nan
    ax.plot(np.linspace(0, 30, nb), env, lw=2.0, color=ORG)
    ax.set_xlim(0, 30)
    ax.set_facecolor("#fffdf9")
    save(fig, "beam_lobes.png")


# 4 — the band x slot occupancy grid, cropped so cells are visible
def occupancy_grid():
    fig, ax = bare(3.4, 1.9)
    crop = Z[:, 210:270]
    ax.imshow(crop, aspect="auto", cmap="Greys", vmin=0, vmax=1.2, interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, crop.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 36, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.7)
    ax.tick_params(which="minor", length=0)
    save(fig, "occupancy_grid.png")


# 5 — the detection curve: chance of a hit against how loud the signal is
def detection_curve():
    from math import erf
    fig, ax = bare(2.0, 1.25)
    xg = np.linspace(-10, 10, 300)
    ax.plot(xg, 0.5 * (1 + np.array([erf(v / (3 * 2 ** .5)) for v in xg])), lw=2.6, color=BLU)
    ax.axvline(0, lw=1.4, color=RED, ls="--")
    ax.set_xlim(-10, 10); ax.set_ylim(-0.03, 1.03)
    save(fig, "detection_curve.png")


# 6 — what the scheduler remembers: the real 36 x 5 observation
def obs_heatmap():
    fig, ax = bare(3.4, 0.95)
    rows = OBS[:36 * 5].reshape(5, 36).astype(float)
    rows = np.clip(rows / np.maximum(np.percentile(rows, 92, axis=1, keepdims=True), 1e-9), 0, 1)
    ax.imshow(rows, cmap=HEAT, aspect="auto", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, 36, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="minor", length=0)
    save(fig, "obs_heatmap.png")


# 7 — the policy's real output: a weight on every band
def band_distribution():
    fig, ax = bare(3.4, 1.1)
    top = int(PROBS.argmax())
    ax.bar(np.arange(36), PROBS, width=0.86,
           color=[RED if i == top else "#9aa4b2" for i in range(36)])
    ax.set_xlim(-0.8, 35.8); ax.set_ylim(0, PROBS.max() * 1.14)
    save(fig, "band_distribution.png")


# 8 — where we pointed against where the emitters were, one whole episode
def waterfall_strip():
    fig, ax = bare(6.4, 1.35)
    ax.imshow(Z, aspect="auto", cmap="Greys", vmin=0, vmax=1.7, interpolation="nearest",
              extent=[0, 600, 35.5, -0.5])
    ax.plot(np.arange(len(RL_BAND)), RL_BAND, lw=0.9, color=RED, alpha=0.85)
    ax.set_xlim(0, 600); ax.set_ylim(35.5, -0.5)
    save(fig, "waterfall_strip.png")


# 9 — the 36-band ladder with the tuned band lit
def band_ladder():
    fig, ax = bare(0.75, 1.9)
    ax.set_xlim(0, 1); ax.set_ylim(35.5, -0.5)
    for b in range(36):
        hot = b == 24
        ax.plot([0.08, 0.66 if hot else 0.50], [b, b], lw=3.4 if hot else 1.2,
                color=RED if hot else "#dfe3e8", solid_capstyle="butt")
    save(fig, "band_ladder.png")


if __name__ == "__main__":
    print("writing", OUT)
    spectrum_bars(); pulse_cloud(); beam_lobes(); occupancy_grid()
    detection_curve(); obs_heatmap(); band_distribution(); waterfall_strip(); band_ladder()
