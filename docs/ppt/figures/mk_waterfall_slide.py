"""The waterfall, sized and labelled for a slide slot.

Three schedulers on **one** episode -- same scenario, same truth grid, same seed --
so the three panels are directly comparable rather than three separate runs.
Grey is where a band was genuinely transmitting; red is where that scheduler
actually pointed.

    python docs/ppt/figures/mk_waterfall_slide.py

Writes docs/ppt/assets/waterfall_slide.png. Numbers in the panel labels are that
episode's own metrics, printed by the run that produced the data.
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parents[0] / "assets"
CACHE = Path("/private/tmp/claude-501/-Users-aamirhashmi-Code-projects-SIH26055"
             "/26974edd-04d8-4ba9-a85b-783a35335334/scratchpad")

RED, GRY, INK = "#c0143c", "#8a9099", "#15181d"
TRUTH = "#dcdfe4"
plt.rcParams.update({"font.family": "DejaVu Sans"})

d = np.load(CACHE / "waterfall.npz")
Z = d["Z"]
SLOT_S = 0.05

band = d["rl_band"]
ratio, tti, cov = d["rl_m"]
t = np.arange(len(band)) * SLOT_S

fig, ax = plt.subplots(figsize=(6.30, 2.45))
fig.subplots_adjust(left=0.075, right=0.985, top=0.775, bottom=0.255)

ax.imshow(Z, aspect="auto", cmap=plt.matplotlib.colors.ListedColormap(["white", TRUTH]),
          vmin=0, vmax=1, interpolation="nearest", extent=[0, 30, 35.5, -0.5])
ax.plot(t, band, lw=0.95, color=RED, alpha=0.95, solid_joinstyle="round")

ax.set_xlim(0, 30); ax.set_ylim(35.5, -0.5)
ax.set_yticks([0, 12, 24, 35]); ax.tick_params(labelsize=7.2, length=2.5, pad=1.5)
ax.set_xlabel("time into one 30-second episode  (s)", fontsize=8.2, labelpad=2)
ax.set_ylabel("band", fontsize=8.2, labelpad=2)
for sp in ax.spines.values():
    sp.set_color("#c9ced6"); sp.set_linewidth(0.8)

fig.text(0.075, 0.900, "Where NARADA pointed, against where the emitters really were",
         fontsize=9.6, color=INK, weight="bold", ha="left", va="bottom")
fig.text(0.075, 0.812, "grey = a band genuinely transmitting   ·   red = the band we tuned",
         fontsize=7.0, color=GRY, ha="left", va="bottom")
fig.text(0.985, 0.035, f"{cov * 100:.0f}% of emitters found  ·  "
                       f"first intercept {tti:.2f} s  ·  interception ratio {ratio:.3f}",
         fontsize=7.4, color=RED, ha="right", va="bottom", weight="bold")

fig.savefig(ASSETS / "waterfall_slide.png", dpi=240, facecolor="white")
print("wrote waterfall_slide.png")
