"""Pareto, sized for a slide corner — legible at ~5 in wide.

Five markers only. The hero is `lstm_balance_v2p_priority_strong_seed2_800k` as
scored in `runs/d74_followup_treatment_comparison/`, which is the surviving,
re-runnable artefact for that checkpoint. The baselines come from
`runs/final_2026-09-11/`; the two runs share `round_robin` and `recency` to four
decimals, which is the check that they are the same scenario set and seeds.

    python docs/ppt/figures/mk_pareto_small.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

FINAL = json.load(open(ROOT / "runs/final_2026-09-11/summary.json"))["ladder"]
FOLLOW = json.load(open(ROOT / "runs/d74_followup_treatment_comparison/summary.json"))["ladder"]

for k in ("round_robin", "recency"):
    a, b = FINAL[k]["aggregate"], FOLLOW[k]["aggregate"]
    for m in ("interception_ratio", "censored_mean_intercept_time_s"):
        assert abs(a[m]["mean"] - b[m]["mean"]) < 5e-4, f"{k}/{m} differs between runs"

RED, BLU, ORG, GRY = "#c0143c", "#2166ac", "#e08214", "#8a9099"

# source, key, label, colour, label offset (dx, dy in axis units), ha
PTS = [
    (FOLLOW, "lstm_balance_v2p_priority_strong_seed2_800k",
     "NARADA", RED, (0.10, 0.0035), "left"),
    (FINAL, "recency", "Best heuristic\n(our own bar)", BLU, (0.10, -0.0060), "left"),
    (FINAL, "apfeld_active_rfs", "Best in literature", ORG, (-0.12, 0.0026), "right"),
    (FINAL, "turing_sweep", "Reference sweep", GRY, (0.00, -0.0075), "center"),
    (FINAL, "round_robin", "Open-loop floor", GRY, (-0.12, 0.0000), "right"),
]

fig, ax = plt.subplots(figsize=(5.15, 3.35))

for src, key, lab, col, (dx, dy), ha in PTS:
    a = src[key]["aggregate"]
    x = a["censored_mean_intercept_time_s"]["mean"]
    y = a["interception_ratio"]["mean"]
    hero = col == RED
    ax.scatter(x, y, s=210 if hero else 95, c=col, zorder=5,
               edgecolors="white", linewidths=1.8)
    ax.annotate(lab, (x, y), xytext=(x + dx, y + dy), ha=ha, va="center",
                fontsize=9.2 if hero else 8.2, color=col,
                weight="bold" if hero else "normal", linespacing=1.15, zorder=6)

ax.annotate("", xy=(2.30, 0.0745), xytext=(2.86, 0.0530),
            arrowprops=dict(arrowstyle="-|>", color="#2ca25f", lw=2.0,
                            shrinkA=0, shrinkB=0), zorder=3)
ax.text(2.92, 0.0520, "better", fontsize=9.0, color="#2ca25f", weight="bold",
        ha="left", va="bottom")

ax.set_xlabel("Time to first intercept  (s)      ← lower is better", fontsize=8.8)
ax.set_ylabel("Interception ratio\nhigher is better →", fontsize=8.8, linespacing=1.3)
ax.set_xlim(2.15, 4.95)
ax.set_ylim(0.045, 0.163)
ax.set_yticks([0.06, 0.09, 0.12, 0.15])
ax.set_xticks([2.5, 3.0, 3.5, 4.0, 4.5])
ax.tick_params(labelsize=8.0, length=3)
ax.grid(True, lw=0.5, color="#e8ebef", zorder=0)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color("#c9ced6")

fig.tight_layout(pad=0.45, rect=(0, 0.085, 1, 1))
fig.text(0.5, 0.012,
         "Off this chart: two strategies that chase ratio alone reach 9.7 s and 14.9 s "
         "by abandoning half the emitters.",
         fontsize=6.8, color=GRY, ha="center", va="bottom", style="italic")
fig.savefig(HERE / "fig1_pareto_small.png", dpi=240, facecolor="white")
print("wrote fig1_pareto_small.png")
