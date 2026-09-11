"""PPT figure 1 — Pareto, redrawn for a slide."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("runs/final_2026-09-11/summary.json"))["ladder"]
G = lambda k: d[k]["aggregate"]

# key, label, colour, (label x, label y), ha, va
PTS = [
    ("lstm_balance_d67_control_seed2_300k_400k",
        "NARADA\nRecurrent PPO", "#c0143c", (2.62, 0.1545), "center", "bottom"),
    ("recency",          "Recency heuristic\nthe bar we set", "#2166ac", (2.42, 0.1035), "left", "top"),
    ("apfeld_active_rfs","Apfeld\nbest in the literature",    "#e08214", (4.62, 0.1345), "left", "center"),
    ("turing_sweep",     "Turing reference sweep",            "#8a9099", (3.60, 0.0735), "center", "top"),
    ("random",           "Random",                            "#8a9099", (4.40, 0.0680), "left", "center"),
    ("round_robin",      "Round-robin — open-loop floor",     "#8a9099", (4.40, 0.0590), "left", "center"),
]

fig, ax = plt.subplots(figsize=(9.6, 5.2))
for key, lab, col, (lx, ly), ha, va in PTS:
    a = G(key)
    x, y = a["censored_mean_intercept_time_s"]["mean"], a["interception_ratio"]["mean"]
    cov = a["emitter_coverage"]["mean"]
    hero = key.startswith("lstm")
    ax.plot([x, lx], [y, ly], lw=0.9, color=col, alpha=.4, zorder=2)
    ax.scatter(x, y, s=1200 * cov ** 2.2, c=col, alpha=.95,
               edgecolors="white", linewidths=2.2, zorder=4 + hero)
    ax.text(lx, ly, lab, fontsize=11.2, color=col, ha=ha, va=va, zorder=6,
            weight="bold" if hero or "bar" in lab else "normal", linespacing=1.25)

ax.annotate("", xy=(5.32, 0.1560), xytext=(5.95, 0.1080),
            arrowprops=dict(arrowstyle="-|>", lw=2.8, color="#2ca25f", alpha=.8))
ax.text(5.98, 0.1035, "better", color="#2ca25f", fontsize=13, weight="bold",
        ha="right", va="top")

ax.text(2.40, 0.0455,
        "Off this chart:  Camper 9.7 s  ·  Apfeld-full 14.9 s\n"
        "They buy a high ratio by abandoning 50–63% of the emitters.",
        fontsize=9.4, color="#555", style="italic", ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.5", fc="#f6f6f6", ec="#dcdcdc"))

ax.set_xlabel("Censored mean intercept time  (s)    ← lower is better", fontsize=11.5)
ax.set_ylabel("Interception ratio     higher is better →", fontsize=11.5)
ax.set_title("The two objectives DRDO names are in direct tension —\n"
             "the learned scheduler is the only rung that improves both at once",
             fontsize=13.5, weight="bold", pad=14)
ax.set_xlim(2.30, 6.30); ax.set_ylim(0.036, 0.168)
ax.grid(alpha=.22, linestyle=":")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.text(0.995, 0.012, "marker size = emitter coverage   |   5,643 episodes · 57 scenarios · 3 seeds",
        transform=ax.transAxes, ha="right", fontsize=8.6, color="#999")
fig.tight_layout()
fig.savefig("/private/tmp/claude-501/-Users-aamirhashmi-Code-projects-SIH26055/90581c6c-2a45-4658-9929-b806de01bddc/scratchpad/fig/fig1_pareto.png", dpi=200, facecolor="white")
print("ok")
