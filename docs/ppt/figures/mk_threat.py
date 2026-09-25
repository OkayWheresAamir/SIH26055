"""PPT figure 4 — the threat block as a decision story, not a bar chart.
Three attempts, two of them wrong. The evidence is a one-line caption each;
the substance is the reasoning that moved us."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

RED, GRN, GRY, INK = "#c0143c", "#1e7a4d", "#9aa0a6", "#17181a"

fig, ax = plt.subplots(figsize=(9.8, 4.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 4.5); ax.axis("off")

ax.text(0.1, 4.26, "How we ended up here", fontsize=14, weight="bold", color=INK)
ax.text(0.1, 3.92, "Two of our three attempts were wrong. Each one told us what to build next.",
        fontsize=10.4, color="#555", style="italic")

STEPS = [
    ("1", "Let the agent work out\nwhat is threatening",
     "It can't. A fire-control radar and a\nweather radar look the same to a\nreceiver that only hears hit / no-hit.",
     "→ priority must come\n   from outside", RED, "✕"),
    ("2", "Flag any band holding\na dangerous emitter",
     "Almost every band holds one — our\nbands overlap by half, so a single band\ncarries several emitters at once.",
     "→ a yes/no flag\n   cannot rank", RED, "✕"),
    ("3", "Ask how MUCH of a band\nis dangerous, not whether",
     "Weight each emitter by its share of\nthat band's traffic. Now bands can be\nordered, and the ordering is usable.",
     "→ this is what\n   we build", GRN, "✓"),
]
x = 0.1
for tag, head, body, verdict, col, mark in STEPS:
    ax.add_patch(FancyBboxPatch((x, 0.85), 3.06, 2.75,
                 boxstyle="round,pad=0.03,rounding_size=0.08",
                 fc="white" if col == RED else "#f2faf5", ec=col, lw=2.2))
    ax.text(x + 0.20, 3.34, tag, fontsize=15, weight="bold", color=col)
    ax.text(x + 2.86, 3.34, mark, fontsize=16, weight="bold", color=col, ha="right")
    ax.text(x + 0.20, 3.02, head, fontsize=11.2, weight="bold", color=INK,
            va="top", linespacing=1.35)
    ax.text(x + 0.20, 2.24, body, fontsize=9.5, color="#3a3d42",
            va="top", linespacing=1.45)
    ax.text(x + 0.20, 1.52, verdict, fontsize=9.8, color=col, weight="bold",
            va="top", style="italic", linespacing=1.3)
    if x < 6:
        ax.add_patch(FancyArrowPatch((x + 3.10, 2.22), (x + 3.36, 2.22),
                     arrowstyle="-|>", mutation_scale=16, lw=2.0, color=GRY))
    x += 3.36

ax.text(5.0, 0.36,
        "And the reason it matters: our own agent was found to reach threatening emitters\n"
        "later than harmless ones — and the gap widened as the agent got better.",
        fontsize=10.2, color=INK, ha="center", va="center", linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.5", fc="#fdf3f5", ec="#e8c3cb"))

fig.tight_layout()
fig.savefig("/private/tmp/claude-501/-Users-aamirhashmi-Code-projects-SIH26055/90581c6c-2a45-4658-9929-b806de01bddc/scratchpad/fig/fig4_threat.png", dpi=200, facecolor="white")
print("ok")
