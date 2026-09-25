"""Flow B and Flow C, second pass — smaller, and the arrows carry the logic.

The v1 versions read as a gallery of images. Here every stage is a text box that
says what it is, every arrow is labelled with the transformation it performs, and
the pictures sit *inside* the boxes as evidence rather than as the content.

    python docs/ppt/figures/mk_slide_flows_v2.py

Writes flowB_world.png and flowC_policy.png beside this file. Real data as before.
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
CACHE = Path("/private/tmp/claude-501/-Users-aamirhashmi-Code-projects-SIH26055"
             "/26974edd-04d8-4ba9-a85b-783a35335334/scratchpad")

RED, BLU, GRY, GRN, ORG = "#c0143c", "#2166ac", "#6b7280", "#2ca25f", "#e08214"
INK, FAINT = "#111827", "#d1d5db"
HEAT = LinearSegmentedColormap.from_list("heat", ["#f8fafc", "#bfdbfe", "#2166ac", "#0b2e4f"])
plt.rcParams.update({"font.family": "DejaVu Sans"})

R = np.load(CACHE / "real.npz")
P = np.load(CACHE / "pdw.npz")
Z, OBS, PROBS = R["Z"], R["obs"], R["probs"]
PDW, LAB = P["d"], P["lab"]


def box(ax, x, y, w, h, fc="white", ec=GRY, lw=1.7, z=2, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.02",
                                fc=fc, ec=ec, lw=lw, zorder=z, **kw))


def arrow(ax, p0, p1, col=GRY, lw=2.1, ls="-", rad=0.0, ms=15, z=5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw,
                                 color=col, linestyle=ls, zorder=z, shrinkA=2, shrinkB=2,
                                 connectionstyle=f"arc3,rad={rad}"))


def cap(ax, x, y, s, fs=8.0, col=GRY, w="normal", ha="center", va="top", st="normal", z=8):
    ax.text(x, y, s, fontsize=fs, color=col, ha=ha, va=va, weight=w, style=st,
            linespacing=1.30, zorder=z)


def inset(fig, rect):
    a = fig.add_axes(rect)
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(False)
    return a


# ============================== FLOW B — the world ===========================
def flow_b():
    W, H = 8.10, 3.80
    fig = plt.figure(figsize=(W, H), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    cap(ax, 0.22, H - 0.08, "THE ENVIRONMENT — built from real radar, not assumed",
        10.2, INK, "bold", ha="left")

    by, bh, bw = 1.72, 1.62, 2.06
    xs = [0.22, 3.02, 5.82]
    heads = ["RECORDED RADAR", "WHAT EACH EMITTER DOES", "THE WORLD, EVERY 50 ms"]
    bodies = ["92 files · 2.2 GB\n644,674 pulses in one 30 s file",
              "3,443 reusable emitters — each beam\nsweeps past, so it comes and goes",
              "36 bands × 600 slots — who is\non air, whether or not we look"]
    for x, hd, bd in zip(xs, heads, bodies):
        box(ax, x, by, bw, bh, ec=BLU)
        cap(ax, x + bw / 2, by + bh - 0.09, hd, 8.6, BLU, "bold")
        cap(ax, x + bw / 2, by + 0.34, bd, 7.3, GRY)

    # labelled arrows — these are what make it a chain rather than a gallery
    for i, lab in enumerate(["read every\npulse", "onto the\nband grid"]):
        x0, x1 = xs[i] + bw, xs[i + 1]
        ay = by + bh * 0.42
        arrow(ax, (x0 + 0.03, ay), (x1 - 0.03, ay), col=BLU)
        cap(ax, (x0 + x1) / 2, ay + 0.09, lab, 6.9, BLU, "bold", va="bottom")

    # evidence inside box 1 — the real pulse cloud
    a1 = inset(fig, [(xs[0] + 0.14) / W, (by + 0.56) / H, 1.78 / W, 0.70 / H])
    idx = np.random.default_rng(0).choice(len(PDW), 9000, replace=False)
    a1.scatter(PDW[idx, 0] / 1e6, PDW[idx, 1] / 1000, c=PDW[idx, 4], s=0.9,
               cmap="magma", vmin=-140, vmax=-40, linewidths=0)
    a1.set_facecolor("#0b1220")

    # evidence inside box 2 — the real beam lobes
    a2 = inset(fig, [(xs[1] + 0.14) / W, (by + 0.56) / H, 1.78 / W, 0.70 / H])
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
    a2.plot(np.linspace(0, 30, nb), env, lw=1.4, color=ORG)
    a2.set_facecolor("#fffaf3")

    # evidence inside box 3 — the real occupancy grid
    a3 = inset(fig, [(xs[2] + 0.14) / W, (by + 0.56) / H, 1.78 / W, 0.70 / H])
    crop = Z[:, 200:280]
    a3.imshow(crop, aspect="auto", cmap="Greys", vmin=0, vmax=1.25, interpolation="nearest")
    a3.set_xticks(np.arange(-0.5, crop.shape[1], 1), minor=True)
    a3.set_yticks(np.arange(-0.5, 36, 1), minor=True)
    a3.grid(which="minor", color="white", lw=0.4)
    a3.tick_params(which="minor", length=0)

    # -- the receiver, and the split ------------------------------------------
    rx, ry, rw, rh = 1.28, 0.78, 5.56, 0.62
    arrow(ax, (xs[2] + bw / 2, by - 0.03), (rx + rw - 0.60, ry + rh + 0.03), col=BLU, rad=0.18)
    box(ax, rx, ry, rw, rh, fc="#eef4fb", ec=BLU, lw=1.9)
    ax.text(rx + 0.20, ry + rh / 2, r"$Y = 1\,[\, S + n \geq \gamma \,]$", fontsize=12.5,
            color=INK, ha="left", va="center", zorder=8)
    cap(ax, rx + 2.06, ry + rh / 2, "THE RECEIVER — a hit only when signal + noise clears\n"
                                    "the threshold. Frozen: we aim it, never improve it",
        7.2, GRY, ha="left", va="center")

    arrow(ax, (rx + 0.55, ry - 0.03), (0.95, 0.34), col=GRY, lw=2.0, rad=0.20)
    arrow(ax, (rx + rw - 0.55, ry - 0.03), (7.15, 0.34), col=RED, lw=2.4, rad=-0.20)
    cap(ax, 0.86, 0.30, "TRUTH → scoring only.\nThe agent never sees it.", 7.8, GRY, "bold")
    cap(ax, 7.24, 0.30, "HIT / NO-HIT → the only\nthing the scheduler is told.", 7.8, RED, "bold")

    fig.savefig(HERE / "flowB_world.png", dpi=240, facecolor="white")
    plt.close(fig)


# ============================== FLOW C — the policy ==========================
def flow_c():
    W, H = 7.60, 3.46
    fig = plt.figure(figsize=(W, H), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    cap(ax, 0.22, H - 0.08, "THE SCHEDULER — one decision, 600 times in 30 seconds",
        10.2, INK, "bold", ha="left")

    by, bh = 1.52, 1.62
    x1, w1 = 0.22, 2.94
    x2, w2 = 3.46, 1.66
    x3, w3 = 5.42, 1.96

    # 1 — what it sees
    box(ax, x1, by, w1, bh, ec=BLU)
    cap(ax, x1 + w1 / 2, by + bh - 0.09, "WHAT IT SEES", 8.6, BLU, "bold")
    a1 = inset(fig, [(x1 + 1.10) / W, (by + 0.50) / H, 1.70 / W, 0.82 / H])
    rows = OBS[:36 * 5].reshape(5, 36).astype(float)
    rows = np.clip(rows / np.maximum(np.percentile(rows, 92, axis=1, keepdims=True), 1e-9), 0, 1)
    a1.imshow(rows, cmap=HEAT, aspect="auto", vmin=0, vmax=1, interpolation="nearest")
    a1.set_xticks(np.arange(-0.5, 36, 1), minor=True)
    a1.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    a1.grid(which="minor", color="white", lw=0.5)
    a1.tick_params(which="minor", length=0)
    for i, n in enumerate(["paid off?", "airtime", "how stale", "we are here", "streak"]):
        yy = by + 1.32 - (i + 0.5) * 0.82 / 5
        ax.text(x1 + 1.04, yy, n, fontsize=6.9, color=INK, ha="right", va="center", zorder=8)
    cap(ax, x1 + 1.95, by + 0.44, "5 numbers × 36 bands + 3 = 183", 7.6, INK, "bold")
    cap(ax, x1 + 1.95, by + 0.24, "all from its own scan history — no ground truth", 7.2, GRY)

    # 2 — the policy
    box(ax, x2, by, w2, bh, fc="#eef4fb", ec=BLU)
    cap(ax, x2 + w2 / 2, by + bh - 0.09, "THE POLICY", 8.6, BLU, "bold")
    cap(ax, x2 + w2 / 2, by + bh - 0.36, "a recurrent network\nthat remembers what\nearlier sweeps\nlooked like, not just\nthe last one", 7.5, GRY)

    # 3 — the output
    box(ax, x3, by, w3, bh, ec=RED)
    cap(ax, x3 + w3 / 2, by + bh - 0.09, "A WEIGHT ON ALL 36", 8.6, RED, "bold")
    a2 = inset(fig, [(x3 + 0.16) / W, (by + 0.52) / H, 1.64 / W, 0.72 / H])
    top = int(PROBS.argmax())
    a2.bar(np.arange(36), PROBS, color=[RED if i == top else "#9aa4b2" for i in range(36)], width=0.88)
    a2.set_ylim(0, PROBS.max() * 1.2)
    cap(ax, x3 + w3 / 2, by + 0.44, "its favourite holds just 11.5%", 7.1, INK, "bold")
    cap(ax, x3 + w3 / 2, by + 0.25, "— it has not stopped searching", 6.9, GRY)

    arrow(ax, (x1 + w1 + 0.03, by + bh * 0.50), (x2 - 0.03, by + bh * 0.50), col=BLU)
    arrow(ax, (x2 + w2 + 0.03, by + bh * 0.50), (x3 - 0.03, by + bh * 0.50), col=RED)

    # optional priority input
    px, py, pwd, phd = 0.22, 0.46, 2.94, 0.68
    box(ax, px, py, pwd, phd, fc="white", ec=GRN, lw=1.6, ls="--")
    cap(ax, px + 0.14, py + phd - 0.08, "OPTIONAL — priority in", 7.6, GRN, "bold", ha="left")
    cap(ax, px + 0.14, py + phd - 0.26, "a threat library or an operator can mark bands as\n"
                                        "more important. Left alone, nothing changes.",
        7.2, GRY, ha="left")
    arrow(ax, (px + pwd + 0.03, py + phd / 2), (x2 + w2 * 0.35, by - 0.03), col=GRN,
          lw=1.7, ls="--", rad=-0.20)

    # the two things you can do with that weight
    oy, oh = 0.30, 0.52
    box(ax, x3 - 1.10, oy, 1.48, oh, fc="#f2fbf5", ec=GRN, lw=1.5)
    cap(ax, x3 - 0.36, oy + oh - 0.08, "✓ draw from it", 7.9, GRN, "bold")
    cap(ax, x3 - 0.36, oy + oh - 0.26, "visits 31 of 36 bands", 7.0, GRY)
    box(ax, x3 + 0.52, oy, 1.48, oh, fc="#fdf2f4", ec=RED, lw=1.5)
    cap(ax, x3 + 1.26, oy + oh - 0.08, "✕ take the top", 7.9, RED, "bold")
    cap(ax, x3 + 1.26, oy + oh - 0.26, "collapses onto one band", 7.0, GRY)
    arrow(ax, (x3 + w3 * 0.34, by - 0.03), (x3 - 0.36, oy + oh + 0.03), col=GRN, lw=1.8, rad=0.10)
    arrow(ax, (x3 + w3 * 0.74, by - 0.03), (x3 + 1.26, oy + oh + 0.03), col=RED, lw=1.8,
          ls="--", rad=-0.10)

    fig.savefig(HERE / "flowC_policy.png", dpi=240, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    flow_b(); flow_c()
    print("wrote flowB_world.png, flowC_policy.png")
