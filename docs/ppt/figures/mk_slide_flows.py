"""Slide flowcharts A / B / C, drawn from real data.

Every panel that could be real is real: the PDW cloud and the beam lobes are read
straight out of `config_2.h5`, the truth grid and the waterfall come from a
`ScanEnv` episode, the observation heatmap is one live 183-vector, and the band
distribution is the rung-17c checkpoint's actual softmax at that step.

    python docs/ppt/figures/mk_slide_flows.py

Writes flowA_pipeline.png, flowB_journey.png, flowC_decision.png beside this file.
"""
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE = Path("/private/tmp/claude-501/-Users-aamirhashmi-Code-projects-SIH26055"
             "/26974edd-04d8-4ba9-a85b-783a35335334/scratchpad")

RED, BLU, GRY, GRN, ORG = "#c0143c", "#2166ac", "#6b7280", "#2ca25f", "#e08214"
INK, FAINT = "#111827", "#d1d5db"
HEAT = LinearSegmentedColormap.from_list("heat", ["#f8fafc", "#bfdbfe", "#2166ac", "#0b2e4f"])

plt.rcParams.update({"font.family": "DejaVu Sans"})

# ----------------------------------------------------------------- real data --
R = np.load(CACHE / "real.npz")
P = np.load(CACHE / "pdw.npz")
S, Z, OBS, PROBS = R["S"], R["Z"], R["obs"], R["probs"]
RL_BAND, RL_Y, RR_BAND = R["rl_band"], R["rl_Y"], R["rr_band"]
PDW, LAB = P["d"], P["lab"]          # columns: ToA us, Freq MHz, PW us, AoA deg, Amp dBm


def box(ax, x, y, w, h, fc="white", ec=GRY, lw=1.6, r=0.018, z=2, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0.004,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z, **kw))


def arrow(ax, p0, p1, col=GRY, lw=2.2, ls="-", rad=0.0, ms=16, z=5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw,
                                 color=col, linestyle=ls, zorder=z, shrinkA=2, shrinkB=2,
                                 connectionstyle=f"arc3,rad={rad}"))


def cap(ax, x, y, s, fs=8.4, col=GRY, w="normal", ha="center", va="top", st="normal"):
    ax.text(x, y, s, fontsize=fs, color=col, ha=ha, va=va, weight=w, style=st,
            linespacing=1.32, zorder=8)


def inset(fig, rect):
    a = fig.add_axes(rect)
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(False)
    return a


# =============================================================== FLOW A =======
# Slide 2, bottom strip. Deliberately short so the Pareto chart keeps its half.
def flow_a():
    fig = plt.figure(figsize=(11.0, 4.62), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 11); ax.set_ylim(0, 4.62); ax.axis("off")

    # -- the spectrum we have to watch, with the receiver's window on it --------
    occ = Z.mean(axis=1)
    sx0, sw, sy, sh = 1.95, 8.70, 3.60, 0.50
    ax.add_patch(Rectangle((sx0, sy), sw, sh, fc="#f8fafc", ec=FAINT, lw=1.0, zorder=1))
    for b in range(36):
        x = sx0 + sw * (b + 0.5) / 36
        hgt = 0.07 + 0.40 * (occ[b] / max(occ.max(), 1e-9)) ** 0.6
        ax.plot([x, x], [sy + 0.04, sy + 0.04 + hgt], lw=3.0, color="#9aa4b2",
                solid_capstyle="round", zorder=2)
    bw = sw / 36
    win = sx0 + sw * 11.5 / 36
    ax.add_patch(Rectangle((win - bw / 2, sy - 0.06), bw, sh + 0.12,
                           fc="none", ec=RED, lw=2.4, zorder=6))
    ax.plot([win, win], [sy + sh + 0.06, sy + sh + 0.22], lw=1.4, color=RED, zorder=6)
    cap(ax, win, sy + sh + 0.26, "±500 MHz — all we can hear at any one instant",
        8.4, RED, "bold", va="bottom")
    cap(ax, sx0 - 0.06, sy + sh / 2, "0.25\nGHz", 7.6, GRY, ha="right", va="center")
    cap(ax, sx0 + sw + 0.06, sy + sh / 2, "18\nGHz", 7.6, GRY, ha="left", va="center")
    cap(ax, 0.30, sy + sh / 2, "THE SPECTRUM\nWE MUST WATCH", 9.6, INK, "bold",
        ha="left", va="center")
    cap(ax, 0.30, sy - 0.08, "36 bands · bar height = how\nbusy that band really was",
        7.9, GRY, ha="left")

    # -- rail 1: open loop ------------------------------------------------------
    y1 = 2.52
    cap(ax, 0.30, y1 + 0.34, "TODAY — open loop", 10.2, GRY, "bold", ha="left", va="center")
    cap(ax, 0.30, y1 + 0.10, "a band list fixed before\nthe mission ever starts", 7.9, GRY,
        ha="left", va="top")
    seq = [7, 8, 9, 10, 11]
    for i, b in enumerate(seq):
        x = 3.16 + i * 0.94
        box(ax, x, y1, 0.70, 0.62, fc="#f3f4f6", ec="#c9ced6", lw=1.4)
        ax.text(x + 0.35, y1 + 0.31, f"{b}", fontsize=12.5, color=GRY,
                ha="center", va="center", weight="bold", zorder=4)
        if i < len(seq) - 1:
            arrow(ax, (x + 0.72, y1 + 0.31), (x + 0.92, y1 + 0.31), col="#c9ced6", lw=1.9, ms=13)
    ax.text(8.30, y1 + 0.31, "✕", fontsize=19, color=RED, ha="center", va="center",
            weight="bold", zorder=8)
    cap(ax, 8.62, y1 + 0.31, "no arrow ever\ncomes back", 8.4, RED, "bold", ha="left", va="center")

    # -- rail 2: the closed loop, each node carrying a real picture -------------
    y0 = 0.88
    cap(ax, 0.30, y0 + 1.06, "NARADA — closed loop", 10.2, RED, "bold", ha="left", va="center")
    cap(ax, 0.30, y0 + 0.82, "the next band is chosen from\nwhat it just heard", 7.9, GRY,
        ha="left", va="top")
    xs = [3.16, 4.92, 6.68, 8.84]
    ws = [1.34, 1.34, 1.72, 1.86]
    hh = 1.12
    for x, w in zip(xs, ws):
        box(ax, x, y0, w, hh, fc="white", ec=RED, lw=2.0)
    for i in range(3):
        arrow(ax, (xs[i] + ws[i] + 0.01, y0 + hh / 2), (xs[i + 1] - 0.01, y0 + hh / 2),
              col=RED, lw=2.1, ms=15)

    labels = ["point at\none band", "hear it,\nor not", "update what it knows\nabout all 36 bands",
              "pick where the\npayoff is now"]
    for x, w, t in zip(xs, ws, labels):
        cap(ax, x + w / 2, y0 + hh - 0.06, t, 8.5, INK, "bold")

    # node 1 — the 36-band ladder with one band lit
    lx, lyb, lyt = xs[0] + 0.28, y0 + 0.08, y0 + 0.68
    for b in range(36):
        yy = lyt - (b + 0.5) * (lyt - lyb) / 36
        hot = (b == 24)
        ax.plot([lx, lx + (0.46 if hot else 0.28)], [yy, yy], lw=3.0 if hot else 0.85,
                color=RED if hot else "#dfe3e8", solid_capstyle="butt", zorder=7)
    ax.text(xs[0] + 0.80, y0 + 0.37, "band 24\nof 36", fontsize=8.8, color=RED, weight="bold",
            ha="left", va="center", zorder=8)

    # node 2 — one bit
    ax.text(xs[1] + 0.44, y0 + 0.48, "●", fontsize=23, color=RED, ha="center", va="center", zorder=8)
    ax.text(xs[1] + 0.92, y0 + 0.48, "○", fontsize=23, color="#c9ced6", ha="center", va="center", zorder=8)
    cap(ax, xs[1] + 0.67, y0 + 0.20, "one bit back", 8.2, GRY, "bold", va="center")

    # node 3 — the real remembered rows
    a = inset(fig, [(xs[2] + 0.13) / 11, (y0 + 0.28) / 4.70, 1.46 / 11, 0.42 / 4.62])
    blk = OBS[:36 * 3].reshape(3, 36)
    ref = np.percentile(blk, 92, axis=1, keepdims=True)
    a.imshow(np.clip(blk / np.maximum(ref, 1e-9), 0, 1), cmap=HEAT,
             aspect="auto", vmin=0, vmax=1, interpolation="nearest")
    a.set_xticks(np.arange(-0.5, 36, 1), minor=True)
    a.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    a.grid(which="minor", color="white", lw=0.5)
    a.tick_params(which="minor", length=0)
    cap(ax, xs[2] + 0.86, y0 + 0.22, "what paid off · airtime · how stale", 7.2, GRY)

    # node 4 — the real softmax
    a = inset(fig, [(xs[3] + 0.15) / 11, (y0 + 0.28) / 4.70, 1.56 / 11, 0.42 / 4.62])
    cols = [RED if i == int(PROBS.argmax()) else "#9aa4b2" for i in range(36)]
    a.bar(np.arange(36), PROBS, color=cols, width=0.9)
    a.set_ylim(0, PROBS.max() * 1.18)
    cap(ax, xs[3] + 0.93, y0 + 0.22, "a live chance for every band", 7.2, GRY)

    # the return arrow — below everything, the thickest line on the slide
    ax.add_patch(FancyArrowPatch((xs[3] + ws[3] / 2, y0 - 0.04), (xs[0] + ws[0] / 2, y0 - 0.04),
                                 arrowstyle="-|>", mutation_scale=20, lw=3.2, color=RED,
                                 connectionstyle="arc3,rad=-0.19", zorder=6))
    ax.text(6.55, 0.42, "50 ms later, again   —   600 of these decisions inside one 30-second look",
            fontsize=9.4, color=RED, weight="bold", ha="center", va="center", zorder=9,
            bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="none"))

    fig.savefig(HERE / "flowA_pipeline.png", dpi=220, facecolor="white")
    plt.close(fig)


# =============================================================== FLOW B =======
# Slide 3 left. The data's journey, with the real artefact drawn at every stage.
def flow_b():
    fig = plt.figure(figsize=(11.0, 6.50), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 11); ax.set_ylim(0, 6.50); ax.axis("off")
    H = 6.50

    cap(ax, 0.30, 6.36, "FROM REAL RECORDED RADAR  →  A WORLD THE SCHEDULER CAN BE TESTED IN",
        11.0, INK, "bold", ha="left", va="top")

    ytop, ph, pw = 4.02, 1.70, 2.36
    xs = [0.30, 2.94, 5.58, 8.22]
    titles = ["REAL RECORDINGS", "EVERY PULSE, AS RECORDED",
              "ONE EMITTER SWEEPING PAST", "WHO IS ON AIR, BAND BY BAND"]
    subs = ["92 scan/stare pairs · 2.2 GB\nradar the Turing Institute simulated",
            "644,674 pulses in one 30 s file\ntime · frequency · power",
            "its beam rotates — we only hear it\nwhen it sweeps across us",
            "36 bands × 600 slots of 50 ms\nthe ground truth of the world"]
    for x, t, s_ in zip(xs, titles, subs):
        box(ax, x, ytop, pw, ph, fc="white", ec=BLU, lw=1.7)
        cap(ax, x + pw / 2, ytop + ph - 0.07, t, 8.2, BLU, "bold")
        cap(ax, x + pw / 2, ytop - 0.09, s_, 7.9, GRY)
    for i in range(3):
        arrow(ax, (xs[i] + pw + 0.02, ytop + ph / 2), (xs[i + 1] - 0.02, ytop + ph / 2),
              col=BLU, lw=2.2)

    # panel 1 — a stack of files
    for k in range(4):
        ax.add_patch(FancyBboxPatch((xs[0] + 0.74 + k * 0.11, ytop + 0.40 + k * 0.078), 0.74, 0.80,
                                    boxstyle="round,pad=0.004,rounding_size=0.03",
                                    fc="white", ec=BLU, lw=1.3, zorder=3 + k))
    ax.text(xs[0] + 1.24, ytop + 0.82, ".h5", fontsize=10.5, color=BLU, weight="bold",
            ha="center", va="center", zorder=9)

    # panel 2 — the REAL pulse cloud
    a2 = inset(fig, [(xs[1] + 0.15) / 11, (ytop + 0.34) / H, 2.06 / 11, 0.96 / H])
    idx = np.random.default_rng(0).choice(len(PDW), 11000, replace=False)
    a2.scatter(PDW[idx, 0] / 1e6, PDW[idx, 1] / 1000, c=PDW[idx, 4], s=1.2,
               cmap="magma", vmin=-140, vmax=-40, linewidths=0)
    a2.set_facecolor("#0b1220")
    cap(ax, xs[1] + pw / 2, ytop + 0.28, "time →   frequency ↑   colour = loudness", 7.0, GRY)

    # panel 3 — the REAL beam lobes: peak power per 0.1 s
    a3 = inset(fig, [(xs[2] + 0.15) / 11, (ytop + 0.34) / H, 2.06 / 11, 0.96 / H])
    swing, who = -1e9, 0
    for lb in np.unique(LAB):
        m = LAB == lb
        if m.sum() < 500:
            continue
        v = PDW[m, 4]
        d = np.percentile(v, 99) - np.percentile(v, 20)
        if d > swing:
            swing, who = d, lb
    m = LAB == who
    t_us, amp = PDW[m, 0], PDW[m, 4]
    nb = 300
    b_i = np.clip((t_us / 30e6 * nb).astype(int), 0, nb - 1)
    env = np.full(nb, -np.inf)
    np.maximum.at(env, b_i, amp)
    env[np.isneginf(env)] = np.nan
    a3.plot(np.linspace(0, 30, nb), env, lw=1.5, color=ORG)
    a3.set_facecolor("#fffaf3")
    a3.set_ylim(np.nanmin(env) - 3, np.nanmax(env) + 3)
    cap(ax, xs[2] + pw / 2, ytop + 0.28, "loudest pulse in every 0.1 s", 7.0, GRY)

    # panel 4 — the REAL truth grid
    a4 = inset(fig, [(xs[3] + 0.15) / 11, (ytop + 0.34) / H, 2.06 / 11, 0.96 / H])
    crop = Z[:, 200:290]
    a4.imshow(crop, aspect="auto", cmap="Greys", vmin=0, vmax=1.25, interpolation="nearest")
    a4.set_xticks(np.arange(-0.5, crop.shape[1], 1), minor=True)
    a4.set_yticks(np.arange(-0.5, 36, 1), minor=True)
    a4.grid(which="minor", color="white", lw=0.45)
    a4.tick_params(which="minor", length=0)
    cap(ax, xs[3] + pw / 2, ytop + 0.28, "one cell = one band, one 50 ms slot", 7.0, GRY)

    # -- the receiver, and the fork -------------------------------------------
    rx, ry, rw, rh = 2.86, 2.06, 5.28, 1.34
    arrow(ax, (xs[3] + pw / 2, ytop - 0.44), (rx + rw + 0.02, ry + rh * 0.72), col=BLU, lw=2.2, rad=0.16)
    box(ax, rx, ry, rw, rh, fc="#eef4fb", ec=BLU, lw=2.0)
    cap(ax, rx + 0.22, ry + rh - 0.10, "OUR RECEIVER MODEL — the one line everything else hangs on",
        8.4, BLU, "bold", ha="left")
    ax.text(rx + 0.24, ry + 0.62, r"$Y \;=\; 1\,[\, S + n \;\geq\; \gamma \,]$",
            fontsize=15, color=INK, ha="left", va="center", zorder=8)
    cap(ax, rx + 0.24, ry + 0.36, "a hit is called when signal + noise clears the threshold.\n"
                                  "frozen hardware — the scheduler aims it, never improves it",
        7.6, GRY, ha="left")
    a5 = inset(fig, [(rx + 3.70) / 11, (ry + 0.40) / H, 1.34 / 11, 0.60 / H])
    from math import erf
    xg = np.linspace(-10, 10, 200)
    a5.plot(xg, 0.5 * (1 + np.array([erf(v / (3 * 2 ** .5)) for v in xg])), lw=2.2, color=BLU)
    a5.axvline(0, lw=1.2, color=RED, ls="--")
    a5.set_facecolor("white")
    for sp in a5.spines.values():
        sp.set_visible(True); sp.set_color(FAINT)
    cap(ax, rx + 4.37, ry + 0.34, "chance we hear it, against\nhow loud it is", 6.9, GRY)

    # the fork
    arrow(ax, (rx + 0.90, ry - 0.02), (1.58, 1.78), col=GRY, lw=2.2, rad=0.14)
    arrow(ax, (rx + rw - 0.90, ry - 0.02), (9.42, 1.78), col=RED, lw=2.6, rad=-0.14)
    cap(ax, 1.42, 1.72, "the truth — kept only for scoring,\nNEVER shown to the agent", 8.2, GRY, "bold")
    cap(ax, 9.58, 1.72, "hit / no-hit — the only thing\nthe scheduler is ever told", 8.2, RED, "bold")

    # -- the waterfall ---------------------------------------------------------
    wy, wh = 0.34, 0.92
    box(ax, 0.30, wy, 10.40, wh, fc="white", ec=INK, lw=1.4)
    a6 = inset(fig, [0.36 / 11, (wy + 0.05) / H, 10.28 / 11, (wh - 0.10) / H])
    a6.imshow(Z, aspect="auto", cmap="Greys", vmin=0, vmax=1.7, interpolation="nearest",
              extent=[0, 600, 35.5, -0.5])
    a6.plot(np.arange(len(RL_BAND)), RL_BAND, lw=0.7, color=RED, alpha=0.80)
    a6.set_xlim(0, 600); a6.set_ylim(35.5, -0.5)
    cap(ax, 5.50, wy - 0.04, "grey = a band genuinely transmitting   ·   red = where we actually "
                             "pointed, across one whole 30-second episode", 8.2, INK, "bold")

    fig.savefig(HERE / "flowB_journey.png", dpi=220, facecolor="white")
    plt.close(fig)


# =============================================================== FLOW C =======
# Slide 3 right. One decision, opened up.
def flow_c():
    H = 5.90
    fig = plt.figure(figsize=(9.6, H), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 9.6); ax.set_ylim(0, H); ax.axis("off")

    cap(ax, 0.28, H - 0.10, "ONE DECISION, OPENED UP   —   this happens 600 times in 30 seconds",
        10.6, INK, "bold", ha="left", va="top")

    # -- what it remembers: the real 36 x 5 observation ------------------------
    ox, oy, ow, oh = 0.28, 3.06, 6.90, 2.42
    box(ax, ox, oy, ow, oh, fc="white", ec=BLU, lw=1.8)
    cap(ax, ox + 0.16, oy + oh - 0.09, "WHAT IT REMEMBERS  —  five numbers for every one of the 36 bands",
        9.0, BLU, "bold", ha="left")

    rows = OBS[:36 * 5].reshape(5, 36).astype(float).copy()
    ref = np.percentile(rows, 92, axis=1, keepdims=True)
    rows = np.clip(rows / np.maximum(ref, 1e-9), 0, 1)
    hm_x, hm_y, hm_w, hm_h = ox + 2.50, oy + 0.80, 4.20, 1.30
    a1 = inset(fig, [hm_x / 9.6, hm_y / H, hm_w / 9.6, hm_h / H])
    a1.imshow(rows, cmap=HEAT, aspect="auto", vmin=0, vmax=1, interpolation="nearest")
    a1.set_xticks(np.arange(-0.5, 36, 1), minor=True)
    a1.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    a1.grid(which="minor", color="white", lw=0.7)
    a1.tick_params(which="minor", length=0)
    names = ["how often this band paid off", "how much airtime it has had",
             "how long since we last looked", "which band we are on right now",
             "hits in a row here"]
    for i, n in enumerate(names):
        yy = hm_y + hm_h - (i + 0.5) * hm_h / 5
        ax.text(hm_x - 0.10, yy, n, fontsize=8.0, color=INK, ha="right", va="center", zorder=8)
    ax.annotate("", xy=(hm_x + hm_w, hm_y - 0.13), xytext=(hm_x, hm_y - 0.13),
                arrowprops=dict(arrowstyle="->", color=GRY, lw=1.1))
    cap(ax, hm_x, hm_y - 0.16, "band 0", 7.4, GRY, ha="left")
    cap(ax, hm_x + hm_w, hm_y - 0.16, "band 35", 7.4, GRY, ha="right")

    for k, s_ in enumerate(["the clock", "last level heard", "current streak"]):
        bx = ox + 0.30 + k * 2.16
        box(ax, bx, oy + 0.14, 1.96, 0.34, fc="#f8fafc", ec=FAINT, lw=1.1)
        cap(ax, bx + 0.98, oy + 0.31, s_, 7.6, GRY, va="center")
    cap(ax, ox + 0.30, oy + 0.62, "and three more, for the whole receiver:", 7.6, GRY, ha="left")

    cap(ax, ox + ow + 0.14, oy + oh - 0.55, "36 × 5\n+ 3\n= 183\nnumbers", 10.0, BLU, "bold",
        ha="left", va="top")
    cap(ax, ox + ow + 0.14, oy + 0.72, "every one of\nthem built from\nits own scan\nhistory — no\nground truth",
        7.8, GRY, ha="left", va="top")

    # -- memory ---------------------------------------------------------------
    mx, my, mw, mh = 1.05, 1.42, 2.66, 1.12
    arrow(ax, (ox + 2.20, oy - 0.02), (mx + mw / 2, my + mh + 0.02), col=BLU, lw=2.2)
    box(ax, mx, my, mw, mh, fc="#eef4fb", ec=BLU, lw=2.0)
    cap(ax, mx + mw / 2, my + mh - 0.12, "RECURRENT NETWORK", 9.0, BLU, "bold")
    cap(ax, mx + mw / 2, my + mh - 0.40, "it carries what it saw in\nearlier sweeps, not only\nwhat it heard just now",
        7.9, GRY)
    ax.add_patch(FancyArrowPatch((mx - 0.03, my + 0.34), (mx - 0.03, my + 0.72),
                                 arrowstyle="-|>", mutation_scale=12, lw=1.8, color=BLU,
                                 connectionstyle="arc3,rad=-1.35", zorder=7))
    cap(ax, mx - 0.40, my + 0.53, "memory", 7.4, BLU, "bold", ha="right", va="center")

    # -- the operator / threat priority input ---------------------------------
    px, py, pw_, ph_ = 0.28, 0.26, 2.50, 0.84
    box(ax, px, py, pw_, ph_, fc="white", ec=GRN, lw=1.8, ls="--")
    cap(ax, px + pw_ / 2, py + ph_ - 0.10, "PRIORITY IN  (optional)", 8.4, GRN, "bold")
    cap(ax, px + pw_ / 2, py + ph_ - 0.34, "an operator or the threat library\nsays which bands matter more.\n"
                                           "left alone, it changes nothing", 7.5, GRY)
    arrow(ax, (px + pw_ / 2, py + ph_ + 0.02), (mx + 0.70, my - 0.02), col=GRN, lw=1.9, ls="--", rad=-0.18)

    # -- the choice -----------------------------------------------------------
    cx, cy, cw, ch = 4.30, 1.42, 5.02, 1.62
    arrow(ax, (mx + mw + 0.02, my + mh / 2), (cx - 0.02, cy + ch / 2), col=BLU, lw=2.2)
    box(ax, cx, cy, cw, ch, fc="white", ec=RED, lw=1.9)
    cap(ax, cx + 0.16, cy + ch - 0.09, "HOW SURE IT IS ABOUT EACH BAND", 9.0, RED, "bold", ha="left")
    a2 = inset(fig, [(cx + 0.24) / 9.6, (cy + 0.42) / H, 4.54 / 9.6, 0.92 / H])
    top = int(PROBS.argmax())
    a2.bar(np.arange(36), PROBS, color=[RED if i == top else "#9aa4b2" for i in range(36)], width=0.88)
    a2.set_ylim(0, PROBS.max() * 1.2)
    cap(ax, cx + cw / 2, cy + 0.36, "even its favourite band holds only 11.5% of the weight — it stays curious",
        7.7, GRY)

    # -- sample vs argmax ------------------------------------------------------
    sy, sh_ = 0.26, 0.84
    box(ax, cx, sy, 2.42, sh_, fc="#f2fbf5", ec=GRN, lw=1.7)
    cap(ax, cx + 1.21, sy + sh_ - 0.10, "✓  draw a band from it", 8.6, GRN, "bold")
    cap(ax, cx + 1.21, sy + sh_ - 0.34, "visits 31 of the 36 bands\nand keeps its coverage", 7.6, GRY)
    box(ax, cx + 2.60, sy, 2.42, sh_, fc="#fdf2f4", ec=RED, lw=1.7)
    cap(ax, cx + 3.81, sy + sh_ - 0.10, "✕  always take the top one", 8.6, RED, "bold")
    cap(ax, cx + 3.81, sy + sh_ - 0.34, "collapses onto a single band\n— we measured this happening", 7.6, GRY)
    arrow(ax, (cx + 1.21, cy - 0.02), (cx + 1.21, sy + sh_ + 0.02), col=GRN, lw=2.0)
    arrow(ax, (cx + 3.81, cy - 0.02), (cx + 3.81, sy + sh_ + 0.02), col=RED, lw=2.0, ls="--")

    fig.savefig(HERE / "flowC_decision.png", dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    flow_a(); flow_b(); flow_c()
    print("wrote flowA_pipeline.png, flowB_journey.png, flowC_decision.png")
