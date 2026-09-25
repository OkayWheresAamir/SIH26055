"""PPT figure 3 — the waterfall. What the receiver actually did with its 30 s.
Three panels, because three panels tell the whole story: the open-loop floor
sweeps blindly, the camper parks, the agent hunts."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rfenv.env import ScanEnv
from rfenv.scenario import Scenario
from rfenv import baselines
from rfenv.constants import SLOT_S, N_SLOTS, N_BANDS

CFG = "config_921"
PANELS = [
    ("round_robin", "Round-robin — the open-loop floor\nsweeps on a fixed cycle, ignores what it hears"),
    ("camper",      "Camper — optimises hit-rate alone\nparks on the busiest band, finds 38 of 72"),
    ("lstm_balance_d67_control_seed2_300k_400k",
                    "NARADA — the learned scheduler\nrevisits what pays, still covers the spectrum"),
]

sc = Scenario.replay(CFG, "stare")
fig, axes = plt.subplots(3, 1, figsize=(10.0, 7.6), sharex=True)
for ax, (key, title) in zip(axes, PANELS):
    env = ScanEnv(scenario=sc)
    obs, info = env.reset(seed=0)
    pol = baselines.make(key, seed=0)
    tuned = np.full(N_SLOTS, -1, dtype=int)
    hits = []
    done = False
    while not done:
        a = int(pol(obs, info))
        s0 = env.t
        obs, r, done, trunc, info = env.step(a)
        for s in range(s0, min(env.t, N_SLOTS)):
            tuned[s] = a
        for j, y in enumerate(info["dwell"]["Y"] if isinstance(info.get("dwell"), dict) else []):
            if y: hits.append((s0 + j, a))
        done = done or trunc

    Z = env.grid.Z
    ax.imshow(Z, aspect="auto", origin="lower", cmap="Greys", alpha=.30,
              extent=[0, 30, -0.5, N_BANDS - 0.5], interpolation="nearest")
    t = np.arange(N_SLOTS) * SLOT_S
    ok = tuned >= 0
    ax.plot(t[ok], tuned[ok], color="#c0143c", lw=0.9, alpha=.95, solid_joinstyle="round")
    ax.set_ylabel("band", fontsize=10)
    ax.set_ylim(-0.5, N_BANDS - 0.5); ax.set_xlim(0, 30)
    ax.set_yticks([0, 12, 24, 35])
    ax.text(0.008, 0.955, title, transform=ax.transAxes, va="top", ha="left",
            fontsize=10.8, weight="bold", linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc", alpha=.93))
    for s in ("top", "right"): ax.spines[s].set_visible(False)

axes[-1].set_xlabel("Time into the 30-second episode  (s)", fontsize=11.5)
fig.suptitle("Where the receiver pointed, against where the emitters actually were\n"
             "grey = a band genuinely transmitting   ·   red = the band we tuned",
             fontsize=13, weight="bold", y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.945])
fig.savefig("/private/tmp/claude-501/-Users-aamirhashmi-Code-projects-SIH26055/90581c6c-2a45-4658-9929-b806de01bddc/scratchpad/fig/fig3_waterfall.png", dpi=190, facecolor="white")
print("ok")
