"""PPT figure 2 — emitters found against time, 5 lines, on one real scenario."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rfenv.env import ScanEnv
from rfenv.scenario import Scenario
from rfenv import baselines
from rfenv.constants import SLOT_S, N_SLOTS

CFG = "config_921"
RUNGS = [
    ("lstm_balance_d67_control_seed2_300k_400k", "NARADA — Recurrent PPO", "#c0143c", 3.0, "-"),
    ("recency",           "Recency heuristic (our bar)", "#2166ac", 2.1, "-"),
    ("round_robin",       "Round-robin (open-loop floor)", "#7f8c8d", 2.0, "--"),
    ("apfeld",            "Apfeld adaptive (literature)", "#d95f02", 2.0, "-"),
    ("camper",            "Camper (max-hit-rate trap)",   "#9467bd", 2.0, "-"),
]

sc = Scenario.replay(CFG, "stare")
fig, ax = plt.subplots(figsize=(9.0, 5.2))
for key, lab, col, lw, ls in RUNGS:
    env = ScanEnv(scenario=sc)
    obs, info = env.reset(seed=0)
    pol = baselines.make(key, seed=0, grid=env.grid if key.endswith("oracle") else None)
    first = {}
    done = False
    while not done:
        a = int(pol(obs, info))
        obs, r, done, trunc, info = env.step(a)
        for e in info["newly_intercepted"]:
            first.setdefault(e, env.t)
        done = done or trunc
    n_det = len(env.detectable)
    t = np.arange(N_SLOTS + 1) * SLOT_S
    cnt = np.zeros(N_SLOTS + 1)
    for slot in first.values():
        cnt[min(int(slot), N_SLOTS):] += 1
    ax.plot(t, cnt, color=col, lw=lw, ls=ls, label=lab, zorder=5 if "NARADA" in lab else 3)
    print(f"{lab:34s} found {int(cnt[-1]):3d} / {n_det}")

ax.axhline(n_det, color="#999", ls=":", lw=1.4)
ax.text(29.6, n_det + 1.0, f"all {n_det} detectable emitters", ha="right",
        fontsize=9.5, color="#777")
ax.set_xlabel("Time into the 30-second episode  (s)", fontsize=11.5)
ax.set_ylabel("Distinct emitters found", fontsize=11.5)
ax.set_title("One scenario, five schedulers: who finds the emitters, and how fast\n"
             f"(Turing replay {CFG}, {n_det} detectable emitters)",
             fontsize=13, weight="bold", pad=12)
ax.set_xlim(0, 30); ax.set_ylim(0, n_det + 6)
ax.grid(alpha=.22, linestyle=":")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(loc="lower right", fontsize=10.2, frameon=True, framealpha=.95,
          edgecolor="#ddd")
fig.tight_layout()
fig.savefig("/private/tmp/claude-501/-Users-aamirhashmi-Code-projects-SIH26055/90581c6c-2a45-4658-9929-b806de01bddc/scratchpad/fig/fig2_discovery.png", dpi=200, facecolor="white")
print("ok")
