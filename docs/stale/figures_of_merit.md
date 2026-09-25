# Figures of merit — the seven PS metrics

`python -m rfenv.compare --figures-of-merit` — 2026-09-19T22:28:02Z. 47 scenarios × 3 seed(s) = 1269 episodes, train split only, stare replays and sampled scenarios only (D36). Reward `reward_balance` throughout.

The problem statement asks for *"figures of merit for interception performance such as probability of detection, probability of false alarm, sensitivity, average intercept rate, average reward/cost function, percentage of correct predictions, and average intercept time error."* All seven are below, one column each.

**Five of the seven are policy-independent** and read the same down their column by design — that is the correct outcome, not a bug to fix (§3, D21, D70):

- **P_d, P_fa, sensitivity** are receiver properties at the frozen operating point (§3, D25, D33). No scheduler and no reward can move them — penalising a false alarm prices a wasted dwell, it does not improve the detector.
- **% correct predictions** and **average intercept-time error** are *model-level* metrics (§2): they measure whether the stare-built environment reproduces Turing's own sweep — the PS's own text assigns *prediction* to the system model. They characterise the environment, not any scheduler, so they too are constant across rungs (D70).

**Only average intercept rate and average reward vary by scheduler.** Every mean is printed over its interquartile range because scenario difficulty spans 2 to 99 emitters (EVALUATION.md §7); the two exact analytic constants (P_fa, sensitivity) have no distribution to show.

| rung | scheduler | P_d | P_fa | sensitivity (dB) | avg intercept rate (/s) | avg reward | % correct predictions | avg intercept-time error (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | `random` | 0.8395 <br><sub>[0.7804, 0.8780]</sub> | 1.350e-03 | -107.16 | 1.070 <br><sub>[0.333, 1.733]</sub> | 220.32 <br><sub>[170.59, 283.15]</sub> | 0.8585 <br><sub>MCC 0.6854; [0.8347, 0.8835]</sub> | 8.42 <br><sub>[2.15, 12.90]</sub> |
| 2 | `round_robin` | 0.8395 <br><sub>[0.7804, 0.8780]</sub> | 1.350e-03 | -107.16 | 1.060 <br><sub>[0.333, 1.667]</sub> | 220.00 <br><sub>[153.20, 280.76]</sub> | 0.8585 <br><sub>MCC 0.6854; [0.8347, 0.8835]</sub> | 8.42 <br><sub>[2.15, 12.90]</sub> |
| 3 | `turing_sweep` | 0.8395 <br><sub>[0.7804, 0.8780]</sub> | 1.350e-03 | -107.16 | 1.069 <br><sub>[0.300, 1.800]</sub> | 230.41 <br><sub>[166.48, 288.64]</sub> | 0.8585 <br><sub>MCC 0.6854; [0.8347, 0.8835]</sub> | 8.42 <br><sub>[2.15, 12.90]</sub> |
| 4 | `camper` | 0.8395 <br><sub>[0.7804, 0.8780]</sub> | 1.350e-03 | -107.16 | 0.621 <br><sub>[0.200, 1.033]</sub> | -343.73 <br><sub>[-406.35, -273.57]</sub> | 0.8585 <br><sub>MCC 0.6854; [0.8347, 0.8835]</sub> | 8.42 <br><sub>[2.15, 12.90]</sub> |
| 5 | `recency` | 0.8395 <br><sub>[0.7804, 0.8780]</sub> | 1.350e-03 | -107.16 | 1.109 <br><sub>[0.333, 1.767]</sub> | 289.98 <br><sub>[235.76, 360.73]</sub> | 0.8585 <br><sub>MCC 0.6854; [0.8347, 0.8835]</sub> | 8.42 <br><sub>[2.15, 12.90]</sub> |
| 6a | `apfeld_active_rfs` | 0.8395 <br><sub>[0.7804, 0.8780]</sub> | 1.350e-03 | -107.16 | 1.090 <br><sub>[0.333, 1.767]</sub> | 292.19 <br><sub>[266.22, 359.10]</sub> | 0.8585 <br><sub>MCC 0.6854; [0.8347, 0.8835]</sub> | 8.42 <br><sub>[2.15, 12.90]</sub> |
| 6 | `apfeld` | 0.8395 <br><sub>[0.7804, 0.8780]</sub> | 1.350e-03 | -107.16 | 0.324 <br><sub>[0.167, 0.433]</sub> | -732.62 <br><sub>[-1069.21, -423.58]</sub> | 0.8585 <br><sub>MCC 0.6854; [0.8347, 0.8835]</sub> | 8.42 <br><sub>[2.15, 12.90]</sub> |
| — | `camper_oracle` *(ref. line)* | 0.8395 <br><sub>[0.7804, 0.8780]</sub> | 1.350e-03 | -107.16 | 0.292 <br><sub>[0.067, 0.533]</sub> | -1356.89 <br><sub>[-1402.83, -1230.09]</sub> | 0.8585 <br><sub>MCC 0.6854; [0.8347, 0.8835]</sub> | 8.42 <br><sub>[2.15, 12.90]</sub> |
| — | `oracle_pulse` *(ref. line)* | 0.8395 <br><sub>[0.7804, 0.8780]</sub> | 1.350e-03 | -107.16 | 0.750 <br><sub>[0.300, 1.067]</sub> | -254.55 <br><sub>[-380.33, 37.25]</sub> | 0.8585 <br><sub>MCC 0.6854; [0.8347, 0.8835]</sub> | 8.42 <br><sub>[2.15, 12.90]</sub> |

Per-scheduler cells are `mean` over the interquartile range across episodes. The five constant columns are identical by design; a **reference line** reads the truth grid and is not a scheduler (§5).

## Which grids each P_d figure is over (D33)

**P_d = 0.8395** here, over the `reference_sweep` cells of *these 47 stare comparison grids* (10,738 occupied). D33 froze the *population rule*, not the set of grids, so this is not interchangeable with:

- **0.85058** — the `runs/validation` figure, over the 47 **scan-replay** grids `validate.py` builds (11,710 occupied). Same rule, different world.
- the P_d over the *full* comparison set when sampled scenarios are included (D46 measured 0.8421 over 57 stare-and-sampled grids). This report's P_d is over the replay grids **only**, because #6 and #7 are undefined on a sampled scenario (no underlying scan recording).

Quote whichever figure you name the grids for, and never one without them (D33, §7).

## Frozen and policy-independent — stated so nobody "fixes" it (§3, D70)

P_d, P_fa and sensitivity are detector properties at the frozen γ and σ; % correct predictions and average intercept-time error are model-faithfulness properties of the stare-built environment. **None of the five moves with the policy.** They read near-identical across `round_robin`, `recency`, `camper` and every `lstm_*` rung, and that is correct — it must not be debugged. Only average intercept rate and average reward are scheduler results.

## Average reward / cost (D7)

Scored under reward `reward_balance` — the selected reward (D47/D68). Average reward is *comparable only within a reward family and never used to rank across different rewards* (D7); it is a per-family scale, not a cross-reward score.

## Average intercept-time error — how it is computed, and its limits (§2, D70)

Per emitter, matched by transmitter label: `|predicted − measured|`, where *predicted* is the stare-built environment's first-intercept slot under Turing's own sweep (D28, censored at 30 s) and *measured* is the emitter's first appearance in the actual scan recording (D36). Mean **8.42 s** over **1,530** emitters detectable in stare *and* present in scan (dropped: 174 predicted-only, 209 scan-only), seed 0.

Two limitations, stated rather than patched — the same two gate 1 carries:

- **scan and stare are independent runs** (D24), so the same emitter has disjoint activity in each. Part of this error is that realisation divergence, not a modelling defect — but an out-of-sample prediction (stare → scan) is exposed to exactly it, so it belongs in the number. This is why the figure is seconds, not the sub-slot agreement of gate 1's per-dwell view.
- **band 0 (250 MHz) is invisible to stare** (D10): 59.0% occupied in the recordings, 0.0% predicted, so its emitters are predicted as never-intercepted and inflate the error where scan caught them.

% correct predictions carries the same band-0 limitation, and is reported with MCC beside it because §7 forbids a bare accuracy on sparse occupancy.

## Provenance

- Written 2026-09-19T22:28:02Z (UTC).
- Scheduler columns: 47 scenarios (47 stare replays, 0 sampled), seeds [0, 1, 2], split `train`, reward `reward_balance`.
- Model-level columns (#6, #7) and P_d: over the 47 stare replay configs, seed 0.
- Operating point: γ = -111.0 dB, σ = 3.0 dB, P_fa = 1.350e-03, sensitivity -107.16 dB, population `reference_sweep` (10,738 occupied cells).
- Every scheduler row is scored by `metrics.scheduler_metrics()` from the artefacts under `runs\baselines`, never from live environment state.
- Scan replays are refused for the scheduler comparison by `compare._check_comparison_scenario` (D36); the model-level columns use scan recordings deliberately, which is gates 1–2's domain, not a comparison scenario.
- P_d, % correct predictions and average intercept-time error re-derive from `receiver.operating_point`, `validate.gate1` and `validate.intercept_time_error` — no figure is transcribed.
