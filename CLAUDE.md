# CLAUDE.md

## What this repository is

A fresh start on SIH26055. It holds the source material, the data, the decisions taken from
them, and the RF environment being built on top.

A previous attempt accumulated derived documents and measured claims that became difficult to
separate from their sources. This repository exists to avoid that. The rules below are the
whole point of it — follow them before doing anything else.

**Build status (2026-09-04).** `rfenv/` has L0 (`scenario.py`), L1 (`truth.py`), L2
(`receiver.py`), L3 (`env.py`), the artefact layer (`metrics.py`, `render.py`), `validate.py`
and now the baseline ladder (`baselines/`, `compare.py`) all built. **`ENVIRONMENT_SPEC.md`
§Build order is complete.**

**Suite status (2026-09-11, on a machine with no training stack and no checkpoints): 261 passed,
290 skipped, 0 failed.** The skip count is high by design — 57 rungs are registered and every
RL-backed one skips without `stable_baselines3` or its checkpoint, because the training stack is
deliberately optional (`requirements.txt`). It must stay that way: a test module that imports the
training stack at *module* level aborts collection of the **whole** suite on such a machine, which
has now happened twice. **Run `python scripts/doctor.py` before saying any chunk is finished** — it
checks the six things that have actually gone wrong here (unpushed commits, uncommitted decisions,
duplicate decision numbers, module-level training imports, rungs without checkpoints, and the
observation width drifting out of the documents) and exits non-zero if a pull would not give
someone what you think you handed them.

**The observation gained a second, opt-in layout on 2026-09-14 (D30 resolved as D71).**
`rfenv/scenario.py` now reads PulseWidth and AoA, the two PDW columns it always silently discarded;
`ScanEnv(obs_version="v1"|"v2")` picks between the original 183-wide vector (default, every
existing checkpoint) and a new 326-wide one that adds them plus a per-band amplitude upgrade. This
is the first observation change that did not invalidate every prior checkpoint (D49, D55, D67 all
did) — "v2" is additive, not a replacement, and `rfenv.rl.common.known_observation_widths()` now
returns both. `_observation_blocks()` builds every block as a named dict; `OBS_LAYOUTS` selects and
orders the subset that ships, so `observation_space` and the flat vector can never disagree. AoA is
stored as `(sin θ, cos θ)`, not a raw angle, and gated on a declared hit the same way `measured_dbm`
already is — what ships is one raw last-bearing reading per band, not the bearing-*clustering*
feature D30's own measurement showed is AoA's real value (still not built). Full account: D71.
**Checkpoints were reorganised the same session** — `runs/checkpoints/` was 150 files flat,
unnavigable; it is now `runs/checkpoints/v1/<model_name>/` (`git mv`, history-preserving), with
`v2/` as the sibling for anything trained on the new layout. A RecurrentPPO run mirroring rung 17c's
own config (`reward_balance`, seed 2, 400k steps) started training on "v2" the same session; no
result yet.

**"v2" widened again the same day, to 362, on a direct request (D72).** `pulse_count` is a sixth
"v2"-only block: the illumination count `C` (`truth.py`) at the loudest slot of a band's last
declared hit, gated on `Y` the same way PulseWidth/AoA are, `log1p`-normalised against the same
reference (`_DENSITY_REF_PULSES = 64`) `reward_balance_improved` already prices `C` by on the
reward side. This reopens D29/D34's own exclusion of pulse count from the observation — a question
already measured and declined once this session by `PDW_COMPLETENESS_AND_BAND_DENSITY_BRIEF.md` on
mechanical grounds: `C` is built with no gamma gate at all, so it counts contributions that would
never individually cross the threshold, and literal `C` has no path to existing outside the
simulator's own bookkeeping. `Y`-gating narrows that gap — only cells the receiver actually declared
a hit on ever write a value — but does not close it: a hit's `pulse_count` can still include
sub-threshold co-located emitters a real per-pulse-resolving receiver would not have logged. Unlike
D71, this one was not free: the three "v2" snapshots D71's own retrain had already produced
(`lstm_balance_v2_control_seed2_s1/s2/s3`, 326-wide, 385,024/400,000 steps) are now permanently
unloadable, the same cost every past observation change has carried. `known_observation_widths()`
now returns `{183, 362}`. Full account: D72.

**The first result on the widened "v2" observation landed 2026-09-18 (D73, `MEASURED`, single
seed).** Two RecurrentPPO checkpoints, both on the 362-wide "v2" layout, same split/hyperparameters/
seed, only the reward differing: `reward_balance` (rung 20d) scored 0.111 interception ratio / 3.30 s
censored intercept time / 36.3% paired-both-vs-recency; `reward_balance_improved_v2` (rung 21a) scored
0.136 / 2.82 s / 45.6% — ahead on every column, the same direction D65 found on "v1". **Not read as
settled** — one training seed per arm, the same caveat D65 gave and never resolved, and not directly
comparable to the "v1" D64/D65 numbers since observation width, reward formula and the sampled-scenario
draw all differ at once. Neither row is promoted as the number to carry forward. Full account: D73,
`EVALUATION.md` §5.

**A band-priority reward was built, trained, and measured null this week (D74, 2026-09-18 to
2026-09-19).** A fully-specified, library-free request: a fourth observation layout, "v2p"
(398-wide = "v2" plus one more 36-wide block, `band_priority`), and an additive reward term in
`ScanEnv.step()` — `reward += priority_coef * band_priority[dwell.band] * len(newly)`, gated on
discovering something new, not on occupying the band (D53's camping exploit is exactly what per-slot
gating would reopen). This is a synthetic, no-library successor to D70 (which sourced its priority
vector from an external threat classification and had its own implementation deleted, uncommitted,
unexplained, before this session) — `band_priority` here is sampled per episode from the env's own
RNG, not read from any emitter-type table. It is **not** a new `REWARDS` entry and does not touch
`reward_gate.py`'s D62 screen.

Two RecurrentPPO checkpoints, matched pair, both complete: treatment
(`lstm_balance_v2p_priority_seed2`, real per-episode elevated bands, rung 22a) and control
(`lstm_balance_v2p_uniform_seed2`, `priority_uniform=True` — same reward-term scale, no real signal,
rung 22b). **The comparison** (round_robin/recency + each rung, 513 episodes apiece): treatment
**0.120** ratio / **3.28 s** cTTI / **0.891** coverage / **44.4%** beats-recency-both; control
**0.128** / **2.70 s** / **0.908** / **61.4%** — **the control beat the treatment on every column**.
Three follow-up checks explain why, in order: camping ruled out directly (only ~7% more airtime on
elevated bands, inconsistent in direction — nowhere near overcommitment); treatment measurably more
diffuse than control across the *whole* spectrum, not just priority bands (entropy 3.255 vs 3.051,
24.95 vs 18.95 of 36 bands touched — plausibly because control's constant input made for an easier,
more stationary training problem than treatment's genuinely varying one); and, decisively, a
**permutation ablation** — the same trained treatment checkpoint run on identical episodes with
`band_priority` either real or shuffled across bands — found its airtime correlates with true
priority equally poorly either way (+0.018 real vs +0.019 shuffled, real beats shuffled in only
13/30 episodes) and performance does not degrade when the signal is corrupted. **The agent never
learned to use `band_priority` at all** — treatment's underperformance is a training-difficulty
story, not a misused-signal story.

`compare.py` gained matching `--band-priority`/`--priority-uniform` flags, resolved *per rung*
rather than only from the CLI: `Rung` (`ladder.py`) carries its own `band_priority`/
`priority_uniform`/... fields, `resolve_priority_kwargs` lets a registered rung's own training
config override the caller's flags, and rung 22a/22b can sit in the same `compare.py` invocation
instead of needing two. `--figures` also writes `priority_animation_config_*.gif` automatically
whenever a compared rung is priority-trained (real signal, not the uniform control) — the episode's
elevated band(s) highlighted directly on the animated schedule, no extra flag needed.

**Not adopted, not promoted, code not removed.** Nothing defaults to `band_priority=True`, so
nothing else is affected by leaving the mechanism registered. Unlike D66's phase-switch gate (a
specific mechanism shown actively worse than its baseline, removed after being measured), this is a
null result at one configuration (`priority_coef=0.5`, 3-6/36 bands, default LSTM capacity, 400k
steps, one seed) with named, untried paths (bigger coefficient, more capacity, more steps) — not
scoped or recommended here, a new experiment if picked up again. Full account: D74; mechanism:
`OBSERVATION_SPACE.md` §2.4; numbers: `MODEL_COMPARISON.md`'s Width 398 section;
`scratch/TRAINING_JOURNEY.md` §18.

**The named paths were tried, the same day, and the answer didn't change (D74 follow-up,
2026-09-19).** A new `priority_reward_bonus` function (`env.py`, still not a `REWARDS` entry) adds a
second, decaying per-slot occupancy term on top of the original discovery bonus — requested directly
despite the D53 camping-exploit risk, and kept clear of it by anchoring its decay to cumulative
`visit_density` (D55's own fix) rather than a resettable streak, verified with a dedicated
ping-pong-vs-sweep test. `priority_coef` 0.5→2.0, the elevated value itself now configurable
(`priority_high`, was a hardcoded constant) raised 3.0→5.0, LSTM hidden size doubled 256→512,
timesteps doubled 400k→800k. Result: treatment (rung 23a) **0.154 / 2.43 s / 0.907 / 73.1%**
beats-recency-both — the highest figure measured anywhere in this project, ahead of D68's settled
best (17c, 54.4%) — against control (rung 23b) **0.121 / 2.74 s / 0.911 / 45.6%**, the opposite
direction from the original pair. **A second permutation ablation says this is not the mechanism
working**: the strengthened checkpoint's airtime correlates with true priority identically whether
fed real or shuffled values (+0.031 both ways, real beats shuffled 12/30 — a coin flip). 23a's lead
is read as single-seed training variance, not a real effect — the direction flipped between the two
pairs while the ablation answer ("no, it doesn't read the signal") stayed the same both times. One
open thread this does surface, separate from the priority question: a 512-wide, 800k-step checkpoint
outperforms every smaller/shorter one measured so far on both arms — whether that holds with the
priority mechanism removed entirely is untested. Verdict unchanged: **not adopted, not promoted,
code not removed.** Full account: D74 (same entry, extended); numbers: `MODEL_COMPARISON.md`'s
Width 398 section; `scratch/TRAINING_JOURNEY.md` §18.

**The agent was made *online*, in three pieces, on 2026-09-19 (D75, D76, D77 — all `BUILT`, none
measured).** A direct request: make the recurrent policy an online learner, let missions run as long
as you like, and make the environment watchable while it runs. Nothing here is a result yet; the
matched pair and the ablation that would make one are pre-registered in D75 and not run.

**D75 — a fourth layout, "v3" (399-wide = "v2p" + `prev_reward` 1).** This is the RL² construction: a
recurrent policy shown what its last action returned can run an adaptation rule inside the episode,
in its hidden state, with no gradient step. **It shipped with three in-context blocks and now has
one, in two same-day amendments (2026-09-20).** First, `prev_action` — asked directly why it was
needed at all, given `step()` already assigns `_current_band = action` and `_prev_action = action`
from the same value, and an LSTM's hidden state can only carry forward what appeared in its *input*
— `current_band` has always been in that input since "v1", so `prev_action` was never adding a
channel the network lacked, only duplicating one it already had; removed, narrowing `OBS_LAYOUTS["v3"]`
436→400, the same class of width change D49/D55/D67/D72 made — `lstm_v3_seed0`/`lstm_v3_seed1` (both
complete 800k-step checkpoints) are now permanently unloadable, paid on purpose. Second, `prev_hit`
— it is `current_hit_streak > 0`, also pre-existing information, with no companion block making it
provably redundant the same direct way, but removed anyway on a direct request to isolate
`prev_reward`'s own effect before the layout's first training run — narrowing again, 400→399, at no
further checkpoint cost (nothing was ever trained on the 400-wide shape). **`prev_reward` is now the
only block "v3" carries beyond "v2p", and remains the only genuinely new information in the
layout** — no layout before "v3" ever exposed the raw per-step reward, only running aggregate
statistics (`hit_rate`, `hit_streak`, `staleness`) — and an ablation corrupting it is the
pre-registered test of whether this agent actually reads it, alongside a `hit_rate` positive control
(if corrupting a block the policy has leaned on since D34 changes nothing either, the ablation is
measuring nothing — the lesson D74 paid for twice).
The reward the policy *sees* is **not** the reward it is trained on: `reward_balance_obs` is
`reward_balance` with `dwell.Y` for `dwell.Z`, because the agent already holds `hit_rate`,
`visit_density`, `staleness` and `n_slots` and could otherwise solve the remaining term for
`0.5*Z.sum()` — learning about emitters it never detected. The two differ by exactly the receiver's
own sensitivity (**Pd = 0.8421**, Pfa = 1.35e-3), which is why `Y` and `Z` are not the same thing
even when you are staring straight at the band. **D29 is not reopened**: the reward still reads
truth, still trains and scores every arm; this is asymmetric actor-critic with the halves kept apart,
and it is what leaves a "v3" rung `deployable=True`.

**D76 — `episode_slots`, so an episode can outrun a recording.** `constants.py` is untouched (D42's
freeze holds, `N_SLOTS` is still 600); this is a per-env length defaulting to it, and the golden
digests in `tests/test_backward_compat_hashes.py` — taken at `d4f4361`, before any of this landed —
assert every existing path is bit-identical. Longer worlds are **stitched** from independent 30 s
draws (`TruthGrid.stitch`), never tiled, because tiling one recording would hand the agent an exactly
periodic world to memorise. Emitter identity is per segment on purpose (merging it would make a
segment-0 emitter unfindable for the rest of the mission), with the honest cost that **there is no
temporal continuity across a seam**. Two blocks would have left their declared Box at length and now
cannot: `clock` divides by this episode's length, and `staleness` is clipped at both sites that write
it — the observation's and the reward's, which must agree (D52) — a clip that provably never binds at
600 (13.930 against a 13.953 ceiling) and would otherwise read 1,674 at an hour. **`episode_metrics()`
carried a latent correctness bug at length**, not a rescale: it censored a missed emitter at the full
episode, so an hour-long mission would charge a segment-0 miss 3,600 s; it now censors at the
emitter's own segment end, which evaluates to exactly 600 at the default. Measured cost: ~78 MB per
simulated hour, built in under a second. Also `rfenv/live.py` — 36 band rows of colour painted
straight to the terminal, no matplotlib, no display, works over SSH, with the full matplotlib panel
in `rfenv/render/live.py` (separate because importing `rfenv.render` forces the Agg backend). Not a
tty means **zero escape bytes**, and a missing plotting backend degrades to the terminal rather than
killing a one-hour run.

**D77 — online fine-tuning (`rfenv/rl/online.py`), a new mode alongside offline training, not instead
of it.** Every rung in `EVALUATION.md` §5, including both of D77's own arms, is still produced by the
ordinary `rfenv.rl.{ppo,recurrent_ppo,dqn}.train()` path — frozen checkpoint, offline, unchanged and
still the default; `rfenv/rl/online.py` only ever loads a checkpoint that path already produced and
keeps adapting it. Verified directly (2026-09-20): plain offline `train()` calls, including on the
narrowed 399-wide "v3", still produce ordinary loadable checkpoints with no code path change. D29 recorded
that *"the restriction would only bite if we ever fine-tuned online, which we do not"*; we now do, and
**the rule is satisfied rather than relaxed** — `ObservableRewardWrapper` feeds the gradient
`reward_balance_obs`, so the agent optimises a quantity a real receiver could compute, while
`total_reward` and `episode_metrics()` still score on the truth-fed reward and an adapted checkpoint
is judged on the same yardstick as every other rung. **Per-mission updates are not offered, and the
arithmetic is the reason**: a 30 s episode is 300-600 decisions against `n_steps=8192`, so one mission
cannot fill a fourteenth of a rollout and its advantages would be dominated by the terminal bootstrap.
Fine-tuning runs only on a continuous grid, where the training rollout fits several times over. It
**stays at `n_steps=8192`** rather than shrinking to buy more updates: `RecurrentPPO` carries the
LSTM state across rollout boundaries but not the gradient, so `n_steps` is the BPTT window, and
shortening it truncates the long-horizon credit assignment an in-context agent exists to learn.
Learning rate, clip range and `target_kl` are the only deliberate departures, and all three are
brakes. Ctrl-C saves a manifested checkpoint
recording the interruption. What gets measured, when it is run, is **coverage per 30 s segment against
segment index**, streamed to `segments.jsonl` as it happens. Full accounts: D75, D76, D77;
mechanism: `OBSERVATION_SPACE.md` §2.5.

**A second recurrent-PPO policy exists, opt-in, and its first result favours the baseline (D78,
2026-09-20).** Rung 9's policy has always been `MlpLstmPolicy` — since the observation is already
flat, its `FlattenExtractor` is a no-op and the architecture every existing checkpoint trained on is
`obs -> LSTM -> actor/critic`. `rfenv/rl/policies.py` adds `MlpFeatureLstmPolicy`, selectable via
`--policy`, not a replacement: `obs -> 2-layer LayerNorm MLP (obs_dim -> 256 -> 256, Tanh) -> LSTM ->
actor/critic`. Only `features_extractor_class`/`features_extractor_kwargs` change from the library
default; `lstm_hidden_size` and the post-LSTM actor/critic heads are untouched, and every
recurrent-PPO guarantee (episode-start masking, hidden-state reset, truncated-BPTT sequence handling,
the rollout buffer) is `sb3_contrib`'s own code, never touched, because the extractor only ever sees a
flat `(batch, obs_dim)` tensor — one live timestep or a whole flattened rollout, transformed one row
at a time, never the sequence dimension. Passed to `RecurrentPPO` as a class object rather than
registered in its `policy_aliases` (a `ClassVar` dict shared process-wide, not this repo's to
mutate); a checkpoint trained this way round-trips through the ordinary `load_checkpoint()` with no
special casing, verified directly. 9 tests (`test_policies.py`).

**The first matched pair (single seed, rung 25a `MlpFeatureLstmPolicy` vs rung 24b's already-trained
`MlpLstmPolicy` control, identical otherwise — reward_balance, "v2p", seed 0, 512-wide LSTM, 800k
steps) has the baseline ahead, decisively for one seed.** Paired against recency (THE BAR): control
wins both-metrics on 62.4% of episodes, MLP-feature policy on 46.8% — a 15.6 pp gap, three times
D47/D68's 5 pp no-selection margin. Interception ratio is essentially tied; the separation is almost
entirely censored intercept time (MLP-feature 3.12 s vs control 2.25 s mean) — the deeper network is
measurably slower to first-detect, not worse at eventually covering the spectrum. Read as suggestive,
not conclusive (one seed, no mechanism check yet, same caution as D64/D73/D74's first pairs); not
adopted, not promoted, code not removed.

**A convergence check says this looks like a plateau, not an unfinished run** — training reward flattened
by ~82k of 800k steps, and evaluating the run's own checkpoint-freq snapshots (200k/400k/600k/800k) on 12
scenarios found no clean upward trend late in training. The 600k snapshot scored marginally best on that
small probe, registered as rung 25b and checked properly on the full 47-config/3-seed set against rung
23a (D74's own strongest checkpoint, 73.1% beats-recency-both) — **which corrected the small-sample
finding rather than confirming it**: 25b scored 36.9% there, clearly worse than the final 800k checkpoint
(25a)'s 46.8%, not better. 23a stays the strongest checkpoint measured in this project by a wide margin.
Full account: D78.

**A third recurrent-PPO policy exists, opt-in, unmeasured (D79, 2026-09-20).** `BandEncoderLstmPolicy`
(`rfenv/rl/policies.py`): `obs -> shared per-band encoder -> mean pool across bands -> concat encoded
global features -> LSTM -> actor/critic`, testing whether making the observation's per-band structure
explicit (36 repeats of an 11-feature description, in "v3") helps, versus handing the network one
undifferentiated vector the way the baseline and D78 both do. The hard part is the gather, not the
network: the flat observation interleaves *blocks* (`hit_rate[0:36]`, `visit_density[0:36]`, ...), not
*bands*, so turning it into `(36, n_band_features)` is a strided, non-contiguous read, not a reshape —
`rfenv/env.py`'s new `band_layout(version)` builds the gather indices once from `_BLOCK_SPECS`'s own
declared widths (per-band exactly when a block is `N_BANDS` wide, global otherwise — nothing hardcoded
by name), verified against synthetic values that encode their own source block, not just checked by
shape. One `band_encoder` (not 36) is applied via `nn.Linear`/`nn.LayerNorm`'s ordinary last-dimension
broadcasting, checked directly (parameter count independent of band count; permuting which band holds
which feature vector before pooling leaves the output unchanged). Mean pooling only, no attention, on
purpose — isolating whether structured encoding helps before touching how the bands are combined.
`obs_version` auto-detects from `observation_space`'s own width, refusing rather than silently
gathering wrong columns on a mismatch. `lstm_hidden_size` and the actor/critic heads are untouched,
same convention D78 set. 21 tests (`test_band_layout.py`, no training stack needed) + 18
(`test_policies.py`, extended). No training result yet — the natural next step is rung 23a's exact
command (reward_balance, "v2p", its own strengthened band-priority settings, seed 2, 512-wide LSTM,
800k steps), differing only in `--policy`, not started without being asked. Full account: D79.

**The four validation gates ran for the first time on 2026-09-04** (`python -m rfenv.validate`,
47 train configs, seed 0, artefacts in `runs/validation/`): **gates 2, 3 and 4 PASS; gate 1 is
MEASURED** — D37 fixed its convention and deliberately left its threshold undecided. Every
criterion was written into `rfenv/validate.py::GATES` *before* the run and is asserted against
D39 by a test, because a threshold chosen once the measurement is visible is not a gate. Numbers
in `EVALUATION.md` §6.

**The baseline ladder ran for the first time on 2026-09-04** (`python -m rfenv.compare --seeds 3
--sampled 10 --figures`): seven schedulers and two truth-reading reference lines over 47 stare
replays + 10 sampled scenarios × 3 seeds = **1,539 episodes**, artefacts in `runs/baselines/`.
Table and figures in `EVALUATION.md` §5; recorded as **D46**. Three decisions came out of it.
**D43** fixed D36's duplicate: rung 2 is now round-robin with **equal airtime per band** (2 slots
per 72-slot cycle) against Turing's 2:1 weighting, and the two rungs now measure different things
— the sweep's weighting is worth +33% interception ratio. **D44** — Apfeld's Algorithm 1
contradicts its own prose; we implement the prose, and every Apfeld number is *our adaptation* of
the no-tracking variant to a binary-detection receiver. **D45** — Apfeld's own "Active RFs"
ablation joins as rung 6a, and the period-estimation half measurably *hurts* on 30 s binary
episodes.

**The bar for RL is rung 5, not round-robin.** A one-line index policy (`argmax(hit rate + gap in
sweeps)`) Pareto-dominates the floor on **70.2%** of episodes; round-robin is beaten by almost
everything.

**RL is on the ladder. Its first result was withdrawn, and the two reasons are both fixed
(2026-09-09).** Rungs 7 (DQN), 8 (PPO) and 9 (Recurrent PPO) are built and registered (**D48**).
The 2,223-episode run in `EVALUATION.md` §5 showed rung 9 with rung 4's camping profile — high
interception ratio, five times the intercept time, a third of the coverage, 0.0–1.8% on the paired
`both` column. **That is now known to be an artefact of two defects, not a finding about the
agent**, and both rows and conclusion are marked superseded in place:

- **D52 — the reward was inverted.** `reward_balance`'s exploration term read a staleness that was
  the *inverse* of the observation's (*when* a band was last seen, not *how long ago*), so the one
  term meant to pull the agent toward neglected bands paid most for revisiting the band it had just
  left. Measured: +0.419 for the band just left against −0.0025 for one untouched for 500 slots.
- **D53 — the camping penalty had a free workaround.** `-1.0 * camp_slots` resets whenever the
  action changes, so a 2-band ping-pong paid exactly what a full sweep paid. Measured over 3 seeds,
  that term ranked the ping-pong (coverage 0.261) **above** round-robin (0.921). It is now charged
  against airtime share: `-3.0 * visit_density[action] * n_slots`.
- **D54 — inference took the argmax of a policy that had not collapsed.** Mean entropy 2.369
  against `ln 36 = 3.584`, modal band holding 0.206 of the mass, argmax on one band for 580 of 586
  steps. Sampling the same checkpoint visits 31 of 36 bands and triples coverage. Both adapters now
  default to sampling; rung 7 (DQN) is the deliberate exception, and torch's generator is seeded
  per rung so seeded runs stay reproducible.

A third change followed from looking at the observation itself. **D55 — two of the three per-band
blocks lived in the bottom tenth of their declared range.** `visit_density` sums to 1 across bands
by construction, pinning its mean at 1/36 = 0.0278 for every scheduler that will ever run;
`staleness` was bimodal, measured mean 0.070 with everything unvisited piled on the 1.0 ceiling.
The decisive evidence was that **rung 5 already corrected one of them by hand** — `recency.py`
multiplied staleness back out by `N_SLOTS / SWEEP_SLOTS`, and its docstring records that skipping
that step collapses the rung into rung 4. That correction now lives in `_observation()` where every
policy gets it. `visit_density` reads in fair shares (1.0 = equal airtime, ceiling 36.0),
`staleness` in reference sweeps (1.0 = one pass overdue, ceiling 13.95), **the box is deliberately
no longer the unit interval**, and `camp_time` is gone — measured, it took exactly two values under
any non-camping policy. **The vector was 146 wide** (extended to 183 by D67, below). Rung 5's
ranking is unchanged (verified on 1,408 of 1,408 steps) and `reward_balance` is numerically
unchanged (verified to 1.6e-7), so D53's table survives.

**Every checkpoint in `runs/checkpoints/` is now dead**, rungs 7 and 8 included — they predated
D49 and D55 finished the job. **No RL row currently in `EVALUATION.md` §5 was produced under
conditions that can answer whether an agent games this ladder.** The training record is
`scratch/TRAINING_JOURNEY.md` §9.

**The reward set is now an axis, and D29's cap of three is lifted (D57).** `greedy` (exploit
corner: declarations plus remembered hit rate, no explore term, no camping cost) and `explore`
(explore corner: staleness minus airtime concentration plus new `(emitter, band)` discoveries) are
the two ends of D14's tension and are *expected* to fail their opposite check — measured, `greedy`
ranks camping above sweeping and `explore` ranks round-robin above rung 5. `weighted` is the single
knob between them at `alpha = 0.3`, chosen from a measured sanity window (**[0.1, 0.4]** once
re-measured against the real rungs, D57's correction below) rather than tuned against a score.

**Corrected 2026-09-10: `weighted` is not the reward to build on.** The original claim here — that
it uniquely separates rung 5 from round-robin — was measured against a hand-written stand-in for
`round_robin` (see D56, directly below) and does not survive being re-measured against the real
rungs. Under D62's screen, `weighted` puts the camper only **0.4σ** below the sweeps, short of the
1.0σ bar, and **fails**. `reward_balance` is the only one of six registered candidates that passes
the screen. `DEFAULT_REWARD` is unchanged at `reward_balance`; `weighted` remains registered as the
axis's midpoint but is not currently a candidate D47 may consider.

**D56 was measured wrong and is withdrawn (2026-09-10).** It reported that `reward_balance`
separates rung 5 from round-robin by only +2.3 ± 11.7 — because it scored a hand-written
`step % N_BANDS` sweep instead of rung 2, which is `EQUAL_AIRTIME_CYCLE` (D43). Against the real
rung the separation is **+59.0 ± 19.4 with rung 5 ahead on 8/8 seeds** — three times the seed
noise, not three percent of it. D53 and D57 used the same stand-in and are corrected in place;
their conclusions survive, only their `round_robin` rows move. **A measurement that substitutes an
obvious-looking reimplementation for a registered rung is not measuring the ladder**, and a test
now enforces it.

**Every reward candidate is now screened before anything trains on it (D62).** Score rungs 2, 4, 5
and 6a under each candidate, 8 seeds, paired per seed: a candidate passes only if it puts rung 5
clearly above rung 2 relative to seed noise and rung 4 clearly below both. No agent needed, minutes
on a CPU. **`hit_z` and `hit_y`, D29's original two, both failed** — they rank the camper above
every sweeping policy, D14's tension in reward form — **and were removed from `REWARDS` entirely
as a consequence (D63).** `REWARDS` now holds four keys: `reward_balance` (the only one that
passes D62), `greedy`, `explore`, `weighted`. Every DQN and PPO run in this repository trained on
one of the two removed candidates; they were already permanently unloadable from D49's observation
change, so nothing currently loadable is lost. `reward_balance_improved` (added 2026-09-10, below)
also passes D62. **D47 ran on 2026-09-10 (D68) and was resolved on 2026-09-11 with matched seeds.**
The first run, one seed per arm, landed 1.2 pp apart — inside D47's 5 pp no-selection margin, so
D47 escalated rather than picking. Two more seeds per arm (4 more runs, strictly sequential) fed a
fuller D61 selection: the treatment arm's pick was unchanged, but the control arm's moved to a
materially stronger checkpoint (net dominance +36.1% against the single-seed pick's +25.0%).
Re-applying D47 on that pair: `reward_balance` **81.9%** paired-both against round-robin,
`reward_balance_improved` **64.3%** (unchanged, same checkpoint as before) — a 17.6 pp gap, decisively
outside the margin. **`reward_balance` is selected.** `reward_balance_improved` stays registered
and passing D62, but is no longer carried forward as a co-equal candidate.

**The train/evaluation leak is closed (D60).** Training sampled `EmitterPool.from_train()` — all 47
development configs — while evaluation ran those same 47 replays plus scenarios sampled from that
same pool. The baselines do not train, so the asymmetry ran one way, ours. The 47 are now split
**35 training / 12 validation** by a rule written before anyone looked at which configs landed
where (`rfenv/split.py`), and the pool is rebuilt from the training half: **2,600 contributions /
1,431 emitters against 843 / 482, with zero shared**. Splitting the config list alone would not
have done it — the pool is assembled *from* the configs.

**Checkpoint selection is pre-registered (D61).** Highest `P(dominates rung 5) − P(dominated by
rung 5)` on the validation half, ties toward fewer steps, fixed in `rfenv/selection.py` before the
run that uses it.

**Corrected 2026-09-10: a clean RL result now exists, headline included, in `EVALUATION.md` §5
(D64, D65).** The 2,223-episode acceptance run above (`runs/acceptance_2026-09-10/`, rung 9c
48.0%/13.5%, rung 9b 43.9%/4.7%) is still contaminated for the reason given — every checkpoint in
it trained on the emitters it was scored against — and stays out of `EVALUATION.md` §5 for that
reason. But a retrain has since run: `reward_balance`, 400k steps, started at commit `176e6a9`,
after both D60 and D62/D63 were in the tree. D61's selection rule, run for the first time on real
candidates rather than stale pre-D60 checkpoints, picked the 400k snapshot at **net dominance
+36.1%** (47.2% dominates rung 5 / 11.1% dominated) on the 12 validation configs — a checkpoint
this repository can defend as clean. Its headline, paired against recency over the full development
set (684 episodes): **73.7% ratio / 31.6% cTTI / 22.8% both**. A parallel run finished the same
day: same split, hyperparameters and seed, only the reward differs (`reward_balance_improved`). Its
own D61-selected checkpoint (200k, weaker than the control on validation at +25.0% net dominance)
scores **83.0% / 39.8% / 31.0%** on the same headline run — ahead of the control there, a reversal
from the validation ranking. **Read as suggestive, not conclusive** — one training seed per arm,
and the control's own four checkpoints swing by more than the gap between the two arms. **Neither
row is promoted as the number to carry forward** — that decision has not been made, and both are
reported side by side in `EVALUATION.md` §5 rather than one superseding the other. Ledger:
`docs/project/ITERATION_LEDGER.md`. Full account: D64 (control), D65 (the paired comparison and the
reversal), `scratch/TRAINING_JOURNEY.md` §14.

**A hard explore/exploit gate was tried and removed the same day (D66).** Two new rungs — a
threshold-based commit/release gate, and the same gate wrapped around the D61-selected checkpoint's
own action choices instead of a sweep — both measured *worse* than the camper baseline on censored
intercept time and coverage. The code was removed after being measured; `D66` stays as the record
and `docs/project/PHASE_SWITCH_FUTURE_WORK.md` carries the two directions (an observation feature,
or a jointly-learned phase/band action) still worth trying if this line of work is picked up again.

**Option A was built the same day: the observation is now 183 wide, not 146 (D67).** Two blocks
appended after `measured_dbm` — `HIT_STREAK` (36-wide, consecutive declared hits per band across
visits, capped and rescaled) and `CURRENT_HIT_STREAK` (the current band's own streak, a
convenience scalar). Existing slice offsets are untouched, so no heuristic rung needed changing.
**Every checkpoint that predates this commit is now permanently unloadable** — D64's, D65's, every
snapshot of both arms — same cost every past observation change has carried (D49, D55), paid again
here on purpose rather than by accident. A gap this exposed is closed alongside it: the test
suite's per-rung checkpoint-usability check was a hand-maintained table that had fallen behind for
every rung added after D55, so this change's first test run produced 85 failures instead of clean
skips; rewritten to build each rung directly and catch the error `require_loadable()` (D49) already
raises, removing the table and the maintenance burden with it. A fresh retrain under 183 is
in progress — training one model at a time, sequentially, after a laptop crash interrupted the
first attempt at running several in parallel. Full account: D67, `scratch/TRAINING_JOURNEY.md`
§15.

**The environment was frozen on 2026-09-04 (D42).** `rfenv/constants.py` is closed — band
geometry, slot clock, dwell schedule, `N₀`, `σ`, `γ`, `PD_POPULATION` — and
`tests/test_freeze.py` enforces it with per-value literals plus a digest tripwire. Until that
file existed the suite read every constant symbolically and would have stayed green through a
change to γ or the band geometry. **Environment and pre-RL work is closed**; it reopens only on
evidence of an actual bug, and if anything on the list moves the environment is re-validated from
gate 1 and every baseline re-run (D25). Not frozen, deliberately: reward candidates and
observation extensions (D29, D30, D34) and the per-episode draw.

**D42 also records what the gates cannot detect** — read it before quoting a gate figure. Gate 2's
0.000 pp is algebraically forced and passes with a wrong band half-width; gate 3's reference is
co-parameterised with the environment; **no gate covers the ±500 MHz half-width**, whose sole
evidence is D3's 99.9851% in-band measurement (re-run 2026-09-04). Gate 1 is the only one whose two
sides use different data. No Turing *performance* result was reproduced because none exists — the
TSRD paper is a deinterleaving benchmark; what was reproduced is Turing's receiver *configuration*,
exactly, asserted against every file by `scenario.load_receiver`.

Two decisions came out of that first run. **D40** — Köksal's `P₁₂(T)` assumes successive receiver
periods are independent, which is false for a deterministic periodic pair, so gate 3 reports it
and never gates on it. **D41** — the recorded non-empty dwell rate is **35.403%**, not the
withdrawn 35.700%; that 0.3 pp "slot quantisation residual" was a band-blind comparison, and
under one convention the pipeline round-trips exactly. D41 is the fourth instance of the same
failure mode, caught on `validate.py`'s first run — which is the machinery working.

**A reference line is not a competitor.** `oracle_pulse` is a ceiling for interception ratio
**only** — measured, it loses censored intercept time to plain round-robin on 80.7% of episodes.
And `camper_oracle` (D14's truth-fed camper, ratio 0.568) is not reachable by any deployable
scheduler: the observation-fed camper gets 0.209, because illumination density is truth-side and
binary declarations are a poor proxy for it (D46).

**A scan replay is not a scheduler-comparison scenario (D36, 2026-09-04).** A scan recording
holds only the pulses Turing's own sweeping receiver was tuned to, so a grid built from one
hands any sweeping scheduler its answer — measured, interception ratio 0.9999 and censored
intercept time 0.00 s. Compare schedulers on **stare replays and sampled scenarios**; scan
replays stay in gates 1 and 2, where that imprint is the mechanism under test.

The three questions L2 was blocked on are all answered: **D28** (what counts as intercepting an
emitter), **D29** (what the reward may read), **D31** (a dwell's reward is the sum of its slots').

**Third consistency audit run 2026-09-03** (`DECISIONS.md` §Consistency audit — 2026-09-03).
The decision spine D1–D31 holds with no contradictions. It found and fixed one implementation
mismatch (**D32** — the sampler could draw the same physical emitter twice) and withdrew two
figures whose measurement convention was never recorded: **`Pd = 0.822`** and **gate 1's
86.19%/71.14%**. Both are relabelled, not silently corrected. **`receiver.py` needs D33 decided**
(which cell population `Pd` averages over) before it can emit a ROC; **D34** ratifies the base
observation vector `env.py` builds to. Neither blocks starting.

## What is authoritative here

| Source | Authority |
|---|---|
| `docs/project/SIH26055_PROBLEM_STATEMENT.md` | **The requirement.** The official DRDO problem statement — what we are being asked to build. Settles scope disputes. |
| `data/turing/**/*.h5` | **Highest for facts about the data.** When anything disagrees with an observed field, the files win. |
| `docs/project/PROJECT_ARCHITECTURE.md` | The working architecture, written by the human team. Default direction; not immutable. |
| `docs/protocol/CLAUDE_CODE_RESEARCH_PROTOCOL.md` | How the reference documents may and may not be used. |
| `docs/project/ENVIRONMENT_SPEC.md` | The consolidated buildable spec for the RF environment. Follows from `DECISIONS.md`; read before building. |
| `docs/project/EVALUATION.md` | The single authority on metrics, baselines, validation gates and protocol. Do not redefine a metric anywhere else. |
| `docs/reference/dataset/TSRD_dataset_paper_arXiv_2602.03856.pdf` | How the data was generated. Primary source on dataset semantics; the HF dataset card is only a summary of it. |
| Everything under `docs/reference/` | Reference material. Classify before use, per the protocol. See `docs/project/RESEARCH_MAP.md`. |
| `rfenv/constants.py` | **The freeze list, as a file.** Band geometry, slot clock, `N₀`, `σ`, `γ`. Frozen once the gates pass; no result may move it (D25). |
| `runs/baselines/comparison.md` | The scheduler comparison as run. Gitignored and rebuilt by `python -m rfenv.compare`; the figures it quotes are ratified in `EVALUATION.md` §5. |

`docs/` is organised by authority: `project/` is authored and governs the build, `protocol/` is
how we work, `reference/` is external material, `teammate-work/` is cross-check only. See
`docs/README.md`.

`rfenv/` is the environment, one module per layer of `ENVIRONMENT_SPEC.md`. Its module
docstrings carry the reasoning; the decisions themselves live in `DECISIONS.md`.

**Search `docs/` before re-reading it.** 366 pages of PDF are invisible to `grep`;
`docsearch` makes them searchable and returns a `file.pdf:p.7` citation with every hit, which
is the "Relevant section/page" the protocol asks for:

```bash
.venv/bin/python -m docsearch "alert confirm dwell time" -k 8
.venv/bin/python -m docsearch "how was the dataset generated" --primary   # skip our summaries
```

It answers `no strong match` when the corpus cannot support a claim, and lists the eight
image-only PDFs it cannot read (`--blind-spots`). See `docsearch/README.md`.

**When to search and when to open the file.** Search when you do not already know which
document holds the answer, and whenever the answer is likely in a PDF — those are unreachable
any other way. Read the file directly when you know which one it is and it is markdown;
`DECISIONS.md`, `ENVIRONMENT_SPEC.md` and `EVALUATION.md` are authoritative and greppable, so
go straight to them. **Either way, open the primary at the cited page before relying on a
number or an equation** — search locates a page, it does not read mathematics off it
(`T_rcv` extracts as `rcv \n T`). Prefer `--primary` when the question is what an external
source claims: our own records in `RESEARCH_MAP.md` and `DECISIONS.md` outrank the papers they
summarise in roughly a fifth of such queries.

Read `docs/project/PROJECT_ARCHITECTURE.md` and `docs/protocol/CLAUDE_CODE_RESEARCH_PROTOCOL.md` before
designing anything.

**`docs/project/DECISIONS.md` records every decision taken, with its evidence and status.** Read it
before proposing anything — a question marked `SETTLED` or `CLOSED` there does not get reopened
without new evidence, and one marked `PROPOSED` is waiting on a human, not on more research.
Add to it whenever a decision is made; that file is where project knowledge survives a session.

## Personal working preferences

Each of us keeps a gitignored `CLAUDE.local.md` at the repo root with our own working
preferences — how we want to be worked with, what we already know, what needs explaining.
Claude Code reads it automatically alongside this file. It is personal and never committed;
write your own rather than editing someone else's.

**Nothing in a `CLAUDE.local.md` overrides this file or the research protocol.** It covers
working style only — never provenance, authority, or the architecture gates below.

## Provenance rules

These are not style preferences. They are why this repository was recreated.

1. **Every factual claim must be traceable to a source you can name** — a specific HDF5 field,
   a specific page of a specific PDF, or a command whose output is in the transcript.
2. **State how you know.** "Verified by opening `config_2.h5`" and "as summarised in a
   document I have not checked against the primary" are different claims. Say which one you
   are making, every time.
3. **A measurement you did not run this session is not a fact.** Re-run it, or label it
   unverified. Never repeat a number because it appears in a document.
4. **A summary of a source is not a second source.** Cite the primary. If the primary is not
   in this repository, the claim is unverified — say so.
5. **Do not import anything from `../SIHProto`.** That repository's code, documents and
   numbers are deliberately not here. Nothing in it is established fact.

## Working rules

- **Do not decide the architecture unilaterally.** For anything that shapes the RF
  environment, the receiver model, ground truth, the reward, the evaluation protocol, or the
  scheduler design: propose it, give the evidence and the alternatives, and **wait for
  confirmation.** Routine implementation choices inside an agreed design do not need approval.
- **One path, not a menu.** Once something is decided, build it. Do not preserve rejected
  approaches as live alternatives, and do not hand back a decision the architecture already
  settles.
- **Keep it simple and writable.** Prefer a model that can be explained in a few sentences and
  implemented directly over one that is more faithful but nobody can hold in their head.
- **Naming.** Do not let an implementation detail become the project's identity.

## Data

`data/` is gitignored — 184 HDF5 files, 2.2 GB, already present locally. The dataset is the
Turing Synthetic Radar Dataset (gated on Hugging Face). Do not re-download it; it is here.

- **47 `scan`/`stare` pairs from the train split** — the development set.
- **45 `scan`/`stare` pairs from the test split** — held out, fetched 2026-08-28 by a rule fixed
  in advance (see `docs/project/RESEARCH_MAP.md`). **Do not touch these during development.** They exist
  so the final evaluation means something; every use must be recorded.
