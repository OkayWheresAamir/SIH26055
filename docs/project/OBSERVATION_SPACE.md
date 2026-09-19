# The observation space — a detailed reference

**What this file is.** A complete, element-by-element reference for `ScanEnv`'s observation
vector: every block, what it means, its dtype, its range, how it's computed, and how long it
persists. It documents the four layouts that exist today ("v1", "v2", "v2p" and "v3" —
D30/D71/D72/D74/D75)
and nothing else — it is not a decision record (that's `DECISIONS.md`,
D34/D49/D55/D67/D30/D71/D72/D74) and not a working brief (that's `STATE_ACTION_FORMULATION.md`). If
a number here disagrees with `rfenv/env.py`, the code wins; this file describes it, it does not
define it.

**Source of truth.** Everything below is read directly from `rfenv/env.py`: `_BLOCK_SPECS`,
`OBS_LAYOUTS`, `_observation_blocks()`, `reset()` and `step()`. Verified against the code this
session, 2026-09-14 (D30/D71), updated the same session for D72's `pulse_count` addition, and again
2026-09-19 for the "v2p" `band_priority` block described in §2.4 (D74, `MEASURED`: the mechanism is
built and registered, but a control-arm comparison plus a permutation ablation found it was never
learned at the scale tested — see §2.4 for the numbers), and again the same day for "v3"'s three
in-context blocks in §2.5 (D75, `BUILT` — no training run yet, so there is no result to quote).

---

## 1. The four layouts

`ScanEnv(obs_version="v1" | "v2" | "v2p" | "v3")` picks which vector gets built. It defaults to `"v1"`.

| | width | dtype | added by | every checkpoint before |
|---|---|---|---|---|
| **"v1"** | 183 | `float32` | D34, extended D49, rescaled D55, extended D67 | — (this is the baseline) |
| **"v2"** | 362 | `float32` | D30, resolved as D71 (2026-09-14), extended by D72 (2026-09-14) | trains on "v1"; "v2" is opt-in, additive |
| **"v2p"** | 398 | `float32` | "v2" + `band_priority` (D74, `MEASURED` — see §2.4) | trains on "v1" or "v2"; "v2p" is opt-in, additive |
| **"v3"** | 436 | `float32` | "v2p" + `prev_action`/`prev_reward`/`prev_hit` (D75, `BUILT` — see §2.5) | trains on any earlier layout; "v3" is opt-in, additive |

**"v1" never changes when "v2" or "v2p" exist.** Unlike every previous observation change in this
project (D49, D55, D67), which widened the vector in place and made every prior checkpoint
permanently unloadable, "v2" and "v2p" are separate, parallel layouts. A checkpoint trained on
"v1" stays exactly as loadable as it always was; nothing about it is affected by "v2"/"v2p"
existing. **A checkpoint trained on D71's own 326-wide "v2" is invalidated by D72 the same way** —
"v2" itself is not immune to widening in place, only "v1" is guaranteed stable; none exist yet, so
nothing on disk is affected. `rfenv.rl.common.known_observation_widths()` returns
`{183, 362, 398, 436}` — all four are live, current, and correct, simultaneously.

**Both layouts are built from a single per-band bookkeeping system.** `ScanEnv` always tracks
every quantity below internally (`self._hit_rate_array`, `self._band_pulse_width`, etc.)
regardless of which `obs_version` you asked for — only the final concatenation into a flat vector
differs. This means a heuristic that only reads the shared blocks (see §5) behaves identically
under either layout.

**Gymnasium contract.** `ScanEnv.observation_space` is `gymnasium.spaces.Box(low=..., high=...,
dtype=np.float32)`, built from the same table this document describes — so the declared bounds
and what actually gets emitted can never drift apart from each other. It is deliberately **not**
the unit interval: two blocks (`visit_density`, `staleness`) declare a ceiling above 1.0 on
purpose (§3).

---

## 2. Element-by-element reference

Every block is per-band unless marked **scalar**. "Per-band" means width 36 (`N_BANDS`), one
number for each of the 36 frequency bands, in band order (band 0 first).

### 2.1 Blocks common to both "v1" and "v2"

These eight blocks exist, in this order, at the *start* of both vectors — so their absolute
vector positions are identical in "v1" and "v2" (see §4's offset tables). All are built purely
from the agent's own scan history; none of them ever reads the truth grid (D29, D34).

| Block | Width | Type | Range | Meaning |
|---|---|---|---|---|
| `hit_rate` | 36 | per-band | `[0.0, 1.0]` | Declared hits on this band ÷ slots looked at this band. `0.0` for a never-visited band (not `NaN`). |
| `visit_density` | 36 | per-band | `[0.0, 36.0]` | Airtime spent on this band so far, in **fair shares**: `1.0` = exactly its equal cut of the episode, `36.0` = the whole episode spent here (full camping). |
| `staleness` | 36 | per-band | `[0.0, ≈13.95]` | How overdue this band is for a look, in **reference sweeps** (one sweep = 43 slots, the schedule's own natural unit): `1.0` = exactly one full pass overdue. A never-visited band reads the ceiling, `N_SLOTS / SWEEP_SLOTS = 600/43 ≈ 13.953` — maximally stale. |
| `current_band` | 36 | per-band, one-hot | `{0.0, 1.0}` | Exactly one `1.0`, at whichever band the most recent dwell was on; `0.0` everywhere else. Lets the agent distinguish "I am here right now" from "I was just here" (which `staleness` alone reads as `0` too). |
| `clock` | 1 | **scalar** | `[0.0, 1.0]` | `t / N_SLOTS` — fraction of the 600-slot episode elapsed. |
| `hit_streak` | 36 | per-band | `[0.0, 1.0]` | Consecutive declared hits on this band across *separate visits* (not reset by visiting a different band, only by a miss on this one). Capped at 5 and divided by 5, so `1.0` means "hot on the last 5-or-more visits." |
| `current_hit_streak` | 1 | **scalar** | `[0.0, 1.0]` | `hit_streak` evaluated at whichever band `current_band` is one-hot on — a direct scalar so a consumer doesn't have to compute the dot product itself. Fully redundant with `hit_streak` + `current_band` together (deliberately — see `DECISIONS.md` D67). |

### 2.2 "v1"-only block

| Block | Width | Type | Range | Meaning |
|---|---|---|---|---|
| `measured_dbm` | 1 | **scalar** | `[0.0, 1.0]` | The **global**, last-dwell-only signal reading: mean of `S + noise` (what the receiver's detector actually read) over the most recent dwell's slots, clamped to `[-120, -20]` dBm and linearly rescaled to `[0, 1]`. Forgets every band but the one just left the instant the agent moves on — this is exactly the weakness "v2" fixes with `measured_dbm_band`. Before any dwell, reads `0.0` (the clamp floor, quietest possible). |

### 2.3 "v2"-only blocks (D30, resolved as D71, 2026-09-14; extended by D72, same day)

Five blocks, appended after `current_hit_streak`, in this order. Four of the five (all but
`measured_dbm_band`) are gated on `Y` (see below). Three (`pulse_width`, `aoa_sin`, `aoa_cos`)
read the two previously-discarded PDW columns, PulseWidth and AoA (`metadata/feature_names`
columns 2 and 3 — `rfenv/scenario.py` now reads them instead of silently dropping them); the
fifth, `pulse_count` (D72), reads `dwell.C`, a quantity that already existed in `DwellResult` but
had never entered the observation before this.

**Why `pulse_count` is 36 wide when `dwell.C` itself isn't.** `DwellResult.C` (`rfenv/receiver.py`)
is per-*slot*, sized to whichever dwell just ran — 1 or 2 elements, one illumination count per slot
of that single dwell. It carries no band dimension on its own. `pulse_count` gets its width from
`self._band_pulse_count`, a persistent `(N_BANDS,)` array `ScanEnv` maintains across the whole
episode (`reset()`): index *b* holds "the `C` value the last time band *b* had a declared hit,"
updated only for the band just dwelt on (`step()`) and left unchanged for the other 35. This is the
same pattern every per-band block in this document follows — one slot of memory per band, not a
reshaping of whatever the current dwell produced — so the agent can compare bands against each
other, not just see the one it is currently tuned to.

| Block | Width | Type | Range | Meaning |
|---|---|---|---|---|
| `measured_dbm_band` | 36 | per-band | `[0.0, 1.0]` | `measured_dbm`'s per-band replacement: the last dwell's mean `S + noise` reading **for that specific band**, clamped/rescaled identically, but persisting until the band is visited again (same convention as `hit_streak`). `0.0` for a never-visited band. |
| `pulse_width` | 36 | per-band | `[0.0, 1.0]` | The pulse width (raw PDW column, microseconds) of the loudest pulse in the last **declared hit** on this band. Clamped to `[0, 200]` µs (measured this session over 6 real recordings: true range 0.007–220.0 µs, p99 = 102.3) and rescaled to `[0, 1]`. `0.0` if this band has never had a declared hit. |
| `aoa_sin` | 36 | per-band | `[0.0, 1.0]` | `(sin θ + 1) / 2`, where θ is the angle of arrival (degrees, converted to radians) of the loudest pulse in the last declared hit on this band. See §3 for why sine/cosine rather than a raw angle. |
| `aoa_cos` | 36 | per-band | `[0.0, 1.0]` | `(cos θ + 1) / 2`, the paired cosine component. `aoa_sin` and `aoa_cos` are always read together. |
| `pulse_count` | 36 | per-band | `[0.0, 1.0]` | `log1p(C) / log1p(64)`, clipped to `[0, 1]` — `C` being the illumination count (`truth.py`) at the loudest slot of the last **declared hit** on this band. `0.0` if this band has never had a declared hit. Same reference constant (`_DENSITY_REF_PULSES = 64`) `reward_balance_improved` already normalises `C` by, reused rather than duplicated (D72). |

**Gating, four of the five "v2"-only per-band signals (`measured_dbm_band` excepted — see
below):** `pulse_width`, `aoa_sin`, `aoa_cos` and `pulse_count` only update on a slot where the
receiver **actually declared a hit** (`Y = 1`). A real receiver only measures a pulse's width and
bearing on a pulse it detected, and D72 extends the same argument to count: a real receiver's own
PDW stream carries a count of the detections it resolved on a hit. Writing any of these on a miss
would be reading truth the receiver never had. `measured_dbm_band` is **not** gated this way:
amplitude is a continuous quantity a receiver reads on every slot it's tuned to, hit or miss, the
same way `measured_dbm` already worked in "v1".

**`pulse_count`'s gap, stated precisely (D72).** The *cell* is gamma-gated, just indirectly: `Y`
is exactly `measured_dbm >= gamma` (`receiver.py`, `Y = measured >= self.gamma`, `measured = S +
noise`), so requiring `Y = True` before `pulse_count` updates already is a gamma test — a noisy
one, since it runs on `measured` (the receiver's own noisy read), not on the true signal, the same
way every real detection in this model works (D26: that noise is *why* `Pd < 1` and `Pfa > 0`
exist at all). What is **not** gamma-gated is what happens *inside* a cell that clears that test:
`C` (`truth.py`) sums every contributing emitter's raw pulse count landing there, with no reference
to any single contribution's own amplitude — a cell can pass the combined-signal gamma test on the
strength of one loud emitter while `C` also counts several quieter co-located ones that would never
individually have cleared `gamma` on their own. So a hit's `pulse_count` can still be inflated by
those sub-threshold contributors a real per-pulse-resolving receiver would not have logged. `Y`-
gating is a real gamma gate on the cell as a whole; it does not, and structurally cannot, extend
that same test down to each contribution `C` adds together — that would need resolving individual
pulses within one merged cell, which D28 already rules out for this receiver model ("not a
per-pulse detector... a real receiver cannot un-mix a cell"). See `DECISIONS.md` D72 and
`PDW_COMPLETENESS_AND_BAND_DENSITY_BRIEF.md` §1 for the full argument on both sides.

**Within a multi-slot dwell with more than one declared hit,** the loudest slot is the
representative reading for `pulse_width`/`aoa_sin`/`aoa_cos`/`pulse_count` — the same rule the truth
grid (`rfenv/truth.py`) already uses to resolve multiple *emitters* sharing one cell (`TruthGrid.PW`/
`.AOA` follow whichever contributor's peak amplitude wins).

### 2.4 "v2p"-only block (D74, `MEASURED` — never learned at the scale tested)

One block, appended after `pulse_count`, on top of everything "v2" already has. Unlike every
other block in this document, it is not derived from anything the receiver measures — it is an
**exogenous** per-episode input, closer in spirit to a mission brief than a sensor reading.

| Block | Width | Type | Range | Meaning |
|---|---|---|---|---|
| `band_priority` | 36 | per-band | `{1.0, 3.0}` | How important this band is *this episode*, supplied to the env, not computed by it. `1.0` = ordinary (the default for every band). `3.0` = elevated. |

**Where the values come from.** At the start of every episode (`reset()`), if the env was built
with `band_priority=True` and `priority_uniform=False`, a random count of bands (`priority_n_bands`,
default 3–6 of the 36) is drawn and elevated to `3.0`; the rest stay at `1.0`. If
`priority_uniform=True` (the control-arm setting — see below), every band stays at `1.0` all
episode, every episode. If `band_priority=False` (the default), the block still exists in a "v2p"
vector — the layout's width never changes based on a constructor flag — but it stays at the
all-ones sentinel the whole time, identical to the `priority_uniform=True` case.

**Not gated on `Y`, unlike every "v2"-only block.** Every other per-band block in §2.3 only
updates when the receiver actually detects something, because each of them reports something the
receiver *measured*. `band_priority` reports something nobody measures — it is handed to the
scheduler the same way a mission brief or an operator's standing order would be — so there is
nothing to gate: it is set once per episode and visible for the whole episode, on every band,
whether or not the agent has ever looked there.

**Why this exists — the reward side, not just the observation.** A block that only sat in the
observation would do nothing on its own; `ScanEnv.step()` also adds a term directly to whatever
`self._reward_fn` (i.e., whichever `REWARDS` candidate is selected) already returned:
`reward += priority_coef * band_priority[dwell.band] * len(newly)` — an *additive* bonus, scaled by
the current band's priority, paid only when the dwell **discovers** something new (`newly`), not
per slot spent there. Gating on discovery rather than per-slot occupancy is deliberate: a per-slot
multiplier would pay repeatedly for camping a high-priority band, the exact exploit D53 already
found and fixed once for the base reward. This term is **not** a new entry in the `REWARDS` dict
and does not touch `reward_gate.py`'s D62 screen — it is added on top, in `env.py`'s `step()`,
composable with whichever reward is selected.

**The control-arm setting, and why it exists.** Because `band_priority` defaults to `1.0` even on
ordinary bands, the reward term above fires on *every* discovery, not only ones on elevated bands —
a uniform scale increase, plus a differential for elevated ones. `priority_uniform=True` keeps that
same uniform scale increase but removes the differential (every band stays at `1.0`), which is what
makes it a genuine control: it isolates whether a trained policy's behaviour comes from actually
reading `band_priority`, versus merely benefiting from a larger reward scale that has nothing to do
with which band it's on.

**Status (D74, `MEASURED`).** Two RecurrentPPO checkpoints tested this —
`lstm_balance_v2p_priority_seed2` (treatment, `priority_uniform=False`) and
`lstm_balance_v2p_uniform_seed2` (control, `priority_uniform=True`) — registered as ladder rungs
22a/22b. **The comparison runs the wrong way**: the control beat the treatment on every headline
column (ratio, cTTI, coverage, beats-recency-both — 61.4% vs 44.4%). Camping was checked directly
and ruled out (only ~7% more airtime on elevated bands than ordinary ones, inconsistent in
direction). Treatment is measurably more diffuse than control across the whole spectrum (entropy
3.255 vs 3.051, 24.95 vs 18.95 of 36 bands touched), plausibly because control's constant input made
for an easier training problem. **A permutation ablation settled it**: the treatment checkpoint's
airtime correlates with true priority equally poorly whether fed the real vector or one shuffled
across bands (+0.018 vs +0.019), and its performance doesn't degrade when the signal is corrupted —
**the agent never learned to use `band_priority`**. Single seed, one comparison run, not read as
settled beyond this configuration.

**Follow-up, same day: tried stronger, same answer.** A new decaying per-slot occupancy term
(`priority_reward_bonus`, on top of the original discovery term, anchored to `visit_density` to stay
clear of D53's camping exploit), `priority_coef` 0.5→2.0, elevated value 3.0→5.0 (`priority_high`,
now configurable), 512-wide LSTM (was 256), 800k timesteps (was 400k). The strengthened treatment
(rung 23a) scored **73.1%** beats-recency-both — the highest in this project — but a second
permutation ablation found its airtime correlates with true priority identically whether fed real or
shuffled values (+0.031 both ways). **Still never learned**, at roughly 4x the incentive and double
the capacity/budget. Full numbers: `MODEL_COMPARISON.md`'s Width 398 section. Not adopted, not
promoted, code not removed — nothing defaults to it. Full account: `DECISIONS.md` D74.

---

### 2.5 "v3"-only blocks (D75, `BUILT` 2026-09-19 — no training run yet)

Three blocks, appended after `band_priority`, that hand the recurrent policy its own last decision
and what that decision returned. This is the RL² construction: a recurrent policy shown its previous
action and previous outcome can run an adaptation rule *inside* the episode, in its hidden state,
with no gradient step — which is what "online" means in the PS's sense of working *"in the absence of
prior reliable intelligence"*.

| block | width | range | what it is |
|---|---|---|---|
| `prev_action` | 36 | `[0, 1]` | one-hot of the last band chosen. **All-zero before the first step** |
| `prev_reward` | 1 | `[0, 1]` | `reward_balance_obs`, clipped to ±6.0 and affinely rescaled |
| `prev_hit` | 1 | `[0, 1]` | `1.0` if the last dwell declared anything (`dwell.Y.any()`), else `0.0` |

**36 of these 38 columns already existed, and the documentation says so up front.** `step()` assigns
`self._current_band = action` and `self._prev_action = action` from the same value, so `prev_action`
is **bit-identical to the `current_band` block at every step after the first**, and `prev_hit` is
`current_hit_streak > 0`. They differ only at the cold start: `current_band` reads one-hot at band 0
after `reset()` — claiming a dwell that never happened — while `prev_action` reads the zero vector,
which is off the one-hot simplex and therefore unreachable by any real action. So **`prev_reward` is
the only genuinely new information in this layout**, and an ablation corrupting `prev_action` alone
would read null by construction. A test pins the equality, so a later change to either block is
noticed rather than discovered.

**`prev_reward` is not the training reward.** It is `reward_balance_obs` — `reward_balance` with
`dwell.Y` substituted for `dwell.Z`, i.e. what the receiver *declared* rather than what was truly
transmitting. This matters because the agent already holds `hit_rate`, `visit_density`, `staleness`
and `n_slots`: given the training reward's value it could solve the remaining term for
`0.5 * dwell.Z.sum()` and learn about emitters it never detected, which no fielded receiver can do.
The two differ by exactly the receiver's own sensitivity — Pd = 0.8421, Pfa = 1.35e-3 at the frozen
operating point — and keeping them apart is what leaves a "v3" rung `deployable=True`. D29 is
unchanged: the *reward* still reads truth, and still trains and scores every arm.

The clamp is `±6.0`, set from the formula's analytic bound (`[-6.0, +3.0]`) rather than from a
percentile, so it cannot bind on real data. It is symmetric so that exactly-zero reward maps to
exactly `0.5`, which makes the first-step fill truthful rather than arbitrary. Measured over 23,025
steps of round_robin/recency/camper: realised range `[-4.5053, +2.5000]`, and **never exactly zero**,
so the sentinel is unambiguous. Full account: `DECISIONS.md` D75.

---

## 3. Two encoding decisions worth understanding, not just reading off the table

**`visit_density` and `staleness` are not on `[0, 1]`, on purpose (D55).** Both used to be
normalised by the episode length, which pinned `visit_density`'s mean at `1/36 = 0.0278` and made
`staleness` bimodal (mean 0.070, with everything-unvisited piled at a `1.0` ceiling) — a feature
that never leaves the bottom tenth of its declared range is one the network has to learn to
amplify before it can use at all. Rescaling them into "fair shares" and "reference sweeps" means
`1.0` means something concrete in each (equal airtime; one pass overdue) instead of being an
arbitrary point near the bottom of a wasted range.

**AoA is circular, so it is stored as `(sin θ, cos θ)`, not a raw angle.** 359° and 1° are almost
the same bearing, but as raw numbers they are nearly maximally far apart — feeding that
difference straight into a network would teach it they're opposites. Splitting into sine and
cosine (each linearly rescaled into `[0, 1]`) preserves the circular topology correctly. This has
a useful side effect, not a separately-designed one: every *real* angle measurement lands on the
unit circle (`sin²θ + cos²θ = 1`), so the rescaled pair `(0.5, 0.5)` — which decodes back to raw
`(0, 0)`, the centre of the circle — can never be produced by an actual bearing reading. It is
therefore a clean, provably-unreachable "never measured" flag, verified directly in
`tests/test_observation_d30.py`.

---

## 4. Exact vector layout (offsets)

Useful for anyone reading a raw `float32` array and needing to know which index is which.
`rfenv/baselines/guard.py` defines the "v1" offsets below as named slices (`HIT_RATE`,
`VISIT_DENSITY`, …) for exactly this purpose; "v2" has no equivalent module yet since no
heuristic rung currently needs to address past index 144 under it (see §5).

### "v1" (183 wide)

| Index range | Block |
|---|---|
| `0 : 36` | `hit_rate` |
| `36 : 72` | `visit_density` |
| `72 : 108` | `staleness` |
| `108 : 144` | `current_band` |
| `144` | `clock` |
| `145` | `measured_dbm` |
| `146 : 182` | `hit_streak` |
| `182` | `current_hit_streak` |

### "v2" (362 wide, D72 appended `pulse_count` after `aoa_cos`)

| Index range | Block |
|---|---|
| `0 : 36` | `hit_rate` |
| `36 : 72` | `visit_density` |
| `72 : 108` | `staleness` |
| `108 : 144` | `current_band` |
| `144` | `clock` |
| `145 : 181` | `measured_dbm_band` |
| `181 : 217` | `hit_streak` |
| `217` | `current_hit_streak` |
| `218 : 254` | `pulse_width` |
| `254 : 290` | `aoa_sin` |
| `290 : 326` | `aoa_cos` |
| `326 : 362` | `pulse_count` (D72) |

### "v2p" (398 wide, appends `band_priority` after `pulse_count`; D74)

| Index range | Block |
|---|---|
| `0 : 36` | `hit_rate` |
| `36 : 72` | `visit_density` |
| `72 : 108` | `staleness` |
| `108 : 144` | `current_band` |
| `144` | `clock` |
| `145 : 181` | `measured_dbm_band` |
| `181 : 217` | `hit_streak` |
| `217` | `current_hit_streak` |
| `218 : 254` | `pulse_width` |
| `254 : 290` | `aoa_sin` |
| `290 : 326` | `aoa_cos` |
| `326 : 362` | `pulse_count` |
| `362 : 398` | `band_priority` |

**The first 144 indices are identical across all three layouts** — `hit_rate`, `visit_density`,
`staleness` and `current_band` sit at exactly the same offsets whether the env is built with
`obs_version="v1"`, `"v2"` or `"v2p"`. Everything from index 144 onward diverges, because
`measured_dbm` (1-wide) is replaced by `measured_dbm_band` (36-wide) at that point and every later
block shifts. "v2" and "v2p" agree on every index up to 362; "v2p" simply appends one more block.

### "v3" (436 wide, appends the three in-context blocks after `band_priority`; D75)

| Index range | Block |
|---|---|
| `0 : 398` | exactly "v2p", unchanged |
| `398 : 434` | `prev_action` |
| `434` | `prev_reward` |
| `435` | `prev_hit` |

Append-only, so every "v2p" offset holds — a test asserts `obs_v3[:398]` is byte-identical to the
"v2p" vector for identically seeded, identically acted environments.

---

## 5. Why heuristic rungs (round_robin, recency, camper, …) don't care which layout runs

`recency.py` (rung 5) reads only `HIT_RATE` and `STALENESS`; `camper.py` reads only `HIT_RATE`.
Both slices sit inside the shared first-144-index range (§4), so both heuristics behave
**identically** whether the environment was built with `obs_version="v1"` or `"v2"` — verified
directly (`tests/test_observation_d30.py::test_v1_and_v2_agree_on_every_block_they_share`, an
identical seeded episode run under both layouts produces byte-identical values on every shared
block). `round_robin` and `turing_sweep` don't read the observation at all. This is what makes it
safe to run `python -m rfenv.compare --obs-version v2 --rungs round_robin,recency,<a v2 checkpoint>`
and get a genuinely fair, same-scenario/seed comparison — the heuristics are not disadvantaged or
changed by the wider vector.

**A trained RL checkpoint is not interchangeable this way.** Its policy network's input layer is
a fixed width, set at training time. A "v1" checkpoint fed a "v2" (362-wide) observation — or vice
versa — raises inside `predict()` (`ValueError: Unexpected observation shape`), not silently. Every
ladder `Rung` records which layout it needs (`Rung.obs_version`, default `"v1"`); `B.make()` builds
the policy, but the **caller** is responsible for constructing `ScanEnv` with the matching
`obs_version` — `rfenv.compare`'s `--obs-version` flag and `rfenv.rl`'s `--obs-version` training flag
both exist for exactly this.

---

## 6. What is deliberately *not* here

Two things a reader might expect to find, and the reason each is absent — plus one that looks
absent but is now present in a narrower form than its literal version:

- **Literal, per-contribution-ungated pulse count, `C`.** Still absent, and still cannot be added
  without reopening D29/D34 again: `truth.py` builds `C` by summing every contributing emitter's
  raw pulse count into a cell with no reference to any single contribution's own amplitude, so it
  can include contributions that would never individually cross `gamma` — an ungated reading would
  hand the policy a number with no path to existing outside the simulator's own bookkeeping
  (`PDW_COMPLETENESS_AND_BAND_DENSITY_BRIEF.md` §1). What **is** now present, "v2" only, is
  `pulse_count` — the same `C`, but `Y`-gated (D72), and `Y` is itself a gamma test
  (`measured_dbm >= gamma`, on the receiver's own noisy reading) — so the cell as a whole is
  gamma-gated indirectly through `Y`; only the per-contribution amplitude check inside `C` is
  missing, and cannot be added without resolving individual pulses within one merged cell, which
  D28 rules out for this receiver model. `Y`-gating narrows the gap without closing
  it (§2.3 above states the residual gap precisely). It remains legitimately available to the
  *reward* unrestricted (D29 permits reading truth there), and
  `reward_balance_improved`/`reward_balance_improved_v2` already use the raw form.
- **Per-emitter attribution / bearing clustering.** `aoa_sin`/`aoa_cos` give one raw "which
  direction did the last hit come from" reading per band. They do **not** give "have I already
  caught this specific emitter" — D30's own measurement found that's AoA's real value, but it
  needs comparing a bearing against a *set* of bearings already seen in that band this episode
  (online clustering), which is a materially larger change (D19 flags the deinterleaving risk) and
  is not built.
- **Raw ToA / Frequency.** Both are already fully spent building the vector's structure rather
  than appearing as standalone fields: ToA becomes the slot clock and every per-band staleness/
  visit-density timing; Frequency becomes the band index (`current_band`'s one-hot). There is no
  leftover "raw ToA value" or "raw frequency in MHz" to add on top — actions are per-band, so
  finer-than-band frequency resolution has no lever to act on.

---

## 7. Cross-references

- `DECISIONS.md` — **D34** (base vector ratified), **D49** (109→146), **D55** (rescaled, `camp_time`
  removed), **D67** (146→183, `hit_streak`), **D30 / D71** (183→326, "v2"), **D72** (326→362,
  `pulse_count`), **D74** (362→398, `band_priority` — built, registered, `MEASURED` as never learned
  at the scale tested; not adopted, not promoted, code not removed).
- `rfenv/env.py` — `_BLOCK_SPECS`, `OBS_LAYOUTS`, `obs_width()`, `ScanEnv._observation_blocks()`
  (every block's own inline derivation and reasoning lives in its docstring).
- `rfenv/baselines/guard.py` — the "v1" slice constants heuristic rungs read by name.
- `tests/test_observation_d30.py` — the "v2"-specific invariants (gating, sentinel, persistence,
  layout agreement, and D72's `pulse_count` transform) as executable tests.
- `STATE_ACTION_FORMULATION.md` — the broader RL-lane working brief this reference doc does not
  replace (action space, reward candidates, amendment history).
- `PDW_COMPLETENESS_AND_BAND_DENSITY_BRIEF.md` — the planning document that preceded D71, including
  the mechanical/policy case against ungated `C` and the `BAND_POWER` alternative; D72 narrows but
  does not overturn its verdict (see D72's own entry for how the two relate).
