# PDW completeness and per-band pulse density: verdict and a phased brief

**Status: not built. This is a proposal document, not a spec of anything running.** No code,
observation, or reward change lands from this brief. It answers two questions that were asked
together — "use all the PDW data" and "put the truth-side per-band pulse count `C` in the
observation" — and they get different answers.

## The request

(A) Use all the data from the PDW as part of the observation space.
(B) For a band with more than one emitter's pulses, give the observation `C` for that band so the
model can learn from it.

## 1. Request (B) — literal `C` in the observation: no, and it is not merely a policy call

Two independent reasons, and they should not be blurred together.

**Mechanically, the current receiver has no process that could produce `C`.** `Receiver.dwell()`
(`rfenv/receiver.py`) collapses every pulse landing in a slot into one combined draw —
`measured = S + noise; Y = measured >= gamma` — before anything is observable. There is no
sub-slot timing model, so "1 loud pulse" and "5 overlapping pulses" are the same scalar to the
receiver. `C[b,t]` (`rfenv/truth.py`) is built independently, straight from `np.add.at` over every
contributing emitter's raw `n_pulses`, with **no `gamma` gate at all** — it counts emitters that
would never even cross the detection threshold. Handing `C` to the observation would not be
"leaking ground truth the model could otherwise reach" — it would hand the policy a number with no
path to existing outside the simulator's own bookkeeping. `DwellResult`'s own docstring calls
`C`/`.pulses` "evaluator-side... counted without reference to gamma."

**Policy-wise, it was already proposed and rejected.** D34's own text: *"What it deliberately
excludes: Pulse count and peak amplitude within the dwell (available at L2, richer than the PS's
minimal hit/miss framing)."* D29 is the governing rule: *"The observation is the policy's input
path. It ships. It must contain only what a deployed receiver has... The reward is a training-time
construct... It may read `Z`, per-emitter own levels, `first_e`, or any other simulator state."*
The "model learns from `C`" goal is **already live on the reward side** —
`reward_balance_improved` already weights its occupancy term by `log1p(C)`, just at half strength
(`_DENSITY_SHRINKAGE = 0.5`; `λ=1.0` has never been swept through the cheap reward screen). That's
the existing, correct, already-built path — it just hasn't been pushed to its limit yet.

**One nuance worth recording precisely, so nobody relies on the wrong backstop.**
`tests/test_env.py::test_the_observation_reads_no_truth_at_all` (lines 407-421) swaps the truth
grid mid-episode and checks the observation snapshot doesn't move. It would **not** catch a
`C`-derived accumulator populated during `step()` — it only detects a live re-read of `self.grid`
at observation-build time, and an already-accumulated feature (exactly how every current feature
works) sails past it. So the real prohibition here is the decision record (D29/D34) and this
project's working rule requiring sign-off on receiver/observation changes — not this test. If this
area is ever revisited, strengthening that test to catch step-time truth accumulation belongs in
the same pass, but is not proposed now.

**Verdict: decline (B) as stated. Do not reopen D34 without new evidence.**

## 2. What actually serves the underlying goal: `BAND_POWER` (proposed, not built here)

The real ask — let the agent sense "this band is busy" — has a deployable answer, not yet built:
a per-band accumulator of the running **max** of `dwell.measured_dbm.mean()` (the real, noisy
signal already in the observation as a scalar today), clamped `[-120,-20]` dBm, normalized to
`[0,1]`, reading `0.0` unvisited. It survives the grid-swap test because it's built purely from
`measured_dbm`, never from `self.grid`.

**The limitation, stated honestly, not as a footnote.** The truth grid combines co-located
emitters by `max` of amplitude, not by count — so `BAND_POWER` cannot tell "one loud emitter" from
"several quieter ones stacked here." It is **correlated with density, not equivalent to a count**.
That gap is the real cost of staying inside the deployability boundary.

`BAND_POWER` is the next concrete observation candidate, through the normal process
(reward-gate-style measurement → its own decision entry → human sign-off), following D67's own
precedent for how an observation addition gets proposed. **Not built in this pass.**

## 3. Request (A) — PulseWidth and AoA: measure before touching L1

**The raw PDW is 5 columns, confirmed against `metadata/feature_names`:**
`['ToA', 'Frequency', 'PulseWidth', 'AoA', 'Amplitude']`. `rfenv/scenario.py:257-259` reads only
columns 0/1/4 — ToA collapses to slot index, Frequency to band index, Amplitude to `peak_dbm`.
**Columns 2 and 3 (PulseWidth, AoA) are never read at all.** This is not an accidental gap — it's
**D30, an already-open, RL-owned decision** ("AoA is a measured PDW field we discard..."),
explicitly deferred until per-band hit rate/visit density/staleness demonstrably can't fix an
exploration failure. That bar hasn't been hit. D19 asks the same open question more generally.
**Neither should be closed by this brief — but a cheap first step doesn't require touching L1.**

PulseWidth has no representational blocker (a scalar, summarizable per band like `BAND_POWER`
would be). AoA is structurally harder: it's a per-emitter bearing, and L1's grid already collapses
multiple emitters per cell via `max` over amplitude — which has no sensible analogue for a bearing.
Any AoA feature needs a fixed-width encoding of a variable-length bearing set (binning, or online
clustering) before it can be a per-band scalar at all — a genuine L1 change, not a small addition.

**Cheapest legitimate first move — a measurement, not code, mirroring how D70 closed its own
question** (three inference attempts measured before ruling out threat-inference; best AUC 0.617,
"nothing clears 0.62"). Concretely: **offline, using the PDW arrays `rfenv/scenario.py` already
loads (no environment or L1 change), compute whether per-band aggregates of PulseWidth
(mean/variance) and AoA (circular variance, or cluster count under a fixed bearing tolerance)
predict per-band `C` or multi-emitter presence better than `BAND_POWER` alone would.** Minutes to
hours of CPU, produces a number in D70's own reporting style, and either motivates opening a
design for the L1 encoding work or closes the question with evidence — the same way D70 closed
threat-inference. This measurement doesn't need its own D-number; it's evidence a later decision
(reopening or re-closing D30) would cite.

**Recommendation: this measurement is the one concrete next action for (A). No observation
feature, no L1 change, until it's done.**

## 4. The honest long-term path: sub-slot pulse simulation — named, not scoped

The only path that gives the agent something resembling real pulse-counting is a materially
different L2 receiver model — each individual pulse gets its own noise draw and threshold test,
instead of one combined per-slot measurement. **D28 already names this: "the honest form... is
per-pulse detection... deferred to v2, after the gates."** This is a receiver-model change, not an
observation-space change, substantially larger than anything else in this document, and needs its
own decision and explicit human sign-off before any design work starts. It becomes worth scoping
only if §3's measurement shows PW/AoA (or a `C`-like signal) would materially move an evaluation
metric, and `BAND_POWER` has actually been tried in training and found insufficient — the same
evidence bar D30 already sets for itself.

## 5. A live coordination risk: D70 wants the same shelf space

**D70 is SETTLED but not yet implemented** (verified: zero lines in `rfenv/` touch it; only
`docs/project/DECISIONS.md` and `docs/project/THREAT_WEIGHTING_BRIEF.md` exist so far). It plans
to append its own 36-wide block — an externally-supplied per-band priority vector `p` — to the
same 183-wide observation vector any `BAND_POWER` work would also extend. **D70's own preamble
already records one real collision from two people extending shared project state concurrently**
(`D64` allocated twice, renumbered to `D70`). Before any future `BAND_POWER` width bump lands,
check `main` for D70's implementation status and append after its new offsets if it landed first;
if not, coordinate slice ordering explicitly with D70's owner before either lands, and record the
agreed ordering in whichever decision lands second.

## Summary

| item | verdict |
|---|---|
| `C` in the observation (as asked) | **No** — mechanically unrecoverable by the current receiver, and already excluded by D34 |
| The underlying goal ("learn from density") | Already live, reward-side, at half strength (`reward_balance_improved`) — sweep `λ` before building anything new |
| `BAND_POWER` (deployable density proxy) | Proposed for a future pass, not built here; correlated with density, not equivalent to a count |
| PulseWidth / AoA | Measure first (§3); do not touch L1 without evidence |
| Sub-slot pulse simulation (v2) | Named, explicitly out of scope; needs its own decision and sign-off |
| Shared resource | Observation width/layout — coordinate with D70 before either lands |

No D-number is allocated by this document. It proposes future decisions (a `BAND_POWER`
observation addition; whatever the PW/AoA measurement motivates) for separate sign-off, each
taking the next number off `main` after a pull at the time it's actually proposed — per D70's own
stated rule, since this project's own numbering has already collided once this way.
