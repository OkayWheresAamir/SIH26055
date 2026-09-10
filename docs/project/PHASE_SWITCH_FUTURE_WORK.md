# Phase-switching: what was tried, and what's still worth trying

**Status: not built. This is a proposal document, not a spec of anything running.**
Read `docs/project/DECISIONS.md` D66 first — it has the numbers that motivate
everything below.

## What "phase switching" means here

None of the clean D60/D61 RecurrentPPO checkpoints ever commit to a band.
Measured directly off episode logs (D64/D65's checkpoints): maximum dwell
streak 6 slots across every sample episode checked, the same order of
magnitude as round-robin's 1-2. The idea behind "phase switching" is not that
camping is bad — D14's whole point is that a policy that *only* exploits or
*only* explores loses on one axis or the other. It's that the agent currently
does neither: it never enters a deliberate exploit phase at all.

## What was tried, and why it doesn't settle the question

D66 built two heuristic rungs — `PhaseSwitch` (a hard, hand-coded commit/release
gate around a round-robin sweep) and `RLPhaseSwitch` (the same gate, but the
explore-phase action came from a trained checkpoint's own `.predict()` instead
of the sweep). Both measured **worse than the plain camper** on censored
intercept time and coverage, and using the trained policy's own suggestions
made every metric worse than the sweep-based version, not better.

That result rules out one specific thing: an *external* gate wrapping a policy
that was never trained alongside one. It does not test whether a policy that
learns to phase-switch **as part of training** would do any better — the two
options below are how you'd actually test that, and neither has been built.

## Option A — give the agent a signal to condition its own switch on

**The idea.** The observation already carries `HIT_RATE` per band, but it's a
*cumulative* rate over the whole elapsed episode — a band that was hot for the
first five slots and cold since looks similar to one that just turned hot,
because the rate smooths over the whole history. Nothing in the current
146-wide vector tells the agent "the band I'm on right now has hit twice in a
row" — which is exactly the quantity the deleted `_CommitReleaseGate.hit_streak`
computed and that D66's heuristic acted on. Add a **per-band (or current-band)
consecutive-hit streak** feature, normalised the way `VISIT_DENSITY`/`STALENESS`
already are (D55), and let the network's own weights learn whether and when to
act on it.

**What it costs, concretely, not hypothetically.**

- Bumps the observation width — 146 → 147 (scalar) or 146+36 (per-band). This
  has happened four times already (109→145→146→147→146, each one D49/D55) and
  **every time it permanently invalidated every existing checkpoint.**
  Concretely, today: `clean_lstm_s4` (D64), `lstm_balance_improved_s2` (D65) —
  both become unloadable the moment this lands.
- Full retrain from zero, both arms, ~74 min each again, then D61 selection
  again, then the headline comparison again. Call it half a day of compute
  end to end, conservatively.
- `guard.py`'s slice constants need a new entry; existing heuristic rungs read
  fixed offsets and are unaffected if the new block is appended at the end
  rather than inserted into the middle.

**Why it might not even work.** Adding a feature doesn't make the agent use
it. This only pays off if two things are both true: the streak signal is
genuinely more useful than what `HIT_RATE`/`VISIT_DENSITY`/`STALENESS` already
encode, *and* whichever reward is in use actually rewards acting on it
correctly. If `reward_balance`'s concentration penalty (D53:
`-3.0 * visit_density * n_slots`) still makes sustained dwelling unattractive
regardless of how legible the signal is, the feature just sits unused — same
failure as today, with a wider vector. **Necessary, probably not sufficient on
its own** — likely needs pairing with a reward change (the `reward_balance_improved`
direction from D65, or a fresh candidate) to have a real chance.

**Before building:** this is an observation change, which CLAUDE.md's working
rules gate behind an explicit proposal and human confirmation — the feature
spec (scalar vs per-band, exact normalisation) needs agreeing before `env.py`
gets touched, the same discipline D34/D55 were held to.

## Option C — the agent decides the phase itself, jointly with the band

**The idea.** Instead of a fixed `Discrete(36)` action (one band per slot),
the policy outputs a phase decision *and* a band — either a wider action space
(`MultiDiscrete([2, 36])`, an explore/exploit bit alongside the band) or true
hierarchical RL (a meta-policy choosing phases at a coarser timescale, a
sub-policy executing within each one).

**Two very different costs hiding under one label.**

- The `MultiDiscrete` version is not exotic — SB3 handles it natively. But the
  phase bit only means something if it changes what happens: either the
  environment enforces it (e.g. "exploit" locks the action to the band chosen
  when exploit began, for some duration) — a real change to `ScanEnv.step`,
  not just the observation — or it doesn't mechanically enforce anything and
  only exists for the reward to shape around, which is a weaker version of
  Option A with extra action-space complexity for no clear extra benefit.
- True hierarchical/options RL is a different training setup entirely.
  sb3-contrib's `RecurrentPPO` doesn't provide it — this means custom training
  infrastructure, not a config change.

**Why this is the expensive option, not the first one to reach for.**

- New action-space contract, unique to this one rung — every metric-reading
  function and comparison test that assumes "one band per slot" needs
  checking against it.
- Reward has to change in tandem, or the phase bit is unrewarded noise the
  network ignores — the same risk as Option A, compounded with a second
  learned decision instead of one.
- Two things learned jointly (when to switch, what to pick) is more failure
  surface, not less. D66 already measured that a hand-designed switch acting
  on a trained policy's outputs made things *worse* — there's no evidence yet
  that adding a *learned* switch dimension fixes the underlying issue rather
  than giving it a new way to misfire.
- Runs directly against CLAUDE.md's own working rule: "Keep it simple and
  writable. Prefer a model that can be explained in a few sentences and
  implemented directly over one that is more faithful but nobody can hold in
  their head." Of everything discussed for this line of work, this is the one
  that argument weighs hardest against.

## Option D — attention over the 36 bands, conditional on Option A

**Not a phase-switching mechanism on its own** — a different question about
*representation*, raised alongside this work and recorded here because it's
the same kind of "should we build this" decision. The observation has real
per-band structure: four 36-wide blocks (`HIT_RATE`, `VISIT_DENSITY`,
`STALENESS`, `CURRENT_BAND`) plus two scalars, and the action is "pick one of
36 bands." The current policy flattens all of that into one vector for an MLP,
which has to learn from scratch that positions 5, 41, 77 and 113 all describe
band 5. A policy that treats the 36 bands as tokens and uses self-attention
across them could learn cross-band relationships natively instead.

**Why it's not next.** Every negative result so far — the D64/D65 checkpoints
never committing to anything, D66's gate making things worse — points at the
reward and the training signal, not at representational capacity. Nothing
measured suggests the current MLP+LSTM is too weak to represent the policy
that would use a streak feature well; the diagnosis has consistently been that
nothing currently rewards using it. Attention changes how the network
processes its inputs, not what it's rewarded for. It's also real engineering
`sb3-contrib` doesn't provide out of the box — a custom `ActorCriticPolicy`
with a Transformer/self-attention feature extractor, its own new
hyperparameters, and almost certainly slower per-step training than the
current MLP+LSTM — which is a lot to spend on a hypothesis nothing has yet
supported.

**When it would become worth it.** After Option A lands and is retrained: if
the network *still* can't exploit per-band relational structure even with a
clean streak feature in hand, that is real evidence the bottleneck is
architectural rather than incentive-based, and attention becomes the
well-motivated next step rather than a guess. Not before.

## If this gets picked up again

Start with **Option A** if any of this is attempted — it's the cheaper test of
"does phase-awareness help at all" (one retrain cycle, not a new action space
and a new reward simultaneously, and not a new policy architecture), and
D66's own result is a reason to expect the answer might still be no even with
a better signal, which is worth finding out before spending on Option C's or
Option D's larger surface.

Whatever gets built, screen any accompanying reward change through
`rfenv/reward_gate.py` (D62) before training on it, and retrain under D60/D61
from the start — there is no shortcut that reuses `clean_lstm_s4`'s or
`lstm_balance_improved_s2`'s weights across an observation-width change.
