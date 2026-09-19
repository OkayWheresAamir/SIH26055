# Evaluation

**The single authority on how anything in this project is measured.** Metric definitions, the
baseline ladder, the validation gates, and the comparison protocol. Rationale for each choice
lives in `docs/project/DECISIONS.md`; this file is the implementable version.

Kept separate from `ENVIRONMENT_SPEC.md` on purpose: the environment must not define its own
scorecard, and duplicated metric definitions drift apart.

---

## 0. The three families, and why the split matters

The PS lists nine figures of merit. Most confusion in this project came from mixing three
different kinds of question. Separating them is what makes the evaluation honest (D21).

| Family | Asks | Varies with the scheduler? |
|---|---|---|
| **Model-level** | Is the environment a faithful model of the data? | No |
| **Receiver-level** | How good is the receiver when it looks? | No |
| **Scheduler-level** | How well did we aim it? | **Yes** |

A scheduler cannot improve P<sub>d</sub> — it can only point the detector at more of the right
cells. Reporting a scheduler comparison on receiver-level metrics is a category error.

---

## 1. Notation

For one scenario over one 30 s episode:

- Bands `b ∈ {0..35}`, slots `t ∈ {0..599}` (50 ms each) — D3, D16.
- `Z[b,t]` — physical occupancy: is any emitter transmitting into this cell? Threshold-free,
  and the PS's binary transmission/non-transmission status (D26).
- `S[b,t]` — true received signal level (dB); the noise floor `N₀` where `Z` is false (D4).
- `a(t)` — band the scheduler is tuned to at slot `t`.
- `Y(t) ∈ {0,1}` — the receiver's **declared** detection at slot `t`, i.e. what the scheduler
  actually observes: `Y = 1` iff `S[a(t),t] + n ≥ γ`, `n ~ N(0, σ)`. `Y` may differ from
  `Z[a(t),t]` because of noise: that difference is exactly what P<sub>d</sub> and
  P<sub>fa</sub> measure.
- `E` — emitters with a non-empty **detectable activity interval**, i.e. whose own received
  level clears γ at some slot (D27). Not every transmitter in the metadata: 19.0% of train
  transmitters never appear in either recording. `on_e` — the start of that interval.
- `first_e` — slot of the first true intercept of `e`; `∞` if never intercepted. An
  intercept of `e` at slot `t` requires **all three** (D5 ∧ D27, restated as D28): the
  scheduler was tuned to a band `e` puts pulses into (`a(t)` ∈ bands of `e` at `t`); **`e`'s
  own** received level in that cell clears γ; and the receiver declared `Y(t) = 1`. The
  own-level clause is what stops a quiet emitter inheriting a loud neighbour's detectability —
  the same rule that defines `on_e`, so numerator and denominator agree and `first_e ≥ on_e`
  always. `Y` itself is declared on the **combined** `S[a(t),t]`, because that is all a real
  receiver has (D28).

**Every metric below is computed identically for every scheduler, including baselines.**

---

## 2. Model-level metrics — do we believe the environment?

Computed by running a **known** schedule (Turing's own sweep) through the environment and
comparing to the actual recordings. These validate the simulator, not any scheduler.

| Metric | Definition |
|---|---|
| **% correct predictions** | Fraction of (band, slot) cells where the environment's predicted detection matches the recorded scan data, under Turing's own schedule. |
| **Average intercept-time error** | Mean absolute difference between per-emitter first-intercept time predicted by the environment and the value measured from the actual scan recording. |

> **PS reading.** *"The model should enable prediction of intercept time and interception ratio
> of a scanning receiver…"* assigns prediction to the **system model**. That is why these two
> live here and not in the scheduler family.

---

## 3. Receiver-level metrics — how good is the detector?

Computed once, at environment freeze time, by sweeping γ. Reported as an ROC curve plus the
chosen operating point. **Identical for every scheduler** (D15, D21).

| Metric | Definition |
|---|---|
| **P<sub>d</sub>** | `P(Y=1 | Z=1)` — declared a hit given an emitter really was transmitting there. |
| **P<sub>fa</sub>** | `P(Y=1 | Z=0)` — declared a hit given the cell was truly empty. |
| **Sensitivity** | The signal level at which P<sub>d</sub> reaches a stated value (e.g. 0.9) at the operating P<sub>fa</sub>. Quoted *with* both, never alone. |

These are per-look conditional probabilities, not properties of the whole grid.

**The cells they are averaged over: the reference-sweep population** — the occupied cells
Turing's own dwell schedule looks at (D33, `SETTLED`; `PD_POPULATION` in `rfenv/constants.py`).
"Cells the receiver actually looked at" cannot be the answer as written: different schedulers
look at different cells, and per-cell detection probability is not uniform (`Φ((S−γ)/σ)`, D29),
so that population would make P<sub>d</sub> scheduler-dependent and contradict D21. Turing's
schedule never varies, so this population is per-look *and* scheduler-independent. Every ROC is
reported with its population stated on it.

**Conditioned on `Z`, not on `(S ≥ γ)`** (D26). Referencing P<sub>d</sub> to a second copy of
`S` thresholded at the same γ is degenerate — it forces `P_d ≥ 0.5` for every γ and the curve
can never sweep. Against threshold-free `Z` it does: measured over the 47 train configs,
P<sub>d</sub> falls as γ rises. At the default operating point `γ = N₀ + 3σ = −111 dB`:
**P<sub>fa</sub> = 1.35e−3** (exact — it is `1 − Φ(3)`) and **sensitivity −107.2 dB**
(= `γ + 1.2816σ`); both re-run 2026-09-03 and confirmed.

**P<sub>d</sub> = 0.851** at γ = −111 over the reference-sweep population (D33). The previously
quoted 0.822 is withdrawn: it matched no population, and the 2026-09-03 re-run gives 0.819 over
stare-replay cells and 0.837 over scan-replay cells, so the figure depends entirely on a choice
that had never been stated. The sweep's *shape* is unaffected by it, and the ROC — not the single
point — remains the deliverable (D15).

---

## 4. Scheduler-level metrics — the actual comparison

This is the table that compares schedulers. **The first two are the PS's primary objectives and
must always be reported together** (D14).

| Metric | Definition | Direction |
|---|---|---|
| **Interception ratio** | Intercepted illuminations ÷ total illuminations — a pulse counts as intercepted iff the scheduler's tuned band window contained it at its ToA. **Per illumination, not per dwell.** | higher better |
| **Censored mean intercept time** | `mean over e ∈ E of (first_e − on_e)`, with **`first_e` set to episode end (30 s) for any emitter never intercepted.** | lower better |
| **Emitter coverage** | `|{e : first_e < ∞}| ÷ |E|` — supporting diagnostic, always printed beside the two above. | higher better |
| **Average intercept rate** | Distinct emitter-intercepts per second of episode. | higher better |
| **Average reward / cost** | The scheduler's own accumulated reward, reported as a scalar. Comparable only within a reward family; never used to rank across different rewards (D7). | — |

**Rewards are judged from outside, and they read what they like** (D29). A reward function is a
training-time construct — training is offline and the policy is frozen before deployment — so it
may read truth-side state (`Z`, per-emitter own levels, `first_e`). Only the *observation* carries
the deployability constraint. Note that D28 puts the first two metrics on opposite sides of that
line: **censored intercept time requires `Y = 1`, interception ratio does not**, so no single
reward is aligned with both, and the D7 comparison has to resolve a Pareto front rather than a
scalar. **No reward can move P<sub>d</sub> or P<sub>fa</sub>** (§3, D15, D21) — penalising false
alarms prices a wasted dwell, it does not improve the receiver.

### Choosing between reward candidates (D47)

Fixed 2026-09-05, **before any training run**, because a rule chosen after the results are visible
is not a rule. The winner among D29's three candidates is **the one that beats round-robin on both
headline metrics on the largest fraction of paired episodes** — same scenario, same seed, same
truth grid, the `both` column of `compare.paired_wins()`. Within 5 pp, nothing is selected: both
are reported and the choice is escalated.

It ranks neither objective, because the problem statement does not. It also needs no exchange rate
between a fraction and seconds, which is the thing no measurement in this project can supply.

Two things must be published beside the winner, and are not optional: **the full §4 table with
spread** (the rule is insensitive to margin — beating by a hair counts as much as beating by
triple), and **the comparison against rung 5**, which is the actual bar (D46). Round-robin is the
paired denominator because it is the PS's named open-loop floor and is identical for every
candidate, so it cancels; it is not the bar.

**Run for the first time 2026-09-10 (D68), against `reward_balance` and `reward_balance_improved`
— the only two candidates D62's screen has passed.** Each represented by its D61-selected
checkpoint (`lstm_balance_d67_control_s3`, `lstm_balance_d67_treatment_s1`), paired against
round-robin: `reward_balance` 65.5% both, `reward_balance_improved` 64.3% both. **1.2 pp apart,
inside the 5 pp margin — no candidate selected, per the rule's own text.** Both clear rung 5
decisively; between themselves, D47 declined to pick.

**Resolved 2026-09-11 with matched seeds.** Two more seeds per arm fed a fuller D61 selection: the
treatment checkpoint was unchanged, but the control arm's pick moved to a materially stronger
checkpoint (`lstm_balance_d67_control_seed2_s3`, net dominance +36.1% against the single-seed
pick's +25.0%). Re-applying D47 on the new pair: `reward_balance` **81.9%** both,
`reward_balance_improved` **64.3%** both (identical to before — its checkpoint never changed).
**17.6 pp apart, decisively outside the margin — `reward_balance` is selected.** Full accounting
in D68.

### Two traps, both measured on real data (D14)

1. **Per-dwell hit rate is misleading.** A camper that parks on the busiest band scores 85–90%
   per-dwell and looks near-optimal, while capturing only 30% of emitters. The per-illumination
   definition above is the honest one and is what the literature uses.
2. **Uncensored intercept time rewards not looking.** Averaged over only *found* emitters, the
   camper appears *faster* (6.5 s) than round-robin — because it only ever finds the loudest
   emitters. Censoring at episode end reverses it correctly (23.8 s vs 9.7 s).

**Rule: never publish interception ratio without coverage and censored intercept time beside
it.** A single scalar hides the entire problem.

---

## 5. Baseline ladder

Every scheduler runs on identical scenarios and seeds, at the same frozen γ (D13) — and on the
**stare replays plus sampled scenarios, never on scan replays** (D36). A scan recording contains
only the pulses Turing's own sweeping receiver was tuned to, so a grid built from one hands any
sweeping scheduler its answer: measured, interception ratio 0.9999 and censored intercept time
0.00 s for the reference sweep on `config_2` scan, against 0.0677 and 4.57 s for the same sweep
on the same config's stare replay. Scan replays keep their role in gates 1 and 2, where that
imprint is the point. `rfenv/compare.py` refuses a scan replay rather than trusting the caller.

| # | Baseline | Purpose |
|---|---|---|
| 1 | **Random** | Lower bound; also the coverage-heavy extreme. |
| 2 | **Round-robin (equal airtime)** | The open-loop strategy the PS explicitly targets. The floor to beat. **Every band gets 2 slots per 72-slot (3.60 s) cycle** — it refuses the sweep's weighting (D43). |
| 3 | **Turing reference sweep** | The dataset's own schedule — makes our numbers comparable to the recordings. Wide bands get double airtime, 43-slot (2.15 s) cycle. |
| 4 | **Greedy static (camper)** | The degenerate exploit. Included *precisely* to show a single metric can be gamed. **Observation-fed**: probes three sweeps, camps on the largest observed hit rate. |
| 5 | **Recency / activity heuristic** | Simple adaptive benchmark: `argmax(hit rate + gap measured in reference sweeps)` over the D34 observation. |
| 6a | **Apfeld: Active RFs** | Apfeld's own ablation — the tentative list with no period estimation (D13, D45). |
| 6 | **Apfeld adaptive** | Published non-learning adaptive strategy (`docs/reference/scheduling/paperSSPD (1).pdf` §II), adapted to a binary-detection receiver (D44). The serious bar. |
| 7 | **DQN** | Ours. Built (D48). **No runnable checkpoint** — all six DQN/PPO checkpoints predate D49's observation change and cannot execute. |
| 8 | **PPO** | Ours. Built (D48). Same: no runnable checkpoint. |
| 9 | **Recurrent PPO (LSTM)** | Ours. Built (D48). The only RL family with runnable checkpoints; measured below. |
| — | **Greedy static, truth-fed** | Reference line: D14's camper, which knew where the pulses were. Not a scheduler. |
| — | **Pulse-capture oracle** | Ceiling **for interception ratio only**. Not a baseline — a reference line. |

Rungs 2 and 3 were the same policy until D43: "step to the next band in order, each for its native
dwell" *is* `constants.dwell_schedule()`, and it reproduced the sweep's row to four decimal places
because it was not a second measurement (D36). Rung 2 is now equal-airtime, which is the
expressible form of D36's "uniform dwell length" — an action is a band and its length is frozen
(D3, D16), so airtime is the only thing left to make uniform.

**A reference line is not a competitor.** Both lines below the dash read the truth grid. The
pulse-capture oracle is a ceiling for interception ratio and **not** for intercept time: measured,
it loses censored intercept time to plain round-robin on 80.7% of episodes, because it is greedy on
illuminations per slot. Each column has a different optimum (D14), so there is no single ceiling
row and this table does not print one.

Beating round-robin is the minimum. **Beating Apfeld is the claim worth making** — stated
precisely, our adaptation of Apfeld's own no-tracking variant on a binary-detection receiver, at
the parameters in `baselines.ApfeldParams` (D44). And on the first run the harder bar turned out
to be rung 5 (below).

### The target, quantified — measured 2026-09-04

`python -m rfenv.compare --seeds 3 --sampled 10`, 47 stare replays + 10 sampled scenarios × 3
seeds = **1,539 episodes**, at the frozen γ = −111 dB, P<sub>fa</sub> = 1.3499e−3, reward `hit_z`.
Artefacts in `runs/baselines/`; full table with interquartile ranges in `comparison.md`. Recorded
as **D46**.

| # | scheduler | interception ratio | censored intercept time (s) | emitter coverage | beats round-robin on **both** |
|---|---|---|---|---|---|
| 1 | random | 0.0669 | 4.16 | 0.858 | 42.1% |
| 2 | round-robin (equal airtime) | 0.0605 | 4.18 | 0.865 | — |
| 3 | Turing reference sweep | 0.0805 | 3.74 | 0.864 | 51.5% |
| 4 | greedy camper (observation-fed) | 0.2088 | 9.67 | 0.497 | 1.8% |
| 5 | **recency / activity** | **0.1104** | **3.20** | **0.897** | **70.2%** |
| 6a | Apfeld: Active RFs | 0.1320 | 4.32 | 0.860 | 53.2% |
| 6 | Apfeld adaptive (no tracking) | 0.2455 | 14.86 | 0.367 | 4.1% |
| — | camper, truth-fed (D14's) | 0.5680 | 15.71 | 0.296 | 7.0% |
| — | pulse-capture oracle | 0.6579 | 8.01 | 0.691 | 19.3% |

"Beats round-robin on both" is paired per episode — same scenario, same seed, same truth grid —
because a mean can clear a mean while losing most scenarios.

### Rung 9 (RL) joins the table — measured 2026-09-09

> **SUPERSEDED 2026-09-09, on both counts — do not quote these rows.** Two defects were found and
> fixed after this run, and each invalidates it independently. **(1) The reward was wrong** (D52):
> `reward_balance`'s exploration term used a staleness that was the inverse of the observation's,
> so it paid most for revisiting the band just left — every checkpoint below was trained against a
> reward that rewarded camping. **(2) Inference was wrong** (D54): every row was produced by taking
> the policy's argmax, and the policy is broad, not collapsed — measured mean entropy 2.369 against
> ln 36 = 3.584, modal band holding 0.206 of the mass. Sampling the same checkpoints takes them
> from 2 distinct bands to 31 and coverage from 0.247 to 0.603. **The "camping" characterised below
> is an artefact of the argmax, not a learned policy.** The rows are kept because the heuristic
> reproduction that licensed them is still sound and because the corrected run has to be compared
> against something. They are re-run once a checkpoint trained on the corrected reward exists.
>
> **A third defect was found afterwards and makes the rows unrunnable as well as unquotable
> (D55).** Two of the three per-band observation blocks lived in the bottom tenth of their declared
> range: `visit_density` sums to 1 across bands by construction, pinning its mean at 1/36 = 0.0278
> for every scheduler that will ever run, and `staleness` was bimodal at mean 0.070 with everything
> unvisited piled on the 1.0 ceiling. Rung 5 -- the bar in this very table -- already corrected the
> second by hand, and its docstring records that skipping that correction collapses it into rung 4.
> `visit_density` now reads in fair shares and `staleness` in reference sweeps, `camp_time` is gone,
> and the vector went to **146** wide at D55 (**183** today, after D67 appended the hit-streak
> blocks; the figure below is D55's state, not the current width). Rung 5's ranking is unchanged (verified on 1,408 of 1,408 steps),
> so **every heuristic and reference row in the table above still stands**. Every RL checkpoint,
> however, is now unloadable.


`python -m rfenv.compare --seeds 3 --sampled 10 --figures --out runs/baselines`, run
2026-09-09T20:30:50Z at the same 57 scenarios × 3 seeds, reward `reward_balance`, 13 buildable
rungs = **2,223 episodes**. **Every heuristic row and both reference lines above reproduce
exactly**, `recency`'s 70.2% included — that reproduction is what licenses the new rows.

| # | scheduler | interception ratio | censored intercept time (s) | emitter coverage | beats round-robin on **both** |
|---|---|---|---|---|---|
| 9a | Recurrent PPO, 100k | 0.1251 | 19.16 | 0.125 | 1.8% |
| 9b | Recurrent PPO, 200k | 0.2265 | 16.26 | 0.252 | **0.0%** |
| 9c | Recurrent PPO, 300k | 0.1695 | 16.26 | 0.248 | 1.8% |
| 9d | Recurrent PPO, 400k | 0.2052 | 15.49 | 0.282 | **0.0%** |

**Rungs 7 and 8 have no row**, and their absence is a result in itself: every DQN and PPO
checkpoint was trained against a narrower observation vector and cannot run at all (D49). What is
in the table is rung 9 only.

**The RL rungs beat rung 5 on interception ratio and lose the comparison anyway.** 0.2265 against
`recency`'s 0.1104, winning that column on 78.9% of paired episodes — while censored intercept time
is four to five times worse and coverage is a third. Set those rows beside rung 4 (0.2088 / 9.67 /
0.497 / 1.8%) and rung 6 (0.2455 / 14.86 / 0.367 / 4.1%) and the profile is the same one: the
rows *look* like the camping exploit rung 4 exists to demonstrate (D14).

**That reading is withdrawn — see the note below and D54.** What produced this profile was taking
the argmax of a policy that had not collapsed, trained against a reward whose exploration term was
inverted (D52). Rung 4 was put in the ladder to show a single metric can be gamed; whether an agent
games it here is now **unmeasured**, because no rung 9 row in this table was produced under
conditions that could answer it.

This is why §4 prints the three metrics together and never one alone. An RL row reported on
interception ratio by itself would read as a win over every heuristic in the table.

**That qualification has since been settled, and it goes the other way (D54).** Every number
above was produced with `deterministic=True` at inference. Reading the action distribution directly
off `lstm_gamma997` over a full seed-0 episode gives mean entropy **2.369** against `ln 36 = 3.584`
and a mean max-probability of **0.206** — the policy is broad, and its mode is merely *sticky*
(argmax on one band for 580 of 586 steps). Sampled instead of argmaxed, the same checkpoint at
seeds 0 and 1 visits **31 of 36 bands** and scores coverage **0.603 / 0.714** against the argmax's
**0.247 / 0.041**.

So the paragraph above is withdrawn as a claim about what the agent learned. It stands as a claim
about what the argmax of an under-trained broad policy does. Both adapters now default to sampling,
rung 7 excepted (a DQN's greedy action *is* its policy) — see D54 for the full reasoning and for
why torch's generator is now seeded per rung.

**What the ladder says, and it is not what D14 predicted.**

1. **The tension is real and belongs to the truth-fed camper.** Ratio 0.568 against round-robin's
   0.061; intercept time 15.71 s against 4.18 s. That reproduces D14's amendment (57.4% / 23.78 s
   / 30.4% coverage). Intercept times are lower across the whole table than D14's because D27
   measures the delay from `on_e` and not from t = 0 — a definition change, not a disagreement.
2. **A camper that cannot see truth is a much weaker camper**: 0.209 against 0.568. Illumination
   density is truth-side (D29), and the most reliably-occupied band is not the busiest one. D14's
   57.4% was never reachable by a deployable scheduler, and the ladder now carries both rows.
3. **The period-estimation half of Apfeld costs more than it buys here** — rung 6 against its own
   ablation 6a: 1.9× the ratio for 3.4× the intercept time and less than half the coverage. A 30 s
   episode and a binary detection series are both ours, not the paper's; D45 has the reasoning.
4. **The bar for the RL rung is rung 5.** A one-line index policy Pareto-dominates the floor on
   70% of episodes and beats the Turing sweep on all three metrics. **The Pareto target is now:
   hold `recency`'s 3.20 s while multiplying its 0.110 interception ratio toward the camper's
   0.209 and the oracle's 0.658.**

   **D56 claimed no candidate encodes that target. It was measured wrong and is withdrawn.**
   It scored a hand-written `step % N_BANDS` sweep instead of rung 2 (`EQUAL_AIRTIME_CYCLE`, D43);
   against the real rung, `reward_balance` separates rung 5 from rung 2 by **+59.0 +/- 19.4 on 8/8
   seeds**. Every candidate is now screened before training (**D62**): rungs 2, 4, 5 and 6a under
   each, 8 seeds, paired per seed. **`reward_balance` is the only one of the original six that
   passed** -- **`hit_z` and `hit_y` both failed**, ranking rung 4 above every sweeping policy
   (D14's tension in reward form), and were **retired from `REWARDS` entirely** 2026-09-10 as a
   consequence. `reward_balance_improved`, added the same day, also passed. **D47 has since run
   (D68) and, once resolved with matched seeds, selected `reward_balance`** -- 81.9% vs
   `reward_balance_improved`'s 64.3% paired-both against round-robin, a 17.6 pp gap, decisively
   outside the rule's own 5 pp no-selection margin. See §"Choosing between reward
   candidates (D47)" above for the full table.

### The development split (D60, D61)

**Until 2026-09-10 RL training sampled `EmitterPool.from_train()` -- every emitter in all 47
development configs -- while this section's scenarios are those same 47 replays plus scenarios
sampled from that same pool.** The heuristic rungs do not train, so the asymmetry ran one way. Any
RL margin measured before that date includes an advantage no fielded scheduler would have.

The 47 are now split **35 training / 12 validation** by a rule fixed in `rfenv/split.py` *before*
anyone looked at which configs landed where, sampling systematically along detectable-emitter count
so both halves span the difficulty range (training mean 40.9, validation 40.2). The pool is rebuilt
from the training half and shares **zero emitters** with the validation half -- splitting the
config list alone is not enough, because the pool is assembled from the configs.

**This section still reports on all 47, deliberately.** The headline belongs on the whole
development set; it is *training* that must not see it. Checkpoint selection happens on the 12
(D61: highest `P(dominates rung 5) - P(dominated by rung 5)`, fixed before the run), and any RL row
added here must state how many configurations were tried to produce it.

**The 2,223-episode acceptance run of 2026-09-10 is not in the table above and will not be.** Its
rung 9 rows -- 9c dominating rung 5 on 48.0% of episodes against 13.5%, 9b on 43.9% against 4.7% --
are arithmetically sound and were produced under the leak. Artefacts in
`runs/acceptance_2026-09-10/`. **This is the comparison D60 said would supersede it, re-measured
after a retrain on the training half — see below.**

### The first clean RL rows (D64, D65) — measured 2026-09-10

Two runs, `reward_balance` (rung 10, the control) and `reward_balance_improved` (rung 11, the
treatment) -- identical seed, hyperparameters and D60 split, differing only in the reward. Each
trained 400k steps, checkpointed every 100k, and each checkpoint selected on the validation half
by D61's pre-registered rule before anything below was measured:

| run | selected checkpoint | dominates rung 5 | dominated by rung 5 | net dominance (D61, validation) |
|---|---|---|---|---|
| control (`reward_balance`) | 400k (`clean_lstm_s4`, rung 10d) | 47.2% | 11.1% | **+36.1%** |
| treatment (`reward_balance_improved`) | 200k (`lstm_balance_improved_s2`, rung 11b) | 36.1% | 11.1% | +25.0% |

`python -m rfenv.compare --rungs round_robin,recency,lstm_balance_clean_400k,lstm_balance_improved_200k_400k --seeds 3 --sampled 10 --figures --out runs/clean_paired_comparison`
— 4 rungs x 57 scenarios x 3 seeds = **684 episodes**:

| # | scheduler | interception ratio | censored intercept time (s) | emitter coverage | beats recency on **both** |
|---|---|---|---|---|---|
| 2 | round_robin | 0.060 | 4.18 | 0.865 | 3.5% |
| 5 | recency | 0.111 | 3.34 | 0.887 | -- |
| 10d | Recurrent PPO, clean (`reward_balance`, 400k) | 0.123 | 3.87 | 0.861 | 22.8% |
| 11b | Recurrent PPO, clean (`reward_balance_improved`, 200k) | 0.143 | 3.51 | 0.860 | **31.0%** |

**Both checkpoints clear the bar** -- both beat `recency` on the joint metric at a rate the floor
(3.5%) does not come close to. **On this headline measure the treatment is ahead of the control**
(31.0% against 22.8%, and ahead on both individual paired columns: ratio 83.0% against 73.7%, cTTI
39.8% against 31.6% -- see `runs/clean_paired_comparison/`), reversing the D61 validation ranking
above.

**Not read as "the improved reward wins."** One training seed per arm: the control's own four
checkpoints swing from +25.0% to +30.6% to +0.0% to +36.1% net dominance across a single run
(D64) -- a larger swing than the 8.2-point headline gap between the two arms -- and the two
selected checkpoints sit at different step counts (400k vs 200k), so the comparison is each
reward's best snapshot rather than a controlled same-point measurement. It is weak evidence in the
direction argued when `reward_balance_improved` was called "unproven rather than rejected" by
D62's screen (which measures discrimination between six fixed heuristics, not what a learner's
gradient consumes), not a settled result. **Matched seed counts per arm is what would settle it**
(D65).

**Neither row above is registered as *the* headline RL number yet** -- that decision (which reward,
if either, becomes the one carried forward past this comparison) has not been made, and both are
reported here as measured rather than one being promoted over the other by this document.

### The first result on the widened "v2" observation (D72, D73) — measured 2026-09-18

D71 added PulseWidth/AoA/per-band amplitude as an opt-in 326-wide "v2" layout; D72 widened it
again to 362 the same day, adding `pulse_count` (gated illumination count). The seed-2 pair D71
started training on 326-wide "v2" never finished (385,024/400,000 steps) and was made permanently
unloadable by D72's width change -- this is the first pair to finish training on the *widened*
"v2", not a continuation of that one.

Same pairing discipline as D64/D65: `reward_balance` (rung 20d, control) against
`reward_balance_improved_v2` (rung 21a, treatment), identical split/hyperparameters/seed (2),
`ScanEnv(obs_version="v2")`. The treatment run was interrupted by two laptop crashes and resumed
both times from its own last snapshot (`RecurrentPPO.load` + `learn(reset_num_timesteps=False)`),
recorded in its manifest.

`python -m rfenv.compare --rungs round_robin,recency,lstm_balance_v2_d72_seed2_400k,lstm_balance_improved_v2_d72_seed2_400k --seeds 3 --sampled 10 --figures --obs-version v2 --out runs/d72_paired_comparison`
-- 4 rungs x 57 scenarios x 3 seeds = **684 episodes**:

| # | scheduler | interception ratio | censored intercept time (s) | emitter coverage | beats recency on **both** |
|---|---|---|---|---|---|
| 2 | round_robin | 0.060 | 4.18 | 0.865 | 3.5% |
| 5 | recency | 0.111 | 3.34 | 0.887 | -- |
| 20d | Recurrent PPO, D72 v2 (`reward_balance`, 400k) | 0.111 | 3.30 | 0.902 | 36.3% |
| 21a | Recurrent PPO, D72 v2 (`reward_balance_improved_v2`, 400k) | 0.136 | 2.82 | 0.912 | **45.6%** |

**Both clear the bar; the treatment is ahead of the control on every column** here too, the same
direction D65 found on "v1". **Not directly comparable to the D64/D65 numbers above** -- observation
width, reward formula (`_DENSITY_SHRINKAGE_V2 = 0.75` against 0.5) and the sampled-scenario draws
all differ at once. **Not read as settled**: one training seed per arm, same caveat D65 gave and
did not resolve. Full account: D73.

**Reproduce the full ladder with:**

```bash
python -m rfenv.compare --seeds 3 --sampled 10 --figures --out runs/baselines
```

Both paired tables are printed: against `round_robin` (the floor, beating it is the minimum) and
against `recency` (**the bar** -- a scheduler that clears the floor and not the bar has not beaten
the ladder). Until 2026-09-10 only the floor comparison was computed, so every RL claim in this
repository had been measured against the wrong reference.

### The full ladder with every live RL rung — measured 2026-09-11 (D71)

**This supersedes every RL row above as the current standing.** All prior RL numbers in this
section compared one run against a *different* run; this is the first in which every loadable
checkpoint was scored against the whole ladder in one run, one scenario set, one set of seeds.

`python -m rfenv.compare --seeds 3 --sampled 10 --figures --out runs/final_2026-09-11`
— 33 rungs x 57 scenarios x 3 seeds = **5,643 episodes**. 24 of 57 registered rungs skipped by
`compare.py`'s guard: every DQN/PPO checkpoint and every 146/147-wide LSTM predates D67 and is
permanently unloadable.

| rung | scheduler | interception ratio | censored intercept time (s) | emitter coverage | intercept rate (/s) |
|---|---|---|---|---|---|
| **17c** | **Recurrent PPO, `reward_balance`, seed 2, 300k** | **0.1304** | **3.04** | **0.9019** | **1.191** |
| 5 | `recency` — **the bar** | 0.1105 | 3.34 | 0.8874 | 1.153 |
| 6a | `apfeld_active_rfs` | 0.1320 | 4.32 | 0.8602 | 1.135 |
| 6 | `apfeld` | 0.2455 | 14.86 | 0.3672 | 0.350 |
| 4 | `camper` | 0.2088 | 9.67 | 0.4971 | 0.660 |
| 3 | `turing_sweep` | 0.0805 | 3.74 | 0.8643 | 1.166 |
| 2 | `round_robin` — the floor | 0.0605 | 4.18 | 0.8650 | 1.118 |
| 1 | `random` | 0.0669 | 4.16 | 0.8583 | 1.117 |
| — | `oracle_pulse` *(reference line)* | 0.6579 | 8.01 | 0.6912 | 0.813 |
| — | `camper_oracle` *(reference line)* | 0.5680 | 15.71 | 0.2960 | 0.295 |

Paired per episode (n = 171): **rung 17c beats `round_robin` on both metrics 81.9% of the time and
`recency` — the bar — 54.4%.** Across all 24 live RL checkpoints the joint column ranges 18.7-54.4%
against the bar (median 35.4%); **23 of 24 beat `apfeld_active_rfs`** at 23.4%, and the weakest
still beats `turing_sweep` (9.9%), `apfeld` (2.9%) and `camper` (0.6%).

**Rung 17c Pareto-dominates rung 5 on all four metrics in the means** — higher ratio, lower
intercept time, higher coverage, higher intercept rate. No rung before it did that.

**Rung 17c is the checkpoint D61's rule selected, on validation data, before this run existed.**
It then scored highest of all 24 here. The rule never saw the data that confirmed it — which is
the whole reason the criterion was fixed in `rfenv/selection.py` in advance (D39's discipline
applied to checkpoints rather than gate thresholds).

**Read the seed spread before quoting any single row.** The 24 checkpoints span 36 percentage
points on the joint column. The control/treatment reward gap argued in D65 and D68 is a few points
inside that spread, so **no single-checkpoint comparison between the two rewards is trustworthy at
this sample size** (D71).

Full account: **D71**. Artefacts: `runs/final_2026-09-11/`.

### Comparison figures

`--figures` writes four kinds of picture into the run directory, all drawn from the artefacts so
they cannot disagree with the table (`rfenv/render.py`):

- **`pareto.png`** — interception ratio against censored intercept time, one marker per rung,
  numbered by rung, area proportional to coverage, reference lines hollow, names in a legend
  beside the axes. D14's finding as a picture: better is up and to the left, and nothing trivial
  is there. The names are off the markers deliberately — five rungs cluster inside one second and
  0.08 of ratio, and labelling them in place makes the one corner a reader came for illegible.
  **Markers are medians and the whiskers are the interquartile range (D58)** — not means, which is
  how the rest of this document reports the same metrics. A marker cannot show a distribution, and
  rung 4's is wide enough to break the figure: its interception ratio has mean 0.2065 against
  median 0.1257, so drawn as means it floated above every adaptive rung *and* stretched the y-axis
  for all of them, since the limits come from the maximum. The table below keeps means because it
  has room to print an interval beside each one; the figure does not, so it collapses differently
  on purpose. The camper's whisker being four times any other rung's is the point of the change.
- **`timeline_<config>.png`** — one row per rung: band against time over the scenario's occupancy,
  with the tuning path, a tick per dwell start (so a 100 ms look reads as one decision, D31/D35),
  declared hits, and a per-emitter strip on the right that turns solid at first intercept. This is
  the picture that makes round-robin and the camper legible as opposite strategies.
- **`discovery_<config>.png`** — distinct emitters intercepted against time, one line per rung,
  with `|E|` as the ceiling. Coverage is one number; this is the path it took.
- The waterfall (§8 artefact 4) is unchanged and still drawn by `render.waterfall`.

Fixed scenarios — `config_81` (1 detectable emitter), `config_2` (19) and `config_921` (72) — so
two runs' figures can be laid side by side.

---

## 6. Validation gates — pass before any scheduler number is quoted

| # | Gate | Why it matters |
|---|---|---|
| **1** | **Out-of-sample prediction.** Build truth from **stare only**, replay Turing's scan schedule, compare predicted detections against the **actual scan recordings** — data never used in construction (D17). Scored **per dwell, against the raw scan ToA stream** (D37). **Reported, not gated:** D37 fixed the convention and left the threshold undecided, so `validate.py` prints `MEASURED` and gate 1 cannot fail; inventing a threshold here would be the failure the gate machinery exists to prevent (D39). **Measured 2026-09-04** over 47 train configs, 23,594 dwells, seed 0, at the frozen γ = −111: **accuracy 0.8585, precision 0.8819, recall 0.6932, MCC 0.6854, per-band r 0.935**, against a 0.3540 base rate (TP 5790 / FP 775 / TN 14466 / FN 2563). These supersede the withdrawn 2026-09-01 scratch figures (86.19% / 87.87% / 71.14% / 0.694 / 0.940 at γ = −110, convention unrecorded) and land inside the audit's 2026-09-03 re-run range. Known limitation, not a defect: band 0 (250 MHz) is **58.97%** occupied in the recordings and 0.00% predicted, because stare cannot see below 500 MHz (D10). | The only gate that is a genuine prediction rather than a fit. If one gate is run, run this one. This is also why no physics signal model is fitted to these same recordings (D25). |
| **2** | **Per-band structure.** Grid built from the scan recording, replayed on the schedule that produced it, thresholded not at all — the non-empty **dwell rate**, per band and in aggregate. Not a γ calibration (D23); a pipeline self-consistency test. Both sides count the same dwells, so there is no sampling noise and the criteria are mechanism bounds: **2a** aggregate \|Δ\| ≤ 0.5 pp, **2b** max per-band \|Δ\| ≤ 1.0 pp, **2c** replayed ≤ recorded (bucketing can drop a pulse at a boundary, never invent one). Fixed in advance in `validate.py::GATES` (D39). **Measured 2026-09-04: replayed 0.35403 against recorded 0.35403 — exact, 0.000 pp aggregate and 0.000 pp on every band.** The previously recorded "35.403% replayed against 35.700% recorded" is **withdrawn**: the 0.3 pp residual was a band-blind recorded side, not slot quantisation (D41). | An aggregate can match while the structure is wrong. |
| **3** | **Theory.** A controlled periodic case against Köksal's closed forms (`docs/reference/scheduling/optimumsearch.pdf` — Eq. (3.8) at body p.18, Table 6-1 at body p.72; PDF pp.33 and 87). Case fixed in advance: one synthetic emitter in band 0 at γ + 5σ, τ_emit 0.50 s, T_emit 3.00 s against the reference sweep's τ_rcv 0.10 s, T_rcv 2.15 s, every one of the 60 whole-slot phases. **3a** \|P₁₂(T₁)_env − discrete reference\| ≤ 0.01; **3b** max first intercept over phase ≥ 10.75 s. **`P₁₂(T) = 1 − [1−P₁₂(T₁)]^(T/T₁)` is reported, never gated** — it assumes successive receiver periods are independent, which is false for a deterministic periodic pair, and gating on it would fail a correct environment (D40). **An environment check, never a scheduler claim:** Köksal assumes prior emitter knowledge and D20 refuses it; the pre-knowledge here is the analyst's, never the agent's. **Measured 2026-09-04: P₁₂(T₁) 0.18333 against reference 0.18333 (\|Δ\| 0.0000); max first intercept 12.90 s ≥ 10.75 s. PASS.** | Independent of the dataset entirely. |
| **4** | **Extremes.** `config_81` (2 transmitters, 1 detectable) and `config_921` (99, 72 detectable), stare replays under `dwell_schedule()`. "Sensibly" is not a predicate, so this is **12 structural assertions** with no tolerances: `first_e ≥ on_e` (D28), every metric inside its definitional range, no NaN or ∞ (censoring is mandatory), 600 slots and 300–600 steps (D35), and the artefacts reproducing `env.episode_metrics()` (D38) — five per config — plus two orderings over the pair. Each restates a decision or a definition; they hold or the code is wrong. **Measured 2026-09-04: 12/12. PASS.** | Catches failures that averages hide. |

**Status: all four ran for the first time on 2026-09-04** — `python -m rfenv.validate`, 47 train
configs, seed 0, artefacts under `runs/validation/`. **The three gated checks pass; gate 1 is
MEASURED.** Criteria were fixed in `rfenv/validate.py::GATES` before the run and are asserted
against D39 by a test, so none of them was chosen after the measurement was visible. The run also
re-measured the operating point and it matches D33 exactly: Pd 0.85058, Pfa 1.3499e−3,
sensitivity −107.155 dB over 11,710 occupied reference-sweep cells.

On pass, **freeze** everything in `rfenv/constants.py`: band geometry, slot clock, native dwell
lengths, truth pipeline, `N₀`, `σ`, γ, metric definitions, and the scenario sampling
distribution (D25).

**The freeze was taken on 2026-09-04 (D42).** It is enforced by `tests/test_freeze.py`, which pins
every value as a literal plus a SHA-256 digest over the whole list — until it existed the suite
read the constants symbolically and would have stayed green through a change to γ or the band
geometry. **D42 also records what the gates cannot detect**, established by fault injection: gate
2's 0.000 pp is algebraically forced and passes with a wrong band half-width, gate 3's reference is
co-parameterised with the environment, and **no gate covers the ±500 MHz half-width** — its sole
evidence is D3's 99.9851% in-band measurement. Read D42 before quoting any gate figure as
stronger than it is.

---

## 7. Protocol

1. **Develop and tune** on the 47 train scenarios only.
2. **Validate** the environment (gates 1–4). No scheduler result is quoted before this.
3. **Freeze** the environment and publish the receiver ROC. ✅ **taken 2026-09-04** (D42).
4. **Compare** all schedulers on identical scenarios and seeds; report the full
   scheduler-level table, never a single metric. ✅ **first run 2026-09-04** — `rfenv/compare.py`,
   1,539 episodes, §5 above and D46. Rung 7 (RL) is the only one missing.
5. **Ablate** — the baseline ladder is the ablation: it shows which component earns the gain.
   Rungs 6a/6 are the first one it has produced (D45).
6. **Test once.** The 45 held-out pairs are touched a single time, at the end, after the system
   is frozen (D8). Record that use. **Not yet done** — nothing in §5 has touched them.

### Reporting rules

- Report **distributions and per-scenario results**, not just a grand mean — scenario difficulty
  spans 2 to 99 emitters.
- Report **repeated-run statistics** (multiple seeds) with spread, not a single run.
- **Never report bare accuracy.** With sparse occupancy, "predict nothing" scores well and is
  operationally useless.
- State the operating point (γ, P<sub>fa</sub>) alongside any scheduler table. γ, σ,
  P<sub>fa</sub> and sensitivity are frozen and data-independent; **P<sub>d</sub> is not** — it
  depends on which grids the reference-sweep population is taken over, so name them. The
  validation run's 0.85058 (47 scan-replay grids) and the comparison run's 0.8421 (57 stare and
  sampled grids) are the same rule applied to different worlds (D33, D46).
- **A reference line is labelled as one in every table it appears in.** The oracles read truth;
  read as competitors they invert the reading of the whole comparison (§5).

---

## 8. Output artefacts the environment must emit

Evaluation is only possible if the environment logs these (see `ENVIRONMENT_SPEC.md` §Outputs):

1. **Episode log** — per slot: time, band chosen, dwell length, declared hit `Y`, true occupancy
   `Z`, pulse count, peak level. `rfenv/metrics.py`, `episode_log.csv`.
2. **Emitter table** — per emitter: detectable activity interval (D27), first/last intercept
   slot, intercept count, bands seen in. `emitter_table.csv`.
3. **Run header** — the episode's scalars: scenario, scheduler, seed, reward, γ, σ, **total
   illuminations** and **total reward** (D38). Added 2026-09-04: §4 is *not* computable from
   artefacts 1 and 2 alone, as this section previously claimed. Interception ratio's
   denominator is grid-level and appears in neither table — the log carries only the numerator —
   and average reward appears in neither either. Both are scalars, so they belong in a header
   rather than in a wider log. **Everything in §4 is computable from artefacts 1–3.** `run.json`.
4. **Waterfall render** — the 36×600 grid as a frequency-vs-time heatmap with the scheduler's
   path and hits overlaid. The same picture drawn from the raw Turing recording should match:
   the standard ESM operator view, and the fastest way to see that the environment is sane.
   `rfenv/render.py`. Note when reading one: a **scan** replay's content lies along Turing's own
   sweep (D36), so a sweeping scheduler's path tracing the bright cells there means nothing.
5. **`metrics.json`** — the three families, one file per run, with the operating point stamped
   on it (§7).
6. **Scheduler-comparison figures** — the Pareto plot, the per-rung timeline and the discovery
   curve, all drawn from artefacts 1–3 so a picture cannot disagree with the table beside it.
   `rfenv/render.py`; written by `python -m rfenv.compare --figures`. §5 §Comparison figures says
   what each shows.

---

*Provenance: every measured figure quoted here traces to a command run against the Turing HDF5
files and is recorded with its evidence in `docs/project/DECISIONS.md`. Where this file and
`docs/project/SIH26055_PROBLEM_STATEMENT.md` disagree, the PS wins.*
