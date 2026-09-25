# The RF Environment, Explained

**The simulated RF world we built for SIH26055 — how it was made from real recordings, how it
generates new scenarios, and how we validated it.** Companion to *NARADA: The Scanning Agent,
Explained*.

---

## 1. Why a simulated environment

A receiver can listen to only one narrow band at a time, and must decide many times a second which
band to check next. To develop a scheduler for that and compare it fairly against others, we need
three things the real world does not give us:

- **Repeatability** — every scheduler must face the identical situation.
- **Ground truth** — to score a scheduler we must know what was transmitting, including in the bands
  it never checked.
- **Speed** — a 30-second mission runs in a fraction of a second.

The problem statement asks for exactly this: *"a simulated RF environment which has truth
information on status of emitters in each band and at each time slot."*

---

## 2. The data

The Turing Synthetic Radar Dataset (TSRD), the dataset named in the problem statement: **184 files,
2.2 GB** of recordings of a receiver listening to radar environments. Each scenario comes as a
pair — a **scan** recording (receiver sweeping) and a **stare** recording (receiver parked and
listening). Each is a list of detected pulses with time, frequency, strength, pulse width and
bearing.

Of 92 scenario pairs, **47** are used for development and **45 are kept untouched** for a clean
final test. Inside the 47, a **35 / 12** split separates the scenarios the environment builds from
and the ones kept for checking, with **zero emitters shared** between them.

---

## 3. A generative RF environment

**Step 1 — Extract and group.** Every recorded pulse is extracted and grouped by the emitter that
produced it.

**Step 2 — Build a reusable emitter pool.** Each emitter's activity becomes a reusable building
block, giving **3,443 emitter contributions**.

**Step 3 — Sample new scenarios.** Emitters are drawn from the pool, with their timing, and dropped
together into a fresh 30-second scenario. Every episode is new — the environment can produce
combinations of emitters and situations the original dataset never contained.

**Step 4 — Fill the grid.** Each scenario becomes a grid of **36 bands × 600 time slots**, 50 ms per
slot, 30 seconds in total.

### The bands

The **36 bands** are the receiver's own, read from each file's metadata and checked against every
file opened: centres 500 MHz apart from 250 MHz to 17,750 MHz, each **±500 MHz** wide, so adjacent
bands overlap by half. We confirmed this against the data itself: **99.985%** of 4.4 million
recorded pulses fall within ±500 MHz of the active band centre.

### What each cell of the grid holds

| Quantity | Meaning |
|---|---|
| Z[b,t] | is any emitter transmitting in band b at slot t |
| S[b,t] | the received signal level |
| C[b,t] | how many pulses illuminate the cell |
| PW[b,t] | pulse width of the loudest pulse in the cell |
| AOA[b,t] | angle of arrival of the loudest pulse in the cell |

This grid is the ground truth: a complete record of everything, everywhere, at every moment. It is
used to **score** a mission, and is never shown to the scheduler. Each emitter's own signal is kept
separately too, so a quiet emitter is only credited as found when *its* signal was strong enough —
not because a louder neighbour happened to share the band.

---

## 4. A realistic receiver

Looking at a band does not reveal the truth — it produces a **noisy measurement**, compared against a
detection threshold:

> Y[b,t] = HIT if signal + noise is above the threshold, otherwise NO HIT

| Receiver parameter | Value |
|---|---|
| Detection threshold | **-111 dBm** |
| Noise spread | 3 dB |
| Probability of detection (P_d) | **0.8506** |
| Probability of false alarm (P_fa) | **0.00135** |
| Sensitivity | **-107.16 dBm** |

So the environment is not noiseless: the receiver occasionally misses a real signal and occasionally
raises a false alarm, just as a real receiver does. A scheduler developed here is developed for a
realistic detector. The scheduler controls **where** the receiver looks; detection quality belongs
to the receiver itself.

---

## 5. The rules of the loop

- **Action.** The scheduler picks one of the 36 bands.
- **Dwell.** The receiver listens for that band's natural dwell — 50 ms for 29 bands, 100 ms for the
  other seven, as in the recordings.
- **Observation.** The scheduler is told HIT or NO HIT, plus what the receiver measured.
- **Repeat.** The clock advances by that dwell; a 30-second mission is **300 to 600 decisions**.

Missions can also run far longer than 30 seconds by stitching fresh scenarios end to end — the basis
for continuous online operation.

---

## 6. What the scheduler sees, and what it is paid

**Seen:** only the receiver's own history — per band, how often looking there paid off, how much
time it has had, how long since it was last checked, and what was last heard there (strength, pulse
width, bearing, pulse count), plus the current band and the mission clock. **Nothing from the truth
grid is ever handed over.** This is the problem statement's *"absence of prior reliable
intelligence"* requirement, built into the code.

**Paid:** one fixed reward formula, computed from the same receiver-side quantities:

> Reward = exploit what pays + explore what is overdue + reward for hits - cost of hogging airtime

The environment can also take a **per-band priority** from the operator, adding a bonus for
discovering emitters on high-threat bands.

---

## 7. Validation

Before any scheduler was evaluated, the environment passed **four validation gates**, each with its
criteria written down in advance:

| Gate | What it checks | Result |
|---|---|---|
| **1. Out-of-sample prediction** | Build truth from **stare** recordings only, replay the receiver's own scan pattern, and predict what the **scan** recordings contain — data never used to build it | **85.85% correct predictions**, MCC 0.685, per-band correlation 0.935 |
| **2. Per-band structure** | Replay the schedule that produced a recording and compare band-by-band activity | **Exact match** — 0.35403 vs 0.35403 |
| **3. Theory** | A controlled periodic emitter against the periodic sweep, versus published closed-form results | **Exact match** — 0.18333 vs 0.18333 |
| **4. Extremes** | The emptiest and busiest scenarios, 12 structural checks | **12 / 12 pass** |

Once validated, every physical constant — band layout, 50 ms clock, dwell lengths, noise, threshold —
was **frozen** and locked by an automated test, so no result can be improved by quietly changing the
world.

---

## 8. How schedulers are scored

Three metrics, always reported together:

| Metric | Meaning | Better |
|---|---|---|
| **Interception ratio** | fraction of all transmitted pulses the receiver was tuned to | higher |
| **Censored intercept time** | average time from an emitter becoming findable to its first detection; an emitter never found is charged the full 30 seconds | lower |
| **Emitter coverage** | fraction of findable emitters detected at all | higher |

Reporting all three together is what stops a scheduler looking good by camping on one busy band: a
camper scores well on interception ratio but badly on intercept time and coverage, and the
environment shows it.

**Fair evaluation.** Every scheduler runs on the identical scenarios with identical noise, and is
compared mission by mission rather than by averages, so scenario difficulty cancels out entirely.

---

## 9. Glossary

| Term | Meaning |
|---|---|
| **Band** | One of 36 frequency ranges the receiver can tune to; ±500 MHz wide, overlapping by half |
| **Slot** | 50 ms of time. A mission is 600 slots |
| **Dwell** | One look at one band — one decision. 50 or 100 ms |
| **Episode / mission** | One 30-second run, starting from zero knowledge |
| **Emitter pool** | Reusable emitter activity extracted from TSRD, sampled into new scenarios |
| **Ground truth** | The complete record of what transmitted where and when; used for scoring only |
| **HIT / NO HIT** | The receiver's declaration after one look |
| **Censored** | Charging a never-found emitter the full mission length |

---

*Built on Gymnasium 1.3 (environment interface), NumPy and h5py 3.16 (TSRD HDF5 access). Figures
are from the project's own runs: the four validation gates, the band-geometry measurement and the
receiver operating point.*
