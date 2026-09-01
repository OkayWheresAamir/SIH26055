# SIH26055 — Smart Scan Strategy for Electronic Warfare

**The official problem statement. This is the most authoritative document in the repository
about *what we are being asked to build*** — above `PROJECT_ARCHITECTURE.md`, which is our
interpretation of it, and above every PDF in `docs/`.

Transcribed verbatim from <https://sih.gov.in/sih2026PS> on 2026-08-28 by the repository owner.
Two mojibake artefacts from the portal's encoding (`receiverâ€™s`) have been corrected to
`receiver's`; nothing else is altered. Where this file and any other document disagree about
the requirement, **this file wins**.

| Field | Value |
|---|---|
| Problem Statement ID | **26055** |
| Title | Smart Scan strategy for Electronic Warfare |
| Organization | DRDO |
| Department | Department of Defence Production / IDEX |
| Category | Software |
| Theme | Robotics and Drones |
| Dataset Link | JC Wise, Radar emitter Database, 2024 — `huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset` |

---

## Description

Development of Smart Scan Strategy for Electronic Warfare in the absence of prior reliable
intelligence of emitters and their operating characteristics.

### Background

Detection of hostile communication or radar signals starts with search / scan of a wide
frequency spectrum which covers relevant emitters. Sensors with typically high sensitivity but
with at least an order lower instantaneous bandwidth compared to overall bandwidth of the
system are used to maintain surveillance over the entire spectrum. This requires a receiver /
receivers to sweep over frequency bands. Hitherto strategies based on pre mission data / prior
data (Open loop) are used. Usually the first priority is to rapidly sweep the entire band with
the best speed possible. Open loop strategies focus only on this requirement and may lose time
to nonthreatening emitters by not giving time to new or threatening ones.

### Detailed Description

This problem statement focusses on development of Smart Scan Strategy for Electronic Warfare.
Interception of signals is a two dimensional search problem since it involves adjusting
receiver's frequency at correct time. This includes building up figures of merit for
interception performance such as probability of detection, probability of false alarm,
sensitivity, Avg intercept rate, Avg Reward / cost function, percentage of correct predictions
and average intercept time error. A system model for the receiver needs to be developed with
measurements obtained from a simulated RF environment which has truth information on status of
emitters in each band and at each time slot. The frequency spectrum for own receiver consists
of many bands.

The status of environment for each frequency band at each time step can be recorded as a
transmission or a non-transmission. The model should enable prediction of intercept time and
interception ratio of a scanning receiver against spatially scanning and frequency agile
emitters. Development of a robust scheduler using machine learning to minimize intercept time
and ensure a high interception rate is the primary objective of the strategy. The model should
then be trained based on hits and misses. Further, approaches to intercept a periodic scan
receiver optimally should be outlined. Algorithms and techniques for the same need to be
developed.

### Expected Solution

Machine learning based Electronic Support receiver scheduler software

---

## What this settles (and what it does not)

Recorded 2026-08-28, when the PS first entered the repository. These are readings of the text
above, not new facts about the data.

### Settled by the PS

1. **The environment is simulated and carries per-band, per-time-slot truth.** *"A system model
   for the receiver needs to be developed with measurements obtained from a simulated RF
   environment which has truth information on status of emitters in each band and at each time
   slot."* This is a direct requirement, and it is the construction we had already converged on.
   The Turing recordings are an input to building and validating that environment, not the
   environment itself.
2. **Environment state is binary occupancy per (band, time slot).** *"The status of environment
   for each frequency band at each time step can be recorded as a transmission or a
   non-transmission."*
3. **Training signal is hits and misses.** *"The model should then be trained based on hits and
   misses."*
4. **The metric set is given, not ours to choose:** probability of detection, probability of
   false alarm, sensitivity, average intercept rate, average reward / cost function, percentage
   of correct predictions, average intercept time error. Plus intercept time and interception
   ratio, named separately in the same paragraph.
5. **Both emitter behaviours must be covered:** *"spatially scanning and frequency agile
   emitters."* Not one or the other.
6. **The receiver's instantaneous bandwidth is at least an order below the total.** Turing's
   scan receiver is a 1000 MHz window over an 18000 MHz range — a factor of 18. Consistent.
7. **The primary objective is minimise intercept time and maximise interception rate.** Not
   classification accuracy, not deinterleaving.
8. **The Turing dataset is named in the PS itself**, so grounding the work in it is expected
   rather than a choice we made.

### Left open by the PS

- **Time-slot duration is unspecified.** "Each time step" is not quantified anywhere.
- **Band count and layout are unspecified.** "The frequency spectrum for own receiver consists
  of many bands" — how many, how wide, whether they overlap, is ours to decide.
- **How a "transmission" in a band-slot is determined** from emitter geometry and power. The PS
  asserts the binary status exists; it does not say how to compute it.
- **Which ML method.** "Machine learning based" and "using machine learning" is as specific as
  it gets. RL is a reasonable reading of "trained based on hits and misses" but is not named.
- **How false alarms arise.** `Pfa` is a required metric, but a noiseless occupancy model
  produces no false alarms. Something must generate them, and the PS does not say what.
- **Ambiguous sentence, flagged for domain review:** *"approaches to intercept a periodic scan
  receiver optimally should be outlined."* "Receiver" here is most likely the hostile
  *emitter*'s periodic scan (the scan-on-scan problem), since one does not intercept a
  receiver. Treated as UNRESOLVED until a domain reviewer confirms.
