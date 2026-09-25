# NARADA — Smart Scan Strategy for Electronic Warfare

**Smart India Hackathon 2026 · Problem Statement SIH26055 · DRDO · Team NIRVANA**

*Right place at the right time.* NARADA is a machine-learning scan scheduler for an Electronic
Support receiver. It decides, every 50–100 ms, which frequency band the receiver should listen to
next — learning from its own hits and misses, with no prior intelligence about the emitters it is
looking for.

---

## The problem

> *"Development of Smart Scan Strategy for Electronic Warfare in the absence of prior reliable
> intelligence of emitters and their operating characteristics."* — SIH26055

An ES receiver covers a wide spectrum but can listen to only one narrow band at a time, so it has
to sweep. Today's receivers follow fixed, pre-mission (open-loop) sweep schedules: they treat a band
where nothing ever happens exactly like a band where a new threat just appeared, and they depend on
intelligence gathered before the mission.

The problem statement asks for a scheduler that:

- **minimises intercept time** and **maximises interception rate** — two objectives that pull
  against each other;
- **learns from hits and misses**, in a simulated RF environment with per-band, per-slot truth;
- works **without prior emitter intelligence**;
- handles **spatially scanning and frequency-agile emitters**;
- is measured on P_d, P_fa, sensitivity, intercept rate, reward, prediction accuracy and
  intercept-time error.

Full text: [`docs/project/SIH26055_PROBLEM_STATEMENT.md`](docs/project/SIH26055_PROBLEM_STATEMENT.md).

---

## Our solution

### 1. A generative RF environment (`rfenv/`)

Built from the **Turing Synthetic Radar Dataset** named in the problem statement (184 HDF5 files,
2.2 GB):

- Every recorded pulse is extracted and grouped by emitter into a reusable **emitter pool**
  (3,443 emitter contributions).
- Emitters are sampled from the pool into fresh **30-second scenarios** — a grid of **36 bands ×
  600 slots of 50 ms**, with the receiver's own geometry (±500 MHz bands, half-overlapping).
- A **realistic receiver model** adds noise to every measurement and compares it against a
  detection threshold: γ = −111 dBm, **P_d = 0.8506, P_fa = 0.00135**, sensitivity −107.16 dBm.
- **Four validation gates** checked the environment before any scheduler was scored — including
  out-of-sample prediction of real scan recordings (**85.85% correct**) and exact agreement with
  theoretical closed forms. All physical constants are then frozen.

### 2. An online Recurrent PPO agent (`rfenv/rl/`)

- The scan problem is treated as a **POMDP**: the receiver only ever sees the band it is tuned to.
- **Recurrent PPO with LSTM memory** carries the scan history between decisions, so the agent
  adapts *inside* a mission without changing any weights.
- **Online learning:** LSTM memory updates every decision; model weights update every 8,192
  decisions (about 9 minutes on a GTX 1650), so the policy keeps adapting after deployment.
- **Truth-free observation:** the agent sees only its own scan history — hit rate, airtime,
  staleness and the last signal heard per band (strength, pulse width, bearing).
- **Fixed reward** built from the same receiver-side quantities:
  `R = 0.5·h + (1.5 − d)·s + 0.5·Σhits − 3.0·d·n` — exploit what pays, explore what is overdue,
  pay for detections, charge for hogging airtime.
- **Operator-guided priority:** optional per-band threat priorities from mission intel steer the
  scan toward high-threat bands.

### 3. A fair evaluation ladder (`rfenv/baselines/`, `rfenv/compare.py`)

Every scheduler — random, round robin, Turing reference sweep, greedy camper, recency heuristic,
Apfeld adaptive (published), PPO, DQN and NARADA — runs on identical scenarios with identical
noise, and is compared mission by mission on **interception ratio, censored intercept time and
emitter coverage together**.

---

## Results

47 scenarios × 3 noise seeds = 141 missions per scheduler. Reward function: `reward_balance`.

| Scheduler | Avg reward | Interception ratio ↑ | Censored intercept time ↓ | Emitter coverage ↑ | Avg intercept rate ↑ |
|---|---|---|---|---|---|
| Round robin (open-loop) | 220 | 0.059 | 3.72 s | 0.877 | 1.060 /s |
| Recency heuristic | 290 | 0.112 | 2.73 s | 0.911 | 1.109 /s |
| **NARADA** | **568.2** | **0.153** | **1.90 s** | **0.925** | **1.144 /s** |

- NARADA beats the open-loop sweep on **both** headline objectives in **83.7%** of missions, and the
  recency heuristic in **73.8%**.
- In a matched comparison, LSTM memory caught **35% more traffic** than a memoryless PPO policy.
- Inference takes **2.5 ms per decision on a single CPU thread** — no GPU needed to run.

---

## Quick start

```bash
python -m venv venv
venv/Scripts/pip install -r requirements.txt     # Windows; use .venv/bin/ on Linux/macOS

# Validate the environment (four gates)
venv/Scripts/python -m rfenv.validate

# Run the scheduler comparison, with figures
venv/Scripts/python -m rfenv.compare --seeds 3 --figures

# Train a Recurrent PPO scheduler
venv/Scripts/python -m rfenv.rl.recurrent_ppo --reward reward_balance --timesteps 200000 --seed 0

# Run the test suite
venv/Scripts/python -m pytest
```

The TSRD data is gated on Hugging Face and not committed; place it under `data/turing/`. The
training stack (`torch`, `stable-baselines3`, `sb3-contrib`) is optional — the environment, the
baselines and most of the test suite run without it. More commands in [`run.md`](run.md).

**Tech stack:** Python 3.12 · NumPy · Gymnasium 1.3 · Stable-Baselines3 2.9 · SB3-Contrib 2.9 ·
PyTorch 2.14 · h5py 3.16 · Matplotlib 3.11

---

## Repository layout

| Path | Contents |
|---|---|
| `rfenv/` | The RF environment: scenarios, truth grid, receiver, Gymnasium interface, metrics, validation |
| `rfenv/rl/` | DQN, PPO and Recurrent PPO trainers, online fine-tuning, custom policies |
| `rfenv/baselines/` | The scheduler ladder |
| `tests/` | Test suite |
| `runs/checkpoints/` | Trained models, with a manifest per checkpoint |
| `docs/project/RF_ENVIRONMENT_EXPLAINED.md` | Plain-language guide to the environment |
| `docs/project/RF_AGENT_EXPLAINED.md` | Plain-language guide to NARADA |
| `docs/project/EVALUATION.md` | Metric definitions, baselines and validation gates |
| `docs/project/DECISIONS.md` | Every design decision, with its evidence |
| `docs/reference/` | Research papers and the dataset paper |

Contributors working with Claude Code should read [`CLAUDE.md`](CLAUDE.md) first.

---

## References

- *The Turing Synthetic Radar Dataset* — Alan Turing Institute, `arXiv:2602.03856`
- *Interception Model of Random Scanning Strategies Against Frequency-Agile Radar in Electronic
  Support* — IEEE
- *An Adaptive Receiver Search Strategy for Electronic Support* — Apfeld, Charlish & Koch,
  Fraunhofer FKIE, 2016
