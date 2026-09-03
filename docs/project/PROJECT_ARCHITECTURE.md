# Project Architecture — Working Specification

## Purpose

This is the **current working architecture** for the SIH project.

Claude Code should follow it as the default direction while building the project, but it is **not a rigid final specification**. If evidence, validation, the dataset, research, or implementation experience shows that an architectural change is needed, Claude should explain the proposed change and its reasoning rather than silently changing direction.

This document describes **what we are building**. It does not prescribe a specific RL algorithm or make CEWS the definition of the architecture.

---

# 1. Core idea

We are building an **interactive RF environment** in which different receiver scheduling strategies can be evaluated under controlled conditions.

The eventual goal is an **adaptive scheduler**, with an **RL agent eventually becoming the learned decision-maker**.

The project should therefore be thought of as:

```text
Turing scenario information
          ↓
      RF scenario
          ↓
Interactive RF environment
          ↓
Receiver observation / feedback
          ↓
       Scheduler
          ↓
Choose next frequency band
          ↓
      Environment
          ↺
```

Eventually:

```text
        Interactive RF Environment
                   ↕
               RL Agent
                   ↓
          choose next band
                   ↓
              Environment
                   ↓
             HIT / MISS
                   ↺
```

The exact RL algorithm, state representation, reward function, network architecture and training procedure are intentionally **not fixed here**.

---

# 2. Environment vs scenario

These are different concepts.

### Environment

The reusable simulator/code that models the RF interaction.

It is responsible for things such as:
- representing RF entities,
- representing receiver constraints,
- advancing time,
- determining whether an observation/detection can occur,
- producing observations/feedback,
- exposing actions to the scheduler.

### Scenario

One particular RF situation/configuration loaded into the environment.

Different scenarios can contain different:
- emitters,
- frequencies,
- timing/PRI behavior,
- beam/scan behavior,
- activity patterns,
- other RF characteristics.

Therefore:

```text
ONE ENVIRONMENT
│
├── Scenario A
├── Scenario B
├── Scenario C
└── Scenario N
```

The current 47 scenarios are therefore **47 scenario inputs**, not 47 environments.

---

# 3. Turing data and scenario construction

The working conceptual split is:

```text
Turing scenario/configuration information
                ↓
        construct/describe
           RF scenario
```

while:

```text
Turing pulse recordings
        ↓
recorded receiver observations
        ↓
validation / comparison / analysis
```

The pulse table should not automatically be treated as a complete description of the hidden RF world.

An interactive environment needs to answer counterfactual questions such as:

> "What happens if the receiver chooses this band at this time?"

A recorded observation sequence does not necessarily contain enough information to answer arbitrary counterfactual actions.

If further dataset investigation shows that this assumption is incomplete or wrong, surface that finding and revisit the model.

---

# 4. Receiver

The receiver is part of the environment and experimental setup.

For initial scheduler comparisons, receiver constraints should be kept controlled and aligned with the intended Turing reference where appropriate.

The reason is fairness:

```text
same RF scenario
+
same receiver constraints
+
different scheduler
=
meaningful scheduler comparison
```

The receiver is therefore not the main architectural contribution we are trying to optimize.

If the receiver model needs to change, that should be an explicit experimental/architectural decision rather than an accidental side effect of implementation.

---

# 5. Scheduler

The scheduler is the decision-making component.

Its job is fundamentally:

> Decide which frequency band the receiver should inspect next.

The environment should support multiple scheduler implementations under the same conditions:

```text
                 SAME ENVIRONMENT
                       │
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
      fixed        heuristic       RL agent
     baseline       baseline       eventual
```

This separation is important because it lets us determine whether an adaptive scheduler actually provides an advantage.

---

# 6. RL

RL is expected to become an important part of the project.

The eventual interaction loop is:

```text
state / history
      ↓
RL policy
      ↓
action: choose band
      ↓
RF environment
      ↓
observation + reward
      ↓
updated state/history
      ↺
```

However, this document deliberately does not lock down:
- the RL algorithm,
- state representation,
- reward,
- neural architecture,
- exploration method,
- training schedule.

Those should be selected after we understand and validate the environment and establish appropriate baselines.

---

# 7. CEWS

CEWS is **not the definition of our project architecture**.

It is an existing implementation/resource that may be useful for:
- simulation infrastructure,
- reusable abstractions,
- existing receiver/emitter models,
- evaluation utilities,
- reference behavior.

We may reuse, adapt, replace or extend CEWS components as appropriate.

Do not force the project to remain CEWS-shaped merely because CEWS is currently available.

If a CEWS limitation conflicts with the project requirements, surface the limitation and propose alternatives.

---

# 8. Environment validation

Before relying on scheduler/RL results, we need confidence that the environment represents the intended problem reasonably.

Validation may include:
- controlled mathematical/theoretical checks,
- simple scenario sanity checks,
- classical scheduler behavior,
- comparisons against relevant Turing observations,
- other independent evidence.

Validation is the phase where environment assumptions can be identified, debugged and changed.

Once the team considers the environment sufficiently validated, create a reproducible/frozen environment version for scheduler comparisons.

---

# 9. Development scenarios and final evaluation

The current 47 selected Turing scenarios are a **development/training subset**.

They are useful for:
- building and debugging the environment,
- repeated testing,
- initial scheduler development,
- RL training if appropriate.

More scenarios can be introduced later for:
- training diversity,
- validation,
- robustness testing,
- final held-out evaluation.

The exact split is an experimental decision and should not be treated as permanently fixed by this document.

The final evaluation should use scenarios that were not used to tune the final system.

---

# 10. Recommended project order

```text
Understand Turing data
        ↓
Build RF scenario representation
        ↓
Build interactive RF environment
        ↓
Validate environment
        ↓
Freeze validated environment
        ↓
Build/compare scheduler baselines
        ↓
Develop RL scheduler
        ↓
Freeze final system
        ↓
Final held-out evaluation
```

Do not optimize an RL agent against an environment whose important behavior is still uncertain.

---

# 11. Architectural flexibility

This is a **working architecture**, not immutable truth.

If a significant change is proposed, Claude should explain:

```text
Current architecture:
...

Problem discovered:
...

Evidence:
...

Proposed change:
...

Why:
...

Affected components:
...

Validation required:
...
```

For a major architectural change, ask for human confirmation before implementing it.

Small implementation choices do not require approval.

---

# 12. What Claude should optimize for

Prioritize:

1. Correctness and reproducibility.
2. Clear separation between environment and scheduler.
3. Ability to run multiple schedulers under identical conditions.
4. Turing-grounded scenario construction.
5. Independent environment validation.
6. Clean future integration of RL.
7. Human-readable assumptions and decisions.
8. Minimal unnecessary coupling to CEWS.

---

# 13. What Claude should not assume

Do not assume:

- CEWS defines the final architecture.
- Every Turing pulse row is an RL training example.
- The 47 scenarios are the complete dataset.
- The 47 scenarios are the final test set.
- One specific RL algorithm has already been selected.
- The reward function is finalized.
- Every Turing metadata field has an obvious interpretation.
- Exact pulse-for-pulse reproduction is automatically required.
- A research paper's proposed architecture must be implemented.

---

# 14. One-sentence architecture

> Build a validated, interactive RF environment grounded in Turing scenarios; keep receiver conditions controlled for fair scheduler comparisons; and eventually place a learned RL scheduler inside that environment to make adaptive frequency-selection decisions.

This is the default direction unless the human team explicitly revises it.
