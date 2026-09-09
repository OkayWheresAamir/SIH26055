"""The generic episode driver.

Not a metrics concern and not a baselines concern: `run_episode` never touches
a CSV, an artefact, or a score, and it defines no policy. It is the one loop
that steps *any* `(obs, info) -> action` callable through `ScanEnv` -- every
baseline rung and the RL rung alike, identically. `EVALUATION.md` §5's ladder
lives outside the environment and consumes only L3 plus this driver.
"""

from __future__ import annotations


def run_episode(env, policy, *, seed: int | None = None, scenario=None):
    """Run one episode to termination and return the environment.

    `policy(observation, info) -> band`. Both arguments, because a schedule that
    depends on the clock -- Turing's own sweep is the obvious one -- needs the slot,
    and `info["slot"]` is where it lives. The observation alone would work
    (`obs[-1]` is normalised episode time) but reading the slot back out of a
    normalised float is a trap waiting for whoever writes the next baseline.

    This is a driver, not a baseline. It defines no policy: the baseline ladder
    (`EVALUATION.md` §5) lives outside the environment and consumes only L3 and
    these artefacts.
    """
    options = {"scenario": scenario} if scenario is not None else None
    obs, info = env.reset(seed=seed, options=options)
    terminated = False
    while not terminated:
        obs, _, terminated, _, info = env.step(int(policy(obs, info)))
    return env
