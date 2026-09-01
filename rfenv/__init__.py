"""SIH26055 — RF environment for electronic-support scan scheduling.

Layers, per docs/project/ENVIRONMENT_SPEC.md:

    L0  scenario.py   emitter contributions, pool, scenarios
    L1  truth.py      the truth grid: S (level), C (pulse count), Z (occupancy)
    L2  receiver.py   dwell mechanics, the detection channel
    L3  env.py        gymnasium interface

In one breath: an emitter is a recorded contribution to the time-frequency grid;
a scenario is a set of emitters and the world is their maximum; the receiver looks
at one band at a time and declares a hit when what it hears beats its threshold;
the scheduler chooses the band.
"""

from rfenv import constants  # noqa: F401
