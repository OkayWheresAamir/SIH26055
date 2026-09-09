"""The waterfall, and the plots that go beside it.

Artefact 3 of `EVALUATION.md` §8, plus the receiver's ROC and the per-band bar
gate 2 reads, plus the multi-scheduler comparison pictures of §5. matplotlib
is a dependency of this package alone -- `metrics/`, `env.py` and the layers
below them never import it, so the environment and its tests run without a
plotting stack installed.

**Package layout.** `_common.py` (the save helper, the two y-axis labelling
conventions), `episode.py` (`waterfall`, `env_frame` -- single-episode /
single-frame views), `comparison.py` (`schedule_timeline`, `discovery_curves`,
`pareto`, `compare_animation` -- the multi-scheduler pictures of §5),
`diagnostics.py` (`roc`, `band_profile` -- receiver and gate-2 pictures).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # write files, never open a window

import matplotlib.pyplot as plt  # noqa: E402,F401

from rfenv.render._common import DPI, LEVEL_VMAX_DB, LEVEL_VMIN_DB  # noqa: E402
from rfenv.render.comparison import (  # noqa: E402
    HIT_COLOUR,
    PATH_COLOUR,
    compare_animation,
    discovery_curves,
    pareto,
    schedule_timeline,
)
from rfenv.render.diagnostics import band_profile, roc  # noqa: E402
from rfenv.render.episode import env_frame, waterfall  # noqa: E402
from rfenv.render._cli import main  # noqa: E402

__all__ = [
    "DPI",
    "HIT_COLOUR",
    "LEVEL_VMAX_DB",
    "LEVEL_VMIN_DB",
    "PATH_COLOUR",
    "band_profile",
    "compare_animation",
    "discovery_curves",
    "env_frame",
    "main",
    "pareto",
    "roc",
    "schedule_timeline",
    "waterfall",
]
