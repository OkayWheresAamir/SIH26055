"""Shared plumbing every view in this package needs: the save helper and the
two y-axis labelling conventions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rfenv.constants import BAND_CENTRES_MHZ, N_BANDS, NOISE_FLOOR_DBM

DPI = 150

# The waterfall colour scale, fixed rather than per-grid, so any two waterfalls
# can be laid side by side and believed -- which is the artefact's whole job.
#
# The bottom is the noise floor: an empty cell sits exactly at N0 and is masked
# out, so nothing is ever drawn there. The top is a clip, not a maximum. Measured
# this session over all 47 train configs, both runs, per-emitter peak level: the
# median emitter peaks at -75.8 dB (scan) / -58.3 dB (stare) and the 95th
# percentile at -30.4 / -33.9, while the loudest single emitter in the whole train
# set reaches +6.40 dB. Scaling to that maximum pushes almost every emitter into
# the bottom fifth of the colourmap and the antenna-pattern structure disappears.
# So the handful of cells above -20 dB saturate, and the 100 dB where the data
# actually lives gets the whole scale.
LEVEL_VMIN_DB = NOISE_FLOOR_DBM
LEVEL_VMAX_DB = -20.0


def _save(fig, path: str | Path | None):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
    return fig


def _band_axis(ax) -> None:
    """Label the y axis by frequency, because that is what an operator reads.

    Band index is linear in frequency -- centres are 250 MHz + 500 MHz per band --
    so the ticks can carry GHz directly with no distortion.
    """
    ticks = np.arange(0, N_BANDS, 5)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{BAND_CENTRES_MHZ[b] / 1000:.2f}" for b in ticks])
    ax.set_ylabel("band centre (GHz)")


def _full_band_axis(ax) -> None:
    """Label every band, not every 5th.

    Used by the live/animated views only (`env_frame`), which have no
    neighbouring row's compactness to protect the way `schedule_timeline`'s do
    -- and "which exact band" is exactly what a viewer following something live
    needs to read off.
    """
    ticks = np.arange(N_BANDS)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{BAND_CENTRES_MHZ[b] / 1000:.2f}" for b in ticks], fontsize=6)
    ax.set_ylabel("band centre (GHz)", fontsize=8)
