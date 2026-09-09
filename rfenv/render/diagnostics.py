"""Receiver characterisation (the ROC) and per-band structure (gate 2)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rfenv.constants import BAND_CENTRES_MHZ, N_BANDS
from rfenv.render._common import _save


def roc(
    curve: dict,
    path: str | Path | None = None,
    *,
    operating: dict | None = None,
    title: str = "Receiver ROC",
    figsize: tuple[float, float] = (5.5, 5.0),
):
    """Pd against Pfa over a gamma sweep -- the receiver deliverable (D15).

    `curve` is what `receiver.roc()` returns; `operating` is what
    `receiver.operating_point()` returns, and marks the frozen point on it.

    **The population is written on the figure**, as D33 requires. Pd depends on it
    entirely -- 0.819, 0.837 and 0.851 at the same gamma over three different
    populations -- and the previously quoted 0.822 was withdrawn precisely because
    a plot and a number had gone out into the world without one.

    Pfa is on a log axis: it runs from about 1 down to the Gaussian tail as gamma
    rises, and on a linear axis the entire useful part of the curve collapses onto
    the left edge.
    """
    pfa = np.asarray(curve["pfa"], dtype=np.float64)
    pd = np.asarray(curve["pd"], dtype=np.float64)
    floor = 1e-12  # log axis cannot show an exactly-zero tail

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(np.maximum(pfa, floor), pd, "-", color="#0a84ff", linewidth=1.6)
    ax.set_xscale("log")
    ax.set_xlabel("P_fa")
    ax.set_ylabel("P_d")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25, which="both")

    if operating is not None:
        ax.plot(max(operating["pfa"], floor), operating["pd"], "o", color="#ff3b30", zorder=5)
        ax.annotate(
            f"γ = {operating['gamma_dbm']:g} dB\n"
            f"P_d = {operating['pd']:.4f}\n"
            f"P_fa = {operating['pfa']:.3e}\n"
            f"sensitivity {operating['sensitivity_dbm']:.2f} dB"
            f" (at P_d = {operating['pd_target']:g})",
            xy=(max(operating["pfa"], floor), operating["pd"]),
            xytext=(12, -46), textcoords="offset points", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#888", alpha=0.9),
        )

    ax.set_title(
        f"{title}\npopulation: {curve['population']} "
        f"({curve['n_occupied']:,} occupied / {curve['n_empty']:,} empty cells)",
        fontsize=10,
    )
    return _save(fig, path)


def band_profile(
    series: dict[str, np.ndarray],
    path: str | Path | None = None,
    *,
    title: str = "Per-band structure",
    ylabel: str = "fraction",
    figsize: tuple[float, float] = (13.0, 3.6),
):
    """Grouped bars over the 36 bands -- gate 2's "not just the aggregate" check.

    `series` maps a label to a length-36 array, so it draws predicted against
    recorded, or any pair worth comparing band by band. Deliberately unopinionated
    about what the arrays mean: gate 2 owns that definition, this draws it.

    Band 0 is expected to be the visible failure when stare-built predictions are
    compared against scan recordings -- stare cannot see below 500 MHz, so the
    band is 59.12% occupied in the recordings and 0.00% predicted (D10, D24). It
    is a stated limitation, and the plot should show it rather than hide it.
    """
    labels = list(series)
    n = len(labels)
    x = np.arange(N_BANDS)
    width = 0.8 / max(n, 1)

    fig, ax = plt.subplots(figsize=figsize)
    for i, label in enumerate(labels):
        ax.bar(x + (i - (n - 1) / 2) * width, np.asarray(series[label], dtype=np.float64),
               width=width, label=label)

    ax.set_xticks(x[::2])
    ax.set_xticklabels([f"{BAND_CENTRES_MHZ[b] / 1000:.1f}" for b in x[::2]], fontsize=8)
    ax.set_xlabel("band centre (GHz)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-1, N_BANDS)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10)
    return _save(fig, path)
