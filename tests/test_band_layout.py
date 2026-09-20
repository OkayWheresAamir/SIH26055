"""`rfenv.env.band_layout`/`obs_version_for_width` (D79): splitting a flat
observation into its per-band and global parts, purely from `_BLOCK_SPECS`'s
declared widths -- no hardcoded block-name list anywhere.

No training stack needed (pure numpy), unlike `test_policies.py`'s D79 half.
"""
from __future__ import annotations

import numpy as np
import pytest

from rfenv.env import (
    N_BANDS,
    OBS_LAYOUTS,
    _BLOCK_SPECS,
    band_layout,
    obs_version_for_width,
    obs_width,
)


@pytest.mark.parametrize("version", sorted(OBS_LAYOUTS))
def test_every_block_is_classified_and_widths_add_up(version):
    layout = band_layout(version)
    assert set(layout.per_band_blocks) | set(layout.global_blocks) == set(OBS_LAYOUTS[version])
    assert set(layout.per_band_blocks).isdisjoint(layout.global_blocks)
    n_global = sum(_BLOCK_SPECS[n][0] for n in layout.global_blocks)
    assert layout.band_index.shape == (N_BANDS, len(layout.per_band_blocks))
    assert layout.global_index.shape == (n_global,)
    assert N_BANDS * len(layout.per_band_blocks) + n_global == obs_width(version)


@pytest.mark.parametrize("version", sorted(OBS_LAYOUTS))
def test_classification_matches_block_specs_width_directly(version):
    """The classification rule itself: width == N_BANDS, nothing else."""
    layout = band_layout(version)
    for name in layout.per_band_blocks:
        assert _BLOCK_SPECS[name][0] == N_BANDS
    for name in layout.global_blocks:
        assert _BLOCK_SPECS[name][0] != N_BANDS


def test_prev_action_would_be_classified_per_band_if_any_layout_used_it():
    """D75 shipped `prev_action` in "v3", then removed it the next day -- it
    is still registered in `_BLOCK_SPECS`, N_BANDS wide, just unused by any
    current `OBS_LAYOUTS` entry. `band_layout`'s classification is purely
    width-based, so this one fact is what guarantees it would never be
    silently dropped into "global" or skipped if a layout used it again.
    """
    assert _BLOCK_SPECS["prev_action"][0] == N_BANDS


@pytest.mark.parametrize("version", sorted(OBS_LAYOUTS))
def test_the_gather_reconstructs_known_per_band_and_global_values(version):
    """Build a flat vector where every value encodes its own block's offset,
    then verify `band_index`/`global_index` gather exactly what each name
    claims to hold -- not just the right shape.
    """
    layout = band_layout(version)
    width = obs_width(version)
    flat = np.zeros(width, dtype=np.float64)
    offsets = {}
    offset = 0
    for name in OBS_LAYOUTS[version]:
        offsets[name] = offset
        block_width = _BLOCK_SPECS[name][0]
        if block_width == N_BANDS:
            flat[offset:offset + block_width] = offset * 1000 + np.arange(block_width)
        else:
            flat[offset:offset + block_width] = offset * 1000 + 900
        offset += block_width

    batch = np.stack([flat, flat * 2.0])  # (2, width)
    band_gathered = batch[:, layout.band_index]      # (2, N_BANDS, n_band_blocks)
    global_gathered = batch[:, layout.global_index]  # (2, n_global)

    for j, name in enumerate(layout.per_band_blocks):
        expected = offsets[name] * 1000 + np.arange(N_BANDS)
        assert np.array_equal(band_gathered[0, :, j], expected)
        assert np.array_equal(band_gathered[1, :, j], expected * 2.0)

    for k, name in enumerate(layout.global_blocks):
        assert global_gathered[0, k] == offsets[name] * 1000 + 900


def test_v1_has_five_per_band_blocks_and_three_global():
    layout = band_layout("v1")
    assert layout.per_band_blocks == (
        "hit_rate", "visit_density", "staleness", "current_band", "hit_streak",
    )
    assert layout.global_blocks == ("clock", "measured_dbm", "current_hit_streak")


def test_v3_has_eleven_per_band_blocks_and_four_global():
    layout = band_layout("v3")
    assert len(layout.per_band_blocks) == 11
    assert layout.global_blocks == ("clock", "current_hit_streak", "prev_reward", "prev_hit")


def test_unknown_obs_version_raises():
    with pytest.raises(ValueError, match="obs_version must be one of"):
        band_layout("v99")


@pytest.mark.parametrize("version", sorted(OBS_LAYOUTS))
def test_obs_version_for_width_is_the_reverse_of_obs_width(version):
    assert obs_version_for_width(obs_width(version)) == version


def test_obs_version_for_width_raises_on_an_unknown_width():
    with pytest.raises(ValueError, match="no registered obs_version"):
        obs_version_for_width(12345)
