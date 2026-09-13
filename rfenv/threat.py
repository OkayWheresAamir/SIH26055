"""D70 -- the example threat library, and the per-band priority vector `p` it produces.

**This is domain judgement applied to a dataset field, not a discovered fact.**
`THREAT_WEIGHTING_BRIEF.md` §5 and `1_THREAT_BRIEF_for_RL_team.pdf` p.5 are explicit: "Label this
as an example, not as our threat model... it goes in code before any scheduler is scored against
it, and it never changes afterwards." `EMITTER_CLASS` is that fixed mapping -- classified once,
before any scheduler sees it, with one line of rationale per name as the brief itself asks for
(§5 "Deleted from the previous draft": no obtainable external doctrine, so publish the mapping as
code with the reasoning attached instead).

**The priority formula is the PDF brief's, not the withdrawn `.md` draft's.** The `.md` draft
(`THREAT_WEIGHTING_BRIEF.md`) proposed a 3-bucket max-over-contributors rule; the PDF
(`1_THREAT_BRIEF_for_RL_team.pdf`, written after running the brief's own §2 gate) replaces it,
because the max rule measurably fails: 77.8% of occupied bands land at `p = HIGH` under `max`,
because most bands hold at least one HIGH-class contributor once 47 configs are pooled -- p is
"near-uniform on the bands that matter." The share-weighted form below (p.3 of the PDF) instead
scored a graded-priority lift of 2.22x at 10-25% airtime against the max rule's flat 1.17x, and
carries 334 distinct values instead of 3. **This module implements the PDF's formula.**

    p[b] = sum_e w(class(e)) * share_e(b)   over emitters e that occupy band b
    share_e(b) = e's illuminations in band b / all illuminations in band b   (pulse-share,
                 not cell-share -- the PDF's own §4 recommendation, "illuminations are the
                 interception-ratio denominator")
    w = {HIGH: 1.0, MEDIUM: 0.5, LOW: 0.0}

then rescaled so **1.0 means "an average band"**, not so the ceiling is 1.0 -- the same
convention D55 applies to `visit_density` (1.0 = equal fair share, not the camping ceiling): divide
by the population mean over *occupied* bands, so a scheduler that cannot tell bands apart at all
sees a vector of all-1.0s, and a band is only ever "elevated" or "depressed" relative to what an
uninformed guess would be.

**Empty bands are read as 1.0 ("ordinary"), not 0.0.** This is one of the two items the PDF brief
(§4.2) leaves open -- "p on empty bands... is undefined. That is a third of the vector." An
unoccupied band carries no threat evidence one way or the other, so treating it as neutral
(1.0, same as the population average) is the choice that does not bias the agent for or against
it; reading it as 0.0 would falsely assert "definitely nothing dangerous here," which is not a
claim the data supports. This default is a routine implementation choice for the *computation*,
not a decision about training-time behaviour (the `p` sampling distribution during training,
PDF §4.3, is still separately open and not decided here).
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from rfenv.constants import GAMMA_DBM, N_BANDS
from rfenv.scenario import Scenario
from rfenv.truth import TruthGrid


class ThreatClass(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


CLASS_WEIGHT: dict[ThreatClass, float] = {
    ThreatClass.HIGH: 1.0,
    ThreatClass.MEDIUM: 0.5,
    ThreatClass.LOW: 0.0,
}

# The 68 distinct `metadata/transmitters/*/.attrs['function']` values across every
# `data/turing/**/*.h5` file (verified this session: `grep`-free, read via h5py over
# every file in `data/turing/`; 68 distinct strings, matching the brief's own count).
# Classified once, per the brief's own category definitions
# (THREAT_WEIGHTING_BRIEF.md §5 / PDF p.1-2):
#   HIGH   -- fire control, engagement, missile guidance, counter-battery, LPI
#   MEDIUM -- air-defence search, early warning, surveillance, maritime patrol
#   LOW    -- weather, marine navigation, airport/ground movement, SAR, ground-penetrating
# Validated: applied to the 47 train configs' detectable emitters (D27), this mapping gives
# HIGH 656/1704 = 38.5%, MEDIUM 777/1704 = 45.6%, LOW 271/1704 = 15.9% -- within 2 emitters
# of the PDF's own published split (656/38.5%, 775/45.5%, 273/16.0%; 653/1704=38.3% in the
# PDF's later re-measurement). Close enough to have almost certainly reproduced the same
# reasoning on every unambiguous name; not claimed to be bit-identical on the few genuinely
# ambiguous ones (multi-role naval/naval-multifunction radars, mainly).
EMITTER_CLASS: dict[str, ThreatClass] = {
    # -- fire control / engagement / missile guidance --
    "AEGIS SPY-1D Naval Fire Control Radar": ThreatClass.HIGH,       # "Fire Control" in the name
    "AN/APG-77 F-22 Raptor AESA Radar": ThreatClass.HIGH,             # APG-series = fighter fire control
    "AN/APG-78 Longbow Fire Control Radar": ThreatClass.HIGH,        # "Fire Control" in the name
    "AN/APG-81 F-35 Fire Control": ThreatClass.HIGH,                  # "Fire Control" in the name
    "AN/APG-83 SABR Fighter": ThreatClass.HIGH,                       # APG-series fighter fire control
    "AN/MPQ-53 Patriot Engagement": ThreatClass.HIGH,                 # "Engagement" in the name
    "AN/SPY-1B AEGIS Naval Radar": ThreatClass.HIGH,                  # AEGIS = naval fire control system
    "AN/SPY-6(V)1 Air & Missile Defense Radar": ThreatClass.HIGH,     # missile-engagement radar
    "AN/SPY-7 Hypersonic Defense": ThreatClass.HIGH,                  # missile-engagement radar
    "CAPTOR-E AESA Fighter Radar": ThreatClass.HIGH,                  # fighter fire-control radar
    "Elta EL/M-2084": ThreatClass.HIGH,                               # counter-battery/weapon-locating radar
    "Iron Dome EL/M-2084 Multi-Mission Radar": ThreatClass.HIGH,      # missile-defence engagement radar
    "KRONOS Grand Naval Radar": ThreatClass.HIGH,                     # naval multi-function fire control
    "MFCR Ballistic Missile Defense Radar": ThreatClass.HIGH,         # "Multi-Function Fire Control" + BMD
    "Patriot AN/MPQ-53 Air Defense Radar": ThreatClass.HIGH,          # Patriot engagement radar
    "SAMPSON Multi-Function Naval Radar": ThreatClass.HIGH,           # naval air-defence fire control
    "Thales NS100 Naval Multi-Function Radar": ThreatClass.HIGH,      # naval multi-function fire control
    "Thales RBE2-AA AESA": ThreatClass.HIGH,                          # Rafale fighter fire-control radar
    "KMR-2500 Ka-band Millimeter Wave Radar": ThreatClass.HIGH,       # mm-wave precision-targeting radar
    # -- counter-battery / weapon-locating --
    "AN/TPQ-36 Firefinder Battlefield Radar": ThreatClass.HIGH,       # Firefinder = counter-battery
    "AN/TPQ-53 Counter-Battery Radar": ThreatClass.HIGH,              # "Counter-Battery" in the name
    "ARTHUR Advanced Weapon Locating System": ThreatClass.HIGH,       # weapon-locating = counter-battery
    "CAESAR C-Band Mobile Artillery Radar": ThreatClass.HIGH,         # artillery-locating radar
    # -- LPI --
    "LPI Naval Stealth Radar": ThreatClass.HIGH,                      # "LPI" in the name
    "MARS-L Low-Probability-of-Intercept Naval Radar": ThreatClass.HIGH,  # "Low-Probability-of-Intercept"

    # -- air-defence search / early warning / surveillance / maritime patrol --
    "AESA Vehicle-Mounted Reconnaissance Radar": ThreatClass.MEDIUM,  # reconnaissance/surveillance
    "AN/APY-10 Maritime Patrol": ThreatClass.MEDIUM,                  # "Maritime Patrol" in the name
    "AN/FPS-117 Long Range Air Surveillance Radar": ThreatClass.MEDIUM,  # "Surveillance" in the name
    "AN/FPS-132 Early Warning": ThreatClass.MEDIUM,                   # "Early Warning" in the name
    "AN/SPQ-9B Surface Search": ThreatClass.MEDIUM,                   # naval search radar
    "AN/SPS-48G 3D Air Search": ThreatClass.MEDIUM,                   # "Air Search" in the name
    "AN/TPS-75 Expeditionary Radar": ThreatClass.MEDIUM,              # general air-surveillance radar
    "AWACS MESA Advanced Airborne Radar": ThreatClass.MEDIUM,         # airborne early-warning/surveillance
    "Agile Multi-Beam Space Surveillance Radar": ThreatClass.MEDIUM,  # "Surveillance" in the name
    "Boeing E-3 Sentry AWACS Radar": ThreatClass.MEDIUM,              # AWACS = airborne early warning
    "Giraffe 4A Advanced Surveillance": ThreatClass.MEDIUM,           # "Surveillance" in the name
    "Ground Master 400 Air Defense": ThreatClass.MEDIUM,              # air-defence search radar
    "IAI EL/M-2106 ATAR 3D Tactical Air Defense": ThreatClass.MEDIUM, # tactical air-defence search radar
    "IBIS-150 Long Range Ground Radar": ThreatClass.MEDIUM,           # ground-surveillance radar
    "Indra Lanza 3D Radar": ThreatClass.MEDIUM,                       # air-defence/surveillance radar
    "JORN Over-The-Horizon Radar": ThreatClass.MEDIUM,                # early-warning surveillance radar
    "Kelvin Hughes SharpEye Coastal Surveillance": ThreatClass.MEDIUM,  # "Surveillance" in the name
    "Lockheed Martin TPS-77": ThreatClass.MEDIUM,                     # long-range air-surveillance radar
    "Nebo-M Anti-Stealth VHF Radar": ThreatClass.MEDIUM,              # long-range early-warning radar
    "Northrop Grumman MESA Airborne SAR": ThreatClass.MEDIUM,         # airborne surveillance/mapping radar
    "RAT-31DL Long Range Surveillance": ThreatClass.MEDIUM,           # "Surveillance" in the name
    "Raytheon Sentinel AN/MPQ-64": ThreatClass.MEDIUM,                # short-range air-defence search radar
    "Rheinmetall X-TAR3D": ThreatClass.MEDIUM,                        # tactical air-defence search radar
    "SMART-L EWC Long Range": ThreatClass.MEDIUM,                     # "Early Warning" long-range radar
    "Saab Giraffe AMB": ThreatClass.MEDIUM,                           # air-surveillance/defence radar
    "Selex Gabbiano T200 Maritime Patrol Radar": ThreatClass.MEDIUM,  # "Maritime Patrol" in the name
    "Selex RAT-31DL": ThreatClass.MEDIUM,                             # long-range surveillance radar
    "Space Fence SSR Space Surveillance Radar": ThreatClass.MEDIUM,   # "Surveillance" in the name
    "TERMA SCANTER 5000 Coastal Surveillance": ThreatClass.MEDIUM,    # "Surveillance" in the name
    "Teledyne FLIR RS8000 Dual-Band Thermal Radar": ThreatClass.MEDIUM,  # thermal surveillance sensor
    "Thales Ground Master 400": ThreatClass.MEDIUM,                   # air-defence search radar
    "VASILIY UHF Long-Range Air Defense Radar": ThreatClass.MEDIUM,   # long-range air-defence search radar

    # -- weather / marine navigation / airport-ground movement / SAR / ground-penetrating --
    "AN/SPN-43C Aircraft Control": ThreatClass.LOW,                   # air-traffic-control radar
    "ASDE-X Airport Surface Detection Radar": ThreatClass.LOW,        # airport surface-movement radar
    "Furuno FAR-2238S X-Band Marine Radar": ThreatClass.LOW,          # marine-navigation radar
    "Furuno FCR-21 Marine Navigation": ThreatClass.LOW,               # "Marine Navigation" in the name
    "GSSI SIR-3000 Ground Penetrating Radar": ThreatClass.LOW,        # "Ground Penetrating" in the name
    "MEXTRA 4D C-Band Airport Surveillance": ThreatClass.LOW,         # airport-surface radar
    "NEXRAD WSR-88D Doppler Weather": ThreatClass.LOW,                # "Weather" in the name
    "NEXRAD WSR-88D Weather Radar": ThreatClass.LOW,                  # "Weather" in the name
    "ROSTOV-ARENA UWB Search & Rescue Radar": ThreatClass.LOW,        # "Search & Rescue" in the name
    "TDWR Terminal Doppler Weather": ThreatClass.LOW,                 # "Weather" in the name
    "VORAX Airfield Surface Movement Radar": ThreatClass.LOW,         # "Airfield Surface Movement"
}

# Sanity check, enforced by test_threat.py: every function string the data actually contains
# must be classified. A new/renamed function in a future dataset revision must fail loudly here
# rather than silently default to some class.
N_CLASSIFIED = len(EMITTER_CLASS)


def classify(function: str) -> ThreatClass:
    """The one place `EMITTER_CLASS` is read. Raises on an unclassified name rather than
    guessing -- the whole point of a fixed-before-scoring library (see module docstring) is
    that it cannot silently grow a new, undocumented entry."""
    try:
        return EMITTER_CLASS[function]
    except KeyError:
        raise ValueError(
            f"{function!r} is not in the threat library (rfenv/threat.py::EMITTER_CLASS). "
            "Classify it there, with a one-line rationale, before scoring anything against it."
        ) from None


def compute_priority(scenario: Scenario, grid: TruthGrid, *, gamma: float = GAMMA_DBM) -> np.ndarray:
    """The per-band priority vector `p`, D70 / PDF p.3's share-weighted formula.

    `p[b] = sum_e w(class(e)) * share_e(b)`, `share_e(b)` = emitter e's pulse share of band b
    (pulse-share, not cell-share -- the PDF's own recommendation), then rescaled so the mean
    over *occupied* bands is 1.0 ("ordinary"), and empty bands read as 1.0 (see module
    docstring). Only detectable emitters (D27) contribute -- an emitter that never clears gamma
    could not have been classified by any real system either, so it carries no priority signal.

    Reads `scenario.contributions` (for `.meta['function']` and `.cells`/`.n_pulses`, truth-side)
    and `grid` (for `detectable_emitters`, truth-side) -- this is intentional and matches D29's
    reward-side permission, not the observation's. `p` itself becomes an *observation* block
    only once D70's retrain step (PDF §3, step 3) is actually built; this function is the
    library-side computation that step would call, not a change to `ScanEnv` on its own.
    """
    contributions = scenario.contributions
    detectable = set(grid.detectable_emitters(gamma).tolist())

    # Per-band total illuminations, over ALL contributors regardless of detectability -- the
    # share denominator is "how loud is this band", not "how loud is this band's threat-scored
    # traffic", so a detectable HIGH emitter sharing a band with an undetectable LOW one still
    # gets its true share, not an inflated one.
    band_total = np.zeros(N_BANDS, dtype=np.float64)
    for c in contributions:
        if not len(c):
            continue
        bands = c.cells[:, 0].astype(np.int64)
        np.add.at(band_total, bands, c.n_pulses.astype(np.float64))

    p_raw = np.zeros(N_BANDS, dtype=np.float64)
    for i, c in enumerate(contributions):
        if i not in detectable or not len(c):
            continue
        weight = CLASS_WEIGHT[classify(c.meta.get("function"))]
        if weight == 0.0:
            continue
        bands = c.cells[:, 0].astype(np.int64)
        pulses = c.n_pulses.astype(np.float64)
        share = np.zeros_like(pulses)
        nonzero = band_total[bands] > 0
        share[nonzero] = pulses[nonzero] / band_total[bands][nonzero]
        np.add.at(p_raw, bands, weight * share)

    occupied = band_total > 0
    if occupied.any():
        mean_occupied = float(p_raw[occupied].mean())
        if mean_occupied > 0:
            p_raw = p_raw / mean_occupied
    # Empty bands carry no evidence -- read as "ordinary" (1.0), not "safe" (0.0). See the
    # module docstring's "Empty bands" paragraph.
    p = np.where(occupied, p_raw, 1.0)
    return p.astype(np.float32)
