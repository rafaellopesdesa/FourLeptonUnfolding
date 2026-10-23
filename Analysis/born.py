"""Tau-inclusive hard four-lepton observables after an ISR Born projection.

These inputs must be the hard leptons before their photon radiation or tau
decay. This module does not infer hard ancestry from stable shower particles.
Exactly four charged leptons are required, and the closest-to-Z pairing is
fixed before projecting the entire quartet with the same Lorentz map.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
import math
from typing import Iterable

import numpy as np
import vector

from Analysis.lhe import KINEMATIC_FIELDS, LHEEvent
from Analysis.selection import Z_MASS
from Tools.born_projection import BornProjectionError, project_to_born
from Tools.four_lepton_kinematics import KinematicError, compute_paired_kinematics


BORN_BRANCH_TYPES = {
    "born_source": np.dtype(np.int8),
    "born_available": np.dtype(np.bool_),
    "born_kinematics_valid": np.dtype(np.bool_),
    "born_type": np.dtype(np.int8),
    "born_n_taus": np.dtype(np.int8),
    "born_shower_event_index": np.dtype(np.int64),
    **{f"born_input_{name}": np.dtype(np.float64)
       for name in ("m_ZZ", "y_ZZ", "pT_ZZ")},
    **{f"born_{name}": np.dtype(np.float64) for name in KINEMATIC_FIELDS},
}

_FLAVORS = {11: "electron", 13: "muon", 15: "tau"}
# Match the native hard-decay momentum check, allowing record precision noise.
_MOMENTUM_REL_TOL = 1.e-6
_MOMENTUM_ABS_TOL = 1.e-7
_EVENT_TYPES = {
    (13, 13): 0, (13, 11): 1, (11, 13): 2, (11, 11): 3,
    (15, 15): 4, (15, 11): 5, (11, 15): 6, (15, 13): 7,
    (13, 15): 8,
}


def output_schema() -> dict[str, np.dtype]:
    """Return an independent copy of the scalar ROOT branch schema."""
    return dict(BORN_BRANCH_TYPES)


def empty_observables() -> dict[str, int | bool | float]:
    """Missing/unsupported quartets have false masks, type -1 and NaNs."""
    return {
        name: (float("nan") if dtype.kind == "f" else
               False if dtype.kind == "b" else
               -1 if name in {"born_type", "born_shower_event_index"} else 0)
        for name, dtype in BORN_BRANCH_TYPES.items()
    }


def empty_columns(size: int) -> dict[str, np.ndarray]:
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 0:
        raise ValueError("column size must be a nonnegative integer")
    return {name: np.full(size, value, dtype=BORN_BRANCH_TYPES[name])
            for name, value in empty_observables().items()}


def _four_vector(momentum):
    names = ("E", "px", "py", "pz")
    if isinstance(momentum, Mapping):
        values = [momentum[name] for name in names]
    else:
        values = [getattr(momentum, name) for name in names]
        values = [value() if callable(value) else value for value in values]
    values = [float(value) for value in values]
    if not all(math.isfinite(value) for value in values) or values[0] <= 0:
        raise BornProjectionError("hard-lepton momenta need finite components and positive energy")
    energy, px, py, pz = values
    if math.hypot(px, py, pz) > energy + _MOMENTUM_ABS_TOL + _MOMENTUM_REL_TOL * energy:
        raise BornProjectionError("hard-lepton momentum is spacelike")
    return vector.obj(**dict(zip(names, values)))


def _choose_pairs(particles: list[tuple[int, object]]) -> tuple[int, ...] | None:
    """Match Analysis.selection's stable closest-Z ordering without cuts."""
    pairs = []
    for first, second in combinations(range(len(particles)), 2):
        pid_one, p4_one = particles[first]
        pid_two, p4_two = particles[second]
        if pid_one != -pid_two:
            continue
        negative, positive = (first, second) if pid_one > 0 else (second, first)
        mass = float((p4_one + p4_two).mass)
        if math.isfinite(mass) and mass >= 0:
            pairs.append((first, second, negative, positive, mass))

    best = None
    for lead in pairs:
        for sublead in pairs:
            if {lead[0], lead[1]}.intersection((sublead[0], sublead[1])):
                continue
            order = (lead[2], lead[3], sublead[2], sublead[3])
            key = (
                abs(lead[4] - Z_MASS),
                abs(sublead[4] - Z_MASS),
                -sum(float(particles[index][1].pt) for index in order),
                tuple(sorted((lead[0], lead[1], sublead[0], sublead[1]))),
            )
            if best is None or key < best[0]:
                best = (key, order)
    return None if best is None else best[1]


def _mass_y_pt(momentum) -> tuple[float, float, float]:
    energy, pz = float(momentum.E), float(momentum.pz)
    rapidity = (math.log(energy + pz) - math.log(energy - pz)) / 2
    return float(momentum.mass), rapidity, math.hypot(float(momentum.px), float(momentum.py))


def born_observables(
    particles: Iterable[tuple[int, object]], source: int, shower_event_index: int = -1,
) -> dict[str, int | bool | float]:
    """Project a complete hard charged-lepton quartet, retaining its source.

    Source codes are 0=missing, 1=original LHE, 2=Pythia native hard leptons,
    3=Herwig native hard leptons. Native sources require a positive, one-based
    accepted shower-event ordinal; LHE/missing sources use -1. Non-leptons
    are ignored. More or fewer than four charged e/mu/tau are unsupported,
    since projecting a selected subset would change the colorless system.

    A well-defined projection sets ``born_available`` even if degenerate
    decay planes prevent some angles. In that case masses, rapidity and pT
    remain available, while ``born_kinematics_valid`` stays false. A missing
    or unprojectable quartet has NaN observables and retains source identity.
    """
    if (isinstance(source, bool) or not isinstance(source, (int, np.integer))
            or source not in {0, 1, 2, 3}):
        raise ValueError("born source must be 0, 1, 2 or 3")
    if (isinstance(shower_event_index, bool)
            or not isinstance(shower_event_index, (int, np.integer))
            or shower_event_index > np.iinfo(np.int64).max
            or (source in {0, 1} and shower_event_index != -1)
            or (source in {2, 3} and shower_event_index < 1)):
        raise ValueError("native sources require a positive int64 shower event index; LHE/missing use -1")
    result = empty_observables()
    result.update(born_source=int(source), born_shower_event_index=int(shower_event_index))
    charged = [(pid, p4) for pid, p4 in particles if abs(pid) in _FLAVORS]
    if source == 0:
        if charged:
            raise ValueError("source 0 denotes missing hard leptons")
        return result
    if len(charged) != 4:
        return result
    try:
        charged = [(pid, _four_vector(p4)) for pid, p4 in charged]
        order = _choose_pairs(charged)
        if order is None:
            return result
        paired = [charged[index][1] for index in order]
        projected = project_to_born(paired)
        before = _mass_y_pt(sum(paired[1:], paired[0]))
        after = _mass_y_pt(sum(projected[1:], projected[0]))
        masses = (float((projected[0] + projected[1]).mass),
                  float((projected[2] + projected[3]).mass))
        if not all(math.isfinite(value) for value in (*before, *after, *masses)):
            return result
    except (BornProjectionError, ValueError, OverflowError, FloatingPointError):
        return result

    first_flavor, second_flavor = abs(charged[order[0]][0]), abs(charged[order[2]][0])
    result.update(born_available=True,
                  born_type=_EVENT_TYPES[(first_flavor, second_flavor)],
                  born_n_taus=sum(abs(pid) == 15 for pid, _ in charged))
    result.update(zip(("born_input_m_ZZ", "born_input_y_ZZ", "born_input_pT_ZZ"), before))
    result.update(zip(("born_m_ZZ", "born_y_ZZ", "born_pT_ZZ"), after))
    result.update(zip(("born_m_Z1", "born_m_Z2"), masses))
    try:
        kinematics = compute_paired_kinematics(*projected, z1_flavor=_FLAVORS[first_flavor])
    except (KinematicError, ValueError, OverflowError, FloatingPointError):
        return result
    values = {f"born_{name}": float(getattr(kinematics, name)) for name in KINEMATIC_FIELDS}
    if all(math.isfinite(value) for value in values.values()):
        result.update(values)
        result["born_kinematics_valid"] = True
    return result


def lhe_born_observables(event: LHEEvent) -> dict[str, int | bool | float]:
    """Project direct status-1 charged leptons with QCD recoil only.

    Extra final-state quarks/gluons are allowed. Photons, neutrinos or any
    other final-state particles make this definition unavailable: removing
    their momenta could select an incomplete colorless decay system. The
    ISR projection does not undo QED radiation or reconstruct tau decays.
    Direct parent taus themselves are supported as charged hard leptons.
    """
    allowed = set(_FLAVORS) | {1, 2, 3, 4, 5, 6, 21}
    if any(particle.status == 1 and abs(particle.pdg_id) not in allowed
           for particle in event.particles):
        return born_observables((), source=1)
    return born_observables(
        ((particle.pdg_id, particle.p4) for particle in event.particles
         if particle.status == 1 and abs(particle.pdg_id) in _FLAVORS),
        source=1,
    )
