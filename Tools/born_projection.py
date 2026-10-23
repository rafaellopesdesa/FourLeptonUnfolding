"""Rapidity-preserving ISR Born projection of a colorless final-state system.

This is the kinematic map in arXiv:2606.11083, Sec. 2.3, Eqs. (14)--(16),
not an NLO matrix-element weight or an inversion of photon radiation.
The same Lorentz transformation is applied to every supplied four-vector.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Iterable

import numpy as np
import vector


BORN_PROJECTION_METADATA = {
    "method": "longitudinal_transverse_inverse_longitudinal_boosts",
    "arxiv_url": "https://arxiv.org/abs/2606.11083",
    "arxiv_version": "2606.11083v2",
    "section": "2.3",
    "equations": [14, 15, 16],
    "preserves": ["total invariant mass", "total rapidity", "all pairwise Lorentz products"],
    "sets": "total transverse momentum to zero",
    "scope": "kinematic ISR projection only; no NLO matrix-element reweighting",
}


class BornProjectionError(ValueError):
    """The supplied system has no numerically well-defined massive rest frame."""


def _components(momentum) -> np.ndarray:
    """Read (E, px, py, pz), retaining extended precision in the arithmetic."""
    names = ("E", "px", "py", "pz")
    if isinstance(momentum, Mapping):
        try:
            values = [momentum[name] for name in names]
        except KeyError as error:
            raise TypeError("four-vector mappings require E, px, py and pz") from error
    elif all(hasattr(momentum, name) for name in names):
        values = [getattr(momentum, name) for name in names]
        values = [value() if callable(value) else value for value in values]
    else:
        raise TypeError("momenta must provide E, px, py and pz components")
    try:
        result = np.asarray(values, dtype=np.longdouble)
    except (TypeError, ValueError) as error:
        raise TypeError("four-vector components must be real scalar numbers") from error
    if result.shape != (4,):
        raise TypeError("four-vector components must be real scalar numbers")
    if not np.all(np.isfinite(result)):
        raise BornProjectionError("four-vector components must be finite")
    return result


def project_to_born(momenta: Iterable[object]) -> tuple[object, ...]:
    """Remove total transverse recoil while preserving mass and rapidity.

    For K=sum(p_i), first remove K_z with a longitudinal boost, remove K_T
    with a transverse boost in that frame, then undo the longitudinal boost.
    The final sum is (M*cosh(y), 0, 0, M*sinh(y)). The longitudinal operation
    uses light-cone components rather than a beta rounded to one. Extended
    arithmetic reduces cancellation, but cannot repair information already
    lost in the input four-vectors; near-lightlike totals are rejected.

    Inputs are unchanged; even an identity projection returns fresh vectors.
    No selection, pairing, mass constraint, or radiation recombination occurs.
    """
    rows = [_components(momentum) for momentum in momenta]
    if not rows:
        raise BornProjectionError("at least one four-vector is required")
    values = np.stack(rows)
    total = np.sum(values, axis=0, dtype=np.longdouble)
    if not np.all(np.isfinite(total)):
        raise BornProjectionError("total four-momentum is nonfinite")
    energy, kx, ky, kz = total
    if energy <= 0:
        raise BornProjectionError("total energy must be positive")
    plus, minus = energy + kz, energy - kz
    if plus <= 0 or minus <= 0:
        raise BornProjectionError("total four-momentum is spacelike or lightlike")
    transverse2 = kx * kx + ky * ky
    transverse_mass2 = plus * minus
    mass2 = transverse_mass2 - transverse2
    if mass2 <= 0:
        raise BornProjectionError("total four-momentum is spacelike or lightlike")
    # Returned vectors use float64. Below this relative invariant mass, input
    # rounding makes the boost ill-conditioned even if longdouble sees m2>0.
    tolerance = np.longdouble(64 * np.finfo(np.float64).eps) * energy * energy
    if mass2 <= tolerance:
        raise BornProjectionError("total four-momentum is numerically near-lightlike")
    if transverse2 == 0:
        projected = values.copy()
    else:
        mass = np.sqrt(mass2)
        transverse_mass = np.sqrt(transverse_mass2)
        exp_y = np.sqrt(plus / minus)

        # B_L: p+ -> exp(-y) p+, p- -> exp(+y) p-.
        pplus = (values[:, 0] + values[:, 3]) / exp_y
        pminus = (values[:, 0] - values[:, 3]) * exp_y
        energy_l = (pplus + pminus) / 2
        z_l = (pplus - pminus) / 2

        # B_T, written with gamma-1 = K_T^2/[M*(m_T+M)] to remain
        # well-behaved for arbitrarily small nonzero transverse recoil.
        dot = kx * values[:, 1] + ky * values[:, 2]
        energy_t = (transverse_mass * energy_l - dot) / mass
        coefficient = (dot / (transverse_mass + mass) - energy_l) / mass
        projected = values.copy()
        projected[:, 1] = values[:, 1] + coefficient * kx
        projected[:, 2] = values[:, 2] + coefficient * ky

        # B_L^-1 restores the original system rapidity, not its original p_z.
        pplus = (energy_t + z_l) * exp_y
        pminus = (energy_t - z_l) / exp_y
        projected[:, 0] = (pplus + pminus) / 2
        projected[:, 3] = (pplus - pminus) / 2

    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(projected, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise BornProjectionError("projected four-vectors exceed finite float64 range")
    return tuple(vector.obj(E=float(row[0]), px=float(row[1]), py=float(row[2]), pz=float(row[3]))
                 for row in result)
