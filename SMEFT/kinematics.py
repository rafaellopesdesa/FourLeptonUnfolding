"""Reconstruct a canonical scalar-Higgs decay from compact Born observables.

The e/mu matrix-element model is massless. The six invariant decay
coordinates determine its four lepton momenta up to an overall proper
rotation and boost, including every crossed-pair invariant for identical
leptons. The orientation relative to the beam is not reconstructed: it is
irrelevant to a scalar decay amplitude, but would matter for production or
non-scalar matrix elements. Tau channels are deliberately unsupported.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


DECAY_FIELDS = ("m_ZZ", "m_Z1", "m_Z2", "cos_theta1", "cos_theta2", "Phi", "type")
REQUIRED_BRANCHES = tuple(f"born_{name}" for name in DECAY_FIELDS)
COMPONENT_ORDER = ("E", "px", "py", "pz")
_COSINE_TOLERANCE = 1.e-12
_CHANNEL_PDGS = np.array([
    [13, -13, 13, -13],  # 4mu
    [13, -13, 11, -11],  # 2mu2e: Z1 is the muon pair
    [11, -11, 13, -13],  # 2e2mu: Z1 is the electron pair
    [11, -11, 11, -11],  # 4e
], dtype=np.int32)


class BornKinematicsError(ValueError):
    """The supplied compact decay coordinates cannot define this model."""


def _where(mask: np.ndarray) -> str:
    if mask.ndim == 0:
        return "scalar input"
    indices = np.flatnonzero(mask)[:5]
    return "flat row indices " + ", ".join(str(int(index)) for index in indices)


def _reject(mask: np.ndarray, message: str) -> None:
    if np.any(mask):
        raise BornKinematicsError(f"{message} ({_where(mask)})")


def reconstruct_born_momenta(
    columns: Mapping[str, object], *, prefix: str = "born_",
) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical rest-frame momenta and charge-ordered lepton PDG IDs.

    Required columns are ``{prefix}m_ZZ``, ``m_Z1``, ``m_Z2``,
    ``cos_theta1``, ``cos_theta2``, ``Phi`` and ``type``. Scalars and arrays
    broadcast to a common event shape. The outputs have shapes
    ``event_shape + (4, 4)`` and ``event_shape + (4,)``; a scalar input gives
    ``(4, 4)`` and ``(4,)``. Momentum components are ``(E, px, py, pz)`` and
    particles are ``(Z1-, Z1+, Z2-, Z2+)``. Positive PDG IDs label negative
    leptons. Units follow the supplied masses, conventionally GeV.

    Z1 points along +x in the four-lepton rest frame, with the first negative
    lepton in the x-z plane. With the repository's +z beam convention this
    chooses cos(theta*)=0 and Phi1=0. The stored production orientation,
    rapidity and transverse momentum are intentionally unused. Pair labels
    are preserved; the caller must include identical-particle exchange in
    its matrix element rather than interpreting these pairs as exclusive
    intermediate states.

    All masses must be finite and positive with m_ZZ > m_Z1 + m_Z2. Both
    polar cosines may touch +/-1; excursions up to 1e-12 from floating-point
    rounding are clipped. Phi is periodic in radians. Unsupported tau
    types, invalid coordinates, or incompatible array shapes raise
    BornKinematicsError. This function does not silently filter rows or
    inspect the ntuple's provenance/availability masks; callers select and
    validate the intended Born rows before reconstruction.
    """
    values = []
    for name in DECAY_FIELDS:
        key = prefix + name
        if key not in columns:
            raise BornKinematicsError(f"missing required decay column {key!r}")
        try:
            value = np.asarray(columns[key], dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise BornKinematicsError(f"decay column {key!r} must be numeric") from error
        _reject(~np.isfinite(value), f"decay column {key!r} must be finite")
        values.append(value)
    try:
        mass, mass1, mass2, cosine1, cosine2, phi, event_type = np.broadcast_arrays(*values)
    except ValueError as error:
        raise BornKinematicsError("decay columns have incompatible event shapes") from error

    _reject((event_type != np.floor(event_type)) | (event_type < 0) | (event_type > 8),
            "decay type must be an integer in 0..8")
    _reject(event_type >= 4, "tau decay channels (types 4..8) are unsupported by the massless e/mu model")
    _reject((mass <= 0) | (mass1 <= 0) | (mass2 <= 0), "decay masses must be positive")
    with np.errstate(over="ignore"):
        pair_sum = mass1 + mass2
    _reject(mass <= pair_sum, "decay requires m_ZZ > m_Z1 + m_Z2; threshold has undefined decay axes")
    _reject(np.abs(cosine1) > 1. + _COSINE_TOLERANCE, "cos_theta1 must lie in [-1, 1]")
    _reject(np.abs(cosine2) > 1. + _COSINE_TOLERANCE, "cos_theta2 must lie in [-1, 1]")
    cosine1 = np.clip(cosine1, -1., 1.)
    cosine2 = np.clip(cosine2, -1., 1.)
    sine1 = np.sqrt((1. - cosine1) * (1. + cosine1))
    sine2 = np.sqrt((1. - cosine2) * (1. + cosine2))

    # Factorized dimensionless Kallen function avoids subtraction of large
    # squared masses and avoids overflow from fourth powers of the scale.
    ratio1, ratio2 = mass1 / mass, mass2 / mass
    momentum = .5 * mass * np.sqrt(
        ((mass - pair_sum) / mass) * (1. + ratio1 + ratio2)
        * (1. - ratio1 + ratio2) * (1. + ratio1 - ratio2)
    )
    energy1 = .5 * mass * (1. + (ratio1 - ratio2) * (ratio1 + ratio2))
    energy2 = .5 * mass * (1. + (ratio2 - ratio1) * (ratio2 + ratio1))
    transverse1 = .5 * mass1 * sine1
    transverse2 = .5 * mass2 * sine2
    zeros = np.zeros_like(mass)

    # Equal-mass daughters have E*=p*=mZ/2 in each pair frame. Writing the
    # boosts through E_Z and |p_Z| avoids divisions by small dilepton masses.
    # For the repository's Phi=atan2(q1.(n1 x n2), -n1.n2), the second
    # negative lepton must have +sin(Phi) along y in this orientation.
    momenta = np.stack((
        np.stack((.5 * (energy1 + momentum * cosine1),
                  .5 * (momentum + energy1 * cosine1), zeros, transverse1), axis=-1),
        np.stack((.5 * (energy1 - momentum * cosine1),
                  .5 * (momentum - energy1 * cosine1), zeros, -transverse1), axis=-1),
        np.stack((.5 * (energy2 + momentum * cosine2),
                  -.5 * (momentum + energy2 * cosine2),
                  transverse2 * np.sin(phi), transverse2 * np.cos(phi)), axis=-1),
        np.stack((.5 * (energy2 - momentum * cosine2),
                  -.5 * (momentum - energy2 * cosine2),
                  -transverse2 * np.sin(phi), -transverse2 * np.cos(phi)), axis=-1),
    ), axis=-2)
    _reject(~np.all(np.isfinite(momenta), axis=(-1, -2)), "reconstructed momenta are nonfinite")
    return momenta, _CHANNEL_PDGS[event_type.astype(np.intp)].copy()
