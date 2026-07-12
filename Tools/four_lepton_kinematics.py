"""Four-lepton observables in the conventions of arXiv:1208.4018.

The public functions take the four-vectors in charge-ordered form::

    electron, positron, muon, antimuon

That is, ``(e-, e+, mu-, mu+)``.  A Scikit-HEP ``vector`` object is the
natural input.  Mappings with ``E, px, py, pz`` keys and sequences ordered as
``(E, px, py, pz)`` are also accepted and converted to ``vector`` objects.

The paper calls the decay polar angles theta_1 and theta_2.  Consequently the
returned observables are ``cos_theta1`` and ``cos_theta2``; names such as
``cos_phi1`` would mix polar and azimuthal angle notation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import atan2, hypot, log, pi, sqrt
from typing import Any, Mapping, Sequence

import numpy as np
import vector


_EPSILON = 1.0e-12


class KinematicError(ValueError):
    """Raised when an angle is undefined for a degenerate configuration."""


@dataclass(frozen=True)
class FourLeptonKinematics:
    """The complete set of four-lepton observables used in this study.

    Angles are in radians and masses/momenta use the units of the inputs.
    ``z1_flavor`` records which opposite-sign pair became Z1 after applying
    the paper's convention ``m_Z1 >= m_Z2``.
    """

    cos_theta_star: float
    cos_theta1: float
    cos_theta2: float
    Phi: float
    Phi1: float
    Psi: float
    m_Z1: float
    m_Z2: float
    m_ZZ: float
    y_ZZ: float
    pT_ZZ: float
    z1_flavor: str

    def as_dict(self) -> dict[str, float | str]:
        """Return a plain dictionary suitable for column-oriented output."""

        return asdict(self)


def _value(component: Any) -> float:
    return float(component() if callable(component) else component)


def _coerce_p4(momentum: Any):
    """Convert a supported scalar four-vector representation to ``vector``."""

    if all(hasattr(momentum, name) for name in ("E", "px", "py", "pz")):
        return vector.obj(
            E=_value(momentum.E),
            px=_value(momentum.px),
            py=_value(momentum.py),
            pz=_value(momentum.pz),
        )

    if isinstance(momentum, Mapping):
        try:
            return vector.obj(
                E=float(momentum["E"]),
                px=float(momentum["px"]),
                py=float(momentum["py"]),
                pz=float(momentum["pz"]),
            )
        except KeyError as error:
            raise TypeError("four-vector mappings need E, px, py, and pz keys") from error

    if isinstance(momentum, Sequence) and not isinstance(momentum, (str, bytes)):
        if len(momentum) != 4:
            raise TypeError("four-vector sequences must be ordered (E, px, py, pz)")
        energy, px, py, pz = momentum
        return vector.obj(E=float(energy), px=float(px), py=float(py), pz=float(pz))

    raise TypeError(
        "expected a vector object, an E/px/py/pz mapping, or an (E, px, py, pz) sequence"
    )


def _spatial(momentum: Any) -> np.ndarray:
    return np.array([momentum.px, momentum.py, momentum.pz], dtype=float)


def _unit(value: np.ndarray, label: str) -> np.ndarray:
    magnitude = float(np.linalg.norm(value))
    if magnitude <= _EPSILON:
        raise KinematicError(f"cannot define {label}: vector magnitude is zero")
    return value / magnitude


def _clip_cosine(value: float) -> float:
    return float(np.clip(value, -1.0, 1.0))


def _wrap_angle(value: float) -> float:
    wrapped = (value + pi) % (2.0 * pi) - pi
    return pi if wrapped == -pi and value > 0.0 else wrapped


def _mass(momentum: Any) -> float:
    mass_squared = float(momentum.mass2)
    if mass_squared < -_EPSILON:
        raise KinematicError("a composite four-vector is spacelike")
    return sqrt(max(0.0, mass_squared))


def compute_kinematics(
    electron: Any,
    positron: Any,
    muon: Any,
    antimuon: Any,
    *,
    beam_axis: Sequence[float] = (0.0, 0.0, 1.0),
) -> FourLeptonKinematics:
    """Calculate the production and decay observables of arXiv:1208.4018.

    Parameters
    ----------
    electron, positron, muon, antimuon:
        Four-vectors in the fixed order ``e-, e+, mu-, mu+``.
    beam_axis:
        Direction of beam 1 in the frame of the supplied four-vectors.  At a
        pp collider the sign of the incoming parton direction is intrinsically
        ambiguous; this implementation uses the laboratory ``+z`` beam by
        default, giving a reproducible signed convention.

    Notes
    -----
    Z1 is the heavier dilepton pair on an event-by-event basis, as in the
    paper.  ``Phi``, ``Phi1`` and ``Psi`` lie in ``[-pi, pi]``.
    """

    e_minus = _coerce_p4(electron)
    e_plus = _coerce_p4(positron)
    mu_minus = _coerce_p4(muon)
    mu_plus = _coerce_p4(antimuon)

    z_e = e_minus + e_plus
    z_mu = mu_minus + mu_plus
    if _mass(z_e) >= _mass(z_mu):
        z1, z2 = z_e, z_mu
        q11, q12 = e_minus, e_plus
        q21, q22 = mu_minus, mu_plus
        z1_flavor = "electron"
    else:
        z1, z2 = z_mu, z_e
        q11, q12 = mu_minus, mu_plus
        q21, q22 = e_minus, e_plus
        z1_flavor = "muon"

    zz = z1 + z2
    m_zz = _mass(zz)
    if m_zz <= _EPSILON:
        raise KinematicError("the four-lepton system must have positive invariant mass")

    try:
        axis = _unit(np.asarray(beam_axis, dtype=float), "beam direction")
    except (TypeError, ValueError) as error:
        raise TypeError("beam_axis must contain three numeric components") from error
    if axis.shape != (3,):
        raise TypeError("beam_axis must contain exactly three components")

    # All plane definitions in Eq. (2) are evaluated in the X/ZZ rest frame.
    z1_x = z1.boostCM_of(zz)
    q11_x = q11.boostCM_of(zz)
    q12_x = q12.boostCM_of(zz)
    q21_x = q21.boostCM_of(zz)
    q22_x = q22.boostCM_of(zz)
    beam = vector.obj(E=1.0, px=axis[0], py=axis[1], pz=axis[2])
    beam_x = beam.boostCM_of(zz)

    q1_hat = _unit(_spatial(z1_x), "Z1 direction in the ZZ rest frame")
    beam_hat = _unit(_spatial(beam_x), "beam direction in the ZZ rest frame")
    n1 = _unit(np.cross(_spatial(q11_x), _spatial(q12_x)), "Z1 decay plane")
    n2 = _unit(np.cross(_spatial(q21_x), _spatial(q22_x)), "Z2 decay plane")
    n_sc = _unit(np.cross(beam_hat, q1_hat), "production plane")

    cos_theta_star = _clip_cosine(float(np.dot(q1_hat, beam_hat)))
    phi = atan2(float(np.dot(q1_hat, np.cross(n1, n2))), -float(np.dot(n1, n2)))
    phi1 = atan2(float(np.dot(q1_hat, np.cross(n1, n_sc))), float(np.dot(n1, n_sc)))
    psi = _wrap_angle(phi1 + 0.5 * phi)

    # Eq. (4): q11 and q21 are the fermions, i.e. the negative leptons.
    q2_z1 = z2.boostCM_of(z1)
    q11_z1 = q11.boostCM_of(z1)
    q1_z2 = z1.boostCM_of(z2)
    q21_z2 = q21.boostCM_of(z2)
    cos_theta1 = _clip_cosine(
        -float(
            np.dot(
                _unit(_spatial(q2_z1), "Z2 direction in the Z1 rest frame"),
                _unit(_spatial(q11_z1), "negative lepton direction in the Z1 rest frame"),
            )
        )
    )
    cos_theta2 = _clip_cosine(
        -float(
            np.dot(
                _unit(_spatial(q1_z2), "Z1 direction in the Z2 rest frame"),
                _unit(_spatial(q21_z2), "negative lepton direction in the Z2 rest frame"),
            )
        )
    )

    energy, pz = float(zz.E), float(zz.pz)
    if energy <= abs(pz):
        raise KinematicError("the four-lepton rapidity is undefined")
    rapidity = 0.5 * log((energy + pz) / (energy - pz))

    return FourLeptonKinematics(
        cos_theta_star=cos_theta_star,
        cos_theta1=cos_theta1,
        cos_theta2=cos_theta2,
        Phi=_wrap_angle(phi),
        Phi1=_wrap_angle(phi1),
        Psi=psi,
        m_Z1=_mass(z1),
        m_Z2=_mass(z2),
        m_ZZ=m_zz,
        y_ZZ=rapidity,
        pT_ZZ=hypot(float(zz.px), float(zz.py)),
        z1_flavor=z1_flavor,
    )


def four_lepton_observables(
    electron: Any,
    positron: Any,
    muon: Any,
    antimuon: Any,
    *,
    beam_axis: Sequence[float] = (0.0, 0.0, 1.0),
) -> dict[str, float | str]:
    """Return the observables as a plain dictionary."""

    return compute_kinematics(
        electron, positron, muon, antimuon, beam_axis=beam_axis
    ).as_dict()
