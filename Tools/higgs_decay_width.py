"""SM scalar-Higgs decay density for H -> ZZ -> e+e- mu+mu-.

This implements the spin-zero helicity amplitudes and the seven-dimensional
angular/mass distribution of arXiv:1208.4018 for the tree-level SM choice
``g1^(0)=1`` and ``g2^(0)=g3^(0)=g4^(0)=0``.  Equivalently,
``a1 = mZ**2 / mH**2`` and ``a2=a3=0``.

The paper specifies this distribution only up to a constant of
proportionality.  The value returned here is therefore a *differential decay
density kernel*, not an absolute partial width in GeV.  It is suitable for
event reweighting, likelihood ratios, and shape comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sqrt
from typing import Any, Sequence

try:
    from .four_lepton_kinematics import FourLeptonKinematics, compute_kinematics
except ImportError:  # Allow direct use with Tools/ on PYTHONPATH.
    from four_lepton_kinematics import FourLeptonKinematics, compute_kinematics


@dataclass(frozen=True)
class HiggsDecayDensity:
    """Components of the differential decay-density kernel."""

    value: float
    angular_factor: float
    mass_factor: float
    amplitude_00: float
    amplitude_pp: float
    amplitude_mm: float
    kinematics: FourLeptonKinematics


def _z_lepton_asymmetry(sin2_theta_w: float) -> float:
    # Overall normalization cancels in A_f.  For charged leptons:
    # T3=-1/2 and Q=-1, so gV=T3-2 Q sin^2(theta_W), gA=T3.
    g_vector = -0.5 + 2.0 * sin2_theta_w
    g_axial = -0.5
    return 2.0 * g_vector * g_axial / (g_vector * g_vector + g_axial * g_axial)


def _mass_factor(m1: float, m2: float, m_h: float, m_z: float, gamma_z: float) -> float:
    upper = 1.0 - ((m1 + m2) / m_h) ** 2
    lower = 1.0 - ((m1 - m2) / m_h) ** 2
    if upper <= 0.0 or lower <= 0.0:
        return 0.0

    propagator1 = (m1 * m1 - m_z * m_z) ** 2 + (m_z * gamma_z) ** 2
    propagator2 = (m2 * m2 - m_z * m_z) ** 2 + (m_z * gamma_z) ** 2
    return sqrt(upper * lower) * m1**3 * m2**3 / (propagator1 * propagator2)


def sm_higgs_decay_density(
    electron: Any,
    positron: Any,
    muon: Any,
    antimuon: Any,
    *,
    m_h: float = 125.10,
    m_z: float = 91.1876,
    gamma_z: float = 2.4952,
    higgs_vev: float = 246.22,
    sin2_theta_w: float = 0.23122,
    beam_axis: Sequence[float] = (0.0, 0.0, 1.0),
) -> HiggsDecayDensity:
    """Evaluate the paper's SM ``0+`` differential decay-density kernel.

    The input order is ``(e-, e+, mu-, mu+)``.  The density is differential
    in ``m_Z1, m_Z2, cos(theta*), Psi, cos(theta1), cos(theta2), Phi``.
    For a scalar, the production variables ``cos(theta*)`` and ``Psi`` are
    uniform and consequently do not appear in the non-constant angular factor.
    """

    for name, parameter in (
        ("m_h", m_h),
        ("m_z", m_z),
        ("gamma_z", gamma_z),
        ("higgs_vev", higgs_vev),
    ):
        if parameter <= 0.0:
            raise ValueError(f"{name} must be positive")
    if not 0.0 < sin2_theta_w < 1.0:
        raise ValueError("sin2_theta_w must lie between zero and one")

    kin = compute_kinematics(
        electron, positron, muon, antimuon, beam_axis=beam_axis
    )
    m1, m2 = kin.m_Z1, kin.m_Z2
    mass_factor = _mass_factor(m1, m2, m_h, m_z, gamma_z)
    if mass_factor == 0.0:
        return HiggsDecayDensity(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, kin)

    x = ((m_h * m_h - m1 * m1 - m2 * m2) / (2.0 * m1 * m2)) ** 2 - 1.0
    x = max(0.0, x)  # protect the physical boundary from round-off

    # Eqs. (9-12), with g1^(0)=1: a1=mZ^2/mH^2 and a2=a3=0.
    transverse_amplitude = m_z * m_z / higgs_vev
    amplitude_00 = -transverse_amplitude * sqrt(1.0 + x)
    amplitude_pp = transverse_amplitude
    amplitude_mm = transverse_amplitude

    c1, c2 = kin.cos_theta1, kin.cos_theta2
    s1 = sqrt(max(0.0, 1.0 - c1 * c1))
    s2 = sqrt(max(0.0, 1.0 - c2 * c2))
    af = _z_lepton_asymmetry(sin2_theta_w)
    a00 = abs(amplitude_00)
    at = transverse_amplitude

    # The relative phases phi++=phi--=pi because A00<0 and A++=A-->0.
    angular = 4.0 * a00 * a00 * s1 * s1 * s2 * s2
    angular += at * at * (
        (1.0 + 2.0 * af * c1 + c1 * c1)
        * (1.0 + 2.0 * af * c2 + c2 * c2)
        + (1.0 - 2.0 * af * c1 + c1 * c1)
        * (1.0 - 2.0 * af * c2 + c2 * c2)
    )
    angular -= 4.0 * a00 * at * cos(kin.Phi) * s1 * s2 * (
        (af + c1) * (af + c2) + (af - c1) * (af - c2)
    )
    angular += 2.0 * at * at * s1 * s1 * s2 * s2 * cos(2.0 * kin.Phi)

    # Tiny negative values can occur only from floating-point cancellation.
    angular = max(0.0, angular)
    return HiggsDecayDensity(
        value=angular * mass_factor,
        angular_factor=angular,
        mass_factor=mass_factor,
        amplitude_00=amplitude_00,
        amplitude_pp=amplitude_pp,
        amplitude_mm=amplitude_mm,
        kinematics=kin,
    )


def differential_decay_width(
    electron: Any,
    positron: Any,
    muon: Any,
    antimuon: Any,
    **kwargs: Any,
) -> float:
    """Return only the differential decay-density value.

    See :func:`sm_higgs_decay_density` for the parameters and the important
    overall-normalization convention.
    """

    return sm_higgs_decay_density(
        electron, positron, muon, antimuon, **kwargs
    ).value
