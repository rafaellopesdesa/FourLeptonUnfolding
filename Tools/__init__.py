"""Reusable four-lepton analysis utilities."""

from .four_lepton_kinematics import (
    FourLeptonKinematics,
    KinematicError,
    compute_kinematics,
    compute_paired_kinematics,
    four_lepton_observables,
)
from .higgs_decay_width import (
    HiggsDecayDensity,
    differential_decay_width,
    sm_higgs_decay_density,
)

__all__ = [
    "FourLeptonKinematics",
    "HiggsDecayDensity",
    "KinematicError",
    "compute_kinematics",
    "compute_paired_kinematics",
    "differential_decay_width",
    "four_lepton_observables",
    "sm_higgs_decay_density",
]
