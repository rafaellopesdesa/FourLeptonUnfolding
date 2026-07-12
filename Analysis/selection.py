"""Lepton pairing and event selection for the H -> 4l fiducial region."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

from Tools.four_lepton_kinematics import (
    FourLeptonKinematics,
    compute_paired_kinematics,
)


Z_MASS = 91.1876
LEPTON_PT_MIN = 5.0
LEPTON_ETA_MAX = 2.7
LEADING_PT_MINIMA = (20.0, 15.0, 10.0)
Z1_MASS_WINDOW = (50.0, 106.0)
Z2_MASS_WINDOW = (12.0, 115.0)
MIN_LEPTON_DELTA_R = 0.1
MIN_SFOS_MASS = 5.0


@dataclass(frozen=True)
class Lepton:
    p4: object
    flavor: str
    charge: int

    def __post_init__(self) -> None:
        if self.flavor not in {"electron", "muon"}:
            raise ValueError("flavor must be 'electron' or 'muon'")
        if self.charge not in {-1, 1}:
            raise ValueError("charge must be -1 or +1")


@dataclass(frozen=True)
class FourLeptonCandidate:
    z1_negative: Lepton
    z1_positive: Lepton
    z2_negative: Lepton
    z2_positive: Lepton
    event_type: int

    @property
    def leptons(self) -> tuple[Lepton, Lepton, Lepton, Lepton]:
        return (
            self.z1_negative,
            self.z1_positive,
            self.z2_negative,
            self.z2_positive,
        )

    @property
    def z1(self):
        return self.z1_negative.p4 + self.z1_positive.p4

    @property
    def z2(self):
        return self.z2_negative.p4 + self.z2_positive.p4

    @property
    def four_lepton(self):
        return self.z1 + self.z2

    def kinematics(self) -> FourLeptonKinematics:
        return compute_paired_kinematics(
            self.z1_negative.p4,
            self.z1_positive.p4,
            self.z2_negative.p4,
            self.z2_positive.p4,
            z1_flavor=self.z1_negative.flavor,
        )


@dataclass(frozen=True)
class SelectionResult:
    candidate: FourLeptonCandidate | None
    selected: bool


def _mass(momentum: object) -> float:
    return float(momentum.mass)


def _event_type(leptons: Iterable[Lepton], z1_flavor: str) -> int:
    flavors = [lepton.flavor for lepton in leptons]
    electron_count = flavors.count("electron")
    muon_count = flavors.count("muon")
    if muon_count == 4:
        return 0
    if electron_count == 2 and muon_count == 2:
        return 1 if z1_flavor == "muon" else 2
    if electron_count == 4:
        return 3
    raise ValueError("candidate is not one of 4mu, 2mu2e, 2e2mu, or 4e")


def _passes_lepton_definition(lepton: Lepton) -> bool:
    return float(lepton.p4.pt) > LEPTON_PT_MIN and abs(float(lepton.p4.eta)) < LEPTON_ETA_MAX


def choose_candidate(leptons: Iterable[Lepton]) -> FourLeptonCandidate | None:
    """Choose two disjoint SFOS pairs using the stated closest-to-mZ rule."""

    accepted = [lepton for lepton in leptons if _passes_lepton_definition(lepton)]
    pairs: list[tuple[int, int, Lepton, Lepton, float]] = []
    for first, second in combinations(range(len(accepted)), 2):
        one, two = accepted[first], accepted[second]
        if one.flavor != two.flavor or one.charge * two.charge != -1:
            continue
        negative, positive = (one, two) if one.charge < 0 else (two, one)
        pairs.append((first, second, negative, positive, _mass(one.p4 + two.p4)))

    best: tuple[tuple[float, float, float, tuple[int, ...]], FourLeptonCandidate] | None = None
    for lead in pairs:
        lead_indices = {lead[0], lead[1]}
        for sublead in pairs:
            if lead_indices.intersection((sublead[0], sublead[1])):
                continue
            selected_leptons = (lead[2], lead[3], sublead[2], sublead[3])
            key = (
                abs(lead[4] - Z_MASS),
                abs(sublead[4] - Z_MASS),
                -sum(float(lepton.p4.pt) for lepton in selected_leptons),
                tuple(sorted((lead[0], lead[1], sublead[0], sublead[1]))),
            )
            candidate = FourLeptonCandidate(
                z1_negative=lead[2],
                z1_positive=lead[3],
                z2_negative=sublead[2],
                z2_positive=sublead[3],
                event_type=_event_type(selected_leptons, lead[2].flavor),
            )
            if best is None or key < best[0]:
                best = (key, candidate)
    return None if best is None else best[1]


def _all_sfos_masses(candidate: FourLeptonCandidate) -> list[float]:
    masses = []
    for first, second in combinations(candidate.leptons, 2):
        if first.flavor == second.flavor and first.charge * second.charge == -1:
            masses.append(_mass(first.p4 + second.p4))
    return masses


def passes_selection(
    candidate: FourLeptonCandidate,
    *,
    four_lepton_mass_window: tuple[float, float] = (105.0, 160.0),
) -> bool:
    """Apply all event-level cuts from the supplied fiducial table."""

    ordered_pts = sorted((float(lepton.p4.pt) for lepton in candidate.leptons), reverse=True)
    if any(ordered_pts[index] <= threshold for index, threshold in enumerate(LEADING_PT_MINIMA)):
        return False

    m_z1, m_z2 = _mass(candidate.z1), _mass(candidate.z2)
    if not Z1_MASS_WINDOW[0] < m_z1 < Z1_MASS_WINDOW[1]:
        return False
    if not Z2_MASS_WINDOW[0] < m_z2 < Z2_MASS_WINDOW[1]:
        return False

    for first, second in combinations(candidate.leptons, 2):
        if float(first.p4.deltaR(second.p4)) <= MIN_LEPTON_DELTA_R:
            return False
    if any(mass <= MIN_SFOS_MASS for mass in _all_sfos_masses(candidate)):
        return False

    m4l = _mass(candidate.four_lepton)
    return four_lepton_mass_window[0] < m4l < four_lepton_mass_window[1]


def select_event(
    leptons: Iterable[Lepton],
    *,
    four_lepton_mass_window: tuple[float, float] = (105.0, 160.0),
) -> SelectionResult:
    candidate = choose_candidate(leptons)
    return SelectionResult(
        candidate=candidate,
        selected=(
            candidate is not None
            and passes_selection(candidate, four_lepton_mass_window=four_lepton_mass_window)
        ),
    )
