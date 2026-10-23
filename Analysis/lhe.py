"""Stream original Les Houches events and calculate hard-record observables.

LHE weights are raw XWGTUP values, not luminosity-scaled analysis weights.
Only direct status-1 electrons/muons enter the four-lepton pairing. No shower
record, acceptance cut, fiducial cut, or Higgs-to-four-lepton substitution is
used here. The event ordinal is local to this original LHE source.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import math
from pathlib import Path
from typing import Iterator
import xml.etree.ElementTree as ET

import numpy as np
import vector

from Analysis.selection import Lepton, choose_candidate
from Tools.four_lepton_kinematics import KinematicError


KINEMATIC_FIELDS = (
    "cos_theta_star", "cos_theta1", "cos_theta2", "Phi", "Phi1", "Psi",
    "m_Z1", "m_Z2", "m_ZZ", "y_ZZ", "pT_ZZ",
)

LHE_BRANCH_TYPES = {
    "lhe_available": np.dtype(np.bool_),
    "lhe_has_four_leptons": np.dtype(np.bool_),
    "lhe_kinematics_valid": np.dtype(np.bool_),
    "lhe_has_higgs": np.dtype(np.bool_),
    "lhe_higgs_kinematics_valid": np.dtype(np.bool_),
    "lhe_has_taus": np.dtype(np.bool_),
    "lhe_event_index": np.dtype(np.int64),
    "lhe_source_id": np.dtype(np.uint64),
    "lhe_process_id": np.dtype(np.int32),
    "lhe_n_final_leptons": np.dtype(np.int32),
    "lhe_type": np.dtype(np.int8),
    "lhe_weight": np.dtype(np.float64),
    "lhe_scale": np.dtype(np.float64),
    "lhe_higgs_m": np.dtype(np.float64),
    "lhe_higgs_y": np.dtype(np.float64),
    "lhe_higgs_pT": np.dtype(np.float64),
    **{f"lhe_{name}": np.dtype(np.float64) for name in KINEMATIC_FIELDS},
}


@dataclass(frozen=True)
class LHEParticle:
    pdg_id: int
    status: int
    mother1: int
    mother2: int
    color1: int
    color2: int
    px: float
    py: float
    pz: float
    energy: float
    mass: float
    lifetime: float
    spin: float

    @property
    def mothers(self) -> tuple[int, int]:
        """Original 1-based MOTHUP endpoints; zero denotes no mother."""
        return self.mother1, self.mother2

    @property
    def p4(self):
        return vector.obj(px=self.px, py=self.py, pz=self.pz, E=self.energy)


@dataclass(frozen=True)
class LHEEvent:
    ordinal: int
    process_id: int
    weight: float
    scale: float
    aqed: float
    aqcd: float
    particles: tuple[LHEParticle, ...]

    @property
    def nup(self) -> int:
        return len(self.particles)


def source_sha256(path: str | Path) -> str:
    """Digest the actual original source bytes, including gzip when present."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_schema() -> dict[str, np.dtype]:
    return dict(LHE_BRANCH_TYPES)


def empty_columns(size: int) -> dict[str, np.ndarray]:
    """Missing LHE rows use explicit false masks, -1 indices/type and NaNs."""
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 0:
        raise ValueError("column size must be a nonnegative integer")
    result = {}
    for name, dtype in LHE_BRANCH_TYPES.items():
        default = np.nan if dtype.kind == "f" else (-1 if name in {"lhe_event_index", "lhe_type"} else 0)
        result[name] = np.full(size, default, dtype=dtype)
    return result


def _tag(element: ET.Element) -> str:
    return str(element.tag).rsplit("}", 1)[-1]


def _finite_float(token: str) -> float:
    value = float(token.replace("D", "E").replace("d", "e"))
    if not math.isfinite(value):
        raise ValueError(f"nonfinite numerical field {token!r}")
    return value


def _event_from_element(element: ET.Element, ordinal: int, path: Path) -> LHEEvent:
    # Direct text holds the HEPEUP block. Child text is XML metadata (rwgt,
    # weights, scales, ...); child tails are direct text again. XML comments
    # are discarded by ElementTree without removing surrounding event text.
    text = "\n".join([element.text or "", *(child.tail or "" for child in element)])
    rows = [line.split("#", 1)[0].strip() for line in text.splitlines()]
    rows = [line for line in rows if line]
    try:
        if not rows or len(rows[0].split()) != 6:
            raise ValueError("event header must contain six HEPEUP fields")
        header = rows[0].split()
        nup, process = int(header[0]), int(header[1])
        if nup < 0:
            raise ValueError("NUP must be nonnegative")
        if not np.iinfo(np.int32).min <= process <= np.iinfo(np.int32).max:
            raise ValueError("IDPRUP is outside the int32 output range")
        weight, scale, aqed, aqcd = map(_finite_float, header[2:])
        if len(rows) - 1 != nup:
            raise ValueError(f"NUP={nup} but found {len(rows) - 1} particle rows")
        particles = []
        for index, row in enumerate(rows[1:], start=1):
            fields = row.split()
            if len(fields) != 13:
                raise ValueError(f"particle {index} must contain 13 HEPEUP fields")
            integers = [int(field) for field in fields[:6]]
            if any(mother < 0 or mother > nup for mother in integers[2:4]):
                raise ValueError(f"particle {index} has a mother outside 0..NUP")
            numbers = [_finite_float(field) for field in fields[6:]]
            particles.append(LHEParticle(*integers, *numbers))
        return LHEEvent(ordinal, process, weight, scale, aqed, aqcd, tuple(particles))
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{path}: LHE event {ordinal}: {error}") from error


def iter_lhe_events(path: str | Path) -> Iterator[LHEEvent]:
    """Yield validated standard LHE(.gz) events without retaining the document.

    The source must be the original XML LHE, not a private shower-reader copy
    with nonstandard event tags. Event ordinals start at one and include all
    events, independent of weight, flavor, or subsequent shower acceptance.
    """
    path = Path(path).expanduser().resolve()
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    ordinal = 0
    stack: list[ET.Element] = []
    try:
        with opener(path, "rb") as stream:
            for phase, element in ET.iterparse(stream, events=("start", "end")):
                tag = _tag(element)
                if phase == "start":
                    if not stack and tag != "LesHouchesEvents":
                        raise ValueError(f"{path}: expected a LesHouchesEvents XML document")
                    if tag == "event" and any(_tag(parent) == "event" for parent in stack):
                        raise ValueError(f"{path}: nested LHE event element")
                    stack.append(element)
                    continue
                stack.pop()
                if tag == "event":
                    ordinal += 1
                    event = _event_from_element(element, ordinal, path)
                    if stack:
                        stack[-1].remove(element)
                    element.clear()
                    yield event
                elif stack and _tag(stack[-1]) in {"LesHouchesEvents", "eventgroup"}:
                    # Header/init and completed event groups must not accumulate.
                    stack[-1].remove(element)
                    element.clear()
    except ET.ParseError as error:
        raise ValueError(f"{path}: malformed LHE XML after {ordinal} complete event(s): {error}") from error


def _mass(momentum) -> float:
    mass2 = float(momentum.mass2)
    return math.sqrt(max(0., mass2)) if math.isfinite(mass2) and mass2 >= -1.e-9 else math.nan


def _mass_y_pt(momentum) -> tuple[float, float, float]:
    energy, pz = float(momentum.E), float(momentum.pz)
    y = .5 * (math.log(energy + pz) - math.log(energy - pz)) if energy > abs(pz) else math.nan
    values = (_mass(momentum), y, math.hypot(float(momentum.px), float(momentum.py)))
    return tuple(value if math.isfinite(value) else math.nan for value in values)


def _terminal_higgs(event: LHEEvent) -> list[LHEParticle]:
    indices = {i for i, particle in enumerate(event.particles, start=1)
               if abs(particle.pdg_id) == 25 and particle.status in {1, 2}}
    # MOTHUP gives the first/last mother indices, unlike Delphes' individual
    # mother pointers. A status copy of a Higgs makes its parent nonterminal.
    nonterminal = set()
    for i in indices:
        particle = event.particles[i - 1]
        first, last = particle.mothers
        if first:
            nonterminal.update(range(first, max(first, last) + 1))
    return [event.particles[i - 1] for i in sorted(indices - nonterminal)]


def event_observables(event: LHEEvent) -> dict[str, float | int | bool]:
    """Return one compact row; source_id is populated by the joining caller."""
    result = {name: values[0].item() for name, values in empty_columns(1).items()}
    result.update(lhe_available=True, lhe_event_index=event.ordinal,
                  lhe_process_id=event.process_id, lhe_weight=event.weight, lhe_scale=event.scale)
    final = [particle for particle in event.particles if particle.status == 1]
    charged = [particle for particle in final if abs(particle.pdg_id) in {11, 13}]
    result["lhe_n_final_leptons"] = len(charged)
    result["lhe_has_taus"] = any(abs(particle.pdg_id) == 15 for particle in final)
    result["lhe_has_higgs"] = any(abs(particle.pdg_id) == 25 and particle.status in {1, 2}
                                   for particle in event.particles)
    higgs = _terminal_higgs(event)
    if len(higgs) == 1:
        values = _mass_y_pt(higgs[0].p4)
        if all(math.isfinite(value) for value in values):
            result.update(zip(("lhe_higgs_m", "lhe_higgs_y", "lhe_higgs_pT"), values))
            result["lhe_higgs_kinematics_valid"] = True
    leptons = [Lepton(particle.p4, "electron" if abs(particle.pdg_id) == 11 else "muon",
                      -1 if particle.pdg_id > 0 else 1) for particle in charged]
    candidate = choose_candidate(leptons, apply_lepton_cuts=False)
    if candidate is None:
        return result
    result.update(lhe_has_four_leptons=True, lhe_type=candidate.event_type)
    result["lhe_m_Z1"], result["lhe_m_Z2"] = _mass(candidate.z1), _mass(candidate.z2)
    result.update(zip(("lhe_m_ZZ", "lhe_y_ZZ", "lhe_pT_ZZ"), _mass_y_pt(candidate.four_lepton)))
    try:
        kinematics = candidate.kinematics()
    except KinematicError:
        return result
    values = {f"lhe_{name}": float(getattr(kinematics, name)) for name in KINEMATIC_FIELDS}
    if all(math.isfinite(value) for value in values.values()):
        result.update(values)
        result["lhe_kinematics_valid"] = True
    return result
