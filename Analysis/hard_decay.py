"""Read native Higgs decays before decay FSR, preserving shower event order.

PYTHIA supplies an explicit native-event snapshot attribute. HERWIG's pinned
QTilde configuration retains the original decay daughters in full HepMC.
Neither path substitutes stable leptons, tau decay products, or dressed truth
for the native charged daughters. Momentum components exposed here are GeV.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterator

import vector


PYTHIA_DEFINITION = "pythia_event_before_decay_fsr"
HERWIG_DEFINITION = "herwig_qtilde_original_decay"
PYTHIA_ATTRIBUTE = "FourLeptonHardDecay"
_CHARGED_LEPTONS = {11, 13, 15}
_REL_TOL = 1.e-6
_ABS_TOL = 1.e-7


@dataclass(frozen=True)
class HardDecayRecord:
    status: str
    source: str
    particles: tuple[tuple[int, object], ...] = ()
    parent: object | None = None
    bosons: tuple[object, ...] = ()

    @property
    def valid(self) -> bool:
        return self.status == "ok"


class _InvalidDecay(ValueError):
    pass


def _momentum(values, scale: float = 1.):
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise _InvalidDecay("invalid_momentum")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise _InvalidDecay("invalid_momentum")
    px, py, pz, energy = (float(value) * scale for value in values)
    if not all(math.isfinite(value) for value in (px, py, pz, energy)):
        raise _InvalidDecay("nonfinite_momentum")
    norm = math.hypot(px, py, pz)
    if energy <= 0 or norm > energy + _ABS_TOL + _REL_TOL * energy:
        raise _InvalidDecay("nonphysical_momentum")
    return vector.obj(px=px, py=py, pz=pz, E=energy)


def _components(momentum) -> tuple[float, float, float, float]:
    return float(momentum.px), float(momentum.py), float(momentum.pz), float(momentum.e)


def _closed(parent, children) -> bool:
    summed = tuple(sum(_components(child)[index] for child in children) for index in range(4))
    tolerance = _ABS_TOL + _REL_TOL * max(float(parent.E), *(float(child.E) for child in children))
    return all(abs(a - b) <= tolerance for a, b in zip(_components(parent), summed))


def _validate(parent, bosons, particles) -> None:
    if len(bosons) != 2 or len(particles) != 4:
        raise _InvalidDecay("invalid_decay_multiplicity")
    for index, boson in enumerate(bosons):
        pair = particles[2 * index:2 * index + 2]
        if any(type(pid) is not int or abs(pid) not in _CHARGED_LEPTONS for pid, _ in pair):
            raise _InvalidDecay("noncharged_or_secondary_decay")
        if pair[0][0] != -pair[1][0]:
            raise _InvalidDecay("non_ossf_z_decay")
        if not _closed(boson, [momentum for _, momentum in pair]):
            raise _InvalidDecay("z_momentum_nonclosure")
    if not _closed(parent, bosons) or not _closed(parent, [momentum for _, momentum in particles]):
        raise _InvalidDecay("higgs_momentum_nonclosure")


def _pythia_decay(event) -> HardDecayRecord:
    try:
        raw = event.attributes[PYTHIA_ATTRIBUTE]
    except KeyError:
        return HardDecayRecord("missing_native_attribute", PYTHIA_DEFINITION)
    # Attributes read from disk are initially UnparsedAttribute objects.
    if not isinstance(raw, str):
        try:
            raw = raw.astype(str)
        except (AttributeError, TypeError, ValueError):
            return HardDecayRecord("invalid_native_attribute", PYTHIA_DEFINITION)
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return HardDecayRecord("invalid_native_attribute", PYTHIA_DEFINITION)
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        return HardDecayRecord("unsupported_attribute_schema", PYTHIA_DEFINITION)
    if payload.get("source") != PYTHIA_DEFINITION:
        return HardDecayRecord("unexpected_attribute_source", PYTHIA_DEFINITION)
    if payload.get("momentum_unit") != "GEV":
        return HardDecayRecord("unexpected_attribute_units", PYTHIA_DEFINITION)
    status = payload.get("status")
    if not isinstance(status, str) or not status:
        return HardDecayRecord("invalid_attribute_status", PYTHIA_DEFINITION)
    if status != "ok":
        return HardDecayRecord(status, PYTHIA_DEFINITION)
    try:
        parent = _momentum(payload.get("parent"))
        raw_bosons = payload.get("bosons")
        raw_particles = payload.get("particles")
        if not isinstance(raw_bosons, list) or not isinstance(raw_particles, list):
            raise _InvalidDecay("invalid_decay_multiplicity")
        bosons = tuple(_momentum(item) for item in raw_bosons)
        particles = []
        for item in raw_particles:
            if not isinstance(item, list) or len(item) != 5 or type(item[0]) is not int:
                raise _InvalidDecay("invalid_native_particle")
            particles.append((item[0], _momentum(item[1:])))
        particles = tuple(particles)
        _validate(parent, bosons, particles)
    except _InvalidDecay as error:
        return HardDecayRecord(str(error), PYTHIA_DEFINITION)
    return HardDecayRecord("ok", PYTHIA_DEFINITION, particles, parent, bosons)


def _children(particle) -> list:
    vertex = particle.end_vertex
    return list(vertex.particles_out) if vertex is not None else []


def _parents(particle) -> list:
    vertex = particle.production_vertex
    return list(vertex.particles_in) if vertex is not None else []


def _decaying_copy(particle, pid: int):
    visited = set()
    while True:
        # HepMC particle IDs identify nodes within one event, not source events.
        if particle.id in visited:
            raise _InvalidDecay("cyclic_decay_history")
        visited.add(particle.id)
        children = _children(particle)
        copies = [child for child in children if child.pid == pid]
        if not copies:
            return particle, children
        if len(copies) != 1:
            raise _InvalidDecay("ambiguous_resonance_copy")
        if len(children) != 1:
            raise _InvalidDecay("radiative_resonance_copy")
        particle = copies[0]


def _herwig_decay(event) -> HardDecayRecord:
    # The config must explicitly certify QTilde/no YFS. Status codes are not
    # used: retained originals commonly have status 11 rather than 2 or 3.
    source = HERWIG_DEFINITION
    unit = str(event.momentum_unit)
    if unit.endswith(".GEV") or unit == "GEV":
        scale = 1.
    elif unit.endswith(".MEV") or unit == "MEV":
        scale = .001
    else:
        return HardDecayRecord("unsupported_hepmc_units", source)
    higgs = [particle for particle in event.particles if particle.pid == 25]
    roots = [particle for particle in higgs if not any(parent.pid == 25 for parent in _parents(particle))]
    if not roots:
        return HardDecayRecord("missing_higgs" if not higgs else "cyclic_decay_history", source)
    if len(roots) != 1:
        return HardDecayRecord("ambiguous_higgs", source)
    try:
        parent_node, z_nodes = _decaying_copy(roots[0], 25)
        if len(z_nodes) != 2 or any(node.pid != 23 for node in z_nodes):
            raise _InvalidDecay("non_zz_or_radiative_higgs_decay")
        parent = _momentum(_components(parent_node.momentum), scale)
        bosons = []
        particles = []
        daughter_ids = set()
        for z_node in z_nodes:
            z_decay, daughters = _decaying_copy(z_node, 23)
            if len(daughters) != 2:
                raise _InvalidDecay("non_two_body_or_radiative_z_decay")
            bosons.append(_momentum(_components(z_decay.momentum), scale))
            for daughter in daughters:
                if daughter.id in daughter_ids:
                    raise _InvalidDecay("shared_decay_daughter")
                daughter_ids.add(daughter.id)
                # Stop here even for taus and even if the lepton later showers.
                particles.append((int(daughter.pid), _momentum(_components(daughter.momentum), scale)))
        bosons, particles = tuple(bosons), tuple(particles)
        _validate(parent, bosons, particles)
    except _InvalidDecay as error:
        return HardDecayRecord(str(error), source)
    return HardDecayRecord("ok", source, particles, parent, bosons)


def extract_hard_decay(event, shower: str, definition: str) -> HardDecayRecord:
    """Return one native decay, or an explicit unavailable reason with no p4s."""
    if shower == "pythia" and definition == PYTHIA_DEFINITION:
        return _pythia_decay(event)
    if shower == "herwig" and definition == HERWIG_DEFINITION:
        return _herwig_decay(event)
    raise ValueError(f"Unsupported hard-decay shower/definition: {shower!r}/{definition!r}")


def iterate_hard_decays(hepmc_path: str | Path, shower: str, definition: str) -> Iterator[tuple[int, HardDecayRecord]]:
    """Read HepMC2/3 in accepted-event order; repeated source IDs remain rows.

    The caller must bind this stream to the validated HepMC/Delphes digests,
    compare complete event-number sequences and counts, and retain row order.
    A source-index dictionary is unsafe for independently reshowered decays.
    """
    import pyhepmc

    if (shower, definition) not in {("pythia", PYTHIA_DEFINITION), ("herwig", HERWIG_DEFINITION)}:
        raise ValueError(f"Unsupported hard-decay shower/definition: {shower!r}/{definition!r}")
    with pyhepmc.open(str(Path(hepmc_path).expanduser().resolve())) as reader:
        for event in reader:
            yield int(event.event_number), extract_hard_decay(event, shower, definition)
