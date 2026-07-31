#!/usr/bin/env python3
"""Reduce Delphes outputs to one compact truth/reconstruction event tree."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import sys
from typing import Iterable

import awkward as ak
import numpy as np
import uproot
import vector

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Analysis.selection import Lepton, SelectionResult, select_event  # noqa: E402
from Tools.four_lepton_kinematics import KinematicError  # noqa: E402


KINEMATIC_FIELDS = (
    "cos_theta_star",
    "cos_theta1",
    "cos_theta2",
    "Phi",
    "Phi1",
    "Psi",
    "m_Z1",
    "m_Z2",
    "m_ZZ",
    "y_ZZ",
    "pT_ZZ",
)

CHANNEL_NAMES = {
    0: "4mu",
    1: "2mu2e",
    2: "2e2mu",
    3: "4e",
}

INPUT_BRANCHES = (
    "Event.Number",
    "Event.Weight",
    "Event.CrossSection",
    "Particle.PID",
    "Particle.E",
    "Particle.Px",
    "Particle.Py",
    "Particle.Pz",
    "Particle.M1",
    "Particle.M2",
    "DressedElectron.PID",
    "DressedElectron.M1",
    "DressedElectron.M2",
    "DressedElectron.E",
    "DressedElectron.Px",
    "DressedElectron.Py",
    "DressedElectron.Pz",
    "DressedMuon.PID",
    "DressedMuon.M1",
    "DressedMuon.M2",
    "DressedMuon.E",
    "DressedMuon.Px",
    "DressedMuon.Py",
    "DressedMuon.Pz",
    # These pre-isolation branches are a schema/version sentinel for the
    # dedicated H4l response card. The compact output does not copy them.
    "RecoElectronNoIso.PT",
    "RecoMuonNoIso.PT",
    "RecoElectron.PT",
    "RecoElectron.Eta",
    "RecoElectron.Phi",
    "RecoElectron.Charge",
    "RecoMuon.PT",
    "RecoMuon.Eta",
    "RecoMuon.Phi",
    "RecoMuon.Charge",
)


@dataclass
class SelectionCounts:
    fiducial: int = 0
    reconstructed: int = 0
    both: int = 0
    fiducial_only: int = 0
    reconstructed_only: int = 0


@dataclass
class SelectionWeightSums:
    fiducial: float = 0.0
    reconstructed: float = 0.0
    both: float = 0.0
    fiducial_only: float = 0.0
    reconstructed_only: float = 0.0


class SelectionDiagnostics:
    """Accumulate count- and nominal-weight selection overlap diagnostics."""

    def __init__(self) -> None:
        self.total = 0
        self.overall = SelectionCounts()
        self.by_channel = {channel: SelectionCounts() for channel in CHANNEL_NAMES}
        self.overall_weighted = SelectionWeightSums()
        self.by_channel_weighted = {
            channel: SelectionWeightSums() for channel in CHANNEL_NAMES
        }

    @staticmethod
    def _add_overlap(
        target: SelectionCounts | SelectionWeightSums,
        truth_selected: bool,
        reco_selected: bool,
        value: int | float,
    ) -> None:
        if truth_selected:
            target.fiducial += value  # type: ignore[assignment]
        if reco_selected:
            target.reconstructed += value  # type: ignore[assignment]
        if truth_selected and reco_selected:
            target.both += value  # type: ignore[assignment]
        elif truth_selected:
            target.fiducial_only += value  # type: ignore[assignment]
        elif reco_selected:
            target.reconstructed_only += value  # type: ignore[assignment]

    def add(
        self, truth: SelectionResult, reco: SelectionResult, *, weight: float = 1.0
    ) -> None:
        if not np.isfinite(weight):
            raise ValueError("selection diagnostics require a finite event weight")
        self.total += 1
        truth_selected = truth.selected
        reco_selected = reco.selected

        self._add_overlap(self.overall, truth_selected, reco_selected, 1)
        self._add_overlap(
            self.overall_weighted, truth_selected, reco_selected, float(weight)
        )

        if truth_selected:
            if truth.candidate is None:
                raise RuntimeError("selected truth event has no four-lepton candidate")
            truth_counts = self.by_channel[truth.candidate.event_type]
            truth_weighted = self.by_channel_weighted[truth.candidate.event_type]
            truth_counts.fiducial += 1
            truth_counts.both += int(reco_selected)
            truth_counts.fiducial_only += int(not reco_selected)
            truth_weighted.fiducial += weight
            truth_weighted.both += weight * int(reco_selected)
            truth_weighted.fiducial_only += weight * int(not reco_selected)

        if reco_selected:
            if reco.candidate is None:
                raise RuntimeError("selected reco event has no four-lepton candidate")
            reco_counts = self.by_channel[reco.candidate.event_type]
            reco_weighted = self.by_channel_weighted[reco.candidate.event_type]
            reco_counts.reconstructed += 1
            reco_counts.reconstructed_only += int(not truth_selected)
            reco_weighted.reconstructed += weight
            reco_weighted.reconstructed_only += weight * int(not truth_selected)


def _ratio(numerator: int | float, denominator: int | float) -> str:
    return "n/a" if denominator == 0 else f"{numerator / denominator:.4f}"


def diagnostic_lines(diagnostics: SelectionDiagnostics) -> list[str]:
    """Format overlap and unfolding-oriented selection metrics for the CLI."""

    overall = diagnostics.overall
    neither = (
        diagnostics.total
        - overall.both
        - overall.fiducial_only
        - overall.reconstructed_only
    )
    lines = [
        (
            f"Fiducial: {overall.fiducial}; reconstructed and selected: "
            f"{overall.reconstructed}"
        ),
        (
            f"Overlap: both={overall.both}; "
            f"fiducial-only={overall.fiducial_only}; "
            f"reconstructed-only={overall.reconstructed_only}; neither={neither}"
        ),
        "Selection diagnostics (unweighted event counts):",
        (
            "channel    Nfid   Nreco   Nboth   Nreco-only"
            "   C_count   eff_count   leakage_count"
        ),
    ]
    rows = [("all", overall), *(
        (CHANNEL_NAMES[channel], diagnostics.by_channel[channel])
        for channel in CHANNEL_NAMES
    )]
    for label, counts in rows:
        lines.append(
            f"{label:<8} "
            f"{counts.fiducial:7d} "
            f"{counts.reconstructed:7d} "
            f"{counts.both:7d} "
            f"{counts.reconstructed_only:12d} "
            f"{_ratio(counts.reconstructed, counts.fiducial):>15} "
            f"{_ratio(counts.both, counts.fiducial):>17} "
            f"{_ratio(counts.reconstructed_only, counts.reconstructed):>26}"
        )
    lines.extend(
        [
            "Selection diagnostics (sum of nominal event weights):",
            (
                "channel       sumWfid      sumWreco      sumWboth"
                "   sumWreco-only   C_weight   eff_weight   leakage_weight"
            ),
        ]
    )
    weighted_rows = [("all", diagnostics.overall_weighted), *(
        (CHANNEL_NAMES[channel], diagnostics.by_channel_weighted[channel])
        for channel in CHANNEL_NAMES
    )]
    for label, sums in weighted_rows:
        lines.append(
            f"{label:<8} "
            f"{sums.fiducial:13.6g} "
            f"{sums.reconstructed:13.6g} "
            f"{sums.both:13.6g} "
            f"{sums.reconstructed_only:15.6g} "
            f"{_ratio(sums.reconstructed, sums.fiducial):>10} "
            f"{_ratio(sums.both, sums.fiducial):>12} "
            f"{_ratio(sums.reconstructed_only, sums.reconstructed):>16}"
        )
    return lines


def available_branch_names(tree: object) -> set[str]:
    """Return recursive branch names without uproot's parent path prefixes."""
    return set(tree.keys(recursive=True, full_paths=False))  # type: ignore[attr-defined]


def discover_inputs(paths: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for item in paths:
        path = Path(item).expanduser().resolve()
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            direct = path / "delphes.root"
            if direct.is_file():
                files.append(direct)
            else:
                files.extend(sorted(path.rglob("delphes.root")))
        else:
            raise FileNotFoundError(f"input does not exist: {path}")
    unique = list(dict.fromkeys(files))
    if not unique:
        raise FileNotFoundError("no delphes.root inputs were found")
    return unique


def _is_hadron(pid: int) -> bool:
    absolute = abs(pid)
    return 100 <= absolute < 1_000_000 or absolute >= 1_000_000_000


def _is_parton(pid: int) -> bool:
    absolute = abs(pid)
    return absolute <= 6 or absolute == 21


def _prompt_mask(
    lepton_pid: list[int],
    lepton_m1: list[int],
    lepton_m2: list[int],
    particle_pid: list[int],
    particle_mass: list[float],
    particle_m1: list[int],
    particle_m2: list[int],
) -> list[bool]:
    """Select non-hadronic leptons from W/Z or hard virtual-photon decays.

    Status-copy records and intermediate taus are traversed, so
    ``Z/W -> tau -> e/mu`` is deliberately accepted. A W or Z terminates the
    relevant decay chain. A photon ancestor is accepted only above 5 GeV,
    retaining continuum ``gamma* -> ll`` while rejecting conversions.
    """

    size = len(particle_pid)

    def mothers(first: int, second: int) -> tuple[int, ...]:
        # Delphes stores two mother indices, not the endpoints of an inclusive
        # particle-index range. Treating M1..M2 as a range walks through
        # unrelated event-record entries and can invent hadron ancestors.
        indices: list[int] = []
        for index in (first, second):
            if 0 <= index < size and index not in indices:
                indices.append(index)
        return tuple(indices)

    @lru_cache(maxsize=None)
    def has_hadron_ancestor(index: int) -> bool:
        """Match Delphes's all-path hadron veto, stopping at incoming partons."""

        pending = [index]
        visited: set[int] = set()
        while pending:
            current = pending.pop()
            if current < 0 or current >= size or current in visited:
                continue
            visited.add(current)

            pid = int(particle_pid[current])
            if _is_hadron(pid):
                return True
            # Stop at the incoming hard-scatter parton. HepMC ancestry may
            # link that parton to a beam proton; following it further would
            # classify every hard-process lepton as a hadron-decay lepton.
            if _is_parton(pid):
                continue
            pending.extend(
                mother
                for mother in mothers(
                    int(particle_m1[current]), int(particle_m2[current])
                )
                if mother != current and mother not in visited
            )
        return False

    @lru_cache(maxsize=None)
    def has_boson_ancestor(index: int, absolute_lepton_pid: int) -> bool:
        """Follow the allowed decay path to W/Z or a hard virtual photon."""

        pending = [index]
        visited: set[int] = set()
        while pending:
            current = pending.pop()
            if current < 0 or current >= size or current in visited:
                continue
            visited.add(current)

            pid = int(particle_pid[current])
            if abs(pid) in {23, 24}:
                return True
            if abs(pid) == 22 and float(particle_mass[current]) > 5.0:
                return True
            # Match the Simulation-side definition: follow status copies of
            # the same lepton and explicitly allowed tau-decay chains only.
            # In particular, do not follow a low-mass conversion photon back
            # to a hard-process boson.
            if abs(pid) not in {absolute_lepton_pid, 15}:
                continue
            pending.extend(
                mother
                for mother in mothers(
                    int(particle_m1[current]), int(particle_m2[current])
                )
                if mother != current and mother not in visited
            )
        return False

    prompt: list[bool] = []
    for pid, first, last in zip(lepton_pid, lepton_m1, lepton_m2):
        ancestry_roots = mothers(int(first), int(last))
        has_ew_boson = any(
            has_boson_ancestor(index, abs(int(pid))) for index in ancestry_roots
        )
        has_hadron = any(has_hadron_ancestor(index) for index in ancestry_roots)
        prompt.append(has_ew_boson and not has_hadron)
    return prompt


def _truth_leptons(rows: dict[str, list], event: int) -> list[Lepton]:
    particle_pid = rows["Particle.PID"][event]
    particle_m1 = rows["Particle.M1"][event]
    particle_m2 = rows["Particle.M2"][event]
    particle_mass = [
        float(
            np.sqrt(
                max(
                    float(energy) ** 2
                    - float(px) ** 2
                    - float(py) ** 2
                    - float(pz) ** 2,
                    0.0,
                )
            )
        )
        for energy, px, py, pz in zip(
            rows["Particle.E"][event],
            rows["Particle.Px"][event],
            rows["Particle.Py"][event],
            rows["Particle.Pz"][event],
        )
    ]
    leptons: list[Lepton] = []
    for branch, flavor in (("DressedElectron", "electron"), ("DressedMuon", "muon")):
        pids = rows[f"{branch}.PID"][event]
        mask = _prompt_mask(
            pids,
            rows[f"{branch}.M1"][event],
            rows[f"{branch}.M2"][event],
            particle_pid,
            particle_mass,
            particle_m1,
            particle_m2,
        )
        for index, prompt in enumerate(mask):
            if not prompt:
                continue
            pid = int(pids[index])
            leptons.append(
                Lepton(
                    p4=vector.obj(
                        E=float(rows[f"{branch}.E"][event][index]),
                        px=float(rows[f"{branch}.Px"][event][index]),
                        py=float(rows[f"{branch}.Py"][event][index]),
                        pz=float(rows[f"{branch}.Pz"][event][index]),
                    ),
                    flavor=flavor,
                    charge=-1 if pid > 0 else 1,
                )
            )
    return leptons


def _reco_leptons(rows: dict[str, list], event: int) -> list[Lepton]:
    leptons: list[Lepton] = []
    for branch, flavor, mass in (
        ("RecoElectron", "electron", 0.00051099895),
        ("RecoMuon", "muon", 0.1056583755),
    ):
        for pt, eta, phi, charge in zip(
            rows[f"{branch}.PT"][event],
            rows[f"{branch}.Eta"][event],
            rows[f"{branch}.Phi"][event],
            rows[f"{branch}.Charge"][event],
        ):
            leptons.append(
                Lepton(
                    p4=vector.obj(pt=float(pt), eta=float(eta), phi=float(phi), mass=mass),
                    flavor=flavor,
                    charge=int(charge),
                )
            )
    return leptons


def _first(value: object, default: float | int) -> float | int:
    if isinstance(value, list):
        return value[0] if value else default
    return value  # type: ignore[return-value]


def _empty_output(size: int, first_event_id: int) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {
        "event_id": np.arange(first_event_id, first_event_id + size, dtype=np.uint64),
        "event_number": np.zeros(size, dtype=np.int64),
        "weight": np.ones(size, dtype=np.float64),
        "cross_section_pb": np.full(size, np.nan, dtype=np.float64),
        "fiducial": np.zeros(size, dtype=np.bool_),
        "reconstructed": np.zeros(size, dtype=np.bool_),
        "truth_type": np.full(size, -1, dtype=np.int8),
        "reco_type": np.full(size, -1, dtype=np.int8),
        "type": np.full(size, -1, dtype=np.int8),
    }
    for level in ("truth", "reco"):
        for field in KINEMATIC_FIELDS:
            output[f"{level}_{field}"] = np.full(size, np.nan, dtype=np.float32)
    return output


def _fill_kinematics(output: dict[str, np.ndarray], level: str, event: int, result: SelectionResult) -> None:
    if result.candidate is None:
        return
    try:
        kinematics = result.candidate.kinematics()
    except KinematicError:
        return
    for field in KINEMATIC_FIELDS:
        output[f"{level}_{field}"][event] = getattr(kinematics, field)


def _fill_event_types(
    output: dict[str, np.ndarray],
    event: int,
    truth: SelectionResult,
    reco: SelectionResult,
) -> None:
    """Store both pairing categories and a reco-first compatibility alias."""

    if truth.candidate is not None:
        output["truth_type"][event] = truth.candidate.event_type
    if reco.candidate is not None:
        output["reco_type"][event] = reco.candidate.event_type
        output["type"][event] = reco.candidate.event_type
    elif truth.candidate is not None:
        output["type"][event] = truth.candidate.event_type


def reduce_chunk(
    arrays: dict[str, ak.Array],
    *,
    first_event_id: int,
    four_lepton_mass_window: tuple[float, float],
    diagnostics: SelectionDiagnostics | None = None,
) -> dict[str, np.ndarray]:
    rows = {name: ak.to_list(array) for name, array in arrays.items()}
    size = len(rows["Event.Number"])
    output = _empty_output(size, first_event_id)

    for event in range(size):
        output["event_number"][event] = int(_first(rows["Event.Number"][event], event))
        event_weight = float(_first(rows["Event.Weight"][event], 1.0))
        output["weight"][event] = event_weight
        output["cross_section_pb"][event] = float(
            _first(rows["Event.CrossSection"][event], np.nan)
        )

        truth = select_event(
            _truth_leptons(rows, event),
            four_lepton_mass_window=four_lepton_mass_window,
        )
        reco = select_event(
            _reco_leptons(rows, event),
            four_lepton_mass_window=four_lepton_mass_window,
        )
        if diagnostics is not None:
            diagnostics.add(truth, reco, weight=event_weight)
        output["fiducial"][event] = truth.selected
        output["reconstructed"][event] = reco.selected
        _fill_event_types(output, event, truth, reco)
        _fill_kinematics(output, "truth", event, truth)
        _fill_kinematics(output, "reco", event, reco)
    return output


def output_schema() -> dict[str, np.dtype]:
    sample = _empty_output(0, 0)
    return {name: values.dtype for name, values in sample.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Delphes ROOT files or directories")
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--tree-name", default="Delphes")
    parser.add_argument("--step-size", default="50 MB", help="uproot chunk size")
    parser.add_argument(
        "--mass-region",
        choices=("extended", "signal"),
        default="extended",
        help="extended: 105<m4l<160 (default); signal: 115<m4l<130",
    )
    args = parser.parse_args()

    inputs = discover_inputs(args.inputs)
    output_path = args.output.expanduser().resolve()
    if output_path in inputs:
        raise ValueError("output file must not overwrite a Delphes input")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mass_window = (105.0, 160.0) if args.mass_region == "extended" else (115.0, 130.0)

    event_id = 0
    diagnostics = SelectionDiagnostics()
    with uproot.recreate(output_path) as output_file:
        output_file.mktree("Analysis", output_schema(), title="Compact H to four-lepton analysis tree")
        output_tree = output_file["Analysis"]
        for input_path in inputs:
            with uproot.open(input_path) as input_file:
                if args.tree_name not in input_file:
                    raise KeyError(f"{input_path} does not contain tree {args.tree_name}")
                tree = input_file[args.tree_name]
                missing = sorted(set(INPUT_BRANCHES).difference(available_branch_names(tree)))
                if missing:
                    raise KeyError(
                        f"{input_path} is missing required branches: {', '.join(missing)}; "
                        "rerun Delphes with the current H4l response card"
                    )
                for arrays in tree.iterate(
                    expressions=INPUT_BRANCHES,
                    step_size=args.step_size,
                    library="ak",
                    how=dict,
                ):
                    reduced = reduce_chunk(
                        arrays,
                        first_event_id=event_id,
                        four_lepton_mass_window=mass_window,
                        diagnostics=diagnostics,
                    )
                    output_tree.extend(reduced)
                    size = len(reduced["event_id"])
                    event_id += size

    print(f"Wrote {event_id} events to {output_path}")
    for line in diagnostic_lines(diagnostics):
        print(line)


if __name__ == "__main__":
    main()
