#!/usr/bin/env python3
"""Reduce Delphes outputs to one compact truth/reconstruction event tree."""

from __future__ import annotations

import argparse
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

INPUT_BRANCHES = (
    "Event.Number",
    "Event.Weight",
    "Event.CrossSection",
    "Particle.PID",
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
    "RecoElectron.PT",
    "RecoElectron.Eta",
    "RecoElectron.Phi",
    "RecoElectron.Charge",
    "RecoMuon.PT",
    "RecoMuon.Eta",
    "RecoMuon.Phi",
    "RecoMuon.Charge",
)


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
    lepton_m1: list[int],
    lepton_m2: list[int],
    particle_pid: list[int],
    particle_m1: list[int],
    particle_m2: list[int],
) -> list[bool]:
    size = len(particle_pid)

    def mothers(first: int, last: int) -> range:
        if first < 0:
            return range(last, last + 1) if 0 <= last < size else range(0)
        stop = last if last >= first else first
        return range(first, min(stop, size - 1) + 1)

    @lru_cache(maxsize=None)
    def has_hadron_ancestor(index: int) -> bool:
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

    return [
        not any(has_hadron_ancestor(index) for index in mothers(int(first), int(last)))
        for first, last in zip(lepton_m1, lepton_m2)
    ]


def _truth_leptons(rows: dict[str, list], event: int) -> list[Lepton]:
    particle_pid = rows["Particle.PID"][event]
    particle_m1 = rows["Particle.M1"][event]
    particle_m2 = rows["Particle.M2"][event]
    leptons: list[Lepton] = []
    for branch, flavor in (("DressedElectron", "electron"), ("DressedMuon", "muon")):
        pids = rows[f"{branch}.PID"][event]
        mask = _prompt_mask(
            rows[f"{branch}.M1"][event],
            rows[f"{branch}.M2"][event],
            particle_pid,
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


def reduce_chunk(
    arrays: dict[str, ak.Array],
    *,
    first_event_id: int,
    four_lepton_mass_window: tuple[float, float],
) -> dict[str, np.ndarray]:
    rows = {name: ak.to_list(array) for name, array in arrays.items()}
    size = len(rows["Event.Number"])
    output = _empty_output(size, first_event_id)

    for event in range(size):
        output["event_number"][event] = int(_first(rows["Event.Number"][event], event))
        output["weight"][event] = float(_first(rows["Event.Weight"][event], 1.0))
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
        output["fiducial"][event] = truth.selected
        output["reconstructed"][event] = reco.selected
        if truth.candidate is not None:
            output["type"][event] = truth.candidate.event_type
        elif reco.candidate is not None:
            output["type"][event] = reco.candidate.event_type
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
    fiducial_count = 0
    reconstructed_count = 0
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
                        "rerun Delphes with the current fiducial-study card"
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
                    )
                    output_tree.extend(reduced)
                    size = len(reduced["event_id"])
                    event_id += size
                    fiducial_count += int(np.count_nonzero(reduced["fiducial"]))
                    reconstructed_count += int(np.count_nonzero(reduced["reconstructed"]))

    print(f"Wrote {event_id} events to {output_path}")
    print(f"Fiducial: {fiducial_count}; reconstructed and selected: {reconstructed_count}")


if __name__ == "__main__":
    main()
