"""Train a varied/nominal endpoint with the existing NSBI ratio estimator.

These are marginal reconstructed-space ratios at a fixed truth mixture, not
conditional detector-response ratios for a Profile OmniFold iteration.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np
import uproot

from Analysis.simulation_metadata import merge_simulation_metadata
from BackgroundRemoval import common
from omnifold.ratios import RatioConfig, RatioEstimator

REGIONS = {"signal": (115.0, 130.0), "sideband": (130.0, 160.0),
           "extended": (105.0, 160.0)}


@dataclass
class Sample:
    features: np.ndarray
    weights: np.ndarray
    splits: np.ndarray
    provenance: dict
    processes: set[str]
    luminosity_fb: float
    source_digests: dict[int, str]


def read_sample(path: Path, *, shower: str, scale: str, region: str,
                fiducial: bool = False, split_seed: int = 12345) -> Sample:
    """Read one merged MC sample; verify physics roles from retained provenance.

    All occurrences of the same LHE source event share their split, including
    across independently reduced nominal and detector-varied samples.
    """
    path = Path(path).expanduser().resolve()
    provenance = common.capture_file_provenance(path)
    common.assert_file_unchanged(path, provenance)
    simulation = merge_simulation_metadata([path])
    if simulation["muon_scale_variation"] != scale or simulation["legacy_nominal_inputs"]:
        raise ValueError(f"{path}: expected explicit muon-scale {scale} metadata")
    features, weights, splits = [], [], []
    luminosity = None
    low, high = REGIONS[region]
    with uproot.open(path) as source:
        if any(name in source for name in ("smeft_hb_metadata", "background_removal_metadata",
                                            "omnifold_metadata")):
            raise ValueError("Systematic endpoints require original, undeformed merged MC")
        if "merge_metadata" not in source or "lhe_metadata" not in source:
            raise ValueError("Merged MC normalization and LHE provenance are required")
        merged = json.loads(str(source["merge_metadata"]))
        if (merged.get("weight_semantics") != "expected_events_at_luminosity"
                or "ensemble_index" in merged or "components" in merged):
            raise ValueError("Systematic endpoints require expected-event MC, not pseudo-data")
        lhe = json.loads(str(source["lhe_metadata"]))
        records = lhe.get("sources", [])
        if lhe.get("schema_version") != 1 or not records:
            raise ValueError("Missing or incompatible LHE source records")
        processes, source_digests = set(), {}
        for record in records:
            process = record.get("process")
            if process not in {"gg_H", "ZZ"} or record.get("shower") != shower:
                raise ValueError(f"{path}: source process/shower does not match {shower}")
            processes.add(process)
            digest = record.get("lhe_sha256", "")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Invalid LHE source checksum")
            identifier = int(record["lhe_source_id"], 16)
            if identifier != int(digest[:16], 16):
                raise ValueError("LHE source identifier disagrees with checksum")
            if source_digests.setdefault(identifier, digest) != digest:
                raise ValueError("LHE source identifier collision")
        if fiducial and processes != {"gg_H"}:
            raise ValueError("Fiducial-signal selection requires a pure gg_H sample")
        tree = source[common.TREE_NAME]
        branches = [*common.FEATURES, "weight", "luminosity_fb", "reconstructed",
                    "reco_m_ZZ", "lhe_available", "lhe_source_id", "lhe_event_index"]
        if fiducial:
            branches.append("fiducial")
        for raw in tree.iterate(branches, step_size="64 MB", library="np", how=dict):
            arrays = raw if isinstance(raw, dict) else {key: raw[key] for key in raw.dtype.names}
            lumi = np.asarray(arrays["luminosity_fb"], dtype=np.float64)
            if lumi.size:
                luminosity = float(lumi[0]) if luminosity is None else luminosity
                if (not np.isfinite(luminosity) or luminosity <= 0
                        or not np.all(np.isclose(lumi, luminosity, rtol=0, atol=1e-9))):
                    raise ValueError("Inconsistent MC luminosity")
            mass = np.asarray(arrays["reco_m_ZZ"], dtype=np.float64)
            reco = np.asarray(arrays["reconstructed"], dtype=bool)
            if np.any(reco & ~np.isfinite(mass)):
                raise ValueError("Reconstructed events require a finite mass")
            keep = reco & (mass > low) & (mass < high)
            if fiducial:
                keep &= np.asarray(arrays["fiducial"], dtype=bool)
            if not np.any(keep):
                continue
            ids = np.asarray(arrays["lhe_source_id"][keep])
            indices = np.asarray(arrays["lhe_event_index"][keep])
            if (ids.dtype.kind not in "iu" or indices.dtype.kind not in "iu"
                    or not np.all(arrays["lhe_available"][keep]) or np.any(indices < 1)
                    or not np.all(np.isin(ids, list(source_digests)))):
                raise ValueError("Selected events require valid source identities")
            values = np.column_stack([arrays[name][keep] for name in common.FEATURES]).astype(np.float32)
            event_weights = np.asarray(arrays["weight"][keep], dtype=np.float64)
            if not np.all(np.isfinite(values)) or not np.all(np.isfinite(event_weights)):
                raise ValueError("Nonfinite selected features or weights")
            # Integer arithmetic preserves the full source ID (no float conversion).
            group = ids.astype(np.uint64) ^ (indices.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15))
            features.append(values)
            weights.append(event_weights)
            # The salt is fixed for both classes; source_id already separates
            # distinct hard-event samples and processes.
            splits.append(common.deterministic_split(group, source="gg_H", seed=split_seed))
    common.assert_file_unchanged(path, provenance)
    if not weights:
        raise ValueError(f"{path}: no selected systematic events")
    provenance["simulation_metadata"] = simulation
    return Sample(np.concatenate(features), np.concatenate(weights), np.concatenate(splits),
                  provenance, processes, luminosity, source_digests)


def train(nominal: Path, varied: Path, *, variation: str, output_dir: Path,
          config: RatioConfig | None = None, region: str = "signal",
          fiducial: bool = False, split_seed: int = 12345, reports: bool = True):
    """Fit target=varied, reference=nominal, retaining a separate net-yield ratio."""
    if variation not in {"theory", "muon_up", "muon_down"}:
        raise ValueError("variation must be theory, muon_up, or muon_down")
    if Path(nominal).resolve() == Path(varied).resolve():
        raise ValueError("Nominal and varied inputs must be different files")
    config = config or RatioConfig(hidden_layers=4, neurons=1024, batch_size=8192, weight_decay=0.0)
    config.validate()
    common.validate_training_weight_decay(config.weight_decay)
    if config.overwrite:
        raise ValueError("Use a new systematic output directory; overwrite is not supported")
    output_dir = Path(output_dir).expanduser().resolve()
    for path in (nominal, varied):
        resolved = Path(path).resolve()
        if output_dir == resolved or output_dir in resolved.parents:
            raise ValueError("Output must not contain or alias an input")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    reference = read_sample(nominal, shower="pythia", scale="nominal", region=region,
                            fiducial=fiducial, split_seed=split_seed)
    target = read_sample(varied, shower="herwig" if variation == "theory" else "pythia",
                         scale="nominal" if variation == "theory" else variation.removeprefix("muon_"),
                         region=region, fiducial=fiducial, split_seed=split_seed)
    if reference.processes != target.processes or not np.isclose(
            reference.luminosity_fb, target.luminosity_fb, rtol=0, atol=1e-9):
        raise ValueError("Nominal and varied process content and luminosity must match")
    for identifier in reference.source_digests.keys() & target.source_digests.keys():
        if reference.source_digests[identifier] != target.source_digests[identifier]:
            raise ValueError("LHE source identifier collision across systematic endpoints")
    metadata = {
        "purpose": "systematic_endpoint", "variation": variation,
        "nuisance": "theory" if variation == "theory" else "experimental",
        "endpoint": -1 if variation == "muon_down" else 1,
        "nominal": reference.provenance, "varied": target.provenance,
        "processes": sorted(reference.processes), "luminosity_fb": reference.luminosity_fb,
        "region": region, "mass_window_open_gev": list(REGIONS[region]),
        "fiducial_signal": fiducial, "split_seed": split_seed,
        "split_identity": "LHE SHA256 prefix and source event ordinal; shared across endpoints",
        "scope": "marginal reconstructed density at the supplied truth mixture; not a conditional response ratio",
    }
    estimator = RatioEstimator.fit(
        target.features, target.weights, reference.features, reference.weights,
        target_splits=target.splits, reference_splits=reference.splits,
        output_dir=output_dir, config=config, metadata=metadata)
    if reports:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from omnifold.diagnostics import plot_ratio_diagnostics
        from omnifold.reporting import save_figures
        figures = plot_ratio_diagnostics(
            estimator, target.features, target.weights, reference.features, reference.weights,
            target_splits=target.splits, reference_splits=reference.splits, title=variation)
        try:
            save_figures(figures, output_dir / "reports")
        finally:
            for figure in figures.values():
                plt.close(figure)
    return estimator


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal", type=Path, required=True)
    parser.add_argument("--varied", type=Path, required=True)
    parser.add_argument("--variation", choices=("theory", "muon_up", "muon_down"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--region", choices=tuple(REGIONS), default="signal")
    parser.add_argument("--fiducial-signal", action="store_true")
    parser.add_argument("--split-seed", type=int, default=12345)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--seed", type=int, default=19110907)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--neurons", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--maximum-epochs", type=int, default=100)
    parser.add_argument("--no-reports", action="store_true")
    args = parser.parse_args(argv)
    if args.cpu_threads < 1 or args.split_seed < 0:
        parser.error("CPU threads must be positive and split seed nonnegative")
    import torch
    torch.set_num_threads(args.cpu_threads)
    config = RatioConfig(hidden_layers=args.hidden_layers, neurons=args.neurons,
                         batch_size=args.batch_size, maximum_epochs=args.maximum_epochs,
                         seed=args.seed, device=args.device, weight_decay=0.0)
    estimator = train(args.nominal, args.varied, variation=args.variation,
                       output_dir=args.output_dir, config=config, region=args.region,
                       fiducial=args.fiducial_signal, split_seed=args.split_seed,
                       reports=not args.no_reports)
    print(json.dumps({"artifact": str(estimator.root), "yield_ratio": estimator.yield_ratio}))


if __name__ == "__main__":
    main()
