from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import uproot

from SMEFT.application import BORN_KINEMATICS, METADATA_NAME, decorate_file
from SMEFT.reweight import expand_inputs, main
from Tools.born_projection import BORN_PROJECTION_METADATA


class Backend:
    metadata = {"backend": "deterministic fixture", "scope": "LO decay-ME deformation"}

    def __init__(self):
        self.calls = []

    def evaluate(self, momenta, pdg_ids):
        self.calls.append((momenta.copy(), pdg_ids.copy()))
        return np.full(len(momenta), .2), np.full(len(momenta), .1)


def fixture(path: Path, sources=(2, 1, 3, 2), *, tau=False, unavailable=False,
            extra_tree=True, metadata_change=None, column_change=None):
    size = len(sources)
    source = np.asarray(sources, dtype=np.int8)
    identifiers = np.where(source == 1, 1, 2).astype(np.uint64)
    columns = {
        "event_id": np.arange(size, dtype=np.uint64),
        "weight": np.resize(np.array([2., -3., 4., -1.]), size),
        "born_source": source, "born_available": np.ones(size, dtype=np.bool_),
        "born_kinematics_valid": np.ones(size, dtype=np.bool_),
        "born_type": np.full(size, 2, dtype=np.int8),
        "born_n_taus": np.zeros(size, dtype=np.int8),
        "born_shower_event_index": np.where(source == 1, -1, np.arange(size) + 1).astype(np.int64),
        "lhe_available": np.ones(size, dtype=np.bool_), "lhe_source_id": identifiers,
        # Repeated source events are normal and must remain distinct rows.
        "lhe_event_index": np.ones(size, dtype=np.int64),
        "omnifold_truth_push": np.tile([1.2, .9], (size, 1)),
    }
    coordinates = {"m_ZZ": 125., "m_Z1": 81., "m_Z2": 29., "cos_theta1": .32,
                   "cos_theta2": -.57, "Phi": 1.12, "cos_theta_star": .1,
                   "Phi1": .3, "Psi": .86, "y_ZZ": .2, "pT_ZZ": 0.}
    for name in BORN_KINEMATICS:
        columns["born_" + name] = np.full(size, coordinates[name])
    background = source == 1
    columns["born_available"][background] = False
    columns["born_kinematics_valid"][background] = False
    columns["born_type"][background] = -1
    for name in BORN_KINEMATICS:
        columns["born_" + name][background] = np.nan
    if tau:
        columns["born_type"][0] = 5
        columns["born_n_taus"][0] = 2
    if unavailable:
        columns["born_available"][-1] = False
        columns["born_kinematics_valid"][-1] = False
        columns["born_type"][-1] = -1
    records = []
    for code in sorted(set(sources)):
        identifier = 1 if code == 1 else 2
        record = {"lhe_source_id": f"{identifier:016x}", "lhe_sha256": f"{identifier:016x}" + "0" * 48,
                  "lhe_event_count": max(size, 1), "process": "ZZ" if code == 1 else "gg_H",
                  "shower": "pythia" if code == 2 else "herwig", "born_source": code,
                  "hard_decay_definition": {1: "lhe_status1_charged_leptons", 2: "pythia_event_before_decay_fsr", 3: "herwig_qtilde_original_decay"}[code]}
        records.append(record)
    strings = {
        "born_metadata": {"schema_version": 1, "projection": deepcopy(BORN_PROJECTION_METADATA), "sources": deepcopy(records)},
        "lhe_metadata": {"schema_version": 1, "event_number": "lhe_event_index_1based", "sources": deepcopy(records)},
        "merge_metadata": {"components": {"ZZ": {"observed_entries": int(background.sum())}, "gg_H": {"observed_entries": int((~background).sum())}}},
    }
    if metadata_change:
        metadata_change(strings)
    if column_change:
        column_change(columns)
    schema = {name: np.dtype((values.dtype, values.shape[1:])) if values.ndim > 1 else values.dtype for name, values in columns.items()}
    with uproot.recreate(path) as root:
        root.mktree("Analysis", schema, title="Original analysis title")
        root["Analysis"].extend(columns)
        if extra_tree:
            root.mktree("Counters", {"count": np.dtype(np.int64)})
            root["Counters"].extend({"count": np.array([4, 7], dtype=np.int64)})
        for name, value in strings.items():
            root[name] = json.dumps(value, sort_keys=True)
        root["notes"] = "Original arbitrary metadata stays byte-for-byte text."
    return columns


class ApplicationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source, self.output = self.directory / "data.root", self.directory / "decorated.root"

    def test_mixed_signal_only_scan_keeps_signed_weights_all_rows_trees_and_metadata(self):
        original = fixture(self.source)
        backend = Backend()
        result = decorate_file(self.source, self.output, [-2., 0., 2.], backend=backend, lambda_tev=2., step_size=2)
        expected_ratio = np.array([.925, 1., 1.125])
        with uproot.open(self.source) as source, uproot.open(self.output) as output:
            arrays = output["Analysis"].arrays(library="np", how=dict)
            for name, values in original.items():
                np.testing.assert_array_equal(arrays[name], values)
            np.testing.assert_allclose(arrays["smeft_hb_ratio"][[0, 2, 3]], np.tile(expected_ratio, (3, 1)))
            np.testing.assert_array_equal(arrays["smeft_hb_ratio"][1], [1., 1., 1.])
            np.testing.assert_allclose(arrays["smeft_hb_weight"], original["weight"][:, None] * arrays["smeft_hb_ratio"])
            np.testing.assert_array_equal(arrays["smeft_hb_status"], [0, 1, 0, 0])
            self.assertTrue(np.all(arrays["smeft_hb_valid"]))
            self.assertEqual(output["Analysis"].title, source["Analysis"].title)
            np.testing.assert_array_equal(output["Counters"]["count"].array(library="np"), [4, 7])
            for key in ("born_metadata", "lhe_metadata", "merge_metadata", "notes"):
                self.assertEqual(str(source[key]), str(output[key]))
            self.assertEqual(json.loads(str(output[METADATA_NAME])), result)
        self.assertEqual(sum(len(p) for p, _ in backend.calls), 3)
        self.assertEqual(result["status_counts"]["background_unity"], 1)
        self.assertEqual(result["effective_coefficients"], [-.5, 0., .5])

    def test_unsupported_higgs_defaults_to_atomic_error_or_explicit_flagged_unity(self):
        original = fixture(self.source, tau=True, unavailable=True)
        with self.assertRaisesRegex(ValueError, "unsupported Higgs rows"):
            decorate_file(self.source, self.output, [1.], backend=Backend())
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.directory.glob(".*.smeft-*")), [])
        backend = Backend()
        decorate_file(self.source, self.output, [1.], backend=backend, unsupported="unity")
        with uproot.open(self.output) as root:
            arrays = root["Analysis"].arrays(library="np", how=dict)
        np.testing.assert_array_equal(arrays["smeft_hb_status"], [3, 1, 0, 2])
        np.testing.assert_array_equal(arrays["smeft_hb_valid"], [False, True, True, False])
        np.testing.assert_array_equal(arrays["smeft_hb_weight"][[0, 1, 3], 0], original["weight"][[0, 1, 3]])
        self.assertEqual(sum(len(p) for p, _ in backend.calls), 1)

    def test_identity_mismatch_collision_missing_metadata_and_unknown_source_fail_closed(self):
        mutations = (
            lambda records: records.pop("born_metadata"),
            lambda records: records["born_metadata"].update(projection={"method": "other"}),
            lambda records: records["born_metadata"]["sources"][0].update(process="gg_H"),
            lambda records: records["born_metadata"]["sources"][1].update(born_source=1),
            lambda records: records["lhe_metadata"]["sources"][2].update(lhe_sha256="0000000000000002" + "1" * 48),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                fixture(self.source, metadata_change=mutation)
                with self.assertRaises(ValueError):
                    decorate_file(self.source, self.output, [0.], backend=Backend(), unsupported="unity")
                self.assertFalse(self.output.exists())
        fixture(self.source, column_change=lambda c: c["born_source"].__setitem__(0, 0))
        with self.assertRaisesRegex(ValueError, "born_source"):
            decorate_file(self.source, self.output, [0.], backend=Backend(), unsupported="unity")
        fixture(self.source, column_change=lambda c: c["lhe_source_id"].__setitem__(0, 9))
        with self.assertRaisesRegex(ValueError, "absent"):
            decorate_file(self.source, self.output, [0.], backend=Backend())

    def test_existing_smeft_and_output_collisions_never_overwrite_input(self):
        fixture(self.source)
        before = self.source.read_bytes()
        with self.assertRaisesRegex(ValueError, "different"):
            decorate_file(self.source, self.source, [0.], backend=Backend(), overwrite=True)
        self.output.hardlink_to(self.source)
        with self.assertRaisesRegex(ValueError, "different"):
            decorate_file(self.source, self.output, [0.], backend=Backend(), overwrite=True)
        self.output.unlink()
        decorate_file(self.source, self.output, [0.], backend=Backend())
        with self.assertRaises(FileExistsError):
            decorate_file(self.source, self.output, [0.], backend=Backend())
        with self.assertRaisesRegex(ValueError, "already contains SMEFT"):
            decorate_file(self.output, self.directory / "twice.root", [0.], backend=Backend())
        self.assertEqual(self.source.read_bytes(), before)

    def test_nonfinite_input_or_backend_failure_leaves_previous_output_unchanged(self):
        fixture(self.source)
        self.output.write_bytes(b"existing output")
        class BadBackend(Backend):
            def evaluate(self, momenta, pdg_ids):
                return np.full(len(momenta), np.nan), np.zeros(len(momenta))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            decorate_file(self.source, self.output, [1.], backend=BadBackend(), overwrite=True)
        self.assertEqual(self.output.read_bytes(), b"existing output")
        fixture(self.source, column_change=lambda c: c["weight"].__setitem__(0, np.nan))
        with self.assertRaisesRegex(ValueError, "finite"):
            decorate_file(self.source, self.output, [1.], backend=Backend(), overwrite=True)
        self.assertEqual(self.output.read_bytes(), b"existing output")

    def test_no_signal_rows_skip_backend_and_remain_valid_background(self):
        fixture(self.source, sources=(1, 1))
        backend = Backend()
        result = decorate_file(self.source, self.output, [-1., 1.], backend=backend)
        self.assertEqual(backend.calls, [])
        self.assertEqual(result["status_counts"]["background_unity"], 2)

    def test_empty_tree_retains_vector_schema_without_backend_calls(self):
        fixture(self.source, sources=())
        backend = Backend()
        result = decorate_file(self.source, self.output, [-1., 0., 1.], backend=backend)
        self.assertEqual(result["entries"], 0)
        self.assertEqual(backend.calls, [])
        with uproot.open(self.output) as root:
            self.assertEqual(root["Analysis"]["smeft_hb_ratio"].array(library="np").shape, (0, 3))
            self.assertEqual(root["Analysis"]["omnifold_truth_push"].array(library="np").shape, (0, 2))

    def test_input_mutation_and_unsupported_root_objects_do_not_publish(self):
        fixture(self.source)
        source = self.source
        class MutatingBackend(Backend):
            def evaluate(self, momenta, pdg_ids):
                with source.open("ab") as stream:
                    stream.write(b"changed during evaluation")
                return super().evaluate(momenta, pdg_ids)
        with self.assertRaisesRegex(RuntimeError, "input changed"):
            decorate_file(self.source, self.output, [1.], backend=MutatingBackend())
        self.assertFalse(self.output.exists())
        fixture(self.source)
        with uproot.update(self.source) as root:
            root["histogram"] = np.histogram([1., 2., 3.], bins=3)
        with self.assertRaisesRegex(TypeError, "refusing to drop"):
            decorate_file(self.source, self.output, [1.], backend=Backend())
        self.assertFalse(self.output.exists())

    def test_custom_weight_and_invalid_coefficient_arguments(self):
        fixture(self.source, column_change=lambda c: c.update(custom_weight=c["weight"] * 2))
        decorate_file(self.source, self.output, [0.], backend=Backend(), weight_branch="custom_weight")
        with uproot.open(self.output) as root:
            np.testing.assert_array_equal(root["Analysis"]["smeft_hb_weight"].array(library="np")[:, 0], [4., -6., 8., -2.])
        for points, scale in (([], 1.), ([np.nan], 1.), ([1.], 0.), ([1.], 1.e300)):
            with self.subTest(points=points, scale=scale), self.assertRaises(ValueError):
                decorate_file(self.source, self.directory / "bad.root", points, backend=Backend(), lambda_tev=scale)

    def test_cli_expands_directory_and_glob_and_passes_build_directory(self):
        inputs, output = self.directory / "inputs", self.directory / "outputs"
        inputs.mkdir()
        fixture(inputs / "first.root")
        fixture(inputs / "second.root")
        self.assertEqual(expand_inputs([str(inputs), str(inputs / "*.root")]), sorted(inputs.glob("*.root")))
        backend = Backend()
        with patch("SMEFT.backend.NativeHBBackend", return_value=backend) as factory, redirect_stdout(io.StringIO()) as stdout:
            result = main([str(inputs), "--output-dir", str(output), "--cHB", "-1", "0", "1", "--build-dir", str(self.directory / "build")])
        self.assertEqual(result, 0)
        factory.assert_called_once_with(build_dir=self.directory / "build")
        self.assertEqual({path.name for path in output.glob("*.root")}, {"first.root", "second.root"})
        self.assertEqual(len(stdout.getvalue().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
