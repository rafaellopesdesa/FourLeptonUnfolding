"""PDF publication and figure ownership checks shared by CLI and notebook."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from omnifold.reporting import save_figures, save_reports


def _figure():
    figure, axis = plt.subplots(figsize=(2, 2))
    axis.plot([0., 1.], [1., 2.])
    return figure


class _Session:
    def __init__(self, directory):
        self.run_dir = Path(directory)
        self.calls = []
        self.fail = None
        self.rows = [{"index": 1, "data_yield": 5.}, {"index": 2, "data_yield": 6.}]

    def history(self):
        return list(self.rows)

    def diagnostics(self, iteration=None):
        self.calls.append(("diagnostics", iteration))
        first = _figure()
        if self.fail == "diagnostics":
            _figure()
            raise RuntimeError("diagnostics failed after creating figures")
        return {"reco_pull_closure": first}

    def compare(self, iteration=None, *, shape_only=False):
        self.calls.append(("compare", iteration, shape_only))
        first = _figure()
        if self.fail == "compare":
            _figure()
            raise RuntimeError("comparison failed after creating figures")
        return {"reco": first, "truth": _figure()}


class ReportingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.addCleanup(plt.close, "all")

    def test_separate_vector_pdfs_sanitize_names_and_keep_notebook_figures_open(self):
        figures = {"reco pull": _figure(), "../../truth": _figure()}
        existing = set(plt.get_fignums())
        output = self.directory / "plots"
        paths = save_figures(figures, output)
        self.assertEqual(paths, [output / "reco_pull.pdf", output / "truth.pdf"])
        for path in paths:
            self.assertEqual(path.read_bytes()[:4], b"%PDF")
            # Plain plotted lines remain PDF vectors, not a raster screenshot.
            self.assertNotIn(b"/Subtype /Image", path.read_bytes())
        self.assertEqual(set(plt.get_fignums()), existing)
        self.assertEqual(sorted(output.iterdir()), sorted(paths))

    def test_collision_is_rejected_before_writing_and_empty_input_is_a_noop(self):
        figure = _figure()
        output = self.directory / "plots"
        with self.assertRaisesRegex(ValueError, "collide"):
            save_figures({"reco/pull": figure, "reco_pull": figure}, output)
        self.assertFalse(output.exists())
        self.assertTrue(plt.fignum_exists(figure.number))
        self.assertEqual(save_figures({}, output), [])
        self.assertFalse(output.exists())

    def test_failed_pdf_save_preserves_prior_file_removes_temporary_and_keeps_figure(self):
        output = self.directory / "plots"
        output.mkdir()
        destination = output / "reco.pdf"
        destination.write_bytes(b"previous complete PDF")
        figure = _figure()

        def fail_after_partial_write(path, **kwargs):
            Path(path).write_bytes(b"incomplete replacement")
            raise OSError("injected disk error")

        with patch.object(figure, "savefig", side_effect=fail_after_partial_write):
            with self.assertRaisesRegex(OSError, "disk error"):
                save_figures({"reco": figure}, output)
        self.assertEqual(destination.read_bytes(), b"previous complete PDF")
        self.assertEqual(list(output.iterdir()), [destination])
        self.assertTrue(plt.fignum_exists(figure.number))

    def test_reports_latest_and_explicit_iteration_layout_history_and_cleanup(self):
        session = _Session(self.directory)
        unrelated = _figure()
        for iteration, shape_only, selected in ((None, False, 2), (1, True, 1)):
            with self.subTest(iteration=iteration, shape_only=shape_only):
                paths = save_reports(session, iteration=iteration, shape_only=shape_only)
                parent = self.directory / "reports" / f"iteration_{selected:04d}"
                comparison = "comparison_shapes" if shape_only else "comparison_yields"
                self.assertEqual(paths, [parent / "ratios" / "reco_pull_closure.pdf",
                                         parent / comparison / "reco.pdf",
                                         parent / comparison / "truth.pdf", parent / "history.json"])
                for path in paths[:-1]:
                    self.assertEqual(path.read_bytes()[:4], b"%PDF")
                self.assertEqual(json.loads(paths[-1].read_text()), session.rows[:selected])
                self.assertEqual(set(plt.get_fignums()), {unrelated.number})
                self.assertEqual(session.calls[-2:], [("diagnostics", selected),
                                                       ("compare", selected, shape_only)])

    def test_partial_plot_generation_failure_closes_created_figures_only(self):
        unrelated = _figure()
        for stage in ("diagnostics", "compare"):
            with self.subTest(stage=stage):
                session = _Session(self.directory / stage)
                session.fail = stage
                with self.assertRaisesRegex(RuntimeError, "after creating figures"):
                    save_reports(session)
                self.assertEqual(set(plt.get_fignums()), {unrelated.number})
                self.assertFalse((session.run_dir / "reports" / "iteration_0002" / "history.json").exists())

    def test_report_save_failure_closes_all_generated_figures(self):
        session = _Session(self.directory)
        unrelated = _figure()
        with patch("omnifold.reporting.save_figures", side_effect=OSError("disk unavailable")):
            with self.assertRaisesRegex(OSError, "disk unavailable"):
                save_reports(session)
        self.assertEqual(set(plt.get_fignums()), {unrelated.number})

    def test_invalid_history_json_preserves_previous_snapshot_and_closes_figures(self):
        session = _Session(self.directory)
        session.rows[-1]["data_yield"] = float("nan")
        parent = self.directory / "reports" / "iteration_0002"
        parent.mkdir(parents=True)
        history_path = parent / "history.json"
        history_path.write_text('[{"index": 2}]\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "JSON compliant"):
            save_reports(session)
        self.assertEqual(history_path.read_text(), '[{"index": 2}]\n')
        self.assertEqual(plt.get_fignums(), [])
        self.assertEqual(sorted(path.name for path in parent.iterdir()),
                         ["comparison_yields", "history.json", "ratios"])

    def test_report_requires_a_completed_integer_iteration(self):
        session = _Session(self.directory)
        for iteration in (0, 3, -1, 1.5, True, "1"):
            with self.subTest(iteration=iteration), self.assertRaises(IndexError):
                save_reports(session, iteration=iteration)
        session.rows = []
        with self.assertRaisesRegex(RuntimeError, "session.step"):
            save_reports(session)
        self.assertEqual(session.calls, [])
        self.assertEqual(plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
