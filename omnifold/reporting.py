"""Shared, headless-compatible PDF reports for notebooks and the CLI.

This module respects the caller's Matplotlib backend. The command-line entry
point selects Agg before generating figures; notebook displays stay live.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import json
import operator
import os
from pathlib import Path
import re
import tempfile
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from .workflow import OmniFoldSession


@contextmanager
def _atomic_path(destination: Path) -> Iterator[Path]:
    """Keep the previous output intact until a complete replacement exists."""
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.stem}.",
                                        suffix=destination.suffix, dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_figures(figures: Mapping[str, Figure], directory: str | Path) -> list[Path]:
    """Save a vector PDF per name, atomically, without closing live figures.

    Unsafe filename characters become underscores. Names which collide
    after sanitization are rejected before writing anything. A failed save
    leaves an existing PDF intact and removes its temporary file. Earlier
    successful files remain available; this is a per-file atomic operation.
    """
    directory = Path(directory).expanduser()
    destinations = []
    seen = set()
    for name, figure in figures.items():
        stem = re.sub(r"[^A-Za-z0-9_-]", "_", str(name)).strip("_") or "figure"
        if stem in seen:
            raise ValueError(f"figure names collide after filename sanitization: {name!r}")
        seen.add(stem)
        destinations.append((directory / f"{stem}.pdf", figure))
    if not destinations:
        return []
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for destination, figure in destinations:
        with _atomic_path(destination) as temporary:
            figure.savefig(temporary, format="pdf", bbox_inches="tight")
        paths.append(destination)
    return paths


def save_reports(
    session: OmniFoldSession, iteration: int | None = None, shape_only: bool = False,
) -> list[Path]:
    """Save ratio, comparison and history reports for one completed iteration.

    PDFs go below ``session.run_dir/reports/iteration_NNNN/ratios`` and
    ``comparison_shapes`` or ``comparison_yields``. The adjacent history.json
    contains the session history through the selected iteration. The return
    value lists all saved PDFs followed by that JSON path.

    All figures created while generating reports are closed, including
    figures created by a plotting call which raises before returning. Other
    already-open notebook figures are preserved. For inline notebook display,
    call the session plot methods and save_figures directly instead.
    """
    history = session.history()
    if not history:
        raise RuntimeError("Run session.step() before saving reports")
    if iteration is None:
        index = int(history[-1]["index"])
    else:
        try:
            if isinstance(iteration, bool):
                raise TypeError
            index = operator.index(iteration)
        except TypeError as error:
            raise IndexError("iteration is a 1-based completed iteration number") from error
    if index not in {int(row["index"]) for row in history}:
        raise IndexError("iteration is a 1-based completed iteration number")
    report_dir = Path(session.run_dir) / "reports" / f"iteration_{index:04d}"
    existing_numbers = set(plt.get_fignums())
    owned_figures = []
    try:
        ratios = session.diagnostics(iteration=index)
        owned_figures.extend(ratios.values())
        paths = save_figures(ratios, report_dir / "ratios")
        comparisons = session.compare(iteration=index, shape_only=shape_only)
        owned_figures.extend(comparisons.values())
        paths.extend(save_figures(comparisons, report_dir / (
            "comparison_shapes" if shape_only else "comparison_yields")))
        report_dir.mkdir(parents=True, exist_ok=True)
        history_path = report_dir / "history.json"
        with _atomic_path(history_path) as temporary:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump([row for row in history if row["index"] <= index],
                          stream, indent=2, allow_nan=False)
                stream.write("\n")
        return [*paths, history_path]
    finally:
        for figure in owned_figures:
            plt.close(figure)
        # Plot generation can fail before it returns the partially built
        # mapping. Such figures are still registered with pyplot.
        for number in set(plt.get_fignums()) - existing_numbers:
            plt.close(number)
