#!/usr/bin/env python3
"""Regenerate the vendored LO cHB backend from pinned MG5 and SMEFTsim sources.

Only generation needs MG5; evaluating the committed sources only needs g++.
Supply unpacked sources of the exact commits recorded in provenance.json.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from vendor_generated import vendor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mg5", type=Path, required=True)
    parser.add_argument("--smeftsim", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    mg = args.mg5.resolve()
    smeft = args.smeftsim.resolve()
    destination = args.destination.resolve()
    if "version = 3.5.17" not in (mg / "VERSION").read_text():
        raise ValueError("Expected MG5 3.5.17")
    name = "SMEFTsim_U35_MwScheme_UFO"
    original = smeft / "UFO_models" / name
    if (original / "version.info").read_text().strip() != "tag: v3.0.2":
        raise ValueError("Expected SMEFTsim 3.0.2")
    model = mg / "models" / name
    if model.exists():
        # Imported UFO caches can depend on restriction contents. A fresh copy
        # ensures regeneration cannot accidentally use an earlier cached model.
        shutil.rmtree(model)
    shutil.copytree(original, model)
    restriction, count = re.subn(
        r"(?m)^    8 0\.000000e\+00 # cHB\s*$", "    8 1.000000e-01 # cHB",
        (model / "restrict_SMlimit_massless.dat").read_text(),
    )
    if count != 1:
        raise ValueError("Expected exactly one cHB entry in the upstream restriction")
    (model / "restrict_OHB.dat").write_text(restriction)
    with tempfile.TemporaryDirectory(prefix="smeft-hb-generation-") as temporary:
        temporary = Path(temporary)
        export = temporary / "standalone"
        commands = temporary / "generate.mg5"
        commands.write_text(
            "set automatic_html_opening False\nset auto_update 0\n"
            f"import model {name}-OHB\n"
            "generate h > e- e+ mu- mu+ QCD=0 SMHLOOP=0 NP<=1\n"
            "add process h > e- e+ e- e+ QCD=0 SMHLOOP=0 NP<=1\n"
            "add process h > mu- mu+ mu- mu+ QCD=0 SMHLOOP=0 NP<=1\n"
            f"output standalone_cpp {export} -f\nquit\n"
        )
        subprocess.run([sys.executable, str(mg / "bin/mg5_aMC"), str(commands)],
                       cwd=mg, check=True)
        if not (export / "Cards/param_card.dat").is_file():
            raise RuntimeError("MG5 did not produce the requested standalone export")
        vendor(export, mg, smeft, destination)


if __name__ == "__main__":
    main()
