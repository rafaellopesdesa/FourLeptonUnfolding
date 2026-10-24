#!/usr/bin/env python3
"""Copy a pinned MG5 standalone export, applying documented interface patches.

Run regenerate_backend.py for the complete generation recipe. This script does
not implement amplitudes; all wavefunctions and amplitudes come from MG5/UFO.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

MODEL = "SMEFTsim_U35_MwScheme_UFO_OHB"
MG_COMMIT = "4492b3e392effbb650447763fef5f67700673f03"
SMEFT_COMMIT = "db7d4a80bdcff424eee27dde71f1eb09ac894039"
CHANNELS = {
    "mixed": ("P1_", "HBMixedProcess", "h_emepmummup"),
    "eeee": ("P2_", "HBEEEEProcess", "h_emepemep"),
    "mmmm": ("P3_", "HBMMMMProcess", "h_mummupmummup"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patch_process(source: str, class_name: str) -> str:
    source = source.replace("CPPProcess", class_name)
    start = source.index(f"void {class_name}::sigmaKin()")
    end = source.index("\n//--------------------------------------------------------------------------", start)
    original = source[start:end]
    helicities = re.search(
        r"static const int helicities\[ncomb\]\[nexternal\] = .*?;", original, re.S
    ).group(0)
    denominators = re.search(
        r"const int denominators\[nprocesses\] = .*?;", original, re.S
    ).group(0)
    matrix = re.search(r"t\[0\] = (matrix_\w+\(\));", original).group(1)
    replacement = f"""void {class_name}::sigmaKin()
{{
  // Interface patch: evaluate every helicity at every coefficient/phase point.
  // No helicity learned at an earlier point may suppress a later contribution.
  pars->setDependentParameters();
  pars->setDependentCouplings();
  const int ncomb = 16;
  {helicities}
  {denominators}
  int perm[nexternal];
  for (int i = 0; i < nexternal; ++i) perm[i] = i;
  matrix_element[0] = 0.;
  jamp2[0][0] = 0.;
  for (int ihel = 0; ihel < ncomb; ++ihel) {{
    calculate_wavefunctions(perm, helicities[ihel]);
    matrix_element[0] += {matrix};
  }}
  matrix_element[0] /= denominators[0];
}}
"""
    source = source[:start] + replacement + source[end:]
    source = re.sub(r"^  pars->print\w+\(\);\s*\n", "", source, flags=re.M)
    # Keep each source's local header include; only C++ class symbols are renamed.
    source = source.replace(f'#include "{class_name}.h"', '#include "CPPProcess.h"')
    return source


def vendor(export: Path, mg: Path, smeft: Path, destination: Path) -> None:
    if "version = 3.5.17" not in (mg / "VERSION").read_text():
        raise ValueError("Expected MG5 3.5.17")
    ufo = smeft / "UFO_models/SMEFTsim_U35_MwScheme_UFO"
    if (ufo / "version.info").read_text().strip() != "tag: v3.0.2":
        raise ValueError("Expected SMEFTsim 3.0.2")
    generated = destination / "generated"
    (generated / "src").mkdir(parents=True, exist_ok=True)
    originals = {}
    for name in [f"HelAmps_{MODEL}.cc", f"HelAmps_{MODEL}.h",
                 f"Parameters_{MODEL}.cc", f"Parameters_{MODEL}.h",
                 "read_slha.cc", "read_slha.h"]:
        path = export / "src" / name
        originals[f"src/{name}"] = sha256(path)
        text = path.read_text()
        if name == "read_slha.cc":
            text = re.sub(r'^  std::cout << "Opened slha file ".*\n', "", text, flags=re.M)
        (generated / "src" / name).write_text(text)
    for channel, (prefix, class_name, _) in CHANNELS.items():
        source_dir, = (export / "SubProcesses").glob(prefix + "*")
        output = generated / "processes" / channel
        output.mkdir(parents=True, exist_ok=True)
        for suffix in ["h", "cc"]:
            path = source_dir / f"CPPProcess.{suffix}"
            originals[str(path.relative_to(export))] = sha256(path)
            text = path.read_text()
            text = (patch_process(text, class_name) if suffix == "cc"
                    else text.replace("CPPProcess", class_name))
            (output / path.name).write_text(text)
    (destination / "cards").mkdir(exist_ok=True)
    card = (export / "Cards/param_card.dat").read_text()
    card, count = re.subn(r"(?m)^(\s+8)\s+1\.000000e-01(\s+# cHB\s*)$",
                         r"\1 0.000000e+00\2", card)
    if count != 1:
        raise ValueError("Expected one generated cHB=0.1 entry")
    (destination / "cards/param_card.dat").write_text(card)
    restriction = mg / "models/SMEFTsim_U35_MwScheme_UFO/restrict_OHB.dat"
    shutil.copyfile(restriction, destination / "cards/restrict_OHB.dat")
    (destination / "licenses").mkdir(exist_ok=True)
    shutil.copyfile(mg / "LICENSE", destination / "licenses/MadGraph5.txt")
    shutil.copyfile(smeft / "LICENSE", destination / "licenses/SMEFTsim.txt")
    manifest = {
        "schema_version": 1,
        "generator": {"name": "MadGraph5_aMC@NLO", "version": "3.5.17", "commit": MG_COMMIT,
                      "archive_url": f"https://api.github.com/repos/mg5amcnlo/mg5amcnlo/tarball/{MG_COMMIT}",
                      "archive_sha256": "a3b68eb489098cf30ca2684e186802c4454a3c2611aa0cbe9bb55e580b336286"},
        "model": {"name": "SMEFTsim_U35_MwScheme_UFO", "version": "3.0.2", "commit": SMEFT_COMMIT,
                  "archive_url": f"https://api.github.com/repos/SMEFTsim/SMEFTsim/tarball/{SMEFT_COMMIT}",
                  "archive_sha256": "64b6014bc6a3749f0040eafd4bfa24e3dc9f80d48c523cde2465f85b8649ad97"},
        "operator": "cHB * (H_dagger H) B_mu_nu B^mu_nu / LambdaSMEFT^2",
        "coefficient": {"parameter": "cHB", "block": "SMEFT", "index": 8,
                        "LambdaSMEFT_GeV": 1000.0},
        "input_scheme": "MwScheme: MW, MZ, Gf",
        "approximation": "LO, massless charged leptons, fixed widths; SMHLOOP=0; at most one dimension-six insertion per amplitude",
        "restriction": "SMlimit_massless with cHB=0.1; all other WCs zero and linearPropCorrections=0. NPprop is removed by this restriction.",
        "commands": ["import model SMEFTsim_U35_MwScheme_UFO-OHB",
                     "generate h > e- e+ mu- mu+ QCD=0 SMHLOOP=0 NP<=1",
                     "add process h > e- e+ e- e+ QCD=0 SMHLOOP=0 NP<=1",
                     "add process h > mu- mu+ mu- mu+ QCD=0 SMHLOOP=0 NP<=1",
                     "output standalone_cpp EXPORT_DIRECTORY -f"],
        "patches": ["Rename CPPProcess classes per flavor channel.",
                    "Replace adaptive sigmaKin helicity cache by a full sum of all 16 helicities; unchanged generated wavefunctions/amplitudes and symmetry factors.",
                    "Suppress informational parameter/coupling and SLHA opening prints.",
                    "Set runtime card cHB=0; generation restriction keeps cHB=0.1 to retain its vertices."],
        "channels": {"mixed": {"pdg": [11,-11,13,-13], "diagrams": 5, "identical_factor": 1},
                     "eeee": {"pdg": [11,-11,11,-11], "diagrams": 10, "identical_factor": 4},
                     "mmmm": {"pdg": [13,-13,13,-13], "diagrams": 10, "identical_factor": 4}},
        "upstream_generated_sha256": originals,
        "generated_sha256": {str(p.relative_to(destination)): sha256(p)
                             for p in sorted(generated.rglob('*')) if p.suffix in {'.cc','.h'}},
        "param_card_sha256": sha256(destination / "cards/param_card.dat"),
        "restriction_sha256": sha256(destination / "cards/restrict_OHB.dat"),
    }
    (generated / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--mg5", type=Path, required=True)
    parser.add_argument("--smeftsim", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    vendor(args.export, args.mg5, args.smeftsim, args.destination)
