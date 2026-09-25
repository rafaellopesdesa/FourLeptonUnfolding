from __future__ import annotations

from pathlib import Path
import re
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]


def card_value(path: Path, key: str) -> float:
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("!", 1)[0].split()
        if len(fields) >= 2 and fields[0] == key:
            return float(fields[1].replace("D", "E").replace("d", "e"))
    raise KeyError(key)


class BeamEnergyTest(unittest.TestCase):
    def test_all_builtin_powheg_cards_use_13p6_tev(self):
        for process in ("gg_H", "ZZ"):
            card = REPO_ROOT / "PowhegCards" / f"{process}.powheg.input"
            with self.subTest(process=process):
                self.assertEqual(card_value(card, "ebeam1"), 6800.0)
                self.assertEqual(card_value(card, "ebeam2"), 6800.0)

    def test_both_showers_use_the_validated_lhe_energy(self):
        script = (REPO_ROOT / "Generation" / "run_generation.sh").read_text(
            encoding="utf-8"
        )
        validation = 'validate_lhe_energy "$INPUT_LHE"'
        self.assertIn(validation, script)
        self.assertLess(script.index(validation), script.index('case "$SHOWER" in'))
        self.assertIn("Beams:frameType = 4", (
            REPO_ROOT / "Generation" / "powheg_pythia8.cc"
        ).read_text(encoding="utf-8"))
        self.assertRegex(script, re.compile(r"LesHouchesReader:FileName \$INPUT_LHE"))
        self.assertIn("sqrt_s_gev=$SQRT_S_GEV", script)

    def test_shell_entrypoints_have_valid_syntax(self):
        scripts = (
            REPO_ROOT / "Generation" / "run_generation.sh",
            REPO_ROOT / "Generation" / "submit_generation.sh",
            REPO_ROOT / "Simulation" / "run_simulation.sh",
            REPO_ROOT / "BatchSubmit" / "unity_generation_job.sh",
        )
        for script in scripts:
            with self.subTest(script=script.name):
                subprocess.run(["bash", "-n", script], check=True)


if __name__ == "__main__":
    unittest.main()
