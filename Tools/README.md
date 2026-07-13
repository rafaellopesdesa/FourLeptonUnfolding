# Four-lepton Python tools

These importable tools implement the conventions and formulas in
[arXiv:1208.4018](https://arxiv.org/abs/1208.4018).

Install the small Python dependency set from the repository root:

```bash
python3 -m pip install -r Tools/requirements.txt
```

The four inputs always have the charge-ordered form `(e-, e+, mu-, mu+)` and
should normally be scalar [Scikit-HEP vector](https://vector.readthedocs.io/)
objects:

```python
import vector

from Tools.four_lepton_kinematics import four_lepton_observables
from Tools.higgs_decay_width import differential_decay_width

electron = vector.obj(E=30.0, px=10.0, py=5.0, pz=27.8388218)
# Define positron, muon and antimuon in the same way.

observables = four_lepton_observables(electron, positron, muon, antimuon)
density = differential_decay_width(electron, positron, muon, antimuon)
```

For a selected `4e`, `4mu`, or mixed-flavor candidate, use
`compute_paired_kinematics(z1_minus, z1_plus, z2_minus, z2_plus,
z1_flavor=...)`. This preserves the analysis pairing order instead of applying
the original mass-ordering convention.

`Z1` is always the heavier dilepton pair. The default incoming-parton
convention is the laboratory `+z` beam; this resolves the unavoidable beam-sign
ambiguity at a proton-proton collider reproducibly.

The scalar-Higgs result is a differential decay-density kernel. Equation (5)
of the paper defines the mass-and-angle distribution only up to an overall
constant, so the result is appropriate for relative weights and shapes, not as
an absolute partial width in GeV. The defaults are `mH = 125.10 GeV`,
`mZ = 91.1876 GeV`, `GammaZ = 2.4952 GeV`, `v = 246.22 GeV`, and the tree-level
SM `0+` coupling choice `g1^(0)=1` (equivalently `a1=mZ^2/mH^2`, with
`a2=a3=0`).
