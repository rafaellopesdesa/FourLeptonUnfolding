import uproot
import numpy as np
import matplotlib.pyplot as plt

files = {
    "ggH Pythia": "production/merged/gg_H_pythia.root",
    "qqZZ Pythia": "production/merged/ZZ_pythia.root",
}

categories = [
    "Fiducial + Reco",
    "Fiducial + Not Reco",
    "Not Fiducial + Reco",
    "Not Fiducial + Not Reco",
]

all_fractions = {}

for label, filename in files.items():
    with uproot.open(filename) as f:
        tree = f["Analysis"]

        fid = tree["fiducial"].array(library="np").astype(bool)
        reco = tree["reconstructed"].array(library="np").astype(bool)

        masks = [
            fid & reco,
            fid & ~reco,
            ~fid & reco,
            ~fid & ~reco,
        ]

        counts = np.array([np.count_nonzero(m) for m in masks], dtype=float)
        fractions = 100 * counts / counts.sum()

        all_fractions[label] = fractions

        print(f"\n{label}")
        print("-" * 50)
        for name, count, frac in zip(categories, counts, fractions):
            print(f"{name:30s} {int(count):10d}   {frac:6.2f}%")

x = np.arange(len(categories))
width = 0.36

fig, ax = plt.subplots(figsize=(11, 6))

for i, (label, fractions) in enumerate(all_fractions.items()):
    offset = (i - 0.5) * width
    bars = ax.bar(x + offset, fractions, width, label=label)

    for bar, value in zip(bars, fractions):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

ax.set_ylabel("Fraction of generated events [%]")
ax.set_title("Fiducial / Reconstruction Event Categories")
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=15, ha="right")
ax.legend()

fig.tight_layout()
fig.savefig("detector_categories.png", dpi=200)

print("\nSaved detector_categories.png")
