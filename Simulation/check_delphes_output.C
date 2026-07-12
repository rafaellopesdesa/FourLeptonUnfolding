#include "TFile.h"
#include "TBranch.h"
#include "TError.h"
#include "TLeaf.h"
#include "TObject.h"
#include "TSystem.h"
#include "TTree.h"

#include <cerrno>
#include <cstdlib>

namespace {
Long64_t expected_entries() {
  const char *text = gSystem->Getenv("DELPHES_EXPECTED_EVENTS");
  if (!text || !*text) return -1;

  errno = 0;
  char *end = nullptr;
  const Long64_t value = std::strtoll(text, &end, 10);
  if (errno || end == text || *end || value < 0) {
    Error("check_delphes_output", "invalid DELPHES_EXPECTED_EVENTS: %s", text);
    return -2;
  }
  return value;
}

bool require_branch(TTree *tree, const char *name) {
  if (tree->GetBranch(name)) return true;
  Error("check_delphes_output", "required branch %s is missing", name);
  return false;
}
}  // namespace

void check_delphes_output() {
  const char *path = gSystem->Getenv("DELPHES_OUTPUT_FILE");
  if (!path || !*path) {
    Error("check_delphes_output", "DELPHES_OUTPUT_FILE is not set");
    gSystem->Exit(2);
    return;
  }

  TFile *file = TFile::Open(path, "UPDATE");
  if (!file || file->IsZombie()) {
    Error("check_delphes_output", "cannot open %s", path);
    gSystem->Exit(3);
    return;
  }

  TTree *tree = dynamic_cast<TTree *>(file->Get("Delphes"));
  if (!tree) {
    Error("check_delphes_output", "Delphes tree is missing in %s", path);
    file->Close();
    gSystem->Exit(4);
    return;
  }

  const Long64_t entries = tree->GetEntries();
  if (entries <= 0) {
    Error("check_delphes_output", "Delphes tree has no entries in %s", path);
    file->Close();
    gSystem->Exit(5);
    return;
  }

  const Long64_t expected = expected_entries();
  if (expected == -2) {
    file->Close();
    gSystem->Exit(6);
    return;
  }
  if (expected >= 0 && entries != expected) {
    Error("check_delphes_output", "event-count mismatch: expected %lld, found %lld",
          expected, entries);
    file->Close();
    gSystem->Exit(7);
    return;
  }

  const char *required[] = {"Event",         "Weight",      "Particle",
                            "StableParticle", "RecoElectron", "RecoMuon",
                            "DressedElectron", "DressedMuon",  "Electron",
                            "Muon"};
  for (const char *name : required) {
    if (!require_branch(tree, name)) {
      file->Close();
      gSystem->Exit(8);
      return;
    }
  }

  TBranch *marker = tree->GetBranch("HasFourRecoLeptons");
  if (!marker) {
    TLeaf *electron_size = tree->GetLeaf("RecoElectron_size");
    TLeaf *muon_size = tree->GetLeaf("RecoMuon_size");
    if (!electron_size || !muon_size) {
      Error("check_delphes_output", "loose reconstructed-lepton counts are missing");
      file->Close();
      gSystem->Exit(9);
      return;
    }

    Bool_t has_four_reco_leptons = false;
    marker = tree->Branch("HasFourRecoLeptons", &has_four_reco_leptons,
                          "HasFourRecoLeptons/O");
    for (Long64_t entry = 0; entry < entries; ++entry) {
      tree->GetEntry(entry);
      has_four_reco_leptons =
          electron_size->GetValueLong64() + muon_size->GetValueLong64() >= 4;
      marker->Fill();
    }
    tree->Write("", TObject::kOverwrite);
  } else if (marker->GetEntries() != entries) {
    Error("check_delphes_output",
          "HasFourRecoLeptons has %lld entries but the tree has %lld",
          marker->GetEntries(), entries);
    file->Close();
    gSystem->Exit(10);
    return;
  }

  Printf("[simulation] Validated full event retention: %lld entries", entries);
  file->Close();
}
