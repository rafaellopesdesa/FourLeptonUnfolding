#include "TFile.h"
#include "TError.h"
#include "TSystem.h"
#include "TTree.h"

void check_delphes_output() {
  const char *path = gSystem->Getenv("DELPHES_OUTPUT_FILE");
  if (!path || !*path) {
    Error("check_delphes_output", "DELPHES_OUTPUT_FILE is not set");
    gSystem->Exit(2);
    return;
  }

  TFile *file = TFile::Open(path, "READ");
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

  Printf("[simulation] Validated Delphes tree: %lld entries", entries);
  file->Close();
}
