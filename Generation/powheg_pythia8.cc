#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC3.h"
#include "Pythia8Plugins/PowhegHooks.h"

#include <iostream>
#include <memory>
#include <string>

int main(int argc, char* argv[]) {
  if (argc != 6) {
    std::cerr
        << "Usage: powheg-pythia8 INPUT.lhe OUTPUT.hepmc3 EVENTS SEED "
           "{gg_H|ZZ}\n";
    return 2;
  }

  const std::string lhe_file = argv[1];
  const std::string output_file = argv[2];
  const long max_events = std::stol(argv[3]);
  const int seed = std::stoi(argv[4]);
  const std::string process = argv[5];
  if (process != "gg_H" && process != "ZZ") {
    std::cerr << "Process must be gg_H or ZZ\n";
    return 2;
  }

  Pythia8::Pythia pythia;
  pythia.readString("Random:setSeed = on");
  pythia.readString("Random:seed = " + std::to_string(seed));
  pythia.readString("Beams:frameType = 4");
  pythia.readString("Beams:LHEF = " + lhe_file);

  // Recommended POWHEG matching defaults from the PYTHIA 8.317 examples.
  pythia.readString("POWHEG:nFinal = " +
                    std::to_string(process == "gg_H" ? 1 : 4));
  pythia.readString("POWHEG:veto = 1");
  pythia.readString("POWHEG:vetoCount = 3");
  pythia.readString("POWHEG:pThard = 0");
  pythia.readString("POWHEG:pTemt = 0");
  pythia.readString("POWHEG:emitted = 0");
  pythia.readString("POWHEG:pTdef = 1");
  pythia.readString("POWHEG:MPIveto = 0");
  pythia.readString("POWHEG:QEDveto = 0");
  pythia.readString("TimeShower:pTmaxMatch = 2");
  pythia.readString("SpaceShower:pTmaxMatch = 2");
  pythia.readString("SpaceShower:pTdampMatch = 3");
  pythia.readString("TimeShower:pTdampMatch = 0");

  // The gg_H LHE contains an undecayed Higgs. For this baseline, force
  // H -> ZZ(*) and Z -> e/mu. The exact four-lepton decay model will be
  // validated and, if needed, upgraded in the next phase of the study.
  if (process == "gg_H") {
    pythia.readString("25:onMode = off");
    pythia.readString("25:onIfMatch = 23 23");
    pythia.readString("23:onMode = off");
    pythia.readString("23:onIfAny = 11 13");
  }

  auto powheg_hooks = std::make_shared<Pythia8::PowhegHooks>();
  pythia.setUserHooksPtr((Pythia8::UserHooksPtr)powheg_hooks);
  if (!pythia.init()) return 3;

  HepMC3::Pythia8ToHepMC3 converter;
  HepMC3::WriterAscii writer(output_file);
  long accepted = 0;
  int consecutive_errors = 0;
  while (max_events == 0 || accepted < max_events) {
    if (!pythia.next()) {
      if (pythia.info.atEndOfFile()) break;
      if (++consecutive_errors >= 20) {
        std::cerr << "Aborting after 20 consecutive PYTHIA errors\n";
        return 4;
      }
      continue;
    }
    consecutive_errors = 0;
    HepMC3::GenEvent event;
    converter.fill_next_event(pythia, &event);
    writer.write_event(event);
    ++accepted;
  }
  writer.close();
  pythia.stat();
  std::cout << "Wrote " << accepted << " events to " << output_file << '\n';
  return accepted > 0 ? 0 : 5;
}

