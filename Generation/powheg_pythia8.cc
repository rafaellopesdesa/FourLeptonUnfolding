#include "Pythia8/Pythia.h"
#include "Pythia8/LesHouches.h"
#include "Pythia8Plugins/HepMC3.h"
#include "Pythia8Plugins/PowhegHooks.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Count source reads, not calls to next() or accepted shower events. PYTHIA
// can reject several hard events inside one next() call; each such read still
// advances this ordinal. LHAupLHEF::setInit only reads run initialization.
class IndexedLHEReader final : public Pythia8::LHAupLHEF {
 public:
  IndexedLHEReader(Pythia8::Info* initial_info, std::istream* events,
                   std::istream* header, bool read_headers, bool set_scales)
      : Pythia8::LHAupLHEF(initial_info, events, header, read_headers,
                          set_scales) {}

  bool setEvent(int = 0) override {
    if (!read_next_source()) return false;
    return setOldEventLHEF();
  }

  // Keep the same source convention if Beams:nSkipLHEFatInit is enabled.
  bool skipEvent(int count) override {
    for (int index = 0; index < count; ++index) {
      if (!read_next_source()) return false;
    }
    return true;
  }

  int source_index() const { return source_index_; }

 private:
  bool read_next_source() {
    if (!setNewEventLHEF()) return false;
    // HepMC's event number is an int. Reject an unrepresentable identity
    // explicitly instead of overflowing and joining the wrong LHE event.
    if (source_index_ == std::numeric_limits<int>::max()) {
      throw std::overflow_error("LHE source index exceeds HepMC event-number range");
    }
    ++source_index_;
    return true;
  }

  int source_index_ = 0;
};

using FourMomentum = std::array<double, 4>;

struct HardDecayParticle {
  int pdg;
  FourMomentum momentum;
};

// These are the original resonance-decay daughters in PYTHIA's full event,
// after production recoil and before decay QED FSR. In PYTHIA 8.317,
// PartonLevel::resonanceShowers copies process daughters, boosts them to the
// showered parent's momentum, and only then starts resonance showers.
// SimpleTimeShower appends daughters/recoiler copies without replacing those
// original four-momenta. Follow H/Z copies, but NEVER follow lepton copies:
// doing so would select radiated leptons or e/mu from a tau decay instead.
// This is deliberately not a claim to reconstruct a pre-ISR hard decay.
struct HardDecaySnapshot {
  std::string status = "missing_higgs";
  std::vector<FourMomentum> parent;
  std::vector<FourMomentum> bosons;
  std::vector<HardDecayParticle> particles;

  std::string json() const {
    std::ostringstream out;
    out << std::setprecision(17)
        << "{\"schema_version\":1,\"source\":"
           "\"pythia_event_before_decay_fsr\",\"momentum_unit\":\"GEV\","
           "\"status\":\"" << status << "\",\"parent\":[";
    const auto write_p4 = [&out](const FourMomentum& p) {
      out << p[0] << ',' << p[1] << ',' << p[2] << ',' << p[3];
    };
    if (!parent.empty()) write_p4(parent.front());
    out << "],\"bosons\":[";
    for (std::size_t i = 0; i < bosons.size(); ++i) {
      if (i) out << ',';
      out << '[';
      write_p4(bosons[i]);
      out << ']';
    }
    out << "],\"particles\":[";
    for (std::size_t i = 0; i < particles.size(); ++i) {
      if (i) out << ',';
      out << '[' << particles[i].pdg << ',';
      write_p4(particles[i].momentum);
      out << ']';
    }
    out << "]}";
    return out.str();
  }
};

FourMomentum four_momentum(const Pythia8::Particle& particle) {
  return {{particle.px(), particle.py(), particle.pz(), particle.e()}};
}

bool valid_momentum(const FourMomentum& p) {
  return p[3] > 0.0 && std::all_of(p.begin(), p.end(),
                                 [](double x) { return std::isfinite(x); });
}

bool closes_momentum(const FourMomentum& parent, const FourMomentum& first,
                     const FourMomentum& second) {
  double scale = 1.0;
  for (std::size_t i = 0; i < parent.size(); ++i)
    scale = std::max(scale, std::max(std::abs(parent[i]),
                                   std::abs(first[i]) + std::abs(second[i])));
  for (std::size_t i = 0; i < parent.size(); ++i)
    if (std::abs(parent[i] - first[i] - second[i]) > 1.e-7 + 1.e-6 * scale)
      return false;
  return true;
}

bool direct_daughters(const Pythia8::Event& event, int index,
                      std::vector<int>& daughters) {
  daughters = event[index].daughterList();
  for (int daughter : daughters) {
    if (daughter <= 0 || daughter >= event.size()) return false;
    const auto& particle = event[daughter];
    // A resonance's original daughters have one parent. A shared or broken
    // history is not enough evidence for a Born decay, even if PDGs match.
    if (particle.mother1() != index ||
        (particle.mother2() != 0 && particle.mother2() != index)) return false;
  }
  return true;
}

int last_resonance_copy(const Pythia8::Event& event, int index,
                        std::vector<int>& visited) {
  while (true) {
    if (std::find(visited.begin(), visited.end(), index) != visited.end())
      return -1;
    visited.push_back(index);
    std::vector<int> daughters;
    if (!direct_daughters(event, index, daughters)) return -1;
    int next = -1;
    for (int daughter : daughters) {
      if (event[daughter].id() != event[index].id()) continue;
      if (next != -1) return -1;  // Ambiguous same-PDG branching.
      next = daughter;
    }
    if (next == -1) return index;
    // A clean recoil copy has exactly one daughter. A same-PDG continuation
    // accompanied by radiation is a different topology, not a pure copy.
    if (daughters.size() != 1) return -1;
    index = next;
  }
}

HardDecaySnapshot capture_hard_decay(const Pythia8::Event& event) {
  HardDecaySnapshot result;
  std::vector<int> higgs_indices, roots;
  for (int index = 1; index < event.size(); ++index) {
    if (event[index].id() != 25) continue;
    higgs_indices.push_back(index);
    bool same_parent = false;
    for (int mother : {event[index].mother1(), event[index].mother2()}) {
      if (mother < 0 || mother >= event.size()) {
        result.status = "invalid_higgs_history";
        return result;
      }
      if (mother > 0 && event[mother].id() == 25) same_parent = true;
    }
    if (!same_parent) roots.push_back(index);
  }
  if (higgs_indices.empty()) return result;
  if (roots.size() != 1) {
    result.status = "ambiguous_higgs";
    return result;
  }
  std::vector<int> higgs_copies;
  const int higgs = last_resonance_copy(event, roots.front(), higgs_copies);
  if (higgs < 0 || higgs_copies.size() != higgs_indices.size()) {
    result.status = "ambiguous_higgs_copies";
    return result;
  }
  std::vector<int> z_indices;
  if (!direct_daughters(event, higgs, z_indices) || z_indices.size() != 2 ||
      event[z_indices[0]].id() != 23 || event[z_indices[1]].id() != 23) {
    result.status = "invalid_higgs_decay";
    return result;
  }
  std::vector<FourMomentum> bosons;
  std::vector<HardDecayParticle> particles;
  std::vector<int> z_copies;
  for (int z_initial : z_indices) {
    const int z = last_resonance_copy(event, z_initial, z_copies);
    if (z < 0) {
      result.status = "ambiguous_z_copies";
      return result;
    }
    std::vector<int> daughters;
    if (!direct_daughters(event, z, daughters) || daughters.size() != 2) {
      result.status = "invalid_z_decay";
      return result;
    }
    const int id = event[daughters[0]].id();
    if ((std::abs(id) != 11 && std::abs(id) != 13 && std::abs(id) != 15) ||
        id != -event[daughters[1]].id()) {
      result.status = "invalid_z_decay";
      return result;
    }
    bosons.push_back(four_momentum(event[z]));
    for (int daughter : daughters)
      particles.push_back({event[daughter].id(), four_momentum(event[daughter])});
  }
  const auto parent = four_momentum(event[higgs]);
  if (!valid_momentum(parent) ||
      !std::all_of(bosons.begin(), bosons.end(), valid_momentum) ||
      !std::all_of(particles.begin(), particles.end(),
                   [](const HardDecayParticle& p) { return valid_momentum(p.momentum); })) {
    result.status = "invalid_momentum";
    return result;
  }
  if (!closes_momentum(parent, bosons[0], bosons[1]) ||
      !closes_momentum(bosons[0], particles[0].momentum, particles[1].momentum) ||
      !closes_momentum(bosons[1], particles[2].momentum, particles[3].momentum)) {
    result.status = "momentum_nonconservation";
    return result;
  }
  result.status = "ok";
  result.parent.push_back(parent);
  result.bosons = std::move(bosons);
  result.particles = std::move(particles);
  return result;
}

}  // namespace

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
  // External streams keep the indexed reader in frameType 4; PYTHIA would
  // replace a filename-backed custom reader during initialization. Separate
  // streams are needed because Reader parses the event stream's run header
  // in its constructor, before setInit reads the header stream.
  std::ifstream lhe_events(lhe_file);
  std::ifstream lhe_header(lhe_file);
  if (!lhe_events || !lhe_header) {
    std::cerr << "Cannot open LHE input " << lhe_file << '\n';
    return 3;
  }
  // The reader constructor requires a writable Info object; Pythia::init()
  // supplies its live Info pointer before setInit/setEvent are invoked. This
  // temporary only supports construction, with the normal PYTHIA logger.
  Pythia8::Info initial_reader_info;
  initial_reader_info.loggerPtr = pythia.info.loggerPtr;
  const auto lhe_reader = std::make_shared<IndexedLHEReader>(
      &initial_reader_info, &lhe_events, &lhe_header,
      pythia.settings.flag("Beams:readLHEFheaders"),
      pythia.settings.flag("Beams:setProductionScalesFromLHEF") ||
          pythia.settings.flag("Beams:setDipoleShowerStartingScalesFromLHEF"));
  pythia.setLHAupPtr(lhe_reader);

  // Recommended POWHEG matching defaults from the PYTHIA 8.317 examples.
  // nFinal is the number of outgoing particles in PYTHIA's *hard shower
  // system*, not the number of stable daughters stored in the LHE record.
  // POWHEG ZZ writes Z Z -> 4l with two status-2 Z resonances. PYTHIA first
  // presents those two resonances (plus an optional POWHEG parton) to this
  // hook and only attaches the four LHE decay leptons afterwards. Therefore
  // the correct Born multiplicities are one Higgs for gg_H and two Z bosons
  // for ZZ.
  const int powheg_n_final = process == "gg_H" ? 1 : 2;
  pythia.readString("POWHEG:nFinal = " +
                    std::to_string(powheg_n_final));
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
  // H -> ZZ(*) and Z -> e/mu/tau. The exact four-lepton decay model will be
  // validated and, if needed, upgraded in the next phase of the study.
  if (process == "gg_H") {
    pythia.readString("25:onMode = off");
    pythia.readString("25:onIfMatch = 23 23");
    pythia.readString("23:onMode = off");
    pythia.readString("23:onIfAny = 11 13 15");
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
    if (lhe_reader->source_index() <= 0) {
      std::cerr << "Accepted event has no LHE source identity\n";
      return 6;
    }
    // This exact source identity survives both HepMC and Delphes Event.Number.
    // It is deliberately not the ordinal in the accepted shower sample.
    event.set_event_number(lhe_reader->source_index());
    if (process == "gg_H") {
      const auto hard_decay = capture_hard_decay(pythia.event);
      event.add_attribute("FourLeptonHardDecay",
          std::make_shared<HepMC3::StringAttribute>(hard_decay.json()));
    }
    writer.write_event(event);
    ++accepted;
  }
  writer.close();
  pythia.stat();
  std::cout << "Wrote " << accepted << " events to " << output_file
            << "; last source LHE index " << lhe_reader->source_index() << '\n';
  return accepted > 0 ? 0 : 5;
}
