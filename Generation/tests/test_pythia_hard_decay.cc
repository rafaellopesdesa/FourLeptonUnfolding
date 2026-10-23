/* Optional native-record test using the installed PYTHIA/HepMC libraries.
From the repository root after sourcing Generation/env.sh:

c++ -std=c++17 $("$GENERATION_PREFIX/bin/pythia8-config" --cxxflags) \
  -UNDEBUG Generation/tests/test_pythia_hard_decay.cc \
  -o /tmp/test_pythia_hard_decay \
  $("$GENERATION_PREFIX/bin/pythia8-config" --ldflags) \
  $("$GENERATION_PREFIX/bin/pythia8-config" --libs) \
  -L"$GENERATION_PREFIX/lib" -L"$GENERATION_PREFIX/lib64" -lHepMC3 \
  -Wl,-rpath,"$GENERATION_PREFIX/lib" -Wl,-rpath,"$GENERATION_PREFIX/lib64"
/tmp/test_pythia_hard_decay
*/
// Exercise the production extractor, rather than a duplicate implementation.
#define main unused_production_driver_main
#include "../powheg_pythia8.cc"
#undef main

#include <cassert>

Pythia8::Event decay_fixture(int second_flavour = 13) {
  Pythia8::Event event;
  event.reset();
  // A production-recoil H copy: the pre-recoil state is intentionally different.
  event.append(25, -22, 0, 0, 2, 2, 0, 0, 10., 0., 0., 126., 125.);
  event.append(25, -22, 1, 1, 3, 4, 0, 0, 0., 0., 0., 125., 125.);
  event.append(23, -22, 2, 0, 5, 6, 0, 0, 0., 0., 30., 75.);
  event.append(23, -22, 2, 0, 7, 8, 0, 0, 0., 0., -30., 50.);
  const double x = std::sqrt(37.5 * 37.5 - 15. * 15.);
  event.append(11, 23, 3, 0, 0, 0, 0, 0, x, 0., 15., 37.5);
  event.append(-11, 23, 3, 0, 0, 0, 0, 0, -x, 0., 15., 37.5);
  event.append(second_flavour, 23, 4, 0, 0, 0, 0, 0, 0., 20., -15., 25.);
  event.append(-second_flavour, 23, 4, 0, 0, 0, 0, 0, 0., -20., -15., 25.);
  return event;
}

int main() {
  auto event = decay_fixture();
  const auto before = capture_hard_decay(event);
  assert(before.status == "ok");
  assert(before.parent.size() == 1 && before.parent[0][0] == 0.);
  assert(before.bosons.size() == 2 && before.particles.size() == 4);
  assert(before.json().find("\"momentum_unit\":\"GEV\"") != std::string::npos);
  assert(before.json().find("pythia_event_before_decay_fsr") != std::string::npos);

  // Native FSR history: preserve the original Z daughters and append the
  // radiating lepton, photon and recoil copy with deliberately different p4.
  event[5].statusNeg();
  event[5].daughters(9, 10);
  event.append(11, 51, 5, 0, 0, 0, 0, 0, 10., 0., 10., 15.);
  event.append(22, 51, 5, 0, 0, 0, 0, 0, 5., 0., 0., 5.);
  event[6].statusNeg();
  event[6].daughters(11, 11);
  event.append(-11, 52, 6, 6, 0, 0, 0, 0, -15., 0., 20., 25.);
  const auto after = capture_hard_decay(event);
  assert(after.json() == before.json());

  // An intermediate Z copy is followed before selecting its direct daughters.
  event = decay_fixture();
  const int z_copy = event.append(event[3]);
  event[z_copy].mothers(3, 3);
  event[3].daughters(z_copy, z_copy);
  event[5].mother1(z_copy);
  event[6].mother1(z_copy);
  assert(capture_hard_decay(event).status == "ok");

  // A Z continuation accompanied by a weak boson is not a clean copy,
  // even though the later Z decay by itself still closes in four-momentum.
  const int weak_sibling = event.append(24, 23, 3, 0, 0, 0, 0, 0,
                                       0., 0., 0., 80., 80.);
  event[3].daughters(z_copy, weak_sibling);
  assert(capture_hard_decay(event).status == "ambiguous_z_copies");
  assert(capture_hard_decay(event).particles.empty());

  // Likewise, do not follow H -> H + photon as if it were an ISR recoil copy.
  event = decay_fixture();
  const int photon = event.append(22, 23, 1, 0, 0, 0, 0, 0,
                                  1., 0., 0., 1.);
  // PYTHIA's reversed daughter endpoints represent two separated daughters.
  event[1].daughters(photon, 2);
  assert(capture_hard_decay(event).status == "ambiguous_higgs_copies");
  assert(capture_hard_decay(event).particles.empty());

  // A tau channel retains taus, even if they later decay into e/mu.
  event = decay_fixture(15);
  event[7].statusNeg();
  event[7].daughters(9, 9);
  event.append(13, 91, 7, 0, 0, 0, 0, 0, 0., 5., -4., 7.);
  const auto tau = capture_hard_decay(event);
  assert(tau.status == "ok" && tau.particles[2].pdg == 15);
  assert(tau.particles[2].momentum[3] == 25.);

  event = decay_fixture();
  event[5].e(36.);
  assert(capture_hard_decay(event).status == "momentum_nonconservation");
  assert(capture_hard_decay(event).particles.empty());
  event[5].e(std::numeric_limits<double>::quiet_NaN());
  assert(capture_hard_decay(event).status == "invalid_momentum");

  event = decay_fixture();
  event[6].id(11);
  assert(capture_hard_decay(event).status == "invalid_z_decay");
  event = decay_fixture();
  event.append(25, 22, 0, 0, 0, 0, 0, 0, 0., 0., 0., 125., 125.);
  assert(capture_hard_decay(event).status == "ambiguous_higgs");
  event = decay_fixture();
  event[2].daughters(2, 2);
  assert(capture_hard_decay(event).status == "ambiguous_higgs_copies");
  event = decay_fixture();
  event[5].mother2(4);
  assert(capture_hard_decay(event).status == "ambiguous_z_copies");
  event.reset();
  assert(capture_hard_decay(event).status == "missing_higgs");

  std::cout << "Native hard decay: production/Z copies, pre-FSR daughters, "
               "tau preservation, invalid topology and momentum checks passed\n";
}
