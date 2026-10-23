/* Optional integration test using the installed PYTHIA/HepMC libraries.
From the repository root after sourcing Generation/env.sh:

c++ -std=c++17 $("$GENERATION_PREFIX/bin/pythia8-config" --cxxflags) \
  -UNDEBUG Generation/tests/test_pythia_source_identity.cc \
  -o /tmp/test_pythia_source_identity \
  $("$GENERATION_PREFIX/bin/pythia8-config" --ldflags) \
  $("$GENERATION_PREFIX/bin/pythia8-config" --libs) \
  -L"$GENERATION_PREFIX/lib" -L"$GENERATION_PREFIX/lib64" -lHepMC3 \
  -Wl,-rpath,"$GENERATION_PREFIX/lib" -Wl,-rpath,"$GENERATION_PREFIX/lib64"
/tmp/test_pythia_source_identity
*/
// Include the actual driver's reader; do not duplicate its implementation.
#define main unused_production_driver_main
#include "../powheg_pythia8.cc"
#undef main

#include <cassert>
#include <sstream>

int main() {
  std::ostringstream fixture;
  fixture << "<LesHouchesEvents version=\"1.0\">\n"
          << "<init>\n2212 2212 6800 6800 0 0 0 0 -4 1\n"
          << "1 0 1 100\n</init>\n";
  for (int index = 1; index <= 4; ++index) {
    const int weight = index == 2 ? -2 : index;
    fixture << "<event>\n3 100 " << weight << " 125 .007 .118\n"
            << "21 -1 0 0 501 502 0 0 62.5 62.5 0 0 9\n"
            << "21 -1 0 0 502 501 0 0 -62.5 62.5 0 0 9\n"
            << "25 1 1 2 0 0 0 0 0 125 125 0 9\n</event>\n";
  }
  fixture << "</LesHouchesEvents>\n";

  std::istringstream events(fixture.str()), header(fixture.str());
  Pythia8::Info info;
  Pythia8::Logger logger;
  Pythia8::WeightContainer weights;
  info.loggerPtr = &logger;
  info.weightContainerPtr = &weights;
  weights.initPtrs(&info);
  IndexedLHEReader reader(&info, &events, &header, true, false);
  Pythia8::LHAup& through_pythia = reader;

  assert(reader.useExternal());  // BeamSetup must keep this reader in frame 4.
  assert(reader.source_index() == 0);
  assert(through_pythia.setInit());
  assert(reader.source_index() == 0);  // Header reading cannot consume event 1.
  assert(reader.idBeamA() == 2212 && reader.eBeamA() == 6800.0);
  assert(reader.idBeamB() == 2212 && reader.eBeamB() == 6800.0);
  assert(reader.strategy() == -4 && reader.xSecSum() == 1.0);

  // Two virtual reads before accepting an event model an internal hard-event
  // rejection followed by a retry. The accepted event must retain source 2.
  assert(through_pythia.setEvent());
  assert(reader.source_index() == 1 && reader.weight() == 1.0);
  assert(through_pythia.setEvent());
  assert(reader.source_index() == 2 && reader.weight() == -2.0);
  assert(through_pythia.skipEvent(1));
  assert(reader.source_index() == 3);
  assert(through_pythia.setEvent());
  assert(reader.source_index() == 4 && reader.weight() == 4.0);
  assert(!through_pythia.setEvent());
  assert(reader.source_index() == 4);

  std::cout << "LHE identity: init, rejected read/retry, skip, signed weights, "
            << "and EOF passed\n";
}
