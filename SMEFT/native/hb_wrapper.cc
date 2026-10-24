// C ABI around MG5-generated LO SMEFTsim amplitudes. No amplitudes are supplied
// by this wrapper. See generated/provenance.json and licenses/ for provenance.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "../generated/processes/mixed/CPPProcess.cc"
#include "../generated/processes/eeee/CPPProcess.cc"
#include "../generated/processes/mmmm/CPPProcess.cc"

namespace {
using Parameters = Parameters_SMEFTsim_U35_MwScheme_UFO_OHB;
std::mutex backend_mutex;
HBMixedProcess mixed_process;
HBEEEEProcess eeee_process;
HBMMMMProcess mmmm_process;
bool process_objects_initialized = false;
bool initialized = false;

void write_error(char* error, std::size_t size, const std::string& message) {
  if (!error || size == 0) return;
  const auto count = std::min(size - 1, message.size());
  std::memcpy(error, message.data(), count);
  error[count] = '\0';
}

struct Event {
  // Parent first, followed by e-/mu-, e+/mu+, e-/mu-, e+/mu+.
  std::array<std::array<double, 4>, 5> momentum{};
  int channel = -1;
};

Event prepare_event(const double* p4, const int* pdg) {
  std::vector<int> electrons, positrons, muons, antimuons;
  for (int i = 0; i < 4; ++i) {
    switch (pdg[i]) {
      case 11: electrons.push_back(i); break;
      case -11: positrons.push_back(i); break;
      case 13: muons.push_back(i); break;
      case -13: antimuons.push_back(i); break;
      default: throw std::runtime_error("Only direct e/mu channels are supported; tau daughters are not massless e/mu");
    }
  }
  Event event;
  std::array<int, 4> order{};
  if (electrons.size() == 1 && positrons.size() == 1 &&
      muons.size() == 1 && antimuons.size() == 1) {
    event.channel = 0;
    order = {{electrons[0], positrons[0], muons[0], antimuons[0]}};
  } else if (electrons.size() == 2 && positrons.size() == 2) {
    event.channel = 1;
    order = {{electrons[0], positrons[0], electrons[1], positrons[1]}};
  } else if (muons.size() == 2 && antimuons.size() == 2) {
    event.channel = 2;
    order = {{muons[0], antimuons[0], muons[1], antimuons[1]}};
  } else {
    throw std::runtime_error("Expected 2e2mu, 4e or 4mu with two opposite-sign same-flavor pairs");
  }
  for (int i = 0; i < 4; ++i) {
    for (int component = 0; component < 4; ++component) {
      const double value = p4[4 * order[i] + component];
      if (!std::isfinite(value)) throw std::runtime_error("Nonfinite four-momentum");
      event.momentum[i + 1][component] = value;
      event.momentum[0][component] += value;
    }
    const auto& p = event.momentum[i + 1];
    if (p[0] <= 0.) throw std::runtime_error("Lepton energy must be positive");
    const double mass2 = p[0] * p[0] - p[1] * p[1] - p[2] * p[2] - p[3] * p[3];
    if (!std::isfinite(mass2) || std::abs(mass2) > 1.e-8 * std::max(1., p[0] * p[0]))
      throw std::runtime_error("Lepton four-momenta must be massless; project them before evaluation");
  }
  // The external Higgs is exactly the sum of its four daughters; no independent
  // parent input can introduce a mismatch. An off-shell external mass is allowed.
  const auto& h = event.momentum[0];
  const double mass2 = h[0] * h[0] - h[1] * h[1] - h[2] * h[2] - h[3] * h[3];
  if (!std::isfinite(mass2) || mass2 <= 0.)
    throw std::runtime_error("Four-lepton parent must have finite positive invariant mass");
  return event;
}

template <class Process>
double calculate(Process& process, std::vector<double*>& momenta) {
  process.setMomenta(momenta);
  process.sigmaKin();
  return process.getMatrixElements()[0];
}

double matrix_element(Event& event, double coefficient) {
  auto* parameters = Parameters::getInstance();
  parameters->mdl_cHB = coefficient;
  // For this restricted model all cHB-dependent amplitude couplings are
  // defined in this function. Fixed widths and EW inputs are never changed.
  parameters->setIndependentCouplings();
  std::vector<double*> momenta;
  for (auto& p : event.momentum) momenta.push_back(p.data());
  double value = 0.;
  if (event.channel == 0) value = calculate(mixed_process, momenta);
  if (event.channel == 1) value = calculate(eeee_process, momenta);
  if (event.channel == 2) value = calculate(mmmm_process, momenta);
  if (!std::isfinite(value) || value < 0.)
    throw std::runtime_error("Generated matrix element is nonfinite or negative (check singular kinematics)");
  return value;
}

void check_input(std::size_t n, const double* p4, const int* pdg, double* output) {
  if (!initialized) throw std::runtime_error("Call hb_init with a valid parameter card first");
  if (n > std::numeric_limits<std::size_t>::max() / 16)
    throw std::runtime_error("Batch size exceeds addressable range");
  if (n && (!p4 || !pdg || !output)) throw std::runtime_error("Null batch pointer");
}

template <class Function>
int protected_call(char* error, std::size_t size, Function function) {
  std::lock_guard<std::mutex> lock(backend_mutex);
  write_error(error, size, "");
  try {
    function();
    return 0;
  } catch (const std::exception& exception) {
    write_error(error, size, exception.what());
  } catch (const char* message) {  // The generated SLHA reader uses this form.
    write_error(error, size, message);
  } catch (...) {
    write_error(error, size, "Unknown native backend failure");
  }
  return 1;
}
}  // namespace

extern "C" int hb_init(const char* card_path, char* error, std::size_t error_size) {
  return protected_call(error, error_size, [&] {
    initialized = false;
    if (!card_path || !*card_path) throw std::runtime_error("Parameter card path is empty");
    SLHAReader card(card_path);
    auto* parameters = Parameters::getInstance();
    parameters->setIndependentParameters(card);
    const double required[] = {parameters->mdl_MW, parameters->mdl_MZ,
      parameters->mdl_Gf, parameters->mdl_LambdaSMEFT, parameters->mdl_WZ};
    if (!std::all_of(std::begin(required), std::end(required),
                     [](double value) { return std::isfinite(value) && value > 0.; }) ||
        parameters->mdl_MW >= parameters->mdl_MZ)
      throw std::runtime_error("Invalid fixed electroweak inputs, cutoff or Z width");
    if (std::abs(parameters->mdl_LambdaSMEFT - 1000.) > 1.e-9)
      throw std::runtime_error("This backend requires the fixed LambdaSMEFT=1000 GeV reference convention");
    parameters->setIndependentCouplings();
    if (!process_objects_initialized) {
      mixed_process.initProc(card_path);
      eeee_process.initProc(card_path);
      mmmm_process.initProc(card_path);
      process_objects_initialized = true;
    }
    initialized = true;
  });
}

// Outputs normalized polynomial coefficients l,q for w(cHB)=1+cHB*l+cHB^2*q.
// Input rows: four direct massless lepton vectors, each ordered [E,px,py,pz].
// Arbitrary input particle ordering is accepted when PDGs follow the same order.
extern "C" int hb_evaluate(std::size_t n, const double* p4, const int* pdg,
                           double* linear, double* quadratic,
                           char* error, std::size_t error_size) {
  return protected_call(error, error_size, [&] {
    check_input(n, p4, pdg, linear);
    if (n && !quadratic) throw std::runtime_error("Null quadratic output pointer");
    if (n) {
      std::fill_n(linear, n, std::numeric_limits<double>::quiet_NaN());
      std::fill_n(quadratic, n, std::numeric_limits<double>::quiet_NaN());
    }
    for (std::size_t i = 0; i < n; ++i) {
      try {
        auto event = prepare_event(p4 + 16 * i, pdg + 4 * i);
        const double sm = matrix_element(event, 0.);
        if (sm <= 0.) throw std::runtime_error("SM matrix element must be positive for a ratio");
        const double plus = matrix_element(event, 1.) / sm;
        const double minus = matrix_element(event, -1.) / sm;
        const double l = 0.5 * (plus - minus);
        double q = 0.5 * (plus + minus) - 1.;
        const double tolerance = 1.e-12 * (1. + std::abs(plus) + std::abs(minus));
        if (q < -tolerance || !std::isfinite(l) || !std::isfinite(q))
          throw std::runtime_error("Nonfinite polynomial coefficient or negative squared-EFT contribution");
        if (q < 0.) q = 0.;  // Only a tiny cancellation error in a nonnegative square.
        if (l * l > 4. * q + 1.e-9 * (1. + l * l + 4. * q))
          throw std::runtime_error("Polynomial coefficients violate the amplitude Cauchy bound");
        linear[i] = l;
        quadratic[i] = q;
      } catch (const std::exception& exception) {
        throw std::runtime_error("event " + std::to_string(i) + ": " + exception.what());
      }
    }
  });
}

// Direct generated |M(cHB)|^2 for independent checks at arbitrary coefficients.
// Includes MG5's identical-particle factor (1/4 for 4e/4mu, unity for 2e2mu),
// which cancels in the same-channel EFT/SM ratio.
extern "C" int hb_matrix_elements(std::size_t n, const double* p4, const int* pdg,
                                  double coefficient, double* output,
                                  char* error, std::size_t error_size) {
  return protected_call(error, error_size, [&] {
    check_input(n, p4, pdg, output);
    if (!std::isfinite(coefficient)) throw std::runtime_error("Coefficient must be finite");
    if (n) std::fill_n(output, n, std::numeric_limits<double>::quiet_NaN());
    for (std::size_t i = 0; i < n; ++i) {
      try {
        auto event = prepare_event(p4 + 16 * i, pdg + 4 * i);
        output[i] = matrix_element(event, coefficient);
      } catch (const std::exception& exception) {
        throw std::runtime_error("event " + std::to_string(i) + ": " + exception.what());
      }
    }
  });
}
