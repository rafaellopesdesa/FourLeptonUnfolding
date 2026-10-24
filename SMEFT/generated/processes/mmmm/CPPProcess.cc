//==========================================================================
// This file has been automatically generated for C++ Standalone by
// MadGraph5_aMC@NLO v. 3.5.17, 2026-10-09
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#include "CPPProcess.h"
#include "HelAmps_SMEFTsim_U35_MwScheme_UFO_OHB.h"

using namespace MG5_SMEFTsim_U35_MwScheme_UFO_OHB; 

//==========================================================================
// Class member functions for calculating the matrix elements for
// Process: h > mu- mu+ mu- mu+ NP<=1 QCD=0 SMHLOOP=0 @3

//--------------------------------------------------------------------------
// Initialize process.

void HBMMMMProcess::initProc(string param_card_name) 
{
  // Instantiate the model class and set parameters that stay fixed during run
  pars = Parameters_SMEFTsim_U35_MwScheme_UFO_OHB::getInstance(); 
  SLHAReader slha(param_card_name); 
  pars->setIndependentParameters(slha); 
  pars->setIndependentCouplings(); 
  // Set external particle masses for this matrix element
  mME.push_back(pars->mdl_MH); 
  mME.push_back(pars->ZERO); 
  mME.push_back(pars->ZERO); 
  mME.push_back(pars->ZERO); 
  mME.push_back(pars->ZERO); 
  jamp2[0] = new double[1]; 
}

//--------------------------------------------------------------------------
// Evaluate |M|^2, part independent of incoming flavour.

void HBMMMMProcess::sigmaKin()
{
  // Interface patch: evaluate every helicity at every coefficient/phase point.
  // No helicity learned at an earlier point may suppress a later contribution.
  pars->setDependentParameters();
  pars->setDependentCouplings();
  const int ncomb = 16;
  static const int helicities[ncomb][nexternal] = {{0, -1, -1, -1, -1}, {0, -1,
      -1, -1, 1}, {0, -1, -1, 1, -1}, {0, -1, -1, 1, 1}, {0, -1, 1, -1, -1},
      {0, -1, 1, -1, 1}, {0, -1, 1, 1, -1}, {0, -1, 1, 1, 1}, {0, 1, -1, -1,
      -1}, {0, 1, -1, -1, 1}, {0, 1, -1, 1, -1}, {0, 1, -1, 1, 1}, {0, 1, 1,
      -1, -1}, {0, 1, 1, -1, 1}, {0, 1, 1, 1, -1}, {0, 1, 1, 1, 1}};
  const int denominators[nprocesses] = {4};
  int perm[nexternal];
  for (int i = 0; i < nexternal; ++i) perm[i] = i;
  matrix_element[0] = 0.;
  jamp2[0][0] = 0.;
  for (int ihel = 0; ihel < ncomb; ++ihel) {
    calculate_wavefunctions(perm, helicities[ihel]);
    matrix_element[0] += matrix_3_h_mummupmummup();
  }
  matrix_element[0] /= denominators[0];
}

//--------------------------------------------------------------------------
// Evaluate |M|^2, including incoming flavour dependence.

double HBMMMMProcess::sigmaHat() 
{
  // Select between the different processes
  if(id1 == 25 && id2 == 13)
  {
    // Add matrix elements for processes with beams (25, 13)
    return matrix_element[0]; 
  }
  else
  {
    // Return 0 if not correct initial state assignment
    return 0.; 
  }
}

//==========================================================================
// Private class member functions

//--------------------------------------------------------------------------
// Evaluate |M|^2 for each subprocess

void HBMMMMProcess::calculate_wavefunctions(const int perm[], const int hel[])
{
  // Calculate wavefunctions for all processes
  int i, j; 

  // Calculate all wavefunctions
  sxxxxx(p[perm[0]], -1, w[0]); 
  oxxxxx(p[perm[1]], mME[1], hel[1], +1, w[1]); 
  ixxxxx(p[perm[2]], mME[2], hel[2], -1, w[2]); 
  oxxxxx(p[perm[3]], mME[3], hel[3], +1, w[3]); 
  ixxxxx(p[perm[4]], mME[4], hel[4], -1, w[4]); 
  FFV1P0_3(w[2], w[1], -pars->GC_3, pars->ZERO, pars->ZERO, w[5]); 
  FFV1P0_3(w[4], w[3], -pars->GC_3, pars->ZERO, pars->ZERO, w[6]); 
  FFV1_3_3(w[4], w[3], pars->GC_197, -pars->GC_135, pars->mdl_MZ, pars->mdl_WZ,
      w[7]);
  FFV1_3_3(w[2], w[1], pars->GC_197, -pars->GC_135, pars->mdl_MZ, pars->mdl_WZ,
      w[8]);
  FFV1P0_3(w[4], w[1], -pars->GC_3, pars->ZERO, pars->ZERO, w[9]); 
  FFV1P0_3(w[2], w[3], -pars->GC_3, pars->ZERO, pars->ZERO, w[10]); 
  FFV1_3_3(w[2], w[3], pars->GC_197, -pars->GC_135, pars->mdl_MZ, pars->mdl_WZ,
      w[11]);
  FFV1_3_3(w[4], w[1], pars->GC_197, -pars->GC_135, pars->mdl_MZ, pars->mdl_WZ,
      w[12]);

  // Calculate all amplitudes
  // Amplitude(s) for diagram number 0
  VVS4_0(w[5], w[6], w[0], pars->GC_268, amp[0]); 
  VVS4_0(w[5], w[7], w[0], pars->GC_318, amp[1]); 
  VVS4_0(w[6], w[8], w[0], pars->GC_318, amp[2]); 
  VVS3_0(w[8], w[7], w[0], pars->GC_284, amp[3]); 
  VVS4_0(w[8], w[7], w[0], pars->GC_326, amp[4]); 
  VVS4_0(w[9], w[10], w[0], pars->GC_268, amp[5]); 
  VVS4_0(w[9], w[11], w[0], pars->GC_318, amp[6]); 
  VVS4_0(w[10], w[12], w[0], pars->GC_318, amp[7]); 
  VVS3_0(w[12], w[11], w[0], pars->GC_284, amp[8]); 
  VVS4_0(w[12], w[11], w[0], pars->GC_326, amp[9]); 

}
double HBMMMMProcess::matrix_3_h_mummupmummup() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 10; 
  const int ncolor = 1; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1}; 
  static const double cf[ncolor][ncolor] = {{1}}; 

  // Calculate color flows
  jamp[0] = +amp[0] + amp[1] + amp[2] + amp[3] + amp[4] - amp[5] - amp[6] -
      amp[7] - amp[8] - amp[9];

  // Sum and square the color flows to get the matrix element
  double matrix = 0; 
  for(i = 0; i < ncolor; i++ )
  {
    ztemp = 0.; 
    for(j = 0; j < ncolor; j++ )
      ztemp = ztemp + cf[i][j] * jamp[j]; 
    matrix = matrix + real(ztemp * conj(jamp[i]))/denom[i]; 
  }

  // Store the leading color flows for choice of color
  for(i = 0; i < ncolor; i++ )
    jamp2[0][i] += real(jamp[i] * conj(jamp[i])); 

  return matrix; 
}



