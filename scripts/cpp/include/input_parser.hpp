
#pragma once
#include "json/json.hpp"
#include <string>

class Grid;

typedef enum CalculationType {
  deformation_curve,
  ground_state
} CalculationType;

typedef struct DeformationCurve
{
  double start;
  double end;
  double step;
} DeformationCurve;

typedef struct GCGParameters
{
  int nev;
  double tol;
  int maxIter;
  int steps;
  double cgTol;
} CGCParameters;
typedef struct
{
  double W0;
  double t0;
  double t1;
  double t2;
  double t3;
  double x0;
  double x1;
  double x2;
  double x3;
  double sigma;
} SkyrmeParameters;
typedef struct HartreeFock
{
  int cycles;
  double energyTol;
  GCGParameters gcg;
} HartreeFock;
typedef struct
{
  GCGParameters initialGCG;
  HartreeFock hf;
} Calculation;

typedef struct
{
  double V0;
  double r0;
  double diffusivity;
  double kappa;
} WoodsSaxonParameters;

typedef struct
{
  double V0;
  double r0;
  double diff;
} WSSpinOrbitParameters;

typedef struct
{
  double window;
  int additionalStates;
  double V0N;
  double V0P;
  double alpha;
} PairingParameters;

/**
 * @brief Gathers the objects related to a calculation.
 */
class InputParser
{
public:
  /**
   * @param inputFile The path to the input file in the input directory.
   */
  InputParser(std::string inputFile);
  nlohmann::json get_json();
  int A, Z;
  Grid get_grid();
  int getA();
  int getZ();
  bool useCoulomb;
  bool useJ;
  double getKappa();
  Calculation getCalculation();
  std::string getOutputName();
  nlohmann::json getWoodsSaxon();
  bool pairing, spinOrbit, COMCorr;
  WoodsSaxonParameters getWS();
  WSSpinOrbitParameters getWSSO();
  std::string outputDirectory;
  std::vector<std::string> log;
  double initialBeta;
  double beta3;
  PairingParameters pairingParameters;

  SkyrmeParameters skyrme;
  CalculationType calculationType;
  DeformationCurve deformationCurve;

private:
  nlohmann::json data;
  // Hamiltonian h;
};
