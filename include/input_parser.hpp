

#pragma once

#include "json/json.hpp"
#include <string>

class EDF;

class Grid;

typedef enum CalculationType {
  deformation_curve,
  ground_state
} CalculationType;

typedef struct DeformationCurve {
  double start;
  double end;
  double step;
} DeformationCurve;

typedef struct MultipoleConstraintInput {
  int iter_start;
  int iter_end;
  int l;
  int m;
  double target;
} MultipoleConstraintInput;

typedef struct GCGParameters {
  int nev;
  double tol;
  int maxIter;
  int steps;
  double cgTol;
} CGCParameters;
typedef struct HartreeFock {
  int cycles;
  double energyTol;
  GCGParameters gcg;
} HartreeFock;
typedef struct {
  GCGParameters initialGCG;
  HartreeFock hf;
} Calculation;

typedef enum PairingType { none, hfb, bcs } PairingType;

typedef struct {
  double V0;
  double r0;
  double diffusivity;
  double kappa;
} WoodsSaxonParameters;

typedef struct {
  double V0;
  double r0;
  double diff;
} WSSpinOrbitParameters;

typedef struct {
  double window;
  int additionalStates;
  double V0;
  double alpha;
  double eta;
  bool windowBoth;
  bool active;
} PairingParameters;

/**
 * @brief Gathers the objects related to a calculation.
 */
class InputParser {
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
  PairingType pairingType;
  WoodsSaxonParameters getWS();

  WSSpinOrbitParameters getWSSO();
  std::string outputDirectory;
  std::vector<std::string> log;
  double initialBeta;
  double beta3;
  double densityMix;
  PairingParameters pairingN, pairingP;
  bool useDIIS;

  std::vector<MultipoleConstraintInput> multipoleConstraints;

  std::shared_ptr<EDF> interaction;
  CalculationType calculationType;
  DeformationCurve deformationCurve;
  WoodsSaxonParameters _woodsSaxonParameters;
  WSSpinOrbitParameters _WSSpinOrbitParameters;
  Calculation calculation;

  double pairingThreshold;
  double constraintsTol;
  int startHFBIter;
  bool constrainCOM;

  bool check();

private:
  nlohmann::json data;
  nlohmann::json woodsSaxonData;
  nlohmann::json hfData;
};
