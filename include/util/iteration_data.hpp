#pragma once
#include "input_parser.hpp"
#include "types.hpp"
#include "util/bcs.hpp"
#include <Eigen/Dense>
#include <memory>

class InputParser;
class Constraint;
class Grid;
class EffectiveMass;
class EDF;

typedef struct QuadrupoleDeformation {
  double beta;
  double gamma;
} QuadrupoleDeformation;

typedef struct UV {
  Eigen::MatrixXd U;
  Eigen::MatrixXd V;
  Eigen::MatrixXd kappa;
  Eigen::VectorXd pairingField;
  double lambda;
  double energy;
} UV;
typedef struct HFBResult {
  UV uv_n;
  UV uv_p;
} HFBResult;

class IterationData {
public:
  std::shared_ptr<EffectiveMass> massN;
  std::shared_ptr<EffectiveMass> massP;
  std::shared_ptr<Eigen::VectorXd> rhoN;
  std::shared_ptr<Eigen::VectorXd> rhoP;
  std::shared_ptr<Eigen::VectorXd> tauN;
  std::shared_ptr<Eigen::VectorXd> tauP;
  std::shared_ptr<Eigen::VectorXd> divJvecN;
  std::shared_ptr<Eigen::VectorXd> divJvecP;
  std::shared_ptr<Eigen::MatrixX3d> nablaRhoN;
  std::shared_ptr<Eigen::MatrixX3d> nablaRhoP;
  std::shared_ptr<Eigen::VectorXd> nabla2RhoN;
  std::shared_ptr<Eigen::VectorXd> nabla2RhoP;
  std::shared_ptr<Real2Tensor> JN;
  std::shared_ptr<Real2Tensor> JP;
  std::shared_ptr<Eigen::MatrixX3d> JvecN;
  std::shared_ptr<Eigen::MatrixX3d> JvecP;
  std::shared_ptr<Eigen::VectorXcd> divJN;
  std::shared_ptr<Eigen::VectorXcd> divJP;
  std::shared_ptr<Eigen::VectorXd> UN;
  std::shared_ptr<Eigen::VectorXd> UP;
  std::shared_ptr<Eigen::VectorXd> UConstr;
  std::shared_ptr<Eigen::MatrixX3d> BN;
  std::shared_ptr<Eigen::MatrixX3d> BP;
  std::shared_ptr<Eigen::VectorXd> UCoul;

  UV HFBResultN;
  UV HFBResultP;
  double oldLambdaN;
  double oldLambdaP;

  Eigen::VectorXd UCDir;

  BCS::BCSResult bcsN;
  BCS::BCSResult bcsP;
  InputParser input;
  double massCorr;
  double energyDiff;
  double constraintTol = 5e-3;

  void mixDensity(const Eigen::MatrixXcd &newDensity,
                  std::vector<std::shared_ptr<Eigen::MatrixXcd>> history);

  typedef struct BCSPairingSolution {
    BCS::BCSResult bcsN;
    BCS::BCSResult bcsP;
  } BCSPairingSolution;

  HFBResult solvePairingHFB(
      const std::pair<Eigen::MatrixXcd, Eigen::VectorXd> &neutronsPair,
      const std::pair<Eigen::MatrixXcd, Eigen::VectorXd> &protonsPair);

  void solvePairingBCS(
      const std::pair<Eigen::MatrixXcd, Eigen::VectorXd> &neutronsPair,
      const std::pair<Eigen::MatrixXcd, Eigen::VectorXd> &protonsPair);

  void updateQuantities(
      const std::pair<Eigen::MatrixXcd, Eigen::VectorXd> &neutronsEigenpair,
      const std::pair<Eigen::MatrixXcd, Eigen::VectorXd> &protonsEigenpair,
      int iter, const std::vector<std::unique_ptr<Constraint>> &constraints);

  double
  constraintEnergy(const std::vector<std::unique_ptr<Constraint>> &constraints);
  double totalEnergyIntegral();
  double rearrangementIntegral();
  double HFEnergy(double SPE,
                  const std::vector<std::unique_ptr<Constraint>> &constraints);
  double Erear();

  double EpairN();
  double EpairP();

  double fermiEnergyN();
  double fermiEnergyP();

  Eigen::MatrixXcd protonsFromPairing(const Eigen::MatrixXcd &phi);
  Eigen::MatrixXcd neutronsFromPairing(const Eigen::MatrixXcd &phi);
  Eigen::MatrixXcd protonsFromBCS(const Eigen::MatrixXcd &phi);
  Eigen::MatrixXcd neutronsFromBCS(const Eigen::MatrixXcd &phi);

  double C0RhoEnergy();
  double C1RhoEnergy();

  double C0rhoDivJEnergy();
  double C1rhoDivJEnergy();

  double C0J2Energy();
  double C1J2Energy();

  double C0TauEnergy();
  double C1TauEnergy();

  double Hso();
  double Hsg();

  double C0nabla2RhoEnergy();
  double C1nabla2RhoEnergy();

  void recomputeLagrange(
      const std::pair<Eigen::MatrixXcd, Eigen::VectorXd> &neutronsPair,
      const std::pair<Eigen::MatrixXcd, Eigen::VectorXd> &protonsPair);

  double kineticEnergy();
  double kineticEnergyEff();
  double coulombEnergy(const Grid &grid);

  double densityUVPIntegral(const Grid &grid);

  double CoulombDirectEnergy(const Grid &grid);
  double SlaterCoulombEnergy(const Grid &grid);

  QuadrupoleDeformation quadrupoleDeformation();
  double betaRealRadius();

  double axis2Exp(char dir);

  std::shared_ptr<EDF> interaction;

  double protonRadius();
  double radius();
  double neutronRadius();
  double chargeRadius(const Eigen::MatrixXcd psiN, const Eigen::MatrixXcd psiP,
                      int N, int Z);
  void
  logData(const std::pair<Eigen::MatrixXcd, Eigen::VectorXd> &neutronsEigenpair,
          const std::pair<Eigen::MatrixXcd, Eigen::VectorXd> &protonsEigenpair,
          const std::vector<std::unique_ptr<Constraint>> &constraints);

  IterationData(InputParser input);
};
