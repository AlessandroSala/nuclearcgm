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

typedef struct QuadrupoleDeformation {
  double beta;
  double gamma;
} QuadrupoleDeformation;

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
  std::vector<std::shared_ptr<Eigen::VectorXd>> rhoNHistory;
  std::vector<std::shared_ptr<Eigen::VectorXd>> rhoPHistory;
  std::vector<std::shared_ptr<Eigen::VectorXd>> tauNHistory;
  std::vector<std::shared_ptr<Eigen::VectorXd>> tauPHistory;
  std::vector<std::shared_ptr<Real2Tensor>> JNHistory;
  std::vector<std::shared_ptr<Real2Tensor>> JPHistory;
  std::vector<std::shared_ptr<Eigen::VectorXd>> divJvecNHistory;
  std::vector<std::shared_ptr<Eigen::VectorXd>> divJvecPHistory;

  Eigen::VectorXd UCDir;

  BCS::BCSResult bcsN;
  BCS::BCSResult bcsP;
  bool useDIIS;
  SkyrmeParameters params;
  InputParser input;
  double massCorr;
  double energyDiff;
  double constraintTol = 5e-3;
  double mu = 0.2;
  int lastConvergedIter;

  void mixDensity(const Eigen::MatrixXcd &newDensity,
                  std::vector<std::shared_ptr<Eigen::MatrixXcd>> history);

  void updateQuantities(
      const std::pair<Eigen::MatrixXcd, Eigen::VectorXd> &neutronsEigenpair,
      const std::pair<Eigen::MatrixXcd, Eigen::VectorXd> &protonsEigenpair,
      int iter, const std::vector<std::unique_ptr<Constraint>> &constraints);

  double
  constraintEnergy(const std::vector<std::unique_ptr<Constraint>> &constraints);
  double totalEnergyIntegral(SkyrmeParameters params, const Grid &grid);
  double rearrangementIntegral(SkyrmeParameters params, const Grid &grid);
  double HFEnergy(double SPE,
                  const std::vector<std::unique_ptr<Constraint>> &constraints);
  double Erear(const Grid &grid);

  double C0RhoEnergy(SkyrmeParameters params, const Grid &grid);
  double C1RhoEnergy(SkyrmeParameters params, const Grid &grid);

  double C0TauEnergy(SkyrmeParameters params, const Grid &grid);
  double C1TauEnergy(SkyrmeParameters params, const Grid &grid);

  double Hso(SkyrmeParameters params, const Grid &grid);
  double Hsg(SkyrmeParameters params, const Grid &grid);

  double C0nabla2RhoEnergy(SkyrmeParameters params, const Grid &grid);
  double C1nabla2RhoEnergy(SkyrmeParameters params, const Grid &grid);

  double kineticEnergy(SkyrmeParameters params, const Grid &grid);
  double kineticEnergyEff(SkyrmeParameters params, const Grid &grid);
  double coulombEnergy(SkyrmeParameters params, const Grid &grid);

  double densityUVPIntegral(const Grid &grid);

  double CoulombDirectEnergy(const Grid &grid);
  double SlaterCoulombEnergy(const Grid &grid);

  QuadrupoleDeformation quadrupoleDeformation();
  double betaRealRadius();

  double axis2Exp(char dir);

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
