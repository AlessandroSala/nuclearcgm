#pragma once
#include "input_parser.hpp"
#include "types.hpp"
#include <Eigen/Dense>
#include <memory>

class InputParser;
class Grid;
class EffectiveMass;

class IterationData {
public:
  std::shared_ptr<EffectiveMass> massN;
  std::shared_ptr<EffectiveMass> massP;
  std::shared_ptr<Eigen::VectorXd> rhoN;
  std::shared_ptr<Eigen::VectorXd> rhoP;
  std::shared_ptr<Eigen::VectorXd> tauN;
  std::shared_ptr<Eigen::VectorXd> tauP;
  std::shared_ptr<Eigen::MatrixX3d> nablaRhoN;
  std::shared_ptr<Eigen::MatrixX3d> nablaRhoP;
  std::shared_ptr<Eigen::MatrixX3d> spinN;
  std::shared_ptr<Eigen::MatrixX3d> spinP;
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
  std::shared_ptr<Eigen::MatrixX3d> BN;
  std::shared_ptr<Eigen::MatrixX3d> BP;
  std::shared_ptr<Eigen::VectorXd> UCoul;
  Eigen::VectorXd UCDir;

  SkyrmeParameters params;
  InputParser input;
  double massCorr;

  void updateQuantities(const Eigen::MatrixXcd &neutrons,
                        const Eigen::MatrixXcd &protons, const Grid &grid);

  double totalEnergyIntegral(SkyrmeParameters params, const Grid &grid);
  double rearrangementIntegral(SkyrmeParameters params, const Grid &grid);
  double HFEnergy(double SPE, const Grid &grid);
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

  IterationData(InputParser input);
};
