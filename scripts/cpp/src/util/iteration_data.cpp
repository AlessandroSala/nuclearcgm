#include "util/iteration_data.hpp"
#include "constants.hpp"
#include "constraint.hpp"
#include "grid.hpp"
#include "input_parser.hpp"
#include "operators/angular_momentum.hpp"
#include "operators/common_operators.hpp"
#include "operators/differential_operators.hpp"
#include "operators/integral_operators.hpp"
#include "util/effective_mass.hpp"
#include "util/fields.hpp"
#include "util/wavefunction.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>

void IterationData::logData(
    const std::pair<Eigen::MatrixXcd, Eigen::VectorXd> &neutronsEigenpair,
    const std::pair<Eigen::MatrixXcd, Eigen::VectorXd> &protonsEigenpair,
    const std::vector<std::unique_ptr<Constraint>> &constraints) {
  auto grid = *Grid::getInstance();
  using std::cout;
  using std::endl;

  double newIntegralEnergy = totalEnergyIntegral(input.skyrme, grid) +
                             kineticEnergy(input.skyrme, grid);

  double SPE =
      0.5 * (neutronsEigenpair.second.sum() + protonsEigenpair.second.sum());
  double newHFEnergy = HFEnergy(2.0 * SPE, constraints);
  cout << "Total energy as integral: " << newIntegralEnergy << endl;
  cout << "Total energy as HF energy: " << newHFEnergy << endl;

  cout << "Direct coulomb energy: " << CoulombDirectEnergy(grid) << endl;
  cout << "Slater exchange energy: " << SlaterCoulombEnergy(grid) << endl;
  std::cout << "X2: " << axis2Exp('x') << std::endl;
  std::cout << "Y2: " << axis2Exp('y') << std::endl;
  std::cout << "Z2: " << axis2Exp('z') << std::endl;
  std::cout << "RN: " << std::sqrt(neutronRadius()) << std::endl;
  std::cout << "RP: " << std::sqrt(protonRadius()) << std::endl;
  std::cout << "CR: "
            << std::sqrt(chargeRadius(neutronsEigenpair.first,
                                      protonsEigenpair.first, N, input.getZ()))
            << std::endl;

  auto [beta, gamma] = quadrupoleDeformation();

  std::cout << "Beta: " << beta << ", Gamma: " << gamma * 180.0 / M_PI << " deg"
            << std::endl;
  std::cout << "Beta w/ real radius: " << betaRealRadius()
            << ", Gamma: " << gamma * 180.0 / M_PI << " deg" << std::endl;
}

IterationData::IterationData(InputParser input) : input(input) {
  params = input.skyrme;
  int A = input.getA();
  using nuclearConstants::m;
  energyDiff = 1.0;

  massCorr = input.COMCorr ? m * ((double)A) / ((double)(A - 1)) : m;

  std::cout << "Created IterationData" << std::endl;
}

double IterationData::protonRadius() {

  auto grid = *Grid::getInstance();
  auto pos = Fields::position();
  using Eigen::VectorXd;
  using Operators::integral;
  VectorXd f = pos.array().pow(2) * rhoP->array();

  return integral(f, grid) / integral(*rhoP, grid);
}
double IterationData::neutronRadius() {

  auto grid = *Grid::getInstance();
  auto pos = Fields::position();
  using Eigen::VectorXd;
  using Operators::integral;
  VectorXd f = pos.array().pow(2) * rhoN->array();

  return integral(f, grid) / integral(*rhoN, grid);
}

double IterationData::radius() {

  auto grid = *Grid::getInstance();
  auto pos = Fields::position();
  using Eigen::VectorXd;
  using Operators::integral;
  VectorXd rho = *rhoN + *rhoP;
  VectorXd f = pos.array().pow(2) * rho.array();

  return integral(f, grid) / integral(rho, grid);
}

double IterationData::chargeRadius(const Eigen::MatrixXcd psiN,
                                   const Eigen::MatrixXcd psiP, int N, int Z) {
  auto grid = *Grid::getInstance();

  using namespace nuclearConstants;
  using Eigen::VectorXcd;
  using Eigen::VectorXd;
  using Operators::integral;
  using Operators::LS;

  double corr = 0.0;
  for (int i = 0; i < psiN.cols(); i++) {
    VectorXcd lsPsi = LS(psiN.col(i), grid);
    auto ls =
        integral((VectorXcd)(lsPsi * psiN.col(i).adjoint()).diagonal(), grid)
            .real();
    corr += muN * ls / h_bar / h_bar;
  }

  for (int i = 0; i < psiP.cols(); i++) {
    VectorXcd lsPsi = LS(psiP.col(i), grid);
    double ls =
        integral((VectorXcd)(lsPsi * psiP.col(i).adjoint()).diagonal(), grid)
            .real();
    corr += muP * ls / h_bar / h_bar;
  }
  corr *= 2.0 * (h_bar * h_bar / m / m) / Z;

  return std::abs(protonRadius() + N * rn / Z + rp + corr);
}

double IterationData::totalEnergyIntegral(SkyrmeParameters params,
                                          const Grid &grid) {
  double energy_C0Rho = C0RhoEnergy(params, grid);
  double energy_C1Rho = C1RhoEnergy(params, grid);

  double energy_C0nabla2Rho = C0nabla2RhoEnergy(params, grid);
  double energy_C1nabla2Rho = C1nabla2RhoEnergy(params, grid);

  double energy_C0Tau = C0TauEnergy(params, grid);
  double energy_C1Tau = C1TauEnergy(params, grid);

  double energy_CoulombDirect =
      input.useCoulomb ? CoulombDirectEnergy(grid) : 0.0;

  double energy_Hso = input.spinOrbit ? Hso(params, grid) : 0.0;

  double energy_Hsg = input.useJ ? Hsg(params, grid) : 0.0;

  std::cout << "C0RhoEnergy: " << energy_C0Rho
            << ", C1RhoEnergy: " << energy_C1Rho
            << ", C0nabla2RhoEnergy: " << energy_C0nabla2Rho
            << ", C1nabla2RhoEnergy: " << energy_C1nabla2Rho
            << ", C0TauEnergy: " << energy_C0Tau
            << ", C1TauEnergy: " << energy_C1Tau
            << ", CoulombDirectEnergy: " << energy_CoulombDirect
            << ", Hso: " << energy_Hso << ", Hsg: " << energy_Hsg << std::endl;

  return energy_C0Rho + energy_C1Rho + energy_C0nabla2Rho + energy_C1nabla2Rho +
         energy_C0Tau + energy_C1Tau + energy_CoulombDirect + bcsN.Epair +
         bcsP.Epair + SlaterCoulombEnergy(grid) + energy_Hso + energy_Hsg;
}

double IterationData::Erear(const Grid &grid) {
  double t0 = params.t0;
  double t3 = params.t3;
  double x0 = params.x0;
  double x3 = params.x3;
  double sigma = params.sigma;

  Eigen::VectorXd ones = Eigen::VectorXd::Ones(grid.get_total_spatial_points());

  Eigen::VectorXd rho0 = *rhoN + *rhoP;
  Eigen::VectorXd rho1 = *rhoN - *rhoP;

  Eigen::VectorXd energy1 =
      sigma * rho1.array().pow(sigma) *
      ((1.0 / 12.0) * t3 * (1 + x3 / 2.0) * rho0.array().pow(2) -
       (1.0 / 12.0) * t3 * (0.5 + x3) *
           (rhoN->array().pow(2) + rhoP->array().pow(2)));
  if (energy1.array().isNaN().any()) {
    energy1 = Eigen::VectorXd::Zero(grid.get_total_spatial_points());
  }

  Eigen::VectorXd energy0 =
      sigma * rho0.array().pow(sigma) *
      ((1.0 / 12.0) * t3 * (1 + x3 / 2.0) * rho0.array().pow(2) -
       (1.0 / 12.0) * t3 * (0.5 + x3) *
           (rhoN->array().pow(2) + rhoP->array().pow(2)));

  using Operators::integral;
  return integral(Eigen::VectorXd(energy0 + energy1), grid);
}
double IterationData::HFEnergy(
    double SPE, const std::vector<std::unique_ptr<Constraint>> &constraints) {
  auto grid = *Grid::getInstance();
  double constraintEnergy = 0.0;
  for (auto &&constraint : constraints) {
    constraintEnergy += constraint->evaluate(this);
  }

  return constraintEnergy +
         0.5 * (bcsN.qpEnergies.sum() + bcsP.qpEnergies.sum() +
                kineticEnergy(params, grid)) -
         0.5 * Erear(grid) + SlaterCoulombEnergy(grid) / 3.0 + bcsN.Epair +
         bcsP.Epair;
}

double IterationData::constraintEnergy(
    const std::vector<std::unique_ptr<Constraint>> &constraints) {
  double constraintEnergy = 0.0;
  for (auto &&constraint : constraints) {
    constraintEnergy += constraint->evaluate(this);
  }
  return constraintEnergy;
}

double IterationData::densityUVPIntegral(const Grid &grid) {
  Eigen::VectorXd vecN = rhoN->array() * UN->array();
  Eigen::VectorXd vecP = rhoP->array() * UP->array();
  Eigen::VectorXd coul = rhoP->array() * UCoul->array();
  Eigen::VectorXd vec = vecN + vecP + coul;

  return Operators::integral(vec, grid);
}

double IterationData::kineticEnergyEff(SkyrmeParameters params,
                                       const Grid &grid) {
  double t0 = params.t0;
  double t3 = params.t3;
  double sigma = params.sigma;

  using Eigen::VectorXd;
  using nuclearConstants::h_bar;
  using nuclearConstants::m;
  using Operators::integral;

  // VectorXd tN = 0.5 * h_bar * h_bar * (*tauN) / (massCorr);
  // VectorXd tP = 0.5 * h_bar * h_bar * (*tauP) / (massCorr);
  VectorXd tN = massN->vector.array() * tauN->array();
  VectorXd tP = massP->vector.array() * tauP->array();
  double kineticEnergy = integral((VectorXd)(tN.matrix() + tP.matrix()), grid);

  return kineticEnergy;
}

double IterationData::kineticEnergy(SkyrmeParameters params, const Grid &grid) {
  double t0 = params.t0;
  double t3 = params.t3;
  double sigma = params.sigma;

  using Eigen::VectorXd;
  using nuclearConstants::h_bar;
  using nuclearConstants::m;
  using Operators::integral;

  VectorXd tN = 0.5 * h_bar * h_bar * (*tauN) / (massCorr);
  VectorXd tP = 0.5 * h_bar * h_bar * (*tauP) / (massCorr);
  double kineticEnergy = integral((VectorXd)(tN.matrix() + tP.matrix()), grid);

  return kineticEnergy;
}

Eigen::MatrixXcd colwiseMatVecMult(const Eigen::MatrixXcd &M,
                                   const Eigen::VectorXd &v) {
  if (M.cols() != v.size())
    throw std::runtime_error("Matrix and vector sizes do not match");

  Eigen::MatrixXcd result(M.rows(), M.cols());
  for (int i = 0; i < M.cols(); i++)
    result.col(i) = M.col(i) * v(i);
  return result;
}

void IterationData::recomputeLagrange(
    const std::pair<Eigen::MatrixXcd, Eigen::VectorXd> &neutronsPair,
    const std::pair<Eigen::MatrixXcd, Eigen::VectorXd> &protonsPair) {

  auto grid = *Grid::getInstance();

  using Wavefunction::kineticDensityLagrange;
  using Wavefunction::soDensityLagrange;

  Eigen::MatrixXcd neutrons, protons;

  neutrons = colwiseMatVecMult(neutronsPair.first, bcsN.v2.array().sqrt());
  protons = colwiseMatVecMult(protonsPair.first, bcsP.v2.array().sqrt());

  *tauN = kineticDensityLagrange(neutrons);
  *tauP = kineticDensityLagrange(protons);

  *JN = soDensityLagrange(neutrons);
  *JP = soDensityLagrange(protons);

  *nablaRhoN = Operators::gradLagrangeNoSpin(*rhoN);
  *nablaRhoP = Operators::gradLagrangeNoSpin(*rhoP);

  *nabla2RhoN = Operators::lapLagrangeNoSpin(*rhoN);
  *nabla2RhoP = Operators::lapLagrangeNoSpin(*rhoP);

  JvecN = std::make_shared<Eigen::MatrixX3d>(Operators::leviCivita(*JN));
  JvecP = std::make_shared<Eigen::MatrixX3d>(Operators::leviCivita(*JP));

  Eigen::VectorXd rho = *rhoN + *rhoP;
  Eigen::MatrixX3d nablaRho = *nablaRhoN + *nablaRhoP;

  *massN =
      EffectiveMass(grid, rho, *rhoN, nablaRho, *nablaRhoN, massCorr, params);
  *massP =
      EffectiveMass(grid, rho, *rhoP, nablaRho, *nablaRhoP, massCorr, params);
}

BCS::BCSResult mixBCS(BCS::BCSResult oldBCS, BCS::BCSResult newBCS,
                      double mix) {
  BCS::BCSResult result;
  result.v2 = oldBCS.v2 * (1.0 - mix) + newBCS.v2 * mix;
  result.u2 = oldBCS.u2 * (1.0 - mix) + newBCS.u2 * mix;
  result.Delta = oldBCS.Delta * (1.0 - mix) + newBCS.Delta * mix;
  result.qpEnergies = oldBCS.qpEnergies * (1.0 - mix) + newBCS.qpEnergies * mix;
  result.lambda = oldBCS.lambda * (1.0 - mix) + newBCS.lambda * mix;
  result.Epair = oldBCS.Epair * (1.0 - mix) + newBCS.Epair * mix;
  return result;
}

void IterationData::updateQuantities(
    const std::pair<Eigen::MatrixXcd, Eigen::VectorXd> &neutronsPair,
    const std::pair<Eigen::MatrixXcd, Eigen::VectorXd> &protonsPair, int iter,
    const std::vector<std::unique_ptr<Constraint>> &constraints) {
  Grid grid = *Grid::getInstance();

  int A = input.getA();
  int Z = input.getZ();

  int N = A - Z;

  double mu = constraints.size() > 0 ? 0.2 : std::min(0.05 + 0.01 * iter, 0.4);
  mu = 0.25;
  // std::cout << "mu: " << mu << std::endl;

  Eigen::VectorXd v2N(neutronsPair.second.size()),
      u2N(neutronsPair.second.size());
  Eigen::VectorXd v2P(protonsPair.second.size()),
      u2P(protonsPair.second.size());
  v2N.setZero();
  v2P.setZero();
  u2N.fill(1.0);
  u2P.fill(1.0);
  for (int i = 0; i < N; ++i) {
    u2N(i) = 0.0;
    v2N(i) = 1.0;
  }
  for (int i = 0; i < Z; ++i) {
    u2P(i) = 0.0;
    v2P(i) = 1.0;
  }
  BCS::BCSResult nullBCSN = {
      .u2 = u2N,
      .v2 = v2N,
      .Delta = Eigen::VectorXd::Zero(N),
      .qpEnergies = v2N.cwiseProduct(neutronsPair.second),
      .lambda = 0.0,
      .Epair = 0.0,
  };
  BCS::BCSResult nullBCSP = {
      .u2 = u2P,
      .v2 = v2P,
      .Delta = Eigen::VectorXd::Zero(Z),
      .qpEnergies = v2P.cwiseProduct(protonsPair.second),
      .lambda = 0.0,
      .Epair = 0.0,
  };
  if (!input.pairing) {
    bcsN = nullBCSN;
    bcsP = nullBCSP;
  } else {
    if (UN == nullptr) {
      Eigen::VectorXd tmpOldDelta(1);
      bcsN = BCS::BCSiter(
          neutronsPair.first, neutronsPair.second, N, input.pairingParameters,
          NucleonType::N, tmpOldDelta,
          0.5 * (neutronsPair.second(N - 1) + neutronsPair.second(N)));
      bcsP = BCS::BCSiter(
          protonsPair.first, protonsPair.second, Z, input.pairingParameters,
          NucleonType::P, tmpOldDelta,
          0.5 * (protonsPair.second(Z - 1) + protonsPair.second(Z)));
    } else {
      bcsN = mixBCS(bcsN,
                    BCS::BCSiter(neutronsPair.first, neutronsPair.second, N,
                                 input.pairingParameters, NucleonType::N,
                                 bcsN.Delta, bcsN.lambda),
                    1.0);
      bcsP = mixBCS(bcsP,
                    BCS::BCSiter(protonsPair.first, protonsPair.second, Z,
                                 input.pairingParameters, NucleonType::P,
                                 bcsP.Delta, bcsP.lambda),
                    1.0);
    }
  }
  std::cout << "N particle number: " << bcsN.v2.sum() << std::endl;
  std::cout << "P particle number: " << bcsP.v2.sum() << std::endl;
  if (std::isnan(bcsN.v2.sum()) || std::isnan(bcsP.v2.sum())) {
    std::cout << "WARNING: NaN particle number" << std::endl;
    bcsN = nullBCSN;
    bcsP = nullBCSP;
  }

  Eigen::MatrixXcd neutrons, protons;

  neutrons = colwiseMatVecMult(neutronsPair.first, bcsN.v2.array().sqrt());
  protons = colwiseMatVecMult(protonsPair.first, bcsP.v2.array().sqrt());

  rhoN = rhoN == nullptr ? std::make_shared<Eigen::VectorXd>(
                               Wavefunction::density(neutrons, grid))
                         : std::make_shared<Eigen::VectorXd>(
                               (*rhoN) * (1.0 - mu) +
                               Wavefunction::density(neutrons, grid) * mu);
  rhoP = rhoP == nullptr ? std::make_shared<Eigen::VectorXd>(
                               Wavefunction::density(protons, grid))
                         : std::make_shared<Eigen::VectorXd>(
                               (*rhoP) * (1.0 - mu) +
                               Wavefunction::density(protons, grid) * mu);
  Eigen::VectorXd rho = (*rhoN) + (*rhoP);
  std::cout << "Integrated density: " << Operators::integral(rho, grid)
            << std::endl;

  nablaRhoN =
      std::make_shared<Eigen::MatrixX3d>(Operators::gradNoSpin(*rhoN, grid));
  nablaRhoP =
      std::make_shared<Eigen::MatrixX3d>(Operators::gradNoSpin(*rhoP, grid));
  Eigen::MatrixX3d nablaRho = (*nablaRhoN) + (*nablaRhoP);

  tauN = tauN == nullptr
             ? std::make_shared<Eigen::VectorXd>(
                   Wavefunction::kineticDensity(neutrons, grid))
             : std::make_shared<Eigen::VectorXd>(
                   (*tauN) * (1.0 - mu) +
                   Wavefunction::kineticDensity(neutrons, grid) * mu);
  tauP = tauP == nullptr
             ? std::make_shared<Eigen::VectorXd>(
                   Wavefunction::kineticDensity(protons, grid))
             : std::make_shared<Eigen::VectorXd>(
                   (*tauP) * (1.0 - mu) +
                   Wavefunction::kineticDensity(protons, grid) * mu);

  nabla2RhoN =
      std::make_shared<Eigen::VectorXd>(Operators::lapNoSpin(*rhoN, grid));
  nabla2RhoP =
      std::make_shared<Eigen::VectorXd>(Operators::lapNoSpin(*rhoP, grid));

  using std::isnan;

  JvecN = std::make_shared<Eigen::MatrixX3d>(
      Eigen::MatrixX3d::Zero(grid.get_total_spatial_points(), 3));
  JvecP = std::make_shared<Eigen::MatrixX3d>(
      Eigen::MatrixX3d::Zero(grid.get_total_spatial_points(), 3));

  JN = JN == nullptr ? std::make_shared<Real2Tensor>(
                           Wavefunction::soDensity(neutrons, grid))
                     : std::make_shared<Real2Tensor>(
                           (*JN) * (1.0 - mu) +
                           Wavefunction::soDensity(neutrons, grid) * mu);
  JP = JP == nullptr ? std::make_shared<Real2Tensor>(
                           Wavefunction::soDensity(protons, grid))
                     : std::make_shared<Real2Tensor>(
                           (*JP) * (1.0 - mu) +
                           Wavefunction::soDensity(protons, grid) * mu);

  JvecN = std::make_shared<Eigen::MatrixX3d>(Operators::leviCivita(*JN));
  JvecP = std::make_shared<Eigen::MatrixX3d>(Operators::leviCivita(*JP));

  divJvecN = divJvecN == nullptr ? std::make_shared<Eigen::VectorXd>(
                                       Wavefunction::divJ(neutrons, grid))
                                 : std::make_shared<Eigen::VectorXd>(
                                       (*divJvecN) * (1.0 - mu) +
                                       Wavefunction::divJ(neutrons, grid) * mu);
  divJvecP =
      divJvecP == nullptr
          ? std::make_shared<Eigen::VectorXd>(Wavefunction::divJ(protons, grid))
          : std::make_shared<Eigen::VectorXd>(
                (*divJvecP) * (1.0 - mu) +
                Wavefunction::divJ(protons, grid) * mu);

  Eigen::VectorXd divJJQN;
  Eigen::VectorXd divJJQP;
  if (input.spinOrbit) {
    divJJQN = *divJvecN + *divJvecN + *divJvecP;
    divJJQP = *divJvecP + *divJvecN + *divJvecP;
  } else {
    divJJQN = Eigen::VectorXd::Zero(grid.get_total_spatial_points());
    divJJQP = Eigen::VectorXd::Zero(grid.get_total_spatial_points());
  }
  std::cout << "Densities updated" << std::endl;

  if (iter != 0 && energyDiff < constraintTol) {
    std::cout
        << "Constraints tolerance reached, updating last converged iteration"
        << std::endl;
  }
  // reset for fields
  mu = 1.0;

  // TODO: dare una sistemata a sta roba
  Eigen::VectorXd tau = *tauN + *tauP;
  Eigen::VectorXd nabla2rho = *nabla2RhoN + *nabla2RhoP;

  EffectiveMass newMassN(grid, rho, *rhoN, nablaRho, *nablaRhoN, massCorr,
                         params);
  EffectiveMass newMassP(grid, rho, *rhoP, nablaRho, *nablaRhoP, massCorr,
                         params);

  Eigen::VectorXd newFieldN = Wavefunction::field(
      rho, *rhoN, tau, *tauN, nabla2rho, *nabla2RhoN, divJJQN, grid, params);

  Eigen::VectorXd newFieldP = Wavefunction::field(
      rho, *rhoP, tau, *tauP, nabla2rho, *nabla2RhoP, divJJQP, grid, params);

  Eigen::VectorXd constraintField =
      Eigen::VectorXd::Zero(grid.get_total_spatial_points());
  // CONSTRAINTS
  for (auto &&constraint : constraints) {
    constraintField += constraint->getField(this);
  }
  // newFieldN += constraintField;
  // newFieldP += constraintField;

  Eigen::VectorXd newFieldCoul;
  if (input.useCoulomb) {
    auto startC = std::chrono::steady_clock::now();

    std::shared_ptr<Eigen::VectorXd> UCDirPtr;

    if (UCoul == nullptr) {
      UCDir = Wavefunction::coulombFieldPoisson(*rhoP, grid, Z, nullptr);
    } else {
      UCDir = Wavefunction::coulombFieldPoisson(
          *rhoP, grid, Z, std::make_shared<Eigen::VectorXd>(UCDir));
    }
    auto endC = std::chrono::steady_clock::now();
    std::cout << "Coulomb Poisson time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(endC -
                                                                       startC)
                     .count()
              << std::endl;

    newFieldCoul = UCDir;
    newFieldCoul += Wavefunction::exchangeCoulombField(*rhoP, grid);
  } else {
    newFieldCoul = Eigen::VectorXd::Zero(grid.get_total_spatial_points());
  }

  if (massN == nullptr) {
    massN = std::make_shared<EffectiveMass>(newMassN);
    massP = std::make_shared<EffectiveMass>(newMassP);
  } else {
    std::cout << "Mass mixing" << std::endl;
    massN->vector = (massN->vector) * (1 - mu) + newMassN.vector * mu;
    massP->vector = (massP->vector) * (1 - mu) + newMassP.vector * mu;
    massN->gradient = (massN->gradient) * (1 - mu) + newMassN.gradient * mu;
    massP->gradient = (massP->gradient) * (1 - mu) + newMassP.gradient * mu;
  }
  if (UP == nullptr) {
    UP = std::make_shared<Eigen::VectorXd>(newFieldP);
    UN = std::make_shared<Eigen::VectorXd>(newFieldN);
  } else {
    *UN = (*UN) * (1 - mu) + newFieldN * mu;
    *UP = (*UP) * (1 - mu) + newFieldP * mu;
  }
  double muConst = 0.8;
  if (UConstr == nullptr) {
    UConstr = std::make_shared<Eigen::VectorXd>(constraintField);
  } else {
    *UConstr = (*UConstr) * (1 - muConst) + constraintField * muConst;
  }
  *UN += *UConstr;
  *UP += *UConstr;

  if (UCoul == nullptr) {
    UCoul = std::make_shared<Eigen::VectorXd>(newFieldCoul);
  } else {
    *UCoul = (*UCoul) * (1 - mu) + newFieldCoul * mu;
  }

  double t1 = params.t1, t2 = params.t2;
  double x1 = params.x1, x2 = params.x2;

  Eigen::MatrixX3d newBN =
      Eigen::MatrixX3d::Zero(grid.get_total_spatial_points(), 3);
  Eigen::MatrixX3d newBP =
      Eigen::MatrixX3d::Zero(grid.get_total_spatial_points(), 3);

  if (input.spinOrbit) {
    newBN += params.W0 * 0.5 * (*nablaRhoN + *nablaRhoN + *nablaRhoP);
    newBP += params.W0 * 0.5 * (*nablaRhoN + *nablaRhoP + *nablaRhoP);
  }
  if (input.useJ) {
    newBN += 0.125 * ((t1 - t2) * (*JvecN) -
                      (t1 * x1 + t2 * x2) * ((*JvecN) + (*JvecP)));
    newBP += 0.125 * ((t1 - t2) * (*JvecP) -
                      (t1 * x1 + t2 * x2) * ((*JvecN) + (*JvecP)));
  }
  if (BN == nullptr) {
    BN = std::make_shared<Eigen::MatrixX3d>(newBN);
    BP = std::make_shared<Eigen::MatrixX3d>(newBP);
  } else {
    *BN = (*BN) * (1 - mu) + newBN * mu;
    *BP = (*BP) * (1 - mu) + newBP * mu;
  }

  std::cout << "Quantities updated" << std::endl << std::endl;
}
