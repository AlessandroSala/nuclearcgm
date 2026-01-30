#include "util/iteration_data.hpp"
#include "EDF.hpp"
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

  double newIntegralEnergy = totalEnergyIntegral() + kineticEnergy();

  double SPE =
      0.5 * (neutronsEigenpair.second.sum() + protonsEigenpair.second.sum());
  double newHFEnergy = HFEnergy(2.0 * SPE, constraints);

  cout << "> Data log" << endl;
  cout << "Integrated EDF: " << newIntegralEnergy
       << ", HF energy: " << newHFEnergy << endl;

  cout << "Coulomb: Direct energy: " << CoulombDirectEnergy(grid)
       << ", exchange energy: " << SlaterCoulombEnergy(grid) << endl;
  std::cout << "X2: " << axis2Exp('x') << ", ";
  std::cout << "Y2: " << axis2Exp('y') << ", ";
  std::cout << "Z2: " << axis2Exp('z') << std::endl;
  std::cout << "RN: " << std::sqrt(neutronRadius()) << ", ";
  std::cout << "RP: " << std::sqrt(protonRadius()) << ", ";
  std::cout << "CR: "
            << std::sqrt(chargeRadius(neutronsEigenpair.first,
                                      protonsEigenpair.first, N, input.getZ()))
            << std::endl;

  auto [beta, gamma] = quadrupoleDeformation();

  std::cout << "Beta: " << beta << ", Gamma: " << gamma * 180.0 / M_PI << " deg"
            << std::endl;
  // std::cout << "Beta w/ physical radius: " << betaRealRadius()
  //           << ", Gamma: " << gamma * 180.0 / M_PI << " deg" << std::endl;
}

IterationData::IterationData(InputParser input) : input(input) {
  interaction = input.interaction;
  int A = input.getA();
  using nuclearConstants::m;
  energyDiff = 1.0;

  massCorr = input.COMCorr ? m * ((double)A) / ((double)(A - 1)) : m;
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

double IterationData::totalEnergyIntegral() {
  auto grid = *Grid::getInstance();
  double energy_C0Rho = C0RhoEnergy();
  double energy_C1Rho = C1RhoEnergy();

  double energy_C0nabla2Rho = C0nabla2RhoEnergy();
  double energy_C1nabla2Rho = C1nabla2RhoEnergy();

  double energy_C0Tau = C0TauEnergy();
  double energy_C1Tau = C1TauEnergy();

  double energy_C0J2 = C0J2Energy();
  double energy_C1J2 = C1J2Energy();

  double energy_C0rhoDivJ = C0rhoDivJEnergy();
  double energy_C1rhoDivJ = C1rhoDivJEnergy();

  double energy_CoulombDirect =
      input.useCoulomb ? CoulombDirectEnergy(grid) : 0.0;

  double energy_Hso = input.spinOrbit ? Hso() : 0.0;

  double energy_Hsg = input.useJ ? Hsg() : 0.0;

  // std::cout << "C0RhoEnergy: " << energy_C0Rho
  //           << ", C1RhoEnergy: " << energy_C1Rho
  //           << ", C0nabla2RhoEnergy: " << energy_C0nabla2Rho
  //           << ", C1nabla2RhoEnergy: " << energy_C1nabla2Rho
  //           << ", C0TauEnergy: " << energy_C0Tau
  //           << ", C1TauEnergy: " << energy_C1Tau
  //           << ", CoulombDirectEnergy: " << energy_CoulombDirect
  //           << ", C0J2Energy: " << energy_C0J2
  //           << ", C1J2Energy: " << energy_C1J2
  //           << ", C0rhoDivJEnergy: " << energy_C0rhoDivJ
  //           << ", C1rhoDivJEnergy: " << energy_C1rhoDivJ;

  return energy_C0Rho + energy_C1Rho + energy_C0nabla2Rho + energy_C1nabla2Rho +
         energy_C0Tau + energy_C1Tau + energy_CoulombDirect + bcsN.Epair +
         bcsP.Epair + SlaterCoulombEnergy(grid) + energy_C0J2 + energy_C1J2 +
         energy_C0rhoDivJ + energy_C1rhoDivJ - HFBResultN.energy -
         HFBResultP.energy;
}

double IterationData::Erear() {
  auto grid = *Grid::getInstance();

  Eigen::VectorXd rho0 = *rhoN + *rhoP;
  Eigen::VectorXd rho1 = *rhoN - *rhoP;
  double sigma = interaction->params.sigma;
  double C0 = interaction->params.C0Drr;
  double C1 = interaction->params.C1Drr;

  Eigen::VectorXd func = -0.5 * sigma *
                         (C0 * rho0.array().pow(sigma + 2) +
                          C1 * rho0.array().pow(sigma) * rho1.array().square());

  using Operators::integral;
  return integral(func, grid);
}

double IterationData::HFEnergy(
    double SPE, const std::vector<std::unique_ptr<Constraint>> &constraints) {
  auto grid = *Grid::getInstance();
  double constraintEnergy = 0.0;
  for (auto &&constraint : constraints) {
    constraintEnergy += constraint->evaluate(this);
  }

  return constraintEnergy +
         0.5 *
             (bcsN.qpEnergies.sum() + bcsP.qpEnergies.sum() + kineticEnergy()) +
         Erear() + SlaterCoulombEnergy(grid) / 3.0 + bcsN.Epair + bcsP.Epair;
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

double IterationData::kineticEnergyEff() {

  using Eigen::VectorXd;
  using nuclearConstants::h_bar;
  using nuclearConstants::m;
  using Operators::integral;

  VectorXd tN = massN->vector.array() * tauN->array();
  VectorXd tP = massP->vector.array() * tauP->array();
  auto grid = *Grid::getInstance();
  double kineticEnergy = integral((VectorXd)(tN.matrix() + tP.matrix()), grid);

  return kineticEnergy;
}

double IterationData::kineticEnergy() {
  using Eigen::VectorXd;
  using nuclearConstants::h_bar;
  using nuclearConstants::m;
  using Operators::integral;

  VectorXd tN = 0.5 * h_bar * h_bar * (*tauN) / (massCorr);
  VectorXd tP = 0.5 * h_bar * h_bar * (*tauP) / (massCorr);
  auto grid = *Grid::getInstance();
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

  *massN = EffectiveMass(grid, rho, *rhoN, nablaRho, *nablaRhoN, massCorr,
                         interaction);
  *massP = EffectiveMass(grid, rho, *rhoP, nablaRho, *nablaRhoP, massCorr,
                         interaction);
}

void IterationData::solvePairingBCS(
    const std::pair<Eigen::MatrixXcd, Eigen::VectorXd> &neutronsPair,
    const std::pair<Eigen::MatrixXcd, Eigen::VectorXd> &protonsPair) {

  int A = input.getA();

  int Z = input.getZ();
  int N = A - Z;

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
      bcsN = BCS::BCSiter(neutronsPair.first, neutronsPair.second, N,
                          input.pairingParameters, NucleonType::N, bcsN.Delta,
                          bcsN.lambda);
      bcsP = BCS::BCSiter(protonsPair.first, protonsPair.second, Z,
                          input.pairingParameters, NucleonType::P, bcsP.Delta,
                          bcsP.lambda);
    }
  }
  std::cout << std::endl;
  std::cout << "N particle number: " << bcsN.v2.sum();
  std::cout << ", P particle number: " << bcsP.v2.sum() << std::endl;
  if (std::isnan(bcsN.v2.sum()) || std::isnan(bcsP.v2.sum())) {
    std::cout << "WARNING: NaN particle number" << std::endl;
    bcsN = nullBCSN;
    bcsP = nullBCSP;
  }
}

Eigen::VectorXcd kappa_field(const Eigen::MatrixXcd &phi,
                             const Eigen::MatrixXcd &kappa_matrix,
                             const Eigen::VectorXd &fermi_factors) {
  Eigen::VectorXcd kappa_r = Eigen::VectorXcd::Zero(phi.rows() / 2);
  Eigen::VectorXcd kappa_complex = Eigen::VectorXcd::Zero(phi.rows() / 2);
  int nStates = phi.cols();
  auto grid = *Grid::getInstance();
  Eigen::VectorXcd tmp_vec = Eigen::VectorXd::Zero(phi.rows() / 2);

  for (int a = 0; a < nStates; ++a)
    for (int b = 0; b < nStates; ++b) {
      tmp_vec.setZero();
      for (int k = 0; k < grid.get_n(); ++k)
        for (int j = 0; j < grid.get_n(); ++j)
          for (int i = 0; i < grid.get_n(); ++i) {
            int idxNS = grid.idxNoSpin(i, j, k);
            for (int s = 0; s < 2; ++s) {
              double sigma = s == 0 ? 1.0 : -1.0;
              int idx = grid.idx(i, j, k, s);
              int idx_other = grid.idx(i, j, k, 1 - s);
              tmp_vec(idxNS) += sigma * phi.col(a)(idx) *
                                (phi.col(b)(idx_other)) * fermi_factors(a) *
                                fermi_factors(b);
            }
          }
      kappa_complex += tmp_vec * kappa_matrix(b, a);
    }
  return kappa_complex;
}

Eigen::MatrixXcd delta_matrix(const Eigen::MatrixXcd &phi,
                              const Eigen::VectorXcd &kappa_r,
                              const Eigen::VectorXd &fermi_factors) {
  auto grid = *Grid::getInstance();
  int nStates = phi.cols();
  Eigen::MatrixXd Delta(nStates, nStates);
  Eigen::MatrixXcd Delta_complex = Eigen::MatrixXd::Zero(nStates, nStates);

  Eigen::VectorXcd tmp_vec =
      Eigen::VectorXcd::Zero(grid.get_total_spatial_points());

  double pairingStrength = -500.0;

  for (int a = 0; a < nStates; ++a)
    for (int b = a; b < nStates; ++b) {
      tmp_vec.setZero();
      for (int k = 0; k < grid.get_n(); ++k)
        for (int j = 0; j < grid.get_n(); ++j)
          for (int i = 0; i < grid.get_n(); ++i) {
            int idxNS = grid.idxNoSpin(i, j, k);
            for (int s = 0; s < 2; ++s) {
              double sigma = s == 0 ? 1.0 : -1.0;
              int idx = grid.idx(i, j, k, s);
              int idx_other = grid.idx(i, j, k, 1 - s);
              tmp_vec(idxNS) += sigma * std::conj(phi.col(a)(idx)) *
                                std::conj(phi.col(b)(idx_other));
            }
          }
      tmp_vec = tmp_vec.array() * kappa_r.array();
      std::complex<double> val = Operators::integralNoSpin(tmp_vec, grid);
      // val *= fermi_factors(a) * fermi_factors(b);
      //   val *= 0.5;
      Delta_complex(a, b) = val;
      Delta_complex(b, a) = -val;
    }
  return -pairingStrength * Delta_complex;
}

HFBResult IterationData::solvePairingHFB(
    const std::pair<Eigen::MatrixXcd, Eigen::VectorXd> &neutronsPair,
    const std::pair<Eigen::MatrixXcd, Eigen::VectorXd> &protonsPair) {

  std::cout << "> Solving pairing HFB" << std::endl;

  const double pairingStrength = -500.0;
  const double toleranceN = 1e-12;
  const int maxBisectionIter = 100;

  const double targetNeutrons = input.getA() - input.getZ();
  const double targetProtons = input.getZ();

  auto solveForSpecies = [&](const Eigen::MatrixXcd &phi,
                             const Eigen::VectorXd &hf_energies, UV &result,
                             double targetNumber, std::string speciesName) {
    const int nStates = hf_energies.size();
    double window = 5.0;
    double smooth = 0.5;
    auto fermi_factor = [&](double e, double lambda) {
      double val = 1.0 / (1.0 + std::exp((e - lambda - window) / smooth));
      val *= 1.0 / (1.0 + std::exp(-(e - lambda + window) / smooth));
      return std::sqrt(val);
    };

    double lambda = result.lambda;

    Eigen::MatrixXcd Delta = Eigen::MatrixXd::Zero(nStates, nStates);

    Eigen::MatrixXcd H_HFB(2 * nStates, 2 * nStates);

    std::cout << "  - Starting HFB loop for " << speciesName
              << " (Target N=" << targetNumber << ")" << std::endl;
    Eigen::MatrixXcd U(nStates, nStates);
    Eigen::MatrixXcd V(nStates, nStates);

    double lambda_min = lambda - 0.7;
    double lambda_max = lambda + 0.7;

    Eigen::VectorXcd kappa_r = result.pairingField;

    Delta = result.Delta;

    Eigen::VectorXd fermi_factors(hf_energies.size());
    for (int i = 0; i < hf_energies.size(); ++i) {
      fermi_factors(i) = fermi_factor(hf_energies(i), lambda);
    }

    kappa_r = kappa_field(phi, result.kappa, fermi_factors);

    double mixDelta = 0.05;
    Delta = delta_matrix(phi, kappa_r, fermi_factors) * mixDelta +
            (1.0 - mixDelta) * Delta;

    std::cout << "Kappa: " << kappa_r.sum() << std::endl;
    std::cout << "Delta: " << Delta.sum() << std::endl;

    double particleDiff = 0.0;

    double b_min = lambda_min;
    double b_max = lambda_max;

    double currentLambda = lambda;

    int bIter;
    for (bIter = 0; bIter < maxBisectionIter; ++bIter) {

      H_HFB.topLeftCorner(nStates, nStates) =
          (hf_energies.array() - currentLambda).matrix().asDiagonal();

      H_HFB.topRightCorner(nStates, nStates) = Delta;

      H_HFB.bottomLeftCorner(nStates, nStates) = -Delta.conjugate();

      H_HFB.bottomRightCorner(nStates, nStates) =
          -1.0 * (hf_energies.array() - currentLambda).matrix().asDiagonal();

      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H_HFB);

      double calculatedNumber = 0.0;
      for (int k = 0; k < nStates; ++k) {
        int qp_index = k + nStates;

        Eigen::VectorXcd W = es.eigenvectors().col(qp_index);

        Eigen::VectorXcd Uk = W.head(nStates);
        Eigen::VectorXcd Vk = W.tail(nStates);

        U.col(k) = Uk;
        V.col(k) = Vk;
        calculatedNumber += std::pow(V.col(k).norm(), 2);
      }

      particleDiff = calculatedNumber - targetNumber;

      if (std::abs(particleDiff) < toleranceN) {
        break;
      }

      if (calculatedNumber > targetNumber) {
        b_max = currentLambda;
      } else {
        b_min = currentLambda;
      }
      currentLambda = (b_min + b_max) / 2.0;
    }
    lambda = currentLambda;
    std::cout << "Bisection converged in " << bIter << " iterations."
              << std::endl;

    double mixKappaMatrix = 1.0;
    Eigen::MatrixXcd kappa = (V.conjugate() * U.transpose()) * mixKappaMatrix +
                             result.kappa * (1.0 - mixKappaMatrix);

    auto pairingEnergy_c = -(Delta.array() * kappa.adjoint().array()).sum();

    std::cout << "  - Converge. Final Lambda: " << lambda << std::endl;
    std::cout << "===> Pairing energy: " << pairingEnergy_c << std::endl;
    double pairingEnergy = pairingEnergy_c.real();

    UV uv = {U, V, lambda, kappa, kappa_r, Delta, pairingEnergy};
    return uv;
  };

  HFBResultN = solveForSpecies(neutronsPair.first, neutronsPair.second,
                               HFBResultN, targetNeutrons, "Neutrons");

  // TODO: both species!!
  // HFBResultP = solveForSpecies(protonsPair.first, protonsPair.second,
  //                             HFBResultP, targetProtons, "Protons");
  HFBResultP = HFBResultN;

  std::cout << "> Pairing HFB completed." << std::endl;
  HFBResult result = {HFBResultN, HFBResultP};
  return result;
}

void IterationData::updateQuantities(
    const std::pair<Eigen::MatrixXcd, Eigen::VectorXd> &neutronsPair,
    const std::pair<Eigen::MatrixXcd, Eigen::VectorXd> &protonsPair, int iter,
    const std::vector<std::unique_ptr<Constraint>> &constraints) {

  std::cout << "> Iteration data update" << std::endl;
  Grid grid = *Grid::getInstance();

  int A = input.getA();
  int Z = input.getZ();

  int N = A - Z;

  // double mu = constraints.size() > 0 ? 0.2 : std::min(0.05 + 0.01 * iter,
  // 0.4);
  double mu = input.densityMix;

  Eigen::MatrixXcd neutrons, protons;

  if (input.pairingType == PairingType::hfb) {
    std::cout << "Iter: " << iter << std::endl;

    int startHFBIter = 3;

    if (iter < startHFBIter) {
      solvePairingBCS(neutronsPair, protonsPair);
      neutrons = colwiseMatVecMult(neutronsPair.first, bcsN.v2.array().sqrt());
      protons = colwiseMatVecMult(protonsPair.first, bcsP.v2.array().sqrt());
    } else {
      if (iter == startHFBIter) {
        HFBResultN.pairingField = Eigen::VectorXcd::Zero(0);
        HFBResultP.pairingField = Eigen::VectorXcd::Zero(0);
        HFBResultN.Delta = Eigen::MatrixXd::Zero(0, 0);
        HFBResultP.Delta = Eigen::MatrixXd::Zero(0, 0);
        std::cout << "Initializing kappa using BCS" << std::endl;
        HFBResultN.kappa = Eigen::MatrixXd::Zero(neutronsPair.second.size(),
                                                 neutronsPair.second.size());
        HFBResultN.lambda = bcsN.lambda;
        double kappa_par = -1.0;

        for (int b = 0; b < neutronsPair.second.size() / 2; ++b) {
          int i = 2 * b;
          int j = 2 * b + 1;

          if (j > i) {
            double uv = std::sqrt(bcsN.u2(i) * bcsN.v2(i));
            HFBResultN.kappa(i, j) = uv;
            HFBResultN.kappa(j, i) = kappa_par * uv;
          }
        }
        HFBResultP.kappa = Eigen::MatrixXd::Zero(protonsPair.second.size(),
                                                 protonsPair.second.size());
        HFBResultP.lambda = bcsP.lambda;
        for (int b = 0; b < protonsPair.second.size() / 2; ++b) {
          int i = 2 * b;
          int j = 2 * b + 1;

          if (j > i) {
            double uv = std::sqrt(bcsP.u2(i) * bcsP.v2(i));
            HFBResultP.kappa(i, j) = uv;
            HFBResultP.kappa(j, i) = kappa_par * uv;
          }
        }
        HFBResultN.pairingField =
            kappa_field(neutronsPair.first, HFBResultN.kappa,
                        Eigen::VectorXd::Ones(neutronsPair.second.size()));
        HFBResultP.pairingField =
            kappa_field(protonsPair.first, HFBResultP.kappa,
                        Eigen::VectorXd::Ones(protonsPair.second.size()));
        HFBResultN.Delta =
            delta_matrix(neutronsPair.first, HFBResultN.pairingField,
                         Eigen::VectorXd::Ones(neutronsPair.second.size()));
        HFBResultP.Delta =
            delta_matrix(protonsPair.first, HFBResultP.pairingField,
                         Eigen::VectorXd::Ones(protonsPair.second.size()));
      }

      auto res = solvePairingHFB(neutronsPair, protonsPair);
      bcsN.Epair = 0.0;
      bcsP.Epair = 0.0;

      HFBResultN = res.uv_n;
      HFBResultP = res.uv_p;
      Eigen::MatrixXcd rho_n_hfb =
          HFBResultN.V.conjugate() * HFBResultN.V.transpose();

      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es_n(rho_n_hfb);
      Eigen::VectorXd v2_n = es_n.eigenvalues();
      std::cout << "Eigenvalues of rho_n_hfb: " << v2_n.transpose()
                << std::endl;
      for (int i = 0; i < v2_n.size(); i++) {
        if (v2_n(i) < 1e-12) {
          v2_n(i) = 0.0;
        }
      }
      Eigen::MatrixXcd W_n = es_n.eigenvectors();
      neutrons = neutronsPair.first;

      neutrons = neutronsPair.first * (W_n * v2_n.cwiseSqrt().asDiagonal());

      Eigen::MatrixXcd rho_p_hfb =
          HFBResultP.V.conjugate() * HFBResultP.V.transpose();

      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es_p(rho_p_hfb);
      Eigen::VectorXd v2_p = es_p.eigenvalues();
      Eigen::MatrixXcd W_p = es_p.eigenvectors();

      for (int i = 0; i < v2_p.size(); i++) {
        if (v2_p(i) < 1e-12) {
          v2_p(i) = 0.0;
        }
      }

      protons = protonsPair.first * (W_p * v2_p.cwiseSqrt().asDiagonal());

      std::cout << "Eigenvalues of protons: " << v2_p.transpose() << std::endl;
    }
  } else {
    solvePairingBCS(neutronsPair, protonsPair);
    neutrons = colwiseMatVecMult(neutronsPair.first, bcsN.v2.array().sqrt());
    protons = colwiseMatVecMult(protonsPair.first, bcsP.v2.array().sqrt());
  }

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

  Eigen::VectorXd divJN;
  Eigen::VectorXd divJP;

  if (input.spinOrbit) {
    divJN = *divJvecN;
    divJP = *divJvecP;
  } else {
    divJN = Eigen::VectorXd::Zero(grid.get_total_spatial_points());
    divJP = Eigen::VectorXd::Zero(grid.get_total_spatial_points());
  }
  Eigen::VectorXd divJ = divJN + divJP;
  std::cout << "Densities updated" << std::endl;

  if (iter != 0 && energyDiff < constraintTol) {
    std::cout
        << "Constraints tolerance reached, updating last converged iteration"
        << std::endl;
  }
  // reset for fields, fields mixing is supported but not done
  mu = 1.0;

  Eigen::VectorXd tau = *tauN + *tauP;
  Eigen::VectorXd nabla2rho = *nabla2RhoN + *nabla2RhoP;

  EffectiveMass newMassN(grid, rho, *rhoN, nablaRho, *nablaRhoN, massCorr,
                         interaction);
  EffectiveMass newMassP(grid, rho, *rhoP, nablaRho, *nablaRhoP, massCorr,
                         interaction);

  Eigen::VectorXd rho1sq = (*rhoN - *rhoP).array().pow(2);

  Eigen::VectorXd newFieldN =
      Wavefunction::field(rho, *rhoN, tau, *tauN, nabla2rho, *nabla2RhoN, divJ,
                          divJN, grid, interaction);

  Eigen::VectorXd newFieldP =
      Wavefunction::field(rho, *rhoP, tau, *tauP, nabla2rho, *nabla2RhoP, divJ,
                          divJP, grid, interaction);

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

    newFieldCoul = UCDir;
    newFieldCoul += Wavefunction::exchangeCoulombField(*rhoP, grid);
  } else {
    newFieldCoul = Eigen::VectorXd::Zero(grid.get_total_spatial_points());
  }

  if (massN == nullptr) {
    massN = std::make_shared<EffectiveMass>(newMassN);
    massP = std::make_shared<EffectiveMass>(newMassP);
  } else {
    // std::cout << "Mass mixing" << std::endl;
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

  Eigen::MatrixX3d newBN =
      Eigen::MatrixX3d::Zero(grid.get_total_spatial_points(), 3);
  Eigen::MatrixX3d newBP =
      Eigen::MatrixX3d::Zero(grid.get_total_spatial_points(), 3);

  if (input.spinOrbit) {
    newBN += -(interaction->params.C0nJ - interaction->params.C1nJ) *
             (*nablaRhoN + *nablaRhoP);
    newBN += -2 * interaction->params.C1nJ * (*nablaRhoN);
    newBP += -(interaction->params.C0nJ - interaction->params.C1nJ) *
             (*nablaRhoN + *nablaRhoP);
    newBP += -2 * interaction->params.C1nJ * (*nablaRhoP);
  }
  if (input.useJ) {
    newBN += 2 * (interaction->params.C0J2 - interaction->params.C1J2) *
             (*JvecN + *JvecP);
    newBN += 4 * interaction->params.C1J2 * (*JvecN);

    newBP += 2 * (interaction->params.C0J2 - interaction->params.C1J2) *
             (*JvecP + *JvecN);
    newBP += 4 * interaction->params.C1J2 * (*JvecP);
  }
  if (BN == nullptr) {
    BN = std::make_shared<Eigen::MatrixX3d>(newBN);
    BP = std::make_shared<Eigen::MatrixX3d>(newBP);
  } else {
    *BN = (*BN) * (1 - mu) + newBN * mu;
    *BP = (*BP) * (1 - mu) + newBP * mu;
  }

  std::cout << "> Iteration data updated" << std::endl << std::endl;
}
