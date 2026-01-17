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
         energy_C0rhoDivJ + energy_C1rhoDivJ;
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

HFBResult IterationData::solvePairingHFB(
    const std::pair<Eigen::MatrixXcd, Eigen::VectorXd> &neutronsPair,
    const std::pair<Eigen::MatrixXcd, Eigen::VectorXd> &protonsPair) {

  std::cout << "> Solving pairing HFB" << std::endl;

  const double pairingStrength = -300.0;
  const double toleranceN = 1e-8;
  const double toleranceL = 1e-4;
  const int maxHFBIter = 100;
  const int maxBisectionIter = 100;

  const double targetNeutrons = input.getA() - input.getZ();
  const double targetProtons = input.getZ();

  auto solveForSpecies = [&](const Eigen::MatrixXcd &phi,
                             const Eigen::VectorXd &hf_energies,
                             std::shared_ptr<Eigen::VectorXcd> &kappa_old,
                             Eigen::MatrixXcd &kappa_matrix,
                             double targetNumber, std::string speciesName,
                             double initLambda) {
    const int nStates = hf_energies.size();
    const int gridSize = phi.rows();

    double lambda = initLambda;
    double lambda_min = -100.0;
    double lambda_max = 50.0;

    Eigen::MatrixXcd Delta = Eigen::MatrixXcd::Zero(nStates, nStates);

    Eigen::MatrixXcd H_HFB(2 * nStates, 2 * nStates);

    std::cout << "  - Starting HFB loop for " << speciesName
              << " (Target N=" << targetNumber << ")" << std::endl;
    Eigen::MatrixXcd U(nStates, nStates);
    Eigen::MatrixXcd V(nStates, nStates);

    for (int iter = 0; iter < maxHFBIter; ++iter) {

      Eigen::VectorXcd kappa_r = Eigen::VectorXcd::Zero(gridSize);
      for (int i = 0; i < nStates; ++i)
        for (int j = 0; j < nStates; ++j)
          kappa_r.array() +=
              phi.col(i).array() * phi.col(j).array() * kappa_matrix(i, j);

      double mu = 0.25;
      if (kappa_old != nullptr) {
        if ((kappa_r - *kappa_old).norm() < toleranceL) {
          std::cout << "Converged" << std::endl;
          break;
        }
        kappa_r = kappa_r * (1.0 - mu) + (*kappa_old) * mu;
        *kappa_old = kappa_r;
      } else {
        std::cout << "New kappa" << std::endl;
        kappa_old = std::make_shared<Eigen::VectorXcd>(kappa_r);
      }

      Eigen::VectorXcd delta_field_spatial = pairingStrength * kappa_r;

      for (int i = 0; i < nStates; ++i) {
        for (int j = i; j < nStates; ++j) {

          Eigen::VectorXcd integrand = phi.col(i).conjugate().array() *
                                       phi.col(j).conjugate().array() *
                                       delta_field_spatial.array();

          auto grid = *Grid::getInstance();
          std::complex<double> val = Operators::integral(integrand, grid);
          Delta(i, j) = val;
          Delta(j, i) = val;
        }
      }

      double particleDiff = 0.0;
      double E_fermi = lambda;

      double b_min = lambda_min;
      double b_max = lambda_max;

      for (int bIter = 0; bIter < maxBisectionIter; ++bIter) {
        double currentLambda = (b_min + b_max) / 2.0;

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
          calculatedNumber += Vk.squaredNorm();
        }

        particleDiff = calculatedNumber - targetNumber;

        if (std::abs(particleDiff) < toleranceN) {
          lambda = currentLambda;
          break;
        }

        if (calculatedNumber > targetNumber) {
          b_max = currentLambda;
        } else {
          b_min = currentLambda;
        }
      }

      Eigen::MatrixXcd kappa_new = V.conjugate() * U.transpose();

      kappa_matrix = 0.5 * kappa_matrix + 0.5 * kappa_new;

      if (iter % 5 == 0) {
        std::cout << "    Iter " << iter << ": Lambda = " << lambda
                  << ", dN = " << particleDiff << std::endl;
      }
    }
    std::cout << "  - Converged. Final Lambda: " << lambda << std::endl;

    double pairingEnergy =
        0.5 * (Delta.array() * kappa_matrix.conjugate().array()).sum().real();
    std::cout << "Pairing energy: " << pairingEnergy << std::endl;
    UV uv = {U, V, lambda};
    return uv;
  };

  auto uv_n =
      solveForSpecies(neutronsPair.first, neutronsPair.second, kappa_n,
                      kappa_matrix_n, targetNeutrons, "Neutrons", oldLambdaN);
  oldLambdaN = uv_n.lambda;

  // TODO: both species!!
  // auto uv_p = solveForSpecies(protonsPair.first, protonsPair.second,
  // kappa_p,
  //                             targetProtons, "Protons");

  std::cout << "> Pairing HFB completed." << std::endl;
  HFBResult result = {uv_n, uv_n};
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

  double mu = constraints.size() > 0 ? 0.2 : std::min(0.05 + 0.01 * iter, 0.4);
  mu = 0.25;

  Eigen::MatrixXcd neutrons, protons;

  if (input.pairingType == PairingType::hfb) {
    std::cout << "Iter: " << iter << std::endl;

    if (iter == 0) {
      solvePairingBCS(neutronsPair, protonsPair);
      neutrons = colwiseMatVecMult(neutronsPair.first, bcsN.v2.array().sqrt());
      protons = colwiseMatVecMult(protonsPair.first, bcsP.v2.array().sqrt());
    } else {
      if (iter == 1) {
        kappa_matrix_n = Eigen::MatrixXcd::Zero(N, N);
        oldLambdaN = neutronsPair.second(N - 1);

        for (int b = 0; b < N / 2; ++b) {
          int i = 2 * b;
          int j = 2 * b + 1;

          if (j > i) {
            double uv = std::sqrt(bcsN.u2(i) * bcsN.v2(i));
            kappa_matrix_n(i, j) = uv;
            kappa_matrix_n(j, i) = -uv;
          }
        }
        kappa_matrix_p = Eigen::MatrixXcd::Zero(Z, Z);
        oldLambdaP = protonsPair.second(Z - 1);
        for (int b = 0; b < Z / 2; ++b) {
          int i = 2 * b;
          int j = 2 * b + 1;

          if (j > i) {
            double uv = std::sqrt(bcsP.u2(i) * bcsP.v2(i));
            kappa_matrix_p(i, j) = -uv;
            kappa_matrix_p(j, i) = uv;
          }
        }
      }

      auto res = solvePairingHFB(neutronsPair, protonsPair);
      Eigen::MatrixXcd rho_n_hfb =
          res.uv_n.V.conjugate() * res.uv_n.V.transpose();

      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es_n(rho_n_hfb);
      Eigen::VectorXd v2_n = es_n.eigenvalues();
      Eigen::MatrixXcd W_n = es_n.eigenvectors();

      neutrons = (neutronsPair.first * W_n) * v2_n.cwiseSqrt().asDiagonal();

      Eigen::MatrixXcd rho_p_hfb =
          res.uv_n.V.conjugate() * res.uv_n.V.transpose();

      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es_p(rho_p_hfb);
      Eigen::VectorXd v2_p = es_p.eigenvalues();
      Eigen::MatrixXcd W_p = es_p.eigenvectors();

      protons = (protonsPair.first * W_p) * v2_p.cwiseSqrt().asDiagonal();
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
  // reset for fields
  mu = 1.0;

  // TODO: dare una sistemata a sta roba
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
