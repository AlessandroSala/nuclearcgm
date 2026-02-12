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
  int N = input.getA() - input.getZ();
  int Z = input.getZ();

  double newIntegralEnergy = totalEnergyIntegral() + kineticEnergy();

  double SPE =
      0.5 * (neutronsEigenpair.second.sum() + protonsEigenpair.second.sum());
  double newHFEnergy = HFEnergy(2.0 * SPE, constraints);

  cout << endl;
  cout << "Integrated EDF: " << newIntegralEnergy
       << ", HF energy: " << newHFEnergy << endl;

  // cout << "Coulomb: Direct energy: " << CoulombDirectEnergy(grid)
  //      << ", exchange energy: " << SlaterCoulombEnergy(grid) << endl;
  cout << "Pairing energy: Neutrons: " << EpairN() << ", Protons: " << EpairP()
       << endl;
  double fermiN = neutronsEigenpair.second(N - 1);
  double fermiP = protonsEigenpair.second(Z - 1);
  if (input.pairingType == PairingType::hfb) {
    if (input.pairingN.active)
      fermiN = HFBResultN.lambda;
    if (input.pairingP.active)
      fermiP = HFBResultP.lambda;
  } else if (input.pairingType == PairingType::bcs) {
    if (input.pairingN.active)
      fermiN = bcsN.lambda;
    if (input.pairingP.active)
      fermiP = bcsP.lambda;
  }

  cout << "Fermi energy: Neutrons: " << fermiN << ", Protons: " << fermiP
       << endl;

  std::cout << "X2: " << axis2Exp('x') << ", ";
  std::cout << "Y2: " << axis2Exp('y') << ", ";
  std::cout << "Z2: " << axis2Exp('z') << std::endl;
  // std::cout << "RN: " << std::sqrt(neutronRadius()) << ", ";
  // std::cout << "RP: " << std::sqrt(protonRadius()) << ", ";
  // std::cout << "CR: "
  //           << std::sqrt(chargeRadius(neutronsEigenpair.first,
  //                                     protonsEigenpair.first, N,
  //                                     input.getZ()))
  //           << std::endl;

  auto [beta, gamma] = quadrupoleDeformation();

  std::cout << "Beta2: " << beta << ", Gamma: " << gamma * 180.0 / M_PI << "°" 
            << std::endl;
  std::cout << std::endl;
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
         energy_C0Tau + energy_C1Tau + energy_CoulombDirect + EpairN() +
         EpairP() + SlaterCoulombEnergy(grid) + energy_C0J2 + energy_C1J2 +
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
         Erear() + SlaterCoulombEnergy(grid) / 3.0 + EpairN() + EpairP();
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

Eigen::VectorXd nullOcc(int n) {
  Eigen::VectorXd occ(n);
  occ.setZero();
  for (int i = 0; i < n; i++) {
    occ(i) = 1.0;
  }
  return occ;
}

Eigen::MatrixXcd IterationData::protonsFromBCS(const Eigen::MatrixXcd &phi) {
  return phi * bcsP.v2.array().sqrt().matrix().asDiagonal();
}
Eigen::MatrixXcd IterationData::neutronsFromBCS(const Eigen::MatrixXcd &phi) {
  return phi * bcsN.v2.array().sqrt().matrix().asDiagonal();
}

Eigen::MatrixXcd
IterationData::protonsFromPairing(const Eigen::MatrixXcd &phi) {
  int Z = input.getZ();

  if (!input.pairingP.active) {
    return phi * nullOcc(Z).asDiagonal();
  }

  if (input.pairingType == PairingType::hfb) {

    Eigen::MatrixXcd rho = HFBResultP.V.conjugate() * HFBResultP.V.transpose();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es_n(rho);
    Eigen::VectorXd v2_n = es_n.eigenvalues();
    for (int i = 0; i < v2_n.size(); i++) {
      if (v2_n(i) < 1e-12) {
        v2_n(i) = 0.0;
      }
    }
    Eigen::MatrixXcd W_n = es_n.eigenvectors();

    return phi * (W_n * v2_n.cwiseSqrt().asDiagonal());
  } else if (input.pairingType == PairingType::bcs) {
    return protonsFromBCS(phi);
  } else {
    return phi * nullOcc(Z).asDiagonal();
  }
}

Eigen::MatrixXcd
IterationData::neutronsFromPairing(const Eigen::MatrixXcd &phi) {
  int N = input.getA() - input.getZ();

  if (!input.pairingN.active) {
    return phi * nullOcc(N).asDiagonal();
  }

  if (input.pairingType == PairingType::hfb) {
    Eigen::MatrixXcd rho = HFBResultN.V.conjugate() * HFBResultN.V.transpose();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es_n(rho);
    Eigen::VectorXd v2_n = es_n.eigenvalues();
    for (int i = 0; i < v2_n.size(); i++) {
      if (v2_n(i) < 1e-12) {
        v2_n(i) = 0.0;
      }
    }
    Eigen::MatrixXcd W_n = es_n.eigenvectors();

    return phi * (W_n * v2_n.cwiseSqrt().asDiagonal());
  } else if (input.pairingType == PairingType::bcs) {
    return neutronsFromBCS(phi);
  } else {
    return phi * nullOcc(N).asDiagonal();
  }
}

void IterationData::recomputeLagrange(
    const std::pair<Eigen::MatrixXcd, Eigen::VectorXd> &neutronsPair,
    const std::pair<Eigen::MatrixXcd, Eigen::VectorXd> &protonsPair) {

  auto grid = *Grid::getInstance();

  using Wavefunction::kineticDensityLagrange;
  using Wavefunction::soDensityLagrange;

  Eigen::MatrixXcd neutrons, protons;

  neutrons = neutronsFromPairing(neutronsPair.first);
  protons = protonsFromPairing(protonsPair.first);

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
      if (input.pairingN.active)
        bcsN = BCS::BCSiter(
            neutronsPair.first, neutronsPair.second, N, input.pairingN,
            NucleonType::N, tmpOldDelta,
            0.5 * (neutronsPair.second(N - 1) + neutronsPair.second(N)));
      else
        bcsN = nullBCSN;
      if (input.pairingP.active)
        bcsP = BCS::BCSiter(
            protonsPair.first, protonsPair.second, Z, input.pairingP,
            NucleonType::P, tmpOldDelta,
            0.5 * (protonsPair.second(Z - 1) + protonsPair.second(Z)));
      else
        bcsP = nullBCSP;
    } else {
      if (input.pairingN.active)
        bcsN = BCS::BCSiter(neutronsPair.first, neutronsPair.second, N,
                            input.pairingN, NucleonType::N, bcsN.Delta,
                            bcsN.lambda);
      else
        bcsN = nullBCSN;
      if (input.pairingP.active)
        bcsP = BCS::BCSiter(protonsPair.first, protonsPair.second, Z,
                            input.pairingP, NucleonType::P, bcsP.Delta,
                            bcsP.lambda);
      else
        bcsP = nullBCSP;
    }
  }
  if (std::isnan(bcsN.v2.sum()) || std::isnan(bcsP.v2.sum())) {
    std::cout << "WARNING: NaN particle number" << std::endl;
    bcsN = nullBCSN;
    bcsP = nullBCSP;
  }
}

Eigen::VectorXd kappa_field(const Eigen::MatrixXcd &phi,
                            const Eigen::MatrixXd &kappa_matrix,
                            const Eigen::VectorXd &fermi_factors, int start,
                            int end) {
  Eigen::VectorXcd kappa_complex = Eigen::VectorXcd::Zero(phi.rows() / 2);
  auto grid = *Grid::getInstance();

#pragma omp parallel
  {
    Eigen::VectorXcd thread_tmp_vec =
        Eigen::VectorXcd::Zero(grid.get_total_spatial_points());
#pragma omp for collapse(2) schedule(dynamic)
    for (int a = start; a < end + 1; ++a)
      for (int b = start; b < end + 1; ++b) {
        for (int k = 0; k < grid.get_n(); ++k)
          for (int j = 0; j < grid.get_n(); ++j)
            for (int i = 0; i < grid.get_n(); ++i) {
              int idxNS = grid.idxNoSpin(i, j, k);
              for (int s = 0; s < 2; ++s) {
                double sigma = s == 0 ? 1.0 : -1.0;
                int idx = grid.idx(i, j, k, s);
                int idx_other = grid.idx(i, j, k, 1 - s);
                thread_tmp_vec(idxNS) +=
                    -(sigma * (phi.col(b)(idx_other)) * (phi.col(a)(idx)) *
                      fermi_factors(a) * fermi_factors(b)) *
                    kappa_matrix(a, b);
              }
            }
      }
#pragma omp critical
    kappa_complex += thread_tmp_vec;
  }

  return kappa_complex.real();
}

Eigen::MatrixXd delta_matrix(const Eigen::MatrixXcd &phi,
                             const Eigen::VectorXd &kappa_r,
                             const Eigen::VectorXd &fermi_factors,
                             PairingParameters params, int start, int end) {
  auto grid = *Grid::getInstance();
  int nStates = end - start + 1;
  Eigen::MatrixXcd Delta_complex = Eigen::MatrixXd::Zero(nStates, nStates);

#pragma omp parallel for collapse(2) schedule(dynamic)
  for (int a = start; a < end + 1; ++a)
    for (int b = start; b < end + 1; ++b) {
      if (b < a)
        continue;
      Eigen::VectorXcd thread_tmp_vec =
          Eigen::VectorXcd::Zero(grid.get_total_spatial_points());
      for (int k = 0; k < grid.get_n(); ++k)
        for (int j = 0; j < grid.get_n(); ++j)
          for (int i = 0; i < grid.get_n(); ++i) {
            int idxNS = grid.idxNoSpin(i, j, k);
            for (int s = 0; s < 2; ++s) {
              double sigma = s == 0 ? 1.0 : -1.0;
              int idx = grid.idx(i, j, k, s);
              int idx_other = grid.idx(i, j, k, 1 - s);
              thread_tmp_vec(idxNS) +=
                  sigma * (phi.col(a)(idx) * (phi.col(b)(idx_other))) *
                  fermi_factors(a) * fermi_factors(b);
            }
          }
      thread_tmp_vec = thread_tmp_vec.array().conjugate() * kappa_r.array();
      std::complex<double> val =
          Operators::integralNoSpin(thread_tmp_vec, grid);
      Delta_complex(a - start, b - start) = val;
      Delta_complex(b - start, a - start) = -val;
    }
  return 0.5 * params.V0 * Delta_complex.real();
}

Eigen::VectorXcd get_mixed_kappa(const Eigen::VectorXcd &kappa_old,
                                 const Eigen::VectorXcd &kappa_new,
                                 double alpha) {
  std::complex<double> overlap = kappa_old.dot(kappa_new);
  std::complex<double> phase_alignment = 1.0;
  double magnitude = std::abs(overlap);

  if (magnitude > 1e-18) {
    phase_alignment = overlap / magnitude;
  }

  return (1.0 - alpha) * kappa_old +
         alpha * (kappa_new * std::conj(phase_alignment));
}

double findLambdaBisection(double &lambdaMin, double &lambdaMax, double dN,
                           double currentLambda) {
  if (dN > 0) {
    lambdaMax = currentLambda;
  } else {
    lambdaMin = currentLambda;
  }
  return (lambdaMax + lambdaMin) / 2.0;
}

Eigen::Vector2d findLambdaNewton(Eigen::Vector2d &curVec, Eigen::Matrix2d &G,
                                 double f, double g, double f_old,
                                 double g_old) {
  Eigen::Vector2d funcVec, funcVecOld;
  funcVec << f, g;
  funcVecOld << f_old, g_old;

  Eigen::VectorXd newLambda = curVec - 0.2 * G.inverse() * funcVec;
  auto dLambda = newLambda - curVec;

  G = G + ((funcVec - funcVecOld) - G * dLambda) * dLambda.transpose() /
              dLambda.squaredNorm();
  return newLambda;
}

Eigen::MatrixXcd numericalTR(const Eigen::MatrixXcd &phi) {
  Eigen::MatrixXcd phi_tr = Eigen::MatrixXcd::Zero(phi.rows(), phi.cols());
  for (int c = 0; c < phi.cols(); c++) {
    if (c % 2 == 1)
      phi_tr.col(c) = Wavefunction::timeReverse(phi.col(c - 1));
    else
      phi_tr.col(c) = phi.col(c);
  }
  return phi_tr;
}

UV nullHFB(double targetNumber, const Eigen::VectorXd &hf_energies) {
  UV hfb;
  hfb.lambda = hf_energies.coeff(targetNumber - 1);
  hfb.V = Eigen::MatrixXd::Zero(hf_energies.size(), hf_energies.size());
  hfb.U = Eigen::MatrixXd::Zero(hf_energies.size(), hf_energies.size());
  hfb.pairingField = Eigen::VectorXd::Zero(hf_energies.size());
  hfb.energy = 0.0;

  for (int i = 0; i < targetNumber; ++i) {
    hfb.V(i, i) = 1.0;
  }
  return hfb;
}

HFBResult IterationData::solvePairingHFB(
    const std::pair<Eigen::MatrixXcd, Eigen::VectorXd> &neutronsPair,
    const std::pair<Eigen::MatrixXcd, Eigen::VectorXd> &protonsPair) {

  const double toleranceN = 1e-12;
  const int maxBisectionIter = 100;

  const double targetNeutrons = input.getA() - input.getZ();
  const double targetProtons = input.getZ();

  unsigned int solveMethod = 0;

  auto solveForSpecies = [&](const Eigen::MatrixXcd &phi,
                             const Eigen::VectorXd &hf_energies, UV &result,
                             double targetNumber, std::string speciesName,
                             PairingParameters params) {
    double window = params.window;
    double smooth = params.window / 10.0;
    auto fermi_factor = [&](double e, double lambda) {
      double val = 1.0 / (1.0 + std::exp((e - lambda - window) / smooth));
      val *= 1.0 / (1.0 + std::exp(-(e - lambda + window) / smooth));
      return std::sqrt(val);
    };

    double lambda = result.lambda;

    Eigen::VectorXd fermi_factors(hf_energies.size());
    for (int i = 0; i < hf_energies.size(); ++i) {
      fermi_factors(i) = fermi_factor(hf_energies(i), lambda);
    }

    int start = 0, end = hf_energies.size() - 1;
    double pairingThreshold = input.pairingThreshold;

    for (int i = 0; i < hf_energies.size(); i += 2) {
      if (fermi_factors(i) > pairingThreshold) {
        start = i;
        break;
      }
    }
    for (int i = start; i < hf_energies.size(); i += 2) {
      if (fermi_factors(i) < pairingThreshold) {
        end = i;
        break;
      }
    }

    const int nStates = end - start + 1;

    Eigen::MatrixXd H_HFB(2 * nStates, 2 * nStates);
    Eigen::MatrixXd H_DISP(nStates, nStates);

    // std::cout << "  - Starting HFB loop for " << speciesName
    //           << " (Target N=" << targetNumber << ")" << std::endl;
    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(nStates, nStates);
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(nStates, nStates);

    double lambda_min = lambda - 1.7;
    double lambda_max = lambda + 1.7;

    Eigen::MatrixXcd phi_rev = numericalTR(phi);

    double mixKappaField = 0.1;
    Eigen::VectorXd kappa_r =
        kappa_field(phi_rev, result.kappa, fermi_factors, start, end);

    Eigen::MatrixXd Delta =
        delta_matrix(phi_rev, kappa_r, fermi_factors, params, start, end);
    if (Delta.norm() < 1e-3) {
      std::cout << "Pairing collapsed for " << speciesName << std::endl;
      return nullHFB(targetNumber, hf_energies);
    }

    double particleDiff = 0.0;

    double b_min = lambda_min;
    double b_max = lambda_max;

    double currentLambda = lambda;
    double currentLambda2 = solveMethod == 1 ? -0.1 : 0.0;

    using Eigen::MatrixXd;
    using Eigen::Vector2d;
    using Eigen::VectorXd;

    Vector2d lambdaVec(2);
    lambdaVec(0) = currentLambda;
    lambdaVec(1) = currentLambda2;

    int bIter;
    Eigen::Matrix2d G = MatrixXd::Zero(2, 2);

    double oldN = 0.0;

    double calculatedNumber = 0.0;
    double dispN = 0.0;
    double dispNTarget = 5.8;
    double oldDispN = 0.0;
    double oldDDispN = 0.0;
    double oldDN = 0.0;
    // V.conj * V.T
    auto rho = V * V.transpose();
    auto kappaMatrix = V * U.transpose();
    Eigen::VectorXd reducedEnergies = hf_energies.segment(start, nStates);
    for (bIter = 0; bIter < maxBisectionIter; ++bIter) {
      auto H_DISP_OD = 4 * kappaMatrix;
      H_DISP = Eigen::MatrixXd::Identity(nStates, nStates) - rho;
      double alpha = 0.05;

      H_HFB.topLeftCorner(nStates, nStates) =
          (Eigen::MatrixXd)((reducedEnergies.array() - currentLambda)
                                .matrix()
                                .asDiagonal()) -
          2 * currentLambda2 * (1 - alpha) * H_DISP;

      H_HFB.topRightCorner(nStates, nStates) =
          Delta - 2 * alpha * currentLambda2 * H_DISP_OD;

      // conjugate delta
      H_HFB.bottomLeftCorner(nStates, nStates) =
          -Delta + 2 * alpha * currentLambda2 * H_DISP_OD;

      H_HFB.bottomRightCorner(nStates, nStates) =
          -1.0 * (Eigen::MatrixXd)((reducedEnergies.array() - currentLambda)
                                       .matrix()
                                       .asDiagonal()) -
          2 * currentLambda2 * (1 - alpha) * H_DISP;
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H_HFB);

      calculatedNumber = 0.0;
      dispN = 0.0;
      for (int k = 0; k < nStates; ++k) {
        int qp_index = k + nStates;

        Eigen::VectorXd W = es.eigenvectors().col(qp_index);

        Eigen::VectorXd Uk = W.head(nStates);
        Eigen::VectorXd Vk = W.tail(nStates);

        U.col(k) = Uk;
        V.col(k) = Vk;
        calculatedNumber += Vk.squaredNorm();
        dispN += 4 * Uk.squaredNorm() * Vk.squaredNorm();
      }
      calculatedNumber += start;

      particleDiff = calculatedNumber - targetNumber;

      if (std::abs(particleDiff) < toleranceN) {
        break;
      }

      if (solveMethod == 0) {
        currentLambda =
            findLambdaBisection(b_min, b_max, particleDiff, currentLambda);
      } else {
        if (bIter == 0) {
          currentLambda += 0.01;
          lambdaVec(0) = currentLambda;

        } else if (bIter == 1) {
          currentLambda2 += 0.001;
          lambdaVec(1) = currentLambda2;

          G(0, 0) = (particleDiff - oldDN) * 100;
          G(1, 0) = ((dispN - dispNTarget) - oldDDispN) * 100;
        } else {
          if (bIter == 2) {
            G(0, 1) = (particleDiff - oldDN) * 1000;
            G(1, 1) = ((dispN - dispNTarget) - oldDDispN) * 1000;
            std::cout << G.determinant() << std::endl;
          }

          auto res = findLambdaNewton(lambdaVec, G, particleDiff,
                                      dispN - dispNTarget, oldDN, oldDDispN);
          lambdaVec = res;
          currentLambda = res(0);
          currentLambda2 = res(1);
          std::cout << "Lambda2: " << currentLambda2 << std::endl;
        }
      }
      oldN = calculatedNumber;
      oldDN = particleDiff;
      oldDispN = dispN;
      oldDDispN = dispN - dispNTarget;
    }
    lambda = currentLambda;

    // V.conj * U.T
    Eigen::MatrixXd kappaNew = (V * U.transpose());

    // KappaNew.adjoint
    auto pairingEnergy = 0.5 * (Delta * kappaNew.transpose()).trace();

    Eigen::MatrixXd fullU =
        Eigen::MatrixXd::Zero(hf_energies.size(), hf_energies.size());
    Eigen::MatrixXd fullV =
        Eigen::MatrixXd::Zero(hf_energies.size(), hf_energies.size());
    fullV.block(start, start, end - start + 1, end - start + 1) = V;
    fullU.block(start, start, end - start + 1, end - start + 1) = U;
    for (int i = 0; i < start; ++i) {
      fullV(i, i) = 1.0;
    }
    for (int i = end + 1; i < hf_energies.size(); ++i) {
      fullU(i, i) = 1.0;
    }
    kappaNew = fullU * fullV.transpose();

    UV uv = {fullU, fullV, kappaNew, kappa_r, lambda, pairingEnergy};
    return uv;
  };

  if (input.pairingN.active) {
    auto new_HFBResultN =
        solveForSpecies(neutronsPair.first, neutronsPair.second, HFBResultN,
                        targetNeutrons, "Neutrons", input.pairingN);
    if ((std::isnan(new_HFBResultN.energy))) {

      std::cout << "HFB routine for neutrons failed, defaulting to "
                   "previous result"
                << std::endl;
    } else if (std::abs(new_HFBResultN.energy) < 1e-12) {
      input.pairingN.active = false;
    } else {
      HFBResultN = new_HFBResultN;
    }
  }

  if (input.pairingP.active) {
    auto new_HFBResultP =
        solveForSpecies(protonsPair.first, protonsPair.second, HFBResultP,
                        targetProtons, "Protons", input.pairingP);
    if ((std::isnan(new_HFBResultP.energy))) {
      std::cout << "HFB routine for neutrons failed, defaulting to "
                   "previous result"
                << std::endl;
    } else if (std::abs(new_HFBResultP.energy) < 1e-12) {
      input.pairingP.active = false;
    } else {
      HFBResultP = new_HFBResultP;
    }
  }

  HFBResult result = {HFBResultN, HFBResultP};
  return result;
}

void IterationData::updateQuantities(
    const std::pair<Eigen::MatrixXcd, Eigen::VectorXd> &neutronsPair,
    const std::pair<Eigen::MatrixXcd, Eigen::VectorXd> &protonsPair, int iter,
    const std::vector<std::unique_ptr<Constraint>> &constraints) {

  // std::cout << "> Iteration data update" << std::endl;
  Grid grid = *Grid::getInstance();

  int A = input.getA();
  int Z = input.getZ();

  int N = A - Z;

  // double mu = constraints.size() > 0 ? 0.2 : std::min(0.05 + 0.01 * iter,
  // 0.4);
  double mu = input.densityMix;

  Eigen::MatrixXcd neutrons, protons;

  if (input.pairingType == PairingType::hfb) {

    if (iter < input.startHFBIter) {
      std::cout << "> BCS step " << (iter + 1) << "/" << input.startHFBIter
                << " to seed HFB" << std::endl;
      solvePairingBCS(neutronsPair, protonsPair);
      neutrons = neutronsFromBCS(neutronsPair.first);
      protons = protonsFromBCS(protonsPair.first);
    } else {
      if (iter == input.startHFBIter) {
        HFBResultN.pairingField = Eigen::VectorXd::Zero(0);
        HFBResultP.pairingField = Eigen::VectorXd::Zero(0);
        std::cout << "> Initializing pairing tensor using BCS" << std::endl;
        HFBResultN.kappa = Eigen::MatrixXd::Zero(neutronsPair.second.size(),
                                                 neutronsPair.second.size());
        HFBResultN.V = Eigen::MatrixXd::Zero(neutronsPair.second.size(),
                                             neutronsPair.second.size());
        HFBResultN.lambda = bcsN.lambda;
        double kappa_par = -1.0;

        for (int i = 0; i < neutronsPair.second.size() - 1; ++i) {
          int j = i + 1;

          if (i % 2 == 0) {
            double uv = std::sqrt(bcsN.u2(i) * bcsN.v2(i));
            HFBResultN.kappa(i, j) = uv;
            HFBResultN.kappa(j, i) = kappa_par * uv;
          }
        }
        HFBResultP.kappa = Eigen::MatrixXd::Zero(protonsPair.second.size(),
                                                 protonsPair.second.size());
        HFBResultP.lambda = bcsP.lambda;
        for (int i = 0; i < protonsPair.second.size() - 1; ++i) {
          int j = i + 1;

          if (i % 2 == 0) {
            double uv = std::sqrt(bcsP.u2(i) * bcsP.v2(i));
            HFBResultP.kappa(i, j) = uv;
            HFBResultP.kappa(j, i) = kappa_par * uv;
          }
        }
      }

      auto res = solvePairingHFB(neutronsPair, protonsPair);
      std::cout << "> HFB routine completed" << std::endl;

      HFBResultN = res.uv_n;
      HFBResultP = res.uv_p;
      neutrons = neutronsFromPairing(neutronsPair.first);
      protons = protonsFromPairing(protonsPair.first);
    }
  } else {
    solvePairingBCS(neutronsPair, protonsPair);
    neutrons = neutronsFromPairing(neutronsPair.first);
    protons = protonsFromPairing(protonsPair.first);
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
  std::cout << "> Densities updated" << std::endl;

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
  if (iter > 0) {
    for (auto &&constraint : constraints) {
      constraintField += constraint->getField(this);
    }
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
  double muConst = 1.0;
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

  // std::cout << "> Iteration data updated" << std::endl << std::endl;
}
