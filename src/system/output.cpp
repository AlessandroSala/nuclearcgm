#include "util/output.hpp"
#include "constraint.hpp"
#include "operators/common_operators.hpp"
#include "spherical_harmonics.hpp"
#include "util/iteration_data.hpp"
#include "util/wavefunction.hpp"
#include <fstream>
#include <iostream>
#include <omp.h>

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cassert>
#include <regex>
#include <utility>

bool contains(const std::vector<std::string> &vec, const std::string &str) {
  return std::find(vec.begin(), vec.end(), str) != vec.end();
}
int find_multipoles_number(const std::vector<std::string> &arr) {
  std::regex pattern(R"(multipoles_(\d+))"); // capture the number
  std::smatch match;

  for (const auto &s : arr) {
    if (std::regex_match(s, match, pattern)) {
      return std::stoi(match[1]); // return the captured number
    }
  }
  return -1; // not found
}
void Output::swapAxes(Eigen::VectorXd &rho, int a1, int a2) {
  if (a1 == a2) {
    return;
  }

  Eigen::VectorXd tmp = rho;
  auto grid = Grid::getInstance();
  const int n = grid->get_n();

#pragma omp parallel for collapse(3)
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        int dest_idx = grid->idxNoSpin(i, j, k);

        std::array<int, 3> src_coords = {i, j, k};

        std::swap(src_coords[a1], src_coords[a2]);

        int src_idx =
            grid->idxNoSpin(src_coords[0], src_coords[1], src_coords[2]);

        rho(dest_idx) = tmp(src_idx);
      }
    }
  }
}

double Output::x2(IterationData *data, const Grid &grid, char dir) {

  auto rho = *(data->rhoN) + *(data->rhoP);
  int n = grid.get_n();
  double res = 0.0;
  for (int k = 0; k < grid.get_n(); ++k) {
    for (int j = 0; j < grid.get_n(); ++j) {
      for (int i = 0; i < grid.get_n(); ++i) {
        int idx = grid.idxNoSpin(i, j, k);
        double ii = grid.get_xs()[i];
        double jj = grid.get_ys()[j];
        double kk = grid.get_zs()[k];
        if (dir == 'x') {
          res += ii * ii * rho(idx);
        } else if (dir == 'y') {
          res += jj * jj * rho(idx);
        } else if (dir == 'z') {
          res += kk * kk * rho(idx);
        }
      }
    }
  }
  return res / rho.sum();
}

Output::Output() : Output("output") {}
Output::Output(std::string folder_) : folder(folder_) {
  namespace fs = std::filesystem;
  fs::create_directory("output");
  fs::create_directory(folder);
}
void Output::matrixToFile(std::string path, Eigen::MatrixXd matrix) {
  std::ofstream file(path);
  file << matrix << std::endl;
  file.close();
}

void Output::shellsToFile(
    std::pair<Eigen::MatrixXcd, Eigen::VectorXd> neutronShells,
    std::pair<Eigen::MatrixXcd, Eigen::VectorXd> protonShells,
    IterationData *iterationData, InputParser &input, int iterations,
    std::vector<double> energies, std::vector<double> HFEnergies,
    double cpuTime, char mode,
    const std::vector<std::unique_ptr<Constraint>> &constraints) {

  auto grid = *Grid::getInstance();
  int N = input.getZ();
  int Z = input.getA() - N;
  auto neutrons = neutronShells.first(Eigen::all, Eigen::seq(0, N - 1));
  auto protons = protonShells.first(Eigen::all, Eigen::seq(0, Z - 1));
  Eigen::VectorXd rho = *(iterationData->rhoN) + *(iterationData->rhoP);

  auto fileMode = mode == 'a' ? std::ios_base::app : std::ios_base::out;

  std::string fileName = input.getOutputName();
  std::ofstream file(folder + "/" + fileName + "_log.txt", std::ios_base::app);

  std::cout << "> Writing to " << folder << std::endl;

#pragma omp parallel
  {
#pragma omp single
    file << "Threads used: " << omp_get_num_threads() << std::endl;
  }
  file << "=== BOX ===" << std::endl;
  auto a = grid.get_a();
  file << "Size: [-" << a << ", " << a << "]" << " fm " << std::endl;
  file << "Steps: " << grid.get_n() << std::endl;
  file << "Step size: " << grid.get_h() << " fm" << std::endl;
  file << std::endl;

  file << "=== NUCLEUS ===" << std::endl;
  file << "Z = " << input.getZ() << ", A = " << input.getA() << std::endl;
  file << std::endl;

  //  nlohmann::json ws = input.getWoodsSaxon();
  //  file << "=== Woods-Saxon ===" << std::endl;
  //  file << "V0: " << ws["V0"] << " MeV" << std::endl;
  //  file << "r0: " << ws["r0"] << " fm" << std::endl;
  //  file << "diff: " << ws["alpha"] << std::endl;
  //  file << "Beta: " << "0.0" << std::endl;
  //  file << std::endl;
  auto toYesNo = [](bool value) { return value ? "YES" : "NO"; };

  file << "=== Functional ===" << std::endl;
  file << "Name: " << input.get_json()["functional"] << std::endl;
  file << "Options: " << "J2 terms: " << toYesNo(input.useJ) << " | "
       << "Spin orbit: " << toYesNo(input.spinOrbit) << " | "
       << "Coulomb: " << toYesNo(input.useCoulomb) << std::endl
       << std::endl;
  // file << std::endl << "Parameters" << std::endl;
  //  file << "t0: " << input.skyrme.t0 << ", ";
  //  file << "t1: " << input.skyrme.t1 << ", ";
  //  file << "t2: " << input.skyrme.t2 << ", ";
  //  file << "t3: " << input.skyrme.t3 << ", ";
  //  file << "W0: " << input.skyrme.W0 << ", ";
  //  file << "x0: " << input.skyrme.x0 << ", ";
  //  file << "x1: " << input.skyrme.x1 << ", ";
  //  file << "x2: " << input.skyrme.x2 << ", ";
  //  file << "x3: " << input.skyrme.x3 << ", ";
  //  file << "alpha: " << input.skyrme.sigma << std::endl;
  //  file << std::endl;

  file << "=== Nuclear data ===" << std::endl;
  double x2int = x2(iterationData, grid, 'x');
  double y2int = x2(iterationData, grid, 'y');
  double z2int = x2(iterationData, grid, 'z');
  double x2Sqrt = std::sqrt(x2int);
  double y2Sqrt = std::sqrt(y2int);
  double z2Sqrt = std::sqrt(z2int);
  double r2Sqrt =
      std::sqrt(x2Sqrt * x2Sqrt + y2Sqrt * y2Sqrt + z2Sqrt * z2Sqrt);
  file << "sqrt<x^2>: " << x2Sqrt << " fm";
  file << ", sqrt<y^2>: " << y2Sqrt << " fm";
  file << ", sqrt<z^2>: " << z2Sqrt << " fm" << std::endl;
  file << "RN: " << std::sqrt(iterationData->neutronRadius()) << " fm"
       << std::endl;
  file << "RP: " << std::sqrt(iterationData->protonRadius()) << " fm"
       << std::endl;
  file << "CR: "
       << std::sqrt(iterationData->chargeRadius(neutronShells.first,
                                                protonShells.first, N, Z))
       << " fm" << std::endl;

  file << std::endl;
  // double roundx2 = std::round(x2Sqrt * 100.0) / 100.0;
  // double roundy2 = std::round(y2Sqrt * 100.0) / 100.0;
  // double roundz2 = std::round(z2Sqrt * 100.0) / 100.0;
  // if(roundx2 > roundz2) {
  //     swapAxes(rho, 0, 2);
  // } else if(roundy2 > roundz2) {
  //     swapAxes(rho, 1, 2);
  // }

  auto [beta, gamma] = iterationData->quadrupoleDeformation();
  double betaRealRadius = iterationData->betaRealRadius();
  file << "Beta2: " << beta << std::endl;
  file << "Beta2 computed from physical radius: " << betaRealRadius
       << std::endl;
  file << std::endl;

  double eKin = iterationData->kineticEnergy();
  double skyrmeEnergy = iterationData->totalEnergyIntegral();

  file << "=== Convergence ===" << std::endl;
  file << "Iterations: " << iterations << std::endl;
  file << "Energy tolerance: " << input.getCalculation().hf.energyTol << " MeV"
       << std::endl;
  file << "CPU time: " << cpuTime << " s" << std::endl;
  file << std::endl;

  file << "=== Density Functional contributions ===" << std::endl;
  file << "E (kin): " << eKin << " MeV" << std::endl;
  file << "C0 Rho0: " << iterationData->C0RhoEnergy() << " MeV" << std::endl;
  file << "C1 Rho1: " << iterationData->C1RhoEnergy() << " MeV" << std::endl;
  file << "C0 nabla2Rho0: " << iterationData->C0nabla2RhoEnergy() << " MeV"
       << std::endl;
  file << "C1 nabla2Rho1: " << iterationData->C1nabla2RhoEnergy() << " MeV"
       << std::endl;
  file << "C0 tau0: " << iterationData->C0TauEnergy() << " MeV" << std::endl;
  file << "C1 tau1: " << iterationData->C1TauEnergy() << " MeV" << std::endl;
  file << "C0 div J: " << iterationData->C0rhoDivJEnergy() << " MeV"
       << std::endl;
  file << "C1 div J: " << iterationData->C1rhoDivJEnergy() << " MeV"
       << std::endl;
  file << "C0 J2: " << iterationData->C0J2Energy() << " MeV" << std::endl;
  file << "C1 J2: " << iterationData->C1J2Energy() << " MeV" << std::endl;
  file << "E coulomb direct: " << iterationData->CoulombDirectEnergy(grid)
       << " MeV" << std::endl;
  file << "E pairing neutrons: " << iterationData->EpairN() << " MeV"
       << std::endl;
  file << "E pairing protons: " << iterationData->EpairP() << " MeV"
       << std::endl;
  file << "E coulomb exchange: " << iterationData->SlaterCoulombEnergy(grid)
       << " MeV" << std::endl;

  double totEnInt = eKin + skyrmeEnergy;
  file << "E_INT: " << totEnInt << " MeV" << std::endl;

  double SPE = 0.5 * (neutronShells.second.sum() + protonShells.second.sum());
  file << std::endl;
  file << "=== HF Energy ===" << std::endl;
  file << "E (SPE): " << 2 * SPE << " MeV" << std::endl;

  double eRea = iterationData->Erear();
  auto E_HF = iterationData->HFEnergy(SPE * 2.0, constraints);
  file << "E (REA): " << eRea << " MeV" << std::endl;
  file << "E_HF: " << E_HF << " MeV" << std::endl;
  file << std::endl;
  file << "E_INT/E_HF - 1: " << 100.0 * ((totEnInt / E_HF) - 1.0) << " %"
       << std::endl;

  file << std::endl
       << "=== Lagrange derivatives recomputed data ===" << std::endl;
  iterationData->recomputeLagrange(neutronShells, protonShells);
  double E_INT_Lagrange =
      iterationData->kineticEnergy() + iterationData->totalEnergyIntegral();

  file << "Lagrange recomputed energy: " << E_INT_Lagrange << " MeV"
       << std::endl
       << std::endl;
  std::cout << "Integrated energy: " << totEnInt << " MeV" << std::endl;
  std::cout << "HF energy: " << E_HF << " MeV" << std::endl;
  std::cout << "Lagrange recomputed energy: " << E_INT_Lagrange << " MeV"
            << std::endl;

  file << "=== Neutrons SP states ===" << std::endl;
  Wavefunction::printShellsToFile(neutronShells, grid, file);
  file << std::endl;
  file << "=== Protons SP states ===" << std::endl;
  Wavefunction::printShellsToFile(protonShells, grid, file);
  file << std::endl;

  using namespace SphericalHarmonics;
  file << "=== Multipole moments ===" << std::endl;

  int l_max = find_multipoles_number(input.log);
  if (l_max > 0) {
    for (int l = 0; l <= std::min(l_max, 5); ++l) {
      file << "l: " << l << std::endl;
      for (int m = -l; m <= l; ++m) {
        file << l << ", " << m << ": " << Q(l, m, rho) << std::endl;
      }
      file << std::endl;
    }
  }

  std::string outputFilesDir = folder + "/" + fileName + "_output/";
  std::filesystem::create_directory(outputFilesDir);

  std::string dataOutputDir = outputFilesDir + "data/";
  std::filesystem::create_directory(dataOutputDir);

  if (contains(input.log, "tot_energies")) {
    std::ofstream totEnFile(dataOutputDir + "int_energies.csv", fileMode);
    for (int i = 0; i < energies.size(); ++i) {
      double e = energies[i];
      totEnFile << std::setprecision(16) << e << std::endl;
    }
    totEnFile.close();
  }
  if (contains(input.log, "hf_energies")) {
    std::ofstream hfEnFile(dataOutputDir + "hf_energies.csv", fileMode);
    for (int i = 0; i < HFEnergies.size(); ++i) {
      double e = HFEnergies[i];
      hfEnFile << std::setprecision(16) << e << std::endl;
    }
    hfEnFile.close();
  }

  std::string dataOutputDirectory = outputFilesDir + "fields/";
  std::filesystem::create_directory(dataOutputDirectory);

  matrixToFile(dataOutputDirectory + "density.csv", rho);
  matrixToFile(dataOutputDirectory + "density_n.csv", *iterationData->rhoN);
  matrixToFile(dataOutputDirectory + "density_p.csv", *iterationData->rhoP);
  matrixToFile(dataOutputDirectory + "kinetic_n.csv", *iterationData->tauN);
  matrixToFile(dataOutputDirectory + "kinetic_p.csv", *iterationData->tauP);
  matrixToFile(dataOutputDirectory + "Field_n.csv", *iterationData->UN);
  matrixToFile(dataOutputDirectory + "Field_p.csv", *iterationData->UP);
  Eigen::VectorXd rhoN = *iterationData->rhoN;
  Eigen::VectorXd rhoP = *iterationData->rhoP;
  Eigen::VectorXd tauN = *iterationData->tauN;
  Eigen::VectorXd tauP = *iterationData->tauP;
  Eigen::VectorXd tau = *iterationData->tauN + *iterationData->tauP;
  Eigen::MatrixXd nablaRho =
      *iterationData->nablaRhoN + *iterationData->nablaRhoP;
  Eigen::VectorXd nablaRhoMod = Operators::mod2(nablaRho);
  Eigen::VectorXd nablaRhoNMod2 = Operators::mod2(*iterationData->nablaRhoN);
  Eigen::VectorXd nablaRhoPMod2 = Operators::mod2(*iterationData->nablaRhoP);
  matrixToFile(dataOutputDirectory + "nabla2rho_n.csv",
               *iterationData->nablaRhoN);
  matrixToFile(dataOutputDirectory + "nabla2rho_p.csv",
               *iterationData->nablaRhoP);
  using std::pow;
  Eigen::VectorXd TF = 3.0 / 5.0 * pow((3 * M_PI * M_PI), 2.0 / 3.0) *
                       rho.array().pow(5.0 / 3.0);
  Eigen::VectorXd TFN = 3.0 / 5.0 * pow((3 * M_PI * M_PI), 2.0 / 3.0) *
                        rhoN.array().pow(5.0 / 3.0);
  Eigen::VectorXd TFP = 3.0 / 5.0 * pow((3 * M_PI * M_PI), 2.0 / 3.0) *
                        rhoP.array().pow(5.0 / 3.0);

  Eigen::VectorXd ones = Eigen::VectorXd::Ones(rho.rows());
  ones.setConstant(1.0);
  Eigen::VectorXd C =
      (ones.array() +
       ((tau.array() * rho.array() - 0.25 * nablaRhoMod.array()) *
        (rho.array() * TF.array() + 1e-12).pow(-1))
           .pow(2))
          .pow(-1);
  Eigen::VectorXd CN =
      (ones.array() +
       ((tauN.array() * rhoN.array() - 0.25 * nablaRhoNMod2.array()) *
        (rhoN.array() * TFN.array() + 1e-12).pow(-1))
           .pow(2))
          .pow(-1);
  Eigen::VectorXd CP =
      (ones.array() +
       ((tauP.array() * rhoP.array() - 0.25 * nablaRhoPMod2.array()) *
        (rhoP.array() * TFP.array() + 1e-12).pow(-1))
           .pow(2))
          .pow(-1);

  matrixToFile(dataOutputDirectory + "C_p.csv", CP);
  matrixToFile(dataOutputDirectory + "C.csv", C);
  matrixToFile(dataOutputDirectory + "C_n.csv", CN);

  double constraintsEnergy = 0.0;
  for (auto &&constraint : constraints) {
    constraintsEnergy += constraint->evaluate(iterationData);
  }
  // JSON output
  nlohmann::json jsonEntry = {
      {"beta", beta},
      {"betaReal", betaRealRadius},
      {"EintLag", E_INT_Lagrange},
      {"Eint", totEnInt},
      {"EpairN", iterationData->EpairN()},
      {"EpairP", iterationData->EpairP()},
      {"gamma", gamma * 180.0 / M_PI},
      {"a", a},
      {"iter", iterations},
      {"constraintsEnergy", constraintsEnergy},
      {"step", grid.get_h()},
  };
  nlohmann::json jsonOutput;
  std::ifstream jsonReader(outputFilesDir + "data/data.json");
  if (jsonReader.good()) {
    try {
      jsonReader >> jsonOutput;
    } catch (const std::exception &e) {
      std::cout << "Error parsing JSON" << std::endl;
      jsonOutput = nlohmann::json::object();
    }
  } else {
    jsonOutput = nlohmann::json::object();
  }

  if (!jsonOutput.contains("data") || !jsonOutput["data"].is_array()) {
    jsonOutput["data"] = nlohmann::json::array();
  }
  jsonOutput["data"].push_back(jsonEntry);

  auto jsonOutputFile = std::ofstream(outputFilesDir + "data/data.json");
  jsonOutputFile << jsonOutput.dump(4);
  jsonOutputFile.close();

  file.close();
}
