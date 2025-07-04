#include "util/output.hpp"
#include "operators/integral_operators.hpp"
#include "util/wavefunction.hpp"
#include "json/json.hpp"
#include <fstream>
#include <iostream>

double x2(std::shared_ptr<IterationData> data, const Grid &grid, char dir) {
  double h = grid.get_h();
  double hh = h * h;
  int n = grid.get_n();
  double res = 0.0;
  for (int i = 0; i < grid.get_n(); ++i) {
    for (int j = 0; j < grid.get_n(); ++j) {
      for (int k = 0; k < grid.get_n(); ++k) {
        int idx = grid.idxNoSpin(i, j, k);
        int ii = (n / 2.0 - i);
        int jj = (n / 2.0 - j);
        int kk = (n / 2.0 - k);
        if (dir == 'x') {
          res += ii * ii * ((*(data->rhoN))(idx) + (*(data->rhoP))(idx));
        } else if (dir == 'y') {
          res += jj * jj * ((*(data->rhoN))(idx) + (*(data->rhoP))(idx));
        } else if (dir == 'z') {
          res += kk * kk * ((*(data->rhoN))(idx) + (*(data->rhoP))(idx));
        }
      }
    }
  }
  return hh * res / ((*(data->rhoN)).sum() + (*(data->rhoP)).sum());
}
Output::Output() : Output("output") {}
Output::Output(std::string folder_) : folder(folder_) {}
void Output::matrixToFile(std::string fileName, Eigen::MatrixXd matrix) {
  std::ofstream file(folder + "/" + fileName);
  file << matrix << std::endl;
  file.close();
}
void Output::shellsToFile(
    std::string fileName,
    std::pair<Eigen::MatrixXcd, Eigen::VectorXd> neutronShells,
    std::pair<Eigen::MatrixXcd, Eigen::VectorXd> protonShells,
    std::shared_ptr<IterationData> iterationData, InputParser input,
    int iterations, std::vector<double> energies, double cpuTime,
    const Grid &grid) {

  std::ofstream file(folder + "/" + input.getOutputName() + ".txt");
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

  file << "=== Skyrme ===" << std::endl;
  file << "t0: " << input.skyrme.t0 << std::endl;
  file << "t3: " << input.skyrme.t3 << std::endl;
  file << "alpha: " << input.skyrme.sigma << std::endl;
  file << std::endl;

  file << "=== Nuclear data ===" << std::endl;
  std::cout << "computing data" << std::endl;
  double x2Sqrt = std::sqrt(x2(iterationData, grid, 'x'));
  double y2Sqrt = std::sqrt(x2(iterationData, grid, 'y'));
  double z2Sqrt = std::sqrt(x2(iterationData, grid, 'z'));
  double r2Sqrt =
      std::sqrt(x2Sqrt * x2Sqrt + y2Sqrt * y2Sqrt + z2Sqrt * z2Sqrt);
  file << "sqrt<x^2>: " << x2Sqrt << " fm";
  file << ", sqrt<y^2>: " << y2Sqrt << " fm";
  file << ", sqrt<z^2>: " << z2Sqrt << " fm" << std::endl;
  file << "sqrt<r^2>: " << r2Sqrt << " fm" << std::endl;
  file << std::endl;

  double eKin = iterationData->kineticEnergy(input.skyrme, grid);
  double skyrmeEnergy = iterationData->totalEnergy(input.skyrme, grid);

  file << "=== Convergence ===" << std::endl;
  file << "Iterations: " << iterations << std::endl;
  file << "Energy tolerance: " << input.getCalculation().hf.energyTol << " MeV"
       << std::endl;
  file << "CPU time: " << cpuTime << " s" << std::endl;
  file << "E (t0, t3): " << skyrmeEnergy << " MeV" << std::endl;
  file << "E (kin): " << eKin << " MeV" << std::endl;
  file << "E: " << eKin + skyrmeEnergy << " MeV" << std::endl;

  double SPE = 0.5 * (neutronShells.second.sum() + protonShells.second.sum());
  file << std::endl;
  file << "=== HF Energy + E_rea ===" << std::endl;
  file << "E (SPE): " << SPE << " MeV" << std::endl;
  file << "E (Kin): " << eKin * 0.5 << " MeV" << std::endl;

  double eRea = skyrmeEnergy - 0.5 * iterationData->densityUVPIntegral(grid);
  double E = eRea + eKin * 0.5 + SPE;
  file << "E (REA): " << eRea << " MeV" << std::endl;
  file << "E: " << E << " MeV" << std::endl;
  file << std::endl;

  file << "=== Neutrons ===" << std::endl;
  Wavefunction::printShellsToFile(neutronShells, grid, file);
  file << std::endl;
  file << "=== Protons ===" << std::endl;
  Wavefunction::printShellsToFile(protonShells, grid, file);
  file << std::endl;

  file << "=== Energies ===" << std::endl;
  for (int i = 0; i < energies.size(); ++i) {
    double e = energies[i];
    file << i << ":  " << e << std::endl;
  }

  matrixToFile("density.csv", *(iterationData->rhoN));

  file.close();
}
