#include "util/utilities.hpp"
#include "grid.hpp"
#include "hamiltonian.hpp"
#include "input_parser.hpp"
#include "kinetic/local_kinetic_potential.hpp"
#include "kinetic/non_local_kinetic_potential.hpp"
#include "skyrme/local_coulomb_potential.hpp"
#include "skyrme/skyrme_so.hpp"
#include "skyrme/skyrme_u.hpp"
#include "solver.hpp"
#include <Eigen/Dense>
#include <cassert>
#include <complex>
#include <iostream>
#include <vector>

using namespace Eigen;
using Complex = std::complex<double>;

std::pair<Eigen::MatrixXcd, Eigen::VectorXd>
Utilities::solve(const ComplexSparseMatrix &matrix,
                 const ComplexDenseMatrix &ConjDir, GCGParameters &calc,
                 const Eigen::MatrixXcd &guess) {
  return gcgm_complex_no_B_lock(
      matrix, guess, ConjDir, guess.cols(), 0.0, calc.maxIter, calc.tol,
      calc.steps, (calc.cgTol), false, EigenpairsOrdering::ASCENDING_ENERGIES);
}

std::pair<Eigen::MatrixXcd, Eigen::VectorXd>
Utilities::solve(const ComplexSparseMatrix &matrix,
                 const ComplexDenseMatrix &ConjDir, GCGParameters &calc,
                 const Eigen::MatrixXcd &guess, int nev) {
  return gcgm_complex_no_B_lock(matrix, guess, ConjDir, nev, 0.0, calc.maxIter,
                                calc.tol, calc.steps, (calc.cgTol), false,
                                EigenpairsOrdering::ASCENDING_ENERGIES);
}

void Utilities::printKV(const std::string &key, double value, int widthKey = 18,
                        int widthVal = 12, int precision = 3,
                        bool scientific = false) {
  using std::cout, std::endl;
  cout << "  " << std::left << std::setw(widthKey) << key << std::right
       << std::setw(widthVal);

  if (scientific)
    cout << std::scientific;
  else
    cout << std::fixed;

  cout << std::setprecision(precision) << value << endl;

  // reset flags
  cout.unsetf(std::ios::floatfield);
}

void Utilities::skyrmeHamiltonian(std::vector<std::shared_ptr<Potential>> &pots,
                                  InputParser input, NucleonType t,
                                  std::shared_ptr<IterationData> data) {
  using std::make_shared;
  auto grid = Grid::getInstance();
  pots.push_back(make_shared<SkyrmeU>(t, data));
  pots.push_back(make_shared<NonLocalKineticPotential>(data, t));
  pots.push_back(make_shared<LocalKineticPotential>(data, t));
  if (input.spinOrbit)
    pots.push_back(make_shared<SkyrmeSO>(data, t));

  if (t == NucleonType::P && input.useCoulomb) {
    pots.push_back(make_shared<LocalCoulombPotential>(data->UCoul));
  }
}
double Utilities::mu20FromBeta(double beta, double R, int A) {
  return beta * 3 * A * R * R / 4 / M_PI;
}
double Utilities::computeDispersion(const ComplexSparseMatrix &h,
                                    const ComplexDenseMatrix &X) {
  ComplexDenseMatrix HX = h * X;
  ComplexDenseMatrix HHX = h.adjoint() * HX;
  auto grid = *Grid::getInstance();

  double sum = 0.0;

  for (int i = 0; i < X.cols(); ++i) {
    std::complex<double> expH =
        (X.col(i).conjugate().cwiseProduct(HX.col(i))).sum() * grid.dV();
    std::complex<double> expHH =
        (X.col(i).conjugate().cwiseProduct(HHX.col(i))).sum() * grid.dV();

    auto diff = expHH.real() - expH.real() * expH.real();

    double dispersion = diff;

    sum += dispersion;
  }

  return sum / X.cols();
}
