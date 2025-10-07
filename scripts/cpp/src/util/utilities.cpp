#include "util/utilities.hpp"
#include "hamiltonian.hpp"
#include "input_parser.hpp"
#include "kinetic/local_kinetic_potential.hpp"
#include "kinetic/non_local_kinetic_potential.hpp"
#include "skyrme/local_coulomb_potential.hpp"
#include "skyrme/skyrme_so.hpp"
#include "skyrme/skyrme_u.hpp"
#include "solver.hpp"
#include <Eigen/Dense>
#include <complex>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cassert>

using namespace Eigen;
using Complex = std::complex<double>;

std::pair<Eigen::MatrixXcd, Eigen::VectorXd>
Utilities::solve(const ComplexSparseMatrix &matrix, GCGParameters &calc,
                 const Eigen::MatrixXcd &guess)
{
  return gcgm_complex_no_B_lock(matrix, guess, guess.cols(), 0.0, calc.maxIter,
                                calc.tol, calc.steps, (calc.cgTol), false);
}

std::pair<Eigen::MatrixXcd, Eigen::VectorXd>
Utilities::solve(const ComplexSparseMatrix &matrix, GCGParameters &calc,
                 const Eigen::MatrixXcd &guess, int nev)
{
  return gcgm_complex_no_B_lock(matrix, guess, nev, 0.0, calc.maxIter,
                                calc.tol, calc.steps, (calc.cgTol), false);
}

void Utilities::skyrmeHamiltonian(std::vector<std::shared_ptr<Potential>> &pots,
                                  InputParser input, NucleonType t,
                                  std::shared_ptr<IterationData> data)
{
  using std::make_shared;
  auto grid = Grid::getInstance();
  pots.push_back(make_shared<SkyrmeU>(input.skyrme, t, data));
  pots.push_back(make_shared<NonLocalKineticPotential>(data, t));
  pots.push_back(make_shared<LocalKineticPotential>(data, t));
  if (input.spinOrbit)
    pots.push_back(make_shared<SkyrmeSO>(data, t));

  if (t == NucleonType::P && input.useCoulomb)
  {
    pots.push_back(make_shared<LocalCoulombPotential>(data->UCoul));
  }
}
