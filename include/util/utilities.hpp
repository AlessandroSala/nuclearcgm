#pragma once
#include "constants.hpp"
#include "types.hpp"
#include <Eigen/Dense>
#include <memory>

class Hamiltonian;
class GCGParameters;
class InputParser;
class IterationData;
class Potential;
class BCSResult;
namespace Utilities {

void printKV(const std::string &key, double value, int widthKey, int widthVal,
             int precision, bool scientific);

double mu20FromBeta(double beta, double R, int A);
std::pair<Eigen::MatrixXcd, Eigen::VectorXd>
solve(const ComplexSparseMatrix &hamiltonian, const ComplexDenseMatrix &ConjDir,
      GCGParameters &calc, const Eigen::MatrixXcd &guess);
std::pair<Eigen::MatrixXcd, Eigen::VectorXd>
solve(const ComplexSparseMatrix &hamiltonian, const ComplexDenseMatrix &ConjDir,
      GCGParameters &calc, const Eigen::MatrixXcd &guess, int nev);
void skyrmeHamiltonian(std::vector<std::shared_ptr<Potential>> &pots,
                       InputParser input, NucleonType t,
                       std::shared_ptr<IterationData> data);
double computeDispersion(const ComplexSparseMatrix &h,
                         const ComplexDenseMatrix &X);
} // namespace Utilities
