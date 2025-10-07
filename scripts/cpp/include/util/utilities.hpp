#pragma once
#include <Eigen/Dense>
#include <memory>
#include "constants.hpp"
#include "types.hpp"

class Hamiltonian;
class GCGParameters;
class InputParser;
class IterationData;
class Potential;
class BCSResult;
namespace Utilities
{

      std::pair<Eigen::MatrixXcd, Eigen::VectorXd>
      solve(const ComplexSparseMatrix &hamiltonian, GCGParameters &calc,
            const Eigen::MatrixXcd &guess);
      std::pair<Eigen::MatrixXcd, Eigen::VectorXd>
      solve(const ComplexSparseMatrix &hamiltonian, GCGParameters &calc,
            const Eigen::MatrixXcd &guess, int nev);
      void skyrmeHamiltonian(std::vector<std::shared_ptr<Potential>> &pots,
                             InputParser input, NucleonType t, std::shared_ptr<IterationData> data);
}
