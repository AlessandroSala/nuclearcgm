#pragma once
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <complex>
#include "types.hpp"

class HFBOperator
{
public:
    HFBOperator(const ComplexSparseMatrix &h_in,
                const Eigen::VectorXcd &Delta_in,
                double lambda_in);

    ComplexDenseVector apply(const ComplexDenseVector &x) const;
    ComplexDenseVector applyShifted(const ComplexDenseVector &x, double W) const;

private:
    const ComplexSparseMatrix &h_sp;
    Eigen::VectorXcd Delta;
    double lambda;
    int N;
};
