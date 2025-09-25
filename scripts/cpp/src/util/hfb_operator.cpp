#include "hfb_operator.hpp"
#include <cassert>

HFBOperator::HFBOperator(const ComplexSparseMatrix &h_in,
                         const Eigen::VectorXcd &Delta_in,
                         double lambda_in)
    : h_sp(h_in), Delta(Delta_in), lambda(lambda_in), N((int)h_in.rows())
{
    assert(h_sp.rows() == h_sp.cols());
    assert((int)Delta.size() == N);
}

ComplexDenseVector HFBOperator::apply(const ComplexDenseVector &x) const
{
    assert((int)x.size() == 2 * N);

    ComplexDenseVector U = x.head(N);
    ComplexDenseVector V = x.tail(N);

    ComplexDenseVector hU = h_sp * U;
    ComplexDenseVector hV = h_sp * V;
    hU.noalias() -= lambda * U;
    hV.noalias() -= lambda * V;

    ComplexDenseVector deltaV = Delta.array() * V.array();
    ComplexDenseVector deltaStarU = Delta.conjugate().array() * U.array();

    ComplexDenseVector y(2 * N);
    y.head(N) = hU + deltaV;
    y.tail(N) = -deltaStarU - hV;

    return y;
}

ComplexDenseVector HFBOperator::applyShifted(const ComplexDenseVector &x, double W) const
{
    ComplexDenseVector y = apply(x);
    y.noalias() -= W * x;
    return y;
}
