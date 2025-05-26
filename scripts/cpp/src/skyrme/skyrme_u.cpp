#include "skyrme/skyrme_u.hpp"

SkyrmeU::SkyrmeU(std::shared_ptr<Eigen::VectorXd> rho_, std::shared_ptr<Eigen::VectorXd> nabla2rho_, std::shared_ptr<Eigen::VectorXd> tau_, std::shared_ptr<Eigen::VectorXcd> divJ_, double t0_, double t1_, double t2_, double t3_, double W0_)
: rho(rho_), nabla2rho(nabla2rho_), tau(tau_), divJ(divJ_), t0(t0_), t1(t1_), t2(t2_), t3(t3_), W0(W0_)
{
}

double SkyrmeU::getValue(double x, double y, double z) const
{
    return 0.0;
}

std::complex<double> SkyrmeU::getElement(int i, int j, int k, int s, int i1, int j1, int k1, int s1, const Grid& grid) const
{
    return 0.0;
}
std::complex<double> SkyrmeU::getElement5p(int i, int j, int k, int s, int i1, int j1, int k1, int s1, const Grid& grid) const
{

    std::complex<double> res = std::complex<double>(0.0, 0.0);
    if(i != i1 || j != j1 || k != k1 || s != s1) {
        return res;
    }
    int iNS = grid.idxNoSpin(i, j, k);
    res+=0.75*t0*(*rho)(iNS);
    res+=(3.0/16.0)*t3*pow((*nabla2rho)(iNS), 2.0);
    res+=(1.0/16.0)*(3*t1 + 5*t2)*(*tau)(iNS);
    res+=(1.0/32.0)*(5*t2 - 9*t1)*(*nabla2rho)(iNS) - 0.75*W0*(*divJ)(iNS);

    return res;
}