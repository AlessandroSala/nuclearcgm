#include "skyrme/local_coulomb_potential.hpp"
#include "omp.h"
#include "constants.hpp"
#include "grid.hpp"
#include <iostream>
LocalCoulombPotential::LocalCoulombPotential(std::shared_ptr<Eigen::VectorXd> rho_)
    : rho(rho_)
{
}
double LocalCoulombPotential::getValue(double x, double y, double z) const
{
    return 0.0;
}
std::complex<double> LocalCoulombPotential::getElement5p(int i, int j, int k, int s, int i1, int j1, int k1, int s1, const Grid &grid) const
{
    double res = 0.0;
    double h = grid.get_h();
    int n = grid.get_n();
    using namespace nuclearConstants;
    if (i != i1 || j != j1 || k != k1 || s != s1)
    {
        return std::complex<double>(0.0, 0.0);
    }
#pragma omp parallel for collapse(3) reduction(+ : res)
    for (int ii = 0; ii < n; ++ii)
    {
        for (int jj = 0; jj < n; ++jj)
        {
            for (int kk = 0; kk < n; ++kk)
            {
                int iNS = grid.idxNoSpin(ii, jj, kk);
                if (ii == i && jj == j && kk == k)
                {
                    res+=(*rho)(iNS)*h*h*1.939285;
                }
                else
                {
                    res += (h * h) * ((*rho)(iNS)) / (sqrt((ii - i) * (ii - i) + (jj - j) * (jj - j) + (kk - k) * (kk - k)));
                }
            }
        }
    }

    res *= 0.5 * e2;

    return std::complex<double>(res, 0.0);
}
