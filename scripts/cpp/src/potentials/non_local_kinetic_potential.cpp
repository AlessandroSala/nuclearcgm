#include <cmath>
#include "kinetic/non_local_kinetic_potential.hpp"
#include "constants.hpp"


NonLocalKineticPotential::NonLocalKineticPotential(std::shared_ptr<Mass> m_)
    : m(m_) {}
double NonLocalKineticPotential::getValue(double x, double y, double z) const
{
    return 1;
}
// TODO: not really, implement 3p derivatives
std::complex<double> NonLocalKineticPotential::getElement(int i, int j, int k, int s, int i1, int j1, int k1, int s1, const Grid &grid) const {

return 0;
}
std::complex<double> NonLocalKineticPotential::getElement5p(int i, int j, int k, int s, int i1, int j1, int k1, int s1, const Grid &grid) const
{
    if(i==i1 && j==j1 && k==k1 && s==s1) {
        return std::complex<double>(0.0, 0.0);
    }
    std::complex<double> val(0.0, 0.0);
    double h = grid.get_h();

    Eigen::Vector3d C = -m->getGradient(i, j, k);

    if (s == s1 && ((i == i1 && j == j1 && std::abs(k1 - k) == 1)))
    {
        val = (k1-k)*8.0*C(2);
    }
    else if( s == s1 && ((i == i1 && k == k1 && std::abs(j1 - j) == 1)))
    {
        val = (j1-j)*8.0*C(1);
    }
    else if( s == s1 && ((j == j1 && k == k1 && std::abs(i1 - i) == 1)))
    {
        val = (i1-i)*8.0*C(0);
    }
    else if (s == s1 && ((i == i1 && j == j1 && std::abs(k1 - k) == 2)))
    {
        val = -(k1-k)*C(2)/2.0;
    }
    else if( s == s1 && ((i == i1 && k == k1 && std::abs(j1 - j) == 2)))
    {
        val = -(j1-j)*C(1)/2.0;
    }
    else if( s == s1 && ((j == j1 && k == k1 && std::abs(i1 - i) == 2)))
    {
        val = -(i1-i)*C(0)/2.0;
    }
    
    return val / (12.0 * h );
}