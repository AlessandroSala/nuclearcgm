#include <cmath>
#include "kinetic/local_kinetic_potential.hpp"
#include "constants.hpp"

LocalKineticPotential::LocalKineticPotential(std::shared_ptr<Mass> m_)
    : m(m_) {}
double LocalKineticPotential::getValue(double x, double y, double z) const
{
    return 1;
}
// TODO: not really, implement 3p derivatives
std::complex<double> LocalKineticPotential::getElement(int i, int j, int k, int s, int i1, int j1, int k1, int s1, const Grid &grid) const {

return 0;
}
std::complex<double> LocalKineticPotential::getElement5p(int i, int j, int k, int s, int i1, int j1, int k1, int s1, const Grid &grid) const
{
    std::complex<double> val;
    double hh = grid.get_h() * grid.get_h();

    using nuclearConstants::h_bar;
    double mass = m->getMass(i, j, k);
    double C = -h_bar*h_bar/(2*mass);

    if (i == i1 && j == j1 && k == k1 && s == s1)
    {
        val = -C*(90.0/12.0) / hh;
    }
    else if (s == s1 && ((i == i1 && j == j1 && std::abs(k1 - k) == 1) ||
                         (i == i1 && k == k1 && std::abs(j1 - j) == 1) ||
                         (j == j1 && k == k1 && std::abs(i1 - i) == 1)))
    {
        val = C*(16.0/12.0) / hh;
    }
    else if (s == s1 && ((i == i1 && j == j1 && std::abs(k1 - k) == 2) ||
                         (i == i1 && k == k1 && std::abs(j1 - j) == 2) ||
                         (j == j1 && k == k1 && std::abs(i1 - i) == 2)))
    {
        val = -C*(1.0/12.0) / hh;
    }
    return val;
}