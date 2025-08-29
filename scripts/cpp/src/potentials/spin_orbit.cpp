#include <cmath>
#include "spin_orbit.hpp"
#include "constants.hpp"

SpinOrbitPotential::SpinOrbitPotential(double V0_, double r0_, double R_)
    : V0(V0_), r0(r0_), R(R_){}
double SpinOrbitPotential::getValue(double x, double y, double z) const {
    double fac = 1;
    double r = sqrt(x*x + y*y + z*z);
    double diff = 0.67;
    if(r > 1e-12) {
        double t = (r-R)/diff;
        fac = -pow(diff*r, -1)*exp(t)/pow(1+exp(t), 2);
    }
    return 0.44*V0*pow((r0/nuclearConstants::h_bar), 2)*fac;
}
std::complex<double> SpinOrbitPotential::getElement(int i, int j, int k, int s, int i1, int j1, int k1, int s1, const Grid& grid) const {
    if (i1 == i && j1 == j && k1 == k && s1 == s)
        return 0;
    SpinMatrix spin(2, 2);
    spin.setZero();
    auto pauli = nuclearConstants::getPauli();

    double h = grid.get_h();
    double x = grid.get_xs()[i], y = grid.get_ys()[j], z = grid.get_zs()[k];
    double ls = getValue(x, y, z);
    if((i + 1 == i1 || i - 1 == i1)&& j == j1 && k == k1)
        spin += (i1-i)*(pauli[1]*z - pauli[2]*y);

    else if(i == i1 && (j + 1 == j1 || j - 1 == j1) && k == k1)
        spin += (j1-j)*(-pauli[0]*z + pauli[2]*x);

    else if(i == i1 && j == j1 && (k + 1 == k1 || k - 1 == k1))
        spin += (k1-k)*(pauli[0]*y - pauli[1]*x);
    else 
        return 0;
    //ls = 0;
    using namespace nuclearConstants;
    spin = -pow(2*h, -1)*std::complex<double>(0, 1.0)*0.5*h_bar*h_bar*spin * ls;

    return spin(s, s1);
}
std::complex<double> SpinOrbitPotential::getElement5p(int i, int j, int k, int s, int i1, int j1, int k1, int s1, const Grid& grid) const {
    if (i1 == i && j1 == j && k1 == k && s1 == s)
        return 0;
    SpinMatrix spin(2, 2);
    spin.setZero();
    auto pauli = nuclearConstants::getPauli();

    double h = grid.get_h();
    double x = grid.get_xs()[i], y = grid.get_ys()[j], z = grid.get_zs()[k];
    double ls = getValue(x, y, z);

    if(std::abs(i - i1) == 1 && j == j1 && k == k1)
        spin += (2.0/3.0)*(i1-i)*(pauli[1]*z - pauli[2]*y);
    else if(std::abs(i - i1) == 2 && j == j1 && k == k1)
        spin += -(i1-i)*(1.0/24.0)*(pauli[1]*z - pauli[2]*y); // i1-i carries factor 2

    else if(i == i1 && std::abs(j - j1) == 1 && k == k1)
        spin += (2.0/3.0)*(j1-j)*(-pauli[0]*z + pauli[2]*x);
    else if(i == i1 && std::abs(j - j1) == 2 && k == k1)
        spin += -(j1-j)*(1.0/24.0)*(-pauli[0]*z + pauli[2]*x); // j1-j carries factor 2

    else if(i == i1 && j == j1 && std::abs(k - k1) == 1)
        spin += (2.0/3.0)*(k1-k)*(pauli[0]*y - pauli[1]*x);
    else if(i == i1 && j == j1 && std::abs(k - k1) == 2)
        spin += -(k1-k)*(1.0/24.0)*(pauli[0]*y - pauli[1]*x); // k1-k carries factor 2
    else 
        return 0;
    //ls = 0;
    using namespace nuclearConstants;
    spin = -pow(h, -1)*std::complex<double>(0, 1.0)*0.5*h_bar*h_bar*spin * ls;

    return spin(s, s1);
}
