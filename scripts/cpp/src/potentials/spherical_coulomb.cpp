#include <cmath>
#include "spherical_coulomb.hpp"

SphericalCoulombPotential::SphericalCoulombPotential(int Z_, double R_)
    : Z(Z_), R(R_) {}
double SphericalCoulombPotential::getValue(double x, double y, double z) const {
    double r = sqrt(x*x + y*y + z*z);
    double fac = pow(r, -1);
    if(r<=R) {
        fac = (3 - pow(r/R, 2))*pow(2*R, -1);
    }
    return fac*1.44*Z;
}
std::complex<double> SphericalCoulombPotential::getElement(int i, int j, int k, int s, int i1, int j1, int k1, int s1, const Grid& grid) const {
    if(i != i1 || j != j1 || k != k1 || s != s1) {
        return 0.0;
    }
    return getValue(grid.get_xs()[i], grid.get_ys()[j], grid.get_zs()[k]);
}