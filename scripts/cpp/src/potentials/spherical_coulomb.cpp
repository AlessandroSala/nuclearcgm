#include <cmath>
#include <iostream>
#include "spherical_coulomb.hpp"

SphericalCoulombPotential::SphericalCoulombPotential(int Z_, double R_)
    : Z(Z_), R(R_) {
        std::cout << "Coulomb potential with Z = " << Z << " and R = " << R << std::endl;
    }
double SphericalCoulombPotential::getValue(double x, double y, double z) const {

    double r = sqrt(x*x + y*y + z*z);
    
    double fac = pow(r, -1);
    if(r<=R) {
        fac = (3 - pow(r/R, 2))*pow(2*R, -1);
    }
    return fac*1.43996*Z;
}