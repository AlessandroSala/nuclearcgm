#include <cmath>
#include "spherical_harmonics.hpp"
#include "woods_saxon/deformed_woods_saxon.hpp"
#include "input_parser.hpp"

DeformedWoodsSaxonPotential::DeformedWoodsSaxonPotential(Parameters::WoodsSaxonParameters params, int A_, int Z_)
: V0(params.V0), A(A_), radius(Radius(params.beta, params.r0, A)), Z(Z_), kappa(params.kappa), diff(params.diff)
    {
    }
DeformedWoodsSaxonPotential::DeformedWoodsSaxonPotential(double V0_, Radius radius_, double diff_, int A_, int Z_, double kappa_, double beta3)
    : V0(V0_), radius(radius_), diff(diff_) , A(A_), Z(Z_), kappa(kappa_), beta3(beta3) {}
double DeformedWoodsSaxonPotential::getValue(double x, double y, double z) const {
    double locR = radius.getRadius(x, y, z);
    double r = sqrt(x*x + y*y + z*z);
    double beta3def = beta3 * radius.r_0 * std::pow(A, 1.0 / 3.0)*SphericalHarmonics::Y(3, 0, x, y, z).real();
    return -V0*(1 + kappa*(A-2*Z)/((double)A)) / (1 + exp((r - (locR+beta3def)) / diff));
}
