#include <cmath>
#include "woods_saxon/deformed_woods_saxon.hpp"
#include "input_parser.hpp"

DeformedWoodsSaxonPotential::DeformedWoodsSaxonPotential(Parameters::WoodsSaxonParameters params, int A_, int Z_)
: V0(params.V0), A(A_), radius(Radius(params.beta, params.r0, A)), Z(Z_), kappa(params.kappa), diff(params.diff)
    {
    }
DeformedWoodsSaxonPotential::DeformedWoodsSaxonPotential(double V0_, Radius radius_, double diff_, int A_, int Z_, double kappa_)
    : V0(V0_), radius(radius_), diff(diff_) , A(A_), Z(Z_), kappa(kappa_) {}
double DeformedWoodsSaxonPotential::getValue(double x, double y, double z) const {
    double locR = radius.getRadius(x, y, z);
    double r = sqrt(x*x + y*y + z*z);
    return -V0*(1 + kappa*(A-2*Z)/((double)A)) / (1 + exp((r - locR) / diff));
}