#include <cmath>
#include "woods_saxon/deformed_woods_saxon.hpp"

DeformedWoodsSaxonPotential::DeformedWoodsSaxonPotential(double V0_, Radius radius_, double diff_)
    : V0(V0_), radius(radius_), diff(diff_) {}
double DeformedWoodsSaxonPotential::getValue(double x, double y, double z) const {
    double locR = radius.getRadius(x, y, z);
    double r = sqrt(x*x + y*y + z*z);
    return -V0 / (1 + exp((r - locR) / diff));
}