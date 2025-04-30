#include <cmath>
#include "woods_saxon.hpp"

WoodsSaxonPotential::WoodsSaxonPotential(double V0_, double R_, double diff_)
    : V0(V0_), R(R_), diff(diff_) {}
double WoodsSaxonPotential::getValue(double x, double y, double z) const {
    return -V0 / (1 + exp((sqrt(x*x + y*y + z*z) - R) / diff));
}