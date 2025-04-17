#include <cmath>
#include "woods_saxon.hpp"

WoodsSaxonPotential::WoodsSaxonPotential(double V0_, double R_, double diff_)
    : V0(V0_), R(R_), diff(diff_) {}
double WoodsSaxonPotential::getValue(double x, double y, double z) const {
    return -V0 / (1 + exp((sqrt(x*x + y*y + z*z) - R) / diff));
}
double WoodsSaxonPotential::getElement(int i, int j, int k, int s, int i1, int j1, int k1, int s1, const Grid& grid) const {
    if(i != i1 || j != j1 || k != k1 || s != s1) {
        return 0.0;
    }
    return getValue(grid.get_xs()[i], grid.get_ys()[j], grid.get_zs()[k]);
}