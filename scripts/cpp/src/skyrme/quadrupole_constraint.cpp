#include "skyrme/quadrupole_constraint.hpp"
#include "spherical_harmonics.hpp"

QuadrupoleConstraint::QuadrupoleConstraint(double mu20) : mu20(mu20) {}

std::complex<double> QuadrupoleConstraint::getElement5p(int i, int j, int k, int s, int i1, int j1, int k1, int s1,
                                          const Grid& grid) const {
    if(i != i1 || j != j1 || k != k1 || s != s1)
        return std::complex<double>(0.0, 0.0);
    double x = grid.get_xs()[i];
    double y = grid.get_ys()[j];
    double z = grid.get_zs()[k];
    double r2 = x * x + y * y + z * z;


    return mu20*r2*SphericalHarmonics::Y(2, 0, x, y, z)*std::sqrt(16.0*M_PI/5);
}

double QuadrupoleConstraint::getValue(double x, double y, double z) const {
    return 0.0;
}

 double QuadrupoleConstraint::evaluate(IterationData* data) const{
    return 0.0;
}
