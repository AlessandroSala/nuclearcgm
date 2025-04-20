#pragma once
#include "potential.hpp"
class SphericalCoulombPotential : public Potential {
public:
    /**
     * @brief Constructs a Coulomb potential of a spherical charge distribution.
     * @param R Nuclear radius parameter (e.g., 1.27 * A^(1/3) fm).
     */
    SphericalCoulombPotential(int Z, double R);

    double getValue(double x, double y, double z) const override;
    std::complex<double> getElement(int i, int j, int k, int s, int i1, int j1, int k1, int s1, const Grid& grid) const override;

public:
    int Z;
    double R;    // Radius parameter
};
