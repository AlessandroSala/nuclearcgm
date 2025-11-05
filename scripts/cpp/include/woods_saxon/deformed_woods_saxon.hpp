
#pragma once
#include "local_potential.hpp"
#include "radius.hpp"
namespace Parameters {
    typedef struct {
        double V0;
        double beta;
        double beta3;
        double r0;
        double diff;
        double kappa;
    } WoodsSaxonParameters;
}
/**
 * @brief Implements a deformed Woods-Saxon potential.
 * V(r) = -V0 / (1 + exp((r(x,y,z) - R) / diff))
 */
class DeformedWoodsSaxonPotential : public LocalPotential {
public:
    /**
     * @brief Constructs a deformed Woods-Saxon potential term.
     * @param V0 Potential depth (positive value, e.g., 51.0 MeV).
     * @param R Nuclear radius parameter (e.g., 1.27 * A^(1/3) fm).
     * @param diff Diffuseness parameter (e.g., 0.67 fm).
     */
    DeformedWoodsSaxonPotential(double V0, Radius radius, double diff, int A, int Z, double kappa, double beta3);
    DeformedWoodsSaxonPotential(Parameters::WoodsSaxonParameters params, int A, int Z);

    /**
     * @brief Calculates the Woods-Saxon contribution to H(n0, n1).
     * Only non-zero for diagonal elements (n0 == n1).
     */
    double getValue(double x, double y, double z) const override;
    Parameters::WoodsSaxonParameters parameters;

public:
    double V0;   // Potential depth
    Radius radius;    // Radius parameter
    double diff; // Diffuseness parameter
    int A, Z; //Nucleues mass, charge
    double kappa; // asymm parameter
    double beta3;
};

