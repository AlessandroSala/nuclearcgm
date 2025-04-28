#pragma once
#include "local_potential.hpp"
/**
 * @brief Implements a Woods-Saxon central potential.
 * V(r) = -V0 / (1 + exp((r - R) / diff))
 */
class WoodsSaxonPotential : public LocalPotential {
public:
    /**
     * @brief Constructs a Woods-Saxon potential term.
     * @param V0 Potential depth (positive value, e.g., 51.0 MeV).
     * @param R Nuclear radius parameter (e.g., 1.27 * A^(1/3) fm).
     * @param diff Diffuseness parameter (e.g., 0.67 fm).
     */
    WoodsSaxonPotential(double V0, double R, double diff);

    /**
     * @brief Calculates the Woods-Saxon contribution to H(n0, n1).
     * Only non-zero for diagonal elements (n0 == n1).
     */
    double getValue(double x, double y, double z) const override;

public:
    double V0;   // Potential depth
    double R;    // Radius parameter
    double diff; // Diffuseness parameter
};
