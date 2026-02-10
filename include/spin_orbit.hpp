#pragma once
#include "potential.hpp"
/**
 * @brief Implements a Spin orbit term for a Woods Saxon potential
 * V(r) = -V0 / (1 + exp((r - R) / diff))
 */
class SpinOrbitPotential : public Potential {
public:
    /**
     * @brief Constructs a Spin Orbit potential term.
     * @param V0 Potential depth (positive value, e.g., 51.0 MeV).
     * @param R Nuclear radius parameter (e.g., 1.27 * A^(1/3) fm).
     * @param diff Diffuseness parameter (e.g., 0.67 fm).
     */
    SpinOrbitPotential(double V0_, double r0_, double R_);

    double getValue(double x, double y, double z) const override;
    std::complex<double> getElement(int i, int j, int k, int s, int i1, int j1, int k1, int s1, const Grid& grid) const ;
    std::complex<double> getElement5p(int i, int j, int k, int s, int i1, int j1, int k1, int s1, const Grid& grid) const override;

public:
    double V0;
    double r0;    
    double R; 
};
