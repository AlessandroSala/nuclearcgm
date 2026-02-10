#pragma once
#include "potential.hpp"
#include "radius.hpp"
#include <Eigen/Dense>
namespace Parameters {
    typedef struct {
        double V0;
        double diff;
        double r0;
        double beta;
    } SpinOrbitParameters;
}
/**
 * @brief Implements a Spin orbit term for a Woods Saxon potential
 * V(r) = -V0 / (1 + exp((r - R) / diff))
 */
class DeformedSpinOrbitPotential : public Potential {
public:
    /**
     * @brief Constructs a Spin Orbit potential term.
     * @param V0 Potential depth (positive value, e.g., 51.0 MeV).
     * @param R Nuclear radius parameter (e.g., 1.27 * A^(1/3) fm).
     * @param diff Diffuseness parameter (e.g., 0.67 fm).
     */
    DeformedSpinOrbitPotential(double V0_, Radius radius_, double diff_);
    DeformedSpinOrbitPotential(Parameters::SpinOrbitParameters p, int A);

    double getValue(double x, double y, double z) const override;
    Eigen::VectorXd getFactor(double x, double y, double z) const;
    std::complex<double> getElement(int i, int j, int k, int s, int i1, int j1, int k1, int s1, const Grid& grid) const ;
    std::complex<double> getElement5p(int i, int j, int k, int s, int i1, int j1, int k1, int s1, const Grid& grid) const override;

public:
    double V0;
    double diff;    
    Radius R; 
};
