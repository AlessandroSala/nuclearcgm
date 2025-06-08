#pragma once 

#include "grid.hpp"
#include <complex>
/**
 * @brief Abstract base class (interface) for potential terms in the Hamiltonian.
 *
 * Each derived class represents a specific physical interaction
 * (e.g., Woods-Saxon, Spin-Orbit) and calculates its contribution
 * to the Hamiltonian matrix elements.
 */
class Potential {
public:
    /**
     * @brief Calculates the contribution of this potential term to a specific
     * Hamiltonian matrix element H(n0, n1).
     *
     * @return double, The contribution of this term to the matrix element.
     */
    virtual double getValue(double x, double y, double z) const = 0;
    virtual std::complex<double> getElement5p(int i, int j, int k, int s, int i1, int j1, int k1, int s1, const Grid& grid) const = 0;

    /**
     * @brief Virtual destructor is required for base classes with virtual functions.
     */
    virtual ~Potential() = default;
};
