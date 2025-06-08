#pragma once
#include <Eigen/Dense>
#include "local_potential.hpp"
#include <memory>

class ExchangeCoulombPotential : public Potential {

public:
/*
    @brief Implements the exchange-correlation coulomb interaction in the Slater approximation.
    @param rho The proton nucleon density
*/
ExchangeCoulombPotential(std::shared_ptr<Eigen::VectorXd> rho);

    std::complex<double> getElement5p(int i, int j, int k, int s, int i1, int j1, int k1, int s1, const Grid& grid) const override;
    double getValue(double x, double y, double z) const override;
private:
    std::shared_ptr<Eigen::VectorXd> rho;
};