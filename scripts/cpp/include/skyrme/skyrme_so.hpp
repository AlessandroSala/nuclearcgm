#pragma once
#include "potential.hpp"
#include <Eigen/Dense>
#include <memory>

class SkyrmeSO : public Potential {
public:
    SkyrmeSO(std::shared_ptr<Eigen::Matrix<double, -1, 3>> W_);

    double getValue(double x, double y, double z) const override;
    std::complex<double> getElement(int i, int j, int k, int s, int i1, int j1, int k1, int s1, const Grid& grid) const override;
    std::complex<double> getElement5p(int i, int j, int k, int s, int i1, int j1, int k1, int s1, const Grid& grid) const override;

public:
    std::shared_ptr<Eigen::Matrix<double, -1, 3>> W;
};
