#pragma once
#include <Eigen/Dense>

class BogoliubovStates
{
public:
    BogoliubovStates(int _points);

    Eigen::MatrixXcd W;
    Eigen::VectorXd energies;

    auto U() const noexcept;
    auto V() const noexcept;

private:
    int points;
};