
#pragma once
#include "local_potential.hpp"
#include <memory>
#include <Eigen/Dense>
/**
 */
class SkyrmeU : public LocalPotential {
public:
    /**
     */
    SkyrmeU(std::shared_ptr<Eigen::VectorXd> rho, std::shared_ptr<Eigen::VectorXd> nabla2rho, std::shared_ptr<Eigen::VectorXd> tau, std::shared_ptr<Eigen::VectorXcd> divJ, double t0, double t1, double t2, double t3, double W0);


public:
    double t0, t1, t2, t3, W0;
    std::shared_ptr<Eigen::VectorXd> rho;
    std::shared_ptr<Eigen::VectorXd> nabla2rho;
    std::shared_ptr<Eigen::VectorXd> tau;
    std::shared_ptr<Eigen::VectorXcd> divJ;
};
