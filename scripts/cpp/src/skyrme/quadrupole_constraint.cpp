#include "skyrme/quadrupole_constraint.hpp"
#include <iostream>
#include "util/fields.hpp"
#include "spherical_harmonics.hpp"
#include "util/iteration_data.hpp"
#include "operators/integral_operators.hpp"

QuadrupoleConstraint::QuadrupoleConstraint(double mu20) : mu20(mu20), C(0.05), lambda(0.0), firstIter(true) {
    residuals.clear();
}
Eigen::VectorXd QuadrupoleConstraint::getField(IterationData* data) {

    auto grid = *Grid::getInstance();
    Eigen::VectorXd rho = *(data->rhoN) + *(data->rhoP);
    double Q20 = SphericalHarmonics::Q(2, 0, rho).real();
    Eigen::VectorXd Y20 = SphericalHarmonics::Y(2, 0).real();
    Eigen::VectorXd pos = Fields::position().array().abs().pow(2);
    Eigen::VectorXd O = Y20.array() * pos.array();

    std::cout << "q: " << Q20 << std::endl;
    std::cout << "mu20: " << mu20 << std::endl;

    std::cout << "Constraint energy: " << evaluate(data) << std::endl;
    if(firstIter) {
        firstIter = false;
        //return Eigen::VectorXd::Zero(data->rhoN->rows());
        return 2.0*C*(Q20 - mu20)* O;
    }

    double gamma = 1.0;

    if(residuals.size() > 1 && std::abs(residuals.back()/residuals[residuals.size()-2]-1) < 1e-2) {
        lambda += gamma*2.0*C*(Q20 - mu20);
        std::cout << "Updated lambda: " << lambda << std::endl;
    }
    double mu = mu20 - lambda/(2.0*C); 
    double alpha = lambda + 2.0*C*(Q20 - mu20);
    std::cout << "alpha: " << alpha << std::endl;

    double residual = (Q20 - mu20);

    residuals.push_back(residual);

    for(int i = 0; i < residuals.size(); ++i) {
        std::cout << residuals[i] << ", ";
    }
    std::cout << std::endl;



    return 2.0*C*(Q20 - mu)* O;
}

 double QuadrupoleConstraint::evaluate(IterationData* data) const{
     Eigen::VectorXd rho = *(data->rhoN) + *(data->rhoP);
     auto grid = *Grid::getInstance();
     double Qc = SphericalHarmonics::Q(2, 0, rho).real();
     return  C*pow(Qc - mu20, 2) + lambda * (Qc - mu20);
}
