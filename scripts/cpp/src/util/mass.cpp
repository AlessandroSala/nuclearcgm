#include "util/mass.hpp"
#include <cmath>
#include "constants.hpp"
#include "operators/differential_operators.hpp"
#include <iostream>

Mass::Mass(std::shared_ptr<Grid> grid, std::shared_ptr<IterationData> data_, SkyrmeParameters p, NucleonType n_)
    : grid_ptr_(grid), params(p), n(n_), data(data_)
{}
//TODO: ambiguità!!
double Mass::getMass(size_t i, size_t j, size_t k) const noexcept {
    using namespace nuclearConstants;
    double res = 0.0;
    double t1 = params.t1, t2 = params.t2;
    double x1 = params.x1, x2 = params.x2;
    int idx = grid_ptr_->idxNoSpin(i, j, k);
    double rho_q = n == NucleonType::N ? (*data->rhoN)(idx) : (*data->rhoP)(idx);
    double rho = (*data->rhoN)(idx) + (*data->rhoP)(idx);

    res += h_bar*h_bar/(2*m);
    //Chabanat
    //res += (1.0/8.0)*(t1*(2+x1)+t2*(2+x2))*((*data->rhoP)(idx) + (*data->rhoN)(idx));
    //res += -(1.0/8.0)*(+t1*(1+2*x1)+t2*(1+2*x2))*rho_q;

    //Engel
    //res += (1.0/4.0)*(t1+t2)*rho;
    //res += (1.0/8.0)*(t2-t1)*rho_q;
    return res;
}
Eigen::VectorXd Mass::getMassVector() const noexcept {
    using namespace nuclearConstants;
    Eigen::VectorXd res(grid_ptr_->get_total_spatial_points());
    double t1 = params.t1, t2 = params.t2;
    double x1 = params.x1, x2 = params.x2;
    auto rho_q = n == NucleonType::N ? (*data->rhoN) : (*data->rhoP);
    auto rho = (*data->rhoN) + (*data->rhoP);
    Eigen::VectorXd ones(grid_ptr_->get_total_spatial_points());
    ones.setOnes();
    res +=  ones*h_bar*h_bar/(2*m);
    //Chabanat
    //res += (1.0/8.0)*(t1*(2+x1)+t2*(2+x2))*((*data->rhoP)(idx) + (*data->rhoN)(idx));
    //res += -(1.0/8.0)*(+t1*(1+2*x1)+t2*(1+2*x2))*rho_q;

    //Engel
    res += (1.0/4.0)*(t1+t2)*rho;
    res += (1.0/8.0)*(t2-t1)*rho_q;
    return res;
}
Eigen::Vector3d Mass::getGradient(size_t i, size_t j, size_t k) const noexcept {
    Eigen::Vector3d grad(3);
    grad.setZero();

    double t1 = params.t1, t2 = params.t2;
    double x1 = params.x1, x2 = params.x2;
    int idx = grid_ptr_->idxNoSpin(i, j, k);
    Eigen::Vector3d nablaRho_q = n == NucleonType::N ? data->nablaRhoN->row(idx) : data->nablaRhoP->row(idx); 
    Eigen::Vector3d nablaRho = data->nablaRhoN->row(idx) + data->nablaRhoP->row(idx);

    //Chabanat
    //grad += (1.0/8.0)*(t1*(2+x1)+t2*(2+x2))*(data->nablaRhoN->row(idx) + data->nablaRhoP->row(idx));
    //grad += -(1.0/8.0)*(t1*(1+2*x1)+t2*(1+2*x2))*nablaRho_q;

    //Engel
    grad += (1.0/4.0)*(t1+t2)*nablaRho;
    grad += (1.0/8.0)*(t2-t1)*nablaRho_q;

    //Bertulani
    //grad = (1.0/16.0)*(3*t1 + 5*t2)*nablaRho;

    return grad;

}
