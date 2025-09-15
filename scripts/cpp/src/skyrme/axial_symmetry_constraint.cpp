#include "skyrme/axial_symmetry_constraint.hpp"
#include <iostream>
#include "util/fields.hpp"
#include "spherical_harmonics.hpp"
#include "util/iteration_data.hpp"
#include "operators/integral_operators.hpp"

AxialSymmetryConstraint::AxialSymmetryConstraint(double mu20) : mu20(mu20), C(0.05), lambda(0.0), firstIter(true) {
    residuals.clear();
}
Eigen::VectorXd AxialSymmetryConstraint::getField(IterationData* data) {
  using Operators::integral;
  using Eigen::VectorXd;
    auto grid = *Grid::getInstance();

  VectorXd x(grid.get_total_spatial_points());
  VectorXd y(grid.get_total_spatial_points());

  for (int i = 0; i < grid.get_n(); ++i) {
    for (int j = 0; j < grid.get_n(); ++j) {
      for (int k = 0; k < grid.get_n(); ++k) {
        int idx = grid.idxNoSpin(i, j, k);
        x(idx) = grid.get_xs()[i];
        y(idx) = grid.get_ys()[j];
      }
    }
  }
  

    Eigen::VectorXd rho = *(data->rhoN) + *(data->rhoP);
    Eigen::VectorXd O = x.array().pow(2) - y.array().pow(2) + 2.0*x.array()*y.array();
    double Q22 = integral((VectorXd)(O.array()*rho.array()), grid);

    std::cout << "q: " << Q22 << std::endl;
    std::cout << "axial symmetry: " << mu20 << std::endl;

    auto constraintEnergy = evaluate(data);
    std::cout << "Constraint energy: " << constraintEnergy << std::endl;

    if(firstIter) {
        firstIter = false;
        //return Eigen::VectorXd::Zero(data->rhoN->rows());
        return 2.0*C*(Q22 - mu20)* O;
    }
    double gamma = 1.0;

    if(residuals.size() > 1 && std::abs(residuals.back()/residuals[residuals.size()-2]-1) < 1e-2) {
        lambda += gamma*2.0*C*(Q22 - mu20);
        std::cout << "Updated lambda: " << lambda << std::endl;
    }
    double mu = mu20 - lambda/(2.0*C); 
    double alpha = lambda + 2.0*C*(Q22 - mu20);
    std::cout << "alpha: " << alpha << std::endl;

    double residual = (Q22 - mu20);

    residuals.push_back(residual);

    for(int i = 0; i < residuals.size(); ++i) {
        std::cout << residuals[i] << ", ";
    }
    std::cout << std::endl;



    return 2.0*C*(Q22 - mu)* O;
}

 double AxialSymmetryConstraint::evaluate(IterationData* data) const{
  using Operators::integral;
  using Eigen::VectorXd;
    auto grid = *Grid::getInstance();

  VectorXd x(grid.get_total_spatial_points());
  VectorXd y(grid.get_total_spatial_points());

  for (int i = 0; i < grid.get_n(); ++i) {
    for (int j = 0; j < grid.get_n(); ++j) {
      for (int k = 0; k < grid.get_n(); ++k) {
        int idx = grid.idxNoSpin(i, j, k);
        x(idx) = grid.get_xs()[i];
        y(idx) = grid.get_ys()[j];
      }
    }
  }
  

    Eigen::VectorXd rho = *(data->rhoN) + *(data->rhoP);
    Eigen::VectorXd O = (x.array().pow(2) - y.array().pow(2)).array().pow(2) + 4.0*x.array().pow(2)*y.array().pow(2);
    double Qc = integral((VectorXd)(O.array()*rho.array()), grid);

     return  C*pow(Qc - mu20, 2) + lambda * (Qc - mu20);
}
