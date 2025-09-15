#pragma once
#include "constraint.hpp"
#include <Eigen/Dense>

class AxialSymmetryConstraint : public Constraint {
    public:
        AxialSymmetryConstraint(double mu20);



    Eigen::VectorXd getField(IterationData* data) override; 

        double evaluate(IterationData* data) const override;


    private:
        double mu20;
        double lambda;
        double C;
        bool firstIter;
        std::vector<double> residuals;

};

