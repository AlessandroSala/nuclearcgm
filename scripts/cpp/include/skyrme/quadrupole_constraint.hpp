#pragma once
#include "constraint.hpp"
#include <Eigen/Dense>

class QuadrupoleConstraint : public Constraint {
    public:
        QuadrupoleConstraint(double mu20);
        QuadrupoleConstraint(double mu20, double lambda);



    Eigen::VectorXd getField(IterationData* data) override; 

        double evaluate(IterationData* data) const override;


    double lambda;
    private:
        double C;
        bool firstIter;
        std::vector<double> residuals;

};

