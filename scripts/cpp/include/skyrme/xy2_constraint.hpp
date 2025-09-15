#pragma once
#include "constraint.hpp"
#include <Eigen/Dense>

class XY2Constraint : public Constraint {
    public:
        XY2Constraint(double mu20);



    Eigen::VectorXd getField(IterationData* data) override; 

        double evaluate(IterationData* data) const override;


    private:
        double mu20;
        double lambda;
        double C;
        bool firstIter;
        std::vector<double> residuals;

};

