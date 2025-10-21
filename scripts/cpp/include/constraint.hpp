#pragma once
#include <Eigen/Dense>
#include "functional_term.hpp"
class Constraint : public FunctionalTerm
{
public:
    double evaluate(IterationData *data) const override;
    virtual Eigen::VectorXd getField(IterationData *data);
    virtual ~Constraint() = default;
    
    double target;
    double lambda;
    double gamma;
};
