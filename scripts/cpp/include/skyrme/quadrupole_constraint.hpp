#pragma once
#include "potential.hpp"
#include "functional_term.hpp"

class QuadrupoleConstraint : public Potential, public FunctionalTerm {
    public:
        QuadrupoleConstraint(double mu20);

        std::complex<double> getElement5p(int i, int j, int k, int s, int i1, int j1, int k1, int s1,
                                          const Grid& grid) const override;
        double getValue(double x, double y, double z) const override;

        double evaluate(IterationData* data) const ;


    private:
        double mu20;
};

