#pragma once

class IterationData;

class FunctionalTerm {
    public:
        virtual double evaluate(IterationData* data) const ;

};



