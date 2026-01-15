#pragma once
#include "input_parser.hpp"

struct FunctionalParameters {
  double C0rr, C1rr, C0Drr, C1Drr, C0rt, C1rt, C0J2, C1J2, C0rdr, C1rdr, C0nJ,
      C1nJ;
  double sigma;
};

class EDF {
public:
  FunctionalParameters params;

  EDF(SkyrmeParameters params);
  EDF(FunctionalParameters params);
};
