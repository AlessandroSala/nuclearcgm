#pragma once
#include "json/json.hpp"

typedef struct SkyrmeParameters {
  double W0;
  double W0_pr;
  double t0;
  double t1;
  double t2;
  double t3;
  double x0;
  double x1;
  double x2;
  double x3;
  double sigma;
  double te;
  double to;
} SkyrmeParameters;

struct FunctionalParameters {
  double C0rr, C1rr, C0Drr, C1Drr, C0rt, C1rt, C0J2, C1J2, C0rdr, C1rdr, C0nJ,
      C1nJ, sigma;
};

class EDF {
public:
  FunctionalParameters params;

  EDF(nlohmann::json functional);
  void setFromInteraction(SkyrmeParameters params);
};
