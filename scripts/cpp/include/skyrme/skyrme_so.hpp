#pragma once
#include "constants.hpp"
#include "input_parser.hpp"
#include "potential.hpp"
#include <Eigen/Dense>
#include <memory>
class IterationData;

class SkyrmeSO : public Potential {
public:
  SkyrmeSO(std::shared_ptr<IterationData> data, NucleonType n);

  double getValue(double x, double y, double z) const override;
  std::complex<double> getElement5p(int i, int j, int k, int s, int i1, int j1,
                                    int k1, int s1,
                                    const Grid &grid) const override;

public:
  std::shared_ptr<IterationData> data;
  NucleonType n;
};
