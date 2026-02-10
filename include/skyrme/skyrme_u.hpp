
#pragma once

#include "constants.hpp"
#include "input_parser.hpp"
#include "local_potential.hpp"
#include <Eigen/Dense>
#include <memory>

class IterationData;

class SkyrmeU : public Potential {
public:
  /**
   */
  SkyrmeU(NucleonType n, std::shared_ptr<IterationData> data);
  double getValue(double x, double y, double z) const override;
  std::complex<double> getElement(int i, int j, int k, int s, int i1, int j1,
                                  int k1, int s1, const Grid &grid) const;
  std::complex<double> getElement5p(int i, int j, int k, int s, int i1, int j1,
                                    int k1, int s1,
                                    const Grid &grid) const override;

public:
  std::shared_ptr<IterationData> data;
  NucleonType n;
};
