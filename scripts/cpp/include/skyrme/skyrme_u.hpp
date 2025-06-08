
#pragma once
#include "util/iteration_data.hpp"
#include "local_potential.hpp"
#include <Eigen/Dense>
#include <memory>
#include "input_parser.hpp"
#include "constants.hpp"

class SkyrmeU : public Potential {
public:
  /**
   */
  SkyrmeU(SkyrmeParameters p, NucleonType n, std::shared_ptr<IterationData> data);
  double getValue(double x, double y, double z) const override;
  std::complex<double> getElement(int i, int j, int k, int s, int i1, int j1,
                                  int k1, int s1,
                                  const Grid &grid) const ;
  std::complex<double> getElement5p(int i, int j, int k, int s, int i1, int j1,
                                    int k1, int s1,
                                    const Grid &grid) const override;

public:
  SkyrmeParameters params;
  std::shared_ptr<IterationData> data;
  NucleonType n;
};
