#include "skyrme/skyrme_u.hpp"
#include "constants.hpp"
#include "util/iteration_data.hpp"

SkyrmeU::SkyrmeU(NucleonType n_, std::shared_ptr<IterationData> data_)
    : data(data_), n(n_) {}

double SkyrmeU::getValue(double x, double y, double z) const { return 0.0; }

std::complex<double> SkyrmeU::getElement(int i, int j, int k, int s, int i1,
                                         int j1, int k1, int s1,
                                         const Grid &grid) const {
  return 0.0;
}
std::complex<double> SkyrmeU::getElement5p(int i, int j, int k, int s, int i1,
                                           int j1, int k1, int s1,
                                           const Grid &grid) const {
  if (i != i1 || j != j1 || k != k1 || s != s1) {
    return std::complex<double>(0.0, 0.0);
  }
  int idx = grid.idxNoSpin(i, j, k);
  double field = n == NucleonType::N ? (*data->UN)(idx) : (*data->UP)(idx);
  return std::complex<double>(field, 0.0);
}
