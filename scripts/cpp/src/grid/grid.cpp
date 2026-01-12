#include "grid.hpp"
#include <cassert>
#include <stdexcept> // For std::invalid_argument
#include <string>    // For std::to_string in error message

// Constructor implementation
Grid::Grid(int n, double a)
    : n_points_(n), a_(a), h_(0.0) // Initialize members
{
  if (n_points_ < 2) {
    throw std::invalid_argument("Grid error: n must be >= 2. Received n = " +
                                std::to_string(n_points_));
  }

  // Calculate h only if n >= 2 (guaranteed by the check above)
  h_ = (2.0 * a_) / (n_points_ - 1);

  // Initialize the coordinate vectors
  initialize_coordinates();
  instance = this;
}

Grid *Grid::instance = nullptr;
Grid *Grid::getInstance() {
  assert(instance);
  return instance;
}

// Private helper function implementation
void Grid::initialize_coordinates() {
  xs_.resize(n_points_);
  ys_.resize(n_points_);
  zs_.resize(n_points_);

  for (size_t i = 0; i < n_points_; ++i) {
    double coord = -a_ + static_cast<double>(i) * h_;
    xs_[i] = coord;
    ys_[i] = coord;
    zs_[i] = coord;
  }
}

// Accessor (get methods) implementation
int Grid::get_n() const noexcept { return n_points_; }

double Grid::get_a() const noexcept { return a_; }

double Grid::get_h() const noexcept { return h_; }

double Grid::dV() const noexcept { return h_ * h_ * h_; }

const std::vector<double> &Grid::get_xs() const noexcept { return xs_; }

const std::vector<double> &Grid::get_ys() const noexcept { return ys_; }

const std::vector<double> &Grid::get_zs() const noexcept { return zs_; }

size_t Grid::get_total_spatial_points() const noexcept {
  return n_points_ * n_points_ * n_points_;
}

size_t Grid::get_total_points() const noexcept {
  return n_points_ * n_points_ * n_points_ * 2;
}
