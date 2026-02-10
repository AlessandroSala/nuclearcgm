
#pragma once
#include <cstddef> // For size_t
#include <vector>

/**
 * @brief Represents a uniform and symmetric 3D spatial grid.
 *
 * Generates and stores coordinate points along the x, y, z axes
 * in the range [-a, a] with 'n' points per dimension.
 * Also provides a function to map indices (i, j, k, s) to a linear index.
 */
class Grid {
public:
  /**
   * @brief Constructs the 3D grid.
   *
   * @param n Number of points along each dimension (must be >= 2).
   * @param a Grid boundary [fm] (range [-a, a]).
   * @throws std::invalid_argument If n < 2.
   */
  Grid(int n, double a);

  static Grid *getInstance();
  // --- Accessors ---

  /** @brief Returns the number of points per dimension (n). */
  int get_n() const noexcept; // noexcept because it cannot fail

  /** @brief Returns the grid boundary (a). */
  double get_a() const noexcept;

  /** @brief Returns the grid spacing (h). */
  double get_h() const noexcept;

  double dV() const noexcept;

  /** @brief Returns the vector of x-coordinates. */
  const std::vector<double> &get_xs() const noexcept;

  /** @brief Returns the vector of y-coordinates. */
  const std::vector<double> &get_ys() const noexcept;

  /** @brief Returns the vector of z-coordinates. */
  const std::vector<double> &get_zs() const noexcept;

  /** @brief Returns the total number of spatial points (n*n*n). */
  size_t get_total_spatial_points() const noexcept;

  /** @brief Returns the total dimension of the Hilbert space (n*n*n*2). */
  size_t get_total_points() const noexcept;

  // --- Indexing ---

  /**
   * @brief Calculates the linear index for a point (i, j, k) with spin s.
   * The order is: s (fast), i, j, k (slow).
   * Assumes indices i, j, k are in [0, n-1] and s is in [0, 1].
   * No bounds checking is performed for performance.
   *
   * @param i Index along x.
   * @param j Index along y.
   * @param k Index along z.
   * @param s Spin index (0 or 1).
   * @return The corresponding linear index [0, 2*n*n*n - 1].
   */
  inline size_t idx(size_t i, size_t j, size_t k, size_t s) const noexcept;
  inline size_t idxNoSpin(size_t i, size_t j, size_t k) const noexcept;
  // NOTE: inline suggests the compiler insert the code directly
  //       at the call site, useful for small, frequently called functions.
  //       The implementation is placed here for this reason, but could be in
  //       the .cpp.

private:
  int n_points_;           // Number of points per dimension (n)
  double a_;               // Boundary [-a, a]
  double h_;               // Grid spacing
  std::vector<double> xs_; // x-coordinates
  std::vector<double> ys_; // y-coordinates
  std::vector<double> zs_; // z-coordinates
  static Grid *instance;

  /** @brief Helper function to calculate coordinates in the constructor. */
  void initialize_coordinates();
};

inline size_t Grid::idxNoSpin(size_t i, size_t j, size_t k) const noexcept {
  return i + n_points_ * (j + n_points_ * k);
}
inline size_t Grid::idx(size_t i, size_t j, size_t k, size_t s) const noexcept {
  return s + 2 * (i + n_points_ * (j + n_points_ * k));
}
