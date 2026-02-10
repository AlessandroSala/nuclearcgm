#include "bogoliubov_states.hpp"

BogoliubovStates::BogoliubovStates(int _points)
    : points(_points) {}

auto BogoliubovStates::U() const noexcept
{
    return W(Eigen::seq(0, points - 1), Eigen::all);
}
auto BogoliubovStates::V() const noexcept
{
    return W(Eigen::seq(points, 2 * points - 1), Eigen::all);
}