#include "util/integral.hpp"

double Integral::wholeSpace(const Eigen::VectorXd& f, const Grid& grid) {
    double res = 0.0;
    double hhh = grid.get_h() * grid.get_h() * grid.get_h();
    for (int i = 0; i < f.size(); ++i) {
        res += f(i) * hhh;
    }
    return res;
}