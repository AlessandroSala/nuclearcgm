#include "operators/angular_momentum.hpp"

Eigen::VectorXd Operators::Jz(const Eigen::VectorXd& x, const Grid& grid) {
    Eigen::VectorXd res(grid.get_total_points());
    int n = grid.get_n();

    
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            for(int k = 0; k < n; ++k) {
                for(int s = 0; s < 2; ++s) {
                    int idx = grid.idx(i, j, k, s);
                    double x = grid.get_xs()[i];
                    double y = grid.get_ys()[j];
                    

                    const int k1_min = std::max(0, k-2);
                    const int k1_max = std::min(n-1, k+2);
                    const int j1_min = std::max(0, j-2);
                    const int j1_max = std::min(n-1, j+2);
                    const int i1_min = std::max(0, i-2);
                    const int i1_max = std::min(n-1, i+2);



                    

                }
            }
        }
    }

}