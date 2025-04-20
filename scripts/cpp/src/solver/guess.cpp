
#include "guess.hpp"
ComplexDenseMatrix gaussian_guess(const Grid& grid, int nev, double a) {
    int n = grid.get_n();  
    
    ComplexDenseMatrix guess(grid.get_total_points(), nev);
    for(int ev = 0; ev < nev; ++ev) {
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            for(int k = 0; k < n; ++k) {
                double r = grid.get_xs()[i]*grid.get_xs()[i] + grid.get_ys()[j]*grid.get_ys()[j] + grid.get_zs()[k]*grid.get_zs()[k];
                r = (sqrt(r))/(5*a);
                for(int s = 0; s < 2; ++s) {
                    guess(grid.idx(i, j, k, s), ev) = pow(r, ev)*ComplexScalar(exp(-pow(r, ev)), 0);
                }
            }
        }
    }
    }
    for (int ev = 0; ev < nev; ++ev) {
        guess.col(ev).normalize();
    }
    return guess;
}