#include "operators/differential_operators.hpp"

std::complex<double> Operators::dvx(const Eigen::VectorXcd& psi, int i, int j, int k, int s, const Grid& grid) {
    std::complex<double> f1 = 0, f2 = 0, f_1 = 0, f_2 = 0;
    int n = grid.get_n();
    std::complex<double> h = grid.get_h();
    
    if(i < n-1) {
        f1 = psi(grid.idx(i+1, j, k, s));
        if(i < n-2) {
            f2 = psi(grid.idx(i+2, j, k, s));
        }
    }
    if(i > 0) {
        f_1 = psi(grid.idx(i-1, j, k, s));
        if(i > 1) {
            f_2 = psi(grid.idx(i-2, j, k, s));
        }
    }
    return dv(f2, f1, f_2, f_1, h);
}
std::complex<double> Operators::dvy(const Eigen::VectorXcd& psi, int i, int j, int k, int s, const Grid& grid) {
    std::complex<double> f1 = 0, f2 = 0, f_1 = 0, f_2 = 0;
    int n = grid.get_n();
    std::complex<double> h = grid.get_h();
    
    if(j < n-1) {
        f1 = psi(grid.idx(i, j+1, k, s));
        if(j < n-2) {
            f2 = psi(grid.idx(i, j+2, k, s));
        }
    }
    if(j > 0) {
        f_1 = psi(grid.idx(i, j-1, k, s));
        if(j > 1) {
            f_2 = psi(grid.idx(i, j-2, k, s));
        }
    }

    return dv(f2, f1, f_2, f_1, h);
}

std::complex<double> Operators::dvz(const Eigen::VectorXcd& psi, int i, int j, int k, int s, const Grid& grid) {
    std::complex<double> f1 = 0, f2 = 0, f_1 = 0, f_2 = 0;
    int n = grid.get_n();
    std::complex<double> h = grid.get_h();
    
    if(k < n-1) {
        f1 = psi(grid.idx(i, j, k+1, s));
        if(k < n-2) {
            f2 = psi(grid.idx(i, j, k+2, s));
        }
    }
    if(k > 0) {
        f_1 = psi(grid.idx(i, j, k-1, s));
        if(k > 1) {
            f_2 = psi(grid.idx(i, j, k-2, s));
        }
    }
    
    return dv(f2, f1, f_2, f_1, h);
}
std::complex<double> Operators::dv(std::complex<double> f2, std::complex<double> f1, std::complex<double> f_2, std::complex<double> f_1, std::complex<double> h) {
    return (-f2 + 8.0*f1 - 8.0*f_1 + f_2)/(12.0*h);
    //return (f1 - f_1)/(2.0*h);
}
std::complex<double> Operators::derivative_along_axis(const Eigen::VectorXcd& psi, int i, int j, int k, int s, const Grid& grid, char axis) {
    int n = grid.get_n();
    std::complex<double> h = grid.get_h();

    auto idx = [&](int ii, int jj, int kk) {
        return axis == 'x' ? grid.idx(ii, j, k, s) :
               axis == 'y' ? grid.idx(i, jj, k, s) :
                             grid.idx(i, j, kk, s);
    };

    int pos = axis == 'x' ? i : axis == 'y' ? j : k;

    if (pos == 0) {
        // Forward difference (2 punti)
        std::complex<double> f0 = psi(idx(pos, pos, pos));
        std::complex<double> f1 = psi(idx(pos + 1, pos + 1, pos + 1));
        return (f1 - f0) / h;
    } else if (pos == 1 || pos == n - 2) {
        // Derivata centrata a 3 punti
        std::complex<double> f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
        std::complex<double> f1 = psi(idx(pos + 1, pos + 1, pos + 1));
        return (f1 - f_1) / (2.0 * h);
    } else if (pos == n - 1) {
        // Backward difference (2 punti)
        std::complex<double> f0 = psi(idx(pos, pos, pos));
        std::complex<double> f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
        return (f0 - f_1) / h;
    } else {
        // Derivata centrata a 5 punti (ordine 4)
        std::complex<double> f_2 = psi(idx(pos - 2, pos - 2, pos - 2));
        std::complex<double> f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
        std::complex<double> f1 = psi(idx(pos + 1, pos + 1, pos + 1));
        std::complex<double> f2 = psi(idx(pos + 2, pos + 2, pos + 2));
        return (-f2 + 8.0*f1 - 8.0*f_1 + f_2) / (12.0 * h);
    }
}


