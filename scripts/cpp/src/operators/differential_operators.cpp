#include "operators/differential_operators.hpp"

Eigen::VectorXcd Operators::dv(const Eigen::VectorXcd& psi, const Grid& grid, char dir) {
    Eigen::VectorXcd res(grid.get_total_points());
    for(int i = 0; i < grid.get_n(); ++i) {
        for(int j = 0; j < grid.get_n(); ++j) {
            for(int k = 0; k < grid.get_n(); ++k) {
                for(int s = 0; s < 2; ++s) {
                    int idx = grid.idx(i, j, k, s);
                    res(idx) = Operators::derivative(psi, i, j, k, s, grid, dir);
                }
            }
        }
    }
    return res;

}
double Operators::derivativeNoSpin(const Eigen::VectorXd& psi, int i, int j, int k, const Grid& grid, char axis) {
    int n = grid.get_n();
    double h = grid.get_h();

    auto idx = [&](int ii, int jj, int kk) {
        return axis == 'x' ? grid.idxNoSpin(ii, j, k) :
               axis == 'y' ? grid.idxNoSpin(i, jj, k) :
                             grid.idxNoSpin(i, j, kk);
    };

    int pos = axis == 'x' ? i : axis == 'y' ? j : k;

    if (pos == 0) {
        // Forward difference (2 punti)
        double f0 = psi(idx(pos, pos, pos));
        double f1 = psi(idx(pos + 1, pos + 1, pos + 1));
        return (f1 - f0) / h;
    } else if (pos == 1 || pos == n - 2) {
        // Derivata centrata a 3 punti
        double f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
        double f1 = psi(idx(pos + 1, pos + 1, pos + 1));
        return (f1 - f_1) / (2.0 * h);
    } else if (pos == n - 1) {
        // Backward difference (2 punti)
        double f0 = psi(idx(pos, pos, pos));
        double f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
        return (f0 - f_1) / h;
    } else {
        // Derivata centrata a 5 punti (ordine 4)
        double f_2 = psi(idx(pos - 2, pos - 2, pos - 2));
        double f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
        double f1 = psi(idx(pos + 1, pos + 1, pos + 1));
        double f2 = psi(idx(pos + 2, pos + 2, pos + 2));
        return (-f2 + 8.0*f1 - 8.0*f_1 + f_2) / (12.0 * h);
    }
}
std::complex<double> Operators::derivativeNoSpin(const Eigen::VectorXcd& psi, int i, int j, int k, const Grid& grid, char axis) {
    int n = grid.get_n();
    double h = grid.get_h();

    auto idx = [&](int ii, int jj, int kk) {
        return axis == 'x' ? grid.idxNoSpin(ii, j, k) :
               axis == 'y' ? grid.idxNoSpin(i, jj, k) :
                             grid.idxNoSpin(i, j, kk);
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
std::complex<double> Operators::derivative(const Eigen::VectorXcd& psi, int i, int j, int k, int s, const Grid& grid, char axis) {
    int n = grid.get_n();
    double h = grid.get_h();

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
std::complex<double> Operators::derivative2(
    const Eigen::VectorXcd& psi, int i, int j, int k, int s,
    const Grid& grid, char axis) {
    
    int n = grid.get_n();
    std::complex<double> h = grid.get_h();

    auto idx = [&](int ii, int jj, int kk) {
        return axis == 'x' ? grid.idx(ii, j, k, s) :
               axis == 'y' ? grid.idx(i, jj, k, s) :
                             grid.idx(i, j, kk, s);
    };

    int pos = axis == 'x' ? i : axis == 'y' ? j : k;

    // Estremi: schema a 2 punti (ordine 1)
    if (pos == 0) {
        std::complex<double> f0 = psi(idx(pos, pos, pos));
        std::complex<double> f1 = psi(idx(pos + 1, pos + 1, pos + 1));
        return (f1 - f0) / (h * h);  // Prima approssimazione grezza
    } else if (pos == 1 || pos == n - 2) {
        // Centrata a 3 punti (ordine 2)
        std::complex<double> f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
        std::complex<double> f0 = psi(idx(pos, pos, pos));
        std::complex<double> f1 = psi(idx(pos + 1, pos + 1, pos + 1));
        return (f_1 - 2.0 * f0 + f1) / (h * h);
    } else if (pos == n - 1) {
        std::complex<double> f0 = psi(idx(pos, pos, pos));
        std::complex<double> f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
        return (f0 - f_1) / (h * h);  // Stessa approssimazione rozza in coda
    } else {
        // Centrata a 5 punti (ordine 4)
        std::complex<double> f_2 = psi(idx(pos - 2, pos - 2, pos - 2));
        std::complex<double> f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
        std::complex<double> f0  = psi(idx(pos, pos, pos));
        std::complex<double> f1  = psi(idx(pos + 1, pos + 1, pos + 1));
        std::complex<double> f2  = psi(idx(pos + 2, pos + 2, pos + 2));
        return (-f2 + 16.0*f1 - 30.0*f0 + 16.0*f_1 - f_2) / (12.0 * h * h);
    }
}


