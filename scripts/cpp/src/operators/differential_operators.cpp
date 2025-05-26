#include "operators/differential_operators.hpp"

Eigen::VectorXd Operators::dvNoSpin(const Eigen::VectorXd& psi, const Grid& grid, char dir) {
    Eigen::VectorXd res(grid.get_total_spatial_points());
    for(int i = 0; i < grid.get_n(); ++i) {
        for(int j = 0; j < grid.get_n(); ++j) {
            for(int k = 0; k < grid.get_n(); ++k) {
                int idx = grid.idxNoSpin(i, j, k);
                res(idx) = Operators::derivativeNoSpin(psi, i, j, k, grid, dir);
            }
        }
    }
    return res;

}
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
Eigen::Matrix<double, Eigen::Dynamic, 3> Operators::gradNoSpin(const Eigen::VectorXd& vec, const Grid& grid) {
    Eigen::Matrix<double, -1, 3> res(vec.rows(), 3);
    res.setZero();
    auto dx = dvNoSpin(vec, grid, 'x');
    auto dy = dvNoSpin(vec, grid, 'y');
    auto dz = dvNoSpin(vec, grid, 'z');
    res.col(0) = dx;
    res.col(1) = dy;
    res.col(2) = dz;
    return res;
}

Eigen::VectorXcd Operators::divNoSpin(const Eigen::MatrixXcd& J, const Grid& grid) {
    int n = grid.get_n();
    Eigen::VectorXcd res(grid.get_total_spatial_points());
    res.setZero();
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            for(int k = 0; k < n; ++k) {
                Eigen::VectorXcd Jx = J.col(0);
                Eigen::VectorXcd Jy = J.col(1);
                Eigen::VectorXcd Jz = J.col(2);

                res(grid.idxNoSpin(i, j, k)) += derivativeNoSpin(Jx, i, j, k, grid, 'x');
                res(grid.idxNoSpin(i, j, k)) += derivativeNoSpin(Jy, i, j, k, grid, 'y');
                res(grid.idxNoSpin(i, j, k)) += derivativeNoSpin(Jz, i, j, k, grid, 'z');
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


