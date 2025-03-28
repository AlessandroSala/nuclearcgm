#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <omp.h>
#include <fstream>

using namespace std;
using namespace Eigen;

const int n = 1000;
const double a = 10.0;
const double h = 2 * a / (n-1);
const double V0 = 57.0;
const double A_val = 16;
const double R = 1.27 * pow(A_val, 1.0/3.0);
const double diff = 0.67;
const double h_bar = 197;
const double m = 939;
const double C = (-2*m/(h_bar*h_bar));
const int max_iter = 5000;
const double tol = 1e-28;
const double omega = 41*pow(A_val, -1.0/3.0)/h_bar;

inline int idx(int i, int j, int k) {
    return i + n * (j + n * k);
}

double pot_osc(double x) {
    return 0.5*(x*x)*m*omega*omega;
}

SparseMatrix<double> matsetup_oscillator(int n, const vector<double>& xs, const vector<double>& ys, const vector<double>& zs) {
    SparseMatrix<double> mat(n, n);
    vector<Triplet<double>> tripletList;

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double val = 0;
            if(i==j)
                val = (pot_osc(xs[i])*C*h*h - 2) / (C * h * h);
            if(i==j-1 || i==j+1)
                val = 1/(C*h*h);

            if (val != 0) {
                #pragma omp critical
                tripletList.emplace_back(i, j, val);
            }
        }
    }
    mat.setFromTriplets(tripletList.begin(), tripletList.end());
    return mat;
}

double find_positive_root(const VectorXd& x, const SparseMatrix<double>& A, const VectorXd& p, const VectorXd& Ax, double xx) {
    VectorXd Ap = A * p;
    double xp = x.dot(p);
    double pp = p.dot(p);
    double pAp = p.dot(Ap);
    double xAp = x.dot(Ap);
    double xAx = x.dot(Ax);

    double a = (pAp) * (xp) - (xAp) * (pp);
    double b = (pAp) * (xx) - (xAx) * (pp);
    double c = (xAp) * (xx) - (xAx) * (xp);

    double delta = b * b - 4 * a * c;
    if (delta < 0) {
        return 0;
    }
    delta = sqrt(delta);
    double lambda1 = (-b + delta) / (2 * a);
    double lambda2 = (-b - delta) / (2 * a);
    if (lambda1 < 0 && lambda2 < 0) {
        return 0;
    }
    else if(lambda1 < 0) return lambda2;
    else if(lambda2 < 0) return lambda1;

    return lambda1 < lambda2 ? lambda1 : lambda2;

}

double compute_beta_FR(const VectorXd& r_new, const VectorXd& r_old) {
    return r_new.dot(r_new) / r_old.dot(r_old);
}

double f(const VectorXd& x, const VectorXd& Ax, double xx) {
    return x.dot(Ax) / xx;
}
VectorXd g(const VectorXd& x, const VectorXd& Ax, double xx) {
    return 2*(Ax - f(x, Ax, xx) * x)/ xx;
}

pair<double, VectorXd> find_eigenpair(const SparseMatrix<double>& A) {
    int N = A.rows();
    VectorXd x = VectorXd::Random(N).normalized();
    for(const auto &e : x) {
        cout << e << endl;
    }
    VectorXd Ax(N);
    VectorXd grad(N);
    VectorXd d(N);
    VectorXd r_new(N);
    VectorXd r_old(N);
    double xx = x.dot(x);
    Ax = A * x;
    grad = g(x, Ax, xx);
    double g0 = grad.dot(grad);
    d = -grad;
    r_new = d;
    
    for (int iter = 0; iter < max_iter; ++iter) {
        r_old = r_new;
        x = x + d*find_positive_root(x, A, d, Ax, xx);
        Ax = A * x;
        xx = x.dot(x);
        grad = g(x, Ax, xx);
        
        if (grad.dot(grad) < tol*g0) break;

        r_new = -grad;
        d = compute_beta_FR(r_new, r_old) * d + r_new;
        //cout << "Iteration " << iter << endl;
    }
    return {f(x, Ax, xx), x};
}


int main() {
    vector<double> xs(n), ys(n), zs(n);
    string path = "output/ho_cpp/";
    for (int i = 0; i < n; ++i) {
        xs[i] = ys[i] = zs[i] = -a + i * h;
    }
    
    SparseMatrix<double> mat = matsetup_oscillator(n, xs, ys, zs);
    cout << "Matrix generated" << endl;

    
    pair<double, VectorXd> eigenpair = find_eigenpair(mat);
    cout << "Smallest eigenvalue (iterative): " << eigenpair.first << endl;
    double e_real = h_bar * omega * 0.5;
    cout << "Real energy: " << e_real << " MeV" << endl;
    cout << "Error: " << ((eigenpair.first)/e_real - 1)*100 << "%" << endl;

    ofstream file(path + "eigenvectors.txt");
    for (int i = 0; i < eigenpair.second.size(); ++i) {
        file << eigenpair.second(i) << endl;
    }
    file.close();

    ofstream file2(path + "x.txt");
    for(const auto &e : xs) {
        file2 << e << endl;
    }
    file2.close();
    //cout << MatrixXd(mat) << endl;
    


    
    return 0;
}
