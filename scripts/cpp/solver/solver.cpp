
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
using namespace Eigen;

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

std::pair<double, VectorXd> find_eigenpair(const SparseMatrix<double>& A, int max_iter = 5000, double tol = 1e-28) {
    int N = A.rows();
    VectorXd x = VectorXd::Random(N).normalized();
    
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

std::pair<double, VectorXd> find_second_eigenpair(const SparseMatrix<double>& A, const VectorXd& c, int max_iter = 5000, double tol = 1e-28) {
    int N = A.rows();
    VectorXd x = VectorXd::Random(N).normalized();
    
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