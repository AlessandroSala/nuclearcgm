
#include "solver.hpp"
#include <chrono>
#include <iostream>
#include <numeric>

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

using namespace Eigen;

double f_constrained(const VectorXd &x, const VectorXd &Ax) {
  return x.dot(Ax);
}
VectorXd g_constrained(const VectorXd &x, const VectorXd &Ax) {
  return Ax - f_constrained(x, Ax) * x;
}

std::pair<double, VectorXd>
find_eigenpair_constrained(const SparseMatrix<double> &A, const VectorXd &X0,
                           int max_iter = 100, double tol = 1e-6) {
  int N = A.rows();
  VectorXd x = X0.normalized();
  VectorXd z(N);
  VectorXd p(N);
  double a, b, c, d, delta, alfa, gamma, l, beta, g0;
  VectorXd Ax(N);
  VectorXd grad(N);
  Ax = A * x;
  l = f_constrained(x, Ax);
  p = g_constrained(x, Ax);
  g0 = p.norm();
  z = A * p;
  for (int iter = 0; iter < max_iter; ++iter) {
    a = z.dot(x);
    b = z.dot(p);
    c = x.dot(p);
    d = p.dot(p);
    delta = pow(l * d - b, 2) - 4 * (b * c - a * d) * (a - l * c);
    alfa = (l * d - b + sqrt(delta)) / (2 * (b * c - a * d));
    gamma = sqrt(1 + 2 * c * alfa + d * pow(alfa, 2));
    l = (l + a * alfa) / (1 + c * alfa); // new
    x = (x + alfa * p) / gamma;
    Ax = (Ax + alfa * z) / gamma;
    grad = Ax - l * x;
    if (grad.norm() < tol * l) {
      break;
    }
    beta = -(grad.dot(z)) / (b);
    p = grad + beta * p;
    z = A * p;
  }
  return {f_constrained(x, Ax), x};
}
typedef std::complex<double> ComplexScalar;
typedef Eigen::Matrix<ComplexScalar, Eigen::Dynamic, Eigen::Dynamic>
    ComplexDenseMatrix;
typedef Eigen::Matrix<ComplexScalar, Eigen::Dynamic, 1> ComplexDenseVector;
typedef Eigen::SparseMatrix<ComplexScalar> ComplexSparseMatrix;

typedef Eigen::MatrixXd DenseMatrix;
typedef Eigen::VectorXd DenseVector;
typedef Eigen::SparseMatrix<double> RealSparseMatrix;

void b_modified_gram_schmidt_complex(ComplexDenseMatrix &V,
                                     const ComplexSparseMatrix &B) {
  if (V.cols() == 0)
    return;

  for (int j = 0; j < V.cols(); ++j) {
    ComplexDenseMatrix BV = B * V;
    ComplexScalar norm_b_sq_complex = V.col(j).adjoint() * BV.col(j);
    double norm_b_sq = std::real(norm_b_sq_complex); // Prendi la parte reale

    if (std::abs(std::imag(norm_b_sq_complex)) > 1e-12 * norm_b_sq) {
      std::cerr << "Warning: Non-real B-norm squared (" << norm_b_sq_complex
                << ") encountered in complex MGS for column " << j << std::endl;
    }

    double norm_b = (norm_b_sq > 1e-24) ? std::sqrt(norm_b_sq) : 0.0;

    if (norm_b < 1e-12) {
      V.col(j).setZero();
      continue;
    }
    V.col(j) /= norm_b;

#pragma omp parallel for
    for (int k = j + 1; k < V.cols(); ++k) {
      ComplexScalar proj = V.col(j).adjoint() * BV.col(k);
      V.col(k) -= proj * V.col(j);
    }
  }
}
void b_modified_gram_schmidt_complex(ComplexDenseMatrix &V,
                                     const RealSparseMatrix &B) {
  if (V.cols() == 0)
    return;

  for (int j = 0; j < V.cols(); ++j) {
    ComplexDenseMatrix BV = B * V;
    ComplexScalar norm_b_sq_complex = V.col(j).adjoint() * BV.col(j);
    double norm_b_sq = std::real(norm_b_sq_complex); // Prendi la parte reale

    if (std::abs(std::imag(norm_b_sq_complex)) > 1e-12 * norm_b_sq) {
      std::cerr << "Warning: Non-real B-norm squared (" << norm_b_sq_complex
                << ") encountered in complex MGS for column " << j << std::endl;
    }

    double norm_b = (norm_b_sq > 1e-24) ? std::sqrt(norm_b_sq) : 0.0;

    if (norm_b < 1e-12) {
      V.col(j).setZero();
      continue;
    }
    V.col(j) /= norm_b;

#pragma omp parallel for
    for (int k = j + 1; k < V.cols(); ++k) {
      ComplexScalar proj = V.col(j).adjoint() * BV.col(k);
      V.col(k) -= proj * V.col(j);
    }
  }
}

std::pair<ComplexDenseMatrix, DenseVector>
rayleighRitz_complex(const ComplexDenseMatrix &V, const ComplexSparseMatrix &A,
                     const RealSparseMatrix &B, int nev) {
  if (V.cols() == 0) {
    std::cerr << "Error: Complex Rayleigh-Ritz called with empty basis V."
              << std::endl;
    return {};
  }

  ComplexDenseMatrix A_proj(V.cols(), V.cols());
  ComplexDenseMatrix B_proj(V.cols(), V.cols());

#pragma omp parallel sections
  {
#pragma omp section
    A_proj.noalias() = V.adjoint() * (A * V);

#pragma omp section
    {
      ComplexDenseMatrix temp = B * V;
      B_proj.noalias() = V.adjoint() * temp;
    }
  }

  Eigen::GeneralizedSelfAdjointEigenSolver<ComplexDenseMatrix> ges;
  ges.compute(A_proj, B_proj);

  if (ges.info() != Eigen::Success) {
    std::cerr << "Error: Complex Rayleigh-Ritz eigenvalue computation failed!"
              << std::endl;
    return {};
  }

  DenseVector eigenvalues_all = ges.eigenvalues();
  ComplexDenseMatrix eigenvectors_proj_all = ges.eigenvectors();

  int num_found = eigenvalues_all.size();
  if (num_found < nev) {
    std::cerr << "Warning: Complex Rayleigh-Ritz found only " << num_found
              << " eigenvalues, requested " << nev << "." << std::endl;
    nev = num_found;
  }
  if (nev <= 0) {
    std::cerr
        << "Error: No eigenvalues found or requested in complex Rayleigh-Ritz."
        << std::endl;
    return {};
  }

  DenseVector lambda_new = eigenvalues_all.head(nev);
  ComplexDenseMatrix C = eigenvectors_proj_all.leftCols(nev);

  return {C, lambda_new};
}

void durMs(const Duration &d) {
  std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(d).count()
            << "ms" << std::endl;
}

std::pair<ComplexDenseMatrix, DenseVector>
gcgm_complex(const ComplexSparseMatrix &A, const RealSparseMatrix &B,
             const ComplexDenseMatrix &X_initial, int nev, double shift = 0.0,
             int max_iter = 100, double tolerance = 1e-8, int cg_steps = 10,
             double cg_tol = 1e-3, bool benchmark = false) {

  auto start = Clock::now();
  auto end = Clock::now();
  std::cout << "GCGM Complex" << std::endl;

  int n = A.rows();
  ComplexSparseMatrix B_complex = B.cast<ComplexScalar>();
  ConjugateGradient<ComplexSparseMatrix, Lower | Upper> cg;
  cg.setMaxIterations(cg_steps);
  cg.setTolerance(cg_tol);

  const int blockSize = std::max(1, nev / 4);
  std::cout << "Using block size " << blockSize << std::endl;

  if (A.rows() != n || A.cols() != n || B.rows() != n || B.cols() != n ||
      X_initial.rows() != n) {
    std::cerr << "Error: Matrix dimensions mismatch (Complex GCGM)."
              << std::endl;
    return {};
  }
  if (X_initial.cols() < nev || nev <= 0) {
    std::cerr << "Error: Initial guess must have at least 'nev' columns, and "
                 "nev > 0 (Complex GCGM)."
              << std::endl;
    return {};
  }

  ComplexDenseMatrix X = X_initial.leftCols(nev);
  ComplexDenseMatrix P = ComplexDenseMatrix::Zero(n, blockSize);
  ComplexDenseMatrix W = ComplexDenseMatrix::Zero(n, blockSize);
  DenseVector Lambda;
  double current_shift = shift;

  b_modified_gram_schmidt_complex(X, B);

  {
    ComplexSparseMatrix A_initial_shifted =
        A + ComplexScalar(current_shift) * B_complex;
    auto rr_init = rayleighRitz_complex(X, A_initial_shifted, B, nev);
    if (rr_init.first.cols() == 0) {
      std::cerr << "Error: Initial Complex Rayleigh-Ritz failed." << std::endl;
      return {};
    }
    X = X * rr_init.first;
    Lambda = rr_init.second -
             DenseVector::Constant(rr_init.second.size(), current_shift);
    b_modified_gram_schmidt_complex(X, B);
  }

  for (int iter = 0; iter < max_iter; ++iter) {

    if (Lambda.size() == nev && Lambda(0) != 0) {
      current_shift = (Lambda(nev - 1) - 100.0 * Lambda(0)) / 99.0;
    }
    std::cout << "Iter: " << iter + 1 << ", Current Shift: " << current_shift
              << std::endl;

    ComplexSparseMatrix A_shifted =
        A + ComplexScalar(current_shift) * B_complex;

    ComplexDenseMatrix BXLambda = B_complex * X * Lambda.asDiagonal();

    cg.compute(A_shifted);
    if (cg.info() != Eigen::Success && iter == 0) {
      std::cerr << "Error: CG compute structure failed (Complex GCGM)."
                << std::endl;
      return {};
    }
    start = Clock::now();
#pragma omp parallel for // Opzionale
    for (int k = 0; k < nev; ++k) {
      W.col(k) = cg.solveWithGuess(BXLambda.col(k), W.col(k)); // W è complesso
    }
    end = Clock::now();
    if (benchmark) {
      std::cout << "Time CG: ";
      durMs(end - start);
    }

    int p_cols = (iter == 0) ? 0 : nev;
    ComplexDenseMatrix V(n, nev + p_cols + nev);
    V.leftCols(nev) = X;
    if (p_cols > 0) {
      V.block(0, nev, n, p_cols) = P;
    }
    V.rightCols(nev) = W;

    start = Clock::now();
    b_modified_gram_schmidt_complex(V, B);
    end = Clock::now();
    if (benchmark) {
      std::cout << "Time B-Ort: ";
      durMs(end - start);
    }

    std::vector<int> keep_cols;
    for (int k = 0; k < V.cols(); ++k) {
      if (V.col(k).squaredNorm() > 1e-20) {
        keep_cols.push_back(k);
      }
    }
    if (keep_cols.size() < nev) {
      std::cerr << "Warning: Complex Basis V rank collapsed below nev ("
                << keep_cols.size() << ") at iteration " << iter << std::endl;
      return {X, Lambda};
    }
    ComplexDenseMatrix V_eff(n, keep_cols.size());
    for (size_t i = 0; i < keep_cols.size(); ++i) {
      V_eff.col(i) = V.col(keep_cols[i]);
    }

    int current_dim = V_eff.cols();
    int rr_nev = std::min(nev, current_dim);

    start = Clock::now();
    auto rr_result = rayleighRitz_complex(V_eff, A_shifted, B, rr_nev);
    end = Clock::now();
    if (benchmark) {
      std::cout << "Time Rayleigh-Ritz: ";
      durMs(end - start);
    }
    if (rr_result.first.cols() == 0) {
      std::cerr << "Error: Complex Rayleigh-Ritz failed in iteration " << iter
                << std::endl;
      return {X, Lambda};
    }

    ComplexDenseMatrix C = rr_result.first;
    DenseVector Lambda_proj = rr_result.second;
    DenseVector Lambda_new =
        Lambda_proj - DenseVector::Constant(Lambda_proj.size(), current_shift);

    ComplexDenseMatrix X_new = V_eff * C;

    ComplexDenseMatrix Residual =
        A * X_new - B_complex * X_new * Lambda_new.asDiagonal();
    double residual_norm = Residual.norm();
    double x_norm = X_new.norm();
    double relative_residual =
        (x_norm > 1e-12) ? (residual_norm / x_norm) : residual_norm;

    std::cout << "Iter: " << iter + 1 << ", Dim(V): " << current_dim
              << ", Rel. Res: " << relative_residual << std::endl;

    if (relative_residual < tolerance) {
      std::cout << "Converged (Complex GCGM) in " << iter + 1 << " iterations."
                << std::endl;
      std::cout << "Eigenvalues: " << Lambda_new.transpose() << std::endl;
      return std::make_pair(X_new, Lambda_new);
    }

    // Wrong logic, P should be conjugate space projection
    P = X_new - X;
    X = X_new;
    Lambda = Lambda_new;
  }

  std::cerr << "Warning: Complex GCGM did not converge within " << max_iter
            << " iterations." << std::endl;
  std::cout << "Eigenvalues: " << Lambda.transpose() << std::endl;
  return {X, Lambda};
}
void recursive_ortho_helper(ComplexDenseMatrix &V, int start, int end);

void recursive_blocked_orthogonalization_from_article(ComplexDenseMatrix &V) {
  if (V.cols() == 0) {
    return;
  }
  recursive_ortho_helper(V, 0, V.cols() - 1);
}

void recursive_ortho_helper(ComplexDenseMatrix &V, int start, int end) {
  int current_block_size = end - start + 1;

  if (current_block_size <= 1) {
    if (current_block_size == 1) {
      double norm = V.col(start).norm();
      if (norm > 1e-12) {
        V.col(start) /= norm;
      } else {
        V.col(start).setZero();
      }
    }
    return;
  }

  int mid = start + (current_block_size / 2);

  recursive_ortho_helper(V, start, mid - 1);

  auto V1 = V.block(0, start, V.rows(), mid - start);
  auto V2 = V.block(0, mid, V.rows(), end - mid + 1);

  for (int i = 0; i < 2; ++i) {
    V2 -= V1 * (V1.adjoint() * V2); // V2 = V2 - V1 * (V1^H * V2)
  }
  recursive_ortho_helper(V, mid, end);
}
void b_modified_gram_schmidt_complex_no_B(ComplexDenseMatrix &V) {
  if (V.cols() == 0)
    return;

  for (int j = 0; j < V.cols(); ++j) {
    ComplexScalar norm_b_sq_complex = V.col(j).adjoint() * V.col(j);
    double norm_b_sq = std::real(norm_b_sq_complex); // Prendi la parte reale

    if (std::abs(std::imag(norm_b_sq_complex)) > 1e-12 * norm_b_sq) {
      std::cerr << "Warning: Non-real B-norm squared (" << norm_b_sq_complex
                << ") encountered in complex MGS for column " << j << std::endl;
    }

    double norm_b = (norm_b_sq > 1e-24) ? std::sqrt(norm_b_sq) : 0.0;

    if (norm_b < 1e-12) {
      V.col(j).setZero();
      continue;
    }
    V.col(j) /= norm_b;

#pragma omp parallel for
    for (int k = j + 1; k < V.cols(); ++k) {
      ComplexScalar proj = V.col(j).adjoint() * V.col(k);
      V.col(k) -= proj * V.col(j);
    }
  }
}

std::pair<ComplexDenseMatrix, DenseVector>
rayleighRitz_complex_no_B(const ComplexDenseMatrix &V,
                          const ComplexSparseMatrix &A, int nev) {
  if (V.cols() == 0) {
    std::cerr << "Error: Complex Rayleigh-Ritz called with empty basis V."
              << std::endl;
    return {};
  }

  ComplexDenseMatrix A_proj(V.cols(), V.cols());
#pragma omp parallel sections
  {
#pragma omp section
    A_proj.noalias() = V.adjoint() * (A * V);
  }

  Eigen::SelfAdjointEigenSolver<ComplexDenseMatrix> ges;
  ges.compute(A_proj);

  if (ges.info() != Eigen::Success) {
    std::cerr << "Error: Complex Rayleigh-Ritz eigenvalue computation failed!"
              << std::endl;
    return {};
  }

  DenseVector eigenvalues_all = ges.eigenvalues();
  ComplexDenseMatrix eigenvectors_proj_all = ges.eigenvectors();

  int num_found = eigenvalues_all.size();
  if (num_found < nev) {
    std::cerr << "Warning: Complex Rayleigh-Ritz found only " << num_found
              << " eigenvalues, requested " << nev << "." << std::endl;
    nev = num_found;
  }
  if (nev <= 0) {
    std::cerr
        << "Error: No eigenvalues found or requested in complex Rayleigh-Ritz."
        << std::endl;
    return {};
  }

  DenseVector lambda_new = eigenvalues_all.head(nev);
  ComplexDenseMatrix C = eigenvectors_proj_all.leftCols(nev);

  return {C, lambda_new};
}

std::pair<ComplexDenseMatrix, DenseVector> gcgm_complex_no_B(
    const ComplexSparseMatrix &A, // A Complessa Hermitiana
    const ComplexDenseMatrix &X_initial, int nev,
    double shift = 0.0, // Shift rimane reale
    int max_iter = 100, double tolerance = 1e-8,
    int cg_steps = 10, // Passi BiCGSTAB (o altro solver complesso)
    double cg_tol = 1e-9, bool benchmark = false, int size = 4) {

  auto start = Clock::now();
  auto end = Clock::now();

  std::cout << "GCGM Complex" << std::endl;
  int maxThreads = omp_get_max_threads();
  std::vector<ConjugateGradient<ComplexSparseMatrix, Lower | Upper>> cgSolvers(
      maxThreads);

  int n = A.rows();

  if (A.rows() != n || A.cols() != n || X_initial.rows() != n) {
    std::cerr << "Error: Matrix dimensions mismatch (Complex GCGM)."
              << std::endl;
    return {};
  }
  if (X_initial.cols() < nev || nev <= 0) {
    std::cerr << "Error: Initial guess must have at least 'nev' columns, and "
                 "nev > 0 (Complex GCGM)."
              << std::endl;
    return {};
  }

  ComplexDenseMatrix X = X_initial.leftCols(nev);
  ComplexDenseMatrix P = ComplexDenseMatrix::Zero(n, nev);
  ComplexDenseMatrix W = ComplexDenseMatrix::Zero(n, nev);
  DenseVector Lambda;
  double current_shift = shift;
  ComplexSparseMatrix Id(n, n);
  Id.setIdentity();

  b_modified_gram_schmidt_complex_no_B(X);
  int converged = 0;

  {
    ComplexSparseMatrix A_initial_shifted =
        A + ComplexScalar(current_shift) * Id;
    auto rr_init = rayleighRitz_complex_no_B(X, A_initial_shifted, nev);
    if (rr_init.first.cols() == 0) {
      std::cerr << "Error: Initial Complex Rayleigh-Ritz failed." << std::endl;
      return {};
    }
    X = X * rr_init.first;
    Lambda = rr_init.second -
             DenseVector::Constant(rr_init.second.size(), current_shift);
    b_modified_gram_schmidt_complex_no_B(X);
  }

  for (int iter = 0; iter < max_iter; ++iter) {

    if (Lambda.size() == nev && Lambda(0) != 0) {
      current_shift = (Lambda(nev - 1) - 100.0 * Lambda(0)) / 99.0;
    }

    std::cout << "Iter: " << iter + 1 << ", Current Shift: " << current_shift
              << " ";

    ComplexSparseMatrix A_shifted = A + ComplexScalar(current_shift) * Id;

    auto smallId = MatrixXd::Identity(Lambda.rows(), Lambda.cols());
    ComplexDenseMatrix BXLambda = X * Lambda.asDiagonal();
    BXLambda += X * current_shift;

    start = Clock::now();

#pragma omp parallel for
    for (int i = 0; i < maxThreads; ++i) {
      cgSolvers[i].setMaxIterations(cg_steps);
      cgSolvers[i].setTolerance(cg_tol);
      cgSolvers[i].compute(A_shifted);
    }

#pragma omp parallel for // Opzionale
    for (int k = 0; k < nev; ++k) {
      int threadId = omp_get_thread_num();

      W.col(k) = cgSolvers[threadId].solveWithGuess(BXLambda.col(k), X.col(k));
    }
    end = Clock::now();
    if (benchmark) {
      std::cout << "Time CG: ";
      durMs(end - start);
    }

    ComplexDenseMatrix V(n, 3 * nev);
    V.leftCols(nev) = X;
    V(seq(0, n), seq(nev, 2 * nev - 1)) = P;
    V.rightCols(nev) = W;

    start = Clock::now();

    b_modified_gram_schmidt_complex_no_B(V);
    end = Clock::now();
    if (benchmark) {
      std::cout << "Time B-Ort: ";
      durMs(end - start);
    }

    std::vector<int> keep_cols;
    for (int k = 0; k < V.cols(); ++k) {
      if (V.col(k).squaredNorm() > 1e-20) {
        keep_cols.push_back(k);
      }
    }
    // if (keep_cols.size() < nev)
    //{
    // std::cerr << "Warning: Complex Basis V rank collapsed below nev (" <<
    // keep_cols.size() << ") at iteration " << iter << std::endl; return {X,
    // Lambda};
    //}
    ComplexDenseMatrix V_eff(n, keep_cols.size());
    for (size_t i = 0; i < keep_cols.size(); ++i) {
      V_eff.col(i) = V.col(keep_cols[i]);
    }

    int current_dim = V_eff.cols();

    int rr_nev = std::min(nev, current_dim);

    start = Clock::now();
    auto rr_result = rayleighRitz_complex_no_B(V_eff, A_shifted, rr_nev);
    end = Clock::now();
    if (benchmark) {
      std::cout << "Time Rayleigh-Ritz: ";
      durMs(end - start);
    }
    if (rr_result.first.cols() == 0) {
      std::cerr << "Error: Complex Rayleigh-Ritz failed in iteration " << iter
                << std::endl;
      return {X, Lambda};
    }

    ComplexDenseMatrix C = rr_result.first;
    DenseVector Lambda_proj = rr_result.second;
    DenseVector Lambda_new =
        Lambda_proj - DenseVector::Constant(Lambda_proj.size(), current_shift);

    ComplexDenseMatrix X_new = V_eff * C;

    ComplexDenseMatrix Residual = A * X_new - X_new * Lambda_new.asDiagonal();

    double residual_norm = Residual.norm();
    double x_norm = X_new.norm();

    double relative_residual =
        (x_norm > 1e-12) ? (residual_norm / x_norm) : residual_norm;

    std::cout << ", Dim(V): " << current_dim
              << ", Rel. Res: " << relative_residual << "\t\r" << std::flush;

    if (relative_residual < tolerance) {
      std::cout << std::endl;
      std::cout << "Converged (Complex GCGM) in " << iter + 1 << " iterations."
                << std::endl;
      std::cout << "Eigenvalues: " << Lambda_new.transpose() << std::endl;
      return std::make_pair(X_new, Lambda_new);
    }
    ComplexDenseMatrix coeff = X.adjoint() * X_new;
    P = X_new - X * coeff;
    X = X_new;
    Lambda = Lambda_new;
  }
  std::cout << std::endl;

  std::cerr << "Warning: Complex GCGM did not converge within " << max_iter
            << " iterations." << std::endl;
  std::cout << "Eigenvalues: " << Lambda.transpose() << std::endl;
  return {X, Lambda};
}

std::pair<ComplexDenseMatrix, DenseVector> gcgm_complex_no_B_lock(
    const ComplexSparseMatrix &A, const ComplexDenseMatrix &X_initial,
    const ComplexDenseMatrix &ConjDir, int nev, double shift = 0.0,
    int max_iter = 100, double tolerance = 1e-8, int cg_steps = 10,
    double cg_tol = 1e-9, bool benchmark = false,
    EigenpairsOrdering ordering = EigenpairsOrdering::MATCH_PREVIOUS) {

  auto start_total = Clock::now();
  auto start_op = Clock::now();
  auto end_op = Clock::now();

  int maxThreads = omp_get_max_threads();
  if (benchmark) {
    std::cout << "GCGM Complex + Lock of converged eigenvectors" << std::endl;
    std::cout << "Using " << maxThreads << " thread(s)." << std::endl;
  }

  int n = A.rows();
  ComplexSparseMatrix Id(n, n);
  Id.setIdentity();
  double current_shift = shift;

  std::vector<Eigen::ConjugateGradient<ComplexSparseMatrix,
                                       Eigen::Lower | Eigen::Upper>>
      cgSolvers(maxThreads);

  ComplexDenseMatrix X0(n, 0);
  ComplexDenseMatrix X = X_initial.leftCols(nev);
  ComplexDenseMatrix P(n, 0);
  if (ConjDir.cols() > 0) {
    if (benchmark)
      std::cout << "Using external conjugate direction" << std::endl;
    P = ConjDir;
  }

  b_modified_gram_schmidt_complex_no_B(X);
  ComplexSparseMatrix A_shifted = A + ComplexScalar(current_shift) * Id;
  auto rr_init = rayleighRitz_complex_no_B(X, A_shifted, X.cols());
  if (rr_init.first.cols() == 0) {
    std::cerr << "Error: Initial Complex Rayleigh-Ritz failed." << std::endl;
    return {};
  }
  X = X * rr_init.first;
  DenseVector Lambda =
      rr_init.second -
      DenseVector::Constant(rr_init.second.size(), current_shift);
  int old_X_cols = X.cols();

  // --- Main GCGM Iteration Loop ---
  for (int iter = 0; iter < max_iter; ++iter) {
    int num_converged = X0.cols();
    int num_active = X.cols();

    if (num_converged == nev) {
      std::cout << std::endl
                << "Converged in " << iter << " iterations." << std::endl;
      DenseVector final_lambda(nev);
      for (int i = 0; i < nev; ++i) {
        final_lambda(i) = (X0.col(i).adjoint() * A * X0.col(i)).value().real();
      }
      std::vector<int> p(nev);
      std::iota(p.begin(), p.end(), 0);
      std::sort(p.begin(), p.end(), [&](int i, int j) {
        return final_lambda(i) < final_lambda(j);
      });
      ComplexDenseMatrix sorted_X0(n, nev);
      DenseVector sorted_lambda(nev);
      for (int i = 0; i < nev; ++i) {
        sorted_X0.col(i) = X0.col(p[i]);
        sorted_lambda(i) = final_lambda(p[i]);
      }
      if (benchmark)
        std::cout << "Final Eigenvalues: " << sorted_lambda.transpose()
                  << std::endl;
      return {sorted_X0, sorted_lambda};
    }

    if (Lambda.size() > 1) {
      current_shift = (Lambda(Lambda.size() - 1) - 100.0 * Lambda(0)) / 99.0;
    }
    A_shifted = A + ComplexScalar(current_shift) * Id;
    if (benchmark)
      std::cout << "Iter: " << iter + 1 << ", Converged: " << num_converged
                << "/" << nev << ", Active: " << num_active
                << ", Shift: " << current_shift << " \r" << std::flush;
    ComplexDenseMatrix W(n, num_active);
    if (num_active > 0) {
      ComplexDenseMatrix BXLambda =
          X * (Lambda.array() + current_shift).matrix().asDiagonal();
      start_op = Clock::now();
#pragma omp parallel for
      for (int i = 0; i < maxThreads; ++i) {
        cgSolvers[i].setMaxIterations(cg_steps);
        cgSolvers[i].setTolerance(cg_tol);
        cgSolvers[i].compute(A_shifted);
      }
#pragma omp parallel for
      for (int k = 0; k < num_active; ++k) {
        int threadId = omp_get_thread_num();
        W.col(k) =
            cgSolvers[threadId].solveWithGuess(BXLambda.col(k), X.col(k));
      }
      end_op = Clock::now();
    }

    int p_cols = P.cols();
    ComplexDenseMatrix V(n, num_converged + num_active + p_cols + num_active);
    V.leftCols(num_converged) = X0;
    V.block(0, num_converged, n, num_active) = X;
    if (p_cols > 0)
      V.block(0, num_converged + num_active, n, p_cols) = P;
    if (num_active > 0)
      V.block(0, num_converged + num_active + p_cols, n, num_active) = W;
    b_modified_gram_schmidt_complex_no_B(V);
    std::vector<int> keep_cols;
    for (int k = 0; k < V.cols(); ++k) {
      if (V.col(k).squaredNorm() > 1e-20)
        keep_cols.push_back(k);
    }
    ComplexDenseMatrix V_eff(n, keep_cols.size());
    for (size_t i = 0; i < keep_cols.size(); ++i) {
      V_eff.col(i) = V.col(keep_cols[i]);
    }

    ComplexDenseMatrix AV_eff = A_shifted * V_eff;
    ComplexDenseMatrix A_proj = V_eff.adjoint() * AV_eff;
    Eigen::SelfAdjointEigenSolver<ComplexDenseMatrix> es(A_proj);
    if (es.info() != Eigen::Success) {
      std::cerr << "\nError: Complex Rayleigh-Ritz eigensolver failed."
                << std::endl;
      return {X0, DenseVector()};
    }
    int rr_nev = std::min(nev, (int)V_eff.cols());
    ComplexDenseMatrix C = es.eigenvectors().leftCols(rr_nev);
    DenseVector Lambda_proj = es.eigenvalues().head(rr_nev);
    ComplexDenseMatrix X_new_full = V_eff * C;
    ComplexDenseMatrix X_new_red = X_new_full.leftCols(X.cols());

    int new_p_cols = 0;
    if (iter == max_iter - 1) { // Do not update P if we are at the last iter
      new_p_cols = X.cols();
      if (new_p_cols > 0) {
        ComplexDenseMatrix tmp;
        tmp.noalias() = X.adjoint() * X_new_red;
        P = X_new_red - X * tmp;
      }
    }
    if (new_p_cols == 0) {
      P.resize(n, 0);
    }

    DenseVector Lambda_new =
        Lambda_proj - DenseVector::Constant(Lambda_proj.size(), current_shift);

    ComplexDenseMatrix Residuals =
        A * X_new_full - X_new_full * Lambda_new.asDiagonal();

    ComplexDenseMatrix updated_X0 = X_new_full.leftCols(num_converged);

    std::vector<int> newly_converged_indices;
    std::vector<int> next_active_indices;
    for (int i = num_converged; i < X_new_full.cols(); ++i) {
      if (Residuals.col(i).norm() < tolerance &&
          (num_converged + newly_converged_indices.size() < nev)) {
        newly_converged_indices.push_back(i);
      } else {
        next_active_indices.push_back(i);
      }
    }

    // Build the next converged set: old (updated) + new
    X0.resize(n, num_converged + newly_converged_indices.size());
    X0.leftCols(num_converged) = updated_X0;
    for (size_t i = 0; i < newly_converged_indices.size(); ++i) {
      X0.col(num_converged + i) = X_new_full.col(newly_converged_indices[i]);
    }

    // Build the next active set from the remaining candidates
    X.resize(n, next_active_indices.size());
    Lambda.resize(next_active_indices.size());
    for (size_t i = 0; i < next_active_indices.size(); ++i) {
      X.col(i) = X_new_full.col(next_active_indices[i]);
      Lambda(i) = Lambda_new(next_active_indices[i]);
    }
  }

  if (benchmark)
    std::cout << std::endl;
  if (X0.cols() == nev) {
    if (benchmark)
      std::cout << "Converged in " << max_iter << " iterations." << std::endl;
  } else {
    if (benchmark)
      std::cerr << "Warning: Corrected GCGM did not converge within "
                << max_iter << " iterations." << std::endl;
  }
  int num_converged_final = X0.cols();
  int num_needed_from_active = nev - num_converged_final;
  int num_active_to_take = std::min((int)X.cols(), num_needed_from_active);

  ComplexDenseMatrix final_X = ComplexDenseMatrix::Zero(n, nev);
  final_X.leftCols(num_converged_final) = X0;
  if (num_active_to_take > 0) {
    final_X.block(0, num_converged_final, n, num_active_to_take) =
        X.leftCols(num_active_to_take);
  }

  DenseVector final_lambda = DenseVector::Zero(nev);
  for (int i = 0; i < num_converged_final; ++i) {
    final_lambda(i) = (X0.col(i).adjoint() * A * X0.col(i)).value().real();
  }
  if (num_active_to_take > 0) {
    final_lambda.segment(num_converged_final, num_active_to_take) =
        Lambda.head(num_active_to_take);
  }

  ComplexDenseMatrix sorted_X(n, nev);
  DenseVector sorted_lambda(nev);

  if (ordering == EigenpairsOrdering::MATCH_PREVIOUS) {

    Eigen::MatrixXd overlap_matrix(nev, nev);
    Eigen::MatrixXcd complex_overlaps(nev, nev); // Keep phases for later

    for (int i = 0; i < nev; ++i) {
      for (int k = 0; k < nev; ++k) {
        std::complex<double> ov = X_initial.col(i).dot(final_X.col(k));
        complex_overlaps(i, k) = ov;
        overlap_matrix(i, k) = std::abs(ov);
      }
    }

    std::vector<bool> assigned_source(nev, false);
    std::vector<bool> assigned_target(nev, false);

    for (int count = 0; count < nev; ++count) {
      double max_val = -1.0;
      int best_target = -1; // Index in X_initial
      int best_source = -1; // Index in final_X

      for (int i = 0; i < nev; ++i) {
        if (assigned_target[i])
          continue;
        for (int k = 0; k < nev; ++k) {
          if (assigned_source[k])
            continue;
          if (overlap_matrix(i, k) > max_val) {
            max_val = overlap_matrix(i, k);
            best_target = i;
            best_source = k;
          }
        }
      }

      if (best_target != -1 && best_source != -1) {
        std::complex<double> ov = complex_overlaps(best_target, best_source);
        std::complex<double> phase =
            (std::abs(ov) > 1e-12) ? (ov / std::abs(ov)) : 1.0;

        sorted_X.col(best_target) = final_X.col(best_source) * phase;
        sorted_lambda(best_target) = final_lambda(best_source);

        assigned_target[best_target] = true;
        assigned_source[best_source] = true;
      } else {
        for (int i = 0; i < nev; ++i)
          if (!assigned_target[i]) {
            for (int k = 0; k < nev; ++k)
              if (!assigned_source[k]) {
                sorted_X.col(i) = final_X.col(k);
                sorted_lambda(i) = final_lambda(k);
                assigned_target[i] = true;
                assigned_source[k] = true;
                break;
              }
          }
      }
    }
  } else {
    int num_converged_final = X0.cols();
    int num_needed_from_active = nev - num_converged_final;
    int num_active_to_take = std::min((int)X.cols(), num_needed_from_active);
    ComplexDenseMatrix final_X = ComplexDenseMatrix::Zero(n, nev);

    final_X.leftCols(num_converged_final) = X0;

    if (num_active_to_take > 0) {

      final_X.block(0, num_converged_final, n, num_active_to_take) =

          X.leftCols(num_active_to_take);
    }

    DenseVector final_lambda = DenseVector::Zero(nev);

    for (int i = 0; i < num_converged_final; ++i) {
      final_lambda(i) = (X0.col(i).adjoint() * A * X0.col(i)).value().real();
    }

    if (num_active_to_take > 0) {
      final_lambda.segment(num_converged_final, num_active_to_take) =
          Lambda.head(num_active_to_take);
    }

    std::vector<int> indices(nev);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](int i, int j) { return final_lambda(i) < final_lambda(j); });

    for (int k = 0; k < nev; ++k) {
      sorted_lambda(k) = final_lambda(indices[k]);
      sorted_X.col(k) = final_X.col(indices[k]);
    }
  }
  // std::cout << sorted_lambda.transpose() << std::endl;

  return {sorted_X, sorted_lambda};
}
