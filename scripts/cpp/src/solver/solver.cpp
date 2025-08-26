
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
// === Typedefs per la versione Complessa ===
typedef std::complex<double> ComplexScalar;
typedef Eigen::Matrix<ComplexScalar, Eigen::Dynamic, Eigen::Dynamic>
    ComplexDenseMatrix;
typedef Eigen::Matrix<ComplexScalar, Eigen::Dynamic, 1> ComplexDenseVector;
typedef Eigen::SparseMatrix<ComplexScalar> ComplexSparseMatrix;

// === Typedefs per la versione Reale (come riferimento) ===
typedef Eigen::MatrixXd DenseMatrix;
typedef Eigen::VectorXd DenseVector;
typedef Eigen::SparseMatrix<double> RealSparseMatrix;

/**
 * @brief Ortogonalizzazione di Gram-Schmidt Modificata Complessa rispetto a B
 * reale. Modifica V sul posto. Assume B Reale Simmetrica Definitia Positiva
 * (SPD).
 *
 * @param V Matrice complessa le cui colonne verranno B-ortogonalizzate.
 * @param B Matrice reale SPD che definisce il prodotto scalare.
 */

void b_modified_gram_schmidt_complex(ComplexDenseMatrix &V,
                                     const RealSparseMatrix &B) {
  if (V.cols() == 0)
    return;

  for (int j = 0; j < V.cols(); ++j) {
    // Normalizza la colonna j: norm_b_sq = V_j^adjoint * B * V_j (dovrebbe
    // essere reale) Nota: B è reale, quindi B*V.col(j) produce un vettore
    // complesso.
    //       V.col(j).adjoint() * (risultato complesso) produce uno scalare
    //       complesso. Ma il risultato V†BV dovrebbe essere reale se B è SPD.
    ComplexDenseMatrix BV = B * V;
    ComplexScalar norm_b_sq_complex = V.col(j).adjoint() * BV.col(j);
    double norm_b_sq = std::real(norm_b_sq_complex); // Prendi la parte reale

    // Aggiungi un controllo sulla parte immaginaria per sicurezza numerica
    if (std::abs(std::imag(norm_b_sq_complex)) > 1e-12 * norm_b_sq) {
      std::cerr << "Warning: Non-real B-norm squared (" << norm_b_sq_complex
                << ") encountered in complex MGS for column " << j << std::endl;
    }

    double norm_b = (norm_b_sq > 1e-24) ? std::sqrt(norm_b_sq) : 0.0;

    if (norm_b < 1e-12) {
      V.col(j).setZero();
      // std::cerr << "Warning: Vector " << j << " has near-zero B-norm in
      // complex MGS." << std::endl;
      continue;
    }
    V.col(j) /= norm_b; // norm_b è reale

// Rendi le colonne successive (k > j) ortogonali alla colonna j
#pragma omp parallel for // Opzionale
    for (int k = j + 1; k < V.cols(); ++k) {
      // proj = V_j^adjoint * B * V_k (prodotto scalare complesso)
      ComplexScalar proj = V.col(j).adjoint() * BV.col(k);
      V.col(k) -= proj * V.col(j); // proj è complesso
    }
  }
}

/**
 * @brief Procedura di Rayleigh-Ritz Complessa per Ax = lambda Bx (A Hermitiana,
 * B Reale SPD).
 *
 * @param V Matrice complessa la cui base (colonne) Spannano il sottospazio.
 * DEVE essere B-ortonormale.
 * @param A Matrice Complessa Hermitiana A.
 * @param B Matrice Reale SPD B.
 * @param nev Numero di autovalori/autovettori desiderati.
 * @return std::pair<ComplexDenseMatrix, DenseVector> Pair contenente:
 * - Matrice C complessa dei coefficienti (colonne sono autovettori proiettati).
 * - Vettore Lambda reale dei corrispondenti autovalori.
 */
std::pair<ComplexDenseMatrix, DenseVector>
rayleighRitz_complex(const ComplexDenseMatrix &V,
                     const ComplexSparseMatrix &A, // A è complessa Hermitiana
                     const RealSparseMatrix &B,    // B è reale SPD
                     int nev) {
  if (V.cols() == 0) {
    std::cerr << "Error: Complex Rayleigh-Ritz called with empty basis V."
              << std::endl;
    return {};
  }

  // 1. Proietta le matrici: A_proj = V^adjoint * A * V, B_proj = V^adjoint * B
  // * V
  ComplexDenseMatrix A_proj(V.cols(), V.cols());
  ComplexDenseMatrix B_proj(V.cols(), V.cols());
  // A_proj = V.adjoint() * A * V;
  // B_proj = V.adjoint() * B * V; // B_proj è Hermitiana (vicina a Identità)

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

  // Debug check: Verifica che B_proj sia vicina all'identità
  // double b_proj_identity_diff = (B_proj -
  // ComplexDenseMatrix::Identity(V.cols(), V.cols())).norm(); if
  // (b_proj_identity_diff > 1e-9) {
  //      std::cerr << "Warning: B_proj deviation from Identity: " <<
  //      b_proj_identity_diff << std::endl;
  // }

  // 2. Risolvi il problema proiettato: A_proj * C = lambda * B_proj * C
  //    Usiamo GeneralizedSelfAdjointEigenSolver per matrici complesse
  Eigen::GeneralizedSelfAdjointEigenSolver<ComplexDenseMatrix> ges;
  ges.compute(A_proj, B_proj); // A_proj Hermitiana, B_proj Hermitiana Def. Pos.

  if (ges.info() != Eigen::Success) {
    std::cerr << "Error: Complex Rayleigh-Ritz eigenvalue computation failed!"
              << std::endl;
    return {};
  }

  DenseVector eigenvalues_all = ges.eigenvalues(); // Gli autovalori sono REALI
  ComplexDenseMatrix eigenvectors_proj_all =
      ges.eigenvectors(); // Gli autovettori sono COMPLESSI

  // 3. Seleziona i 'nev' più piccoli
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

/**
 * @brief Algoritmo GCGM Complesso per Ax = lambda Bx (A Hermitiana, B Reale
 * SPD).
 *
 * @param A Matrice Complessa Hermitiana A.
 * @param B Matrice Reale SPD B.
 * @param X_initial Stima iniziale Complessa per gli autovettori.
 * @param nev Numero di autocoppie da calcolare.
 * @param shift Shift iniziale (reale).
 * @param max_iter Max iterazioni GCGM.
 * @param tolerance Tolleranza sul residuo relativo.
 * @param cg_steps Numero di passi del solver iterativo (es. BiCGSTAB) per W.
 * @return std::pair<ComplexDenseMatrix, DenseVector> Pair contenente:
 * - Matrice X complessa degli autovettori.
 * - Vettore Lambda reale degli autovalori.
 */
std::pair<ComplexDenseMatrix, DenseVector>
gcgm_complex(const ComplexSparseMatrix &A, // A Complessa Hermitiana
             const RealSparseMatrix &B,    // B Reale SPD
             const ComplexDenseMatrix &X_initial, int nev,
             double shift = 0.0, // Shift rimane reale
             int max_iter = 100, double tolerance = 1e-8,
             int cg_steps = 10, // Passi BiCGSTAB (o altro solver complesso)
             double cg_tol = 1e-3, bool benchmark = false) {

  auto start = Clock::now();
  auto end = Clock::now();
  std::cout << "GCGM Complex" << std::endl;
  // Codice per info OpenMP/thread (invariato)
  // ... (come nella versione originale) ...

  int n = A.rows();
  ComplexSparseMatrix B_complex = B.cast<ComplexScalar>();
  ConjugateGradient<ComplexSparseMatrix, Lower | Upper> cg;
  cg.setMaxIterations(cg_steps);
  cg.setTolerance(cg_tol);

  // Block size definition
  const int blockSize = std::max(1, nev / 4);
  std::cout << "Using block size " << blockSize << std::endl;

  // --- Validazione Input --- (controlli dimensioni simili)
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

  // --- Inizializzazione Complessa ---
  ComplexDenseMatrix X = X_initial.leftCols(nev);
  ComplexDenseMatrix P = ComplexDenseMatrix::Zero(n, blockSize);
  ComplexDenseMatrix W = ComplexDenseMatrix::Zero(n, blockSize);
  DenseVector Lambda;           // Autovalori reali
  double current_shift = shift; // Shift reale

  b_modified_gram_schmidt_complex(X, B); // Rendi X B-ortonormale

  // Rayleigh-Ritz iniziale
  {
    ComplexSparseMatrix A_initial_shifted =
        A + ComplexScalar(current_shift) * B_complex; // Shift A
    auto rr_init = rayleighRitz_complex(X, A_initial_shifted, B, nev);
    if (rr_init.first.cols() == 0) {
      std::cerr << "Error: Initial Complex Rayleigh-Ritz failed." << std::endl;
      return {};
    }
    X = X * rr_init.first; // X = X * C (complesso)
    Lambda =
        rr_init.second - DenseVector::Constant(rr_init.second.size(),
                                               current_shift); // Lambda reale
    b_modified_gram_schmidt_complex(X, B); // Ri-ortogonalizza
  }

  // --- Iterazioni GCGM Complesso ---
  for (int iter = 0; iter < max_iter; ++iter) {

    // 1. Genera W (approx. A_shifted * W = B * X * diag(Lambda))
    if (Lambda.size() == nev && Lambda(0) != 0) {
      // Strategia shift (invariata, usa Lambda reale)
      current_shift = (Lambda(nev - 1) - 100.0 * Lambda(0)) / 99.0;
      // Controlli shift
    }
    std::cout << "Iter: " << iter + 1 << ", Current Shift: " << current_shift
              << std::endl;

    // A_shifted è Complessa Hermitiana (potenzialmente indefinita)
    ComplexSparseMatrix A_shifted =
        A + ComplexScalar(current_shift) * B_complex;

    // BXLambda è Complessa
    ComplexDenseMatrix BXLambda = B_complex * X * Lambda.asDiagonal();

    // Usa BiCGSTAB (o altro solver iterativo complesso) invece di CG
    cg.compute(A_shifted); // Analizza pattern
    // if(iter > 0) cg.setMaxIterations(std::min(cg_steps, 10));
    // std::cout << "Using " << cg.maxIterations() << " iterations" <<
    // std::endl;

    if (cg.info() != Eigen::Success && iter == 0) {
      std::cerr << "Error: CG compute structure failed (Complex GCGM)."
                << std::endl;
      // Questo errore è meno probabile con BiCGSTAB che con CG
      return {};
    }
    start = Clock::now();
#pragma omp parallel for // Opzionale
    for (int k = 0; k < nev; ++k) {
      // Risolvi A_shifted * w_k = (BXLambda)_k
      W.col(k) = cg.solveWithGuess(BXLambda.col(k), W.col(k)); // W è complesso
      // Controllo errore solve opzionale
    }
    end = Clock::now();
    if (benchmark) {
      std::cout << "Time CG: ";
      durMs(end - start);
    }

    // 2. Costruisci V = [X, P, W] (tutte complesse)
    int p_cols = (iter == 0) ? 0 : nev;
    ComplexDenseMatrix V(n, nev + p_cols + nev);
    V.leftCols(nev) = X;
    if (p_cols > 0) {
      V.block(0, nev, n, p_cols) = P;
    }
    V.rightCols(nev) = W;

    start = Clock::now();
    // 3. B-Ortogonalizza V (complessa)
    b_modified_gram_schmidt_complex(V, B);
    end = Clock::now();
    if (benchmark) {
      std::cout << "Time B-Ort: ";
      durMs(end - start);
    }

    // --- Gestione Dipendenze Lineari (come prima, ma su V complessa) ---
    std::vector<int> keep_cols;
    for (int k = 0; k < V.cols(); ++k) {
      // Usa squaredNorm() che funziona per complessi (è ||v||^2 = v†v)
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

    // 4. Rayleigh-Ritz su V_eff (complessa)
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

    ComplexDenseMatrix C = rr_result.first;     // Coefficienti complessi
    DenseVector Lambda_proj = rr_result.second; // Autovalori reali shftati
    DenseVector Lambda_new =
        Lambda_proj - DenseVector::Constant(Lambda_proj.size(),
                                            current_shift); // Autovalori reali

    ComplexDenseMatrix X_new = V_eff * C; // Autovettori complessi

    // 5. Verifica Convergenza (Residuo complesso, norma reale)
    ComplexDenseMatrix Residual =
        A * X_new - B_complex * X_new * Lambda_new.asDiagonal();
    double residual_norm = Residual.norm(); // Norma di Frobenius (reale)
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

    // 6. Aggiorna P (complesso)
    P = X_new - X;
    // Eventuale ri-ortogonalizzazione di P

    // 7. Aggiorna X (complesso) e Lambda (reale)
    X = X_new;
    Lambda = Lambda_new;
    // Eventuale ri-ortogonalizzazione di X
    // if (iter % 5 == 0) b_modified_gram_schmidt_complex(X, B);

  } // Fine ciclo iterazioni

  std::cerr << "Warning: Complex GCGM did not converge within " << max_iter
            << " iterations." << std::endl;
  std::cout << "Eigenvalues: " << Lambda.transpose() << std::endl;
  return {X, Lambda};
}
void recursive_ortho_helper(ComplexDenseMatrix &V, int start, int end);

/**
 * @brief Implements the Recursive Blocked Orthogonalization from the article
 * (Algorithm 4).
 *
 * This function serves as a user-friendly wrapper for the main recursive
 * implementation. It is adapted for the standard L2 inner product to match the
 * context of the provided code snippet (b_modified_gram_schmidt_complex_no_B).
 *
 * @param V The matrix whose columns will be orthonormalized in place.
 */
void recursive_blocked_orthogonalization_from_article(ComplexDenseMatrix &V) {
  if (V.cols() == 0) {
    return;
  }
  // Initial call to the recursive helper function on the entire matrix
  recursive_ortho_helper(V, 0, V.cols() - 1);
}

/**
 * @brief Helper function implementing the core logic of Algorithm 4.
 *
 * @param V The matrix being operated on.
 * @param start The starting column index of the current block.
 * @param end The ending column index of the current block.
 */
void recursive_ortho_helper(ComplexDenseMatrix &V, int start, int end) {
  int current_block_size = end - start + 1;

  // Base Case: If the block contains one or zero vectors, just normalize it.
  // This corresponds to lines 3-9 in Algorithm 4.
  if (current_block_size <= 1) {
    if (current_block_size == 1) {
      double norm = V.col(start).norm();
      if (norm > 1e-12) {
        V.col(start) /= norm;
      } else {
        V.col(start).setZero();
      }
    }
    return; // Return if block is size 1 or empty
  }

  // Recursive Step: Divide the current block into two halves.
  // Corresponds to line 2 in Algorithm 4.
  int mid = start + (current_block_size / 2);

  // 1. Orthonormalize the first half recursively.
  // Corresponds to line 11 in Algorithm 4.
  recursive_ortho_helper(V, start, mid - 1);

  // 2. Make the second half orthogonal to the first half using a block
  // operation. This is the key step (line 16 in Algorithm 4) that leverages
  // Level-3 BLAS.
  auto V1 = V.block(0, start, V.rows(), mid - start);
  auto V2 = V.block(0, mid, V.rows(), end - mid + 1);

  // The paper suggests repeating this step for stability
  // (re-orthogonalization). We will perform it twice, a common practice known
  // as CGS2.
  for (int i = 0; i < 2; ++i) {
    V2 -= V1 * (V1.adjoint() * V2); // V2 = V2 - V1 * (V1^H * V2)
  }

  // 3. Orthonormalize the now-modified second half recursively.
  // Corresponds to line 17 in Algorithm 4.
  recursive_ortho_helper(V, mid, end);
}
void b_modified_gram_schmidt_complex_no_B(ComplexDenseMatrix &V) {
  if (V.cols() == 0)
    return;

  for (int j = 0; j < V.cols(); ++j) {
    // Normalizza la colonna j: norm_b_sq = V_j^adjoint * B * V_j (dovrebbe
    // essere reale) Nota: B è reale, quindi B*V.col(j) produce un vettore
    // complesso.
    //       V.col(j).adjoint() * (risultato complesso) produce uno scalare
    //       complesso. Ma il risultato V†BV dovrebbe essere reale se B è SPD.
    ComplexScalar norm_b_sq_complex = V.col(j).adjoint() * V.col(j);
    double norm_b_sq = std::real(norm_b_sq_complex); // Prendi la parte reale

    // Aggiungi un controllo sulla parte immaginaria per sicurezza numerica
    if (std::abs(std::imag(norm_b_sq_complex)) > 1e-12 * norm_b_sq) {
      std::cerr << "Warning: Non-real B-norm squared (" << norm_b_sq_complex
                << ") encountered in complex MGS for column " << j << std::endl;
    }

    double norm_b = (norm_b_sq > 1e-24) ? std::sqrt(norm_b_sq) : 0.0;

    if (norm_b < 1e-12) {
      V.col(j).setZero();
      // std::cerr << "Warning: Vector " << j << " has near-zero B-norm in
      // complex MGS." << std::endl;
      continue;
    }
    V.col(j) /= norm_b; // norm_b è reale

// Rendi le colonne successive (k > j) ortogonali alla colonna j
#pragma omp parallel for // Opzionale
    for (int k = j + 1; k < V.cols(); ++k) {
      // proj = V_j^adjoint * B * V_k (prodotto scalare complesso)
      ComplexScalar proj = V.col(j).adjoint() * V.col(k);
      V.col(k) -= proj * V.col(j); // proj è complesso
    }
  }
}

/**
 * @brief Procedura di Rayleigh-Ritz Complessa per Ax = lambda Bx (A Hermitiana,
 * B Reale SPD).
 *
 * @param V Matrice complessa la cui base (colonne) Spannano il sottospazio.
 * DEVE essere B-ortonormale.
 * @param A Matrice Complessa Hermitiana A.
 * @param B Matrice Reale SPD B.
 * @param nev Numero di autovalori/autovettori desiderati.
 * @return std::pair<ComplexDenseMatrix, DenseVector> Pair contenente:
 * - Matrice C complessa dei coefficienti (colonne sono autovettori proiettati).
 * - Vettore Lambda reale dei corrispondenti autovalori.
 */
std::pair<ComplexDenseMatrix, DenseVector> rayleighRitz_complex_no_B(
    const ComplexDenseMatrix &V,
    const ComplexSparseMatrix &A, // A è complessa Hermitiana
    int nev) {
  if (V.cols() == 0) {
    std::cerr << "Error: Complex Rayleigh-Ritz called with empty basis V."
              << std::endl;
    return {};
  }

  // 1. Proietta le matrici: A_proj = V^adjoint * A * V, B_proj = V^adjoint * B
  // * V
  ComplexDenseMatrix A_proj(V.cols(), V.cols());
  // A_proj = V.adjoint() * A * V;
  // B_proj = V.adjoint() * B * V; // B_proj è Hermitiana (vicina a Identità)

#pragma omp parallel sections
  {
#pragma omp section
    A_proj.noalias() = V.adjoint() * (A * V);
  }

  // Debug check: Verifica che B_proj sia vicina all'identità
  // double b_proj_identity_diff = (B_proj -
  // ComplexDenseMatrix::Identity(V.cols(), V.cols())).norm(); if
  // (b_proj_identity_diff > 1e-9) {
  //      std::cerr << "Warning: B_proj deviation from Identity: " <<
  //      b_proj_identity_diff << std::endl;
  // }

  // 2. Risolvi il problema proiettato: A_proj * C = lambda * B_proj * C
  //    Usiamo GeneralizedSelfAdjointEigenSolver per matrici complesse
  Eigen::SelfAdjointEigenSolver<ComplexDenseMatrix> ges;
  ges.compute(A_proj); // A_proj Hermitiana, B_proj Hermitiana Def. Pos.

  if (ges.info() != Eigen::Success) {
    std::cerr << "Error: Complex Rayleigh-Ritz eigenvalue computation failed!"
              << std::endl;
    return {};
  }

  DenseVector eigenvalues_all = ges.eigenvalues(); // Gli autovalori sono REALI
  ComplexDenseMatrix eigenvectors_proj_all =
      ges.eigenvectors(); // Gli autovettori sono COMPLESSI

  // 3. Seleziona i 'nev' più piccoli
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

/**
 * @brief Algoritmo GCGM Complesso per Ax = lambda Bx (A Hermitiana, B Reale
 * SPD).
 *
 * @param A Matrice Complessa Hermitiana A.
 * @param B Matrice Reale SPD B.
 * @param X_initial Stima iniziale Complessa per gli autovettori.
 * @param nev Numero di autocoppie da calcolare.
 * @param shift Shift iniziale (reale).
 * @param max_iter Max iterazioni GCGM.
 * @param tolerance Tolleranza sul residuo relativo.
 * @param cg_steps Numero di passi del solver iterativo (es. BiCGSTAB) per W.
 * @return std::pair<ComplexDenseMatrix, DenseVector> Pair contenente:
 * - Matrice X complessa degli autovettori.
 * - Vettore Lambda reale degli autovalori.
 */
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
  // Codice per info OpenMP/thread (invariato)
  // ... (come nella versione originale) ...
  int maxThreads = omp_get_max_threads();
  std::vector<ConjugateGradient<ComplexSparseMatrix, Lower | Upper>> cgSolvers(
      maxThreads);

  int n = A.rows();

  // --- Validazione Input --- (controlli dimensioni simili)
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

  // --- Inizializzazione Complessa ---
  ComplexDenseMatrix X = X_initial.leftCols(nev);
  ComplexDenseMatrix P = ComplexDenseMatrix::Zero(n, nev);
  ComplexDenseMatrix W = ComplexDenseMatrix::Zero(n, nev);
  DenseVector Lambda;           // Autovalori reali
  double current_shift = shift; // Shift reale
  ComplexSparseMatrix Id(n, n);
  Id.setIdentity();

  b_modified_gram_schmidt_complex_no_B(X); // Rendi X B-ortonormale
  int converged = 0;

  // Rayleigh-Ritz iniziale

  {
    ComplexSparseMatrix A_initial_shifted =
        A + ComplexScalar(current_shift) * Id; // Shift A
    auto rr_init = rayleighRitz_complex_no_B(X, A_initial_shifted, nev);
    if (rr_init.first.cols() == 0) {
      std::cerr << "Error: Initial Complex Rayleigh-Ritz failed." << std::endl;
      return {};
    }
    X = X * rr_init.first; // X = X * C (complesso)
    Lambda =
        rr_init.second - DenseVector::Constant(rr_init.second.size(),
                                               current_shift); // Lambda reale
    b_modified_gram_schmidt_complex_no_B(X); // Ri-ortogonalizza
  }

  // --- Iterazioni GCGM Complesso ---
  for (int iter = 0; iter < max_iter; ++iter) {

    // 1. Genera W (approx. A_shifted * W = B * X * diag(Lambda))
    if (Lambda.size() == nev && Lambda(0) != 0) {
      // Strategia shift (invariata, usa Lambda reale)
      current_shift = (Lambda(nev - 1) - 100.0 * Lambda(0)) / 99.0;
      // Controlli shift
    }

    std::cout << "Iter: " << iter + 1 << ", Current Shift: " << current_shift
              << " ";

    // A_shifted è Complessa Hermitiana (potenzialmente indefinita)
    ComplexSparseMatrix A_shifted = A + ComplexScalar(current_shift) * Id;

    auto smallId = MatrixXd::Identity(Lambda.rows(), Lambda.cols());
    // BXLambda è Complessa
    ComplexDenseMatrix BXLambda = X * Lambda.asDiagonal();
    BXLambda += X * current_shift;

    start = Clock::now();

// Prepare cg solvers
#pragma omp parallel for
    for (int i = 0; i < maxThreads; ++i) {
      cgSolvers[i].setMaxIterations(cg_steps);
      cgSolvers[i].setTolerance(cg_tol);
      // Se usi un precondizionatore, impostalo qui per ogni solver
      // cg_solvers[i].setPreconditioner(...);
      cgSolvers[i].compute(A_shifted); // Ogni thread prepara il suo solver
      // Verifica cg_solvers[i].info() se necessario
    }

#pragma omp parallel for // Opzionale
    for (int k = 0; k < nev; ++k) {
      int threadId = omp_get_thread_num();

      // Risolvi A_shifted * w_k = (BXLambda)_k
      W.col(k) = cgSolvers[threadId].solveWithGuess(BXLambda.col(k),
                                                    X.col(k)); // W è complesso
      // Controllo errore solve opzionale
    }
    end = Clock::now();
    if (benchmark) {
      std::cout << "Time CG: ";
      durMs(end - start);
    }

    // 2. Costruisci V = [X, P, W] (tutte complesse)
    ComplexDenseMatrix V(n, 3 * nev);
    V.leftCols(nev) = X;
    V(seq(0, n), seq(nev, 2 * nev - 1)) = P;
    V.rightCols(nev) = W;

    start = Clock::now();

    // 3. B-Ortogonalizza V (complessa)
    b_modified_gram_schmidt_complex_no_B(V);
    end = Clock::now();
    if (benchmark) {
      std::cout << "Time B-Ort: ";
      durMs(end - start);
    }

    // --- Gestione Dipendenze Lineari (come prima, ma su V complessa) ---
    std::vector<int> keep_cols;
    for (int k = 0; k < V.cols(); ++k) {
      // Usa squaredNorm() che funziona per complessi (è ||v||^2 = v†v)
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

    // 4. Rayleigh-Ritz su V_eff (complessa)
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

    ComplexDenseMatrix C = rr_result.first;     // Coefficienti complessi
    DenseVector Lambda_proj = rr_result.second; // Autovalori reali shftati
    DenseVector Lambda_new =
        Lambda_proj - DenseVector::Constant(Lambda_proj.size(),
                                            current_shift); // Autovalori reali

    ComplexDenseMatrix X_new = V_eff * C; // Autovettori complessi

    // 5. Verifica Convergenza (Residuo complesso, norma reale)
    ComplexDenseMatrix Residual = A * X_new - X_new * Lambda_new.asDiagonal();

    double residual_norm = Residual.norm(); // Norma di Frobenius (reale)
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
    // P = X_new - X;
    //  Eventuale ri-ortogonalizzazione di P

    // 7. Aggiorna X (complesso) e Lambda (reale)
    X = X_new;
    Lambda = Lambda_new;
    // Eventuale ri-ortogonalizzazione di X
    // if (iter % 5 == 0) b_modified_gram_schmidt_complex(X, B);

  } // Fine ciclo iterazioni
  std::cout << std::endl;

  std::cerr << "Warning: Complex GCGM did not converge within " << max_iter
            << " iterations." << std::endl;
  std::cout << "Eigenvalues: " << Lambda.transpose() << std::endl;
  return {X, Lambda};
}
