
#include <vector>
#include <iostream> 
#include <utility> 
#include <cmath>  
#include <algorithm>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Core>
#include <iostream>
using namespace Eigen;

double f_constrained(const VectorXd& x, const VectorXd& Ax) {
    return x.dot(Ax);
}
VectorXd g_constrained(const VectorXd& x, const VectorXd& Ax ) {
    return Ax - f_constrained(x, Ax) * x;
}

std::pair<double, VectorXd> find_eigenpair_constrained(const SparseMatrix<double>& A, const VectorXd& X0, int max_iter = 100, double tol = 1e-6) {
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
        delta = pow(l*d - b,2) - 4*(b*c - a*d)*(a-l*c);
        alfa = (l*d - b + sqrt(delta))/(2*(b*c - a*d));
        gamma = sqrt(1+2*c*alfa+d*pow(alfa, 2));
        l = (l+a*alfa)/(1+c*alfa); // new
        x = (x+alfa*p)/gamma;
        Ax = (Ax+alfa*z)/gamma;
        grad = Ax - l*x;
        if (grad.norm() < tol*l) {
            break;
        }
        beta = -(grad.dot(z))/(b);
        p = grad + beta*p;
        z = A*p;
    }
    return {f_constrained(x, Ax), x};

}


// Typedef per chiarezza (usa SparseMatrix se A, B sono sparse)
typedef MatrixXd DenseMatrix;
typedef VectorXd DenseVector;
// typedef SparseMatrix<double> SparseMatrix; // Alternativa Sparsa

/**
 * @brief Ortogonalizzazione di Gram-Schmidt Modificata rispetto al prodotto scalare indotto da B.
 * Modifica V sul posto. Assume che B sia Simmetrica Definitia Positiva (SPD).
 *
 * @param V Matrice le cui colonne verranno B-ortogonalizzate.
 * @param B Matrice SPD che definisce il prodotto scalare.
 */
void b_modified_gram_schmidt(DenseMatrix& V, const DenseMatrix& B) {
    if (V.cols() == 0) return;

    for (int j = 0; j < V.cols(); ++j) {
        // Normalizza la colonna j
        double norm_b_sq = V.col(j).transpose() * B * V.col(j);
        double norm_b = (norm_b_sq > 1e-24) ? std::sqrt(norm_b_sq) : 0.0;

        if (norm_b < 1e-12) {
            // La colonna è (quasi) zero nel prodotto B o linearmente dipendente
            // Potrebbe essere necessario gestirla (es. rimuoverla), qui la azzeriamo
            V.col(j).setZero();
            // std::cerr << "Warning: Vector " << j << " has near-zero B-norm in MGS." << std::endl;
            continue; // Passa alla colonna successiva
        }
        V.col(j) /= norm_b;

        // Rendi le colonne successive (k > j) ortogonali alla colonna j
        #pragma omp parallel for // Opzionale: parallelizza il ciclo esterno se V è grande
        for (int k = j + 1; k < V.cols(); ++k) {
            double proj = V.col(k).transpose() * B * V.col(j);
            V.col(k) -= proj * V.col(j);
        }
    }
    // Potrebbe essere necessaria una seconda passata (ri-ortogonalizzazione) per stabilità numerica
    // o l'uso di metodi a blocchi come descritto nel PDF per migliori performance/stabilità.
}


/**
 * @brief Procedura di Rayleigh-Ritz per il problema generalizzato Ax = lambda Bx.
 *
 * @param V Matrice la cui base (colonne) Spannano il sottospazio di proiezione. V DEVE essere B-ortonormale.
 * @param A Matrice A del problema agli autovalori.
 * @param B Matrice B del problema agli autovalori (SPD).
 * @param nev Numero di autovalori/autovettori desiderati (tipicamente i più piccoli).
 * @return std::pair<DenseMatrix, DenseVector> Pair contenente:
 * - Matrice C dei coefficienti degli autovettori nella base V (le colonne sono gli autovettori proiettati).
 * - Vettore Lambda dei corrispondenti autovalori calcolati.
 * Restituisce matrici/vettori vuoti in caso di errore.
 */
std::pair<DenseMatrix, DenseVector> rayleighRitz(
    const DenseMatrix& V,
    const DenseMatrix& A,
    const DenseMatrix& B,
    int nev)
{
    if (V.cols() == 0) {
        std::cerr << "Error: Rayleigh-Ritz called with empty basis V." << std::endl;
        return {};
    }

    // 1. Proietta le matrici sul sottospazio V
    DenseMatrix A_proj = V.transpose() * A * V;
    // Se V è B-ortonormale, B_proj = V' * B * V dovrebbe essere l'identità (numericamente)
    DenseMatrix B_proj = V.transpose() * B * V;
    // Si potrebbe verificare che B_proj sia vicina a I: (B_proj - DenseMatrix::Identity(V.cols(), V.cols())).norm() < tolerance

    // 2. Risolvi il problema agli autovalori proiettato (piccolo): A_proj * C = lambda * B_proj * C
    GeneralizedSelfAdjointEigenSolver<DenseMatrix> ges;
    ges.compute(A_proj, B_proj);

    if (ges.info() != Success) {
        std::cerr << "Error: Rayleigh-Ritz eigenvalue computation failed!" << std::endl;
        return {};
    }

    DenseVector eigenvalues_all = ges.eigenvalues();
    DenseMatrix eigenvectors_proj_all = ges.eigenvectors(); // Queste sono le colonne di C

    // 3. Seleziona i 'nev' autovalori/autovettori desiderati (i più piccoli)
    //    GeneralizedSelfAdjointEigenSolver li ordina già in modo crescente.
    int num_found = eigenvalues_all.size();
    if (num_found < nev) {
         std::cerr << "Warning: Rayleigh-Ritz found only " << num_found << " eigenvalues, requested " << nev << "." << std::endl;
         nev = num_found; // Prendi tutti quelli trovati
    }
     if (nev <= 0) {
         std::cerr << "Error: No eigenvalues found or requested in Rayleigh-Ritz." << std::endl;
         return {};
     }


    DenseVector lambda_new = eigenvalues_all.head(nev);
    DenseMatrix C = eigenvectors_proj_all.leftCols(nev);

    return {C, lambda_new};
}

/**
 * @brief Implementazione semplificata dell'algoritmo GCGM (Generalized Conjugate Gradient Method).
 * Risolve Ax = lambda Bx per i `nev` autovalori più piccoli e i corrispondenti autovettori.
 * Basato principalmente su Algorithm 1 del PDF, con semplificazioni.
 *
 * @param A Matrice A (densa o sparsa - codice attuale per densa).
 * @param B Matrice B (densa o sparsa, SPD).
 * @param X_initial Stima iniziale per gli autovettori (colonne). Deve avere almeno `nev` colonne.
 * @param nev Numero di autocoppie da calcolare.
 * @param max_iter Numero massimo di iterazioni GCGM.
 * @param tolerance Tolleranza sulla norma relativa del residuo per la convergenza.
 * @param cg_steps Numero di passi del Gradiente Coniugato (CG) per la soluzione approssimata di W.
 * @return std::pair<DenseMatrix, DenseVector> Pair contenente:
 * - Matrice X degli autovettori calcolati (colonne).
 * - Vettore Lambda dei corrispondenti autovalori.
 * Restituisce matrici/vettori vuoti in caso di errore grave.
 */
std::pair<DenseMatrix, DenseVector> gcgm(
    const SparseMatrix<double>& A,           // Usa SparseMatrix se necessario
    const SparseMatrix<double>& B,           // Usa SparseMatrix se necessario
    const DenseMatrix& X_initial,
    int nev,
    double shift = 0.0,
    int max_iter = 100,
    double tolerance = 1e-8,
    int cg_steps = 10              // Passi CG per W (come da PDF step 2)
) {
    int n = A.rows(); // Dimensione del problema

    // --- Validazione Input ---
    if (A.rows() != n || A.cols() != n || B.rows() != n || B.cols() != n || X_initial.rows() != n) {
        std::cerr << "Error: Matrix dimensions mismatch." << std::endl;
        return {};
    }
    if (X_initial.cols() < nev || nev <= 0) {
         std::cerr << "Error: Initial guess must have at least 'nev' columns, and nev > 0." << std::endl;
         return {};
    }

    // --- Inizializzazione ---
    DenseMatrix X = X_initial.leftCols(nev); // Usa le prime 'nev' colonne della stima iniziale
    DenseMatrix P = DenseMatrix::Zero(n, nev); // Blocco P precedente (inizialmente vuoto/zero)
    DenseMatrix W = DenseMatrix::Zero(n, nev); // Blocco W (nuova direzione)
    double current_shift = shift;

    // Rendi X B-ortonormale all'inizio
    b_modified_gram_schmidt(X, B);
    // Qui bisognerebbe verificare e gestire eventuali colonne nulle (dipendenze lineari)

    // Rayleigh-Ritz iniziale per ottenere Lambda
    DenseVector Lambda;
    {
        DenseMatrix A_initial_shifted = A + current_shift * B; // Usa shift iniziale

        auto rr_init = rayleighRitz(X, A_initial_shifted, B, nev);
        if (rr_init.first.cols() == 0) { // Fallimento RR iniziale
            std::cerr << "Error: Initial Rayleigh-Ritz failed." << std::endl;
            return {};
        }
        // Aggiorna X = X * C e Lambda
        X = X * rr_init.first;
        Lambda = rr_init.second - DenseVector::Constant(rr_init.second.size(), current_shift);
        // Ri-ortogonalizza X dopo la combinazione lineare (importante!)
        b_modified_gram_schmidt(X, B);
    }

    // --- Iterazioni GCGM ---
    for (int iter = 0; iter < max_iter; ++iter) {

        // 1. Genera W (cfr. Algorithm 1, Step 2)
        //    Risolvi approx. A * W = B * X * diag(Lambda) usando 'cg_steps' passi di CG.
        //    L'uso di Lambda qui segue l'idea dell'inverse power iter.
        // DenseMatrix BXLambda = B * X * Lambda.asDiagonal();
        //if (Lambda.size() > 0 && Lambda(0) < 0) {
             //// Stima semplice e conservativa, potrebbe non essere ottimale
             //current_shift = std::max(current_shift, -Lambda(0) + 1e-4);
        //}
        // Una strategia più robusta (come da Algoritmo 2, Step 5) potrebbe essere:
        if (Lambda.size() == nev && Lambda(0) != 0) {
           current_shift = (Lambda(nev-1) - 100.0 * Lambda(0)) / 99.0;
           // Aggiungi controlli per evitare shift troppo grandi o NaN
        }
        // Assicurati che lo shift scelto renda A+shift*B definita positiva!
        std::cout << "Iter: " << iter + 1 << ", Current Shift: " << current_shift << std::endl;

        SparseMatrix<double> A_shifted = A + current_shift * B;
        DenseMatrix BXLambda = B * X * Lambda.asDiagonal();
        
        // Configura il solver CG (per matrici dense A - usare versioni per sparse se A è sparsa)
        ConjugateGradient<SparseMatrix<double>, Lower|Upper> cg;
        cg.setMaxIterations(cg_steps); // Limita i passi come da PDF
        cg.compute(A_shifted); // Precomputa se possibile (per A densa, non fa molto)

        if (cg.info() != Success && iter == 0) { // Controlla solo la prima volta se compute fallisce
             std::cerr << "Error: CG compute structure failed. Is A SPD?" << std::endl;
             return {};
        }

        #pragma omp parallel for // Opzionale: parallelizza il solve per colonna
        for (int k = 0; k < nev; ++k) {
             // Risolvi A * w_k = (BXLambda)_k con X_k come guess iniziale (come da PDF)
             W.col(k) = cg.solveWithGuess(BXLambda.col(k), X.col(k));
             // Non controlliamo l'errore di solve qui, è un'approssimazione
             // if(cg.info()!=Success) { ... }
        }

        // 2. Costruisci lo spazio V = [X, P, W]
        //    Gestione semplificata: P ha sempre 'nev' colonne dopo la prima iterazione.
        int p_cols = (iter == 0) ? 0 : nev; // P è vuoto solo alla prima iterazione
        DenseMatrix V(n, nev + p_cols + nev);
        V.leftCols(nev) = X;
        if (p_cols > 0) {
            V.block(0, nev, n, p_cols) = P;
        }
        V.rightCols(nev) = W;

        // 3. B-Ortogonalizza V
        b_modified_gram_schmidt(V, B);
        // --- Gestione robusta delle dipendenze lineari ---
        // Dopo l'ortogonalizzazione, alcune colonne di V potrebbero essere nulle o quasi.
        // Andrebbero identificate e rimosse, aggiornando la dimensione effettiva di V.
        // Questa parte è omessa per semplicità ma critica per la robustezza.
        // Esempio concettuale (non implementato robustamente):
        std::vector<int> keep_cols;
        for(int k=0; k < V.cols(); ++k) {
            if (V.col(k).squaredNorm() > 1e-20) { // Tolleranza per colonna non nulla
                keep_cols.push_back(k);
            }
        }
        if (keep_cols.size() < nev) {
             std::cerr << "Warning: Basis V rank collapsed below nev (" << keep_cols.size() << ") at iteration " << iter << std::endl;
             // Potrebbe indicare convergenza o problemi numerici. Usciamo con l'ultimo risultato valido.
             return {X, Lambda};
        }
        // Crea una nuova matrice V_eff con solo le colonne valide
        DenseMatrix V_eff(n, keep_cols.size());
        for(size_t i=0; i < keep_cols.size(); ++i) {
            V_eff.col(i) = V.col(keep_cols[i]);
        }
        // Da qui si usa V_eff invece di V


        // 4. Rayleigh-Ritz sul sottospazio V (o V_eff)
        int current_dim = V_eff.cols(); // Dimensione effettiva della base
        int rr_nev = std::min(nev, current_dim); // Non possiamo ottenere più autovettori della dimensione della base

        auto rr_result = rayleighRitz(V_eff, A_shifted, B, rr_nev);
        if (rr_result.first.cols() == 0) {
             std::cerr << "Error: Rayleigh-Ritz failed in iteration " << iter << std::endl;
             return {X, Lambda}; // Restituisci l'ultimo risultato valido
        }

        DenseMatrix C = rr_result.first;          // Coefficienti nella base V_eff
        DenseMatrix Lambda_proj = rr_result.second; // Nuovi autovalori
        DenseVector Lambda_new = Lambda_proj - DenseVector::Constant(Lambda_proj.size(), current_shift); // Nuovi autovalori

        DenseMatrix X_new = V_eff * C; // Nuovi candidati autovettori

        // 5. Verifica Convergenza
        //    Calcola la norma del blocco residuo R = A*X_new - B*X_new*Lambda_new
        DenseMatrix Residual = A * X_new - B * X_new * Lambda_new.asDiagonal();
        double residual_norm = Residual.norm(); // Norma di Frobenius
        double x_norm = X_new.norm();
        double relative_residual = (x_norm > 1e-12) ? (residual_norm / x_norm) : residual_norm;

        std::cout << "Iter: " << iter + 1
                  << ", Dim(V): " << current_dim
                  << ", Rel. Res: " << relative_residual << std::endl;

        if (relative_residual < tolerance) {
            std::cout << "Converged in " << iter + 1 << " iterations." << std::endl;
            std::cout << "Eigenvalues: " << Lambda_new.transpose() << std::endl;
            return std::make_pair(X_new, Lambda_new);
        }

        // 6. Aggiorna P (direzione di ricerca precedente)
        //    P = X_new - X (semplice differenza, come in molti metodi CG-like)
        //    Alternative più complesse dal PDF (basate su C) potrebbero essere più efficienti/robuste.
        P = X_new - X;
        // Eventualmente ri-ortogonalizzare P rispetto a X_new? P = P - X_new * (X_new.transpose() * B * P)

        // 7. Aggiorna X e Lambda per la prossima iterazione
        X = X_new;
        Lambda = Lambda_new;
        // Opzionale ma consigliato: ri-ortogonalizza X ogni tanto per mantenere la B-ortonormalità
        // if (iter % 5 == 0) b_modified_gram_schmidt(X, B);

    } // Fine ciclo iterazioni

    std::cerr << "Warning: GCGM did not converge within " << max_iter << " iterations." << std::endl;
    return {X, Lambda}; // Restituisci l'ultimo risultato ottenuto
}

MatrixXd random_orthonormal_matrix(int n, int k) {
    MatrixXd A = MatrixXd::Random(n, k);            // matrice random
    HouseholderQR<MatrixXd> qr(A);                  // QR decomposition
    MatrixXd Q = qr.householderQ() * MatrixXd::Identity(n, k); // primi k vettori ortonormali
    return Q;
}