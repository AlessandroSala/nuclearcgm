
#include <vector>
#include <iostream> // Per output/debug
#include <utility> // Per std::pair
#include <cmath>   // Per std::sqrt
#include <algorithm> // Per std::min
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Core>
#include <iostream>
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

double f_constrained(const VectorXd& x, const VectorXd& Ax) {
    return x.dot(Ax);
}
VectorXd g_constrained(const VectorXd& x, const VectorXd& Ax ) {
    return Ax - f_constrained(x, Ax) * x;
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
std::pair<double, VectorXd> find_eigenpair_constrained_dense(const MatrixXd& A, const VectorXd& X0, int max_iter = 100, double tol = 1e-6) {
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

std::pair<VectorXd, MatrixXd> accelerated_cgm_test(const SparseMatrix<double>& A, const MatrixXd& X0, int max_iter = 100, double tol = 1e-6) {
    std::pair<double, VectorXd> pair = find_eigenpair_constrained(A, VectorXd::Random(A.rows()), tol = tol, max_iter = max_iter);
    std::cout << "Eigenvalue found: " << pair.first << std::endl;
    VectorXd eigenvalues(1);
    eigenvalues(0) = pair.first;
    return std::pair<VectorXd, MatrixXd>(eigenvalues, pair.second);

}

std::pair<MatrixXd, MatrixXd> rayleigh_ritz(const SparseMatrix<double>& A, const MatrixXd& S) {
    MatrixXd M(S.cols(), S.cols());
    MatrixXd tmp(S.cols(), S.cols());
    MatrixXd vals = MatrixXd::Zero(S.cols(), S.cols());

    M = S.transpose() * S;

    for(int i = 0; i < M.rows(); i++) {
        for(int j=0; j < M.cols(); j++) {
            if(i!=j) {
                M(i, j) = 0; 
            } else {
                M(i, j) = pow(M(i,j), -0.5); 
            }
        }
    }
    LLT<MatrixXd> lltOfM(M*S.transpose()*S*M);
    SelfAdjointEigenSolver<MatrixXd> es;
    
    tmp = lltOfM.matrixU().toDenseMatrix().inverse().transpose() * M * S.transpose() * A * S * M * lltOfM.matrixU().toDenseMatrix().inverse();
    es = SelfAdjointEigenSolver<MatrixXd>(tmp);
    for(int i = 0; i<S.cols(); ++i) {
        vals(i,i) = es.eigenvalues()(i);
    }

    return std::pair<MatrixXd, MatrixXd>(es.eigenvectors(), vals);
}


// Restituisce: pair<VectorXd, MatrixXd>
// VectorXd: I k autovalori di Ritz selezionati (ordinati)
// MatrixXd: I k autovettori di Ritz corrispondenti (coefficienti Y per la base S)
std::pair<VectorXd, MatrixXd> rayleigh_ritz_correct(const SparseMatrix<double>& A, const MatrixXd& S, int k) {
    // Calcola le matrici proiettate
    MatrixXd StS = S.transpose() * S;
    MatrixXd StAS = S.transpose() * (A * S); // Efficienza: calcola prima A*S se S è denso

    // Risolvi il problema agli autovalori generalizzato: StAS * Y = StS * Y * Lambda
    // Nota: Assicurati che StAS sia simmetrica (potrebbe richiedere (StAS + StAS.transpose())/2 per stabilità)
    GeneralizedSelfAdjointEigenSolver<MatrixXd> ges;
    ges.compute(StAS, StS); // Risolve A*y = lambda*B*y dove A=StAS, B=StS

    if (ges.info() != Success) {
        // Gestisci l'errore - il solver potrebbe fallire se StS è singolare
        // (le colonne di S sono linearmente dipendenti)
        std::cerr << "GeneralizedSelfAdjointEigenSolver failed!" << std::endl;
        // Potresti dover uscire o implementare una strategia di fallback
        // (es. ortogonalizzare S e riprovare, o ridurre la dimensione del blocco)
        throw std::runtime_error("Eigenvalue solver failed");
    }

    // ges.eigenvalues() restituisce autovalori ordinati in modo crescente.
    // ges.eigenvectors() restituisce gli autovettori corrispondenti Y come colonne.

    // Estrai i k più piccoli autovalori e i corrispondenti autovettori Y
    VectorXd ritz_values_k = ges.eigenvalues().head(k);
    MatrixXd ritz_vectors_Y_k = ges.eigenvectors().leftCols(k);

    return std::make_pair(ritz_values_k, ritz_vectors_Y_k);
}
std::pair<VectorXd, MatrixXd> lobpcg_correct(const SparseMatrix<double>& A, const MatrixXd& X0, int max_iter = 100, double tol = 1e-6) { // Rimosso max_iter_acgm ridondante
    int N = A.rows();
    int k = X0.cols();

    // --- Orthonormalize X0 (importante per il primo passo) ---
    // Householder QR decomposition
    HouseholderQR<MatrixXd> qr(X0);
    MatrixXd X = qr.householderQ() * MatrixXd::Identity(N, k);
    // ---------------------------------------------------------

    MatrixXd P = MatrixXd::Zero(N, k); // P è inizializzato a zero
    MatrixXd R;
    MatrixXd W;
    MatrixXd S;
    MatrixXd X_new;
    std::pair<VectorXd, MatrixXd> rr_pair; // Ora contiene VectorXd per gli autovalori

    // Calcolo iniziale di Ritz e residuo
    rr_pair = rayleigh_ritz_correct(A, X, k); // Usa X ortonormale iniziale
    // X = X * rr_pair.second; // Non serve riaggiornare X qui, è già la base usata
    MatrixXd Lambda = rr_pair.first.asDiagonal(); // Matrice diagonale degli autovalori
    R = A * X - X * Lambda; // Calcola il residuo iniziale

    for (int iter = 0; iter < max_iter; ++iter) {
         // --- Calcolo norma residuo (esempio) ---
         // Potresti voler usare una norma specifica B-norm se B non fosse identità
         double residual_norm = 0;
         for(int j=0; j<k; ++j) {
            // Norma del residuo per ciascun autovettore, o norma Frobenius totale?
            // Qui usiamo la norma Frobenius della matrice R
             residual_norm = R.norm(); // O R.col(j).norm() se vuoi convergenza individuale
         }
         std::cout << "Iteration " << iter << " | Residual Norm: " << residual_norm << std::endl;
         std::cout << "Eigenvalues: " << rr_pair.first.transpose() << std::endl; // Stampa il vettore

         if (residual_norm < tol) {
             break;
         }
         // --- Fine calcolo norma residuo ---


        // Precondizionamento (qui T=I) e ortogonalizzazione W (opzionale ma raccomandato)
        W = R; // T*W = R => W = R perché T=I

        // --- Ortogonalizzazione di W rispetto a X (B-ortogonalizzazione se B!=I) ---
        // Proietta W fuori dallo spazio di X: W = W - X * (X^T * W)  (poiché X^T*X = I)
        // Se X non fosse garantito ortonormale, servirebbe (X^T * X)^-1 * (X^T * W)
        MatrixXd XtW = X.transpose() * W;
        W = W - X * XtW;
        // Potrebbe essere necessaria una ri-ortogonalizzazione di W stesso se le sue colonne
        // diventano dipendenti, specialmente se si usa un precondizionatore.
        // Per B=I, T=I, potrebbe non essere strettamente necessario all'inizio.
        // ----------------------------------------------------------------------


        // Costruzione sottospazio S
        if (iter == 0) {
            S.resize(N, 2 * k);
            S.block(0, 0, N, k) = X;
            S.block(0, k, N, k) = W;
        } else {
             // --- Ortogonalizzazione di P rispetto a X ---
             MatrixXd XtP = X.transpose() * P;
             P = P - X * XtP;
             // --- Ortogonalizzazione di P rispetto a W (opzionale) ---
             // MatrixXd WtP = W.transpose() * P; // Serve W^T W se W non ortonormale
             // HouseholderQR<MatrixXd> qr_w(W); // QR per ortonormalizzare W
             // MatrixXd Q_w = qr_w.householderQ() * MatrixXd::Identity(N, k);
             // P = P - Q_w * (Q_w.transpose() * P);
             // ----------------------------------------------------------

            S.resize(N, 3 * k);
            S.block(0, 0, N, k) = X;
            S.block(0, k, N, k) = W;
            S.block(0, 2 * k, N, k) = P;
        }

        // Passo Rayleigh-Ritz sul sottospazio S
        rr_pair = rayleigh_ritz_correct(A, S, k); // Trova i k migliori nel sottospazio S
        VectorXd current_eigenvalues = rr_pair.first;
        MatrixXd Y = rr_pair.second; // Coefficienti per la base S

        // Aggiorna X, P, R
        // Y ha dimensioni S.cols() x k
        MatrixXd X_prev = X; // Salva il vecchio X per calcolare P dopo

        X = S * Y; // Aggiorna gli autovettori approssimati
        Lambda = current_eigenvalues.asDiagonal();
        R = A * X - X * Lambda; // Calcola il nuovo residuo

        // Aggiorna P: P_new = W*Y_w + P_old*Y_p
        // Y è partizionato [Y_x^T, Y_w^T, Y_p^T]^T (se iter > 0)
        // Y_x sono le prime k righe, Y_w le successive k, Y_p le ultime k (se esistono)
        if (iter == 0) { // S = [X_prev, W] -> Y è (2k x k)
            P = W * Y.block(k, 0, k, k); // P_new = W * Y_w
        } else { // S = [X_prev, W, P_prev] -> Y è (3k x k)
            P = W * Y.block(k, 0, k, k) + P * Y.block(2 * k, 0, k, k); // P_new = W*Y_w + P_prev*Y_p
        }
         // --- Assicurarsi che X sia ortonormale per il prossimo giro (opzionale ma buono) ---
         // HouseholderQR<MatrixXd> qr_x(X);
         // X = qr_x.householderQ() * MatrixXd::Identity(N, k);
         // Se fai questo, devi ricalcolare Lambda = X^T A X e poi R = AX - X Lambda
         // ---------------------------------------------------------------------------
    }
    // Ritorna l'ultimo set di autovalori e autovettori calcolati
    return std::make_pair(rr_pair.first, X); // Restituisce VectorXd e MatrixXd
}

// --- ATTENZIONE ---
// Gli include di Eigen sono necessari per la compilazione.
// Sono stati omessi come richiesto, ma devono essere presenti nel file C++.
// Assicurati che Eigen sia installato e accessibile dal tuo compilatore.
// Se A e B sono matrici sparse, usa Eigen::SparseMatrix<double> e
// decommenta <Eigen/SparseCore>. Le operazioni dovranno essere adattate.

// Usiamo matrici dense per semplicità. Cambia a SparseMatrix se necessario.
using namespace Eigen;

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
    const DenseMatrix& A,           // Usa SparseMatrix se necessario
    const DenseMatrix& B,           // Usa SparseMatrix se necessario
    const DenseMatrix& X_initial,
    int nev,
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

    // Rendi X B-ortonormale all'inizio
    b_modified_gram_schmidt(X, B);
    // Qui bisognerebbe verificare e gestire eventuali colonne nulle (dipendenze lineari)

    // Rayleigh-Ritz iniziale per ottenere Lambda
    DenseVector Lambda;
    {
        auto rr_init = rayleighRitz(X, A, B, nev);
        if (rr_init.first.cols() == 0) { // Fallimento RR iniziale
            std::cerr << "Error: Initial Rayleigh-Ritz failed." << std::endl;
            return {};
        }
        // Aggiorna X = X * C e Lambda
        X = X * rr_init.first;
        Lambda = rr_init.second;
        // Ri-ortogonalizza X dopo la combinazione lineare (importante!)
        b_modified_gram_schmidt(X, B);
    }

    // --- Iterazioni GCGM ---
    for (int iter = 0; iter < max_iter; ++iter) {

        // 1. Genera W (cfr. Algorithm 1, Step 2)
        //    Risolvi approx. A * W = B * X * diag(Lambda) usando 'cg_steps' passi di CG.
        //    L'uso di Lambda qui segue l'idea dell'inverse power iter.
        DenseMatrix BXLambda = B * X * Lambda.asDiagonal();

        // Configura il solver CG (per matrici dense A - usare versioni per sparse se A è sparsa)
        // Nota: CG richiede che A sia SPD. Se A non lo è, si dovrebbe usare un solver diverso
        //       (es. MINRES se A è simmetrica indefinita) o introdurre uno shift (Algorithm 2).
        //       Qui assumiamo A SPD o usiamo CG come approssimazione generica.
        ConjugateGradient<DenseMatrix, Lower|Upper> cg;
        cg.setMaxIterations(cg_steps); // Limita i passi come da PDF
        cg.compute(A); // Precomputa se possibile (per A densa, non fa molto)

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

        auto rr_result = rayleighRitz(V_eff, A, B, rr_nev);
        if (rr_result.first.cols() == 0) {
             std::cerr << "Error: Rayleigh-Ritz failed in iteration " << iter << std::endl;
             return {X, Lambda}; // Restituisci l'ultimo risultato valido
        }

        DenseMatrix C = rr_result.first;          // Coefficienti nella base V_eff
        DenseVector Lambda_new = rr_result.second; // Nuovi autovalori

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


// Esempio di utilizzo (main):
/*
int main() {
    // --- Definire le matrici A e B e la stima iniziale X_initial ---
    // Esempio: Problema standard A*x = lambda*x => B = Identità
    int n = 100; // Dimensione
    int nev = 5; // Numero di autovalori/vettori

    DenseMatrix A = DenseMatrix::Random(n, n);
    A = A + A.transpose(); // Rendi A simmetrica
    // Per renderla definita positiva (per CG):
    // A = A * A.transpose() + DenseMatrix::Identity(n,n) * 0.1;

    DenseMatrix B = DenseMatrix::Identity(n, n); // Problema standard

    DenseMatrix X_initial = DenseMatrix::Random(n, nev); // Stima iniziale casuale

    // --- Chiamare GCGM ---
    int max_iterations = 200;
    double tolerance = 1e-9;
    int cg_steps = 15;

    std::pair<DenseMatrix, DenseVector> result = gcgm(A, B, X_initial, nev, max_iterations, tolerance, cg_steps);

    // --- Controllare e usare i risultati ---
    if (result.first.cols() > 0) {
        DenseMatrix Eigenvectors = result.first;
        DenseVector Eigenvalues = result.second;

        std::cout << "\nConverged Eigenvalues:\n" << Eigenvalues << std::endl;
        // std::cout << "\nConverged Eigenvectors (first 5 rows):\n" << Eigenvectors.topRows(5) << std::endl;

        // Verifica opzionale del residuo finale
        DenseMatrix FinalResidual = A * Eigenvectors - B * Eigenvectors * Eigenvalues.asDiagonal();
        std::cout << "\nFinal Relative Residual Norm: " << FinalResidual.norm() / Eigenvectors.norm() << std::endl;
    } else {
        std::cout << "\nGCGM failed to compute eigenvalues." << std::endl;
    }

    return 0;
}
*/

std::pair<MatrixXd, MatrixXd> lobpcg(const SparseMatrix<double>& A, const MatrixXd& X0, int max_iter = 100, double tol = 1e-6, int max_iter_acgm = 1000) {
    int N = A.rows();
    int k = X0.cols();
    MatrixXd eigenvectors(N, k);
    MatrixXd M(k, k);
    MatrixXd X = X0;
    MatrixXd W(N, k);
    MatrixXd P(N, k);
    MatrixXd S;
    MatrixXd Q(N, N);
    VectorXd eigenvalues(k);
    VectorXd tmp(N);
    MatrixXd R(N, k);
    MatrixXd X_new(N, k);
    std::pair<MatrixXd, MatrixXd> rr_pair;

    rr_pair = rayleigh_ritz(A, X);
    X = X * rr_pair.first;
    
    R = A * X - X * rr_pair.second;


    for(int iter = 0; iter < max_iter_acgm; ++iter) {
        if (R.norm() < tol) {
            break;
        }
        W = R; //Solve BW = R
        if(iter==0) {
            S = MatrixXd(N, 2*k);
            S(all, seq(0, k-1)) = X;
            S(all, seq(k, 2*k-1)) = W;
        } else {
            S = MatrixXd(N, 3*k);
            S(all, seq(0, k-1)) = X;
            S(all, seq(k, 2*k-1)) = W;
            S(all, seq(2*k, 3*k-1)) = P;
        }


        rr_pair = rayleigh_ritz(A, S);
        X_new = S * rr_pair.first(all, seq(0, k-1));
        R = A * X_new - X_new * rr_pair.second(seq(0, k-1), seq(0, k-1));
        P = S(all, seq(k, S.cols()-1)) * rr_pair.first(seq(k, rr_pair.first.cols()-1), seq(0, k-1));
        X = X_new;

        
        std::cout << "Iteration " << iter << std::endl;
        std::cout << "Eigenvalues: " << rr_pair.second << std::endl;

        
    }
    return std::make_pair(rr_pair.second, X_new);
}

std::pair<VectorXd, MatrixXd> accelerated_cgm(const SparseMatrix<double>& A, const MatrixXd& X0, int max_iter = 100, double tol = 1e-6, int max_iter_acgm = 1000) {
    int N = A.rows();
    int k = X0.cols();
    MatrixXd eigenvectors(N, k);
    MatrixXd M(k, k);
    MatrixXd W = X0;
    MatrixXd W_copy = X0;
    MatrixXd Q(N, N);
    VectorXd eigenvalues(k);
    std::pair<double, VectorXd> pair;
    SelfAdjointEigenSolver<MatrixXd> es;
    VectorXd sum(N);
    VectorXd tmp(N);
    std::vector<std::pair<double, VectorXd>> pairs(k);

    for(int iter = 0; iter < max_iter_acgm; ++iter) {
        for(int i = 0; i < k; ++i) {
            Q.setIdentity();
            if(i > 0) {
                Q -=  (W(all, seq(0, i-1)) * W(all, seq(0, i-1)).transpose());
            }
            pair = find_eigenpair_constrained_dense(Q*A*Q, Q*VectorXd::Random(N), max_iter, tol);
            W(all, i) = pair.second;
            eigenvalues(i) = pair.first;
            tmp = pair.second;
            pairs[i] = std::make_pair(eigenvalues(i), tmp);
            
        }
        
        M = W.transpose() * A * W;
        es = SelfAdjointEigenSolver<MatrixXd>(M);
        eigenvectors = es.eigenvectors();
        W_copy = MatrixXd(W);
        //for(int i = 0; i < k; ++i) {
            //sum = VectorXd::Zero(N);
            //for(int j = 0; j < k; ++j) {
                //sum += es.eigenvectors()(j, i) * W_copy(all, j);
            //}
            //W(all, i) = sum;

        //}
        W = W_copy * es.eigenvectors();
        
        std::cout << "Iteration " << iter << std::endl;
        std::cout << "Eigenvalues: " << eigenvalues << std::endl;
        std::cout << "Diag eigenvalules: " << es.eigenvalues() << std::endl;

        
    }
    return std::pair<VectorXd, MatrixXd>(es.eigenvalues(), W);
}


MatrixXd random_orthonormal_matrix(int n, int k) {
    MatrixXd A = MatrixXd::Random(n, k);            // matrice random
    HouseholderQR<MatrixXd> qr(A);                  // QR decomposition
    MatrixXd Q = qr.householderQ() * MatrixXd::Identity(n, k); // primi k vettori ortonormali
    return Q;
}