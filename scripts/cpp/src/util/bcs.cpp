#include "util/bcs.hpp"
#include <iostream>

namespace BCS
{

    MatrixXd
    compute_Gij(const MatrixXcd &phi, double V0)
    {
        int N = phi.rows();
        int M = phi.cols();
        MatrixXd G(M, M);
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < M; ++j)
            {
                double sum = 0.0;
                for (int p = 0; p < N; ++p)
                {
                    sum += std::norm(phi(p, i)) * std::norm(phi(p, j));
                }
                G(i, j) = V0 * sum;
            }
        }
        return G;
    }

    BCSResult solveBCS(const VectorXd &eps, const MatrixXd &Gij,
                       int A, double initDelta = 1.0,
                       double tol = 1e-6, int maxIter = 100)
    {
        int M = eps.size();
        VectorXd Delta = VectorXd::Constant(M, initDelta);
        double lambda = eps(M / 2); // crude init guess: middle of spectrum

        for (int iter = 0; iter < maxIter; ++iter)
        {
            // compute quasiparticle energies, occupations
            VectorXd E(M), u(M), v(M), kappa(M);
            for (int i = 0; i < M; ++i)
            {
                double xi = eps(i) - lambda;
                double Ei = std::sqrt(xi * xi + Delta(i) * Delta(i));
                E(i) = Ei;
                v(i) = std::sqrt(0.5 * (1.0 - xi / Ei));
                u(i) = std::sqrt(1.0 - v(i) * v(i));
                kappa(i) = u(i) * v(i);
            }

            // adjust lambda to fix particle number
            auto fN = [&](double lam)
            {
                double Nsum = 0.0;
                for (int i = 0; i < M; ++i)
                {
                    double xi = eps(i) - lam;
                    double Ei = std::sqrt(xi * xi + Delta(i) * Delta(i));
                    double vi2 = 0.5 * (1.0 - xi / Ei);
                    Nsum += vi2;
                }
                return Nsum;
            };

            // TODO: is it a window?
            double window = 20.0;
            double Nlow = fN(*std::min_element(eps.data(), eps.data() + M) - window);
            double Nhigh = fN(*std::max_element(eps.data(), eps.data() + M) + window);
            double lamLow = *std::min_element(eps.data(), eps.data() + M) - window;
            double lamHigh = *std::max_element(eps.data(), eps.data() + M) + window;

            for (int b = 0; b < 40; ++b)
            {
                double mid = 0.5 * (lamLow + lamHigh);
                double Nmid = fN(mid);
                if (Nmid > A)
                    lamHigh = mid;
                else
                    lamLow = mid;
            }
            lambda = 0.5 * (lamLow + lamHigh);

            // recompute with new lambda
            for (int i = 0; i < M; ++i)
            {
                double xi = eps(i) - lambda;
                double Ei = std::sqrt(xi * xi + Delta(i) * Delta(i));
                v(i) = std::sqrt(0.5 * (1.0 - xi / Ei));
                u(i) = std::sqrt(1.0 - v(i) * v(i));
                kappa(i) = u(i) * v(i);
            }

            // update Delta
            VectorXd DeltaNew(M);
            DeltaNew = -Gij * kappa;

            // check convergence
            if ((DeltaNew - Delta).norm() < tol)
            {
                Delta = DeltaNew;
                break;
            }
            Delta = 0.5 * Delta + 0.5 * DeltaNew; // mix
        }

        // final u,v with converged Delta
        VectorXd u(M), v(M);
        for (int i = 0; i < M; ++i)
        {
            double xi = eps(i) - lambda;
            double Ei = std::sqrt(xi * xi + Delta(i) * Delta(i));
            v(i) = std::sqrt(0.5 * (1.0 - xi / Ei));
            u(i) = std::sqrt(1.0 - v(i) * v(i));
        }
        std::cout << "Lambda: " << lambda << ", Delta: " << Delta << std::endl;
        return {u, v, Delta, lambda};
    }

    /**
     * Construct UV matrix (2N x M) from orbitals and BCS (u,v)
     * Convention: top block U = phi * diag(u), bottom block V = phi * diag(v)
     */
    MatrixXcd buildUV(const MatrixXcd &phi, const VectorXd &u, const VectorXd &v)
    {
        int N = phi.rows();
        int M = phi.cols();
        MatrixXcd UV(2 * N, M);
        UV.topRows(N) = phi * u.asDiagonal();
        UV.bottomRows(N) = phi * v.asDiagonal();
        return UV;
    }

    /**
     * Full driver: from orbitals, energies, particle number -> UV
     */
    BCSResult BCSiter(const MatrixXcd &phi, const VectorXd &eps,
                      int A, double V0)
    {
        MatrixXd Gij = compute_Gij(phi, V0);
        BCSResult res = solveBCS(eps, Gij, A);
        std::cout << "Computed v" << std::endl;
        std::cout << res.v.transpose() << std::endl;
        return res;
    }
}