#include "util/bcs.hpp"
#include "operators/integral_operators.hpp"
#include "util/wavefunction.hpp"
#include "grid.hpp"
#include <iostream>

namespace BCS
{

    MatrixXd
    compute_G_pairing(const MatrixXcd &phi,
                      const std::vector<int> &unique_pairs,
                      const std::vector<int> &all_partners,
                      PairingParameters params, NucleonType t)
    {
        auto grid = *Grid::getInstance();
        int num_pairs = unique_pairs.size();
        MatrixXcd G_p = MatrixXcd::Zero(num_pairs, num_pairs);
        MatrixXcd phi_bar = Wavefunction::timeReverse(phi);

        for (int p = 0; p < num_pairs; ++p)
        {
            int i = unique_pairs[p];
            int i_bar = all_partners[i];
            for (int q = 0; q < num_pairs; ++q)
            {
                int j = unique_pairs[q];
                int j_bar = all_partners[j];

                VectorXcd P_i(grid.get_total_spatial_points());
                VectorXcd P_j(grid.get_total_spatial_points());
                for (int x = 0; x < grid.get_n(); ++x)
                    for (int y = 0; y < grid.get_n(); ++y)
                        for (int z = 0; z < grid.get_n(); ++z)
                        {
                            {
                                {
                                    int up = grid.idx(x, y, z, 0);
                                    int down = grid.idx(x, y, z, 1);
                                    std::complex<double> a = phi(up, i);
                                    std::complex<double> b = phi(down, i);
                                    std::complex<double> c = phi_bar(up, i);
                                    std::complex<double> d = phi_bar(down, i);
                                    std::complex<double> e = phi(up, j);
                                    std::complex<double> f = phi(down, j);
                                    std::complex<double> g = phi_bar(up, j);
                                    std::complex<double> h = phi_bar(down, j);

                                    int P_idx = grid.idxNoSpin(x, y, z);

                                    P_i(P_idx) = a * d - b * c;
                                    P_j(P_idx) = e * h - f * g;
                                }
                            }
                        }

                // integrand to integrate:
                VectorXcd integrand = P_i.conjugate().cwiseProduct(P_j);

                double V0 = t == NucleonType::N ? params.V0N : params.V0P;
                G_p(p, q) = V0 * Operators::integralNoSpin(integrand, grid);
            }
        }
        return G_p.real();
    }

    MatrixXd pairs(const MatrixXcd &phi)
    {
        auto res = (phi.adjoint() * Wavefunction::timeReverse(phi)).cwiseAbs2().eval();

        return res;
    }
    std::vector<int> match_time_reversal(const Eigen::MatrixXcd &O, double threshold = 1e-6)
    {
        int M = O.rows();
        using Entry = std::tuple<double, int, int>;
        std::vector<Entry> list;
        list.reserve(M * M);
        for (int k = 0; k < M; ++k)
        {
            for (int j = 0; j < M; ++j)
            {
                double val = std::abs(O(k, j));
                if (val > threshold)
                    list.emplace_back(val, k, j);
            }
        }

        std::sort(list.begin(), list.end(),
                  [](const Entry &a, const Entry &b)
                  { return std::get<0>(a) > std::get<0>(b); });

        std::vector<int> partner(M, -1);
        std::vector<char> used(M, 0);

        for (auto &e : list)
        {
            int k = std::get<1>(e);
            int j = std::get<2>(e);
            if (used[k] || used[j])
                continue;
            if (k == j)
                continue;
            partner[k] = j;
            partner[j] = k;
            used[k] = used[j] = 1;
        }

        return partner;
    }

    BCSResult solveBCS(const VectorXd &eps_pairs, const MatrixXd &G_pairing,
                       int A, double window, const VectorXd &oldDelta, double oldLambda, double initDelta = 0.01,
                       double tol = 1e-12, int maxIter = 20000, double PairingPrec = 1e-12)
    {
        using std::abs;
        using std::isfinite;
        using std::isnan;

        const double EPS_SMALL = 1e-12;

        int num_pairs = eps_pairs.size();
        if (num_pairs == 0)
            return {}; // empty

        // ----- INITIALIZE -----
        MatrixXd G = 0.5 * (G_pairing + G_pairing.transpose());

        double G_avg = G.diagonal().mean();

        VectorXd Delta = VectorXd::Constant(num_pairs, initDelta);
        std::cout << "Old Delta size: " << oldDelta.size() << ", norm: " << oldDelta.norm() << std::endl;
        if (oldDelta.size() == num_pairs && oldDelta.norm() > 1e-2)
        {
            Delta = oldDelta;
        }
        else
        {
            std::cout << "Reinitializing Delta" << std::endl;
        }

        double lambda = oldLambda;
        if (!isfinite(lambda) || isnan(lambda))
        {
            int mid = std::clamp((A / 2) - 1, 0, num_pairs - 1);
            lambda = eps_pairs(mid);
        }

        VectorXd kappa = VectorXd::Zero(num_pairs);
        VectorXd v = VectorXd::Zero(num_pairs);
        VectorXd u = VectorXd::Zero(num_pairs);

        // --- STABILITY IMPROVEMENTS ---
        double mixDelta = 0.3; // More conservative mixing
        double mixLambda = 0.3;
        double maxDeltaStepFrac = 0.8; // Smaller max step

        for (int iter = 0; iter < maxIter; ++iter)
        {
            double prevLambda = lambda;

            // 1) Compute quasiparticle energies
            VectorXd Eqp(num_pairs);
            for (int p = 0; p < num_pairs; ++p)
            {
                double xi_prev = eps_pairs(p) - prevLambda;
                Eqp(p) = std::sqrt(xi_prev * xi_prev + Delta(p) * Delta(p));
            }

            // 2) Algebraic update of lambda
            double sum1 = 0.0;
            double sumInv = 0.0;
            for (int p = 0; p < num_pairs; ++p)
            {
                sum1 += (1.0 - eps_pairs(p) / Eqp(p));
                sumInv += 1.0 / Eqp(p);
            }

            double rawNewLambda = (sumInv < EPS_SMALL) ? prevLambda : (static_cast<double>(A) - sum1) / sumInv;
            lambda = mixLambda * rawNewLambda + (1.0 - mixLambda) * lambda;

            // 3) Recompute u, v, kappa with updated lambda
            for (int p = 0; p < num_pairs; ++p)
            {
                double xi = eps_pairs(p) - lambda;
                double Ei = std::sqrt(xi * xi + Delta(p) * Delta(p));
                if (Ei < EPS_SMALL)
                    Ei = EPS_SMALL;
                double vp2 = 0.5 * (1.0 - xi / Ei);
                vp2 = std::clamp(vp2, 0.0, 1.0); // Simplified clamping
                v(p) = std::sqrt(vp2);
                u(p) = std::sqrt(1.0 - vp2);
                kappa(p) = u(p) * v(p);
            }

            // --- 4) Update DeltaNew with a SMOOTH window function ---
            VectorXd DeltaNew = VectorXd::Zero(num_pairs);
            const double window_smoothness = window / 10.0; // Sharpness of the cutoff

            // Pre-calculate window weights for efficiency
            VectorXd window_weights(num_pairs);
            for (int p = 0; p < num_pairs; ++p)
            {
                window_weights(p) = 1.0 / (1.0 + std::exp((std::abs(eps_pairs(p) - lambda) - window) / window_smoothness));
            }

            for (int p = 0; p < num_pairs; ++p)
            {
                double sum = 0.0;
                if (window_weights(p) > EPS_SMALL)
                { // Only compute if state p is relevant
                    for (int q = 0; q < num_pairs; ++q)
                    {
                        if (window_weights(q) > EPS_SMALL)
                        { // Only include state q if relevant
                            sum += G(p, q) * kappa(q) * window_weights(q);
                        }
                    }
                    DeltaNew(p) = -sum * window_weights(p);
                }
                else
                {
                    DeltaNew(p) = 0.0;
                }
            }

            // 5) Clamp per-step Delta changes
            // for (int p = 0; p < num_pairs; ++p)
            //{
            //    double maxStep = maxDeltaStepFrac * std::max(0.2, std::abs(Delta(p)));
            //    double d = DeltaNew(p) - Delta(p);
            //    if (std::abs(d) > maxStep)
            //    {
            //        DeltaNew(p) = Delta(p) + std::copysign(maxStep, d);
            //    }
            //}

            // 6) Mix Delta for stability
            VectorXd DeltaMixed = (1.0 - mixDelta) * Delta + mixDelta * DeltaNew;

            // 7) Convergence tests
            double dnorm = (DeltaMixed - Delta).norm();
            double ldiff = std::abs(rawNewLambda - prevLambda);

            Delta = DeltaMixed;

            if (ldiff < PairingPrec && dnorm < tol)
            {
                // Final consistency calculation (optional but good practice)
                // ... (your final loop to re-calculate v, u, kappa)
                std::cout << "BCS Iterations: " << iter << std::endl;
                break;
            }
        }

        // ... Final calculations and return statement
        // (Your remaining code is fine)
        // Build outputs: final v2/u2, Delta, pairing energy etc.
        double mix = 1.0;
        Delta = mix * Delta + (1.0 - mix) * oldDelta;
        lambda = mix * lambda + (1.0 - mix) * oldLambda;
        VectorXd final_v2(num_pairs), final_u2(num_pairs);
        for (int p = 0; p < num_pairs; ++p)
        {
            double xi = eps_pairs(p) - lambda;
            double Ei = std::sqrt(xi * xi + Delta(p) * Delta(p));
            if (Ei < EPS_SMALL)
                Ei = EPS_SMALL;
            double vp2 = 0.5 * (1.0 - xi / Ei);
            vp2 = std::clamp(vp2, 0.0, 1.0);
            final_v2(p) = vp2;
            final_u2(p) = 1.0 - vp2;
        }

        // Final kappa for pairing energy
        kappa = final_u2.array().sqrt() * final_v2.array().sqrt();
        double Epair = -0.5 * kappa.dot(Delta);
        return {final_u2, final_v2, Delta, eps_pairs.array() * final_v2.array(), lambda, Epair};
    }

    // Assumes Eigen is included and BCSResult is defined as before
    BCSResult BCSiter(const MatrixXcd &phi, const VectorXd &eps,
                      int A, PairingParameters params, NucleonType t, const VectorXd &oldDelta, double oldLambda)
    {
        int M = phi.cols();
        // Step 1: Find time-reversal partners for all states
        auto O = pairs(phi);
        auto timeReversalPairs = match_time_reversal(O);

        // Step 2: Identify unique pairs and create necessary mappings
        std::vector<int> unique_pairs;
        unique_pairs.reserve(M / 2);
        std::vector<int> state_to_pair_map(M);
        std::vector<bool> visited(M, false);
        VectorXd oldDeltaUnique(M / 2);

        for (int i = 0; i < M; ++i)
        {
            if (!visited[i])
            {
                int partner = timeReversalPairs[i];
                if (partner != i && partner != -1)
                {
                    int pair_index = unique_pairs.size();
                    unique_pairs.push_back(i); // Add the first state of the pair

                    state_to_pair_map[i] = pair_index;
                    state_to_pair_map[partner] = pair_index;

                    visited[i] = true;
                    visited[partner] = true;
                }
            }
        }

        // Step 3: Prepare inputs for the pair-based BCS solver
        int num_pairs = unique_pairs.size();
        VectorXd eps_pairs(num_pairs);
        for (int p = 0; p < num_pairs; ++p)
        {
            // Energies are degenerate, so we take the energy of the first state in the pair
            eps_pairs(p) = eps(unique_pairs[p]);
        }
        for (int i = 0; i < oldDeltaUnique.size(); i++)
            oldDeltaUnique(i) = oldDelta(unique_pairs[i]);

        if (oldDelta.size() != num_pairs * 2)
            oldDeltaUnique = oldDelta;

        // Step 4: Compute the correct pairing matrix and solve BCS
        MatrixXd G_p = compute_G_pairing(phi, unique_pairs, timeReversalPairs, params, t);
        BCSResult pair_results = solveBCS(eps_pairs, G_p, A, params.window, oldDeltaUnique, oldLambda);

        // Step 5: "Unpack" the results from pairs back to the full M states
        VectorXd u2_full = VectorXd::Zero(M);
        VectorXd v2_full = VectorXd::Zero(M);
        VectorXd Delta_full = VectorXd::Zero(M);

        for (int i = 0; i < M; ++i)
        {
            if (visited[i]) // Only paired states will have non-zero results
            {
                int pair_idx = state_to_pair_map[i];
                u2_full(i) = pair_results.u2(pair_idx);
                v2_full(i) = pair_results.v2(pair_idx);
                Delta_full(i) = pair_results.Delta(pair_idx);
            }
        }

        // std::cout << v2_full.transpose() << std::endl;

        std::cout << "Converged Lambda: " << pair_results.lambda << ", Epair: " << pair_results.Epair << std::endl;
        BCSResult final_results = {u2_full, v2_full, Delta_full, eps.array() * v2_full.array(), pair_results.lambda, pair_results.Epair};
        return final_results;
    }
}