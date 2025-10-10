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
        MatrixXd G_p = MatrixXd::Zero(num_pairs, num_pairs);

        for (int p = 0; p < num_pairs; ++p)
        {
            for (int q = 0; q < num_pairs; ++q)
            {
                int i = unique_pairs[p];
                int i_bar = all_partners[i];
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
                                    std::complex<double> c = phi(up, i_bar);
                                    std::complex<double> d = phi(down, i_bar);
                                    std::complex<double> e = phi(up, j);
                                    std::complex<double> f = phi(down, j);
                                    std::complex<double> g = phi(up, j_bar);
                                    std::complex<double> h = phi(down, j_bar);

                                    int P_idx = grid.idxNoSpin(x, y, z);

                                    P_i(P_idx) = a * d - b * c;
                                    P_j(P_idx) = e * h - f * g;
                                }
                            }
                        }

                // integrand to integrate:
                VectorXcd integrand = P_i.conjugate().cwiseProduct(P_j);

                double V0 = t == NucleonType::N ? params.V0N : params.V0P;
                G_p(p, q) = V0 * Operators::integralNoSpin(integrand, grid).real();
            }
        }
        return G_p;
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
                       int A, double window, const VectorXd &oldDelta, double oldLambda, double initDelta = 1.0,
                       double tol = 1e-6, int maxIter = 500)
    {
        using std::abs;
        const double EPS_SMALL = 1e-12;

        int num_pairs = eps_pairs.size();
        if (num_pairs == 0)
            return {}; // empty

        // Initialize Delta: prefer oldDelta if provided and same size
        double G_avg = G_pairing.diagonal().mean();

        initDelta = std::max(0.1, 0.5 * G_avg); // Adaptive initial gap

        VectorXd Delta = VectorXd::Constant(num_pairs, initDelta);
        if (oldDelta.size() == num_pairs)
        {
            std::cout << "Initializing to guess delta" << std::endl;
            Delta = oldDelta;
        }

        // Initial guess for lambda: try centre around Fermi index but safe-guard indexing
        double lambda = oldLambda;
        if (!(lambda == lambda))
        { // NaN check: if oldLambda not sensible, estimate
            int mid = std::max(0, std::min(num_pairs - 1, A / 2));
            lambda = eps_pairs(mid);
        }

        VectorXd kappa = VectorXd::Zero(num_pairs);
        VectorXd v = VectorXd::Zero(num_pairs);
        VectorXd u = VectorXd::Zero(num_pairs);

        // mixing parameters (tune these if you need stronger/softer mixing)
        double mixDelta = 0.5;         // mixing applied each iteration for Delta
        double mixLambda = 1.0;        // set <1.0 to underrelax lambda updates (we solve lambda exactly each iter)
        double maxDeltaStepFrac = 0.5; // limit per-iteration fractional change in Delta

        for (int iter = 0; iter < maxIter; ++iter)
        {
            // 1) Given current Delta, solve lambda (chemical potential) so that N = A
            auto compute_N = [&](double lam) -> double
            {
                double Nsum = 0.0;
                for (int p = 0; p < num_pairs; ++p)
                {

                    double xi = eps_pairs(p) - lam;
                    double Ei = std::sqrt(xi * xi + Delta(p) * Delta(p));
                    if (Ei < EPS_SMALL)
                        Ei = EPS_SMALL;
                    double vp2 = 0.5 * (1.0 - xi / Ei);
                    Nsum += 2.0 * vp2;
                }
                return Nsum;
            };
            // bracket lambda
            double lamLo = eps_pairs.minCoeff() - 4.0 * window - 10.0;
            double lamHi = eps_pairs.maxCoeff() + 4.0 * window + 10.0;
            double Nlo = compute_N(lamLo);
            double Nhi = compute_N(lamHi);

            // Expand bracket if necessary (robust)
            int expand_count = 0;
            while (Nlo > A && expand_count++ < 100)
            {
                lamLo -= 2.0 * window + 10.0;
                Nlo = compute_N(lamLo);
            }
            expand_count = 0;
            while (Nhi < A && expand_count++ < 100)
            {
                lamHi += 2.0 * window + 10.0;
                Nhi = compute_N(lamHi);
            }

            // Now bisection (robust and monotonic)
            double lamMid = lambda;
            for (int b = 0; b < 100; ++b)
            {
                lamMid = 0.5 * (lamLo + lamHi);
                double Nmid = compute_N(lamMid);
                if (std::abs(Nmid - A) < 1e-8)
                    break;
                if (Nmid > A)
                    lamHi = lamMid;
                else
                    lamLo = lamMid;
            }

            // Optionally underrelax lambda change (but we solved it robustly so full update is fine)
            double newLambda = lamMid;
            lambda = mixLambda * newLambda + (1.0 - mixLambda) * lambda;

            // 2) Recompute u, v, kappa with the solved lambda
            for (int p = 0; p < num_pairs; ++p)
            {
                double xi = eps_pairs(p) - lambda;
                double Ei = std::sqrt(xi * xi + Delta(p) * Delta(p));
                if (Ei < EPS_SMALL)
                    Ei = EPS_SMALL;
                double vp2 = 0.5 * (1.0 - xi / Ei);
                // ensure numerical bound
                if (vp2 < 0.0)
                    vp2 = 0.0;
                if (vp2 > 1.0)
                    vp2 = 1.0;
                v(p) = std::sqrt(vp2);
                u(p) = std::sqrt(1.0 - vp2);
                kappa(p) = u(p) * v(p);
            }

            // 3) Update DeltaNew from pairing interaction within window
            VectorXd DeltaNew = VectorXd::Zero(num_pairs);
            for (int p = 0; p < num_pairs; ++p)
            {
                if (std::abs(eps_pairs(p) - lambda) < window)
                {
                    double sum = 0.0;
                    for (int q = 0; q < num_pairs; ++q)
                    {
                        if (std::abs(eps_pairs(q) - lambda) < window)
                        {
                            sum += G_pairing(p, q) * kappa(q);
                        }
                    }
                    DeltaNew(p) = -sum;
                }
                else
                {
                    DeltaNew(p) = 0.0;
                }
            }

            // 4) Clamp per-step Delta changes to avoid huge jumps (safety)
            for (int p = 0; p < num_pairs; ++p)
            {
                double maxStep = maxDeltaStepFrac * std::max(1.0, std::abs(Delta(p)));
                double d = DeltaNew(p) - Delta(p);
                if (std::abs(d) > maxStep)
                {
                    DeltaNew(p) = Delta(p) + std::copysign(maxStep, d);
                }
            }

            // 5) Mix Delta for stability
            VectorXd DeltaMixed = (1.0 - mixDelta) * Delta + mixDelta * DeltaNew;

            // 6) Convergence test: check Delta change (and optionally lambda)
            double dnorm = (DeltaMixed - Delta).norm();
            double ldiff = std::abs(lambda - newLambda); // newLambda was the 'raw' solved one before underrelax
            Delta = DeltaMixed;

            if (dnorm < tol && ldiff < 1e-8)
            {
                // Recompute kappa with final Delta/lambda to produce consistent outputs
                for (int p = 0; p < num_pairs; ++p)
                {
                    double xi = eps_pairs(p) - lambda;
                    double Ei = std::sqrt(xi * xi + Delta(p) * Delta(p));
                    if (Ei < EPS_SMALL)
                        Ei = EPS_SMALL;
                    double vp2 = 0.5 * (1.0 - xi / Ei);
                    if (vp2 < 0.0)
                        vp2 = 0.0;
                    if (vp2 > 1.0)
                        vp2 = 1.0;
                    v(p) = std::sqrt(vp2);
                    u(p) = std::sqrt(1.0 - vp2);
                    kappa(p) = u(p) * v(p);
                }
                break;
            }

            // continue iterations
        } // end SCF loop

        // Optional smoothing with the provided oldDelta/oldLambda (kept but smaller mixing)
        if (oldDelta.size() == num_pairs)
        {
            double finalMix = 0.25;
            lambda = (1.0 - finalMix) * oldLambda + finalMix * lambda;
            Delta = (1.0 - finalMix) * oldDelta + finalMix * Delta;
            // recompute v/u/kappa after smoothing
            for (int p = 0; p < num_pairs; ++p)
            {
                double xi = eps_pairs(p) - lambda;
                double Ei = std::sqrt(xi * xi + Delta(p) * Delta(p));
                if (Ei < EPS_SMALL)
                    Ei = EPS_SMALL;
                double vp2 = 0.5 * (1.0 - xi / Ei);
                if (vp2 < 0.0)
                    vp2 = 0.0;
                if (vp2 > 1.0)
                    vp2 = 1.0;
                v(p) = std::sqrt(vp2);
                u(p) = std::sqrt(1.0 - vp2);
                kappa(p) = u(p) * v(p);
            }
        }

        // Build outputs: final v2/u2, Delta, pairing energy etc.
        VectorXd final_v2(num_pairs), final_u2(num_pairs);
        for (int p = 0; p < num_pairs; ++p)
        {
            double xi = eps_pairs(p) - lambda;
            double Ei = std::sqrt(xi * xi + Delta(p) * Delta(p));
            if (Ei < EPS_SMALL)
                Ei = EPS_SMALL;
            double vp2 = 0.5 * (1.0 - xi / Ei);
            if (vp2 < 0.0)
                vp2 = 0.0;
            if (vp2 > 1.0)
                vp2 = 1.0;
            final_v2(p) = vp2;
            final_u2(p) = 1.0 - vp2;
        }

        double avgGap = 0.0;
        double denom = final_v2.sum();
        if (denom > EPS_SMALL)
            avgGap = final_v2.dot(Delta) / denom;
        else
            avgGap = Delta.mean();

        // Pairing energy: commonly E_pair = -0.5 * sum_p Delta_p * kappa_p (depends on your convention)
        double Epair = -0.5 * kappa.dot(Delta);

        std::cout << "Converged Lambda: " << lambda << ", Avg gap: " << avgGap << ", Epair: " << Epair << std::endl;
        std::cout << "Delta: " << Delta.transpose() << std::endl;

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
                // This check handles cases where a state might be its own partner
                // or if no partner was found (-1). Such states don't form Cooper pairs.
                if (partner != i && partner != -1)
                {
                    int pair_index = unique_pairs.size();
                    unique_pairs.push_back(i); // Add the first state of the pair
                    oldDeltaUnique(pair_index) = oldDelta(i);

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

        // Step 4: Compute the correct pairing matrix and solve BCS
        MatrixXd G_p = compute_G_pairing(phi, unique_pairs, timeReversalPairs, params, t);
        BCSResult pair_results = solveBCS(eps_pairs, G_p, A, params.window, oldDelta, oldLambda);

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

        BCSResult final_results = {u2_full, v2_full, Delta_full, eps.array() * v2_full.array(), pair_results.lambda, pair_results.Epair};
        return final_results;
    }
}