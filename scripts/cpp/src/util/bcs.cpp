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

    // Assumes Eigen is included and BCSResult is defined as before
    BCSResult solveBCS(const VectorXd &eps_pairs, const MatrixXd &G_pairing,
                       int A, double Ecut, const VectorXd &oldDelta, double oldLambda,
                       double initDelta = 1.0, double tol = 1e-6, int maxIter = 200,
                       double smoothWidth = 0.5)
    {
        using std::abs;
        const double EPS_SMALL = 1e-12;

        int num_pairs = eps_pairs.size();
        if (num_pairs == 0)
            return {};

        // Initialize Delta
        VectorXd Delta = (oldDelta.size() == num_pairs) ? oldDelta : VectorXd::Constant(num_pairs, initDelta);

        // Initialize lambda estimate
        double lambda = (oldLambda == oldLambda) ? oldLambda : eps_pairs(A / 2);

        VectorXd kappa(num_pairs);

        // Iteration parameters
        int lambdaBisections = 60;

        for (int iter = 0; iter < maxIter; ++iter)
        {
            // 1) Solve for lambda to enforce particle number N = A with the current Delta
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

            // --- Find lambda via bisection ---
            double lamLo = eps_pairs.minCoeff() - 2 * Ecut;
            double lamHi = eps_pairs.maxCoeff() + 2 * Ecut;
            for (int b = 0; b < lambdaBisections; ++b)
            {
                double lamMid = 0.5 * (lamLo + lamHi);
                if (compute_N(lamMid) > A)
                    lamHi = lamMid;
                else
                    lamLo = lamMid;
            }
            lambda = 0.5 * (lamLo + lamHi);

            // 2) Compute kappa = u*v for the current lambda and Delta
            for (int p = 0; p < num_pairs; ++p)
            {
                double xi = eps_pairs(p) - lambda;
                double Ei = std::sqrt(xi * xi + Delta(p) * Delta(p));
                if (Ei < EPS_SMALL)
                    Ei = EPS_SMALL;
                double vp2 = 0.5 * (1.0 - xi / Ei);
                kappa(p) = std::sqrt(vp2 * (1.0 - vp2));
            }

            // 3) Smooth cutoff centered around lambda
            VectorXd f(num_pairs);
            for (int p = 0; p < num_pairs; ++p)
            {
                double arg = (abs(eps_pairs(p) - lambda) - Ecut) / smoothWidth;
                if (arg > 50.0)
                    f(p) = 0.0;
                else
                    f(p) = 1.0 / (1.0 + std::exp(arg));
            }

            // 4) Build new Delta from gap equation (no mixing)
            VectorXd DeltaNew = VectorXd::Zero(num_pairs);
            for (int p = 0; p < num_pairs; ++p)
            {
                if (f(p) < EPS_SMALL)
                    continue;

                double sum = 0.0;
                for (int q = 0; q < num_pairs; ++q)
                    sum += f(p) * G_pairing(p, q) * f(q) * kappa(q);

                DeltaNew(p) = -sum;
            }

            // 5) Convergence check
            double dnorm = (DeltaNew - Delta).norm();
            Delta = DeltaNew;

            if (dnorm < tol)
                break;
        }

        // 6) Final occupations
        VectorXd final_v2(num_pairs), final_u2(num_pairs);
        for (int p = 0; p < num_pairs; ++p)
        {
            double xi = eps_pairs(p) - lambda;
            double Ei = std::sqrt(xi * xi + Delta(p) * Delta(p));
            if (Ei < EPS_SMALL)
                Ei = EPS_SMALL;
            double vp2 = 0.5 * (1.0 - xi / Ei);
            final_v2(p) = vp2;
            final_u2(p) = 1.0 - vp2;
        }

        double final_Epair = -Delta.dot(kappa);

        return {final_u2, final_v2, Delta, eps_pairs.array() * final_v2.array(), lambda, final_Epair};
    }

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

        std::cout << v2_full.transpose() << std::endl;

        BCSResult final_results = {u2_full, v2_full, Delta_full, eps.array() * v2_full.array(), pair_results.lambda, pair_results.Epair};
        return final_results;
    }
}