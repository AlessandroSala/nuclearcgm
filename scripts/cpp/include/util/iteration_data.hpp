#pragma once
#include <Eigen/Dense>
#include <memory>
class Grid;

class IterationData {
    public:
        std::shared_ptr<Eigen::VectorXd> rhoN;
        std::shared_ptr<Eigen::VectorXd> rhoP;
        std::shared_ptr<Eigen::VectorXd> tauN;
        std::shared_ptr<Eigen::VectorXd> tauP;
        std::shared_ptr<Eigen::MatrixX3d> nablaRhoN;
        std::shared_ptr<Eigen::MatrixX3d> nablaRhoP;
        std::shared_ptr<Eigen::VectorXd> nabla2RhoN;
        std::shared_ptr<Eigen::VectorXd> nabla2RhoP;
        std::shared_ptr<Eigen::VectorXcd> divJN;
        std::shared_ptr<Eigen::VectorXcd> divJP;
        std::shared_ptr<Eigen::MatrixX3cd> JN;
        std::shared_ptr<Eigen::MatrixX3cd> JP;

        void updateQuantities(const Eigen::MatrixXcd& neutrons, const Eigen::MatrixXcd& protons, int A, int Z, const Grid& grid);

        IterationData();

};