#pragma once
#include <fstream>
#include <Eigen/Dense>
class Output
{
private:
    std::string folder;

public:
    Output();
    Output(std::string folder_);
    void matrixToFile(std::string fileName, Eigen::MatrixXd matrix);
    void shellsToFile(std::string fileName, std::pair<Eigen::MatrixXcd, Eigen::VectorXd> shells);
};
