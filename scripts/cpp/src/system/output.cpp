#include "util/output.hpp"

Output::Output()
: Output("output")
{
}
Output::Output(std::string folder_)
:folder(folder_)
{
}

void Output::matrixToFile(std::string fileName, Eigen::MatrixXd matrix)
{
    std::ofstream file(folder + "/" + fileName);
    file << matrix << std::endl;
    file.close();
}
void Output::shellsToFile(std::string fileName, std::pair<Eigen::MatrixXcd, Eigen::VectorXd> shells) {

    std::ofstream file(folder + "/" + fileName);
    file << "l,j,mj,P,energy_mev,deg" << std::endl;
    


}
