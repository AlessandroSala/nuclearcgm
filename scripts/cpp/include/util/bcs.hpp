
#include <Eigen/Dense>
namespace BCS
{
    using namespace Eigen;

    typedef struct
    {
        VectorXd u;
        VectorXd v;
        VectorXd Delta;
        double lambda;
    } BCSResult;

    BCSResult BCSiter(const MatrixXcd &phi, const VectorXd &eps,
                      int A, double V0);
}