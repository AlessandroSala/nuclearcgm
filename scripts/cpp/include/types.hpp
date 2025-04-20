
#pragma once

#include<Eigen/Dense>
#include<Eigen/Sparse>

//Complex data types
typedef std::complex<double> ComplexScalar;
typedef Eigen::Matrix<ComplexScalar, Eigen::Dynamic, Eigen::Dynamic> ComplexDenseMatrix;
typedef Eigen::Matrix<ComplexScalar, Eigen::Dynamic, 1> ComplexDenseVector;
typedef Eigen::SparseMatrix<ComplexScalar> ComplexSparseMatrix;
typedef Eigen::Matrix<ComplexScalar, 2, 2> SpinMatrix;

//Real data types
typedef Eigen::MatrixXd DenseMatrix;
typedef Eigen::VectorXd DenseVector;
typedef Eigen::SparseMatrix<double> RealSparseMatrix;