#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>

class AssimVector {
    public:
    Eigen::VectorXd vec;
    std::vector<Eigen::MatrixXd::Index> shapes;

    // Constructor by flattening the matrices into 1D vector
    AssimVector(const std::vector<Eigen::MatrixXd> &matrices);
    AssimVector(const Eigen::VectorXd &vec, 
                const std::vector<Eigen::MatrixXd::Index> &shapes);

    // Reconstruction of the matrices from the 1D vector
    std::vector<Eigen::MatrixXd> to_matrices() const;
};

#endif // UTILS_H
