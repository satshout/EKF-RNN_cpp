#include "utils.h"
#include <Eigen/Dense>
#include <utility>
#include <vector>

// Constructor of the AssimVector class
AssimVector::AssimVector(const std::vector<Eigen::MatrixXd> &matrices) {
    shapes = std::vector<Eigen::MatrixXd::Index>();
    std::vector<double> concat;
    
    for (const auto &matrix : matrices) {
        Eigen::MatrixXd::Index rows = matrix.rows();
        Eigen::MatrixXd::Index cols = matrix.cols();
        shapes.push_back(rows);
        shapes.push_back(cols);
        
        for (int i = 0; i < matrix.size(); ++i) {
            concat.push_back(matrix.data()[i]);
        }
    }
    
    vec = Eigen::Map<Eigen::VectorXd>(concat.data(), concat.size());
}

AssimVector::AssimVector(const Eigen::VectorXd &vec, 
                   const std::vector<Eigen::MatrixXd::Index> &shapes) {
    this->vec = vec;
    this->shapes = shapes;
}

// Reconstruction of the matrices from the 1D vector
std::vector<Eigen::MatrixXd> AssimVector::to_matrices() const {
    std::vector<Eigen::MatrixXd> matrices;
    int index = 0;
    
    for (size_t i = 0; i < shapes.size(); i += 2) {
        Eigen::MatrixXd::Index rows = shapes.at(i);
        Eigen::MatrixXd::Index cols = shapes.at(i + 1);
        Eigen::MatrixXd matrix = Eigen::Map<const Eigen::MatrixXd>(vec.data() + index, rows, cols);
        matrices.push_back(matrix);
        index += rows * cols;
    }
    
    return matrices;
}
