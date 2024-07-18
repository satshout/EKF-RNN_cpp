#ifndef KALMANFILTER_H
#define KALMANFILTER_H

#include <vector>
#include <Eigen/Dense>
#include "utils.h" // Include the utils header for AssimVector class

class Snapshot {
    public:
    Eigen::MatrixXd Pf;
    AssimVector wa;
    Eigen::MatrixXd Pa;
    
    // Constructor
    Snapshot(const Eigen::MatrixXd &Pf, const AssimVector &wa, const Eigen::MatrixXd &Pa) : Pf(Pf), wa(wa), Pa(Pa) {}
};

// Get LzdX
Eigen::MatrixXd get_LzdX(const AssimVector &X0, const AssimVector &dX);

// Get LzU
Eigen::MatrixXd get_LzU(const Eigen::MatrixXd &U, const AssimVector &X0);

// Get tangent linear model (TLM)
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> get_tlm(const Eigen::MatrixXd &xt, const Eigen::MatrixXd &h_prev,
                                                     const AssimVector &wt);

// Observation model
Eigen::MatrixXd Hobs(const Eigen::MatrixXd &yo_prev, const Eigen::MatrixXd &h_prev, const AssimVector &w);

// Kalman Filter
Snapshot KalmanFilter(const Snapshot &PW_prev,
                      const Eigen::MatrixXd &h_prev, const Eigen::MatrixXd &yo_prev, 
                      const Eigen::MatrixXd &H, const Eigen::MatrixXd &yo, const Eigen::MatrixXd &R);

// Forecast
std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd> Forecast(const Eigen::MatrixXd &xt, const Eigen::MatrixXd &h_prev, const Eigen::MatrixXd &Px,
                                                                       const AssimVector &wt, const Eigen::MatrixXd &Pw, const Eigen::MatrixXd &Lz, const Eigen::MatrixXd &H);

#endif // KALMANFILTER_H
