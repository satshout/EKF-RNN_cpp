#include "KalmanFilter.h"
#include <vector>
#include <Eigen/Dense>
#include "RNN.h"
#include "utils.h"

Eigen::MatrixXd get_LzdX(const AssimVector &X0, const AssimVector &dX) {
    std::vector<Eigen::MatrixXd> X0_matrices = X0.to_matrices();
    std::vector<Eigen::MatrixXd> dX_matrices = dX.to_matrices();
    Eigen::MatrixXd dz = RNNstep_tlm(X0_matrices.at(0), X0_matrices.at(1), X0_matrices.at(2), X0_matrices.at(3), X0_matrices.at(4), X0_matrices.at(5), X0_matrices.at(6),
                                     dX_matrices.at(0), dX_matrices.at(1), dX_matrices.at(2), dX_matrices.at(3), dX_matrices.at(4), dX_matrices.at(5), dX_matrices.at(6));
    return dz;
}

Eigen::MatrixXd get_LzU(const Eigen::MatrixXd &U, const AssimVector &X0) {
    std::vector<Eigen::MatrixXd> concat;
    AssimVector dX(U.col(0), X0.shapes);
    for (int i = 0; i < U.cols(); ++i) {
        dX = AssimVector(U.col(i), X0.shapes);
        Eigen::MatrixXd Vi = get_LzdX(X0, dX);
        concat.push_back(Vi);
    }
    Eigen::MatrixXd LzU(concat.at(0).rows(), U.cols());
    for (int i = 0; i < U.cols(); ++i) {
        LzU.col(i) = concat.at(i);
    }
    return LzU;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> get_tlm(const Eigen::MatrixXd &xt, const Eigen::MatrixXd &h_prev,
                                                     const AssimVector &wt) {
    auto wt_matrices = wt.to_matrices();
    AssimVector X0({xt, h_prev, wt_matrices[0], wt_matrices.at(1), wt_matrices.at(2), wt_matrices.at(3), wt_matrices.at(4)});
    
    Eigen::MatrixXd E = Eigen::MatrixXd::Identity(X0.vec.size(), X0.vec.size());
    Eigen::MatrixXd TLM = get_LzU(E, X0);
    
    Eigen::MatrixXd Lz = TLM.leftCols(xt.size()); // dx1/dx2
    Eigen::MatrixXd Lh = TLM.block(0, xt.size(), TLM.rows(), h_prev.size()); // 
    Eigen::MatrixXd H = TLM.rightCols(TLM.cols() - xt.size() - h_prev.size());
    
    return std::make_tuple(Lz, H);
}

Eigen::MatrixXd Hobs(const Eigen::MatrixXd &yo_prev, const Eigen::MatrixXd &h_prev, const AssimVector &w) {
    std::vector<Eigen::MatrixXd> w_matrices = w.to_matrices();
    Eigen::VectorXd ht, zt;
    std::tie(ht, zt) = RNNstep(yo_prev, h_prev, w_matrices[0], w_matrices[1], w_matrices[2], w_matrices[3], w_matrices[4]);
    return zt;
}

Snapshot KalmanFilter(const Snapshot &PW_prev,
                                                                           const Eigen::MatrixXd &h_prev, const Eigen::MatrixXd &yo_prev, 
                                                                           const Eigen::MatrixXd &H, const Eigen::MatrixXd &yo, const Eigen::MatrixXd &R) {
    Eigen::MatrixXd Pa_prev_sym = (PW_prev.Pa + PW_prev.Pa.transpose()) / 2.0;
    
    // Forecast
    AssimVector wf = PW_prev.wa;
    Eigen::MatrixXd Pf = Pa_prev_sym;
    
    // Analysis
    Eigen::MatrixXd K = Pf * H.transpose() * (H * Pf * H.transpose() + R).inverse();
    
    Eigen::VectorXd wa_vec = wf.vec + K * (yo - Hobs(yo_prev, h_prev, wf));
    
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(wa_vec.size(), wa_vec.size());
    Eigen::MatrixXd Pa = (I - K * H) * Pf;

    return Snapshot(Pf, AssimVector(wa_vec, wf.shapes), Pa);
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd> Forecast(const Eigen::MatrixXd &xt, const Eigen::MatrixXd &h_prev, const Eigen::MatrixXd &Px,
                                                                       const AssimVector &wt, const Eigen::MatrixXd &Pw, const Eigen::MatrixXd &Lz, const Eigen::MatrixXd &H) {
    std::vector<Eigen::MatrixXd> wt_matrices = wt.to_matrices();
    Eigen::VectorXd ht, zt;
    std::tie(ht, zt) = RNNstep(xt, h_prev, wt_matrices.at(0), wt_matrices.at(1), wt_matrices.at(2), wt_matrices.at(3), wt_matrices.at(4));
    
    // Forecast covariance
    Eigen::MatrixXd Pz = Lz * Px * Lz.transpose() + H * Pw * H.transpose();
    
    return std::make_tuple(ht, zt, Pz);
}
