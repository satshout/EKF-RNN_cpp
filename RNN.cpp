#include "RNN.h"
#include <Eigen/Dense>
#include <utility>
#include <vector>

// Define activation functions
Eigen::MatrixXd tanh(const Eigen::MatrixXd &x) {
    return x.array().tanh();
}

Eigen::MatrixXd tanh_prime(const Eigen::MatrixXd &x) {
    return 1.0 - x.array().tanh().square();
}

Eigen::MatrixXd id(const Eigen::MatrixXd &x) {
    return x;
}


Eigen::MatrixXd id_prime(const Eigen::MatrixXd &x) {
    return Eigen::MatrixXd::Ones(x.rows(), x.cols());
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> RNNstep(const Eigen::MatrixXd &xt, 
                                                    const Eigen::MatrixXd &h_prev, 
                                                    const Eigen::MatrixXd &Win, 
                                                    const Eigen::MatrixXd &Wrec, 
                                                    const Eigen::MatrixXd &b_rec, 
                                                    const Eigen::MatrixXd &Wout, 
                                                    const Eigen::MatrixXd &b_out, 
                                                    Eigen::MatrixXd (*h_actF)(const Eigen::MatrixXd &),
                                                    Eigen::MatrixXd (*z_actF)(const Eigen::MatrixXd &)) {
    Eigen::VectorXd ht = h_actF(Win * xt + Wrec * h_prev + b_rec);
    Eigen::VectorXd zt = z_actF(Wout * ht + b_out);
    return std::make_pair(ht, zt);
}

Eigen::VectorXd RNNstep_tlm(const Eigen::MatrixXd &xt, 
                            const Eigen::MatrixXd &h_prev, 
                            const Eigen::MatrixXd &Win, 
                            const Eigen::MatrixXd &Wrec, 
                            const Eigen::MatrixXd &b_rec, 
                            const Eigen::MatrixXd &Wout, 
                            const Eigen::MatrixXd &b_out, 
                            const Eigen::MatrixXd &dx, 
                            const Eigen::MatrixXd &dh_prev, 
                            const Eigen::MatrixXd &dWin, 
                            const Eigen::MatrixXd &dWrec, 
                            const Eigen::MatrixXd &db_rec, 
                            const Eigen::MatrixXd &dWout, 
                            const Eigen::MatrixXd &db_out, 
                            Eigen::MatrixXd (*h_actF)(const Eigen::MatrixXd &),
                            Eigen::MatrixXd (*z_actF)(const Eigen::MatrixXd &),
                            Eigen::MatrixXd (*h_actF_prime)(const Eigen::MatrixXd &),
                            Eigen::MatrixXd (*z_actF_prime)(const Eigen::MatrixXd &)) {
    Eigen::VectorXd w_rec = Win * xt + Wrec * h_prev + b_rec;
    Eigen::VectorXd dw_rec = (dWin * xt + Win * dx) + (dWrec * h_prev + Wrec * dh_prev) + db_rec;
    Eigen::VectorXd ht = h_actF(w_rec);
    Eigen::VectorXd dh = h_actF_prime(w_rec).array() * dw_rec.array();

    Eigen::VectorXd dz = z_actF_prime(Wout * ht + b_out).array() * (dWout * ht + Wout * dh + db_out).array();
    return dz;
}
