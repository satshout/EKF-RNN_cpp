#ifndef RNN_H
#define RNN_H

#include <Eigen/Dense>

// Define activation functions
Eigen::MatrixXd tanh(const Eigen::MatrixXd &x);
Eigen::MatrixXd tanh_prime(const Eigen::MatrixXd &x);
Eigen::MatrixXd id(const Eigen::MatrixXd &x);
Eigen::MatrixXd id_prime(const Eigen::MatrixXd &x);

// RNN step function
std::pair<Eigen::VectorXd, Eigen::VectorXd> RNNstep(const Eigen::MatrixXd &xt, 
                                                    const Eigen::MatrixXd &h_prev, 
                                                    const Eigen::MatrixXd &Win, 
                                                    const Eigen::MatrixXd &Wrec, 
                                                    const Eigen::MatrixXd &b_rec, 
                                                    const Eigen::MatrixXd &Wout, 
                                                    const Eigen::MatrixXd &b_out, 
                                                    Eigen::MatrixXd (*h_actF)(const Eigen::MatrixXd &) = tanh, 
                                                    Eigen::MatrixXd (*z_actF)(const Eigen::MatrixXd &) = id);

// RNN tangent linear model function
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
                            Eigen::MatrixXd (*h_actF)(const Eigen::MatrixXd &) = tanh, 
                            Eigen::MatrixXd (*z_actF)(const Eigen::MatrixXd &) = id, 
                            Eigen::MatrixXd (*h_actF_prime)(const Eigen::MatrixXd &) = tanh_prime, 
                            Eigen::MatrixXd (*z_actF_prime)(const Eigen::MatrixXd &) = id_prime);

#endif // RNN_H
