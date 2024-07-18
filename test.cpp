#include "RNN.h"
#include "utils.h"
#include "KalmanFilter.h"
#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;


int main()
{
    // Eigen Test -------------------
    MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;


    // RNN Test ---------------------
    VectorXd xt(3), h_prev(2), b_rec(2), b_out(3);
    MatrixXd Win(2, 3), Wrec(2, 2), Wout(3, 2);
    xt << 0.5, 0.2, 0.8;
    h_prev << 0.1, 0.3;
    Win << 0.2, 0.3, 0.1, 
           0.4, 0.5, 0.6;
    Wrec << 0.5, 0.4, 
            0.1, 0.6;
    b_rec << 0.1, 0.2;
    Wout << 0.5, 0.4, 
            0.2, 0.1, 
            0.7, 0.8;
    b_out << 0.1, 0.2, 0.4;

    std::cout << "W_in: " << std::endl << Win << std::endl;
    std::cout << "xt" << std::endl << xt << std::endl;
    std::cout << "Win x xt" << std::endl << Win * xt << std::endl;

    VectorXd ht0(2), zt0(3);
    std::tie(ht0, zt0) = RNNstep(xt, h_prev, Win, Wrec, b_rec, Wout, b_out);
    //VectorXd ht(2), zt(3);
    //ht = Win * xt + Wrec * h_prev + b_rec;
    //zt = Wout * ht + b_out;
    //pair<Eigen::MatrixXd, Eigen::MatrixXd> htzt = make_pair(ht, zt);
    //
    std::cout << "ht: " << std::endl << ht0 << std::endl;
    std::cout << "zt: " << std::endl << zt0 << std::endl;


    // Kalman Filter Test ---------------------
    VectorXd yo_prev(3), yo(3);
    MatrixXd R(3, 3);
    yo_prev << 0.1, 0.2, 0.3;
    yo << 0.2, 0.3, 0.4;
    R = 0.1 * MatrixXd::Identity(3, 3);

    std::vector<MatrixXd> matrices{Win, Wrec, b_rec, Wout, b_out};
    std::cout << "matrices[0]: " << std::endl << matrices.at(0);

    AssimVector wa_prev(matrices);
    std::cout << "wa_prev.vec: " << std::endl << wa_prev.vec << std::endl;

    int Nw = wa_prev.vec.size();
    MatrixXd Pf_0(Nw, Nw), Pa_prev(Nw, Nw);
    Pa_prev = MatrixXd::Identity(Nw, Nw);
    Pf_0 = MatrixXd::Zero(Nw, Nw);

    Snapshot PW_prev(Pf_0, wa_prev, Pa_prev);


    MatrixXd Lz, H;
    std::tie(Lz, H) = get_tlm(yo_prev, h_prev, wa_prev);
    std::cout << "Pa_prev: " << std::endl << Pa_prev << std::endl;
    std::cout << "H.shape: " << H.rows() << "x" << H.cols() << std::endl;


    Snapshot PW = KalmanFilter(PW_prev, h_prev, yo_prev, H, yo, R);
    std::cout << "PW.Pf: " << std::endl << PW.Pf << std::endl;
    std::cout << "PW.wa.vec: " << std::endl << PW.wa.vec << std::endl;
    std::cout << "PW.Pa: " << std::endl << PW.Pa << std::endl;

    Eigen::VectorXd ht, zt;
    Eigen::MatrixXd Pz;
    std::tie(ht, zt, Pz) = Forecast(yo_prev, h_prev, R, PW.wa, PW.Pa, Lz, H);

    /*
    */
}
