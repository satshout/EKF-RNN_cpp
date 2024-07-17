#include "RNN.h"
//#include "utils.h"
#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;


int main()
{
    MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;
    
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

    auto htzt = RNNstep(xt, h_prev, Win, Wrec, b_rec, Wout, b_out);
    //VectorXd ht(2), zt(3);
    //ht = Win * xt + Wrec * h_prev + b_rec;
    //zt = Wout * ht + b_out;
    //std::pair<Eigen::MatrixXd, Eigen::MatrixXd> htzt = std::make_pair(ht, zt);
    //
    std::cout << "ht: " << std::endl << htzt.first << std::endl;
    std::cout << "zt: " << std::endl << htzt.second << std::endl;
}
