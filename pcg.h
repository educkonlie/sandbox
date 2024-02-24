#pragma once
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <chrono>
#include <cstdio>
//#include <pthread.h>
#include "BlockSparseMatrix.h"
using namespace Eigen;
//using namespace Sophus;
using namespace std;

void leastsquare_pcg_orig(MatXXc &A, VecXc &b, VecXc &x, scalar tor, int maxiter)
{
    static int num_of_iter = 0;
    static int num_of_pcg = 0;
    num_of_pcg++;
    int i = 0;
//        MatXXc M_inv = (A.transpose() * A).diagonal().asDiagonal().inverse();
    VecXc lambda = VecXc::Zero(A.cols());
    for (int i = 0; i < A.cols(); i++)
        lambda(i) = 1.0 / (A.col(i).transpose() * A.col(i));
    MatXXc M_inv = lambda.asDiagonal();

    x = VecXc::Zero(A.cols());

    VecXc Atb = A.transpose() * b;
    VecXc r = Atb; // - A.transpose() * (A * x);
    VecXc d = M_inv * r;

    scalar delta_new = r.transpose() * d;
    scalar delta_0 = delta_new;
    scalar delta_old;
    scalar alpha;
    scalar beta;

    std::cout << "cg delta_0: " << delta_0 << std::endl;
    while (i < maxiter && delta_new > tor * tor * delta_0) {
        VecXc q = A.transpose() * (A * d);
        alpha = delta_new / (d.transpose() * q);
        x = x + alpha * d;

        if (i % 50 == 0)
            r = Atb - A.transpose() * (A * x);
        else
            r = r - alpha * q;

        VecXc s = lambda.asDiagonal() * r;
        delta_old = delta_new;
        delta_new = r.transpose() * s;

//            std::cout << "cg delta_new: " << delta_new << std::endl;
        num_of_iter++;

        beta = delta_new / delta_old;
        d = s + beta * d;
        i++;
    }
    std::cout << "iters:        " << i << std::endl;
    std::cout << "total iters:  " << num_of_iter << std::endl;
    std::cout << "iters per pcg:" << num_of_iter / num_of_pcg << std::endl;
}
