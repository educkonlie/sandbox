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

#ifdef BLOCK_SPARSE_MAT_test
void leastsquare_pcg_orig(MatXXd &M_inv, MatXXd &A, VecXd &b, VecXd &x, double tor, int maxiter)
{
    static int num_of_iter = 0;
    static int num_of_pcg = 0;
    num_of_pcg++;
    int i = 0;

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

//        VecXc s = lambda.asDiagonal() * r;
        VecXc s = M_inv * r;
        delta_old = delta_new;
        delta_new = r.transpose() * s;

        num_of_iter++;

        beta = delta_new / delta_old;
        d = s + beta * d;
        i++;
    }
    std::cout << "iters:        " << i << std::endl;
    std::cout << "total iters:  " << num_of_iter << std::endl;
    std::cout << "iters per pcg:" << num_of_iter / num_of_pcg << std::endl;
}
#endif

void leastsquare_pcg_BlockSparse(MatXXd &M_inv, BlockSparseMatrix<POSE_SIZE> &A, VecXd &b, VecXd &x, double tor, int maxiter)
{
    static int num_of_iter = 0;
    static int num_of_pcg = 0;
    num_of_pcg++;
    int i = 0;

//    int C = A.getC();

    x = VecXd::Zero(A.block_cols() * POSE_SIZE);

    VecXd Atb = VecXd::Zero(A.block_cols() * POSE_SIZE);
    A.transpose_right_multiply(b, Atb);
//    VecXc Atb2 = A_dense.transpose() * b;
//    std::cout << "Atb:  " << Atb.transpose() << std::endl;
//    std::cout << "Atb2: " << Atb2.transpose() << std::endl;
    VecXd r = Atb; // - A.transpose() * (A * x);
    VecXd d = M_inv * r;

    double delta_new = r.transpose() * d;
    double delta_0 = delta_new;
    double delta_old;
    double alpha;
    double beta;

    std::cout << "cg delta_0: " << delta_0 << std::endl;

    VecXd q = VecXd::Zero(A.block_cols() * POSE_SIZE);
    while (i < maxiter && delta_new > tor * tor * delta_0) {
//        VecXc q = A.transpose() * (A * d);
        A.AAq(d, q);
        alpha = delta_new / (d.transpose() * q);
        x = x + alpha * d;

        if (i % 50 == 0) {
            VecXd temp = VecXd::Zero(A.block_cols() * POSE_SIZE);
            A.AAq(x, temp);
            r = Atb - temp;
        } else {
            r = r - alpha * q;
        }

//        VecXc s = lambda.asDiagonal() * r;
        VecXd s = M_inv * r;
        delta_old = delta_new;
        delta_new = r.transpose() * s;

        num_of_iter++;

        beta = delta_new / delta_old;
        d = s + beta * d;
        i++;
    }
    std::cout << "iters:        " << i << std::endl;
    std::cout << "total iters:  " << num_of_iter << std::endl;
    std::cout << "iters per pcg:" << num_of_iter / num_of_pcg << std::endl;
}