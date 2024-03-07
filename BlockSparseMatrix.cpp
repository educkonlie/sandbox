//
// Created by ruankefeng on 24-2-19.
//
#include "BlockSparseMatrix.h"

#ifdef BLOCK_SPARSE_MAT_test
#if 1
#include <cstdlib>
#include <ctime>
//#include "pcg.h"
int main()
{
    srand(static_cast<unsigned int>(time(0)));

    BlockSparseMatrix<3> mat(100, 8);
    /// tt用来为mat做存储，因为mat内部是不占用堆的
    /// MatXXd在没有初始化的时候也不占用堆
    std::vector <MatXXd> tt;
    for (int i = 0; i < 1000; i++) {
        MatXXd temp;
        tt.push_back(temp);
    }
    int k = 0;
    for (int i = 0; i < 100; i++) {
        int rows_per_block = rand() % 3 + 1;
//        std::cout << "rows: " << rows_per_block << std::endl;
        for (int j = 0; j < 8; j++) {
            if (rand() % 2 == 1) {
                tt[k] = MatXXd::Random(rows_per_block, 3);
//                std::cout << "\tcols: " << j << std::endl;
//                std::cout << "\ttemp\n" << tt[k] << std::endl;
                mat.add(i, j, tt[k]);
                k++;
            }
        }
        mat.set_startRow_of_block(i, rows_per_block);
    }
    MatXXd m = MatXXd::Zero(mat.rows(), 8 * 3);
    mat.toDenseMatrix(m);
    std::cout << "mat:\n" << m << std::endl;

    VecXd b = VecXd::Random(mat.rows());

    std::cout << "ldlt:\n" << (m.transpose() * m).ldlt().solve(m.transpose() * b).transpose() << std::endl;

//    void leastsquare_pcg_BlockSparse(MatXXd &M_inv, BlockSparseMatrix<3> &A, VecXd &b, VecXd &x, double tor, int maxiter)

    MatXXd M_inv = MatXXd::Zero(8 * 3, 8 * 3);
    MatXXd M_inv2 = MatXXd::Identity(8 * 3, 8 * 3);
    mat.get_M_inv(M_inv);
    VecXd x = VecXd::Zero(8 * 3);
    mat.leastsquare_pcg_BlockSparse(M_inv, b, x, 1e-8, 1000);
    std::cout << "pcg sparse:\n" << x.transpose() << std::endl;
//    mat.leastsquare_pcg_BlockSparse(M_inv2, b, x, 1e-8, 1000);
//    std::cout << "pcg sparse Identity:\n" << x.transpose() << std::endl;
    x.setZero();
    mat.least_square_conjugate_gradient(M_inv, b, x, 1000, 1e-8);
    std::cout << "pcg new:\n" << x.transpose() << std::endl;
//    VecXd b2(b.rows());
//    mat.right_multiply(x, b2);
//    std::cout << "Ax = :" << b2.transpose() << std::endl;


//    leastsquare_pcg_orig(M_inv, m, b, x, 1e-3, 1000);
//    std::cout << "pcg dense:\n" << x.transpose() << std::endl;
    return 0;
}
#endif
#endif

#if 0
#include <cstdlib>
#include <ctime>
//#include <Eigen/Core>
#include <Eigen/Sparse>
#include <random>
//#include "pcg.h"
int main()
{
    std::srand(static_cast<unsigned int>(time(0)));

    Eigen::SparseMatrix<double, Eigen::RowMajor> mat(1000, 1000);
    std::vector<Eigen::Triplet<double> > triplets;

    for (int i = 0; i < 6000; i++)
        triplets.push_back(Eigen::Triplet<double>(rand() % 1000, rand() % 1000, rand() * 0.395));

    mat.setFromTriplets(triplets.begin(), triplets.end());
    VecXd p = VecXd::Random(1000);
    VecXd q =  mat.adjoint() * p;

    std::cout << q.transpose() << std::endl;
    return 0;
}
#endif