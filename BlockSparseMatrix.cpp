//
// Created by ruankefeng on 24-2-19.
//
#include "BlockSparseMatrix.h"

#ifdef BLOCK_SPARSE_MAT_test
#include <cstdlib>
#include <ctime>
#include "pcg.h"
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

#if 0
    VecXd q = VecXd::Random(9);
    std::cout << "q\n" << q.transpose() << std::endl;
    std::cout << "A * q\n" << (m * q).transpose() << std::endl;

    VecXd AAq = VecXd::Zero(9);
    mat.AAq(q, AAq);

    std::cout << "AAq\n" << AAq.transpose() << std::endl;
    std::cout << "A * A * q\n" << (m.transpose() * (m * q)).transpose() << std::endl;

    MatXXd *p;
    for (int i = 0; i < mat.block_rows(); i++)
        for (int j = 0; j < mat.block_cols(); j++) {
            if (mat.find(i, j, &p))
                std::cout << i << " " << j << ":\n" << *p << std::endl;
        }

    MatXXd diag[1000];
#pragma omp parallel for
    for (int j = 0; j < mat.block_cols(); j++) {
        for (int i = 0; i < mat.block_rows(); i++) {
            if (mat.find(i, j, &p)) {
                if (diag[j].size() == 0) {
                    diag[j] = (*p).transpose() * (*p);
                } else {
                    diag[j] += (*p).transpose() * (*p);
                }
            }
        }
    }
    for (int j = 0; j < mat.block_cols(); j++) {
        std::cout << "diag: " << j << "\n" << diag[j] << std::endl;
    }
    MatXXd diag2[1000];
    for (int j = 0; j < mat.block_cols(); j++)
        for (int i = 0; i < mat.block_rows(); i++) {
            if (mat.find(i, j, &p)) {
                if (diag2[j].size() == 0) {
                    diag2[j] = (*p).transpose() * (*p);
                } else {
                    diag2[j] += (*p).transpose() * (*p);
                }
            }
        }
    for (int j = 0; j < mat.block_cols(); j++) {
        std::cout << "diag2: " << j << "\n" << diag2[j] << std::endl;
    }
#endif
//    void leastsquare_pcg_BlockSparse(MatXXd &M_inv, BlockSparseMatrix<3> &A, VecXd &b, VecXd &x, double tor, int maxiter)

    MatXXd M_inv = MatXXd::Zero(8 * 3, 8 * 3);
    mat.get_M_inv(M_inv);
    VecXd x = VecXd::Zero(8 * 3);
    leastsquare_pcg_BlockSparse(M_inv, mat, b, x, 1e-3, 1000);
    std::cout << "pcg sparse:\n" << x.transpose() << std::endl;
    leastsquare_pcg_orig(M_inv, m, b, x, 1e-3, 1000);
    std::cout << "pcg dense:\n" << x.transpose() << std::endl;
    return 0;
}
#endif