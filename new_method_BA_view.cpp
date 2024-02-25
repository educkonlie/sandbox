//
// Created by ruankefeng on 24-2-6.
//
// used for test for new_method_BA
#if 0
#include "new_method_BA.h"
#include <random>

#include <cstdlib>
#include <ctime>
//#include "pcg.h"
int main()
{
    srand(static_cast<unsigned int>(time(0)));

    Eigen::Matrix<Mat33d, Eigen::Dynamic, Eigen::Dynamic> m = Eigen::Matrix<Mat33d, Eigen::Dynamic, Eigen::Dynamic>::Random(3, 3);

    Eigen::SparseMatrix<double> J(1000, 500);
    VecXd r;
    std::cout << m(0, 1) << std::endl;
    int rand_num = rand() % 5;
//    BlockSparseMatrix<3, 4> mat(10, 5);
//    MatXXd temp;
//    temp.conservativeResize(3, 4);
    for (int i = 0; i < 30; i++) {
        MatXXd temp = MatXXd::Random(3, 4);
//        temp.conservativeResize(3, 4);
        temp(0, 0) = i;
        temp(2, 3) = i * 2;
        mat.add(rand() % 2, rand() % 5, temp);
    }
    MatXXd m = MatXXd::Zero(3 * 10, 4 * 5);
    mat.toDenseMatrix(m);
    std::cout << "mat:\n" << m << std::endl;

    /*std::vector<mySparseRowVec<6>> new_mat;
    for (int i = 0; i < 30; i++) {
        mySparseRowVec<6> rowVec(3);
        VecXd block = VecXd::Random(6);
        rowVec.add(rand() % 3, block);
        new_mat.push_back(rowVec);
    }

    VecXd temp;
    for (int i = 0; i < 30; i++) {
        temp = VecXd::Zero(3 * 6);
        new_mat[i].toDenseVec(temp);
        std::cout << temp.transpose() << std::endl;
    }*/
    return 0;
}
#endif
