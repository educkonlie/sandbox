//
// Created by ruankefeng on 24-2-19.
//

#ifndef BUNDLE_ADJUSTMENT_BLOCKSPARSEMATRIX_H
#define BUNDLE_ADJUSTMENT_BLOCKSPARSEMATRIX_H

#define BLOCK_SPARSE_MAT_test

#ifdef BLOCK_SPARSE_MAT_test
#include <iostream>

#include <Eigen/Core>
//#include <eigen3/Eigen/Core>
#include <map>
#include <Eigen/Cholesky>
#include <Eigen/src/Core/Matrix.h>

//#define scalar double
typedef Eigen::Matrix<double,2,1> Vec2d;
typedef Eigen::Matrix<double,3,1> Vec3d;
typedef Eigen::Matrix<double, 3, 3> Mat33d;
typedef Eigen::Matrix<double,16,1> Vec16d;
typedef Eigen::Matrix<double,8,1> Vec8d;
typedef Eigen::Matrix<double,2,6> Mat26d;
typedef Eigen::Matrix<double,2,3> Mat23d;

//typedef Eigen::Matrix<scalar,2,1> Vec2c;
//typedef Eigen::Matrix<scalar,3,1> Vec3c;
//typedef Eigen::Matrix<scalar, 3, 3> Mat33c;
//typedef Eigen::Matrix<scalar,16,1> Vec16c;
//typedef Eigen::Matrix<scalar,8,1> Vec8c;
//typedef Eigen::Matrix<scalar,2,6> Mat26c;
//typedef Eigen::Matrix<scalar,2,3> Mat23c;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatXXdr;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatXXd;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VecXd;
//typedef Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatXXcr;
//typedef Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatXXc;
//typedef Eigen::Matrix<scalar, Eigen::Dynamic, 1> VecXc;

typedef std::vector<Vec3d, Eigen::aligned_allocator<Vec3d>> VecVec3d;
#else
#include "globalPBA.h"
#endif

/// 1X6 1X3 1X1
/// Jp  Jl  r
/// 6X6 blockdiagonal
///有上述四种block，然后set_block(startRow, startCol, block)
/// 之前pcg中的手撸线程池太复杂了，应该可以简化成用openmp
/// 其实本项目中只需要template <1, 6>这种类型，可以增加一个transpose操作，和跟VecX的矩阵乘法

/// 组每块的大小不一定相等的块稀疏矩阵，组的过程中不干预每块的大小是否匹配
/// 等到矩阵制作完成（map已经塞完）再开始为每一个块行和块列计算相对应的实际行列值
///
template <int C> class BlockSparseMatrix {
public:
    BlockSparseMatrix(int BlockRows, int BlockCols) {
        _BlockRows = BlockRows;
        _BlockCols = BlockCols;
        for (int i = 0; i < _BlockRows; i++) {
            _startRow_of_block.push_back(0);
            _rows_of_block.push_back(0);
        }
        _rows = 0;
    }
    ~BlockSparseMatrix() {
        _blocked_mat.clear();
    }
    int getC() {
        return C;
    }
    int block_rows() {
        return _BlockRows;
    }
    int block_cols() {
        return _BlockCols;
    }
    /// axpy   x.row(i) += a * x.row(j)

    bool find(int startRow, int startCol, MatXXd **ret) {
        auto it = _blocked_mat.find(std::make_pair(startRow, startCol));
        if (it == _blocked_mat.end())
            return false;
        *ret = it->second;
//        std::cout << "it->second\n" << *(it->second) << std::endl;
        return true;
    }
//    inline std::map<std::pair<int, int>, MatXXd *>::iterator

    inline void add(int startRow, int startCol, MatXXd &block) {
//        assert(block.rows() == R);
//        assert(block.cols() == C);
//        if (startRow >= Rs.size())
//        assert(startRow < block_rows());
//        assert(startCol < block_cols());
        auto it = _blocked_mat.find(std::make_pair(startRow, startCol));
        if (it != _blocked_mat.end()) {
            *(it->second) = *(it->second) + block;
        } else {
            _blocked_mat.insert(std::make_pair(std::make_pair(startRow, startCol), &block));
        }
    }

    void set_startRow_of_block(int i, int k) {
        static int prev = 0;
        if (i == 0) {
            _startRow_of_block[i] = 0;
            prev = k;
            _rows_of_block[i] = k;
        } else {
            _startRow_of_block[i] = _startRow_of_block[i - 1] + prev;
            prev = k;
            _rows_of_block[i] = k;
        }
        _rows = _startRow_of_block[i] + _rows_of_block[i];
        _BlockRows = i + 1;
    }

    void toDenseMatrix(MatXXd &mat) {
//        for (int i = 0; i < _startRow_of_block.size(); i++)
//            std::cout << "..." << _startRow_of_block[i] << std::endl;
        for (int i = 0; i < _BlockRows; i++)
            for (int j = 0; j < _BlockCols; j++) {
                auto it = _blocked_mat.find(std::make_pair(i, j));
                if (it != _blocked_mat.end()) {
//                    std::cout << i << " " << j << "\n" << *(it->second) << std::endl;
//                    std::cout << "start row: " << _startRow_of_block[i] << std::endl;
                    mat.block(_startRow_of_block[i], j * C,
                              _rows_of_block[i], C)
                            = *(it->second);
                }
            }
    }

    void right_multiply(VecXd &q, VecXd &Aq) {
        std::vector<VecXd> temp;
        for (int i = 0; i < this->block_rows(); i++) {
            temp.push_back(VecXd::Zero(this->_rows_of_block[i]));
        }
        /// 并行化
//#pragma omp parallel for collapse(2)
//#pragma omp parallel for
        for (int i = 0; i < this->block_rows(); i++) {
            for (int j = 0; j < this->block_cols(); j++) {
                auto it = this->_blocked_mat.find(std::make_pair(i, j));
                if (it != _blocked_mat.end()) {
                    temp[i] += *(it->second) * q.middleRows(j * C, C);
                }
            }
        }
//#pragma omp parallel for
        for (int i = 0; i < this->block_rows(); i++) {
            Aq.middleRows(this->_startRow_of_block[i], this->_rows_of_block[i]) = temp[i];
        }
    }
    void transpose_right_multiply(VecXd &q, VecXd &Atq) {
        std::vector<VecXd> temp;
        for (int i = 0; i < this->block_cols(); i++) {
            temp.push_back(VecXd::Zero(C));
        }
        /// 并行化
//#pragma omp parallel for
        for (int j = 0; j < this->block_cols(); j++) {
            for (int i = 0; i < this->block_rows(); i++) {
//#pragma omp parallel for  reduction(+:Atq)
                auto it = this->_blocked_mat.find(std::make_pair(i, j));
                if (it != _blocked_mat.end()) {
                    temp[j] += (*(it->second)).transpose() *
                            q.middleRows(_startRow_of_block[i], _rows_of_block[i]);
                }
            }
        }
//#pragma omp parallel for
        for (int i = 0; i < this->block_cols(); i++) {
            Atq.middleRows(i * C, C) = temp[i];
        }
    }
    int rows() {
        return _rows;
    }
    void AAq(VecXd &q, VecXd &AAq) {
        VecXd Aq = VecXd::Zero(_rows);
        right_multiply(q, Aq);
        transpose_right_multiply(Aq, AAq);
    }
    void get_M_inv(MatXXd &M_inv) {
        MatXXd diag[10000];
        MatXXd *p;

//        for (int i = 0; i < this->block_rows(); i++)
//            for (int j = 0; j < this->block_cols(); j++) {
//                auto it = _blocked_mat.find(make_pair(i, j));
//                if (it != _blocked_mat.end()) {
//                    auto t = *(it->second);
//                    std::cout << "it:\n" << t.cols() << std::endl;
//                }
//            }
////////  必须预指定哪几个线程处理哪几条，要不会有问题
//#pragma omp parallel for
        for (int j = 0; j < this->block_cols(); j++)
            for (int i = 0; i < this->block_rows(); i++) {
                if (this->find(i, j, &p)) {
                    if (diag[j].size() == 0) {
                        diag[j] = (*p).transpose() * (*p);
//                        std::cout << "*p\n" << *p << std::endl;
//                        std::cout << "diag[j]\n" << diag[j].rows() << " " << diag[j].cols() << std::endl;
                    } else {
                        diag[j] += (*p).transpose() * (*p);
                    }
                }
            }
        for (int j = 0; j < this->block_cols(); j++) {
            M_inv.block(j * C, j * C, C, C) =
                    diag[j].selfadjointView<Eigen::Upper>().llt().solve(MatXXd::Identity(C, C));
//            M_inv.block(j * C, j * C, C, C) =
//                    diag[j] * (MatXXd::Identity(C, C) - );
        }
//        for (int j = 0; j < this->block_cols() * C; j++) {
//            if (M_inv(j, j) == 0.0)
//                M_inv(j, j) = 1.0;
//            else
//                M_inv(j, j) = 1.0 / M_inv(j, j);
//        }
    }
    void leastsquare_pcg_BlockSparse(MatXXd &M_inv, VecXd &b, VecXd &x, double tor, int maxiter)
    {
        static int num_of_iter = 0;
        static int num_of_pcg = 0;
        num_of_pcg++;
        int i = 0;

//    int C = A.getC();

        x = VecXd::Zero(this->block_cols() * C);

        VecXd Atb = VecXd::Zero(this->block_cols() * C);
        this->transpose_right_multiply(b, Atb);
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

        VecXd q = VecXd::Zero(this->block_cols() * C);
        while (i < maxiter && delta_new > tor * tor * delta_0) {
//        VecXc q = A.transpose() * (A * d);
//            std::cout << "AAq start.." << std::endl;
            this->AAq(d, q);
//            std::cout << "AAq end.." << std::endl;
//            std::cout << delta_new << std::endl;
            alpha = delta_new / (d.transpose() * q);
            x = x + alpha * d;

            if (i % 5 == 0) {
//            if (true) {
                VecXd temp = VecXd::Zero(this->block_cols() * C);
                this->AAq(x, temp);
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
    void least_square_conjugate_gradient(MatXXd &M_inv, VecXd &b, VecXd &x,
                                         int iters,
                                         double tol_error)
    {
//        using std::sqrt;
//        using std::abs;

        double tol = tol_error;
        int maxIters = iters;

//        int m = this->rows(), n = this->cols();
        int m = this->rows();
        int n = this->block_cols() * C;

        x = VecXd::Zero(n);

//        VecXd residual        = b - this->right_multiply(x);
        VecXd res = b;
        VecXd r = VecXd::Zero(n);
        this->transpose_right_multiply(res, r);

        VecXd Atb = VecXd::Zero(n);
        this->transpose_right_multiply(b, Atb);

        double rhsNorm2 = Atb.squaredNorm();

        if(rhsNorm2 == 0) {
            x.setZero();
            iters = 0;
            tol_error = 0;
            return;
        }
        double threshold = tol*tol*rhsNorm2;
        double residualNorm2 = r.squaredNorm();
        if (residualNorm2 < threshold) {
            iters = 0;
            tol_error = std::sqrt(residualNorm2 / rhsNorm2);
            std::cout << "tol_error: " << tol_error << std::endl;
            return;
        }

        VecXd p(n);
//    p = precond.solve(normal_residual);                         // initial search direction
        p = M_inv * r;

        VecXd z(n), tmp(m);
        double absNew = r.dot(p);  // the square of the absolute value of r scaled by invM
        int i = 0;

        while(i < maxIters) {
            this->right_multiply(p, tmp);

            double alpha = absNew / tmp.squaredNorm();      // the amount we travel on dir
            x += alpha * p;                                 // update solution
            res -= alpha * tmp;                        // update residual
//            r = A.adjoint() * res;     // update residual of the normal equation
            this->transpose_right_multiply(res, r);

            residualNorm2 = r.squaredNorm();
            if(residualNorm2 < threshold)
                break;

//        z = precond.solve(normal_residual);             // approximately solve for "A'A z = normal_residual"
            z = M_inv * r;

            double absOld = absNew;
            absNew = r.dot(z);  // update the absolute value of r
            double beta = absNew / absOld;              // calculate the Gram-Schmidt value used to create the new search direction
            p = z + beta * p;                               // update search direction
            i++;

//            tol_error = std::sqrt(residualNorm2 / rhsNorm2);
            std::cout << "resN2: " << residualNorm2 << " threshold: " << threshold << std::endl;
        }
        tol_error = std::sqrt(residualNorm2 / rhsNorm2);
        std::cout << "tol_error: " << tol_error << std::endl;
        iters = i;
    }
private:
//    std::map<std::pair<int, int>, MatXXd *> _blocked_mat;
    std::map<std::pair<int, int>, MatXXd *> _blocked_mat;
    int _BlockRows;
    int _BlockCols;
    int _rows;
    std::vector<int> _startRow_of_block;
    std::vector<int> _rows_of_block;
};

#if 0
void least_square_conjugate_gradient(const BlockSparseMatrix &A, VecXd &b, VecXd &x,
                                     MatXXd &M_inv, int &iters,
                                     double &tol_error)
{
    using std::sqrt;
    using std::abs;

    double tol = tol_error;
    int maxIters = iters;

    int m = A.rows(), n = A.cols();

    VecXd residual        = b - A * x;
    VecXd normal_residual = A.adjoint() * residual;

    double rhsNorm2 = (A.adjoint() * b).squaredNorm();
    if(rhsNorm2 == 0) {
        x.setZero();
        iters = 0;
        tol_error = 0;
        return;
    }
    double threshold = tol*tol*rhsNorm2;
    double residualNorm2 = normal_residual.squaredNorm();
    if (residualNorm2 < threshold) {
        iters = 0;
        tol_error = sqrt(residualNorm2 / rhsNorm2);
        return;
    }

    VecXd p(n);
//    p = precond.solve(normal_residual);                         // initial search direction
    p = M_inv * normal_residual;

    VecXd z(n), tmp(m);
    double absNew = normal_residual.dot(p);  // the square of the absolute value of r scaled by invM
    int i = 0;
    while(i < maxIters) {
        tmp.noalias() = A * p;

        double alpha = absNew / tmp.squaredNorm();      // the amount we travel on dir
        x += alpha * p;                                 // update solution
        residual -= alpha * tmp;                        // update residual
        normal_residual = A.adjoint() * residual;     // update residual of the normal equation

        residualNorm2 = normal_residual.squaredNorm();
        if(residualNorm2 < threshold)
            break;

//        z = precond.solve(normal_residual);             // approximately solve for "A'A z = normal_residual"
        z = M_inv * normal_residual;

        double absOld = absNew;
        absNew = normal_residual.dot(z);  // update the absolute value of r
        double beta = absNew / absOld;              // calculate the Gram-Schmidt value used to create the new search direction
        p = z + beta * p;                               // update search direction
        i++;
    }
    tol_error = sqrt(residualNorm2 / rhsNorm2);
    iters = i;
}
#endif


/// += means = or +=
/// rowVec += myMat.row(i)
///    rowVec.xpy(myMat[i]);
///  myMat.row(i) += rowVec
///    myMat[i].xpy(rowVec)

#endif //BUNDLE_ADJUSTMENT_BLOCKSPARSEMATRIX_H
