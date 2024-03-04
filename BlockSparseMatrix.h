//
// Created by ruankefeng on 24-2-19.
//

#ifndef BUNDLE_ADJUSTMENT_BLOCKSPARSEMATRIX_H
#define BUNDLE_ADJUSTMENT_BLOCKSPARSEMATRIX_H

#define BLOCK_SPARSE_MAT_test

#ifdef BLOCK_SPARSE_MAT_test
#include <iostream>

#include <Eigen/Core>
#include <map>
#include <Eigen/Cholesky>
#include <Eigen/src/Core/Matrix.h>

#define scalar double
typedef Eigen::Matrix<double,2,1> Vec2d;
typedef Eigen::Matrix<double,3,1> Vec3d;
typedef Eigen::Matrix<double, 3, 3> Mat33d;
typedef Eigen::Matrix<double,16,1> Vec16d;
typedef Eigen::Matrix<double,8,1> Vec8d;
typedef Eigen::Matrix<double,2,6> Mat26d;
typedef Eigen::Matrix<double,2,3> Mat23d;

typedef Eigen::Matrix<scalar,2,1> Vec2c;
typedef Eigen::Matrix<scalar,3,1> Vec3c;
typedef Eigen::Matrix<scalar, 3, 3> Mat33c;
typedef Eigen::Matrix<scalar,16,1> Vec16c;
typedef Eigen::Matrix<scalar,8,1> Vec8c;
typedef Eigen::Matrix<scalar,2,6> Mat26c;
typedef Eigen::Matrix<scalar,2,3> Mat23c;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatXXdr;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatXXd;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VecXd;
typedef Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatXXcr;
typedef Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatXXc;
typedef Eigen::Matrix<scalar, Eigen::Dynamic, 1> VecXc;

typedef std::vector<Vec3d, Eigen::aligned_allocator<Vec3d>> VecVec3d;
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
#pragma omp parallel for
        for (int i = 0; i < this->block_rows(); i++) {
            for (int j = 0; j < this->block_cols(); j++) {
                auto it = this->_blocked_mat.find(std::make_pair(i, j));
                if (it != _blocked_mat.end()) {
                    temp[i] += *(it->second) * q.middleRows(j * C, C);
                }
            }
        }
#pragma omp parallel for
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
#pragma omp parallel for
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
#pragma omp parallel for
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
        MatXXd diag[1000];
        MatXXd *p;
#pragma omp parallel for
        for (int j = 0; j < this->block_cols(); j++)
            for (int i = 0; i < this->block_rows(); i++) {
                if (this->find(i, j, &p)) {
                    if (diag[j].size() == 0) {
                        diag[j] = (*p).transpose() * (*p);
                    } else {
                        diag[j] += (*p).transpose() * (*p);
                    }
                }
            }
        for (int j = 0; j < this->block_cols(); j++)
            M_inv.block(j * C, j * C, C, C) =
                    diag[j].selfadjointView<Eigen::Upper>().llt().solve(MatXXd::Identity(C, C));
    }
private:
    std::map<std::pair<int, int>, MatXXd *> _blocked_mat;
    int _BlockRows;
    int _BlockCols;
    int _rows;
    std::vector<int> _startRow_of_block;
    std::vector<int> _rows_of_block;
};



/// += means = or +=
/// rowVec += myMat.row(i)
///    rowVec.xpy(myMat[i]);
///  myMat.row(i) += rowVec
///    myMat[i].xpy(rowVec)

#endif //BUNDLE_ADJUSTMENT_BLOCKSPARSEMATRIX_H
