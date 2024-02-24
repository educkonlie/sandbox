//
// Created by ruankefeng on 24-2-19.
//

#ifndef BUNDLE_ADJUSTMENT_BLOCKSPARSEMATRIX_H
#define BUNDLE_ADJUSTMENT_BLOCKSPARSEMATRIX_H

#define BLOCK_SPARSE_MAT_test

#ifdef BLOCK_SPARSE_MAT_test
#include <iostream>

#include <Eigen/Core>

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
template <int R, int C> class BlockSparseMatrix {
public:
    BlockSparseMatrix(int BlockRows, int BlockCols) {
        MatXXd temp;
        for (int i = 0; i < BlockRows; i++)
            for (int j = 0; j < BlockCols; j++)
                _blocked_mat.push_back(temp);
        _BlockRows = BlockRows;
        _BlockCols = BlockCols;
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
//    void axpy(BlockSparseMatrix &x, double a) {
//
//    }
//    void xpy(BlockSparseMatrix &x,)

    inline void add(int startRow, int startCol, MatXXd &block) {
        assert(block.rows() == R);
        assert(block.cols() == C);
        auto it = find(startRow, startCol);
//        if ((*it).size() == 0)
//            std::cout << 0 << std::endl;
//        else
//            std::cout << (*it) << std::endl;
        if (it != _blocked_mat.end() && (*it).size() != 0) {
            *it = *it + block;
        } else if (it != _blocked_mat.end()){
            *it = block;
        }
    }
    inline std::vector<MatXXd>::iterator find(int startRow, int startCol) {
        if (_blocked_mat.empty())
            return _blocked_mat.end();
        auto it = _blocked_mat.begin();
        return it + (startRow * _BlockCols + startCol);
    }
    void toDenseMatrix(MatXXd &mat) {
        assert(mat.rows() == R * _BlockRows);
        assert(mat.cols() == C * _Blockcols);
        for (int i = 0; i < _BlockRows; i++)
            for (int j = 0; j < _BlockCols; j++) {
                auto it = find(i, j);
                if (it != _blocked_mat.end() && (*it).data() != NULL)
                    mat.block(i * R, j * C, R, C) = *it;
            }
    }

private:
    std::vector<MatXXd > _blocked_mat;
    int _BlockRows;
    int _BlockCols;
};

/// 行向量，按列分块
template <int C> class mySparseRowVec {
public:
    mySparseRowVec(int BlockCols) {
        VecXd temp;
        for (int j = 0; j < BlockCols; j++)
            _blocked_vec.push_back(temp);
        _BlockCols = BlockCols;
    }
    ~mySparseRowVec() {
        _blocked_vec.clear();
    }
    inline int BlockCols() {
        return _BlockCols;
    }
    // = or +=
    inline void add(int startCol, VecXd &block) {
        assert(block.cols() == C);
        assert(startCol < _BlockCols);

        if ((_blocked_vec[startCol]).size() == 0)
            _blocked_vec[startCol] = block;
        else
            _blocked_vec[startCol] += block;
    }
    inline VecXd middleBlock(int i) {
        return _blocked_vec[i];
    }
    /// 将稀疏行x乘上标量a加到本稀疏行上去
    void axpy(mySparseRowVec &x, double a) {
        for (int i = 0; i < x.BlockCols(); i++) {
            add(i, a * x.middleBlock(i));
        }
    }
    /// 将稀疏行x加到本稀疏行上去
    void xpy(mySparseRowVec &x) {
        for (int i = 0; i < x.BlockCols(); i++) {
            add(i, x.middleBlock(i));
        }
    }
    void toDenseVec(VecXd &vec) {
        assert(vec.rows() == C * _Blockcols);
        for (int j = 0; j < _BlockCols; j++) {
            if (_blocked_vec[j].size() == 0)
                continue;
            vec.middleRows(j * C, C) = _blocked_vec[j];
        }
    }
//    void scalarMul(double a) {
//        for (int i = 0; i < _BlockCols; i++) {
//            if (_blocked_vec[i].size() == 0)
//                continue;
//            _blocked_vec[i] = a * _blocked_vec[i];
//        }
//    }
    /// 稀疏行 × q == scalar
    double Aq(VecXd &q) {
        assert(q.rows() == _BlockCols * C);
        double sum = 0.0;
        for (int i = 0; i < _BlockCols; i++) {
            if (_blocked_vec[i].size() == 0)
                continue;
            sum += (_blocked_vec[i]).transpose() * q.middleRows(i * C, C);
        }
        return sum;
    }

private:
    std::vector<VecXd > _blocked_vec;
    int _BlockCols;
};

/// += means = or +=
/// rowVec += myMat.row(i)
///    rowVec.xpy(myMat[i]);
///  myMat.row(i) += rowVec
///    myMat[i].xpy(rowVec)

#endif //BUNDLE_ADJUSTMENT_BLOCKSPARSEMATRIX_H
