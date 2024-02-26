#include "new_method_BA.h"
#include <random>

void myEdge::computeResidual(VecXd &r) {
    // _error = _measurement - f(v -> _estimate)
    // compute projection error ...
    /// project得到u, v
    /// 一个相机观测一个路标点，得到观测值：
//    _r = _pose->GetPixelValue(_landmark->estimate()) - _measurement;
    r = _pose->GetPixelValue(_landmark->estimate()) - _measurement;
//    std::cout << "GetPixelValue: " << (_pose->GetPixelValue(_landmark->estimate())).transpose() << std::endl;
//    std::cout << "_measurement: " << _measurement.transpose() << std::endl;
//    std::cout << "r inside " << r.transpose() << std::endl;
}

// Let g2o compute jacobian for you
//    G2O_MAKE_AUTO_AD_FUNCTIONS;
// 参照slambook14 page 186-187，但是有两个区别，这里还多了畸变模型，并且归一化平面为负，所以还是需要重新推导一下
//! 线性化直和
void myEdge::linearizeOplus(MatXXd &Jp, MatXXd &Jl) {
    //! 有两种顶点，一个是位姿+内参，一个是点（三维坐标）
    auto cam_est = _pose->estimate();
    auto P= _landmark->estimate();

    Sophus::SO3d R = cam_est.so3().cast<double>();
    Vec3d t = cam_est.translation();

    // 生成估计值T处的扰动{\delta}{\xi}的雅克比
    Vec3d Pc = R * P + t;

    double Xc = Pc[0];
    double Yc = Pc[1];
    double Zc = Pc[2];

    double Zc2 = Zc * Zc;
    double Xc2 = Xc * Xc;
    double Yc2 = Yc * Yc;

    Eigen::Matrix<double, 2, 3> E1;
    Eigen::Matrix<double, 2, 6> E2;
    // row 0
    E1(0, 0) = fx / Zc;  E1(0, 1) = 0;           E1(0, 2) = -fx * Xc / Zc2;
    // row 1
    E1(1, 0) = 0;        E1(1, 1) = fy / Zc;     E1(1, 2) = -fy * Yc / Zc2;

    E2.block(0, 0, 2, 3) = E1;

    E2(0, 3) = -fx * Xc * Yc / Zc2;
    E2(0, 4) = fx + fx * Xc2 / Zc2;
    E2(0, 5) = -fx * Yc / Zc;

    E2(1, 3) = -fy - fy * Yc2 / Zc2;
    E2(1, 4) = fy * Xc * Yc / Zc2;
    E2(1, 5) = fy * Xc / Zc;

//!   2 × 6
//    _Jp = E2;
    Jp = E2;
//!   2 × 3
//    _Jl = E1 * (R.matrix());
    Jl = E1 * (R.matrix());
}

void qr3(MatXXd &Jp, MatXXd &Jl, VecXd &Jr) {
    MatXXd temp1, temp2;
    VecXd temp3;
    int nres = Jl.rows();
    int cols = Jl.cols();
    assert(nres > 3);
    // i: row
    // j: col
    for (int j = 0; j < cols; j++) {
        double pivot = Jl(j, j);
        for (int i = j + 1; i < nres; i++) {
            if (std::abs(Jl(i, j)) < 1e-10)
                continue;
            double a = Jl(i, j);
            double r = sqrt(pivot * pivot + a * a);
            double c = pivot / r;
            double s = a / r;
            pivot = r;
            assert(std::isfinite(r));
            assert(std::abs(r) > 1e-10);
// 变0的，先到temp
            temp1 = -s * Jp.row(j) + c * Jp.row(i);
            temp2 = -s * Jl.row(j) + c * Jl.row(i);
            temp3 = -s * Jr.row(j) + c * Jr.row(i);
// 变大的.  j是pivot，在上面，i在下面
            Jp.row(j) = c * Jp.row(j) + s * Jp.row(i);
            Jl.row(j) = c * Jl.row(j) + s * Jl.row(i);
            Jr.row(j) = c * Jr.row(j) + s * Jr.row(i);
// 变0的, temp => i
            Jp.row(i) = temp1;
            Jl.row(i) = temp2;
            Jr.row(i) = temp3;

            Jl(j, j) = pivot = r;
            Jl(i, j) = 0;
        }
    }
}
/// 对于该点下面的所有边，计算Jp Jl r
void myOptimizer::_linearize_one_landmark(myLandmark *l) {
    MatXXd Jp;
    MatXXd Jl;
    VecXd   r;

    int Jp_Cols = l->edges.size() * 6;
    int Jp_Rows = l->edges.size() * 2;
    l->Jp = MatXXd::Zero(Jp_Rows, Jp_Cols);
    l->Jl = MatXXd::Zero(Jp_Rows, 3);
    l->r  = VecXd::Zero(Jp_Rows);

    for (int i = 0; i < l->edges.size(); i++) {
        auto edge = l->edges[i];
        edge->computeResidual(r);
//        std::cout << "r outside: " << r.transpose() << std::endl;
        edge->linearizeOplus(Jp, Jl);
        l->Jp.block(i * 2, i * 6, 2, 6) = Jp;
        l->Jl.block(i * 2, 0, 2, 3) = Jl;
        l->r.middleRows(i * 2, 2)  = r;
    }

    /// 计算单路标的残差能量
    l->energy = (l->r).transpose() * (l->r);
//    std::cout << "l->energy " << l->energy << std::endl;
    /// 保存未边缘化的Jp Jl r用于回代求解delta_landmark
    l->orig_Jp = l->Jp;
    l->orig_Jl = l->Jl;
    l->orig_r  = l->r;

    assert(l->Jp.rows() > LAND_SIZE);

    qr3(l->Jp, l->Jl, l->r);
//    std::cout << "Jp before:\n" << l->Jp << std::endl;
    if (l->Jp.rows() <= LAND_SIZE) {
        l->Jp = MatXXd::Zero(0, 0);
        l->r = VecXd::Zero(0);
    } else {
        l->Jp = l->Jp.bottomRows(l->Jp.rows() - LAND_SIZE);
//        std::cout << "Jp after:\n" << l->Jp << std::endl;
        l->r = l->r.bottomRows(l->r.rows() - LAND_SIZE);
    }
    // to
}

/// 将一个landmark对应的所有边的Jp r' (Jl已经被边缘化)放入大的稀疏矩阵J, r
#define TRIPLET_LIST
#ifdef TRIPLET_LIST
//tripletList.reserve(estimation_of_entries);
//for(...)
//{
// ...
//tripletList.push_back(T(i,j,v_ij));
//}
//SparseMatrixType mat(rows,cols);
//mat.setFromTriplets(tripletList.begin(), tripletList.end());
// mat is ready to go!
/*
void myOptimizer::_toSparseMatrix(int startRow, int startCol,
                    MatXXd *blk) {
//    std::vector<T > tripletList;
    for (int i = 0; i < (*blk).rows(); i++)
        for (int j = 0; j < (*blk).cols(); j++)
            _tripletList.push_back(T(startRow + i, startCol + j, (*blk)(i, j)));
}
*/
#endif
double myOptimizer::_compose1() {
    int startRow, startCol;
    startRow = startCol = 0;
    double energy = 0.0;

#ifdef TRIPLET_LIST
    _tripletList.clear();
#endif
    for (auto l : _allLandmarks) {
        assert(l);
        assert(l->edges);

        if (l->Jp.rows() == 0)
            continue;
        for (int i = 0; i < l->edges.size(); i++) {
            startCol = l->edges[i]->getPose()->pose_id * POSE_SIZE;
//            std::cout << "startCol: " << startCol << std::endl;
//            std::cout << "startRow: " << startRow << std::endl;
            MatXXd blk = l->Jp.middleCols(i * POSE_SIZE, POSE_SIZE);
//            std::cout << "blk rows: " << blk.rows() << std::endl;
//            std::cout << "blk cols: " << blk.cols() << std::endl;
            {
                for (int m = 0; m < blk.rows(); m++)
                    for (int n = 0; n < blk.cols(); n++)
#ifdef TRIPLET_LIST
                        _tripletList.push_back(T(startRow + m, startCol + n, blk(m, n)));
#else
                        this->_big_J.insert(startRow + m, startCol + n) = blk(m, n);
#endif
            }
        }
//        std::cout << "done........" << std::endl;
//        std::cout << "big_J: row col: " << _big_J.rows() << " " << _big_J.cols() << std::endl;
        this->_big_r.middleRows(startRow, l->r.rows()) = l->r;
        energy += l->energy;
        startRow += l->Jp.rows() /*- LAND_SIZE*/;
    }
#ifdef TRIPLET_LIST
    this->_big_J.setFromTriplets(_tripletList.begin(), _tripletList.end());
#endif
    std::cout << "all done........" << energy << std::endl;
    return energy;
}
void myOptimizer::_compute1(VecXd &dx) {
    Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double> > lscg;
    lscg.setMaxIterations(1000);
    lscg.setTolerance(1e-2);
    lscg.compute(_big_J);
    dx = lscg.solve(_big_r);
//    std::cout << "lscg  x:\n" << dx.transpose() << std::endl;
    std::cout << "lscg iter: " << lscg.iterations() << std::endl;
    std::cout << "lscg error: " << lscg.error() << std::endl;
}
