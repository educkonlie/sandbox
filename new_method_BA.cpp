#include "new_method_BA.h"
#include <random>

#if 0
Sophus::SE3d AddNoiseinPose(Sophus::SE3d &pose)
{
#ifdef DIRECT_METHOD
    static std::normal_distribution<double> pose_rotation_noise(0., 0.001);
    static std::normal_distribution<double> pose_position_noise(0., 0.001);
#else
    static std::normal_distribution<double> pose_rotation_noise(0., 0.01);
    static std::normal_distribution<double> pose_position_noise(0., 0.01);
#endif
    static std::default_random_engine generator;
//    for (size_t i = 0; i < poses.size(); ++i) {
        // 给相机位置和姿态初值增加噪声
        Mat33d noise_R, noise_X, noise_Y, noise_Z;
        noise_X = Eigen::AngleAxis<double>(pose_rotation_noise(generator),
                                           Vec3d::UnitX());
        noise_Y = Eigen::AngleAxis<double>(pose_rotation_noise(generator),
                                           Vec3d::UnitY());
        noise_Z = Eigen::AngleAxis<double>(pose_rotation_noise(generator),
                                           Vec3d::UnitZ());
        noise_R = noise_X * noise_Y * noise_Z;
        Vec3d noise_t(pose_position_noise(generator),
                      pose_position_noise(generator),
                      pose_position_noise(generator));
//        rootBA::Quaternion<Scalar> noise_q_wc(noise_R);
//        assert(i >= 2);
//        if (i < 2) {
//            noise_t.setZero();
//            noise_q_wc.setIdentity();
//        }
    return Sophus::SE3d(noise_R, noise_t) * pose;
//        poses_noisy.push_back(Sophus::SE3d(noise_R, noise_t) * poses[i]);
//        cameras[i].setNoise(noise_t, noise_q_wc);
//    }
}
Vec3d AddNoiseinPoint(Vec3d &point)
{
    static std::default_random_engine generator;
#ifdef DIRECT_METHOD
    static std::normal_distribution<double> landmark_position_noise(0., 0.02);
#else
    static std::normal_distribution<double> landmark_position_noise(0., 0.2);
#endif
//    static std::normal_distribution<double> landmark_position_noise(0., 0.0);
// 为初值添加随机噪声
    Vec3d noise(landmark_position_noise(generator),
                    landmark_position_noise(generator),
                    landmark_position_noise(generator));
    return point + noise;
}
#endif

void myEdge::computeResidual(VecXd &r) {
    // _error = _measurement - f(v -> _estimate)
    // compute projection error ...
    /// project得到u, v
    /// 一个相机观测一个路标点，得到观测值：
//    _r = _pose->GetPixelValue(_landmark->estimate()) - _measurement;
    r = _pose->GetPixelValue(_landmark->estimate()) - _measurement;
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
        edge->linearizeOplus(Jp, Jl);
        l->Jp.block(i * 2, i * 6, 2, 6) = Jp;
        l->Jl.block(i * 2, 0, 2, 3) = Jl;
        l->r.middleRows(i * 2, 2)  = r;
    }

    /// 计算单路标的残差能量
    l->energy = r.norm();
    /// 保存未边缘化的Jp Jl r用于回代求解delta_landmark
    l->orig_Jp = l->Jp;
    l->orig_Jl = l->Jl;
    l->orig_r  = l->r;

    //qr(l->Jp, l->Jl, l->r);
    // to
}

/// 将一个landmark对应的所有边的Jp r' (Jl已经被边缘化)放入大的稀疏矩阵J, r
//tripletList.reserve(estimation_of_entries);
//for(...)
//{
// ...
//tripletList.push_back(T(i,j,v_ij));
//}
//SparseMatrixType mat(rows,cols);
//mat.setFromTriplets(tripletList.begin(), tripletList.end());
// mat is ready to go!
void myOptimizer::_toSparseMatrix(int startRow, int startCol,
                    MatXXd &blk) {
    std::vector<Eigen::Triplet<double> > tripletList;
    for (int i = 0; i < blk.rows(); i++)
        for (int j = 0; j < blk.cols(); j++)
            tripletList.push_back(T(startRow + i, startCol + j, blk(i, j)));
    this->_big_J.setFromTriplets(tripletList.begin(), tripletList.end());
}
double myOptimizer::_compose1() {
    int startRow, startCol;
    startRow = startCol = 0;
    double energy = 0.0;
    for (auto l : _allLandmarks) {
        assert(l->Jp.rows() > LAND_SIZE);
//        l->Jp = l->Jp.bottomRows(l->Jp.rows() - LAND_SIZE);
        for (int i = 0; i < l->edges.size(); i++) {
            startCol = l->edges[i]->getPose()->pose_id * POSE_SIZE;
            _toSparseMatrix(startRow, startCol,
                            l->Jp.block(LAND_SIZE, i * POSE_SIZE,
                                        l->Jp.rows() - LAND_SIZE, POSE_SIZE))
        }
        this->_big_r.middleRows(startRow, l->r.rows()) = l->r;
        energy += l->energy;
        startRow += l->Jp.rows() - LAND_SIZE;
    }
}
void myOptimizer::_compute1(VecXd &dx) {
    Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double> > lscg;
    lscg.setMaxIterations(1000);
    lscg.setTolerance(1e-2);
    lscg.compute(_big_J);
    dx = lscg.solve(_big_r);
    std::cout << "lscg  x:\n" << dx.transpose() << std::endl;
    std::cout << "lscg iter: " << lscg.iterations() << std::endl;
    std::cout << "lscg error: " << lscg.error() << std::endl;
}
