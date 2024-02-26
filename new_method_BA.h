#pragma once
#include "globalPBA.h"

///////////////////////////////////////////////////////////////////////////////////////////////
class myPose {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    myPose() {
        this->setEstimate(Sophus::SE3d());
    }
    ~myPose() {}
    // update _estimate
    void oplusImpl(VecXd dx) {
        this->setEstimate(Sophus::SE3d::exp(-dx) * this->estimate());
    }
    /// project得到u, v
    /// 一个相机观测一个路标点，得到观测值：
    /// new cam->GetPixelValue(landmark->estimate()) -> Vec16d
    Vec2d GetPixelValue(Vec3d point) {
        double u, v;
        Vec16d ret;
        _project(point, u, v);
        return Vec2d(u, v);
    }
    void setEstimate(Sophus::SE3d cam) { _estimate = cam; }
    Sophus::SE3d estimate() { return _estimate; }
    void setPoseId(int id) { pose_id = id;}
    int pose_id;
private:
    Sophus::SE3d _estimate;
    inline void _project(Vec3d &point, double &u, double &v) {
        //! v0里保存的似乎是Tcw，但是从poses读进来的应该是Twc
        Vec3d pc = this->estimate() * point;
        pc /= pc[2];
        u = pc[0] * fx + cx;
        v = pc[1] * fy + cy;
    }
};
class myLandmark;
class myEdge {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // 对color和img的关联
    myEdge() {}
    ~myEdge() {}

    void setPose(myPose *p) {
        this->_pose = p;
    }
    void setLandmark(myLandmark *l) {
        this->_landmark = l;
    }
    void setMeasurement(Vec2d m) {
        _measurement = m;
    }

    // _r = _measurement - f(v -> _estimate)
    void computeResidual(VecXd &r);
    // Let g2o compute jacobian for you
    //! 线性化直和
    void linearizeOplus(MatXXd &Jp, MatXXd &Jl);

    myPose *getPose() {
        return _pose;
    }
private:
    myPose *_pose;
    myLandmark *_landmark;
    Vec2d _measurement;
};
class myEdge;
class myLandmark {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    myLandmark() { _estimate = Vec3d(0, 0, 0); }
    ~myLandmark() {}

    void setEstimate(Vec3d p) { _estimate = p; }
    Vec3d estimate() { return _estimate; }
    void oplusImpl(Vec3d update) { _estimate += update; }
    void addEdge(myEdge *edge) {
        edges.push_back(edge);
    }
    MatXXd Jp;
    MatXXd Jl;
    VecXd   r;
    double energy;
    /// 回代求解三维点的dy的时候需要
    MatXXd orig_Jp;
    MatXXd orig_Jl;
    VecXd  orig_r;
    std::vector<myEdge *> edges;
private:
    Vec3d _estimate;
};

/// 优化器的输入是所有的边，每条边经过一次线性化求得两行J，一次computeError得到两行r
/// rootba应当以landmark作为基本分组单位，一个landmark会带有好几条边，生成一个Jp块，一个Jl块，一列r
class myOptimizer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    myOptimizer() {}
    ~myOptimizer() {}

    void addPose(myPose *p) {
        _allPoses.push_back(p);
    }
    void addLandmark(myLandmark *l) {
        _allLandmarks.push_back(l);
    }

    double solveSystem(VecXd &dx) {
        double energy = 0.0;

        /// 这个for可以用openmp并行化
        for (auto l : _allLandmarks) {
            _linearize_one_landmark(l);
        }
        ///     leastsquare_pcg(preconditioner, sparse_J, sparse_r);
///     return energy for compare, and to decide the next iterate
        /// 这个不可以并行化
        energy = _compose1();
        /// dx是所有pose的更新值
        /// _compute可以并行化
        _compute1(dx);
        return energy;
    }
    /// big circle for LM
    double optimize(int iters) {
        double old_energy = -1;
        VecXd dx;
        int big_J_rows = 0;
        int big_J_cols = 0;
        for (auto l : _allLandmarks)
            for (auto e : l->edges)
                big_J_rows += 2;
        for (auto p : _allPoses)
            big_J_cols += POSE_SIZE;

        std::cout << "big_J rows: " << big_J_rows << " cols: " << big_J_cols << std::endl;
        _big_J = Eigen::SparseMatrix<double>(big_J_rows, big_J_cols);
        _big_r = VecXd::Zero(big_J_rows);

        for (int iter = 0; iter < iters; iter++) {
            std::cout << "start....solve..." << std::endl;
            double energy = solveSystem(dx);
            std::cout << "iter: " << iter << " energy: " << energy << std::endl;
            assert(energy >= 0);
//            if (iter > 0 && energy > old_energy)
//                break;
            old_energy = energy;
            update(dx);
        }
        return old_energy;
    }
    //  Jp * dx + Jl * dy = r
//  Jl * dy = r - Jp * dx
// 已求得dx =>  dy = Jl.ldlt().solve(r - Jp * dx)
//  T = exp(dx) * T => p->oplus
//  P = P + dy   => l->oplus
    void update(VecXd &dx) {
        /// 可以并行化
/// update all poses
        for (int i = 0; i < _allPoses.size(); i++) {
            auto *p = _allPoses[i];
            p->oplusImpl(dx.middleRows(i * POSE_SIZE, POSE_SIZE));
        }
        // update all landmarks
        for (auto l : _allLandmarks) {
            VecXd local_dx = VecXd::Zero(l->edges.size() * POSE_SIZE);
            for (int i = 0; i < l->edges.size(); i++) {
                myPose *p = l->edges[i]->getPose();
                local_dx.middleRows(i * POSE_SIZE, POSE_SIZE)
                        = dx.middleRows(p->pose_id * POSE_SIZE, POSE_SIZE);
            }
            MatXXd H = (l->orig_Jl).transpose() * (l->orig_Jl);
            VecXd r = l->orig_r - l->orig_Jp * local_dx;
            VecXd b = (l->orig_Jl).transpose() * r;
            VecXd dy = H.ldlt().solve(b);
            l->oplusImpl(-dy);
        }
    }
private:
    std::vector<T > _tripletList;
    std::vector<myPose *> _allPoses;
    std::vector<myLandmark *> _allLandmarks;
    Eigen::SparseMatrix<double> _big_J;
    VecXd _big_r;
    void _linearize_one_landmark(myLandmark *l);
    void _toSparseMatrix(int startRow, int startCol, MatXXd blk);
    /// 维护并计算最小二乘方程，使用eigen的稀疏矩阵
    double _compose1();
    void   _compute1(VecXd &dx);
    /// 维护并计算最小二乘方程，使用自制的BlockSparseMatrix
//    double _compose2(BlockSparseMatrix<double> &J, BlockSparseMatrix<double> &r, myLandmark *l);
//    void   _compute2(VecXd &dx);
};