
#pragma once

#include <iostream>

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include <Eigen/Core>

#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

#include <pangolin/pangolin.h>
#include <boost/format.hpp>

//#define DIRECT_METHOD

typedef std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> VecSE3d;
typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVec3d;
typedef Eigen::Matrix<double,2,1> Vec2d;
typedef Eigen::Matrix<double,3,1> Vec3d;
typedef Eigen::Matrix<double, 3, 3> Mat33d;
typedef Eigen::Matrix<double,16,1> Vec16d;
typedef Eigen::Matrix<double,8,1> Vec8d;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatXXdr;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatXXd;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VecXd;

typedef Eigen::Map<Eigen::VectorXd> VectorRef;

using namespace std;

Sophus::SE3d AddNoiseinPose(Sophus::SE3d &pose);
Vec3d AddNoiseinPoint(Vec3d &point);

extern float fx, fy, cx, cy;
//class Cam : public g2o::BaseVertex<6, Sophus::SE3d>;
//class Landmark : public g2o::BaseVertex<3, Vec3d>;
//class EdgeDirectProjection : public g2o::BaseBinaryEdge<16, Vec16d, Cam, Landmark>;
// g2o vertex that use sophus::SE3d as pose
class Cam : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Cam(int host_id, cv::Mat &target, double a, double b)
    {
        _host_id = host_id;
        _aff_a = a;
        _aff_b = b;
        targetImg = target;
    }
    ~Cam() {}

    inline int get_host_id() { return _host_id; }
    inline double get_a() { return _aff_a; }
    inline double get_b() { return _aff_b; }

    bool read(std::istream &is) {}
    bool write(std::ostream &os) const {}

    virtual void setToOriginImpl() {
        this->setEstimate(Sophus::SE3d());
    }

    // update _estimate
    virtual void oplusImpl(const double *update_) {
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> update(update_);
//        Sophus::SO3d R = Sophus::exp(Vector3d(update[0], update[1], update[2])));
//        Vector3d t = Vector3d(update[3], update[4], update[5]);
        this->setEstimate(Sophus::SE3d::exp(update) * this->estimate());
//        cout << "===========" << endl;
    }
    /// project得到u, v
    /// 一个相机观测一个路标点，得到观测值：
    /// new cam->GetPixelValue(landmark->estimate()) -> Vec16d
#ifdef DIRECT_METHOD
    Vec16d GetPixelValue(Vec3d point)
    {
        double u, v;
        Vec16d ret;
        _project(point, u, v);

        if (u < 0.0 || v < 0.0)
            return Vec16d::Zero();
        int k = 0;
        //! [-2, 1] X [-2, 1]，总共16个点
        for (int i = -2; i <= 1; i++)
            for (int j = -2; j <= 1; j++) {
                //! _measurement是老图里的色彩，根据灰度一致假设赋为该路标在新的路标坐标和相机位姿下的投影估计值
                ret[k] = _GetPixelValue(u + i, v + j);
                k++;
            }
        return ret;
    }
#else
    Vec2d GetPixelValue(Vec3d point)
    {
        double u, v;
        Vec16d ret;
        _project(point, u, v);
        return Vec2d(u, v);
    }
#endif
private:
    int _host_id;
    double _aff_a;
    double _aff_b;
//    double _fx, _fy, _cx, _cy;
    cv::Mat targetImg;  // the target image
// bilinear interpolation
#ifdef DIRECT_METHOD
    inline float _GetPixelValue(float x, float y) {
        uchar *data = &(this->targetImg.data[int(y) * this->targetImg.step + int(x)]);
        float xx = x - floor(x);
        float yy = y - floor(y);
        return float(
                (1 - xx) * (1 - yy) * data[0] +
                xx * (1 - yy) * data[1] +
                (1 - xx) * yy * data[this->targetImg.step] +
                xx * yy * data[this->targetImg.step + 1]
        );
    }
#endif
    inline void _project(Vec3d &point, double &u, double &v)
    {
        //! v0里保存的似乎是Tcw，但是从poses读进来的应该是Twc
//        Sophus::SO3d R = (v0->estimate()).so3();
//        Vector3d t = (v0->estimate()).translation();
        Vec3d pc = this->estimate() * point;
        pc /= pc[2];
        u = pc[0] * fx + cx;
        v = pc[1] * fy + cy;

//        if (u < 0)
//            u = 0;
//        if (v < 0)
//            v = 0;

#ifdef DIRECT_METHOD
        // 如果变为outlier点，则使用临近的边界值（即不好不坏)
        if (u - 2 < 0)
            u = -1;
        if (u + 1 >= this->targetImg.cols)
//            u = this->targetImg.cols - 2;
            u = -1;
        if (v - 2 < 0)
            v = -1;
        if (v + 1 >= this->targetImg.rows)
//            v = this->targetImg.rows - 2;
            v = -1;
#endif
//        cout << "rectified " << "u " << u << " v " << v <<endl;
    }
};

class Landmark : public g2o::BaseVertex<3, Vec3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Landmark(int host_id, double a, double b)
    {
        _host_id = host_id;
        _aff_a = a;
        _aff_b = b;
    }
    ~Landmark() {}

    inline int get_host_id() { return _host_id; }
    inline double get_a() { return _aff_a; }
    inline double get_b() { return _aff_b; }

    virtual void setToOriginImpl() override {
        _estimate = Vec3d(0, 0, 0);
    }

    virtual void oplusImpl(const double *update) override {
        _estimate += Vec3d(update[0], update[1], update[2]);
    }

    virtual bool read(std::istream &in) {}
    virtual bool write(std::ostream &out) const {}
private:
    int _host_id;
    double _aff_a;
    double _aff_b;
};
//long long g_outlier = 0;
// TODO edge of projection error, implement it
// 16x1 error, which is the errors in patch
// 跟特征法主要是在这个error的计算里有区别。
// 同时，应该归属于VertexPoint的color，和

#ifdef DIRECT_METHOD
class EdgeDirectProjection : public g2o::BaseBinaryEdge<16, Vec16d, Cam, Landmark> {
#else
class EdgeDirectProjection : public g2o::BaseBinaryEdge<2, Vec2d, Cam, Landmark> {
#endif
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // 对color和img的关联
    EdgeDirectProjection() {}
    ~EdgeDirectProjection() {}

#ifdef DIRECT_METHOD
    void set_ab()
    {
        auto cam = (Cam *) _vertices[0];
        auto landmark = (Landmark *) _vertices[1];
        //! host_id target_id同一个也不要紧
        this->_a = exp(cam->get_a() - landmark->get_a());
        this->_b = cam->get_b() - _a * landmark->get_b();
    }
#endif
    // _error = _measurement - f(v -> _estimate)
    //! computeError获取一条边的两个顶点，一个顶点是投影面，即target，一个顶点是路标点
    //! 投影面需要保存aff_g2l_t，一张修正了大小的img，位姿
    //! 路标点需要报存host_id，aff_g2l_h，
    virtual void computeError() override {
        // TODO START YOUR CODE HERE
        // compute projection error ...
        auto cam = (Cam *) _vertices[0];
        auto landmark = (Landmark *) _vertices[1];
        /// project得到u, v
        /// 一个相机观测一个路标点，得到观测值：
#ifdef DIRECT_METHOD
        /// new cam->GetPixelValue(landmark->estimate()) -> Vec16d
        Vec16d b_ = _b * Vec16d::Identity();
        _error = cam->GetPixelValue(landmark->estimate()) - (_a * _measurement + b_);
#else
        _error = cam->GetPixelValue(landmark->estimate()) - _measurement;
#endif
        // TODO END YOUR CODE HERE
    }
    // Let g2o compute jacobian for you
//    G2O_MAKE_AUTO_AD_FUNCTIONS;
#if 1
// 参照slambook14 page 186-187，但是有两个区别，这里还多了畸变模型，并且归一化平面为负，所以还是需要重新推导一下
    // 畸变模型在page 102
    //! 线性化直和
    virtual void linearizeOplus() override {
        //! 有两种顶点，一个是位姿+内参，一个是点（三维坐标）
        auto cam = (Cam *) _vertices[0];
        auto landmark = (Landmark *) _vertices[1];
        auto pose_est = cam->estimate();
        auto P= landmark->estimate();

        Sophus::SO3d R = pose_est.so3().cast<double>();
        Vec3d t = pose_est.translation();

        // 生成估计值T处的扰动{\delta}{\xi}的雅克比
        Vec3d Pc = R * P + t;
//        Vec3d Pcc = pose_est * P;
//        std::cout << Pc.transpose() << std::endl;
//        std::cout << Pcc.transpose() << std::endl;
//        exit(0);

        double Xc = Pc[0];
        double Yc = Pc[1];
        double Zc = Pc[2];

        double Zc2 = Zc * Zc;
        double Xc2 = Xc * Xc;
        double Yc2 = Yc * Yc;

        double r2 = (Xc * Xc + Yc * Yc + Zc * Zc) / (Zc * Zc);

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
//        _jacobianOplusXi << E, E * (-Sophus::SO3d::hat(R * P).matrix());
        _jacobianOplusXi = E2;
//      生成三维点point的雅克比
//!   2 × 3
        _jacobianOplusXj = E1 * (R.matrix());
//        cout << "Xi " << _jacobianOplusXi << endl;
//        cout << "Xj " << _jacobianOplusXj << endl;
    }
#endif
    virtual bool read(std::istream &in) {}
    virtual bool write(std::ostream &out) const {}

private:
    double _a = 1.0;
    double _b = 0.0;
};
