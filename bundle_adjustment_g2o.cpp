#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel_impl.h>
#include <iostream>

#include "common.h"
#include "sophus/se3.hpp"

#include "IndexThreadReduce.h"

#include <ctime>
#include <cstdlib>
#include <chrono>
#include <cstdio>
//#include <pthread.h>

#include "pcg.h"
#include "qr.h"

using namespace Eigen;
using namespace Sophus;
using namespace std;

/// 姿态和内参的结构
struct PoseAndIntrinsics {
    PoseAndIntrinsics() {}

    /// set from given data address
    explicit PoseAndIntrinsics(double *data_addr) {
        rotation = SO3d::exp(Vector3d(data_addr[0], data_addr[1], data_addr[2]));
        translation = Vector3d(data_addr[3], data_addr[4], data_addr[5]);
        focal = data_addr[6];
        k1 = data_addr[7];
        k2 = data_addr[8];
    }

    /// 将估计值放入内存
    const void set_to(double *data_addr) {
        auto r = rotation.log();
        for (int i = 0; i < 3; ++i) data_addr[i] = r[i];
        for (int i = 0; i < 3; ++i) data_addr[i + 3] = translation[i];
        data_addr[6] = focal;
        data_addr[7] = k1;
        data_addr[8] = k2;
    }

    SO3d rotation;
    Vector3d translation = Vector3d::Zero();
    double focal = 0;
    double k1 = 0, k2 = 0;
};

/// 位姿加相机内参的顶点，9维，前三维为so3 R，接下去为t, f, k1, k2
class VertexPoseAndIntrinsics : public g2o::BaseVertex<9, PoseAndIntrinsics> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPoseAndIntrinsics() {}

    virtual void setToOriginImpl() override {
        _estimate = PoseAndIntrinsics();
    }

    //! 可以借用g2o的接口，实现自己的J, r的维护，和delta的更新，以及回退
    //! oplus是直和
    virtual void oplusImpl(const double *update) override {
        _estimate.rotation = SO3d::exp(Vector3d(update[0], update[1], update[2])) * _estimate.rotation;
        _estimate.translation += Vector3d(update[3], update[4], update[5]);
        _estimate.focal += update[6];
        _estimate.k1 += update[7];
        _estimate.k2 += update[8];
    }

    /// 根据估计值投影一个点
    Vector2d project(const Vector3d &point) {
        // Rp + t
        Vector3d pc = _estimate.rotation * point + _estimate.translation;
        pc = -pc / pc[2]; /// 注意这里有负号，归一化平面似乎是在相机背后
        double r2 = pc.squaredNorm();
        double distortion = 1.0 + r2 * (_estimate.k1 + _estimate.k2 * r2);
        return Vector2d(_estimate.focal * distortion * pc[0],
                        _estimate.focal * distortion * pc[1]);
    }

    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}
};

class VertexPoint : public g2o::BaseVertex<3, Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPoint() {}

    virtual void setToOriginImpl() override {
        _estimate = Vector3d(0, 0, 0);
    }

    virtual void oplusImpl(const double *update) override {
        _estimate += Vector3d(update[0], update[1], update[2]);
    }

    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}
};

//! one row for an edge, .......Jp_i.......  .........Jl_j........  res = P(i, j) - observe
class EdgeProjection :
    public g2o::BaseBinaryEdge<2, Vector2d, VertexPoseAndIntrinsics, VertexPoint> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // _error = f(_measurement, v->estimate())
    virtual void computeError() override {
        auto v0 = (VertexPoseAndIntrinsics *) _vertices[0];
        auto v1 = (VertexPoint *) _vertices[1];
        auto proj = v0->project(v1->estimate());
        _error = proj - _measurement;
    }

    // not use numeric derivatives
    // TODO compute J
#if 0
// 参照slambook14 page 186-187，但是有两个区别，这里还多了畸变模型，并且归一化平面为负，所以还是需要重新推导一下
    // 畸变模型在page 102
    //! 线性化直和
    virtual void linearizeOplus() override {
        //! 有两种顶点，一个是位姿+内参，一个是点（三维坐标）
        auto v0 = (VertexPoseAndIntrinsics *) _vertices[0];
        auto v1 = (VertexPoint *) _vertices[1];
        auto v0_est = v0->estimate();
        auto P= v1->estimate();

        double f = v0_est.focal;
        double k1 = v0_est.k1;
        double k2 = v0_est.k2;

        Sophus::SO3d R = v0_est.rotation;
        Vector3d t = v0_est.translation;

        // 生成估计值T处的扰动{\delta}{\xi}的雅克比
        Vector3d Pc = v0_est.rotation * P + v0_est.translation;
        double Xc = Pc[0];
        double Yc = Pc[1];
        double Zc = Pc[2];

        double r2 = (Xc * Xc + Yc * Yc + Zc * Zc) / (Zc * Zc);
        double D = 1 + k1 * r2 + k2 * r2 * r2;
        double A = (k1 + 2 * k2 * r2) * (2 * Xc) / (Zc * Zc);
        double B = (k1 + 2 * k2 * r2) * (2 * Yc) / (Zc * Zc);
        double C = (k1 + 2 * k2 * r2) * (-2 * (Xc * Xc + Yc * Yc)) / (Zc * Zc * Zc);
//        D = 1;
//        A = B = C = 0;

        Matrix<double, 2, 3> E;
        // row 0
        E(0, 0) = -f * (Xc * A + D) / Zc;
        E(0, 1) = -f * Xc * B / Zc;
        E(0, 2) = -f * Xc * (-D / (Zc * Zc) + C / Zc);
        // row 1
        E(1, 0) = -f * Yc * A / Zc;
        E(1, 1) = -f * (Yc * B + D) / Zc;
        E(1, 2) = -f * Yc * (-D / (Zc * Zc) + C / Zc);
//
//        Matrix<double, 3, 6> F;
//        Matrix3d Pc_hat = Matrix3d::Zero();
//        Pc_hat(0, 1) = - Zc;
//        Pc_hat(0, 2) = Yc;
//        Pc_hat(1, 0) = Zc;
//        Pc_hat(1, 2) = - Xc;
//        Pc_hat(2, 0) = - Yc;
//        Pc_hat(2, 1) = Xc;

        Matrix<double, 2, 3> Jk = Matrix<double, 2, 3>::Zero();
        Jk(0, 0) = -D * Xc / Zc;
        Jk(0, 1) = -r2 * f * Xc / Zc;
        Jk(0, 2) = -r2 * r2 * f * Xc / Zc;
        Jk(1, 0) = -D * Yc / Zc;
        Jk(1, 1) = -r2 * f * Yc / Zc;
        Jk(1, 2) = -r2 * r2 * f * Yc / Zc;
//        Jk = Matrix<double, 2, 3>::Zero();

//!   2 × 9
        _jacobianOplusXi << -E * Sophus::SO3d::hat(R * P).matrix(), E, Jk;

//      生成三维点point的雅克比
//!   2 × 3
        _jacobianOplusXj
                = E * R.matrix();
//        cout << "Xi " << _jacobianOplusXi << endl;
//        cout << "Xj " << _jacobianOplusXj << endl;
    }
    // TODO done
#endif
    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}

};

void SolveBA(BALProblem &bal_problem);


//void test_my_solver_pcg_and_sc();
//void test_qr();
//void test_householderQr();
//void test_pcg_parallel();
//
//void test_my_thread_pool();

#ifdef NEW_METHOD
struct Jacobi {
    vector<int > cams;
//    int point;
    vector<Matrix<double, 2, 9> > Jps;
    vector<Matrix<double, 2, 3> > Jls;
    vector<Matrix<double, 2, 1> > Jrs;
};

void my_linearizeOplus(Matrix<double, 2, 9> &pose,
                       Matrix<double, 2, 3> &landmark,
                       VertexPoseAndIntrinsics *v0, VertexPoint *v1);
void my_computeError(VertexPoseAndIntrinsics *v0, VertexPoint *v1, Vector2d measurement, Vector2d &residual)
{
//    auto v0 = (VertexPoseAndIntrinsics *) _vertices[0];
//    auto v1 = (VertexPoint *) _vertices[1];
    auto proj = v0->project(v1->estimate());
    residual = proj - measurement;
}
#endif

class TicToc {
public:
    TicToc() {tic();}

    void tic() {start = std::chrono::system_clock::now();}

    double toc()
    {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        return elapsed_seconds.count() /** 1000*/;
    }

private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};

TicToc timer_ACC1;
double times_ACC1 = 0.0;
TicToc timer_ACC2;
double times_ACC2 = 0.0;
TicToc timer_ACC3;
double times_ACC3 = 0.0;
TicToc timer_ACC4;
double times_ACC4 = 0.0;
TicToc timer_ACC5;
double times_ACC5 = 0.0;

void my_solver_normal_equation(MatXX &, VectorXd &, MatXX &, VectorXd &);
void my_solver_sparse(size_t &st, MatXX &J_total, VectorXd &r_total, MatXX &J, VectorXd &r);
void SolveBA(BALProblem &bal_problem)
{
    const int point_block_size = bal_problem.point_block_size();
    const int camera_block_size = bal_problem.camera_block_size();
    double *points = bal_problem.mutable_points();
    double *cameras = bal_problem.mutable_cameras();

    // pose dimension 9, landmark is 3
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<9, 3>> BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
    // use LM
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
//    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
//        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    /// build g2o problem
    const double *observations = bal_problem.observations();
    // vertex
    vector<VertexPoseAndIntrinsics *> vertex_pose_intrinsics;
    vector<VertexPoint *> vertex_points;
    //! 制作Jp,  列数为num_cameras * Jp
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        VertexPoseAndIntrinsics *v = new VertexPoseAndIntrinsics();
        double *camera = cameras + camera_block_size * i;
        v->setId(i);
        v->setEstimate(PoseAndIntrinsics(camera));
        optimizer.addVertex(v);
        vertex_pose_intrinsics.push_back(v);
    }
    //! 制作Jl, 列数为num_points * Jl
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        VertexPoint *v = new VertexPoint();
        double *point = points + point_block_size * i;
        v->setId(i + bal_problem.num_cameras());
        v->setEstimate(Vector3d(point[0], point[1], point[2]));
        // g2o在BA中需要手动设置待Marg的顶点
        // 然后g2o会知道这个点需要被shur
        v->setMarginalized(true);
        optimizer.addVertex(v);
        vertex_points.push_back(v);
    }

    // edge
    //! 制作residual, 行数为num_observations

#ifdef NEW_METHOD
    vector<struct Jacobi > Js;
    for (int i = 0; i < bal_problem.num_points(); i++) {
        struct Jacobi *temp = new(struct Jacobi);
//        temp.point = i;
        Js.push_back(*temp);
    }
#endif
    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        EdgeProjection *edge = new EdgeProjection;
        //! 应该是通过i关联到相应的v0, v1
        //! 一条边对应一次观测，对应一个三维路标 和 一个相机
        //! 改为PBA后， 也要沿用这个做法，事先从DSO中为每一个点记录共视图
//        std::cout << "cam: " << bal_problem.camera_index()[i] << std::endl;
//        std::cout << "point: " << bal_problem.point_index()[i] << std::endl;
#ifdef NEW_METHOD
        Js[bal_problem.point_index()[i]].cams.push_back(bal_problem.camera_index()[i]);

        Matrix<double, 2, 9> Jp = Matrix<double, 2, 9>::Zero();
        Matrix<double, 2, 3> Jl = Matrix<double, 2, 3>::Zero();
        Matrix<double, 2, 1> Jr = Matrix<double, 2, 1>::Zero();
        my_linearizeOplus(Jp, Jl, vertex_pose_intrinsics[bal_problem.camera_index()[i]],
                          vertex_points[bal_problem.point_index()[i]]);
//        std::cout << "residual:" << std::endl;
//        std::cout << Vector2d(observations[2 * i + 0], observations[2 * i + 1])  << std::endl;
        my_computeError(vertex_pose_intrinsics[bal_problem.camera_index()[i]],
                        vertex_points[bal_problem.point_index()[i]],
                        Vector2d(observations[2 * i + 0], observations[2 * i + 1]),
                        Jr);
        Js[bal_problem.point_index()[i]].Jps.push_back(Jp);
        Js[bal_problem.point_index()[i]].Jls.push_back(Jl);
        Js[bal_problem.point_index()[i]].Jrs.push_back(Jr);
#endif

        //! camera_index和point_index应该是edge_index -> camera_index,  edge_index -> point_index
        edge->setVertex(0, vertex_pose_intrinsics[bal_problem.camera_index()[i]]);
        edge->setVertex(1, vertex_points[bal_problem.point_index()[i]]);

        edge->setMeasurement(Vector2d(observations[2 * i + 0], observations[2 * i + 1]));
        edge->setInformation(Matrix2d::Identity());
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(edge);
    }
#ifdef NEW_METHOD
//    ...........pose..........         r
//    ....pose.................         r
    MatXX H = MatXX::Zero(bal_problem.num_cameras() * 9, bal_problem.num_cameras() * 9);
    VectorXd b = VectorXd::Zero(bal_problem.num_cameras() * 9);

    MatXX J = MatXX::Zero(bal_problem.num_observations() * 2, bal_problem.num_cameras() * 9);
    VectorXd r = VectorXd::Zero(bal_problem.num_observations() * 2);

    std::cout << "H rows:\n" << H.rows() << std::endl;
    std::cout << "H cols:\n" << H.cols() << std::endl;
    std::cout << "J rows:\n" << J.rows() << std::endl;
    std::cout << "J cols:\n" << J.cols() << std::endl;

    size_t start_row = 0;
    for (int i = 0; i < bal_problem.num_points(); i++) {
//        std::cout << "\npoint: " << i << std::endl;
        MatXX J1, J2;
        int ncams = Js[i].cams.size();
        J1 = MatXX::Zero(ncams * 2, ncams * 9 + 1);
        J2 = MatXX::Zero(ncams * 2, 3);
        for (int j = 0; j < ncams; j++) {
            //!将每个landmark的J, r扩展到J1, J2中，其中J2为需要被边缘化的landmark
            J1.block(j * 2, j * 9, 2, 9) = Js[i].Jps[j];
            J1.block(j * 2, ncams * 9, 2, 1) = Js[i].Jrs[j];
            J2.block(j * 2, 0, 2, 3) = Js[i].Jls[j];
        }
//        timer_ACC1.tic();
        qr(J1, J2);
//        times_ACC1 += timer_ACC1.toc();
        //! 去掉上3行
        //! temp1为Jp,  temp2为r
        MatXX temp1 = MatXX::Zero(J1.rows() - 3, bal_problem.num_cameras() * 9);
        VectorXd temp2 = J1.block(3, ncams * 9, J1.rows() - 3, 1);

        for (int j = 0; j < Js[i].cams.size(); j++) {
            temp1.block(0, Js[i].cams[j] * 9, J1.rows() - 3, 9)
                    = J1.block(3, j * 9, J1.rows() - 3, 9);
        }
        timer_ACC1.tic();
        my_solver_normal_equation(H, b, temp1, temp2);
        times_ACC1 += timer_ACC1.toc();

        timer_ACC2.tic();
        my_solver_sparse(start_row, J, r, temp1, temp2);
        times_ACC2 += timer_ACC2.toc();
//        std::cout << "start_row: " << start_row << std::endl;
//        std::cout << "J1:\n" << J1 << std::endl;
//        std::cout << "J2:\n" << J2 << std::endl;
    }

    MatXX J_prime = J.block(0, 0, start_row, J.cols());
//    r.resize(start_row);
    VectorXd r_prime = r.segment(0, start_row);

    std::cout << "\nstart................." << J_prime.rows() << " " << J_prime.cols() << std::endl;

//    std::cout << "\nH rows: " << H.rows() << std::endl;
//    std::cout << "H cols: " << H.cols() << std::endl;
//    std::cout << "H size: " << H.size() << std::endl;
//    std::cout << "b rows: " << b.rows() << std::endl;
//    std::cout << "b size: " << b.size() << std::endl;

    MatXX H2 = J_prime.transpose() * J_prime;
    VectorXd b2 = J_prime.transpose() * r_prime;

    timer_ACC3.tic();
//    std::cout << "....b1....\n" << b.transpose() << std::endl;
//    std::cout << "....b2....\n" << (J_prime.transpose() * r_prime).transpose() << std::endl;
//    std::cout << "H1:\n" << H << std::endl;


    for (int i = 0; i < H.rows(); i++)
        for (int j = 0; j < H.cols(); j++)
            if (std::abs(H(i, j) - H2(i, j)) > 1e-5)
                std::cout << "......H....." << H(i, j) << " " << H2(i, j) << " "
                        << std::abs(H(i, j) - H2(i, j)) << std::endl;
    for (int i = 0; i < b2.rows(); i++)
        if (std::abs(b2(i) - b(i)) > 1e-5)
            std::cout << "...b..." << b(i) << " " << b2(i) << " "
                    << std::abs(b2(i) - b(i)) << std::endl;

//    std::cout << "H2:\n" << (J_prime.transpose() * J_prime) << std::endl;
    std::cout << "1 H b x:\n" << H.ldlt().solve(b).transpose() << std::endl;
    times_ACC3 += timer_ACC3.toc();

    std::cout << "2 H b x:\n" << H2.ldlt().solve(b2).transpose() << std::endl;


    timer_ACC4.tic();
    Eigen::LeastSquaresConjugateGradient<MatXX > lscg;
    lscg.setMaxIterations(10000);
    lscg.setTolerance(1e-4);
    lscg.compute(J_prime);
    VectorXd x = lscg.solve(r_prime);
    times_ACC4 += timer_ACC4.toc();

    timer_ACC5.tic();
    std::cout << "qr solve x: " << J.colPivHouseholderQr().solve(r).transpose() << std::endl;
    std::cout << "qr solve x: " << J_prime.colPivHouseholderQr().solve(r_prime).transpose() << std::endl;
    times_ACC5 += timer_ACC5.toc();

    std::cout << "\n#iterations:     " << lscg.iterations() << std::endl;
    std::cout << "estimated error: " << lscg.error()      << std::endl;
    std::cout << "x: " << x.transpose() << std::endl;

    std::cout << "prepare q r: " << times_ACC1 << std::endl;
    std::cout << "prepare J r: " << times_ACC2 << std::endl;
    std::cout << "solve H b: " << times_ACC3 << std::endl;
    std::cout << "solve J r: " << times_ACC4 << std::endl;
    std::cout << "solve qr : " << times_ACC5 << std::endl;
#endif

    optimizer.initializeOptimization();
    optimizer.optimize(40);

    // set to bal problem
    //! output
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        double *camera = cameras + camera_block_size * i;
        auto vertex = vertex_pose_intrinsics[i];
        auto estimate = vertex->estimate();
        //! output to camera
        estimate.set_to(camera);
    }
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        double *point = points + point_block_size * i;
        auto vertex = vertex_points[i];
        //! output to point
        for (int k = 0; k < 3; ++k) point[k] = vertex->estimate()[k];
    }
}

#ifdef NEW_METHOD
void my_linearizeOplus(Matrix<double, 2, 9> &pose,
                       Matrix<double, 2, 3> &landmark,
                       VertexPoseAndIntrinsics *v0, VertexPoint *v1)
{
//! 有两种顶点，一个是位姿+内参，一个是点（三维坐标）
    auto v0_est = v0->estimate();
    auto P = v1->estimate();

    double f = v0_est.focal;
    double k1 = v0_est.k1;
    double k2 = v0_est.k2;

    Sophus::SO3d R = v0_est.rotation;
    Vector3d t = v0_est.translation;

// 生成估计值T处的扰动{\delta}{\xi}的雅克比
    Vector3d Pc = v0_est.rotation * P + v0_est.translation;
    double Xc = Pc[0];
    double Yc = Pc[1];
    double Zc = Pc[2];

    double r2 = (Xc * Xc + Yc * Yc + Zc * Zc) / (Zc * Zc);
    double D = 1 + k1 * r2 + k2 * r2 * r2;
    double A = (k1 + 2 * k2 * r2) * (2 * Xc) / (Zc * Zc);
    double B = (k1 + 2 * k2 * r2) * (2 * Yc) / (Zc * Zc);
    double C = (k1 + 2 * k2 * r2) * (-2 * (Xc * Xc + Yc * Yc)) / (Zc * Zc * Zc);

//        D = 1;
//        A = B = C = 0;

    Matrix<double, 2, 3> E;
// row 0
    E(0, 0) = -f * (Xc * A + D) / Zc;
    E(0, 1) = -f * Xc * B / Zc;
    E(0, 2) = -f * Xc * (-D / (Zc * Zc) + C / Zc);
// row 1
    E(1, 0) = -f * Yc * A / Zc;
    E(1, 1) = -f * (Yc * B + D) / Zc;
    E(1, 2) = -f * Yc * (-D / (Zc * Zc) + C / Zc);
//
    Matrix<double, 2, 3> Jk = Matrix<double, 2, 3>::Zero();
    Jk(0, 0) = -D * Xc / Zc;
    Jk(0, 1) = -r2 * f * Xc / Zc;
    Jk(0, 2) = -r2 * r2 * f * Xc / Zc;
    Jk(1, 0) = -D * Yc / Zc;
    Jk(1, 1) = -r2 * f * Yc / Zc;
    Jk(1, 2) = -r2 * r2 * f * Yc / Zc;
//        Jk = Matrix<double, 2, 3>::Zero();

//!   2 × 9
    pose.block<2, 3>(0, 0) = -E * Sophus::SO3d::hat(R * P).matrix();
    pose.block<2, 3>(0, 3) = E;
    pose.block<2, 3>(0, 6) = Jk;

//      生成三维点point的雅克比
//!   2 × 3
//    _jacobianOplusXj
//            = E * R.matrix();
//        cout << "Xi " << _jacobianOplusXi << endl;
//        cout << "Xj " << _jacobianOplusXj << endl;
    landmark = E * R.matrix();
}
#endif

void my_solver_normal_equation(MatXX &H, VecX &b, MatXX &J, VecX &r)
{
    H += J.transpose() * J;
    b += J.transpose() * r;
}
void my_solver_sparse(size_t &start_row, MatXX &J_total, VecX &r_total, MatXX &J, VecX &r)
{
    J_total.block(start_row, 0, J.rows(), J.cols()) = J;
    r_total.segment(start_row, r.rows()) = r;

    start_row += J.rows();
}
#if 0
void test_conservertiveResize()
{
    MatXX A = MatXX::Random(10, 100);
    timer_ACC1.tic();
    for (int i = 0; i < 500; i++) {
        A.conservativeResize(A.rows() + 9, A.cols());
    }
    times_ACC1 += timer_ACC1.toc();
    timer_ACC2.tic();
    for (int i = 0; i < 500; i++) {
        A.setZero();
    }
    times_ACC2 += timer_ACC2.toc();
    std::cout << times_ACC1 << std::endl;
    std::cout << times_ACC2 << std::endl;
}

void test_my_solver_pcg_and_sc()
{
    MatXX J = MatXX::Random(76000, 792);
    VecX r = VecX::Random(76000);

    IndexThreadReduce<Vec10> *thread_pool = new IndexThreadReduce<Vec10>();

    my *my1 = new my();

    std::cout << "x size\n" << J.cols() << std::endl;
    std::cout << "residual size\n" << r.rows() << std::endl;

    VecX x;
    VecX z;

    MatXX A[76];
    VecX b[76];
    for (int i = 0; i < 76; i++) {
        A[i] = J.block(i * 1000, 0, 1000, 792);
        b[i] = r.segment(i * 1000, 1000);
    }

    timer_ACC1.tic();
//    my1->cg(J, r, x, 1e-6, J.cols());
//    std::cout << "c g  x:\n" << x.transpose() << std::endl;
    my1->pcgMT(thread_pool, A, b, 76, x, 1e-8, A[0].cols(), false);
    std::cout << "pcgMT noMT x:\n" << x.transpose() << std::endl;
    times_ACC1 += timer_ACC1.toc();

//    z.setZero();

    timer_ACC2.tic();
//    my_pcg(J, r, z, 1e-10, J.cols());
//    std::cout << "pcg  x:\n" << z.transpose() << std::endl;
    my1->pcgMT(thread_pool, A, b, 76, z, 1e-8, A[0].cols(), true);
    std::cout << "pcgMT   MT x:\n" << z.transpose() << std::endl;
    times_ACC2 += timer_ACC2.toc();

//    z.setZero();

    timer_ACC3.tic();
    my1->pcg(J, r, z, 1e-8, J.cols());
    std::cout << "pcg  x:\n" << z.transpose() << std::endl;
//    my_pcgMT_tbb(A, b, 76, z, 1e-6, A[0].cols());
//    std::cout << "pcg_tbb x:\n" << z.transpose() << std::endl;
    times_ACC3 += timer_ACC3.toc();

    timer_ACC4.tic();
    std::cout << "H b  x:\n" << (J.transpose() * J).ldlt().solve(J.transpose() * r).transpose() << std::endl;
    times_ACC4 += timer_ACC4.toc();

#if 0
    timer_ACC5.tic();
    Eigen::LeastSquaresConjugateGradient<MatXX > lscg;
//    lscg.setMaxIterations(100);
    lscg.setTolerance(1e-8);
    lscg.compute(J);
    VecX y = lscg.solve(r);

    std::cout << "lscg  x:\n" << y.transpose() << std::endl;
    times_ACC5 += timer_ACC5.toc();
    std::cout << "lscg iter:\n" << lscg.iterations() << std::endl;
#endif
    std::cout << "......" << std::endl;
    std::cout << times_ACC1 << std::endl;
    std::cout << times_ACC2 << std::endl;
    std::cout << times_ACC3 << std::endl;
    std::cout << times_ACC4 << std::endl;
    std::cout << times_ACC5 << std::endl;

    delete my1;
}
#endif



//! 并行计算的线程池
//! 并行计算的线程池是将比如1000条数据用10个线程处理，每个线程处理100条，然后写到10个缓冲区里，等10个线程join了后，
//! 再用一个串行操作(for循环)把缓冲区数据累加输出。
//! 在这里面没有锁，也没有线程切换（之前一直没有搞清楚，没有跟并发区分清楚）

#if 1
int main(int argc, char **argv)
{
    if (argc != 2) {
        cout << "usage: bundle_adjustment_g2o bal_data.txt" << endl;
        return 1;
    }

    BALProblem bal_problem(argv[1]);
    bal_problem.Normalize();
    bal_problem.Perturb(0.1, 0.5, 0.5);
    bal_problem.WriteToPLYFile("initial.ply");
//    bal_problem.myDraw("init");
    SolveBA(bal_problem);
    bal_problem.WriteToPLYFile("final.ply");
    bal_problem.myDraw("final");

    return 0;
}
#endif
/*
 * In this code:

The matrix_multiply_avx function performs matrix multiplication using AVX intrinsics.
 It uses nested loops to iterate through the matrices A, B, and C.

Within the innermost loop, AVX intrinsics are used to load rows of matrix A,
 broadcast values from matrix B, and perform the multiply-and-accumulate operation.

The result is stored back in matrix C using _mm256_storeu_ps.

In the main function, matrices A and B are initialized with appropriate values,
 and the matrix_multiply_avx function is called to compute the result in matrix C.
 Make sure to adjust the matrix size (N) and populate matrices A and B
 with the desired data before running the code.
 Additionally, consider aligning your data properly for better performance.
 */
#if 0
#include <immintrin.h>  //! avx
#include <emmintrin.h>  //! sse2
#include <xmmintrin.h>
#include <stdio.h>

#define N 8  // Matrix size (N x N)

//! 一个数乘一个向量
//! float * [float, float, float...float]
void scalar_mul_vector_avx(float *s, float *B, float *C)
{
    __m256 sum = _mm256_setzero_ps();  // Initialize a 256-bit vector to zero

    __m256 a = _mm256_loadu_ps(B);  // Load a row of matrix A
    __m256 b = _mm256_broadcast_ss(s);  // Broadcast a value from matrix B
//    sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));  // Multiply and accumulate
    sum = _mm256_fmadd_ps(a, b, sum);

    _mm256_storeu_ps(C, sum);  // Store the result in matrix C
}
void scalar_mul_vector_128(float *s,  float *B, float *C)
{
    __m128 sum = _mm_setzero_ps();

    __m128 a = _mm_loadu_ps(B);
    __m128 b = _mm_broadcast_ss(s);
    sum = _mm_fmadd_ps(a, b, sum);

    _mm_storeu_ps(C, sum);
}

void matrix_mul_vector_avx(float *A, float *b, float *C)
{
    __m256 sum = _mm256_setzero_ps();  // Initialize a 256-bit vector to zero
    for (int i = 0; i < 8; i++) {
        __m256 x = _mm256_loadu_ps(A + i * 8);  //! 它是按照A的类型来计算i的步长的
        __m256 y = _mm256_broadcast_ss(b + i);  // Broadcast a value from matrix B
//        sum = _mm256_add_ps(sum, _mm256_mul_ps(x, y));  // Multiply and accumulate
        sum = _mm256_fmadd_ps(x, y, sum);
    }
    _mm256_storeu_ps(C, sum);
}
void matrix_mul_vector_128(float *A, float *b, float *C)
{
    __m128 sum = _mm_setzero_ps();  // Initialize a 256-bit vector to zero
    for (int i = 0; i < 4; i++) {
        __m128 x = _mm_loadu_ps(A + i * 4);  //! 它是按照A的类型来计算i的步长的
        __m128 y = _mm_broadcast_ss(b + i);  // Broadcast a value from matrix B
//        sum = _mm256_add_ps(sum, _mm256_mul_ps(x, y));  // Multiply and accumulate
        sum = _mm_fmadd_ps(x, y, sum);
    }
    _mm_storeu_ps(C, sum);
}

#include <emmintrin.h>
#include <stdio.h>

#define N 4  // Matrix size (N x N)

void matrix_multiply_sse2(float* A, float* B, float* C) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            __m128 sum = _mm_setzero_ps();  // Initialize a 128-bit vector to zero

            for (int k = 0; k < N; ++k) {
                __m128 a = _mm_loadu_ps(&A[i * N + k]);  // Load a row of matrix A
                __m128 b = _mm_loadu_ps(&B[k * N + j]);  // Load a column of matrix B
                sum = _mm_add_ps(sum, _mm_mul_ps(a, b));  // Multiply and accumulate
            }

            _mm_storeu_ps(&C[i * N + j], sum);  // Store the result in matrix C
        }
    }
}

int main() {
    float A[N][N];  // Matrix A
    float B[N][N];  // Matrix B
    float C[N][N];  // Result Matrix C

    // Initialize matrices A and B (you should fill these with appropriate values)
    // ...

    matrix_multiply_sse2((float*)A, (float*)B, (float*)C);

    // Print the result matrix C
    printf("Result Matrix C:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%f ", C[i][j]);
        }
        printf("\n");
    }

    return 0;
}
#endif