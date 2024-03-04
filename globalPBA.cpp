#include "globalPBA.h"
#include <random>

double global_pose_R_error = 0.01;
double global_pose_t_error = 0.01;
double global_landmark_t_error = 0.2;
Sophus::SE3d AddNoiseinPose(Sophus::SE3d &pose)
{
    static std::normal_distribution<double> pose_rotation_noise(0., global_pose_R_error);
    static std::normal_distribution<double> pose_position_noise(0., global_pose_t_error);
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
    static std::normal_distribution<double> landmark_position_noise(0., global_landmark_t_error);
//    static std::normal_distribution<double> landmark_position_noise(0., 0.0);
// 为初值添加随机噪声
    Vec3d noise(landmark_position_noise(generator),
                landmark_position_noise(generator),
                landmark_position_noise(generator));
    return point + noise;
}

// 参照slambook14 page 186-187，但是有两个区别，这里还多了畸变模型，并且归一化平面为负，所以还是需要重新推导一下
// 畸变模型在page 102
//! 线性化直和
void EdgeDirectProjection::linearizeOplus() {
//! 有两种顶点，一个是位姿+内参，一个是点（三维坐标）
    auto pose = (Pose *) _vertices[0];
    auto landmark = (Landmark *) _vertices[1];
    auto cam_est = pose->estimate();
    auto P= landmark->estimate();

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
//        _jacobianOplusXi << E, E * (-Sophus::SO3d::hat(R * P).matrix());
    _jacobianOplusXi = E2;
//      生成三维点point的雅克比
//!   2 × 3
    _jacobianOplusXj = E1 * (R.matrix());
//        cout << "Xi " << _jacobianOplusXi << endl;
//        cout << "Xj " << _jacobianOplusXj << endl;
}