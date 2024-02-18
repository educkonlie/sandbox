#include "globalPBA.h"
#include <random>


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