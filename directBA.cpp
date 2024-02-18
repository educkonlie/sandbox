#include "globalPBA.h"

//! 关于术语使用
//! 输入输出使用 camera point  表明是纯数值
//! 内部优化包括g2o优化使用 pose landmark，这是两个vertex

// global variables
std::string camera_file = "/mnt/data/dso-4-reading/result.txt";
std::string point_file = "/mnt/data/dso-4-reading/point_cloud.txt";

/*
        dx 0 dy -2
        dx -1 dy -1
        dx 1 dy -1
        dx -2 dy 0
        dx 0 dy 0
        dx 2 dy 0
        dx -1 dy 1
        dx 0 dy 2
*/
int pattern_dx[8] = {0, -1, 1, -2, 0, 2, -1, 0};
int pattern_dy[8] = {-2, -1, -1, 0, 0, 0, 1, 2};

// intrinsics
//float fx = 277.34;
//float fy = 291.402;
//float cx = 312.234;
//float cy = 239.777;

//float fx = 706.01;
//float fy = 703.522;
//float cx = 600.911;
//float cy = 182.127;

float fx = 705.919;
float fy = 703.522;
float cx = 600.928;
float cy = 182.121;
//float fx = 700.0;
//float fy = 700.0;
//float cx = 0.0;
//float cy = 0.0;

// plot the cams and points for you, need pangolin
void Draw(std::string title, const VecSE3d &cams, const VecVec3d points[], const vector<int > &host);
void printResult(std::string file, const VecSE3d &cams);

int main(int argc, char **argv)
{
    // read poses and points
    VecSE3d cams;
    VecVec3d points[100 * 100];
    std::vector<Vec8d > color[100 * 100];

    std::ifstream fin(camera_file);
    while (!fin.eof()) {
#if 1
        double data[12];
        for (auto &d: data)
            fin >> d;
        Mat33d R;
        R(0, 0) = data[0];
        R(0, 1) = data[1];
        R(0, 2) = data[2];
        R(1, 0) = data[4];
        R(1, 1) = data[5];
        R(1, 2) = data[6];
        R(2, 0) = data[8];
        R(2, 1) = data[9];
        R(2, 2) = data[10];
        Vec3d t;
        t[0] = data[3];
        t[1] = data[7];
        t[2] = data[11];
        cams.push_back(Sophus::SE3d(Sophus::SO3d(R), t));
#endif
        if (!fin.good())
            break;
    }
    fin.close();

    fin.open(point_file);
    int num_of_host = 0;
    vector<int > host;
    vector<int > marginalizedAt;

    while (!fin.eof()) {
        int num_of_points;
        int host_id;
        int marg_id;
        fin >> host_id;
        fin >> marg_id;
        fin >> num_of_points;
        host.push_back(host_id);
        marginalizedAt.push_back(marg_id);
        for (int i = 0; i < num_of_points; i++) {
            double xyz[3] = {0};
            for (int j = 0; j < 3; j++)
                fin >> xyz[j];
            if (xyz[0] == 0)
                break;
            points[host_id].push_back(Eigen::Vector3d(xyz[0], xyz[1], xyz[2]));
            Vec8d c;
            for (int i = 0; i < 8; i++)
                fin >> c[i];
            color[host_id].push_back(c); // color数组元素跟points数组元素一一对应。

            if (fin.good() == false)
                break;
        }
    }
    fin.close();

    int points_num = 0;
    for (int i = 0; i < cams.size(); i++)
        points_num += points[i].size();

    cout << "cams: " << cams.size() << ", points: " << points_num << endl;
    cout << "observations(edges): " << cams.size() * points_num << endl;
    int obs = 0;
    for (int i = 0; i < host.size(); i++) {
        int host_id = host[i];
        int marg_id = marginalizedAt[i];
        for (int j = host_id; j < marg_id; j++) {
            obs += points[host_id].size();
        }
    }
    cout << "observations(edges): " << obs << endl;

#if 1
    // build optimization problem
// 构建图优化，先设定g2o
    std::cout << "1....." << std::endl;
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;  // pose is 6, landmark is 3
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型

    std::cout << "2....." << std::endl;
    // 梯度下降方法，可以从GN, LM, DogLeg 中选
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
//    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
//            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

    std::cout << "3....." << std::endl;
    g2o::SparseOptimizer optimizer;     // 图模型
    optimizer.setAlgorithm(solver);   // 设置求解器
    optimizer.setVerbose(true);       // 打开调试输出

    std::cout << "4....." << std::endl;

    // TODO add vertices, edges into the graph optimizer
    // START YOUR CODE HERE
    //! 按host循环
    // 一个路标一个顶点，一个位姿也是一个顶点。
    vector<Landmark *> landmarks[100 * 100]; // 最多100 * 100个host
    int a = 1;
    for (int host_id : host) {
        for (Vec3d p : points[host_id]) {
            Landmark *landmark = new Landmark();
            landmark->setId(a++);
            landmark->setEstimate(p);
            landmark->setMarginalized(true);

            optimizer.addVertex(landmark);
            landmarks[host_id].push_back(landmark);
        }
    }
    std::cout << "5....." << std::endl;

    vector<Pose *> poses;
    for (int i = 0; i < cams.size(); i++) {
        Pose *pose = new Pose();
        pose->setId(a++);
//        assert(v_s->getId() == i + points_num + 10);
        //! 这里应该错了，应该是Tcw而不是Twc
        pose->setEstimate(cams[i].inverse());
        optimizer.addVertex(pose);
        poses.push_back(pose);
    }
    std::cout << "6....." << std::endl;
    // 每条边赋值一个color作为观测值，一个观测一条边，总共poses * points条边.
    //! 这里是稠密图了
//    int obs = 0;
    for (int l = 0; l < host.size(); l++) {
        int host_id = host[l];
        int marg_id = marginalizedAt[l];
//            points[host_id]
//            poses[host_id]....poses[marg_id]
        for (Landmark *land : landmarks[host_id]) {
            for (int i = host_id; i < marg_id; i++) {
                Vec2d measure = poses[i]->GetPixelValue(land->estimate());
                if (measure[0] < 10)
                    continue;
                if (measure[1] < 10)
                    continue;
                if (measure[0] >= 1224 - 10)
                    continue;
                if (measure[1] >= 368 - 10)
                    continue;
                //! 按j计数point，i为j对应的s的id和marginalizeAt（也可以
                EdgeDirectProjection *edge = new EdgeDirectProjection(/*images[i]*/);
                edge->setVertex(0, poses[i]); //! 投影的面从host_id....marg_id
                edge->setVertex(1, land); //! 投影的点为所有的vp[host_id]
                edge->setMeasurement(measure);
                edge->setInformation(Eigen::Matrix<double, 2, 2>::Identity());
                edge->setRobustKernel(new g2o::RobustKernelHuber());
                optimizer.addEdge(edge);
//                break;
            }
        }
    }
    // END YOUR CODE HERE

    /// 接下来还有一个很重要的步骤，就是将从DSO取得的poses, points加上噪声，制作poses_noisy, points_noisy
    /// 作为优化器的输入。优化在功能上也就是滤波降噪。
    VecSE3d cams_noisy;
    VecVec3d points_noisy[100 * 100];
    for (Sophus::SE3d cam : cams)
        cams_noisy.push_back(AddNoiseinPose(cam));
    for (int host_id : host) {
        for (Vec3d &pt: points[host_id])
            points_noisy[host_id].push_back(AddNoiseinPoint(pt));
    }

    for (int i = 0; i < cams_noisy.size(); i++)
        poses[i]->setEstimate(cams_noisy[i].inverse());

    for (int host_id : host) {
        for (int i = 0; i < landmarks[host_id].size(); i++)
            landmarks[host_id][i]->setEstimate(points_noisy[host_id][i]);
    }

    std::cout << "7....." << std::endl;
    // perform optimization
    Draw(string("before"), cams_noisy, points_noisy, host);
    optimizer.initializeOptimization(0);
    optimizer.optimize(20);
//    optimizer.optimize(0);

    // TODO fetch data from the optimizer
    // START YOUR CODE HERE
    for (int i = 0; i < poses.size(); i++)
        cams[i] = poses[i]->estimate().inverse();
    for (int host_id : host) {
        for (int j = 0; j < points[host_id].size(); j++)
            points[host_id][j] = landmarks[host_id][j]->estimate();
    }
    // END YOUR CODE HERE
//    cout << "outlier: " << g_outlier << endl;
    // plot the optimized points and poses
#endif
    Draw(string("after"), cams, points, host);
    printResult(string("/mnt/data/dso-4-reading/gpba_result.txt"), cams);
    return 0;
}
