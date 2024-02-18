#include "globalPBA.h"

//! 关于术语使用
//! 输入输出使用 pose point  表明是纯数值
//! 内部优化包括g2o优化使用 cam landmark  表明是两个有着具体内容的对象

// global variables
std::string pose_file = "/mnt/data/dso-4-reading/result.txt";
std::string point_file = "/mnt/data/dso-4-reading/point_cloud.txt";
std::string aff_calib_file = "/mnt/data/dso-4-reading/aff_calib.txt";

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

// plot the poses and points for you, need pangolin
//void Draw(const VecSE3 &poses, const VecVec3d &points);
void Draw(std::string title, const VecSE3d &poses, const VecVec3d points[], const vector<int > &host);
void printResult(std::string file, const VecSE3d &poses);

int main(int argc, char **argv)
{
    // read poses and points
    VecSE3d poses;
    VecVec3d points[100 * 100];
    std::vector<Vec8d > color[100 * 100];
    std::vector<Vec16d > color16[100 * 100];

    std::ifstream fin(pose_file);
    while (!fin.eof()) {
#if 0
        double timestamp = 0;
        fin >> timestamp;
//        if (timestamp == 0)
//            break;
        double data[7];
        for (auto &d: data)
            fin >> d;
        poses.push_back(Sophus::SE3d(
                Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
                Eigen::Vector3d(data[0], data[1], data[2])
        ));
#endif
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
        poses.push_back(Sophus::SE3d(Sophus::SO3d(R), t));
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

    fin.open(aff_calib_file);
    std::vector<double > affLLa;
    std::vector<double > affLLb;

    while (!fin.eof()) {
        double a, b;
        fin >> a;
        fin >> b;
        affLLa.push_back(a);
        affLLb.push_back(b);
        if (fin.good() == false)
            break;
    }
    assert(affLLa.size() == poses.size());
    assert(affLLb.size() == poses.size());
    fin.close();
    std::cout << "aff_calib read " << affLLa.size() << std::endl;
    fin.close();

    int points_num = 0;
    for (int i = 0; i < poses.size(); i++)
        points_num += points[i].size();

    cout << "poses: " << poses.size() << ", points: " << points_num << endl;
    cout << "observations(edges): " << poses.size() * points_num << endl;
    int obs = 0;
    for (int i = 0; i < host.size(); i++) {
        int host_id = host[i];
        int marg_id = marginalizedAt[i];
        for (int j = host_id; j < marg_id; j++) {
            obs += points[host_id].size();
        }
    }
    cout << "observations(edges): " << obs << endl;

    // read images
    vector<cv::Mat> images;
    boost::format fmt("/mnt/data/kitti_dataset/sequences/06-sample/image_0/%06d.png");
    for (int i = 0; i < poses.size(); i++) {
        cv::Mat orig_image = cv::imread((fmt % (i + 0)).str(), 0);
//        cout << i << "orig image rows: " << orig_image.rows << " orig image cols: " << orig_image.cols << endl;
        cv::Mat image;
        cv::resize(orig_image, image, cv::Size(1224, 368));
        images.push_back(image);
//        cout << i << "image rows: " << images[i].rows << " image cols: " << images[i].cols << endl;
    }

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
            Landmark *landmark = new Landmark(host_id, affLLa[host_id], affLLb[host_id]);
            landmark->setId(a++);
            landmark->setEstimate(p);
            landmark->setMarginalized(true);

            optimizer.addVertex(landmark);
            landmarks[host_id].push_back(landmark);
        }
    }
    std::cout << "5....." << std::endl;

    vector<Cam *> cams;
    for (int i = 0; i < poses.size(); i++) {
        Cam *cam = new Cam(i, images[i], affLLa[i], affLLb[i]);

        cam->setId(a++);
//        assert(v_s->getId() == i + points_num + 10);
        //! 这里应该错了，应该是Tcw而不是Twc
        cam->setEstimate(poses[i].inverse());
        optimizer.addVertex(cam);
        cams.push_back(cam);
    }
#if 0
    for (int host_id : host) {
        for (int j = 0; j < points[host_id].size(); j++) {
            /*
            Vec3d pc = (poses[host_id].inverse()) * points[host_id][j];
            pc /= pc[2];
            double u = pc[0] * fx + cx;
            double v = pc[1] * fy + cy;
            if (u - 2 < 0 || u + 1 >= images[host_id].cols || v - 2 < 0 || v + 1 >= images[host_id].rows) {
//            if (u - 4 < 0 || u + 2 >= images[host_id].cols || v - 4 < 0 || v + 2 >= images[host_id].rows) {
                std::cout << "error!!!!!!!!!!!!" << std::endl;
            }
             */
            Vec16d tmp;
            tmp = cams[host_id]->GetPixelValue(points[host_id][j]);
                    /*
            int k = 0;
            for (int x = -2; x <= 1; x++)
                for (int y = -2; y <= 1; y++)
                    tmp[k++] = GetPixelValue(images[host_id], u + x, v + y);
                     */
            color16[host_id].push_back(tmp);
        }
    }
#endif

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
#ifdef DIRECT_METHOD
            Vec16d measure = cams[host_id]->GetPixelValue(land->estimate());
#endif
            for (int i = host_id; i < marg_id; i++) {
#ifndef DIRECT_METHOD
                Vec2d measure = cams[i]->GetPixelValue(land->estimate());
                if (measure[0] < 10)
                    continue;
                if (measure[1] < 10)
                    continue;
                if (measure[0] >= images[i].cols - 10)
                    continue;
                if (measure[1] >= images[i].rows - 10)
                    continue;
#endif
                //! 按j计数point，i为j对应的s的id和marginalizeAt（也可以
                EdgeDirectProjection *edge = new EdgeDirectProjection(/*images[i]*/);
                edge->setVertex(0, cams[i]); //! 投影的面从host_id....marg_id
                edge->setVertex(1, land); //! 投影的点为所有的vp[host_id]
#ifdef DIRECT_METHOD
                edge->set_ab();
#endif
                edge->setMeasurement(measure);
//                edge->setMeasurement(color16[host_id][j]);
//                edge->setMeasurement(color[host_id][j]);
//                edge->setInformation(Eigen::Matrix<double, 8, 8>::Identity());
#ifdef DIRECT_METHOD
                edge->setInformation(Eigen::Matrix<double, 16, 16>::Identity());
#else
                edge->setInformation(Eigen::Matrix<double, 2, 2>::Identity());
#endif
                edge->setRobustKernel(new g2o::RobustKernelHuber());
                optimizer.addEdge(edge);
//                break;
            }
        }
    }
    // END YOUR CODE HERE

    /// 接下来还有一个很重要的步骤，就是将从DSO取得的poses, points加上噪声，制作poses_noisy, points_noisy
    /// 作为优化器的输入。优化在功能上也就是滤波降噪。
    VecSE3d poses_noisy;
    VecVec3d points_noisy[100 * 100];
    for (Sophus::SE3d &pose : poses)
        poses_noisy.push_back(AddNoiseinPose(pose));
    for (int host_id : host) {
        for (Vec3d &pt: points[host_id])
            points_noisy[host_id].push_back(AddNoiseinPoint(pt));
    }

    for (int i = 0; i < poses_noisy.size(); i++)
        cams[i]->setEstimate(poses_noisy[i].inverse());

    for (int host_id : host) {
        for (int i = 0; i < landmarks[host_id].size(); i++)
            landmarks[host_id][i]->setEstimate(points_noisy[host_id][i]);
    }

    std::cout << "7....." << std::endl;
    // perform optimization
    Draw(string("before"), poses_noisy, points_noisy, host);
    optimizer.initializeOptimization(0);
    optimizer.optimize(20);
//    optimizer.optimize(0);

    // TODO fetch data from the optimizer
    // START YOUR CODE HERE
    for (int i = 0; i < poses.size(); i++)
        poses[i] = cams[i]->estimate().inverse();
    for (int host_id : host) {
        for (int j = 0; j < points[host_id].size(); j++)
            points[host_id][j] = landmarks[host_id][j]->estimate();
    }
    // END YOUR CODE HERE
//    cout << "outlier: " << g_outlier << endl;
    // plot the optimized points and poses
#endif
    Draw(string("after"), poses, points, host);
    printResult(string("/mnt/data/dso-4-reading/gpba_result.txt"), poses);
    return 0;
}
