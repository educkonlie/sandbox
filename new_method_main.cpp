#include "new_method_BA.h"

//! 关于术语使用
//! 输入输出使用 camera point  表明是纯数值
//! 内部优化包括g2o优化使用 pose landmark，这是两个vertex

// global variables
//std::string camera_file = "/mnt/data/dso-4-reading/build/bin/result.txt";
std::string camera_file = "/mnt/data/dso-4-reading/groundTruthPose/kitti/00.txt";
std::string point_file  = "/mnt/data/dso-4-reading/groundTruthPose/kitti/00_pc.txt";

float fx = 705.919;
float fy = 703.522;
float cx = 600.928;
float cy = 182.121;

int first_cam = 0;
int last_cam  = 0;
bool global_start = false;
// plot the cams and points for you, need pangolin
void Draw(std::string title, const VecSE3d &cams, const VecVec3d points[], const vector<int > &host, std::mutex &m);
void printResult(std::string file, const VecSE3d &cams);
void input(std::string &camera_file, std::string &point_file,
           VecSE3d &cams, VecVec3d points[],
           std::vector<Vec8d> color[],
           std::vector<int> &host,
           std::vector<int> &marginalizedAt);

//! 先基于g2o的改造成自己的Pose和Landmark，并且测试一下计算residual和energy和jacobi是否正确
int main(int argc, char **argv)
{
    // read poses and points
    VecSE3d cams;
    VecVec3d points[100 * 100];
    std::vector<Vec8d > color[100 * 100];
    vector<int > host;
    vector<int > marginalizedAt;

    // 寻找camera_file, point_file，如果不存在，则等待生成
    std::mutex m;

    input(camera_file, point_file, cams, points, color, host, marginalizedAt);
    std::thread runthread([&]() {
        Draw(string("before"), cams, points, host, m);
    });
    while (1) {
        m.lock();
        if (global_start) {
            m.unlock();
            break;
        }
        m.unlock();
    }

#if 1
    myOptimizer *my_optimizer = new myOptimizer();
    std::cout << "4....." << std::endl;

    // START YOUR CODE HERE
    //! 按host循环
    vector<myLandmark *> mylandmarks[100 * 100];
    for (int host_id : host) {
        for (Vec3d p : points[host_id]) {
            myLandmark *mylandmark = new myLandmark();
            mylandmark->setEstimate(p);
            my_optimizer->addLandmark(mylandmark);
            mylandmarks[host_id].push_back(mylandmark);
        }
    }
    std::cout << "5....." << std::endl;

    vector<myPose *> myposes;
    for (int i = 0; i < cams.size(); i++) {
        myPose *mypose = new myPose();
        mypose->setPoseId(i);
        /// 要注意是相机到世界还是世界到相机
        mypose->setEstimate(cams[i].inverse());
        my_optimizer->addPose(mypose);
        myposes.push_back(mypose);
    }
    std::cout << "6....." << std::endl;

    for (int l = 0; l < host.size(); l++) {
        int host_id = host[l];
        int marg_id = marginalizedAt[l];
//            points[host_id]
//            poses[host_id]....poses[marg_id]
        for (myLandmark *myland : mylandmarks[host_id]) {
            for (int i = host_id; i < marg_id; i++) {
                Vec2d my_measure = myposes[i]->GetPixelValue(myland->estimate());
                if (my_measure[0] < 10)
                    continue;
                if (my_measure[1] < 10)
                    continue;
                if (my_measure[0] >= 1224 - 10)
                    continue;
                if (my_measure[1] >= 368 - 10)
                    continue;
                //! 按j计数point，i为j对应的s的id和marginalizeAt（也可以
                myEdge *myedge = new myEdge();
                myedge->setPose(myposes[i]); //! 投影的面从host_id....marg_id
                myedge->setLandmark(myland); //! 投影的点为所有的vp[host_id]
                myedge->setMeasurement(my_measure);

                myland->addEdge(myedge);
//                myedge->setRobustKernel(new g2o::RobustKernelHuber());
//                break;
            }
        }
    }

    /// 接下来还有一个很重要的步骤，就是将从DSO取得的poses, points加上噪声，制作poses_noisy, points_noisy
    /// 作为优化器的输入。优化在功能上也就是滤波降噪。
    VecSE3d cams_noisy;
    VecVec3d points_noisy[100 * 100];
    m.lock();
    for (Sophus::SE3d cam : cams)
        cams_noisy.push_back(AddNoiseinPose(cam));
    for (int host_id : host) {
        for (Vec3d &pt: points[host_id])
            points_noisy[host_id].push_back(AddNoiseinPoint(pt));
    }


    for (int i = 0; i < cams_noisy.size(); i++) {
        myposes[i]->setEstimate(cams_noisy[i].inverse());
        cams[i] = cams_noisy[i];
    }

    for (int host_id : host) {
        for (int i = 0; i < points_noisy[host_id].size(); i++) {
            mylandmarks[host_id][i]->setEstimate(points_noisy[host_id][i]);
            points[host_id][i] = points_noisy[host_id][i];
        }

    }
    m.unlock();

    std::cout << "7....." << std::endl;
    // perform optimization
//    for (int k = 0; k < 10; k++) {
        my_optimizer->optimize(5);

        m.lock();
        for (int i = 0; i < cams.size(); i++)
            cams[i] = myposes[i]->estimate().inverse();
        for (int host_id: host) {
            for (int j = 0; j < points[host_id].size(); j++)
                points[host_id][j] = mylandmarks[host_id][j]->estimate();
        }
        m.unlock();
//    }
#endif
    runthread.join();
    printResult(string("/mnt/data/dso-4-reading/gpba_result.txt"), cams);
    return 0;
}
void input(std::string &camera_file, std::string &point_file,
           VecSE3d &cams, VecVec3d points[],
           std::vector<Vec8d> color[],
           vector<int> &host,
           vector<int> &marginalizedAt) {
    std::ifstream fin(camera_file);
    while (!fin.eof()) {
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
        if (!fin.good())
            break;
    }
    fin.close();

    fin.open(point_file);
    int num_of_host = 0;


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
}
