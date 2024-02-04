//
// Created by xiang on 1/4/18.
// this program shows how to perform direct bundle adjustment
//
#include <iostream>

using namespace std;

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

using namespace Eigen;

#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

#include <pangolin/pangolin.h>
#include <boost/format.hpp>

typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> VecSE3;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVec3d;

// global variables
//string pose_file = "../poses.txt";
string pose_file = "/mnt/data/dso-4-reading/result.txt";
//string pose_file = "/home/ruankefeng/DSO/build/bin/result.txt";

//string points_file = "../points.txt";
string points_file = "/mnt/data/dso-4-reading/point_cloud.txt";

string aff_calib_file = "/mnt/data/dso-4-reading/aff_calib.txt";

/*
        ................
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

float fx = 705.918;
float fy = 703.522;
float cx = 600.927;
float cy = 182.12;

// bilinear interpolation
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
            (1 - xx) * (1 - yy) * data[0] +
            xx * (1 - yy) * data[1] +
            (1 - xx) * yy * data[img.step] +
            xx * yy * data[img.step + 1]
    );
}

// g2o vertex that use sophus::SE3d as pose
class VertexSophus : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VertexSophus(int host_id, double a, double b)
    {
        _host_id = host_id;
        _aff_a = a;
        _aff_b = b;
    }

    ~VertexSophus() {}

    int get_host_id()
    {
        return _host_id;
    }
    double get_a()
    {
        return _aff_a;
    }
    double get_b()
    {
        return _aff_b;
    }

    bool read(std::istream &is) {}

    bool write(std::ostream &os) const {}

    virtual void setToOriginImpl() {
//        _estimate = Sophus::SE3d();
        this->setEstimate(Sophus::SE3d());
    }

    // update _estimate
    virtual void oplusImpl(const double *update_) {
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> update(update_);
        this->setEstimate(Sophus::SE3d::exp(update) * this->estimate());
//        cout << "===========" << endl;
    }
private:
    int _host_id;
    double _aff_a;
    double _aff_b;
};


class VertexPoint : public g2o::BaseVertex<3, Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPoint(int host_id, double a, double b)
    {
        _host_id = host_id;
        _aff_a = a;
        _aff_b = b;
    }
    int get_host_id()
    {
        return _host_id;
    }
    double get_a()
    {
        return _aff_a;
    }
    double get_b()
    {
        return _aff_b;
    }

    virtual void setToOriginImpl() override {
        _estimate = Vector3d(0, 0, 0);
    }

    virtual void oplusImpl(const double *update) override {
        _estimate += Vector3d(update[0], update[1], update[2]);
//        cout << "....." << endl;
    }

    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}
private:
    int _host_id;
    double _aff_a;
    double _aff_b;
};
long long g_outlier = 0;
// TODO edge of projection error, implement it
// 16x1 error, which is the errors in patch
// 跟特征法主要是在这个error的计算里有区别。
// 同时，应该归属于VertexPoint的color，和
typedef Eigen::Matrix<double,16,1> Vector16d;
typedef Eigen::Matrix<double,8,1> Vector8d;
//class EdgeDirectProjection : public g2o::BaseBinaryEdge<16, Vector16d, VertexSophus, VertexPoint> {
class EdgeDirectProjection : public g2o::BaseBinaryEdge<8, Vector8d, VertexSophus, VertexPoint> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // 对color和img的关联
    EdgeDirectProjection(cv::Mat &target) {
//        this->origColor = color;
        this->targetImg = target;
    }

    ~EdgeDirectProjection() {}

    // _error = _measurement - f(v -> _estimate)
    //! computeError获取一条边的两个顶点，一个顶点是投影面，即target，一个顶点是路标点
    //! 投影面需要保存aff_g2l_t，一张修正了大小的img，位姿
    //! 路标点需要报存host_id，aff_g2l_h，
    virtual void computeError() override {
        // TODO START YOUR CODE HERE
        // compute projection error ...
        auto v0 = (VertexSophus *) _vertices[0];
        auto v1 = (VertexPoint *) _vertices[1];

        //! host_id target_id同一个也不要紧
        double a = 1.0;
        double b = 0.0;
//        double a = exp(v0->get_a() - v1->get_a());
//        double b = v1->get_b() - a * v0->get_b();

        //! v0里保存的似乎是Tcw，但是从poses读进来的应该是Twc
//        Sophus::SO3d R = (v0->estimate()).so3();
//        Vector3d t = (v0->estimate()).translation();

        Vector3d pc = (v0 -> estimate().inverse()) * (v1 -> estimate());
        pc /= pc[2];
        double u = pc[0] * fx + cx;

        double v = pc[1] * fy + cy;
//        cout << "u " << u << " v " << v <<endl;

        if (u - 2 < 0 || u + 1 >= this->targetImg.cols || v - 2 < 0 || v + 1 >= this->targetImg.rows) {
//            for (int k = 0; k < 16; k++)
            for (int k = 0; k < 8; k++)
                _error[k] = 0;
            g_outlier++;
           // return;
        }

        // 如果变为outlier点，则使用临近的边界值（即不好不坏)
        if (u - 2 < 0)
            u = 2;
        if (u + 1 >= this->targetImg.cols)
            u = this->targetImg.cols - 2;
        if (v - 2 < 0)
            v = 2;
        if (v + 1 >= this->targetImg.rows)
            v = this->targetImg.rows - 2;
        int k = 0;
        /*
        //! [-2, 1] X [-2, 1]，总共16个点
        for (int i = -2; i <= 1; i++)
            for (int j = -2; j <= 1; j++) {
                //! _measurement是老图里的色彩，根据灰度一致假设赋为该路标在新的路标坐标和相机位姿下的投影估计值
                _error[k++] = GetPixelValue(this->targetImg, u + i, v + j) - _measurement[k];
            }
        */
        for (int i = 0; i < 8; i++) {
            int dx = pattern_dx[i];
            int dy = pattern_dy[i];
//            _error[k] = GetPixelValue(this->targetImg, u + dx, v + dy) - (a * _measurement[k] + b);
            _error[k] = GetPixelValue(this->targetImg, u + dx, v + dy) - _measurement[k];
            k++;
        }

        // TODO END YOUR CODE HERE
    }

        // Let g2o compute jacobian for you
//    G2O_MAKE_AUTO_AD_FUNCTIONS;

        virtual bool read(istream &in) {}

        virtual bool write(ostream &out) const {}

        private:
        cv::Mat targetImg;  // the target image
//        float *origColor = nullptr;   // 16 floats, the color of this point
        bool outlier = false;
};

// plot the poses and points for you, need pangolin
//void Draw(const VecSE3 &poses, const VecVec3d &points);
void Draw(string title, const VecSE3 &poses, const VecVec3d points[], const vector<int > &host);

int main(int argc, char **argv) {

    // read poses and points
    VecSE3 poses;
    VecVec3d points[100 * 100];
    vector<Vector8d > color[100 * 100];
    ifstream fin(pose_file);

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
        Eigen::Matrix<double, 3, 3> R;
        R(0, 0) = data[0];
        R(0, 1) = data[1];
        R(0, 2) = data[2];
        R(1, 0) = data[4];
        R(1, 1) = data[5];
        R(1, 2) = data[6];
        R(2, 0) = data[8];
        R(2, 1) = data[9];
        R(2, 2) = data[10];
        Eigen::Vector<double, 3> t;
        t[0] = data[3];
        t[1] = data[7];
        t[2] = data[11];
//        std::cout << t.transpose() << std::endl;
        poses.push_back(Sophus::SE3d(Sophus::SO3d(R), t));
#endif
        if (!fin.good())
            break;
    }
    fin.close();
    poses.pop_back();

//    Draw(poses, points);
//    return 0;

//    vector<double *> color;  // 由color组成的数组。所谓的color其实周边的16个像素值
//    vector<Vector16d > color;
    fin.open(points_file);
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
            Vector8d c;
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
        //! TODO 输入affLL.a affLL.b
//        for (int i = 0; i < poses.size(); i++) {
        double a, b;
        fin >> a;
        fin >> b;
        std::cout << a << " " << b << std::endl;
        affLLa.push_back(a);
        affLLb.push_back(b);
//        }
        //! END
        if (fin.good() == false)
            break;
    }
    assert(affLLa.size() == poses.size());
    assert(affLLb.size() == poses.size());
    fin.close();
    std::cout << "aff_calib read " << affLLa.size() << std::endl;

    /*
    while (!fin.eof()) {
        double xyz[3] = {0};
        for (int i = 0; i < 3; i++)
            fin >> xyz[i];
        if (xyz[0] == 0)
            break;
        points.push_back(Eigen::Vector3d(xyz[0], xyz[1], xyz[2]));
//        double *c = new double[16];
#if 0
        Vector16d c;
        for (int i = 0; i < 16; i++)
            fin >> c[i];
        color.push_back(c); // color数组元素跟points数组元素一一对应。
#endif
        Vector8d c;
        for (int i = 0; i < 8; i++)
            fin >> c[i];
        color.push_back(c); // color数组元素跟points数组元素一一对应。

        if (fin.good() == false)
            break;
    }
     */
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
//    points_num = 0;
//    for (int i = 0; i < poses.size(); i++)
//        points_num += points[i].size();
//    cout << "points_num: " << points_num << endl;


    // read images
    vector<cv::Mat> images;
//    boost::format fmt("../%d.png");
    boost::format fmt("/mnt/data/kitti_dataset/sequences/06/image_0/%06d.png");
    for (int i = 0; i < poses.size(); i++) {
        cv::Mat orig_image = cv::imread((fmt % (i + 0)).str(), 0);
//        cout << i << "orig image rows: " << orig_image.rows << " orig image cols: " << orig_image.cols << endl;
        cv::Mat image;
        cv::resize(orig_image, image, cv::Size(1224, 368));
        images.push_back(image);
//        cout << i << "image rows: " << images[i].rows << " image cols: " << images[i].cols << endl;
    }

//    Draw("before", poses, points, host);

#if 1

    // build optimization problem
//    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> DirectBlock;  // 求解的向量是6＊1的
//    typedef g2o::LinearSolverDense<DirectBlock::PoseMatrixType> LinearSolverType; // 线性求解器类型
//    DirectBlock::LinearSolverType *linearSolver = new g2o::LinearSolverDense<DirectBlock::PoseMatrixType>();
//    DirectBlock *solver_ptr = new DirectBlock(linearSolver);
//    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr); // L-M
//    g2o::SparseOptimizer optimizer;
//    optimizer.setAlgorithm(solver);
//    optimizer.setVerbose(true);

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
//    for (int host_id : host) {
//        for (Vector3d pw : points[host_id]) {
//            glColor3f(0, (1.0 * host_id) / poses.size(),  1.0 - (1.0 * host_id) / poses.size());
//            glVertex3d(pw[0], pw[1], pw[2]);
//        }
//    }
    vector<VertexPoint *> vertices_points[100 * 100]; // 最多100 * 100个host
//    std::cout << "4.5....." << std::endl;
    int a = 1;
    for (int host_id : host) {
//        std::cout << "4.6....." << host_id << std::endl;
//        std::cout << "4.6..a..." << affLLa.size() << std::endl;
//        std::cout << "4.6..b..." << affLLb.size() << std::endl;
//        std::cout << affLLa[host_id] << "......." << affLLb[host_id] << std::endl;
        for (Vector3d p : points[host_id]) {
            VertexPoint * v_p = new VertexPoint(host_id, affLLa[host_id], affLLb[host_id]);
            v_p->setId(a++);
            v_p->setEstimate(p);
            v_p->setMarginalized(true);
            optimizer.addVertex(v_p);
            vertices_points[host_id].push_back(v_p);
        }
    }
//    return 0;
//    assert(a == points_num);
//    vector<VertexPoint *> vertices_points;
//    for (int i = 0; i < points.size(); i++) {
//        VertexPoint * v_p = new VertexPoint();
//        v_p->setId(i); //id [0, p.size()]
//        //! points[i]为三维向量
//        v_p->setEstimate(points[i]);
//        v_p->setMarginalized(true);
//        optimizer.addVertex(v_p);
//        vertices_points.push_back(v_p);
//    }

    std::cout << "5....." << std::endl;

    vector<VertexSophus *> vertices_sophus;
    for (int i = 0; i < poses.size(); i++) {
        VertexSophus *v_s = new VertexSophus(i, affLLa[i], affLLb[i]);
//        v_s->setId(i + points.size());
//        v_s->setId(i + points_num);
        v_s->setId(a++);
//        assert(v_s->getId() == i + points_num + 10);
      //! 这里应该错了，应该是Tcw而不是Twc
        v_s->setEstimate(poses[i]);
        optimizer.addVertex(v_s);
        vertices_sophus.push_back(v_s);
    }
    std::cout << "6....." << std::endl;
    // 每条边赋值一个color作为观测值，一个观测一条边，总共poses * points条边.
    //! 这里是稠密图了
//    int obs = 0;
    for (int l = 0; l < host.size(); l++) {
        int host_id = host[l];
        int marg_id = marginalizedAt[l];
//            poses[host_id]....poses[marg_id]
//            points[host_id]
//        printf("from [%d] to [%d], projected [%ld] points\n", host_id, marg_id, points[host_id].size());
        for (int i = host_id; i < marg_id; i++) {
            for (int j = 0; j < vertices_points[host_id].size(); j++) {
                //! 按j计数point，i为j对应的s的id和marginalizeAt（也可以
                EdgeDirectProjection *edge = new EdgeDirectProjection(images[i]);

                edge->setVertex(0, vertices_sophus[i]); //! 投影的面从host_id....marg_id
                edge->setVertex(1, (vertices_points[host_id])[j]); //! 投影的点为所有的vp[host_id]
                edge->setMeasurement(Vector8d((color[host_id])[j]));
                edge->setInformation(Eigen::Matrix<double, 8, 8>::Identity());
                edge->setRobustKernel(new g2o::RobustKernelHuber());
                optimizer.addEdge(edge);
            }
           // break;
        }
    }
//    for (int i = 0; i < poses.size(); i++) {
//        for (int j = 0; j < points.size(); j++) {
//            //! 按j计数point，i为j对应的s的id和marginalizeAt（也可以
//            EdgeDirectProjection *edge = new EdgeDirectProjection(images[i]);
//            edge->setVertex(0, vertices_sophus[i]); //传入的是指针
//            edge->setVertex(1, vertices_points[j]);
//            edge->setMeasurement(Vector8d(color[j]));
//            edge->setInformation(Eigen::Matrix<double, 8, 8>::Identity());
//            edge->setRobustKernel(new g2o::RobustKernelHuber());
//            optimizer.addEdge(edge);
//        }
//    }
    // END YOUR CODE HERE

    std::cout << "7....." << std::endl;
    // perform optimization
    optimizer.initializeOptimization(0);
    optimizer.optimize(30);
//    optimizer.optimize(0);

    // TODO fetch data from the optimizer
    // START YOUR CODE HERE
    for (int i = 0; i < poses.size(); i++)
        poses[i] = vertices_sophus[i]->estimate();
    for (int host_id : host) {
        for (int j = 0; j < points[host_id].size(); j++)
            points[host_id][j] = vertices_points[host_id][j]->estimate();
    }
    // END YOUR CODE HERE

    cout << "outlier: " << g_outlier << endl;
    // plot the optimized points and poses
#endif
    Draw(string("after"), poses, points, host);

    // delete color data
//    for (auto &c: color) delete[] c;
    return 0;
}

void Draw(string title, const VecSE3 &poses, const VecVec3d points[], const vector<int > &host) {
//    if (poses.empty() || points.empty()) {
//        cerr << "parameter is empty!" << endl;
//        return;
//    }

    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind(title, 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));


    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

        // draw poses
        float sz = 0.5;
        int width = 1224, height = 368;
//        for (auto &Tcw: poses) {
        for (auto &Twc: poses) {
            glPushMatrix();

            //! 这样就cam to world了
//            Sophus::Matrix4f m = Tcw.inverse().matrix().cast<float>();
            Sophus::Matrix4f m = Twc.matrix().cast<float>();
            glMultMatrixf((GLfloat *) m.data());

            //! 下面的是在cam坐标系下画的相机，然后会用gl的Matrix乘上去转换为世界坐标系
            glColor3f(1, 1, 0);
            glLineWidth(2);
            glBegin(GL_LINES);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
            glEnd();

            glPopMatrix();
        }

        // points
        glPointSize(2);
        glBegin(GL_POINTS);
        for (int host_id : host) {
            for (Vector3d pw : points[host_id]) {
                //! 第i个点
//                std::cout << host_id / poses.size() << std::endl;
//                glColor3f(0, pw[2] / 4, 1.0 - pw[2] / 4);
                glColor3f(0, (1.0 * host_id) / poses.size(),  1.0 - (1.0 * host_id) / poses.size());
//                if (host_id % 2 == 0)
//                    glColor3f(0, 1.0, 0);
//                else
//                    glColor3f(0, 0, 1.0);
                glVertex3d(pw[0], pw[1], pw[2]);
            }
        }
        /*
        for (size_t i = 0; i < points.size(); i++) {
            //! 第i个点
            glColor3f(0.0, points[i][2] / 4, 1.0 - points[i][2] / 4);
            glVertex3d(points[i][0], points[i][1], points[i][2]);
        }
        */
        glEnd();

        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
}

