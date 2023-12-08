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
string pose_file = "../poses.txt";
string points_file = "../points.txt";

// intrinsics
float fx = 277.34;
float fy = 291.402;
float cx = 312.234;
float cy = 239.777;

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

    VertexSophus() {}

    ~VertexSophus() {}


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
        cout << "===========" << endl;
    }
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
//        cout << "....." << endl;
    }

    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}

};
long long g_outlier = 0;
// TODO edge of projection error, implement it
// 16x1 error, which is the errors in patch
// 跟特征法主要是在这个error的计算里有区别。
// 同时，应该归属于VertexPoint的color，和
typedef Eigen::Matrix<double,16,1> Vector16d;
class EdgeDirectProjection : public g2o::BaseBinaryEdge<16, Vector16d, VertexSophus, VertexPoint> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // 对color和img的关联
    EdgeDirectProjection(cv::Mat &target) {
//        this->origColor = color;
        this->targetImg = target;
    }

    ~EdgeDirectProjection() {}

    // _error = _measurement - f(v -> _estimate)
    virtual void computeError() override {
        // TODO START YOUR CODE HERE
        // compute projection error ...
        auto v0 = (VertexSophus *) _vertices[0];
        auto v1 = (VertexPoint *) _vertices[1];

        Vector3d pc = (v0 -> estimate()) * (v1 -> estimate());
        pc /= pc[2];
        double u = pc[0] * fx + cx;
        double v = pc[1] * fy + cy;
//        cout << "u " << u << " v " << v <<endl;
        // 如果变为outlier点，则使用临近的边界值（即不好不坏)
        if (u - 2 < 0)
            u = 2;
        if (u + 1 >= this->targetImg.cols)
            u = this->targetImg.cols - 2;
        if (v - 2 < 0)
            v = 2;
        if (v + 1 >= this->targetImg.rows)
            v = this->targetImg.rows - 2;
//        cout << "u " << u << " v " << v <<endl;
//        if (this->outlier == true) {
//            this->outlier = false;
//            g_outlier--;
//        }
//        if (u - 2 < 0 || u + 1 >= this->targetImg.cols || v - 2 < 0 || v + 1 >= this->targetImg.rows) {
//            _error = Vector16d::Zero();
//            g_outlier++;
//            this->outlier = true;
//        } else {
        int k = 0;
        for (int i = -2; i <= 1; i++)
            for (int j = -2; j <= 1; j++) {
                _error[k++] = GetPixelValue(this->targetImg, u + i, v + j) - _measurement[k];
//                cout << "_error " << _error << endl;
            }
//        cout << _error << endl;
//        }
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
void Draw(const VecSE3 &poses, const VecVec3d &points);

int main(int argc, char **argv) {

    // read poses and points
    VecSE3 poses;
    VecVec3d points;
    ifstream fin(pose_file);

    while (!fin.eof()) {
        double timestamp = 0;
        fin >> timestamp;
        if (timestamp == 0)
            break;
        double data[7];
        for (auto &d: data)
            fin >> d;
        poses.push_back(Sophus::SE3d(
                Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
                Eigen::Vector3d(data[0], data[1], data[2])
        ));
        if (!fin.good())
            break;
    }
    fin.close();

//    vector<double *> color;  // 由color组成的数组。所谓的color其实周边的16个像素值
    vector<Vector16d > color;
    fin.open(points_file);
    while (!fin.eof()) {
        double xyz[3] = {0};
        for (int i = 0; i < 3; i++)
            fin >> xyz[i];
        if (xyz[0] == 0)
            break;
        points.push_back(Eigen::Vector3d(xyz[0], xyz[1], xyz[2]));
//        double *c = new double[16];
        Vector16d c;
        for (int i = 0; i < 16; i++)
            fin >> c[i];
        color.push_back(c); // color数组元素跟points数组元素一一对应。

        if (fin.good() == false)
            break;
    }
    fin.close();

    cout << "poses: " << poses.size() << ", points: " << points.size() << endl;
    cout << "observations(edges): " << poses.size() * points.size() << endl;

    // read images
    vector<cv::Mat> images;
    boost::format fmt("../%d.png");
    for (int i = 0; i < 6; i++) {
        images.push_back(cv::imread((fmt % (i + 1)).str(), 0));
        cout << "imggg " << images[i].rows << endl;
    }

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
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;  // pose is 6, landmark is 3
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型
    // 梯度下降方法，可以从GN, LM, DogLeg 中选
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;     // 图模型
    optimizer.setAlgorithm(solver);   // 设置求解器
    optimizer.setVerbose(true);       // 打开调试输出

    // TODO add vertices, edges into the graph optimizer
    // START YOUR CODE HERE
    // 一个路标一个顶点，一个位姿也是一个顶点。
    vector<VertexPoint *> vertices_points;
    for (int i = 0; i < points.size(); i++) {
        VertexPoint * v_p = new VertexPoint();
        v_p->setId(i);
        v_p->setEstimate(points[i]);
        v_p->setMarginalized(true);
        optimizer.addVertex(v_p);
        vertices_points.push_back(v_p);
    }
    vector<VertexSophus *> vertices_sophus;
    for (int i = 0; i < poses.size(); i++) {
        VertexSophus *v_s = new VertexSophus();
        v_s->setId(i + points.size());
        v_s->setEstimate(poses[i]);
        optimizer.addVertex(v_s);
        vertices_sophus.push_back(v_s);
    }
    // 每条边赋值一个color作为观测值，一个观测一条边，总共poses * points条边.
    for (int i = 0; i < poses.size(); i++) {
        for (int j = 0; j < points.size(); j++) {
//    for (int i = 0; i < 10; i++) {
//        for (int j = 0; j < 100; j++) {
            EdgeDirectProjection *edge = new EdgeDirectProjection(images[i]);
            edge->setVertex(0, vertices_sophus[i]); //传入的是指针
            edge->setVertex(1, vertices_points[j]);
            edge->setMeasurement(Vector16d(color[j]));
            edge->setInformation(Eigen::Matrix<double, 16, 16>::Identity());
            edge->setRobustKernel(new g2o::RobustKernelHuber());
            optimizer.addEdge(edge);
        }
    }
    // END YOUR CODE HERE

    // perform optimization
    optimizer.initializeOptimization(0);
    optimizer.optimize(200);

    // TODO fetch data from the optimizer
    // START YOUR CODE HERE
    for (int i = 0; i < poses.size(); i++)
        poses[i] = vertices_sophus[i]->estimate();
    for (int j = 0; j < points.size(); j++)
        points[j] = vertices_points[j]->estimate();
    // END YOUR CODE HERE

    cout << "outlier: " << g_outlier << endl;
    // plot the optimized points and poses
    Draw(poses, points);

    // delete color data
//    for (auto &c: color) delete[] c;
    return 0;
}

void Draw(const VecSE3 &poses, const VecVec3d &points) {
    if (poses.empty() || points.empty()) {
        cerr << "parameter is empty!" << endl;
        return;
    }

    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
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
        float sz = 0.1;
        int width = 640, height = 480;
        for (auto &Tcw: poses) {
            glPushMatrix();
            Sophus::Matrix4f m = Tcw.inverse().matrix().cast<float>();
            glMultMatrixf((GLfloat *) m.data());
            glColor3f(1, 0, 0);
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
        for (size_t i = 0; i < points.size(); i++) {
            glColor3f(0.0, points[i][2] / 4, 1.0 - points[i][2] / 4);
            glVertex3d(points[i][0], points[i][1], points[i][2]);
        }
        glEnd();

        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
}

