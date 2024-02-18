//
// Created by ruankefeng on 24-2-6.
//
#include "globalPBA.h"
extern float fx, fy, cx, cy;
void Draw(string title, const VecSE3d &poses, const VecVec3d points[], const vector<int > &host)
{
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind(title.c_str(), 1024, 768);
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


    //! wait for ESC or Tab
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
            for (Vec3d pw : points[host_id]) {
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
    pangolin::QuitAll();
}
void printResult(std::string file, const VecSE3d &poses)
{
    std::ofstream myfile;
    myfile.open (file.c_str());
    myfile << std::setprecision(15);

    bool first = true;
    for (Sophus::SE3d pose : poses) {
        const Eigen::Matrix<double,3,3> R = pose.so3().matrix();
        const Eigen::Matrix<double,3,1> T = pose.translation().transpose();
        if (!first)
            myfile << "\n";
        myfile<< R(0,0) <<" "<<R(0,1)<<" "<<R(0,2)<<" "<<T(0,0)<<" "<<
              R(1,0) <<" "<<R(1,1)<<" "<<R(1,2)<<" "<<T(1,0)<<" "<<
              R(2,0) <<" "<<R(2,1)<<" "<<R(2,2)<<" "<<T(2,0);
        first = false;
    }
    myfile.close();
}