//
// Created by ruankefeng on 24-2-6.
//
#include "globalPBA.h"
extern float fx, fy, cx, cy;
extern int first_cam, last_cam;
void Draw(string title, const VecSE3d &cams, const VecVec3d points[], const vector<int > &host)
{
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind(title.c_str(), 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

//    const int UI_WIDTH = 180;
        // parameter reconfigure gui
//    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));
//    pangolin::Var<int> settings_cams_first("ui.first_camera", 0, 0, 1000, false);
//    pangolin::Var<int> settings_cams_last("ui.last_camera", 500, 1, 1000, false);

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

//        first_cam = settings_cams_first.Get();
//        last_cam  = settings_cams_last.Get();

        // draw poses
        float sz = 0.5;
        int width = 1224, height = 368;
        for (auto &Twc: cams) {
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
                glColor3f(0, (1.0 * host_id) / cams.size(),  1.0 - (1.0 * host_id) / cams.size());
//                if (host_id % 2 == 0)
//                    glColor3f(0, 1.0, 0);
//                else
//                    glColor3f(0, 0, 1.0);
                glVertex3d(pw[0], pw[1], pw[2]);
            }
        }
        glEnd();

        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
    pangolin::QuitAll();
}
void printResult(std::string file, const VecSE3d &cams)
{
    std::ofstream myfile;
    myfile.open (file.c_str());
    myfile << std::setprecision(15);

    bool first = true;
    for (Sophus::SE3d cam : cams) {
        const Eigen::Matrix<double,3,3> R = cam.so3().matrix();
        const Eigen::Matrix<double,3,1> T = cam.translation().transpose();
        if (!first)
            myfile << "\n";
        myfile<< R(0,0) <<" "<<R(0,1)<<" "<<R(0,2)<<" "<<T(0,0)<<" "<<
              R(1,0) <<" "<<R(1,1)<<" "<<R(1,2)<<" "<<T(1,0)<<" "<<
              R(2,0) <<" "<<R(2,1)<<" "<<R(2,2)<<" "<<T(2,0);
        first = false;
    }
    myfile.close();
}

void run()
{
    double w = 400;
    double h = 800;
    printf("START PANGOLIN!\n");

    pangolin::CreateWindowAndBind("Main",2*w,2*h);
    const int UI_WIDTH = 180;

    glEnable(GL_DEPTH_TEST);

    // parameter reconfigure gui
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));
    pangolin::Var<int> settings_cams_first("ui.first_camera", 0, 0, 1000, false);
    pangolin::Var<int> settings_cams_last("ui.last_camera", 500, 1, 1000, false);

    {
        // Default hooks for exiting (Esc) and fullscreen (tab).
        while (!pangolin::ShouldQuit()) {
            // Clear entire screen
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            first_cam = settings_cams_first.Get();
            last_cam  = settings_cams_last.Get();
            // update parameters
//            int settings_pointCloudMode_i = settings_pointCloudMode.Get();
//        bool settings_showActiveConstraints_b = settings_showActiveConstraints.Get();
//            double settings_absVarTH_d = settings_absVarTH.Get();
            // Swap frames and Process Events
            pangolin::FinishFrame();
//        if(needReset) reset_internal();
        }
    }
    printf("QUIT Pangolin thread!\n");
//    printf("I'll just kill the whole process.\nSo Long, and Thanks for All the Fish!\n");
//    exit(1);
}