#pragma once
#include "common.h"
#include "pcg.h"
//! 要将三列数据全部QR化，需要第一列，第二列，第三列，依次进行
//  1 2 3
//  2 4 6
//  4 5 6
//  7 8 9
// =>
//  a x x
//  0 b x
//  0 0 c
//  0 0 0
// Jp 和 Jr合成Jp

void qr(MatXX &Jp, MatXX &Jl)
{
    MatXX temp1, temp2;
    int nres = Jl.rows();
    int cols = Jl.cols();
    assert(nres > 3);
    // i: row
    // j: col
    for (int j = 0; j < cols; j++) {
        rkf_scalar pivot = Jl(j, j);
        for (int i = j + 1; i < nres; i++) {
            if (std::abs(Jl(i, j)) < 1e-10)
                continue;
            rkf_scalar a = Jl(i, j);
            rkf_scalar r = sqrt(pivot * pivot + a * a);
            rkf_scalar c = pivot / r;
            rkf_scalar s = a / r;
            pivot = r;

// 变0的，先到temp
            temp1 = -s * Jl.row(j) + c * Jl.row(i);
            temp2 = -s * Jp.row(j) + c * Jp.row(i);
// 变大的.  j是pivot，在上面，i在下面
            Jl.row(j) = c * Jl.row(j) + s * Jl.row(i);
            Jp.row(j) = c * Jp.row(j) + s * Jp.row(i);
// 变0的, temp => i
            Jl.row(i) = temp1;
            Jp.row(i) = temp2;

            Jl(j, j) = pivot = r;
            Jl(i, j) = 0;
        }
    }
}
void qr2(MatXX &Jl)
{
    MatXX temp1, temp2;
    int nres = Jl.rows();
    int cols = Jl.cols();
    assert(nres > 3);
    // i: row
    // j: col
    for (int j = 0; j < cols; j++) {
        rkf_scalar pivot = Jl(j, j);
        for (int i = j + 1; i < nres; i++) {
            if (std::abs(Jl(i, j)) < 1e-10)
                continue;
            rkf_scalar a = Jl(i, j);
            rkf_scalar r = sqrt(pivot * pivot + a * a);
            rkf_scalar c = pivot / r;
            rkf_scalar s = a / r;
            pivot = r;

// 变0的，先到temp
            temp1 = -s * Jl.row(j) + c * Jl.row(i);
// 变大的.  j是pivot，在上面，i在下面
            Jl.row(j) = c * Jl.row(j) + s * Jl.row(i);
// 变0的, temp => i
            Jl.row(i) = temp1;

            Jl(j, j) = pivot = r;
            Jl(i, j) = 0;
        }
    }
}
void qr3(MatXX &Jp, MatXX &Jl, VecX &Jr) {
    MatXX temp1, temp2;
    VecX temp3;
    int nres = Jl.rows();
    int cols = Jl.cols();
    assert(nres > 3);
    // i: row
    // j: col
    for (int j = 0; j < cols; j++) {
        rkf_scalar pivot = Jl(j, j);
        for (int i = j + 1; i < nres; i++) {
            if (std::abs(Jl(i, j)) < 1e-10)
                continue;
            rkf_scalar a = Jl(i, j);
            rkf_scalar r = sqrt(pivot * pivot + a * a);
            rkf_scalar c = pivot / r;
            rkf_scalar s = a / r;
            pivot = r;
            assert(std::isfinite(r));
            assert(std::abs(r) > 1e-10);
// 变0的，先到temp
            temp1 = -s * Jp.row(j) + c * Jp.row(i);
            temp2 = -s * Jl.row(j) + c * Jl.row(i);
            temp3 = -s * Jr.row(j) + c * Jr.row(i);
// 变大的.  j是pivot，在上面，i在下面
            Jp.row(j) = c * Jp.row(j) + s * Jp.row(i);
            Jl.row(j) = c * Jl.row(j) + s * Jl.row(i);
            Jr.row(j) = c * Jr.row(j) + s * Jr.row(i);
// 变0的, temp => i
            Jp.row(i) = temp1;
            Jl.row(i) = temp2;
            Jr.row(i) = temp3;

            Jl(j, j) = pivot = r;
            Jl(i, j) = 0;
        }
    }
}

#define XI 8
// struct of JM rM
//  JM_marg   JM_remained   JM_CPARS  rM
// 然后将上面整体进行QR分解，rM会比JM多一行非零，但这个是正常现象
// 最后将JM_marg对应的行列删除，将底下为0的行删除
// 如果不做marg，那么仍旧可以通过求解(J.transpose() * JM).ldlt().solve(J.transpose() * rM)
// 来比较，如果解不变，则化简有效
void marg_frame(MatXX &J, VecX &r, MatXX &J_new, VecX &r_new, int nframes, int idx)
{
    MatXX Jr = MatXX::Zero(J.rows(), J.cols() + 1);
    //! 组合Jr，将CPARS从最左边移为J_pose J_cpars r的结构
    Jr.leftCols(nframes * 8) = J.middleCols(CPARS, J.cols() - CPARS);
    Jr.middleCols(nframes * 8, CPARS) = J.leftCols(CPARS);
    Jr.rightCols(1) = r;

//! 将需要marg的帧移到第0帧
    if (idx != 0) {
        MatXX temp1 = Jr.leftCols(idx * 8);
        MatXX temp2 = Jr.middleCols(idx * 8, 8);
        Jr.leftCols(8) = temp2;
        Jr.middleCols(8, idx * 8) = temp1;
    }

    //! qr分解，以及化简，即删除多余的零行
    qr2(Jr);
    MatXX Jr_temp = Jr.topRows(Jr.cols());

    //! 去掉前8列，和上8行 (marg)
    Jr = Jr_temp.bottomRightCorner(Jr_temp.rows() - 8, Jr_temp.cols() - 8);

    //! 将化简后的Jr分为J, r
    J = Jr.leftCols(CPARS + nframes * 8 - 8);
    r = Jr.rightCols(1);

    //! 输出为J_new, r_new
    J_new = MatXX::Zero(J.rows(), J.cols());
    r_new = r;
    //! 再把CPARS换回头部
    J_new.leftCols(CPARS) = J.middleCols(nframes * 8 - 8, CPARS);
    J_new.middleCols(CPARS, nframes * 8 - 8) = J.leftCols(nframes * 8 - 8);
}
void marg_frame2(MatXX &J, VecX &r, int idx)
{
    MatXX J_new = MatXX::Zero(J.rows(), J.cols() - 8);
    MatXX Jl = MatXX::Zero(J.rows(), 8);

//! 将需要marg的帧移到Jl
//    if (idx != 0) {
        J_new.leftCols(CPARS + idx * 8) = J.leftCols(CPARS + idx * 8);
        Jl = J.middleCols(CPARS + idx * 8, 8);
        J_new.rightCols(J_new.cols() - CPARS - idx * 8)
                = J.rightCols(J.cols() - CPARS - idx * 8 - 8);
//    }

    //! qr分解
//    qr3(J_new, Jl, r);
    qr3(J_new, Jl, r);

    //! 去掉上8行 (marg)
    J = J_new.bottomRows(J_new.rows() - 8);
    VecX r_new = r;
    r = r_new.bottomRows(r_new.rows() - 8);
}
void compress_Jr(MatXX &J, VecX &r)
{
    MatXX Jr = MatXX::Zero(J.rows(), J.cols() + 1);
    //! 组Jr
    Jr.leftCols(J.cols()) = J;
    Jr.rightCols(1) = r;

    //! qr分解，以及化简，即删除多余的零行
    qr2(Jr);
    Jr.conservativeResize(Jr.cols(), Jr.cols());
//    MatXX temp = Jr.topRows(Jr.cols());
//    Jr = temp;

    //! 将化简后的Jr分为J, r
    J = Jr.leftCols(Jr.cols() - 1);
    r = Jr.rightCols(1);
}
//!
void no_marg_frame(MatXX &J, VecX &r, MatXX &J_new, VecX &r_new, int nframes)
{
    MatXX Jr = MatXX::Zero(J.rows(), J.cols() + 1);
    //! 组Jr，把CPARS放到后面
//    Jr.leftCols(nframes * 8) = J.middleCols(CPARS, J.cols() - CPARS);
//    Jr.middleCols(nframes * 8, CPARS) = J.leftCols(CPARS);
//    Jr.rightCols(1) = r;
    Jr.leftCols(CPARS + nframes * 8) = J;
    Jr.rightCols(1) = r;

    //! qr分解，以及化简，即删除多余的零行
    qr2(Jr);
    Jr.conservativeResize(Jr.cols(), Jr.cols());
//    MatXX temp = Jr.topRows(Jr.cols());
//    Jr = temp;

    //! 将化简后的Jr分为J, r
    J = Jr.leftCols(CPARS + nframes * 8);
    r = Jr.rightCols(1);

    //! 输出为J_new, r_new
//    J_new = MatXX::Zero(J.rows(), J.cols());
//    r_new = r;
//    J_new = J;
//    r_new = r;
    //! 把CPARS换回头部
//    J_new.leftCols(CPARS) = J.middleCols(nframes * 8, CPARS);
//    J_new.middleCols(CPARS, nframes * 8) = J.leftCols(nframes * 8);
}
void add_lambda_frame(MatXX &J, VecX &r, int idx, Vec8 Lambda, Vec8 alpha)
{
    int old_rows = J.rows();
    //! 扩展底下8行
    J.conservativeResize(J.rows() + 8, J.cols());
    r.conservativeResize(r.rows() + 8);
//    J.bottomRows(8) = MatXX::Zero(8, J.cols());
    J.bottomRows(8).setZero();
    r.bottomRows(8).setZero();
    //! 底下idx对应的8行8列设置为Lambda，r底下设置为 Lambda * alpha
    J.block(old_rows, CPARS + idx * 8, 8, 8) = Lambda.asDiagonal();
    r.bottomRows(8) = Lambda.asDiagonal() * alpha;
}
void test_marg_frame() {
    int num_of_frames = 25;
    int idx = 2;
    MatXX J = MatXX::Random(2000, CPARS + num_of_frames * 8);
    VecX r = VecX::Random(2000);
    Vec8 Lambda = Vec8::Random(8);
    Vec8 alpha  = Vec8::Random(8);
//    add_lambda_frame(J, r, 2, Lambda, alpha);

//    my *my1 = new my();

    Matrix<double, Dynamic, Dynamic> J_new;
    VectorXd r_new;
    J_new = J.cast<double>();
    r_new = r.cast<double>();
//    VecX x;
//    my1->pcg(J, r, x, 1e-8, 10);
    std::cout << "x double    : "
            << (J_new.transpose() * J_new).ldlt().solve(J_new.transpose() * r_new).transpose()
            << std::endl;
//    std::cout << "x scalar    : "
//              << (J.transpose() * J).ldlt().solve(J.transpose() * r).transpose()
//              << std::endl;

//    std::cout << "x    : " << x.transpose() << std::endl;
//    marg_frame(J, r, J_new, r_new, num_of_frames, 2);
//    no_marg_frame(J, r, J_new, r_new, num_of_frames);

//    compress_Jr(J, r);
    marg_frame2(J, r, 2);

    J_new = J.cast<double>();
    r_new = r.cast<double>();

//    std::cout << "new x: "
//            << (J_new.transpose() * J_new).ldlt().solve(J_new.transpose() * r_new).transpose()
//            << std::endl;
//    my1->pcg(J, r, x, 1e-8, 10);
    std::cout << "new x scalar: "
              << (J.transpose() * J).ldlt().solve(J.transpose() * r).transpose()
              << std::endl;
    std::cout << "new x double: "
              << (J_new.transpose() * J_new).ldlt().solve(J_new.transpose() * r_new).transpose()
              << std::endl;

//    std::cout << "new x: " << x.transpose() << std::endl;
    std::cout << "J.rows(): " << J.rows() << std::endl;
    std::cout << "J.cols(): " << J.cols() << std::endl;
    std::cout << "r.rows(): " << r.rows() << std::endl;
}
void test_qr()
{
//    MatXX Jp = MatXX::Random(10, 10);
    MatXX I = MatXX::Identity(10, 10);
    MatXX A = MatXX::Random(10, 3);

    MatXX Qt = I;
    MatXX R = A;

    qr(Qt, R);
//    std::cout << "I\n" << I << std::endl;
    std::cout << "A\n" << A << std::endl;
//    std::cout << "Qt\n" << Qt << std::endl;
    std::cout << "R\n" << R << std::endl;
    std::cout << "Q * R\n" << Qt.transpose() * R << std::endl;
    std::cout << "Qt * A\n" << Qt * A << std::endl;
    std::cout << "test 2............" << std::endl;
    A = MatXX::Random(20, 3 * 3);
    MatXX Al = A.block(0, 0, 20, 3);
    MatXX Ap = A.block(0, 3, 20, 3 * 2);
    qr(Ap, Al);

    std::cout << "A:\n" << A << std::endl;
    std::cout << "Ap:\n" << Ap << std::endl;
    std::cout << "Al:\n" << Al << std::endl;
}

