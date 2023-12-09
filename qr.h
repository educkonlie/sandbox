#include "common.h"
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
        double pivot = Jl(j, j);
//        std::cout << "pivot: " << pivot << std::endl;
        for (int i = j + 1; i < nres; i++) {
#if false
            double a;
            while ((a = Jl(i, j)) == 0 && i < nres) {
                i++;
            }
#else
            double a = Jl(i, j);
#endif
//            std::cout << "a: " << a << std::endl;
//            std::cout << "i, j: " << i << " " << j << std::endl;
            if (i == nres) {
//                assert(std::abs(pivot) > 0.0000001);
                if (pivot == 0.0)
                    pivot = 0.000001;
                Jl(j, j) = pivot;
                std::cout << "......pivot...." << pivot << std::endl;
                break;
            }
            double r = sqrt(pivot * pivot + a * a);
            double c = pivot / r;
            double s = a / r;
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
        double pivot = Jl(j, j);
//        std::cout << "pivot: " << pivot << std::endl;
        for (int i = j + 1; i < nres; i++) {
#if false
            double a;
            while ((a = Jl(i, j)) == 0 && i < nres) {
                i++;
            }
#else
            double a = Jl(i, j);
#endif
//            std::cout << "a: " << a << std::endl;
//            std::cout << "i, j: " << i << " " << j << std::endl;
            if (i == nres) {
//                assert(std::abs(pivot) > 0.0000001);
                if (pivot == 0.0)
                    pivot = 0.000001;
                Jl(j, j) = pivot;
                std::cout << "......pivot...." << pivot << std::endl;
                break;
            }
            double r = sqrt(pivot * pivot + a * a);
            double c = pivot / r;
            double s = a / r;
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
#define CPARS 2
#define XI 3
// struct of JM rM
//  JM_marg   JM_remained   JM_CPARS  rM
// 然后将上面整体进行QR分解，rM会比JM多一行非零，但这个是正常现象
// 最后将JM_marg对应的行列删除，将底下为0的行删除
// 如果不做marg，那么仍旧可以通过求解(J.transpose() * JM).ldlt().solve(J.transpose() * rM)
// 来比较，如果解不变，则化简有效
void marg_frame()
{
    int num_of_frames = 4;
//    int idx = 2;
//    MatXX Jr = MatXX::Random(20, CPARS + num_of_frames * XI + 1);
    MatXX Jr = MatXX::Random(20, 9);
    std::cout << "Jr_old:\n" << Jr << std::endl;

//    VectorXd r = VectorXd::Random(20);
//    std::cout << "J:\n" << J << std::endl;
#if 0
    MatXX Jp = J.leftCols(CPARS);
    MatXX Jl = J.rightCols(num_of_frames * XI);

    std::cout << "Jp:\n" << Jp << std::endl;
    std::cout << "Jl:\n" << Jl << std::endl;

    //! move mid col to head
    MatXX temp = Jl.middleCols(idx * XI, XI);
    Jl.middleCols(idx * XI, XI) = Jl.leftCols(XI);
    Jl.leftCols(XI) = temp;
    std::cout << "Jp:\n" << Jp << std::endl;
    std::cout << "Jl:\n" << Jl << std::endl;

    qr(Jp, Jl);

    std::cout << "Jp:\n" << Jp << std::endl;
    std::cout << "Jl:\n" << Jl << std::endl;
#endif
    MatXX J = Jr.leftCols(Jr.cols() - 1);
    VectorXd r = Jr.rightCols(1);
    std::cout << "x:\n" << (J.transpose() * J).ldlt().solve(J.transpose() * r).transpose() << std::endl;

    qr2(Jr);
    MatXX Jr_new = Jr.topRows(Jr.cols());
    std::cout << "Jr_new:\n" << Jr_new << std::endl;

    J = Jr_new.leftCols(Jr_new.cols() - 1);
    r = Jr_new.rightCols(1);
    std::cout << "x:\n" << (J.transpose() * J).ldlt().solve(J.transpose() * r).transpose() << std::endl;
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
    std::cout << "test 2............" << std::endl;
    A = MatXX::Random(20, 3 * 3);
    MatXX Al = A.block(0, 0, 20, 3);
    MatXX Ap = A.block(0, 3, 20, 3 * 2);
    qr(Ap, Al);

    std::cout << "A:\n" << A << std::endl;
    std::cout << "Ap:\n" << Ap << std::endl;
    std::cout << "Al:\n" << Al << std::endl;
}
