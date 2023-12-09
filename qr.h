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

