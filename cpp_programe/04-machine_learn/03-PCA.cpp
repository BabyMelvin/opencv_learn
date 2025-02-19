#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    // 生成一些数据
    Mat data = (Mat_<float>(4, 2) << 1, 2, 2, 3, 3, 4, 4, 5);

    // 创建PCA对象
    PCA pca(data, Mat(), PCA::DATA_AS_ROW, 1);

    // 投影数据
    Mat projected = pca.project(data);
    cout << "Projected data: " << endl << projected << endl;

    return 0;
}