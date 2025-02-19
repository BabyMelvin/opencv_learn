#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
// K-Means clustering algorithm

// 我们生成了100个二维数据点，并使用K-Means算法将其分为3个簇。
// cv::kmeans函数的参数包括数据、簇数、标签、终止条件等。
int main()
{
    // 生成随机数据
    Mat data(50, 2, CV_32FC1);
    randu(data, Scalar(0, 0), Scalar(100, 100));

    // 设置K值和迭代条件
    int K = 3;
    Mat labels, centers;
    kmeans(data, K, labels,
        TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 10, 1.0),
        3, KMEANS_PP_CENTERS, centers);

    // 输出结果
    cout << "Labels:" << labels << endl;
    cout << "Centers:" << centers << endl;

    return 0;
}