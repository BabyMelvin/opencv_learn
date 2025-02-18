#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main()
{
    // 读取图像
    Mat img = imread("lena.jpg");

    // 检查图像是否成功加载
    if (img.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    // 裁剪与缩放
    Rect roi(100, 100, 200, 200); // 裁剪区域,x, y, w, h
    Mat cropped = img(roi); // 裁剪

    // 缩放图像
    Mat resizeImage;
    resize(img, resizeImage, Size(100, 100)); // 缩放


    // 拷贝与克隆图像
    Mat clonedImage = img.clone(); // 拷贝
    Mat copiedImage = img; // 克隆


    // 旋转图像
    Mat RotateImage;
    Point2f center(img.cols / 2.0, img.rows / 2.0); // 旋转中心
    double angle = 45; // 旋转角度
    double scale = 1.0; // 缩放比例
    Mat rotMat = getRotationMatrix2D(center, angle, scale); // 获取旋转矩阵
    warpAffine(img, RotateImage, rotMat, img.size()); // 旋转

    // 显示图像
    namedWindow("Display window", WINDOW_AUTOSIZE);
    imshow("Display window", img);

    // 等待按键
    waitKey(0);

    // 销毁窗口
    destroyWindow("Display window");

    return 0;
}
