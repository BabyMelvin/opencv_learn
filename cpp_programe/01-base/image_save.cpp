#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// basic function to save an image
int main() {
    Mat img = imread("lena.jpg");
    if (img.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    // 获取图像属性
    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();
    cout << "Image width: " << width << endl;
    cout << "Image height: " << height << endl;
    cout << "Image channels: " << channels << endl;

    // 修改图像属性
    // 获取像素点
    Vec3b pixel = img.at<Vec3b>(100, 100);
    cout << "Pixel value (B, G, R): " << (int)pixel[0] << ", " << (int)pixel[1] << ", " << (int)pixel[2] << endl;

    // 修改像素点
    img.at<Vec3b>(100, 100) = Vec3b(255, 0, 0); // 将像素设置成蓝色


    // 颜色空间转换
    Mat img_gray;
    cvtColor(img, img_gray, COLOR_BGR2GRAY); // 转换成灰度图
    bool isSuccess = imwrite("lena_copy.jpg", img);
    if(isSuccess)
        cout << "Image saved as lena_copy.jpg" << endl;
    else
        cout << "Error in saving the image" << endl;

    imshow("Image", img_gray);

    return 0;
}