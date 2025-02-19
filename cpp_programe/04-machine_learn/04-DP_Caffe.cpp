#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// 一个简单的图像分类器，使用了Caffe模型。
// deploy.prototxt 是模型的配置文件，model.caffemodel是模型的权重文件。
int main()
{
    // 加载预训练的Caffe模型
    dnn::Net net = dnn::readNetFromCaffe("deploy.prototxt", "model.caffemodel");

    // 检查模型是否加载成功
    if (net.empty())
    {
        cerr << "Can't load network by using the following files: " << endl;
        cerr << "prototxt: " << "deploy.prototxt" << endl;
        cerr << "caffemodel: " << "model.caffemodel" << endl;
        return -1;
    }

    // 读取测试图像
    Mat img = imread("test.jpg");
    if (img.empty())
    {
        cerr << "Can't read the image." << endl;
        return -1;
    }

    // 预处理图像
    Mat blob = dnn::blobFromImage(img, 1, Size(224, 224), Scalar(104, 117, 123), false, false);

    // 设置输入层
    net.setInput(blob);

    // 前向传播
    Mat prob = net.forward();

    // 输出结果
    Point classIdPoint;
    double confidence;
    minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
    int classId = classIdPoint.x;

    cout << "The image is classified as: " << classId << endl;
    cout << "Confidence: " << confidence << endl;
    cout << "Caffe model loaded successfully." << endl;

    return 0;
}