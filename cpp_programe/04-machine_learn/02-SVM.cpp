#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// SVM training function
int main()
{
    Mat trainData = ((Mat_<float>(4, 2)) << 1, 1, 1,2, 2, 1, 2, 2);
    Mat labels = ((Mat_<int>(4, 1)) << 1, -1, 1, -1);

    // 创建SVM分类器
    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setKernel(ml::SVM::LINEAR);
    svm->setC(1);

    // 训练SVM分类器
    svm->train(trainData, ml::ROW_SAMPLE, labels);

    // 预测新数据
    Mat testData = ((Mat_<float>(1, 2)) << 1.5, 1.5);
    float response = svm->predict(testData);
    cout << "Prediction: " << response << endl;
    return 0;
}