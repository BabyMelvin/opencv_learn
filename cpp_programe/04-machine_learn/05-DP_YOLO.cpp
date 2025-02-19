#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace dnn;
using namespace std;

int main(int argc, char **argv)
{
    // Load the YOLO model
    Net net = readNet("yolov3.weights", "yolov3.cfg");

    // load the input image
    Mat img = imread("dog.jpg");
    if (img.empty()) {
        cout << "Failed to load the image" << endl;
        return -1;
    }

    // preporcess the input image
    Mat blob = blobFromImage(img, 1 / 255.0, Size(416, 416), Scalar(0, 0, 0), true, false);

    // set the input blob for the network
    net.setInput(blob);

    // forward pass through the network
    vector<Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());

    // decode the output
    for (size_t i = 0; i < outs.size(); ++i) {
        float *data = (float *)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > 0.5) {
                int centerX = (int)(data[0] * img.cols);
                int centerY = (int)(data[1] * img.rows);
                int width = (int)(data[2] * img.cols);
                int height = (int)(data[3] * img.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                rectangle(img, Point(left, top), Point(left + width, top + height), Scalar(0, 255, 0), 2);
            }
        }
    }

    // display the output image
    imshow("YOLO detections", img);
    waitKey(0);

    return 0;
}