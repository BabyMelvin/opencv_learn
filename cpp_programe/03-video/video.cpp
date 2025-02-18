#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    VideoCapture cap(0); // open the default camera

    if (!cap.isOpened())  // check if we succeeded
        return -1;

    while (true)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera

        if (frame.empty()) // check if frame is empty
            break;
        Mat edges;
        // 应用canny边缘检测算法
        Canny(frame, edges, 50, 150);
        imshow("Live Video", edges); // show the frame in a window named "Live Video"

        if (waitKey(30) == 27) // wait for 30ms or for 'esc' key to exit
            break;
    }

    cap.release(); // release the camera
    destroyAllWindows(); // close all windows
}