#include <iostream>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include "add.h"
using namespace cv;
using namespace std;

int main()
{
    cout << "Hello world" << endl;
    cout << "2 + 3 =" << Add(2, 3) << endl;

    // Read image
    Mat img = imread("D:\\python_programe\\opencv_learn\\cpp_programe\\helloworld\\astra_color.jpg", IMREAD_COLOR);
    if (img.empty())
    {
        cout << "Could not open or find the image" << endl;
        return -1;
    }
    // Show image
    namedWindow("Example1", WINDOW_AUTOSIZE);
    imshow("Example1", img);
    waitKey(0); // Wait for key press

    destroyAllWindows(); // Close all windows

    return 0;
}