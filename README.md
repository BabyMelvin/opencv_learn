# opencv_learn
# 学好图像处理，方便算法学习  python版
# 学会图像识别
# 学好方法，看C++版本
## 1.core module
核心模块，包含了图像处理的基础功能（如图像数组的表示和操作）。
- Mat: OpenCV 中用于存储图像和矩阵的基本数据结构。
- Scalar: 用于表示颜色或像素值。
- Point、Size、Rect: 用于表示点、尺寸和矩形。

基本绘图函数: cv.line()、cv.circle()、cv.rectangle()、cv.putText() 等。

使用场景：
- 图像的基本操作
- 绘制几何图形和文本

## 2.imgproc module
图像处理模块，提供图像的各种操作，如滤波、图像变换、形态学操作等。
主要类和函数:

- 图像滤波: cv.blur()、cv.GaussianBlur()、cv.medianBlur() 等。

- 几何变换: cv.resize()、cv.warpAffine()、cv.warpPerspective() 等。

- 颜色空间转换: cv.cvtColor()（如 BGR 转灰度、BGR 转 HSV）。

- 阈值处理: cv.threshold()、cv.adaptiveThreshold()。

- 边缘检测: cv.Canny()、cv.Sobel()、cv.Laplacian()。

应用场景:

- 图像平滑、锐化、边缘检测。

- 图像缩放、旋转、仿射变换。

- 图像二值化、颜色空间转换。

## 3.highgui module

图形用户界面模块，提供显示图像和视频的功能。
功能: 提供高层 GUI 和媒体 I/O 功能，用于图像的显示和交互。

主要类和函数:

- 图像显示: cv.imshow()、cv.waitKey()、cv.destroyAllWindows()。
- 视频捕获: cv.VideoCapture()、cv.VideoWriter()。
- 鼠标和键盘事件: cv.setMouseCallback()。

应用场景:

- 显示图像和视频。
- 捕获摄像头或视频文件。
- 处理用户交互（如鼠标点击、键盘输入）。


## 4.imgcodecs module
## 5.videoio module
## 6.calib3d module
 相机校准和 3D 重建模块。
主要类和函数:

- 相机校准: cv.calibrateCamera()、cv.findChessboardCorners()。
- 3D 重建: cv.solvePnP()、cv.reprojectImageTo3D()。

应用场景:

- 相机标定（用于去除镜头畸变）。
- 3D 重建（如从 2D 图像恢复 3D 信息）。


## 7.feature2d module
 特征检测与匹配模块，包含了角点、边缘、关键点检测等。
主要类和函数:

- 特征检测: cv.SIFT_create()、cv.ORB_create()、cv.SURF_create()。
- 特征匹配: cv.BFMatcher()、cv.FlannBasedMatcher()。
- 关键点绘制: cv.drawKeypoints()。

应用场景:

- 图像特征提取和匹配。
- 图像拼接、物体识别。


## 8.video module
提供视频处理的功能，如视频捕捉、视频流的处理等。
主要类和函数:

- 背景减除: cv.createBackgroundSubtractorMOG2()、cv.createBackgroundSubtractorKNN()。
- 光流法: cv.calcOpticalFlowPyrLK()。
- 目标跟踪: cv.TrackerKCF_create()、cv.TrackerMOSSE_create()。

应用场景:

- 视频中的运动检测。
- 目标跟踪（如行人、车辆跟踪）。

## 9.object module
目标检测模块。
主要类和函数:

- Haar 特征分类器: cv.CascadeClassifier()（用于人脸检测）。
- HOG 特征分类器: 用于行人检测。

应用场景:

- 人脸检测、行人检测。


## 10.dnn module
 深度学习模块
功能: 提供深度学习功能，支持加载和运行预训练的深度学习模型。

主要类和函数:

- 模型加载: cv.dnn.readNetFromCaffe()、cv.dnn.readNetFromTensorflow()。
- 前向传播: net.forward()。

应用场景:

- 图像分类、目标检测、语义分割。

## 11.ml module
功能: 提供机器学习算法。

主要类和函数:

- 支持向量机 (SVM): cv.ml.SVM_create()。
- K 均值聚类 (K-Means): cv.kmeans()。
- 神经网络 (ANN): cv.ml.ANN_MLP_create()。

应用场景:

- 图像分类、聚类分析。
## 12.photo module
## 13.stitching module
## 14.cuda module
