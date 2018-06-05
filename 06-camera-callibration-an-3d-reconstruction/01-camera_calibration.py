import cv2
import matplotlib.pylab as plt
import numpy as np
import glob

from numpy.core.tests.test_mem_overlap import xrange


def chess_board_corner():
    """
        用cv2.findChessboardCorners()来找象棋板上样式。所以需要传递要找的
    样式如8x8格子，或者是5x5格子。这个例子我们用7x6格子。
        返回角点如果样式获得，返回值为True.角点顺序放置(从左到右，从上到下)

    参见：
        这个函数并不能从所有图片中获得想要的样式。所以一个好的选择来写代码是，打开
    相机检测没帧来获得好的样式。
        一旦样式被找到，找到角点并以列表存储。同样在读下一帧之前需要一些间隔，为了在不同方向
    上调整象棋板。一直找，直到满足需要的数量。即使样例提供，但是不知道14张有多少张好的。所以所要读取
    所有，并找到好的。

    参见：
        除了象棋板，也采用一些圆形网格，cv2.findCircleGrid()来找样式。据说采用圆形样式相对
    需要数量少一些。

    当找到角点，使用cv2.cornerSubPix()来提高精度。
    :return:
    """
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points ,like (0,0,0),(1,0,0),(2,0,0)...(6,5,0)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    # array to store object points and image points from all the images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 3D points in image plane

    images = glob.glob('image/cheese/*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

        # if found ,add object points,image points(after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            imgpoints.append(corners2)

            # draw and display the corners
            img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(5000)

    # calibration
    # 现在我们找到object points和image point 准备去聚焦。使用cv2.calibrateCamera()，将返回
    # 相机矩阵，扭曲系数，旋转平移向量等
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    np.savez('B.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    cv2.destroyAllWindows()


def undistortion_opencv():
    """
    cv2.getOptimalNewCameraMatrix()基于自由缩放参数细化相机。
    如果缩放参数`alpha=0`,返回最少不想要像素的不变形图片。所以可能移除图像一些像素点。
    如果`alpha=1`所有像素被保留额外黑色图片。返回图片ROI用来裁剪结果

    :return:
    """
    img = cv2.imread("image/cheese/cheese.jpg")
    h, w = img.shape[:2]
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points ,like (0,0,0),(1,0,0),(2,0,0)...(6,5,0)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    # array to store object points and image points from all the images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 3D points in image plane

    images = glob.glob('image/cheese/*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

        # if found ,add object points,image points(after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            imgpoints.append(corners2)

            # draw and display the corners
            img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(5000)
            cv2.destroyAllWindows()
            break

    # calibration
    # 现在我们找到object points和image point 准备去聚焦。使用cv2.calibrateCamera()，将返回
    # 相机矩阵，扭曲系数，旋转平移向量等
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("+++++++++++++++++++")
    np.savez('B.npz', mtx, dist, rvecs, tvecs)

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # 1. using cv2.undistort()
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imshow('calbresult', dst)

    # 2.using remapping
    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imshow('calbresult2', dst)

    # Re-projection Error
    # 首先平移object points变为 image points 使用cv2.projectPoints()
    # 当我们计算出我们计算的平移和找算术角点的绝对泛数。找到平均错误，所有聚焦相片平均错误。
    mean_error = 0
    tot_error = 0
    for i in xrange(objpoints):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2 / len(imgpoints2))
        tot_error += error

    print("total error:", mean_error / len(objpoints))
    cv2.destroyAllWindows()


# 可以将相机矩阵和扭曲系数存储起来将来使用
chess_board_corner()
