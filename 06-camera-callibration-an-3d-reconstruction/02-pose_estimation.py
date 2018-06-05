import cv2
import matplotlib.pylab as plt
import numpy as np
import glob


def draw_3D_axis():
    """

    :return:
    """

    # load previously saved data
    with np.load('B.npz') as X:
        mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]
    # 创建停止条件，object points(象棋板点3D点)和坐标轴点，3D空间点作为坐标轴画3D空间。
    # 画轴长为3，X轴从(0,0,0)到(3,0,0)，Y和Z轴从(0,0,0)到(0,0,-3)。负数代表相机的方向
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    # axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
    axis = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
                       [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])

    # 加载每个图片，7x6方格，我们发现，利用子角像素细化。然后计算旋转和平移
    # 一旦有这些平移矩阵，利用他们投影到图片平面的坐标轴点上。
    # 简单说，我们图片平面找到3D空间对应的点(3,0,0)（0,3,0)（0，0，3）
    for fname in glob.glob('image/cheese/*.jpg'):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # find the rotation and translation vectors
            ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

            # img = draw(img, corners2, imgpts)
            img = draw_cube(img, corners2, imgpts)
            cv2.imshow('img', img)

            k = cv2.waitKey(0) & 0xff
            if k == 's':
                cv2.imwrite(fname[:6] + ".png", img)
    cv2.destroyAllWindows()


def draw(img, corners, imgpts):
    """
    从象棋盘上取点画图
    :param img:
    :param corners:
    :param imgpts:
    :return:
    """
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)

    return img


def draw_cube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw groud floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 4)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img


draw_3D_axis()
