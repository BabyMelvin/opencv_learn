import cv2
import numpy as np
import matplotlib.pyplot as plt


def hough_transform_in_opencv():
    """
    cv2.HoughLines()
    返回：(ρ，θ），ρ单位像素，θ弧度（radians）。
    输入：
        1.二值图片
            使用阀值，或者在使用Hough变换之前,使用Canny边检测
        2,3参数ρ，θ的精度
        4，最小选举认为是直线。
            选举点数据根据线上的数目个数，代表待检测线的最短长度
    :return:
    """
    img = cv2.imread('image/dave.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # NOTE：注意，当精度为1时候，可能找不到线
    lines = cv2.HoughLines(edges, 2, np.pi / 180, 200)
    print(len(lines))
    for i in np.arange(0, len(lines)):
        for rho, theta in lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('image_houghlines', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# hough_transform_in_opencv()
def probabilistic_hough_ransform():
    """
    基于Robust 线检测
    参数：
        minLineLength：比这短丢弃。
        maxLineGap:   线之间最大允许认为是单独的直线。
    返回两点
    :return:
    """
    img = cv2.imread('image/dave.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    minLineLength = 200
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
    print(lines.shape)
    for i in np.arange(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('houghLinesP', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


probabilistic_hough_ransform()
