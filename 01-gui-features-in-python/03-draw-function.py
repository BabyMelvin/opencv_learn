import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
    目标：
        1.画不同的几何形状
    共用参数：
        img      :你想画形状的图片
        color    :形状颜色,BGR。传递元组参数(255,0,0)
        thickness:厚度，-1填充封闭图形
        lineType :8-connected,anti-aliased line(反锯齿形),cv2.LINE_AA曲线会很好
"""


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(x, y)


def drawing_line():
    """
        传递起始坐标
    :return:
    """
    # 创建一个黑图
    img = np.zeros((512, 512, 3), np.uint8)
    # 画一个斜线5个像素宽
    img = cv2.line(img, (0, 0), (100, 100), (255, 0, 0), 5)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# drawing_line()


def draw_rectangle():
    # 创建一个黑图
    img = np.zeros((512, 512, 3), np.uint8)
    # 画一个长方形,左上和右下
    img = cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', mouse_callback)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# draw_rectangle()


def draw_circle():
    # 创建一个黑图
    img = np.zeros((512, 512, 3), np.uint8)
    # 画一个圆,-1表示实心
    img = cv2.circle(img, (377, 63), 63, (0, 0, 255), -1)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# draw_circle()


def draw_ellipse():
    # 创建一个黑图
    img = np.zeros((512, 512, 3), np.uint8)
    # 画一个椭圆ellipse,中心，(横，竖)，起始角度,起始偏移，结束偏移,颜色，厚度   逆时针为正
    img = cv2.ellipse(img, (256, 256), (100, 50), 90, 0, 180, 255, -1)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# draw_ellipse()


def draw_polygon():
    # 创建一个黑图
    img = np.zeros((512, 512, 3), np.uint8)
    # 画一个多边形polygon
    pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
    print(pts)
    # 其中-1 表示自己不知道多少，知道到后面是1，2
    pts = pts.reshape((-1, 1, 2))
    print([pts])
    img = cv2.polylines(img, [pts], True, (0, 255, 255))
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# draw_polygon()


def paint_text():
    # 创建一个黑图
    img = np.zeros((512, 512, 3), np.uint8)
    # 写个字
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'OpenCV', (10, 100), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# paint_text()
