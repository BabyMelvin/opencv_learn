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


def drawing_line():
    """
        传递起始坐标
    :return:
    """
    # 创建一个黑图
    img = np.zeros((512, 512, 3), np.uint8)
    print(img)
    # 画一个斜线5个像素宽
    img = cv2.line(img, (0, 0), (100, 100), (255, 0, 0), 5)
    # 画一个长方形
    img = cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
    # 画一个圆
    img = cv2.circle(img, (477, 63), 63, (0, 0, 255), -1)
    # 画一个椭圆ellipse
    img = cv2.ellipse(img, (256, 256), (100, 50), 0, 0, 180, 255, -1)
    # 画一个多边形polygon
    pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    img = cv2.polylines(img, [pts], True, (0, 255, 255))
    # 写个字
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


drawing_line()
