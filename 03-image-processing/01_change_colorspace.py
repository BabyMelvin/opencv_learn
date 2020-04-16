import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
    目标：
        颜色空间转换:BGR<->Gray,BGR<->HSV
        抽取视频的颜色对象
"""


def change_color_space():
    """
        方法有很多，我们用最广泛的两种: BGR<->Gray,BGR<->HSV
        主要由flag确定
        HSV,hue[0 179] 色度 saturate[0 255]饱和度 value[0 255]纯度
        cv2.cvtColor(img,flogs)
        不同软件可能不同的缩放，需要正常化这些范围
    :return:
    """
    # 查看opencv中多有的flag
    flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
    print(flags)


# change_color_space()


def object_track():
    """
        提取颜色的物体，HSV比RGB更容易显示一个颜色。我们将提取一个蓝色物体：
            1.获取视频的每帧
            2.将BGR转成HSV颜色空间
            3.设定HSV图片蓝色空间阀值范围
            4.能够单独提取看色物体，做我们想做的任何事
        结果：
            1.获取的图像会有噪音
    :return:
    """
    cap = cv2.VideoCapture(0)
    while (1):
        # 获取每一帧
        _, frame = cap.read()

        # 转换颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 在HSV的蓝色空间的范围
        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # HSV图片阀值去获取蓝色
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # 将mask和原图位与操作
        res = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        cv2.imshow('res', res)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()


# object_track()


def find_HSV_value():
    """
        [H-10,100,100]和[H+10,255,255]作为范围
    :return:
    """
    green = np.uint8([[[0, 255, 0]]])
    hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
    print(hsv_green)


# find_HSV_value()
