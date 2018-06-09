import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
    将window和trackbar绑定
    cv2.getTrackbarPos()
        1.trackbar name
        2.window name attached
        3.default value
        4.maximum
        5.callback function
    可以添加按钮和开关
         
"""


def button_listen():
    print("button click")


def code_demo():
    """
        一个开关只有，否则屏幕始终是black
    :return:
    """

    def nothing(x):
        pass

    # 定义一个黑色图片，一个窗口
    img = np.zeros((300, 512, 3), np.uint8)
    cv2.namedWindow('image')

    # 创建轨迹及条控制颜色变化
    cv2.createTrackbar('R', 'image', 0, 255, nothing)
    cv2.createTrackbar('G', 'image', 0, 255, nothing)
    cv2.createTrackbar('B', 'image', 0, 255, nothing)
    # 创建开关
    switch = "0:OFF \n 1 : ON"
    cv2.createTrackbar(switch, 'image', 0, 1, nothing)
    cv2.createButton('开关', button_listen)
    while True:
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k is 27:
            break
        # 获得当前4个轨迹条位置
        r = cv2.getTrackbarPos('R', 'image')
        g = cv2.getTrackbarPos('G', 'image')
        b = cv2.getTrackbarPos('B', 'image')
        s = cv2.getTrackbarPos(switch, 'image')

        if s is 0:
            img[:] = 0
        else:
            img[:] = [b, g, r]
    cv2.destroyAllWindows()


code_demo()
