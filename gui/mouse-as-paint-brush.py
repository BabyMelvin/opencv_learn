import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
    学会处理鼠标事件
"""


def simple_demo():
    """
    双击鼠标左键，显示圆
    :return:
    """
    # 获得所有的事件信息
    events = [i for i in dir(cv2) if 'EVENT' in i]
    print(events)

    # 鼠标回调函数
    def draw_circle(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(img, (x, y), 10, (255, 0, 0), -1)

    # 创建黑的图，窗口，并绑定窗口和窗口
    img = np.zeros((512, 512, 3), np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)
    while True:
        cv2.imshow('image', img)
        if cv2.waitKey(20) & 0XFF == 27:
            break
    cv2.destroyAllWindows()


def more_advanced_mouse():
    """
    帮助理解对象跟踪，图像分割
    :return:
    """
    drawing = False
    mode = True
    ix, iy = -1, -1

    # 鼠标回调函数
    def draw_circle(event, x, y, flags, param):
        nonlocal ix, iy, drawing, mode
        if event == cv2.EVENT_LBUTTONDBLCLK:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing is True:
                if mode is True:
                    cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
                else:
                    cv2.circle(img, (ix, iy), 5, (0, 0, 255), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if mode is True:
                cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
            else:
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

    img = np.zeros((512, 512, 3), np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)
    while True:
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
        elif k == 27:
            break
    cv2.destroyAllWindows()


more_advanced_mouse()
# simple_demo()
