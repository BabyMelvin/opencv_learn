import cv2
import matplotlib.pylab as plt
import numpy as np


def scale_invariable_feature_transform():
    """
        首先关键点检测，并且画下他们。
            第一步，构建SIFT对象，传递不同的参数
    :return:
    """
    img = cv2.imread('image/building.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT()

    # detect:找到图片中关键点，可以传入mask 选定范围
    # 关键点属性(x,y)，有意义区域大小，特定方向的角度，代表指定的要点
    kp = sift.detect(gray, None)

    # 如果找到了关键点，compute获得找到点的描述信息
    kp, des = sift.compute(gray, kp)

    # 如果没找到关键点，直接一步找到关键点和描述信息
    kp, des = sift.detectAndCompute(gray, None)

    # flag:cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    # 画圆，在关键点位置。
    img = cv2.drawKeypoints(gray, kp)

    cv2.imshow('bulding.jpg', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


scale_invariable_feature_transform()
