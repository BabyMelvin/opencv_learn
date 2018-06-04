import cv2
import matplotlib.pylab as plt
import numpy as np


def orb_in_opencv():
    """
        用cv2.ORB()或者feature2d统统接口创建ORB
        许多可选参数：
            nFeatures：获得特征最大数目，默认500
            scoreType: 表示使用Harris score(默认)还是FAST score来排名特征
            WTA_K: 每个方向BRIEF描述元素产生点数目
                默认是两个，同时选择两个点。
                    用来匹配，NORM_HAMMING距离被使用。
                如果是3或者4，匹配描述使用NORM_HAMMING2
    :return:
    """
    img = cv2.imread('image/cube.jpg', 0)
    img2 = img.copy()
    img3 = img2.copy()

    # initiate STAR detector
    orb = cv2.ORB_create()

    # find the keypoints with ORB
    kp = orb.detect(img, None)

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)

    # draw only keypoints location,not size and orientation
    img3 = cv2.drawKeypoints(img, kp, img2, color=(255, 0, 0), flags=0)
    plt.imshow(img3), plt.show()


orb_in_opencv()
