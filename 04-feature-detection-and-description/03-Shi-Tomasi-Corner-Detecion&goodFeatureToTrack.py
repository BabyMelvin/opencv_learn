import cv2
import matplotlib.pylab as plt
import numpy as np


def good_features_to_track():
    """
        cv2.goodFeauturesToTrack()
        能够找到最强的转角（通过Shi-Tomasi或Harris Corner细化的）
        然后具体化要找转角的数目。
        质量水平值0~1，表示最低转角最低质量（每个不期望的）。当转角被检测到
        我们将会提供最小欧几里得距离。

        所有信息，找到图片的转角。所有转角在质量水平以下，然后降序排列剩下的转角，
        。然后函数获得一个最强转角，抛去咋最小具体方位内最强点，并且返回N个最强转角。
    :return:
    """
    img = cv2.imread('image/cube.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)

    plt.imshow(img), plt.show()


good_features_to_track()
