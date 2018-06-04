import cv2
import matplotlib.pylab as plt
import numpy as np


def lucas_kanade_optical_flow():
    """
        OpenCV只提供一个方法:cv2.calcOpticalFlowPyrLK()
        决定点采用cv2.goodFeaturesToTracker()
        第一帧，检测一些Shi-Tomasi角点，然后迭代跟踪这些点利用Lucas-Kanade光学流方法

        cv2.calcOpticalFlowPyrLK()传递前一帧，前一个点集合下一帧。
            返回下一点带有数字状态，如果是1那么就发现这些点，否则为0.
        依次迭代，将前点到下一帧中

        代码不检测下一个关键点的正确性。即使图片中任何特征点都消失，能够找到下一个点可能接近它。
        所以为了真正鲁棒检测，角点应该特定间隔。OpenCV对这个样本每5个帧取一个样本
        同样采用向后检测，来获得较好的点
    :return:
    """
    cap = cv2.VideoCapture('image/slow.flv')

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # parameters for the lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # create a mask imgage for drawing purposes
    mask = np.zeros_like(old_frame)
    while (1):
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # caculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # select the good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

        img = cv2.add(frame, mask)

        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break;

        # now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cv2.destroyAllWindows()
    cap.release()


def dense_optical_flow_in_opencv():
    """
    下面汇总爱到密光流的方法。获得一个2通道数组，光流向量(u,v)。我们找到大小和方向。颜色编程，为了
    更好的看到。方向对应于Hue值图像。大小对应价值面。
    :return:
    """
    cap = cv2.VideoCapture('image/vtest.avi')

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)

    hsv[..., 1] = 255
    while (1):
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow('frame2', rgb)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('image/opticalfb.png', frame2)
            cv2.imwrite('image/opticalhsv.png', rgb)
        prvs = next

    cap.release()
    cv2.destroyAllWindows()


# lucas_kanade_optical_flow()
dense_optical_flow_in_opencv()
