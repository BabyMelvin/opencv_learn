import cv2
import matplotlib.pylab as plt
import numpy as np


def meanshift_in_opencv():
    """
        为了使用meanshift，首先假定目标，找到直方图，为了对计算meanshift投影目标到每帧。
        同样需要提供初始窗口的位置。
        对于直方图，值考虑Hue。避免低亮度值错误，低亮度也不使用cv2.inRange()
    :return:
    """
    cap = cv2.VideoCapture('image/slow.flv')

    # take first frame of the video
    ret, frame = cap.read()

    # setup initial location of window
    r, h, c, w = 250, 90, 400, 125  # simply hardcoded the value
    track_window = (c, r, w, h)

    # set up the ROI for tracking
    roi = frame[r:r + h, c:c + w + w]
    hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # setup the termination criteria,either 10 iteration or move by at least 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while (1):

        ret, frame = cap.read()

        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            # apply meanshift to get the new location
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)

            # draw it on image
            x, y, w, h = track_window
            img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
            cv2.imshow('img2', img2)

            k = cv2.waitKey(60) & 0xff
            if k == 27:
                break
            else:
                cv2.imwrite("image/" + chr(k) + ".jpg", img2)
        else:
            break
    cv2.destroyAllWindows()
    cap.release()


def camshift_in_opencv():
    """
    几乎和meanshift一样，但是返回一个旋转方块，方框参数用作传递给下一搜索窗口的迭代。
    :return:
    """
    cap = cv2.VideoCapture('image/slow.flv')

    # take first frame of the video
    ret, frame = cap.read()

    # set up initial location of the window
    r, h, c, w = 250, 90, 400, 125  # simply hardcoded the values
    track_window = (c, r, w, h)

    # set up the ROI for tracking
    roi = frame[r:r + h, c:c + w]
    hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # setup the termination criteria ,either 10 iteration or move by at least 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    while (1):
        ret, frame = cap.read()

        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            # apply meanshift to get the new location
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)

            # draw it on image
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)

            img2 = cv2.polylines(frame, [pts], True, 255, 2)
            cv2.imshow('img2', img2)

            k = cv2.waitKey(60) & 0xff
            if k == 27:
                break
            else:
                cv2.imwrite(chr(k) + ".jpg", img2)
        else:
            break

    cv2.destroyAllWindows()
    cap.release()


# meanshift_in_opencv()
camshift_in_opencv()
