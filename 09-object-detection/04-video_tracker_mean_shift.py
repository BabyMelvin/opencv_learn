import cv2
import numpy as np

"""
初始化窗口：在视频的第一帧中，手动或自动选择一个目标区域，作为初始窗口。
计算质心：在当前窗口中，计算目标区域的质心（即像素点的均值）。
移动窗口：将窗口中心移动到质心位置。
迭代：重复步骤 2 和 3，直到窗口中心不再变化或达到最大迭代次数。
"""
# Load the video
video = cv2.VideoCapture(0)

# 读取第一帧
ret, frame = video.read()

# 设置初始化窗口
x, y, w, h = 300, 200, 100, 50
track_window = (x, y, w, h)

# 设置ROI
roi = frame[y:y+h, x:x+w]

# 转换为HSV色彩空间
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# 创建掩码并计算直方图
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, 0, 255, cv2.NORM_MINMAX)

# 设置终止条件
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = video.read()

    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 计算反射投影
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # 应用MeanShift算法
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # 绘制跟踪和结果
        x, y, w, h = track_window
        img2 = cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
        cv2.imshow('MeanShift Tracking', img2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

video.release()
cv2.destroyAllWindows()

