import cv2
import numpy as np

# camshift tracker

# 读取视频内容
cap = cv2.VideoCapture(0)

# 读取第一帧
ret, frame = cap.read()

# 1.设置初始化串钩窗口
x, y, w, h = 300, 200, 100, 50  # 初始窗口
track_window = (x, y, w, h)

# 设置ROT
roi = frame[y:y+h, x:x+w]

# 转换为HSV色彩空间
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# 创建掩膜并计算直方图
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, 0, 255, cv2.NORM_MINMAX)

# 设置终止条件
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # 2.更新目标位置
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        # 绘制跟踪结果
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame, [pts], True, 255, 2)

        cv2.imshow('Camshift Tracking', img2)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()