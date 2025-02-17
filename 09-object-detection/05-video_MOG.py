import cv2
import numpy as np

# 视频背景去除， MOG算法

# 创建MOG背景分割器
mog = cv2.createBackgroundSubtractorMOG2()

# 打开视频文件
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 背景分割
    fgmask = mog.apply(frame)

    # 显示结果
    cv2.imshow('frame', frame)
    cv2.imshow('fgmask', fgmask)

    # 按下q键退出
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()