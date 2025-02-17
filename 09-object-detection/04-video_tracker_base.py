import cv2
import numpy as np

# 视频目标跟踪 meanshift 算法

# 创建视频对象
cap = cv2.VideoCapture(0)

# 设置视频参数
cap.set(3, 640)  # 设置宽度
cap.set(4, 480)  # 设置高度

# 定义颜色范围
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])

# 定义目标追踪函数
def track_object(frame):
    # 转换为 HSV 色彩空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 寻找蓝色目标
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 开运算提取目标区域
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # 寻找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历轮廓
    for contour in contours:
        # 计算轮廓的矩
        M = cv2.moments(contour)

        # 计算轮廓的质心
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # 绘制轮廓
        cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)

        # 绘制质心
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    return frame

# 循环读取视频帧
while True:
    # 读取视频帧
    ret, frame = cap.read()

    # 目标追踪
    frame = track_object(frame)

    # 显示视频帧
    cv2.imshow('Video', frame)

    # 等待按键
    key = cv2.waitKey(1) & 0xFF

    # 按 q 键退出
    if key == ord('q'):
        break

# 释放视频对象
cap.release()

cap.closeAllWindows()