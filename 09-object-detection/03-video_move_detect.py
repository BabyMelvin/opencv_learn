import cv2

# 运动检测, 可以通过计算帧之间的差异来检测运动物体

# 读取视频文件
cap = cv2.VideoCapture(0)

# 定义两个阈值
threshold = 5
last_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 计算当前帧与上一帧的差异
    if last_frame is not None:
        diff = cv2.absdiff(frame, last_frame)

        # 计算差异的平均值
        diff_mean = cv2.mean(diff)[0]
        # 如果差异大于阈值，则检测到运动物体
        if diff_mean > threshold:
            cv2.putText(frame, "Motion detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 更新上一帧
    last_frame = frame.copy()

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()