import numpy as np
import cv2

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 检测轮廓
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, threshold_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制轮廓
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    # 处理每一帧
    cv2.imshow("Video Mine", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# hsv = np.arange(0, 27).reshape(3, 3, 3)
# h = hsv[:, :, 0]
# print(hsv)
# print("h:{0}".format(h))
