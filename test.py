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

    # Canny 检测
    edges_canny = cv2.Canny(frame, 100, 200)    # Canny 检测

    # sobel计算x,y方向梯度
    sobel_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)

    #计算梯度幅值
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)


    #Laplacian 算子
    laplacian = cv2.Laplacian(frame, cv2.CV_64F)


    # 绘制轮廓, 这里会改变frame
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    # 处理每一帧
    cv2.imshow("Video Mine", frame)

    # 显示canny 边缘检测
    cv2.imshow("Canny Edges", edges_canny)

    # 显示sobel边缘检测
    cv2.imshow("Sobel X", sobel_x)
    cv2.imshow("Sobel Y", sobel_y)
    cv2.imshow("Sobel Combined", sobel_combined)

    # 显示Laplacian
    cv2.imshow("Laplacian", laplacian)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# hsv = np.arange(0, 27).reshape(3, 3, 3)
# h = hsv[:, :, 0]
# print(hsv)
# print("h:{0}".format(h))
