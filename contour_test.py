import numpy as np
import cv2

i = 1
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 二值化处理， 需要是灰度图
    _, binary = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)


    # 查找轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if i == 1:
        print(f"contours nums:{len(contours)}, hierarchy count:{hierarchy.shape}")

    for contour in contours:
        area = cv2.contourArea(contour)
        length = cv2.arcLength(contour, True)

        # 边界矩形信息
        x, y, w, h = cv2.boundingRect(contour)
        # 边界矩形绘画
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


        # 最小外接矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)


        # 最小外接圆
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(frame, center, radius, (255, 0, 0), 2)

        if i == 1:
            print(f"Contour Area: {area}, Length: {length}")

    i = i + 1

    cv2.drawContours(frame, contours, -1, (255, 0, 0), 2)

    cv2.imshow("Contour", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()