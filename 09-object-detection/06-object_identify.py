import cv2
import numpy as np

# 物体识别

# 1.加载图像: 读取搜索图像和模板图像。
# 2.模板匹配: 使用cv2.matchTemplate()函数进行模板匹配。
# 3.提取特征点: 使用cv2.minMaxLoc()函数提取特征点。
# 4.绘制矩形: 使用cv2.rectangle()函数绘制矩形。
# 5.显示结果: 显示搜索图像和矩形。

#  1.加载图像
template_image = cv2.imread('template_face.jpg', cv2.IMREAD_GRAYSCALE)

# 获取图像宽高
# h, w, *_ = img.shape
w, h = template_image.shape[::-1]


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2.模板匹配
    result = cv2.matchTemplate(frame_gray, template_image, cv2.TM_CCOEFF_NORMED)

    # 设置匹配阈值
    threshold = 0.8
    loc = np.where(result >= threshold)


    # 3.图像标记匹配位置
    for pt in zip(*loc[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    # 4.显示结果
    cv2.imshow('Match Image', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


