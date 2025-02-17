
import cv2

"""
Haar 特征分类器进行人脸检测的
"""

# 加载 Haar 特征分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 读取视频文件
cap = cv2.VideoCapture(0)

while True:
    # 读取一帧视频
    ret, frame = cap.read()

    if not ret:
        break

    # 灰度化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 人脸检测
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    # 绘制矩形框
    for (x, y, w, h) in faces:
        cv2.imwrite("template_face.jpg", frame[y:y+h, x:x+w, :])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 显示视频
    cv2.imshow('Video', frame)

    # 等待按键
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()