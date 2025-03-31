import sys
from os import path
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import time

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


class RecordVideo(QWidget):
    image_data = QtCore.pyqtSignal(tuple)

    def __init__(self, camera_port=0, parent=None):
        super().__init__(parent)
        self.camera = cv2.VideoCapture(camera_port)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        self.timer = QtCore.QBasicTimer()

    def start_recording(self):
        self.timer.start(0, self)

    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return

        t = time.time()
        read, data = self.camera.read()
        if read:
            # 发送端
            self.image_data.emit((data, t))

class FaceDetectionWidget(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = QtGui.QImage()
        self._red = (0, 0, 255)
        self._width = 2
        self._min_size = (30, 30)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 数据接收端
    def image_data_slot(self, data):
        image, t = data

        # process image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # draw detect result
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), self._red, self._width)

        self.image = self.get_qimage(image)

        # rescale window to image size
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()

    def get_qimage(self, image):
        height, width, colors = image.shape
        bytes_per_line = 3 * width

        image = QImage(image.data,
                       width,
                       height,
                       bytes_per_line,
                       QImage.Format_RGB888)

        # cv return iamge in BGR format
        image = image.rgbSwapped()
        return image

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)

        self.image = QImage()

class MainWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.face_detection_widget = FaceDetectionWidget(self)
        self.record_video = RecordVideo()
        image_data_slot = self.face_detection_widget.image_data_slot
        self.record_video.image_data.connect(image_data_slot)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.face_detection_widget)
        self.setLayout(layout)
        self.run_button = QPushButton('Start')
        layout.addWidget(self.run_button)
        self.run_button.clicked.connect(self.record_video.start_recording)
        self.setLayout(layout)

def main():
    app = QtWidgets.QApplication(sys.argv)

    main_window = QtWidgets.QMainWindow()
    main_window.setWindowTitle("Yolo Demos")
    ui = MainWidget()
    main_window.setGeometry(30, 30, 1000, 1000)
    main_window.setCentralWidget(ui)
    main_window.show()

    sys.exit(app.exec_())

if __name__ == '__main__':
    # Get the script directory
    script_dir = path.dirname(path.abspath(__file__))
    print(script_dir)
    main()