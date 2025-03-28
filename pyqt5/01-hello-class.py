import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


class Window(QWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__()
        self.resize(200, 50)
        self.setWindowTitle("Hello PyQt5")
        self.setGeometry(100, 100, 300, 200)
        self.b = QLabel(self)
        self.b.setText("Hello World")
        font = QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.b.setFont(font)
        self.b.setAlignment(Qt.AlignCenter)
        self.b.move(100, 100)

def main():
    app = QApplication(sys.argv)
    ex = Window()
    ex.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()