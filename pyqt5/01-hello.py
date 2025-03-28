import sys

from PyQt5.QtWidgets import *

def window():
    app = QApplication(sys.argv)
    win = QWidget()
    win.setWindowTitle('Hello PyQt5')
    win.setGeometry(100, 100, 300, 200)
    b = QLabel(win)
    b.setText("Hello World")
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    window()