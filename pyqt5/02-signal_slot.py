import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

def window():
    app = QApplication(sys.argv)
    win = QWidget()
    win.setWindowTitle('PyQt5')
    b1 = QPushButton(win)
    b1.setText('Button1')
    b1.move(30, 50)
    b1.clicked.connect(lambda: print('Button1 clicked'))

    label = QLabel(win)
    label.move(80, 20)
    label.setText('Hello World')

    b2 = QPushButton(win)
    b2.setText('Button2')
    b2.move(150, 50)

    b2.clicked.connect(b2_clicked)

    win.show()
    sys.exit(app.exec_())

def b2_clicked():
    print('Button2 clicked')

if __name__ == '__main__':
    window()