import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

def window():
    app = QApplication(sys.argv)
    win = QWidget()
    win.setWindowTitle('PyQt5')
    win.setGeometry(30, 30, 600, 600)
    # 创建一个 QLabel 控件
    label = QLabel(win)
    label.setText('Hello World')

    QLineEdit(win)

    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    window()