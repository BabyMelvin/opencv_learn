import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

def window():
    app = QApplication(sys.argv)
    win = QWidget()
    win.setWindowTitle('PyQt5')

    win.setGeometry(30, 30, 300, 200)
    b1 = QPushButton(win)
    b1.setText('Button1')
    b1.move(30, 50)

    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    window()