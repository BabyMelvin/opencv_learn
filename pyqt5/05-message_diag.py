import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


def window():
    app = QApplication(sys.argv)
    win = QWidget()
    win.setWindowTitle('PyQt5')

    b = QPushButton(win)
    b.setText('Button1')
    b.move(30, 50)
    b.clicked.connect(show_dialog)
    win.show()
    sys.exit(app.exec_())

def show_dialog():
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)

    msg.setText('This is a message box')
    msg.setInformativeText("This is additional information")
    msg.setWindowTitle('Message box')
    msg.setDetailedText('The details are as follows:')
    msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    msg.buttonClicked.connect(msg_btn)

    retval = msg.exec_()

def msg_btn(i):
    print('Button clicked is:', i.text())

if __name__ == '__main__':
    window()