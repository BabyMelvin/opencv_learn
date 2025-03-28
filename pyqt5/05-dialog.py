import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
def window():
    app = QApplication(sys.argv)
    win = QWidget()
    win.setWindowTitle('PyQt5')
    win.setGeometry(30, 30, 300, 200)

    btn= QPushButton(win)
    btn.setText('Button1')
    btn.move(30, 50)
    btn.clicked.connect(show_dialog)

    win.show()
    sys.exit(app.exec_())

def show_dialog():
    dlg = QDialog()
    dlg.setWindowTitle('Dialog')
    b1 = QPushButton('ok', dlg)
    dlg.setGeometry(30, 30, 300, 200)
    dlg.setWindowModality(Qt.ApplicationModal)
    dlg.exec_()

if __name__ == '__main__':
    window()