import sys

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class ComboBox(QComboBox):
    def __init__(self, title, parent):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        print(e)

        if e.mimeData().hasText():
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        self.addItem(e.mimeData().text())



class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        lo = QFormLayout()
        lo.addRow(QLabel('Type some text in textbox and drag it into combo box'))

        edit = QLineEdit()
        edit.setDragEnabled(True)


        combo = ComboBox("Button", self)
        lo.addRow(edit, combo)
        self.setLayout(lo)
        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('Drag and Drop')



def main():
    app = QApplication(sys.argv)
    ex = Example()
    ex.show()

    sys.exit(app.exec_())
if __name__ == '__main__':
    main()