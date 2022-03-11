from PyQt5.QtCore import QDir, Qt, pyqtSlot, QThread
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap
from PyQt5.QtWidgets import (QDesktopWidget, QDialog, QAction, QApplication, QFileDialog, QLabel,
        QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy, QTextEdit)
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from xmhighlighter import XMLHighlighter
from motion_clipper import MotionClipper
from dialogn import ClipDialog
import functools

class MotionClipperWindow(QMainWindow):
    def __init__(self):
        super(MotionClipperWindow, self).__init__()
        self._thread = None
        
        self.printer = QPrinter()

        #Create a QTextEdit widget
        self.xmlviewer = QTextEdit()
 
        self.setCentralWidget(self.xmlviewer)

        self.createActions()
        self.createMenus()
        
        #Create our XMLHighlighter derived from QSyntaxHighlighter
        self.highlighter = XMLHighlighter(self.xmlviewer.document())
        self.xmlviewer.setPlainText("")
 
        self.setWindowTitle("Motion Clipper")
        
        self.motionClipper = None
        self.xml_text = None
        self.fileName = None

    def handleProgressUpdated(self, value):
        try:
            self.clipDialog.ui.progressBar.setValue(value)
        except:
            pass
        QApplication.processEvents()
    
    def handleProgressTextUpdated(self, value):
        try:
            self.clipDialog.ui.infoLabel.setText(value)
        except:
            pass
        QApplication.processEvents()
        
    def handleFinishedUpdated(self, result_filename):
        self.clipDialog.ui.progressBar.setValue(100)
        self.on_worker_done()
        QApplication.processEvents()
        
        self.mb = QMessageBox()
        self.mb.setIcon(QMessageBox.Information)
        self.mb.setWindowTitle('Finished')
        #result_filename = self.fileName[:self.fileName.rindex(".")] + "_clipped."+self.fileName[self.fileName.rindex(".")+1:]
        
        self.mb.setText('Processing finished.\n The result saved as ' + result_filename + ".")
        self.mb.setStandardButtons(QMessageBox.Ok)
        self.mb.show()
        self.mb.exec_()
        
        
    def open(self):
        self.fileName, _ = QFileDialog.getOpenFileName(self, "Open Sequence Final Cut Pro XML File",
                QDir.currentPath())
        if self.fileName:            
            print(self.fileName)
            if ".fcpxmld" in self.fileName:
                self.fileName = self.fileName + "/Info.fcpxml"
            self.motionClipper = MotionClipper(self.fileName)
            self.motionClipper.progressValueUpdated.connect(self.handleProgressUpdated)
            self.motionClipper.progressTextUpdated.connect(self.handleProgressTextUpdated)
            self.motionClipper.finishedUpdated.connect(self.handleFinishedUpdated)
            
            self.xml_text = ET.tostring(self.motionClipper.tree.getroot(), encoding='utf8').decode('utf8')
            
            self.xmlviewer.setPlainText(self.xml_text)
            self.printAct.setEnabled(True)
            self.clipMotion.setEnabled(True)
            
            self.xmlviewer.setReadOnly(True)

    def print_(self):
        dialog = QPrintDialog(self.printer, self)
        if dialog.exec_():
            painter = QPainter(self.printer)
            rect = painter.viewport()

    def about(self):
        QMessageBox.about(self, "About Motion Clipper",
                "<p>The <b>Motion Clipper</b> is software solution to clip movements in videos represented by Final Cut PRO XML files."
                "</p>")

    def createActions(self):
        self.openAct = QAction("&Open...", self, shortcut="Ctrl+O",
                triggered=self.open)

        self.printAct = QAction("&Print...", self, shortcut="Ctrl+P",
                enabled=False, triggered=self.print_)

        self.exitAct = QAction("E&xit", self, shortcut="Ctrl+Q",
                triggered=self.close)

        self.clipMotion = QAction("Clip Motion", self, shortcut="Ctrl+R",
                enabled=False, triggered=self.openClipperDialog)

        self.aboutAct = QAction("&About", self, triggered=self.about)
        
        
    def openClipperDialog(self):
        self.motionClipper = MotionClipper(self.fileName)
        self.motionClipper.progressValueUpdated.connect(self.handleProgressUpdated)
        self.motionClipper.progressTextUpdated.connect(self.handleProgressTextUpdated)
        self.motionClipper.finishedUpdated.connect(self.handleFinishedUpdated)
        
        self.clipDialog = QDialog()
        self.clipDialog.ui = ClipDialog()
        self.clipDialog.ui.setupUi(self.clipDialog)
        self.clipDialog.setAttribute(Qt.WA_DeleteOnClose)
        self.clipDialog.ui.startButton.clicked.connect(self.run)
        self.clipDialog.finished.connect(self.onClipDialogClosed)
        self.clipDialog.exec_()

    def onClipDialogClosed(self):
        print("closed")
        self.toggle(False)
        
    def run(self):
        self.clipDialog.ui.setParams()
        self.clipDialog.ui.startButton.setEnabled(False)
        self.toggle(True)
        #self.motionClipper.process(self.clipDialog.ui.show_detection, self.clipDialog.ui.min_area, self.clipDialog.ui.alpha, self.clipDialog.ui.threshold, self.clipDialog.ui.width)
        

    def createMenus(self):
        self.fileMenu = QMenu("&File", self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addAction(self.printAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.viewMenu = QMenu("&Action", self)
        self.viewMenu.addAction(self.clipMotion)

        self.helpMenu = QMenu("&Help", self)
        self.helpMenu.addAction(self.aboutAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.helpMenu)

    def toggle(self, enable):
        if enable:
            if not self._thread:
                self._thread = QThread()

            self.motionClipper.moveToThread(self._thread)
            self.motionClipper.sgnFinished.connect(self.on_worker_done)
            
            self._thread.started.connect(functools.partial(self.motionClipper.process_fcpx, self.clipDialog.ui.show_detection, self.clipDialog.ui.min_area, self.clipDialog.ui.alpha, self.clipDialog.ui.threshold, self.clipDialog.ui.width, self.clipDialog.ui.minMotionFrames, self.clipDialog.ui.minNonMotionFrames, self.clipDialog.ui.nonMotionBeforeStart))
            self._thread.start()
        else:
            print('stopping the worker object')
            self.motionClipper.stop()

    @pyqtSlot()
    def on_worker_done(self):
        print('workers job was interrupted manually')
        self._thread.quit()
        self._thread.wait()
        self._thread = None

if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    mcWindow = MotionClipperWindow()
    sizeObject = QDesktopWidget().screenGeometry(-1)
    mcWindow.setFixedSize(sizeObject.width()*0.7,sizeObject.height()*0.7)
    mcWindow.show()
    sys.exit(app.exec_())
    
# pyinstaller --paths c:\Users\Kryvol\Anaconda3\envs\tensorflow\Lib\site-packages\PyQt5\Qt\bin\ --windowed --add-binary c:\Users\Kryvol\Anaconda3\envs\tensorflow\Library\bin\opencv_ffmpeg330_64.dll;. MotionClipper.py
#
