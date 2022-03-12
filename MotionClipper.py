from PyQt6.QtCore import QDir, Qt, pyqtSlot, QThread, QRunnable, QThreadPool
from PyQt6.QtGui import QImage, QPainter, QPalette, QPixmap, QGuiApplication
from PyQt6.QtWidgets import (QMainWindow, QApplication, QDialog, QFileDialog, QLabel, QMenu, QMessageBox, QScrollArea, QTextEdit)
from PyQt6.QtGui import QAction
from PyQt6.QtPrintSupport import QPrintDialog, QPrinter
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from xmhighlighter import XMLHighlighter
from motion_clipper import MotionClipper
from dialogn import ClipDialog
import functools
import os
import traceback


class MotionClipperWindow(QMainWindow):
    def __init__(self):
        super(MotionClipperWindow, self).__init__()
        self.threadpool = QThreadPool()
        
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
        self.mb.setIcon(QMessageBox.Icon.Information)
        self.mb.setWindowTitle('Finished')
        #result_filename = self.fileName[:self.fileName.rindex(".")] + "_clipped."+self.fileName[self.fileName.rindex(".")+1:]
        
        self.mb.setText('Processing finished.\n The result saved as ' + result_filename + ".")
        self.mb.setStandardButtons(QMessageBox.StandardButton.Ok)
        self.mb.show()
        self.mb.exec()
        
        
    def open(self):
        self.fileName, _ = QFileDialog.getOpenFileName(self, "Open Sequence Final Cut Pro XML File",
                QDir.currentPath(), filter = "fcpxmld(*.fcpxmld);;fcpxml(*.fcpxml);;xml(*.xml)")
        if self.fileName:            
            print(self.fileName)
            _, self.extension = os.path.splitext(self.fileName)
            if ".fcpxmld" in self.fileName:
                self.fileName = self.fileName + "/Info.fcpxml"

            self.motionClipper = MotionClipper(self.fileName)
            self.motionClipper.notifier.progressValueUpdated.connect(self.handleProgressUpdated)
            self.motionClipper.notifier.progressTextUpdated.connect(self.handleProgressTextUpdated)
            self.motionClipper.notifier.finishedUpdated.connect(self.handleFinishedUpdated)
            self.motionClipper.notifier.sgnFinished.connect(self.on_worker_done)
            
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
        self.motionClipper.notifier.progressValueUpdated.connect(self.handleProgressUpdated)
        self.motionClipper.notifier.progressTextUpdated.connect(self.handleProgressTextUpdated)
        self.motionClipper.notifier.finishedUpdated.connect(self.handleFinishedUpdated)
        self.motionClipper.notifier.sgnFinished.connect(self.on_worker_done)
        
        self.clipDialog = QDialog()
        self.clipDialog.ui = ClipDialog()
        self.clipDialog.ui.setupUi(self.clipDialog)
        self.clipDialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.clipDialog.ui.startButton.clicked.connect(self.run)
        self.clipDialog.finished.connect(self.onClipDialogClosed)
        self.clipDialog.exec()

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
        try:
            if enable:

                self.motionClipper.setParams(self.clipDialog.ui.show_detection, self.clipDialog.ui.min_area, self.clipDialog.ui.alpha, self.clipDialog.ui.threshold, self.clipDialog.ui.width, 
                    self.clipDialog.ui.minMotionFrames, self.clipDialog.ui.minNonMotionFrames, self.clipDialog.ui.nonMotionBeforeStart, self.clipDialog.ui.nonMotionAfter, self.clipDialog.ui.minFramesToKeep)
                
                self.threadpool.start(self.motionClipper)

            else:
                print('stopping the worker object')
                self.motionClipper.stop()
        except Exception as ex:
            print(traceback.format_exc())

    @pyqtSlot()
    def on_worker_done(self):
        print('workers job was interrupted or finished')
        #self._thread.quit()
        #self._thread.wait()
        #self._thread = None

if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    mcWindow = MotionClipperWindow()

    sizeObject = QGuiApplication.primaryScreen().availableGeometry()
    #sizeObject = QDesktopWidget().screenGeometry(-1)
    mcWindow.setFixedSize(sizeObject.width()*0.7,sizeObject.height()*0.7)
    mcWindow.show()

    sys.exit(app.exec())
    
# pyinstaller --paths c:\Users\Kryvol\Anaconda3\envs\tensorflow\Lib\site-packages\PyQt6\Qt\bin\ --windowed --add-binary  c:\Users\Kryvol\Anaconda3\envs\tensorflow\Lib\site-packages\cv2\opencv_ffmpeg342_64.dll;. MotionClipper.py
#
