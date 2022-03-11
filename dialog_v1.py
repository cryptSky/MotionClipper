# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dialog.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class ClipDialog(object):
    def __init__(self):
               
        self.min_area = 500
        self.alpha = 0.2
        self.threshold = (32, 255)
        self.width = 1000
        self.show_detection = False
        
    def setupUi(self, Dialog):
        Dialog.setObjectName("Clipper parameters")
        Dialog.resize(600, 400)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)
        self.verticalLayoutWidget = QtWidgets.QWidget(Dialog)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 570, 370))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox = QtWidgets.QGroupBox(self.verticalLayoutWidget)
        self.groupBox.setObjectName("groupBox")
        self.gridLayoutWidget = QtWidgets.QWidget(self.groupBox)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(9, 19, 501, 231))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.alphaEdit = QtWidgets.QLineEdit(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.alphaEdit.sizePolicy().hasHeightForWidth())
        self.alphaEdit.setSizePolicy(sizePolicy)
        self.alphaEdit.setObjectName("alphaEdit")
        self.gridLayout.addWidget(self.alphaEdit, 1, 1, 1, 1)
        self.minThreshEdit = QtWidgets.QLineEdit(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.minThreshEdit.sizePolicy().hasHeightForWidth())
        self.minThreshEdit.setSizePolicy(sizePolicy)
        self.minThreshEdit.setObjectName("minThreshEdit")
        self.gridLayout.addWidget(self.minThreshEdit, 3, 1, 1, 1)
        self.alphaLabel = QtWidgets.QLabel(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.alphaLabel.sizePolicy().hasHeightForWidth())
        self.alphaLabel.setSizePolicy(sizePolicy)
        self.alphaLabel.setObjectName("alphaLabel")
        self.gridLayout.addWidget(self.alphaLabel, 1, 0, 1, 1)
        self.maxThreshLabel = QtWidgets.QLabel(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.maxThreshLabel.sizePolicy().hasHeightForWidth())
        self.maxThreshLabel.setSizePolicy(sizePolicy)
        self.maxThreshLabel.setObjectName("maxThreshLabel")
        self.gridLayout.addWidget(self.maxThreshLabel, 4, 0, 1, 1)
        self.minThreshLabel = QtWidgets.QLabel(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.minThreshLabel.sizePolicy().hasHeightForWidth())
        self.minThreshLabel.setSizePolicy(sizePolicy)
        self.minThreshLabel.setObjectName("minThreshLabel")
        self.gridLayout.addWidget(self.minThreshLabel, 3, 0, 1, 1)
        self.minAreaLabel = QtWidgets.QLabel(self.gridLayoutWidget)
        self.minAreaLabel.setObjectName("minAreaLabel")
        self.gridLayout.addWidget(self.minAreaLabel, 0, 0, 1, 1)
        self.processingWidthLabel = QtWidgets.QLabel(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.processingWidthLabel.sizePolicy().hasHeightForWidth())
        self.processingWidthLabel.setSizePolicy(sizePolicy)
        self.processingWidthLabel.setObjectName("processingWidthLabel")
        self.gridLayout.addWidget(self.processingWidthLabel, 5, 0, 1, 1)
        self.minAreaEdit = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.minAreaEdit.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.minAreaEdit.sizePolicy().hasHeightForWidth())
        self.minAreaEdit.setSizePolicy(sizePolicy)
        self.minAreaEdit.setObjectName("minAreaEdit")
        self.gridLayout.addWidget(self.minAreaEdit, 0, 1, 1, 1)
        self.maxThreshEdit = QtWidgets.QLineEdit(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.maxThreshEdit.sizePolicy().hasHeightForWidth())
        self.maxThreshEdit.setSizePolicy(sizePolicy)
        self.maxThreshEdit.setObjectName("maxThreshEdit")
        self.gridLayout.addWidget(self.maxThreshEdit, 4, 1, 1, 1)
        self.processingWidthEdit = QtWidgets.QLineEdit(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.processingWidthEdit.sizePolicy().hasHeightForWidth())
        self.processingWidthEdit.setSizePolicy(sizePolicy)
        self.processingWidthEdit.setObjectName("processingWidthEdit")
        self.gridLayout.addWidget(self.processingWidthEdit, 5, 1, 1, 1)
        self.showVideoCheckBox = QtWidgets.QCheckBox(self.groupBox)
        self.showVideoCheckBox.setGeometry(QtCore.QRect(10, 270, 391, 20))
        self.showVideoCheckBox.setObjectName("showVideoCheckBox")
        self.startButton = QtWidgets.QPushButton(self.groupBox)
        self.startButton.setGeometry(QtCore.QRect(10, 310, 93, 31))
        self.startButton.setObjectName("startButton")
        self.progressBar = QtWidgets.QProgressBar(self.groupBox)
        self.progressBar.setGeometry(QtCore.QRect(110, 310, 401, 31))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout.addWidget(self.groupBox)
        
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.groupBox.setTitle(_translate("Dialog", "Set Parameters"))
        self.alphaEdit.setText(_translate("Dialog", str(self.alpha)))
        self.minThreshEdit.setText(_translate("Dialog", str(self.threshold[0])))
        self.alphaLabel.setText(_translate("Dialog", "       Alpha:"))
        self.maxThreshLabel.setText(_translate("Dialog", "       Max threshold:"))
        self.minThreshLabel.setText(_translate("Dialog", "       Min threshold:"))
        self.minAreaLabel.setText(_translate("Dialog", "      Minimum area:     "))
        self.processingWidthLabel.setText(_translate("Dialog", "       Processing width"))
        self.minAreaEdit.setText(_translate("Dialog", str(self.min_area)))
        self.maxThreshEdit.setText(_translate("Dialog", str(self.threshold[1])))
        self.processingWidthEdit.setText(_translate("Dialog", str(self.width)))
        self.showVideoCheckBox.setText(_translate("Dialog", "Show detection video (processing will be slower)"))
        self.showVideoCheckBox.setChecked(self.show_detection)
        self.startButton.setText(_translate("Dialog", "Start"))
        
    def setParams(self):
        
        self.min_area = int(self.minAreaEdit.getText())
        self.alpha = float(self.alphaEdit.getText())
        minThresh = int(self.minThreshEdit.getText())
        maxThresh = int(self.maxThreshEdit.getText())
        self.threshold = (minThresh, maxThresh)
        self.width = int(self.processingWidthEdit.getText())
        self.show_detection = self.showVideoCheckBox.getChecked()
        
        #self.accept()
        
        

