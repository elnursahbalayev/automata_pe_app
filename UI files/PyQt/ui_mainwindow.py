# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main.ui'
##
## Created by: Qt User Interface Compiler version 6.4.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QMetaObject, QRect,
                            QSize)
from PySide6.QtGui import (QFont, QIcon)
from PySide6.QtWidgets import (QFrame, QGridLayout, QLabel,
                               QLayout, QLineEdit, QMenuBar,
                               QPushButton, QRadioButton, QStatusBar,
                               QTabWidget, QTextBrowser, QVBoxLayout, QWidget)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1132, 868)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.layoutWidget = QWidget(self.centralwidget)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.layoutWidget.setGeometry(QRect(20, 30, 1101, 801))
        self.gridLayout = QGridLayout(self.layoutWidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.label_tools = QLabel(self.layoutWidget)
        self.label_tools.setObjectName(u"label_tools")

        self.gridLayout.addWidget(self.label_tools, 0, 0, 1, 1)

        self.line = QFrame(self.layoutWidget)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.gridLayout.addWidget(self.line, 1, 0, 1, 1)

        self.tab_results = QTabWidget(self.layoutWidget)
        self.tab_results.setObjectName(u"tab_results")
        self.tab_pdg = QWidget()
        self.tab_pdg.setObjectName(u"tab_pdg")
        self.button_upload_file = QPushButton(self.tab_pdg)
        self.button_upload_file.setObjectName(u"button_upload_file")
        self.button_upload_file.setGeometry(QRect(160, 320, 171, 91))
        font = QFont()
        font.setPointSize(19)
        self.button_upload_file.setFont(font)
        self.button_upload_file.setStyleSheet(u"color: rgb(255, 255, 255);\n"
"background-color: rgb(0, 189, 139);")
        self.tab_results.addTab(self.tab_pdg, "")
        self.tab_pvt = QWidget()
        self.tab_pvt.setObjectName(u"tab_pvt")
        self.tab_results.addTab(self.tab_pvt, "")
        self.tab_wt = QWidget()
        self.tab_wt.setObjectName(u"tab_wt")
        self.survey_data_upload_push_button = QPushButton(self.tab_wt)
        self.survey_data_upload_push_button.setObjectName(u"survey_data_upload_push_button")
        self.survey_data_upload_push_button.setGeometry(QRect(190, 10, 131, 24))
        self.tab_results.addTab(self.tab_wt, "")

        self.gridLayout.addWidget(self.tab_results, 1, 1, 4, 1)

        self.tab_tools = QTabWidget(self.layoutWidget)
        self.tab_tools.setObjectName(u"tab_tools")
        self.tab_tools_cleaning = QWidget()
        self.tab_tools_cleaning.setObjectName(u"tab_tools_cleaning")
        self.layoutWidget1 = QWidget(self.tab_tools_cleaning)
        self.layoutWidget1.setObjectName(u"layoutWidget1")
        self.layoutWidget1.setGeometry(QRect(0, 0, 130, 166))
        self.verticalLayout_2 = QVBoxLayout(self.layoutWidget1)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.label_data_cleaning = QLabel(self.layoutWidget1)
        self.label_data_cleaning.setObjectName(u"label_data_cleaning")

        self.verticalLayout_2.addWidget(self.label_data_cleaning)

        self.label_delete_null_values = QLabel(self.layoutWidget1)
        self.label_delete_null_values.setObjectName(u"label_delete_null_values")

        self.verticalLayout_2.addWidget(self.label_delete_null_values)

        self.radioButton_keep_null = QRadioButton(self.layoutWidget1)
        self.radioButton_keep_null.setObjectName(u"radioButton_keep_null")

        self.verticalLayout_2.addWidget(self.radioButton_keep_null)

        self.radioButton__remove_null = QRadioButton(self.layoutWidget1)
        self.radioButton__remove_null.setObjectName(u"radioButton__remove_null")

        self.verticalLayout_2.addWidget(self.radioButton__remove_null)

        self.label_delete_duplicate = QLabel(self.layoutWidget1)
        self.label_delete_duplicate.setObjectName(u"label_delete_duplicate")

        self.verticalLayout_2.addWidget(self.label_delete_duplicate)

        self.radioButton_keep_duplicate = QRadioButton(self.layoutWidget1)
        self.radioButton_keep_duplicate.setObjectName(u"radioButton_keep_duplicate")

        self.verticalLayout_2.addWidget(self.radioButton_keep_duplicate)

        self.radioButton_remove_duplicate = QRadioButton(self.layoutWidget1)
        self.radioButton_remove_duplicate.setObjectName(u"radioButton_remove_duplicate")

        self.verticalLayout_2.addWidget(self.radioButton_remove_duplicate)

        icon = QIcon()
        icon.addFile(u":/images/paintbrush.png", QSize(), QIcon.Normal, QIcon.Off)
        self.tab_tools.addTab(self.tab_tools_cleaning, icon, "")
        self.tab_tools_values = QWidget()
        self.tab_tools_values.setObjectName(u"tab_tools_values")
        self.label_gauge_details = QLabel(self.tab_tools_values)
        self.label_gauge_details.setObjectName(u"label_gauge_details")
        self.label_gauge_details.setGeometry(QRect(11, 21, 73, 16))
        self.label_gauge_depth = QLabel(self.tab_tools_values)
        self.label_gauge_depth.setObjectName(u"label_gauge_depth")
        self.label_gauge_depth.setGeometry(QRect(11, 43, 99, 16))
        self.lineEdit_depth2 = QLineEdit(self.tab_tools_values)
        self.lineEdit_depth2.setObjectName(u"lineEdit_depth2")
        self.lineEdit_depth2.setGeometry(QRect(11, 93, 133, 22))
        self.lineEdit_depth1 = QLineEdit(self.tab_tools_values)
        self.lineEdit_depth1.setObjectName(u"lineEdit_depth1")
        self.lineEdit_depth1.setGeometry(QRect(11, 65, 133, 22))
        icon1 = QIcon()
        icon1.addFile(u":/images/gear.png", QSize(), QIcon.Normal, QIcon.Off)
        self.tab_tools.addTab(self.tab_tools_values, icon1, "")

        self.gridLayout.addWidget(self.tab_tools, 2, 0, 1, 1)

        self.label_statistics = QLabel(self.layoutWidget)
        self.label_statistics.setObjectName(u"label_statistics")

        self.gridLayout.addWidget(self.label_statistics, 3, 0, 1, 1)

        self.text_statistics = QTextBrowser(self.layoutWidget)
        self.text_statistics.setObjectName(u"text_statistics")

        self.gridLayout.addWidget(self.text_statistics, 4, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1132, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.tab_results.setCurrentIndex(2)
        self.tab_tools.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label_tools.setText(QCoreApplication.translate("MainWindow", u"Tools", None))
        self.button_upload_file.setText(QCoreApplication.translate("MainWindow", u"Upload Files", None))
        self.tab_results.setTabText(self.tab_results.indexOf(self.tab_pdg), QCoreApplication.translate("MainWindow", u"PDG", None))
        self.tab_results.setTabText(self.tab_results.indexOf(self.tab_pvt), QCoreApplication.translate("MainWindow", u"PVT and Fractional flow", None))
        self.survey_data_upload_push_button.setText(QCoreApplication.translate("MainWindow", u"Upload Survey File", None))
        self.tab_results.setTabText(self.tab_results.indexOf(self.tab_wt), QCoreApplication.translate("MainWindow", u"Well trajectory", None))
        self.label_data_cleaning.setText(QCoreApplication.translate("MainWindow", u"Data Cleaning", None))
        self.label_delete_null_values.setText(QCoreApplication.translate("MainWindow", u"Delete null values?", None))
        self.radioButton_keep_null.setText(QCoreApplication.translate("MainWindow", u"Keep", None))
        self.radioButton__remove_null.setText(QCoreApplication.translate("MainWindow", u"Remove", None))
        self.label_delete_duplicate.setText(QCoreApplication.translate("MainWindow", u"Delete duplicate values?", None))
        self.radioButton_keep_duplicate.setText(QCoreApplication.translate("MainWindow", u"Keep", None))
        self.radioButton_remove_duplicate.setText(QCoreApplication.translate("MainWindow", u"Remove", None))
        self.tab_tools.setTabText(self.tab_tools.indexOf(self.tab_tools_cleaning), "")
        self.label_gauge_details.setText(QCoreApplication.translate("MainWindow", u"Gauge details", None))
        self.label_gauge_depth.setText(QCoreApplication.translate("MainWindow", u"Enter gauge depth", None))
        self.tab_tools.setTabText(self.tab_tools.indexOf(self.tab_tools_values), "")
        self.label_statistics.setText(QCoreApplication.translate("MainWindow", u"Statistics", None))
    # retranslateUi

