from PySide6.QtWidgets import QMainWindow, QFileDialog

from ui_mainwindow import Ui_MainWindow
import pandas as pd
from Backend.wellpath import Wellpath


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, app):
        super().__init__()
        self.setupUi(self)
        self.app = app

        self.survey_data_upload_push_button.clicked.connect(self.read_survey_csv)

    def get_survey_data_path(self):
        self.path_to_survey_data, _ = QFileDialog.getOpenFileName()

    def read_survey_csv(self):
        self.get_survey_data_path()
        survey_df = pd.read_csv(self.path_to_survey_data)

        self.dev = Wellpath.wellpath_deviation(survey_df)


