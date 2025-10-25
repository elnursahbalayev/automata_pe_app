from Backend.wellpath import Wellpath
import pandas as pd
import streamlit as st
class WellpathUi:
    def __init__(self):
        pass

    def upload_ui(self,survey_file):
        self.survey_file = survey_file
        if self.survey_file is not None:
            survey_df = pd.read_csv(self.survey_file)
            # st.write(survey_df.head())

            dev = Wellpath.wellpath_deviation(survey_df)

            col1, col2 = st.columns(2)
            col1.plotly_chart(Wellpath.plot_wellpath(dev), use_container_width=True)

            col2.plotly_chart(Wellpath.plot_wellpath_3d(dev), use_container_width=True)