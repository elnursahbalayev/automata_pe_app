from Backend.buildup_recognition import BuildUp
import pandas as pd
import streamlit as st


class BuildUpUi:
    def __init__(self):
        self.buildup = BuildUp()

    def upload_ui(self, buildup_pressure_df, buildup_rate_df):
        self.buildup.read_data(buildup_pressure_df, buildup_rate_df)
        self.buildup.format_data()

        st.plotly_chart(self.buildup.plot_pressure_rate(), use_container_width=True)
        self.buildup.plot_pressure_der_rate()
        self.buildup.plot_pressure_der_cleaned_rate()
        st.plotly_chart(self.buildup.plot_buildup_zones(), use_container_width=True)