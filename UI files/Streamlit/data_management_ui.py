import streamlit as st
import tkinter as tk
from tkinter import filedialog
import pandas as pd

class DataManagementUi:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.wm_attributes('-topmost', 1)

    def upload_ui(self):
        if 'dirname' not in st.session_state:
            st.session_state.dirname = None

        if 'select_folder' not in st.session_state:
            st.session_state.select_folder = False

        clicked = st.button('Select a folder')
        if clicked or st.session_state.select_folder:
            st.session_state.select_folder = True
            if st.session_state.dirname is not None:
                self.dirname = st.session_state.dirname
            else:
                self.dirname = st.text_input('Selected folder', filedialog.askdirectory(master=self.root))
                st.session_state.dirname = self.dirname

        self.monthly_prod_data = st.file_uploader('Upload monthly production data')
        st.write('Required columns: (Well, NPDCode, On Stream, Oil, Gas, Water, Date)')
        self.pvt_data =st.file_uploader('Upload PVT data')
        st.write('Required columns: (Pressure, Rs, Bo, Bw, Bg)')
        self.survey_file = st.file_uploader('Upload survey data')
        st.write('Required columns: (MD, INC, AZI, TVD, X-offset, Y-offset, Z-offset, NPDCode, Well, Date)')
        self.rf_files = st.file_uploader('Upload Recovery Factor Benchmark data', accept_multiple_files=True)
        st.write('Required columns: (Sand, Fault Block, Block Thickness (ft), Sw (%), poro (%) OOIP (MMSTB), Well, Latitude, Longitude, Perforation Length (ft), perforation permeability (md), perf NTG, EUR (MMSTB))')
        self.buildup_pressure = st.file_uploader('Upload Pressure data for Buildup recognition')
        st.write('Required columns: (Pressure)')
        self.buildup_rate = st.file_uploader('Upload Rate data for Buildup recognition')
        st.write('Required columns: (Rate)')

    
    def return_data(self):
        if self.monthly_prod_data is not None and self.pvt_data is not None and self.dirname is not None:
            return self.monthly_prod_data, self.pvt_data, self.dirname
        else:
            st.warning('No data uploaded')
            return 'No data uploaded', 'No data uploaded', None

    def return_survey_data(self):
        if self.survey_file is not None:
            return self.survey_file

    def return_rf_data(self):
        if self.rf_files is not None:
            return self.rf_files
        
    def return_buildup_data(self):
        if self.buildup_pressure is not None and self.buildup_rate is not None:
            self.buildup_pressure = pd.read_csv(self.buildup_pressure,sep='\t', on_bad_lines='skip')
            self.buildup_rate = pd.read_csv(self.buildup_rate,sep='\t', on_bad_lines='skip')

            return self.buildup_pressure, self.buildup_rate
        
        else:
            return None, None