import streamlit as st
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from Backend.dataPreprocessing import dataprocess
from Backend.pdgProcessing import pdg_process
from Backend.pvtProcessing import pvt_process
from Streamlit.res_management_ui import Res_Management
from Streamlit.data_management_ui import DataManagementUi
from Streamlit.buildup_ui import BuildUpUi
from Streamlit.glo_ui import  GLOUi
from Streamlit.wellpath_ui import WellpathUi

import os
class PdaUi:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.wm_attributes('-topmost', 1)
        self.dp = dataprocess()
        self.pdg_proc = pdg_process()
        self.pvt = pvt_process()
        self.oripath = os.getcwd()
        self.resman = Res_Management()
        self.buildupui = BuildUpUi()
        self.gloui = GLOUi()
        self.wellpathui = WellpathUi()

        # if 'dirname' not in st.session_state:
        #     st.session_state.dirname=None
        #
        # if 'select_folder' not in st.session_state:
        #     st.session_state.select_folder = False

    def upload_ui(self, RF_data, buildup_pressure_df, buildup_rate_df, survey_file):

        self.pdg_tab, self.pvt_tab, self.gas_lift_tab, self.res_man_tab, self.well_trajectory_tab = st.tabs(['PDG data', 'Production data', 'Gas Lift Optimization', 'Reservoir Management', 'Well Trajectory'])

        with self.pdg_tab:
            # clicked = st.button('Select a folder')
            # if clicked or st.session_state.select_folder:
            #     st.session_state.select_folder = True
            #     if st.session_state.dirname is not None:
            #         self.dirname = st.session_state.dirname
            #     else:
            #         self.dirname = st.text_input('Selected folder', filedialog.askdirectory(master=self.root))
            #         st.session_state.dirname = self.dirname
            #     st.write(st.session_state.dirname)
                # self.dirname = st.session_state.dirname
                self.read_files()
                st.success('Uploaded successfully')
                self.choice_null = st.radio('Keep or remove null values', ('Keep', 'Remove'))
                self.choice_duplicate = st.radio('Keep or remove duplicate values', ('Keep', 'Remove'))

                statistics_tab, null_values_tab, pdg_distribution_tab, pressure_difference_tab, fluid_density_tab, buildup_recog_tab = st.tabs(['Statistics',
                                                                                                                             'Null Values',
                                                                                                                             'Pressure and Temperature Distribution',
                                                                                                                             'Pressure vs Time Difference',
                                                                                                                             'Fluid Density', 'Buildup Recognition'])
                with null_values_tab:
                    self.process_files()

                with statistics_tab:
                    self.show_statistics()

                with pdg_distribution_tab:
                    # self.plot_pdg_graphs()
                    st.plotly_chart(self.pdg_proc.visualize_well(self.PDG_df), use_container_width=True)

                with pressure_difference_tab:
                    st.plotly_chart(self.pdg_proc.plot_date_vs_pressure_difference(self.PDG_df),
                                    use_container_width=True)
                with fluid_density_tab:
                    self.fluid_density_calculation()

                with buildup_recog_tab:
                    self.buildupui.upload_ui(buildup_pressure_df, buildup_rate_df)

        with self.pvt_tab:
            # self.monthly_prod_data, self.pvt_data = None, None
            if (self.monthly_prod_data is not None and self.pvt_data is not None) and (self.monthly_prod_data !='No data uploaded' and self.pvt_data !='No data uploaded'):
                self.monthly_prod_data = pd.read_csv(self.monthly_prod_data)
                self.pvt_data = pd.read_csv(self.pvt_data)

                self.pvt_processing_and_calculation()

                calendar_rate_tab, technical_rate_tab, outliers_tab = st.tabs(['Calendar Rate', 'Technical Rate', 'Outliers detection'])

                with calendar_rate_tab:
                    st.plotly_chart(self.pvt.plot_fg_fw_fo_vs_time_calendar_rate(self.PVT_df),use_container_width=True)

                with technical_rate_tab:
                    st.plotly_chart(self.pvt.plot_production_data_technical_rate(self.PVT_df), use_container_width=True)

                with outliers_tab:
                    self.detect_outliers()

        with self.res_man_tab:
            self.resman.upload_ui(self.PVT_df, RF_data)

        with self.gas_lift_tab:
            self.gloui.upload_ui()
        
        with self.well_trajectory_tab:
            self.wellpathui.upload_ui(survey_file)




    # @st.cache
    def read_files(self):
        # reading file one by one and send number of files. Initiate number of txt and csv files in the folder
        os.chdir(self.dirname)
        list = os.listdir()
        print('There are {} files in the folder'.format(len(list)))
        list_csv = [x for x in list if 'CSV' in str.upper(x)]
        list_txt = [x for x in list if 'TXT' in str.upper(x)]
        self.PDG_df = pd.DataFrame()
        df_csv = pd.DataFrame()
        df_txt = pd.DataFrame()

        # from each list, each file will be processed and read. Each file with the same format will be combined into one singular dataframe
        if list_csv:
            print(list_csv)
            filetype = 'csv'
            print('--------------------Retrieving data from files----------------------------------------------')
            for i, file in enumerate(list_csv):
                filepath = os.path.join(self.dirname, file)
                df_temp = self.dp.build_dataframe(filepath, filetype)
                df_temp = self.dp.formatting(df_temp)
                # appending multiple dataframe every iteration into single list
                if df_csv.empty: df_csv = df_csv.append(df_temp, ignore_index=True)
                if not df_csv.empty:
                    if df_csv.columns.tolist() == df_temp.columns.tolist(): df_csv = df_csv.append(df_temp,
                                                                                                   ignore_index=True)
            print("Data processed")
            print('------------------------------------------------------------------------------------------------')

        if list_txt:
            print(list_txt)
            filetype = 'txt'
            print('--------------------Retrieving data from files--------------------------------------------------')
            for i, file in enumerate(list_txt):
                filepath = os.path.join(self.dirname, file)
                df_temp = self.dp.build_dataframe(filepath, filetype)
                df_temp = self.dp.formatting(df_temp)
                # appending multiple dataframe every iteration into single list
                if df_txt.empty: df_txt = df_txt.append(df_temp, ignore_index=True)
                if not df_txt.empty:
                    if df_txt.columns.tolist() == df_temp.columns.tolist(): df_txt = df_csv.append(df_temp,
                                                                                                   ignore_index=True)
            print("Data processed")
            print('------------------------------------------------------------------------------------------------')

        if not list_txt and not list_csv:
            # if no file are present
            print('No file processed')

            # combine the list of csv and txt if the format is the same
        if df_csv.columns.tolist() == df_txt.columns.tolist():
            PDG_df = df_csv.append(df_txt, ignore_index=True)
        else:
            try:
                if not df_csv.empty and self.PDG_df.empty and df_txt.empty: self.PDG_df = df_csv
                if not df_txt.empty and self.PDG_df.empty and df_csv.empty: self.PDG_df = df_txt
            except:
                print('Inconsistent data with different column values detected.')
                os._exit(0)
        os.chdir(self.oripath)

    
    def process_files(self):
        self.PDG_df = self.dp.sorting(self.PDG_df)
        # choice = input('Choose null value handling option.\n1) Remove\n2) Ignore.\nEither press 1 or 2: ').lower()
        # choice_duplicate = input(
        #     'Choose duplicate value handling option.\n1) Remove\n2) Ignore.\nEither press 1 or 2: ').lower()
        choice = '1'
        choice_duplicate = '1'


        self.PDG_df, fig1, fig2 = self.dp.datahandling(self.PDG_df, choice, choice_duplicate)
        if choice == '2':
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.plotly_chart(fig1, use_container_width=True)


        self.PDG_df = self.dp.dataresampling(self.PDG_df)
        self.num_gauge, gauge_type, _, _, _ = self.dp.num_gaugedetect(self.PDG_df)
        st.write("There are {} type of reading detected with {} number of gauge! ".format(len(gauge_type), len(self.num_gauge)))

    def show_statistics(self):
        if self.PDG_df is not None:
            st.write(self.PDG_df.describe().T)


    def fluid_density_calculation(self):
        dp_gauges = []
        for i in range(len(self.num_gauge)):
            try:
                dp = float(st.text_input(f'Enter depth for gauge {i+1}'))
                dp_gauges.append(dp)
            except:
                pass
        if len(dp_gauges) == len(self.num_gauge):
            self.PDG_df = self.pdg_proc.calculate_fluid_density(self.PDG_df, dp_gauges)
            st.plotly_chart(self.pdg_proc.plot_fluid_density(self.PDG_df), use_container_width=True)

    def pvt_processing_and_calculation(self):
        self.PVT_df = self.pvt.pvtPreprocessing(self.monthly_prod_data, self.PDG_df, self.pvt_data)
        self.PVT_df = self.pvt.calculate_total_day(self.PVT_df)  # total day for technical production calculation
        self.PVT_df = self.pvt.unitConversion(self.PVT_df)
        # calculaself.ting water, oil and gas technical production
        self.PVT_df = self.pvt.calculate_technical_production(self.PVT_df)
        # calculaself.ting gor and wc
        self.PVT_df = self.pvt.calculate_gor_technical(self.PVT_df)
        self.PVT_df = self.pvt.calculate_wc_technical(self.PVT_df)
        # calculaself.ting the fw, fg and fo and free gas
        self.PVT_df = self.pvt.calculate_free_gas_technical(self.PVT_df)
        self.PVT_df = self.pvt.calculate_fw_technical(self.PVT_df)
        self.PVT_df = self.pvt.calculate_fg_technical(self.PVT_df)
        self.PVT_df = self.pvt.calculate_fo_technical(self.PVT_df)

    def detect_outliers(self):
        self.PVT_df, fig = self.pvt.detect_outliers(self.PVT_df, column='fo')
        st.plotly_chart(fig)
        self.pvt.fit(self.PVT_df)
        self.PVT_df, fig_2, fig_res = self.pvt.predict(self.PVT_df)
        st.plotly_chart(fig_res, use_container_width=True)
        st.plotly_chart(fig_2, use_container_width=True)