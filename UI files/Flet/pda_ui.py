import flet as ft
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import os
import plotly.io as pio
from Backend.dataPreprocessing import dataprocess
from Backend.pdgProcessing import pdg_process
from Backend.pvtProcessing import pvt_process
import sys
import importlib.util

# Import modules from UI files/Flet directory
def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import the required modules
res_management_ui = import_from_path("res_management_ui", os.path.join("UI files", "Flet", "res_management_ui.py"))
buildup_ui = import_from_path("buildup_ui", os.path.join("UI files", "Flet", "buildup_ui.py"))
glo_ui = import_from_path("glo_ui", os.path.join("UI files", "Flet", "glo_ui.py"))
wellpath_ui = import_from_path("wellpath_ui", os.path.join("UI files", "Flet", "wellpath_ui.py"))

class PdaUi:
    def __init__(self, page):
        self.page = page
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.wm_attributes('-topmost', 1)
        self.dp = dataprocess()
        self.pdg_proc = pdg_process()
        self.pvt = pvt_process()
        self.oripath = os.getcwd()
        self.resman = res_management_ui.Res_Management(page)
        self.buildupui = buildup_ui.BuildUpUi(page)
        self.gloui = glo_ui.GLOUi(page)
        self.wellpathui = wellpath_ui.WellpathUi(page)
        self.PDG_df = None
        self.PVT_df = None
        self.monthly_prod_data = None
        self.pvt_data = None
        self.dirname = None

    def upload_ui(self, RF_data, buildup_pressure_df, buildup_rate_df, survey_file):
        # Create a container for the UI elements
        container = ft.Column(spacing=20)
        
        # Create tabs
        tabs = ft.Tabs(
            selected_index=0,
            animation_duration=300,
            tabs=[
                ft.Tab(
                    text="PDG data",
                    content=self.create_pdg_tab(buildup_pressure_df, buildup_rate_df)
                ),
                ft.Tab(
                    text="Production data",
                    content=self.create_pvt_tab()
                ),
                ft.Tab(
                    text="Gas Lift Optimization",
                    content=self.gloui.upload_ui()
                ),
                ft.Tab(
                    text="Reservoir Management",
                    content=self.resman.upload_ui(self.PVT_df, RF_data)
                ),
                ft.Tab(
                    text="Well Trajectory",
                    content=self.wellpathui.upload_ui(survey_file)
                ),
            ],
            expand=1
        )
        
        container.controls.append(tabs)
        return container
    
    def create_pdg_tab(self, buildup_pressure_df, buildup_rate_df):
        # Create a container for the PDG tab
        pdg_container = ft.Column(spacing=20)
        
        # Select folder button
        select_folder_btn = ft.ElevatedButton(
            text="Select a folder",
            on_click=self.select_folder
        )
        
        self.folder_path_text = ft.Text("No folder selected")
        
        # Add folder selection controls
        pdg_container.controls.extend([
            select_folder_btn,
            self.folder_path_text,
            ft.Divider()
        ])
        
        # Create sub-tabs for PDG data
        pdg_subtabs = ft.Tabs(
            selected_index=0,
            animation_duration=300,
            tabs=[
                ft.Tab(
                    text="Statistics",
                    content=self.create_statistics_tab()
                ),
                ft.Tab(
                    text="Null Values",
                    content=self.create_null_values_tab()
                ),
                ft.Tab(
                    text="Pressure and Temperature Distribution",
                    content=self.create_pdg_distribution_tab()
                ),
                ft.Tab(
                    text="Pressure vs Time Difference",
                    content=self.create_pressure_difference_tab()
                ),
                ft.Tab(
                    text="Fluid Density",
                    content=self.create_fluid_density_tab()
                ),
                ft.Tab(
                    text="Buildup Recognition",
                    content=self.buildupui.upload_ui(buildup_pressure_df, buildup_rate_df)
                ),
            ],
            expand=1
        )
        
        pdg_container.controls.append(pdg_subtabs)
        return pdg_container
    
    def create_statistics_tab(self):
        # Create a container for the statistics tab
        stats_container = ft.Column(spacing=10)
        
        if self.PDG_df is not None:
            # Convert DataFrame describe to a readable format
            stats_text = ft.Text("Statistics will be shown here after data is loaded")
            try:
                stats_df = self.PDG_df.describe().T
                stats_text = ft.DataTable(
                    columns=[
                        ft.DataColumn(ft.Text("Statistic")),
                        ft.DataColumn(ft.Text("count")),
                        ft.DataColumn(ft.Text("mean")),
                        ft.DataColumn(ft.Text("std")),
                        ft.DataColumn(ft.Text("min")),
                        ft.DataColumn(ft.Text("25%")),
                        ft.DataColumn(ft.Text("50%")),
                        ft.DataColumn(ft.Text("75%")),
                        ft.DataColumn(ft.Text("max")),
                    ],
                    rows=[
                        ft.DataRow(
                            cells=[
                                ft.DataCell(ft.Text(index)),
                                ft.DataCell(ft.Text(f"{row['count']:.2f}")),
                                ft.DataCell(ft.Text(f"{row['mean']:.2f}")),
                                ft.DataCell(ft.Text(f"{row['std']:.2f}")),
                                ft.DataCell(ft.Text(f"{row['min']:.2f}")),
                                ft.DataCell(ft.Text(f"{row['25%']:.2f}")),
                                ft.DataCell(ft.Text(f"{row['50%']:.2f}")),
                                ft.DataCell(ft.Text(f"{row['75%']:.2f}")),
                                ft.DataCell(ft.Text(f"{row['max']:.2f}")),
                            ]
                        ) for index, row in stats_df.iterrows()
                    ]
                )
            except:
                pass
            
            stats_container.controls.append(stats_text)
        else:
            stats_container.controls.append(ft.Text("No data loaded yet"))
        
        return stats_container
    
    def create_null_values_tab(self):
        # Create a container for the null values tab
        null_container = ft.Column(spacing=10)
        
        # Radio buttons for null values handling
        null_radio = ft.RadioGroup(
            content=ft.Column([
                ft.Radio(value="Keep", label="Keep null values"),
                ft.Radio(value="Remove", label="Remove null values"),
            ]),
            on_change=self.handle_null_change
        )
        
        # Radio buttons for duplicate values handling
        duplicate_radio = ft.RadioGroup(
            content=ft.Column([
                ft.Radio(value="Keep", label="Keep duplicate values"),
                ft.Radio(value="Remove", label="Remove duplicate values"),
            ]),
            on_change=self.handle_duplicate_change
        )
        
        null_container.controls.extend([
            ft.Text("Null values handling:"),
            null_radio,
            ft.Text("Duplicate values handling:"),
            duplicate_radio,
            ft.Divider(),
        ])
        
        # Add a placeholder for the null values plot
        self.null_plot_view = ft.Container(
            content=ft.Text("Process data to see null values plot"),
            width=800,
            height=500
        )
        null_container.controls.append(self.null_plot_view)
        
        # Add a button to process the files
        process_btn = ft.ElevatedButton(
            text="Process Files",
            on_click=self.process_files_click
        )
        null_container.controls.append(process_btn)
        
        # Add a text field to display gauge information
        self.gauge_info_text = ft.Text("")
        null_container.controls.append(self.gauge_info_text)
        
        return null_container
    
    def create_pdg_distribution_tab(self):
        # Create a container for the PDG distribution tab
        dist_container = ft.Column(spacing=10)
        
        if self.PDG_df is not None:
            try:
                # Create the PDG distribution plot
                fig = self.pdg_proc.visualize_well(self.PDG_df)
                html = pio.to_html(fig, full_html=False)
                plot_view = ft.WebView(
                    html=html,
                    width=800,
                    height=500
                )
                dist_container.controls.append(plot_view)
            except Exception as e:
                dist_container.controls.append(ft.Text(f"Error creating plot: {str(e)}"))
        else:
            dist_container.controls.append(ft.Text("No data loaded yet"))
        
        return dist_container
    
    def create_pressure_difference_tab(self):
        # Create a container for the pressure difference tab
        diff_container = ft.Column(spacing=10)
        
        if self.PDG_df is not None:
            try:
                # Create the pressure difference plot
                fig = self.pdg_proc.plot_date_vs_pressure_difference(self.PDG_df)
                html = pio.to_html(fig, full_html=False)
                plot_view = ft.WebView(
                    html=html,
                    width=800,
                    height=500
                )
                diff_container.controls.append(plot_view)
            except Exception as e:
                diff_container.controls.append(ft.Text(f"Error creating plot: {str(e)}"))
        else:
            diff_container.controls.append(ft.Text("No data loaded yet"))
        
        return diff_container
    
    def create_fluid_density_tab(self):
        # Create a container for the fluid density tab
        density_container = ft.Column(spacing=10)
        
        if self.PDG_df is not None and hasattr(self, 'num_gauge'):
            # Create input fields for gauge depths
            gauge_inputs = []
            for i in range(len(self.num_gauge)):
                gauge_input = ft.TextField(
                    label=f"Enter depth for gauge {i+1}",
                    keyboard_type=ft.KeyboardType.NUMBER,
                    on_change=lambda e, idx=i: self.update_gauge_depth(e, idx)
                )
                gauge_inputs.append(gauge_input)
            
            density_container.controls.extend(gauge_inputs)
            
            # Add a button to calculate fluid density
            calc_btn = ft.ElevatedButton(
                text="Calculate Fluid Density",
                on_click=self.calculate_fluid_density_click
            )
            density_container.controls.append(calc_btn)
            
            # Add a placeholder for the fluid density plot
            self.density_plot_view = ft.Container(
                content=ft.Text("Calculate fluid density to see plot"),
                width=800,
                height=500
            )
            density_container.controls.append(self.density_plot_view)
        else:
            density_container.controls.append(ft.Text("No data loaded yet or gauges not detected"))
        
        return density_container
    
    def create_pvt_tab(self):
        # Create a container for the PVT tab
        pvt_container = ft.Column(spacing=20)
        
        if (self.monthly_prod_data is not None and self.pvt_data is not None) and (self.monthly_prod_data != 'No data uploaded' and self.pvt_data != 'No data uploaded'):
            try:
                # Process PVT data
                self.pvt_processing_and_calculation()
                
                # Create sub-tabs for PVT data
                pvt_subtabs = ft.Tabs(
                    selected_index=0,
                    animation_duration=300,
                    tabs=[
                        ft.Tab(
                            text="Calendar Rate",
                            content=self.create_calendar_rate_tab()
                        ),
                        ft.Tab(
                            text="Technical Rate",
                            content=self.create_technical_rate_tab()
                        ),
                        ft.Tab(
                            text="Outliers detection",
                            content=self.create_outliers_tab()
                        ),
                    ],
                    expand=1
                )
                
                pvt_container.controls.append(pvt_subtabs)
            except Exception as e:
                pvt_container.controls.append(ft.Text(f"Error processing PVT data: {str(e)}"))
        else:
            pvt_container.controls.append(ft.Text("No PVT data uploaded or data is incomplete"))
        
        return pvt_container
    
    def create_calendar_rate_tab(self):
        # Create a container for the calendar rate tab
        calendar_container = ft.Column(spacing=10)
        
        if self.PVT_df is not None:
            try:
                # Create the calendar rate plot
                fig = self.pvt.plot_fg_fw_fo_vs_time_calendar_rate(self.PVT_df)
                html = pio.to_html(fig, full_html=False)
                plot_view = ft.WebView(
                    html=html,
                    width=800,
                    height=500
                )
                calendar_container.controls.append(plot_view)
            except Exception as e:
                calendar_container.controls.append(ft.Text(f"Error creating plot: {str(e)}"))
        else:
            calendar_container.controls.append(ft.Text("No PVT data processed yet"))
        
        return calendar_container
    
    def create_technical_rate_tab(self):
        # Create a container for the technical rate tab
        technical_container = ft.Column(spacing=10)
        
        if self.PVT_df is not None:
            try:
                # Create the technical rate plot
                fig = self.pvt.plot_production_data_technical_rate(self.PVT_df)
                html = pio.to_html(fig, full_html=False)
                plot_view = ft.WebView(
                    html=html,
                    width=800,
                    height=500
                )
                technical_container.controls.append(plot_view)
            except Exception as e:
                technical_container.controls.append(ft.Text(f"Error creating plot: {str(e)}"))
        else:
            technical_container.controls.append(ft.Text("No PVT data processed yet"))
        
        return technical_container
    
    def create_outliers_tab(self):
        # Create a container for the outliers tab
        outliers_container = ft.Column(spacing=10)
        
        if self.PVT_df is not None:
            try:
                # Detect outliers and create plots
                self.PVT_df, fig = self.pvt.detect_outliers(self.PVT_df, column='fo')
                html1 = pio.to_html(fig, full_html=False)
                plot_view1 = ft.WebView(
                    html=html1,
                    width=800,
                    height=400
                )
                
                self.pvt.fit(self.PVT_df)
                self.PVT_df, fig_2, fig_res = self.pvt.predict(self.PVT_df)
                
                html2 = pio.to_html(fig_res, full_html=False)
                plot_view2 = ft.WebView(
                    html=html2,
                    width=800,
                    height=400
                )
                
                html3 = pio.to_html(fig_2, full_html=False)
                plot_view3 = ft.WebView(
                    html=html3,
                    width=800,
                    height=400
                )
                
                outliers_container.controls.extend([
                    ft.Text("Outliers Detection", size=16, weight=ft.FontWeight.BOLD),
                    plot_view1,
                    ft.Divider(),
                    ft.Text("Prediction Results", size=16, weight=ft.FontWeight.BOLD),
                    plot_view2,
                    ft.Divider(),
                    ft.Text("Prediction Plot", size=16, weight=ft.FontWeight.BOLD),
                    plot_view3
                ])
            except Exception as e:
                outliers_container.controls.append(ft.Text(f"Error detecting outliers: {str(e)}"))
        else:
            outliers_container.controls.append(ft.Text("No PVT data processed yet"))
        
        return outliers_container
    
    def select_folder(self, e):
        self.dirname = filedialog.askdirectory(master=self.root)
        if self.dirname:
            self.folder_path_text.value = f"Selected folder: {self.dirname}"
            self.page.update()
            self.read_files()
            # Update UI to show success message
            self.folder_path_text.value = f"Selected folder: {self.dirname} (Uploaded successfully)"
            self.page.update()
    
    def handle_null_change(self, e):
        self.choice_null = e.control.value
        self.page.update()
    
    def handle_duplicate_change(self, e):
        self.choice_duplicate = e.control.value
        self.page.update()
    
    def process_files_click(self, e):
        if self.PDG_df is not None:
            self.PDG_df = self.dp.sorting(self.PDG_df)
            
            # Convert radio button values to dataprocessing options
            choice = '1' if self.choice_null == 'Remove' else '2'
            choice_duplicate = '1' if self.choice_duplicate == 'Remove' else '2'
            
            self.PDG_df, fig1, fig2 = self.dp.datahandling(self.PDG_df, choice, choice_duplicate)
            
            # Display the appropriate figure based on the choice
            if choice == '2':
                html = pio.to_html(fig2, full_html=False)
            else:
                html = pio.to_html(fig1, full_html=False)
                
            self.null_plot_view.content = ft.WebView(
                html=html,
                width=800,
                height=500
            )
            
            self.PDG_df = self.dp.dataresampling(self.PDG_df)
            self.num_gauge, gauge_type, _, _, _ = self.dp.num_gaugedetect(self.PDG_df)
            
            self.gauge_info_text.value = f"There are {len(gauge_type)} type of reading detected with {len(self.num_gauge)} number of gauge!"
            self.page.update()
    
    def update_gauge_depth(self, e, idx):
        if not hasattr(self, 'dp_gauges'):
            self.dp_gauges = [None] * len(self.num_gauge)
        
        try:
            self.dp_gauges[idx] = float(e.control.value)
        except:
            pass
        
        self.page.update()
    
    def calculate_fluid_density_click(self, e):
        if hasattr(self, 'dp_gauges') and all(dp is not None for dp in self.dp_gauges):
            self.PDG_df = self.pdg_proc.calculate_fluid_density(self.PDG_df, self.dp_gauges)
            
            # Create the fluid density plot
            fig = self.pdg_proc.plot_fluid_density(self.PDG_df)
            html = pio.to_html(fig, full_html=False)
            
            self.density_plot_view.content = ft.WebView(
                html=html,
                width=800,
                height=500
            )
            self.page.update()
    
    def read_files(self):
        # reading file one by one and send number of files. Initiate number of txt and csv files in the folder
        os.chdir(self.dirname)
        list_files = os.listdir()
        print('There are {} files in the folder'.format(len(list_files)))
        list_csv = [x for x in list_files if 'CSV' in str.upper(x)]
        list_txt = [x for x in list_files if 'TXT' in str.upper(x)]
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
                if df_csv.empty: df_csv = pd.concat([df_csv, df_temp], ignore_index=True)
                if not df_csv.empty:
                    if df_csv.columns.tolist() == df_temp.columns.tolist(): df_csv = pd.concat([df_csv, df_temp],
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
                if df_txt.empty: df_txt = pd.concat([df_txt, df_temp], ignore_index=True)
                if not df_txt.empty:
                    if df_txt.columns.tolist() == df_temp.columns.tolist(): df_txt = pd.concat([df_csv, df_temp],
                                                                                               ignore_index=True)
            print("Data processed")
            print('------------------------------------------------------------------------------------------------')

        if not list_txt and not list_csv:
            # if no file are present
            print('No file processed')

            # combine the list of csv and txt if the format is the same
        if df_csv.columns.tolist() == df_txt.columns.tolist():
            self.PDG_df = pd.concat([df_csv, df_txt], ignore_index=True)
        else:
            try:
                if not df_csv.empty and self.PDG_df.empty and df_txt.empty: self.PDG_df = df_csv
                if not df_txt.empty and self.PDG_df.empty and df_csv.empty: self.PDG_df = df_txt
            except:
                print('Inconsistent data with different column values detected.')
                os._exit(0)
        os.chdir(self.oripath)
    
    def pvt_processing_and_calculation(self):
        try:
            # Read the data files
            monthly_prod_df = pd.read_csv(self.monthly_prod_data)
            pvt_df = pd.read_csv(self.pvt_data)
            
            self.PVT_df = self.pvt.pvtPreprocessing(monthly_prod_df, self.PDG_df, pvt_df)
            self.PVT_df = self.pvt.calculate_total_day(self.PVT_df)  # total day for technical production calculation
            self.PVT_df = self.pvt.unitConversion(self.PVT_df)
            # calculating water, oil and gas technical production
            self.PVT_df = self.pvt.calculate_technical_production(self.PVT_df)
            # calculating gor and wc
            self.PVT_df = self.pvt.calculate_gor_technical(self.PVT_df)
            self.PVT_df = self.pvt.calculate_wc_technical(self.PVT_df)
            # calculating the fw, fg and fo and free gas
            self.PVT_df = self.pvt.calculate_free_gas_technical(self.PVT_df)
            self.PVT_df = self.pvt.calculate_fw_technical(self.PVT_df)
            self.PVT_df = self.pvt.calculate_fg_technical(self.PVT_df)
            self.PVT_df = self.pvt.calculate_fo_technical(self.PVT_df)
        except Exception as e:
            print(f"Error in PVT processing: {e}")
            self.PVT_df = None