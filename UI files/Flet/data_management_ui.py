import flet as ft
import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog

class DataManagementUi:
    def __init__(self, page):
        self.page = page
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.wm_attributes('-topmost', 1)
        self.dirname = None
        self.monthly_prod_data = None
        self.pvt_data = None
        self.survey_file = None
        self.rf_files = []
        self.buildup_pressure = None
        self.buildup_rate = None
        
    def upload_ui(self):
        # Create a container for the UI elements
        container = ft.Column(spacing=20)
        
        # Select folder button
        select_folder_btn = ft.ElevatedButton(
            text="Select a folder",
            on_click=self.select_folder
        )
        
        self.folder_path_text = ft.Text("No folder selected")
        
        # File upload buttons with descriptions
        monthly_prod_upload = ft.Column([
            ft.ElevatedButton(
                "Upload monthly production data",
                on_click=lambda e: self.upload_file(e, "monthly_prod")
            ),
            ft.Text("Required columns: (Well, NPDCode, On Stream, Oil, Gas, Water, Date)", size=12)
        ])
        
        pvt_upload = ft.Column([
            ft.ElevatedButton(
                "Upload PVT data",
                on_click=lambda e: self.upload_file(e, "pvt")
            ),
            ft.Text("Required columns: (Pressure, Rs, Bo, Bw, Bg)", size=12)
        ])
        
        survey_upload = ft.Column([
            ft.ElevatedButton(
                "Upload survey data",
                on_click=lambda e: self.upload_file(e, "survey")
            ),
            ft.Text("Required columns: (MD, INC, AZI, TVD, X-offset, Y-offset, Z-offset, NPDCode, Well, Date)", size=12)
        ])
        
        rf_upload = ft.Column([
            ft.ElevatedButton(
                "Upload Recovery Factor Benchmark data",
                on_click=lambda e: self.upload_file(e, "rf")
            ),
            ft.Text("Required columns: (Sand, Fault Block, Block Thickness (ft), Sw (%), poro (%) OOIP (MMSTB), Well, Latitude, Longitude, Perforation Length (ft), perforation permeability (md), perf NTG, EUR (MMSTB))", size=12)
        ])
        
        buildup_pressure_upload = ft.Column([
            ft.ElevatedButton(
                "Upload Pressure data for Buildup recognition",
                on_click=lambda e: self.upload_file(e, "buildup_pressure")
            ),
            ft.Text("Required columns: (Pressure)", size=12)
        ])
        
        buildup_rate_upload = ft.Column([
            ft.ElevatedButton(
                "Upload Rate data for Buildup recognition",
                on_click=lambda e: self.upload_file(e, "buildup_rate")
            ),
            ft.Text("Required columns: (Rate)", size=12)
        ])
        
        # Status text for file uploads
        self.monthly_prod_status = ft.Text("Not uploaded")
        self.pvt_status = ft.Text("Not uploaded")
        self.survey_status = ft.Text("Not uploaded")
        self.rf_status = ft.Text("Not uploaded")
        self.buildup_pressure_status = ft.Text("Not uploaded")
        self.buildup_rate_status = ft.Text("Not uploaded")
        
        # Add all elements to the container
        container.controls.extend([
            select_folder_btn,
            self.folder_path_text,
            ft.Divider(),
            ft.Row([monthly_prod_upload, self.monthly_prod_status]),
            ft.Row([pvt_upload, self.pvt_status]),
            ft.Row([survey_upload, self.survey_status]),
            ft.Row([rf_upload, self.rf_status]),
            ft.Row([buildup_pressure_upload, self.buildup_pressure_status]),
            ft.Row([buildup_rate_upload, self.buildup_rate_status]),
        ])
        
        return container
    
    def select_folder(self, e):
        self.dirname = filedialog.askdirectory(master=self.root)
        if self.dirname:
            self.folder_path_text.value = f"Selected folder: {self.dirname}"
            self.page.update()
    
    def upload_file(self, e, file_type):
        file_path = filedialog.askopenfilename(master=self.root)
        if not file_path:
            return
            
        if file_type == "monthly_prod":
            self.monthly_prod_data = file_path
            self.monthly_prod_status.value = f"Uploaded: {os.path.basename(file_path)}"
        elif file_type == "pvt":
            self.pvt_data = file_path
            self.pvt_status.value = f"Uploaded: {os.path.basename(file_path)}"
        elif file_type == "survey":
            self.survey_file = file_path
            self.survey_status.value = f"Uploaded: {os.path.basename(file_path)}"
        elif file_type == "rf":
            self.rf_files.append(file_path)
            self.rf_status.value = f"Uploaded {len(self.rf_files)} files"
        elif file_type == "buildup_pressure":
            self.buildup_pressure = file_path
            self.buildup_pressure_status.value = f"Uploaded: {os.path.basename(file_path)}"
        elif file_type == "buildup_rate":
            self.buildup_rate = file_path
            self.buildup_rate_status.value = f"Uploaded: {os.path.basename(file_path)}"
            
        self.page.update()
    
    def return_data(self):
        if self.monthly_prod_data is not None and self.pvt_data is not None and self.dirname is not None:
            return self.monthly_prod_data, self.pvt_data, self.dirname
        else:
            return 'No data uploaded', 'No data uploaded', None
    
    def return_survey_data(self):
        if self.survey_file is not None:
            return self.survey_file
        return None
    
    def return_rf_data(self):
        if self.rf_files:
            return self.rf_files
        return None
    
    def return_buildup_data(self):
        if self.buildup_pressure is not None and self.buildup_rate is not None:
            try:
                buildup_pressure_df = pd.read_csv(self.buildup_pressure, sep='\t', on_bad_lines='skip')
                buildup_rate_df = pd.read_csv(self.buildup_rate, sep='\t', on_bad_lines='skip')
                return buildup_pressure_df, buildup_rate_df
            except Exception as e:
                print(f"Error reading buildup data: {e}")
        
        return None, None