from Backend.wellpath import Wellpath
import pandas as pd
import flet as ft
import plotly.io as pio

class WellpathUi:
    def __init__(self, page):
        self.page = page
        
    def upload_ui(self, survey_file):
        # Create a container for the UI elements
        container = ft.Column(spacing=20)
        
        if survey_file is None:
            container.controls.append(ft.Text("No survey file uploaded"))
            return container
            
        # Process the survey data
        try:
            survey_df = pd.read_csv(survey_file)
            dev = Wellpath.wellpath_deviation(survey_df)
            
            # Create 2D wellpath plot
            wellpath_2d_fig = Wellpath.plot_wellpath(dev)
            wellpath_2d_html = pio.to_html(wellpath_2d_fig, full_html=False)
            wellpath_2d_view = ft.WebView(
                html=wellpath_2d_html,
                width=400,
                height=500
            )
            
            # Create 3D wellpath plot
            wellpath_3d_fig = Wellpath.plot_wellpath_3d(dev)
            wellpath_3d_html = pio.to_html(wellpath_3d_fig, full_html=False)
            wellpath_3d_view = ft.WebView(
                html=wellpath_3d_html,
                width=400,
                height=500
            )
            
            # Create a row to display the plots side by side
            plots_row = ft.Row([
                ft.Column([
                    ft.Text("2D Wellpath", size=20, weight=ft.FontWeight.BOLD),
                    wellpath_2d_view
                ]),
                ft.Column([
                    ft.Text("3D Wellpath", size=20, weight=ft.FontWeight.BOLD),
                    wellpath_3d_view
                ])
            ])
            
            container.controls.append(plots_row)
            
        except Exception as e:
            container.controls.append(ft.Text(f"Error processing survey file: {str(e)}"))
            
        return container