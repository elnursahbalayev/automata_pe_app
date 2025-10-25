from Backend.buildup_recognition import BuildUp
import pandas as pd
import flet as ft
import plotly.io as pio

class BuildUpUi:
    def __init__(self, page):
        self.page = page
        self.buildup = BuildUp()
        
    def upload_ui(self, buildup_pressure_df, buildup_rate_df):
        # Create a container for the UI elements
        container = ft.Column(spacing=20)
        
        # Process the data
        self.buildup.read_data(buildup_pressure_df, buildup_rate_df)
        self.buildup.format_data()
        
        # Convert Plotly figures to HTML and display them in WebViews
        pressure_rate_fig = self.buildup.plot_pressure_rate()
        pressure_rate_html = pio.to_html(pressure_rate_fig, full_html=False)
        pressure_rate_view = ft.WebView(
            url=pressure_rate_html,
            width=800,
            height=500
        )
        
        # Process other plots
        self.buildup.plot_pressure_der_rate()
        self.buildup.plot_pressure_der_cleaned_rate()
        
        buildup_zones_fig = self.buildup.plot_buildup_zones()
        buildup_zones_html = pio.to_html(buildup_zones_fig, full_html=False)
        buildup_zones_view = ft.WebView(
            url=buildup_zones_html,
            width=800,
            height=500
        )
        
        # Add plots to the container
        container.controls.extend([
            ft.Text("Pressure and Rate Plot", size=20, weight=ft.FontWeight.BOLD),
            pressure_rate_view,
            ft.Divider(),
            ft.Text("Buildup Zones Plot", size=20, weight=ft.FontWeight.BOLD),
            buildup_zones_view
        ])
        
        return container