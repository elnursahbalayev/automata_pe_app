from Backend.pePlots import Plot as pplt
import pandas as pd
import flet as ft
import plotly.io as pio

class PePlotsUi:
    def __init__(self, page):
        self.page = page
        self.pplt = pplt()
        self.p_avg_res = 0
        
    def upload_ui(self, PVT_df):
        # Create a container for the UI elements
        container = ft.Column(spacing=20)
        
        # Create tabs
        tabs = ft.Tabs(
            selected_index=0,
            animation_duration=300,
            tabs=[
                ft.Tab(
                    text="Chan Plot",
                    content=self.create_chan_tab(PVT_df)
                ),
                ft.Tab(
                    text="Hall Plot",
                    content=self.create_hall_tab(PVT_df)
                ),
                ft.Tab(
                    text="VRR Plot",
                    content=self.create_vrr_tab(PVT_df)
                ),
            ],
            expand=1
        )
        
        container.controls.append(tabs)
        return container
    
    def create_chan_tab(self, PVT_df):
        # Process the data and create the Chan plot
        self.PVT_df, fig_chan = pplt.chanplot(PVT_df)
        
        # Convert Plotly figure to HTML and display it in a WebView
        chan_html = pio.to_html(fig_chan, full_html=False)
        chan_view = ft.WebView(
            html=chan_html,
            width=800,
            height=500
        )
        
        # Create the tab content
        return ft.Column([
            ft.Text("Chan Plot", size=20, weight=ft.FontWeight.BOLD),
            chan_view
        ])
    
    def create_hall_tab(self, PVT_df):
        # Create input field for average reservoir pressure
        p_avg_res_input = ft.TextField(
            label="Please enter average pressure of the reservoir",
            value="0",
            width=300,
            on_change=self.on_p_avg_res_change
        )
        
        # Create a button to generate the plot
        generate_button = ft.ElevatedButton(
            text="Generate Hall Plot",
            on_click=lambda e: self.update_hall_plot(e, PVT_df)
        )
        
        # Create a container for the plot
        self.hall_plot_container = ft.Container(
            content=ft.Text("Enter average reservoir pressure and click Generate"),
            width=800,
            height=500
        )
        
        # Create the tab content
        return ft.Column([
            ft.Text("Hall Plot", size=20, weight=ft.FontWeight.BOLD),
            p_avg_res_input,
            generate_button,
            self.hall_plot_container
        ])
    
    def on_p_avg_res_change(self, e):
        try:
            self.p_avg_res = float(e.control.value)
        except ValueError:
            # Handle invalid input
            pass
    
    def update_hall_plot(self, e, PVT_df):
        if self.p_avg_res > 0:
            WI_volume = [7000]  # This has to be changed later as file input
            
            # Process the data and create the Hall plot
            self.PVT_df, fig_hall = pplt.hallplot(self.PVT_df, WI_volume, self.p_avg_res)
            
            # Convert Plotly figure to HTML and display it in a WebView
            hall_html = pio.to_html(fig_hall, full_html=False)
            hall_view = ft.WebView(
                html=hall_html,
                width=800,
                height=500
            )
            
            # Update the plot container
            self.hall_plot_container.content = hall_view
            self.page.update()
    
    def create_vrr_tab(self, PVT_df):
        # Process the data and create the VRR plot
        self.PVT_df, fig_vrr = pplt.plot_VRR(self.PVT_df)
        
        # Convert Plotly figure to HTML and display it in a WebView
        vrr_html = pio.to_html(fig_vrr, full_html=False)
        vrr_view = ft.WebView(
            html=vrr_html,
            width=800,
            height=500
        )
        
        # Create the tab content
        return ft.Column([
            ft.Text("VRR Plot", size=20, weight=ft.FontWeight.BOLD),
            vrr_view
        ])