import flet as ft
import pandas as pd
from Backend.rfBenchmark import RFB
import numpy as np
from PIL import Image
import plotly.io as pio

class RFUI:
    def __init__(self, page):
        self.page = page
        
    def uploadUI(self, RF_data):
        # Create a container for the UI elements
        container = ft.Column(spacing=20)
        
        if not RF_data or len(RF_data) < 2:
            container.controls.append(ft.Text("No RF data uploaded or insufficient data (need at least 2 files)"))
            return container
            
        rfb = RFB()
        rfb2 = RFB()
        
        df_rf_1 = rfb.load_data(RF_data[0])
        df_rf_2 = rfb2.load_data(RF_data[1])
        
        df_rf_1, rfb_df_grouped = rfb.dataformat()
        df_rf_2, rfb_df_grouped2 = rfb2.dataformat()
        
        # Create tabs
        tabs = ft.Tabs(
            selected_index=0,
            animation_duration=300,
            tabs=[
                ft.Tab(
                    text="P50 Plot",
                    content=self.create_p50_tab(rfb)
                ),
                ft.Tab(
                    text="Outliers Detection",
                    content=self.create_outliers_tab(rfb)
                ),
                ft.Tab(
                    text="Margin Plot",
                    content=self.create_margin_tab(rfb)
                ),
                ft.Tab(
                    text="Bubble map",
                    content=self.create_bubble_map_tab(rfb, rfb_df_grouped, rfb_df_grouped2)
                ),
            ],
            expand=1
        )
        
        container.controls.append(tabs)
        return container
    
    def create_p50_tab(self, rfb):
        # Process the data and create the P50 plot
        fig = rfb.plot_p50()
        
        # Convert Plotly figure to HTML and display it in a WebView
        p50_html = pio.to_html(fig, full_html=False)
        p50_view = ft.WebView(
            html=p50_html,
            width=800,
            height=500
        )
        
        # Create the tab content
        return ft.Column([
            ft.Text("P50 Plot", size=20, weight=ft.FontWeight.BOLD),
            p50_view
        ])
    
    def create_outliers_tab(self, rfb):
        # Process the data and create the outliers plot
        fig = rfb.detect_outliers_iso_forest()
        
        # Convert Plotly figure to HTML and display it in a WebView
        outliers_html = pio.to_html(fig, full_html=False)
        outliers_view = ft.WebView(
            html=outliers_html,
            width=800,
            height=500
        )
        
        # Create the tab content
        return ft.Column([
            ft.Text("Outliers Detection", size=20, weight=ft.FontWeight.BOLD),
            outliers_view
        ])
    
    def create_margin_tab(self, rfb):
        # Process the data and create the margin plot
        fig = rfb.plot_margins()
        
        # Convert Plotly figure to HTML and display it in a WebView
        margin_html = pio.to_html(fig, full_html=False)
        margin_view = ft.WebView(
            html=margin_html,
            width=800,
            height=500
        )
        
        # Create the tab content
        return ft.Column([
            ft.Text("Margin Plot", size=20, weight=ft.FontWeight.BOLD),
            margin_view
        ])
    
    def create_bubble_map_tab(self, rfb, rfb_df_grouped, rfb_df_grouped2):
        # Load the image
        try:
            source = Image.open(r"images\MicrosoftTeams-image.png")
        except:
            # Use a fallback if the image is not found
            source = None
            
        coef, intercept = rfb.get_coef_and_intercept()
        
        # Creating comparison dataset for 2 wells
        df_compare = pd.concat([rfb_df_grouped[['fault block', 'Kh']], rfb_df_grouped2[['fault block', 'Kh']]], axis=0)
        df_compare = df_compare.groupby('fault block', axis=0, as_index=False).sum()
        df_compare.rename(columns={'Kh': 'exp Kh'}, inplace=True)
        df_compare = df_compare.merge(rfb_df_grouped[['fault block', 'Kh', 'OOIP, MMSTB']], on=['fault block'])
        
        # Calculating RF & exponential RF
        df_compare['RF'] = np.power(10, coef * np.log10(df_compare['Kh']) + intercept)
        df_compare['exp RF'] = np.power(10, coef * np.log10(df_compare['exp Kh']) + intercept)
        
        # Calculating RF_10 & exponential RF_10
        df_compare['RF_10'] = df_compare['RF'] + (rfb.lr.intercept_ - df_compare['RF'].std() * rfb.margin_multiplier)
        df_compare['exp RF_10'] = df_compare['exp RF'] + (rfb.lr.intercept_ - df_compare['exp RF'].std() * rfb.margin_multiplier)
        
        # Calculating RF_90 & exponential RF_90
        df_compare['RF_90'] = df_compare['RF'] - (rfb.lr.intercept_ - df_compare['RF'].std() * rfb.margin_multiplier)
        df_compare['exp RF_90'] = df_compare['exp RF'] - (rfb.lr.intercept_ - df_compare['exp RF'].std() * rfb.margin_multiplier)
        
        # Calculating incrementals for RF, RF_10 & RF_90
        df_compare['incremental RF'] = df_compare['exp RF'] - df_compare['RF']
        df_compare['incremental RF_10'] = df_compare['exp RF_10'] - df_compare['RF_10']
        df_compare['incremental RF_90'] = df_compare['exp RF_90'] - df_compare['RF_90']
        
        # Calculating EUR
        df_compare['incremental EUR, MMSTB'] = df_compare['incremental RF'] / 100 * df_compare['OOIP, MMSTB']
        df_compare['incremental EUR_10, MMSTB'] = df_compare['incremental RF_10'] / 100 * df_compare['OOIP, MMSTB']
        df_compare['incremental EUR_90, MMSTB'] = df_compare['incremental RF_90'] / 100 * df_compare['OOIP, MMSTB']
        
        # Validation of incremental RF
        if (df_compare['incremental RF'] < 100).all() == False:
            print('Warning!!! RF is bigger than 100% which is impossible!!!')
            
        # Calculating total EUR increase
        eur_increase_total = df_compare['incremental EUR, MMSTB'].sum()
        eur_increase_total_10 = df_compare['incremental EUR_10, MMSTB'].sum()
        eur_increase_total_90 = df_compare['incremental EUR_90, MMSTB'].sum()
        
        # Calculating radius of investigation
        rfb_df_grouped['Boi'] = 1.23
        rfb_df_grouped['radius, ft'] = np.sqrt(
            ((rfb_df_grouped['EUR, MMSTB'] * 1_000_000) / (rfb_df_grouped['RF'] / 100) * rfb_df_grouped['Boi']) / (
                (rfb_df_grouped['poro %'] / 100) * (rfb_df_grouped['perf NTG'] / 100) * (
                    1 - (rfb_df_grouped['Sw %'] / 100)) * rfb_df_grouped['block thickness, ft'] * np.pi))
        rfb_df_grouped['radius, km'] = rfb_df_grouped['radius, ft'] * 0.0003048
        
        # Create text displays for EUR values
        eur_total_text = ft.Text(f"Estimated Ultimate Recovery (Total): {eur_increase_total:.2f}")
        eur_10_text = ft.Text(f"Estimated Ultimate Recovery (10): {eur_increase_total_10:.2f}")
        eur_90_text = ft.Text(f"Estimated Ultimate Recovery (90): {eur_increase_total_90:.2f}")
        
        # Create scatter plot
        rfb_df_grouped, scatter_fig = rfb.scatterplot()
        
        # Create map plot if source image is available
        if source:
            map_fig = rfb.mapplot(source)
            map_html = pio.to_html(map_fig, full_html=False)
            map_view = ft.WebView(
                html=map_html,
                width=800,
                height=500
            )
        else:
            map_view = ft.Text("Map image not found")
        
        # Create the tab content
        return ft.Column([
            ft.Text("Bubble Map", size=20, weight=ft.FontWeight.BOLD),
            eur_total_text,
            eur_10_text,
            eur_90_text,
            ft.Divider(),
            map_view
        ])