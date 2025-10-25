import flet as ft
import os
from PIL import Image
import sys
import importlib.util

# Import modules from UI files/Flet directory
def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import the required modules
data_management_ui = import_from_path("data_management_ui", os.path.join("UI files", "Flet", "data_management_ui.py"))
pda_ui = import_from_path("pda_ui", os.path.join("UI files", "Flet", "pda_ui.py"))

def main(page: ft.Page):
    # Configure the page
    page.title = "Production Enhancement Dashboard"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 20
    page.window_width = 1200
    page.window_height = 800
    
    # Load logo
    try:
        logo_path = os.path.join("images", "logo.png")
        logo = ft.Image(src=logo_path, width=200)
    except Exception as e:
        print(f"Error loading logo: {e}")
        logo = ft.Text("Production Enhancement Dashboard", size=30, weight=ft.FontWeight.BOLD)
    
    # Initialize UI components
    dmu = data_management_ui.DataManagementUi(page)
    pdaui = pda_ui.PdaUi(page)
    
    # Create tabs
    tabs = ft.Tabs(
        selected_index=0,
        animation_duration=300,
        tabs=[
            ft.Tab(
                text="Data Loading",
                content=dmu.upload_ui()
            ),
            ft.Tab(
                text="Data QC/Analysis",
                content=ft.Container(
                    content=ft.Text("Load data in the Data Loading tab first"),
                    padding=20
                )
            ),
            ft.Tab(
                text="Data Export and Reporting",
                content=ft.Container(
                    content=ft.Text("Export and reporting features will be available here"),
                    padding=20
                )
            ),
        ],
        expand=1
    )
    
    # Function to update the Data QC/Analysis tab when data is loaded
    def update_pda_tab():
        monthly_prod_data, pvt_data, dirname = dmu.return_data()
        rf_data = dmu.return_rf_data()
        buildup_pressure_df, buildup_rate_df = dmu.return_buildup_data()
        survey_file = dmu.return_survey_data()
        
        if dirname is not None and rf_data is not None and buildup_pressure_df is not None and buildup_rate_df is not None and survey_file is not None:
            # Update the PDA UI with the loaded data
            pdaui.monthly_prod_data = monthly_prod_data
            pdaui.pvt_data = pvt_data
            pdaui.dirname = dirname
            
            # Update the second tab content
            tabs.tabs[1].content = pdaui.upload_ui(rf_data, buildup_pressure_df, buildup_rate_df, survey_file)
            page.update()
    
    # Add a button to check and update data
    update_button = ft.ElevatedButton(
        text="Check and Load Data for Analysis",
        on_click=lambda _: update_pda_tab()
    )
    
    # Create the main layout
    page.add(
        ft.Column([
            logo,
            ft.Divider(),
            update_button,
            tabs
        ])
    )

# Run the app
if __name__ == "__main__":
    ft.app(target=main)