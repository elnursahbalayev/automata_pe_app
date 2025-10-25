import flet as ft
from Flet.infill_well_ui import PePlotsUi
from Flet.rf_ui import RFUI

class Res_Management:
    def __init__(self, page):
        self.page = page
        self.pplt = PePlotsUi(page)
        self.rfui = RFUI(page)
        
    def upload_ui(self, PVT_df, RF_data):
        # Create a container for the UI elements
        container = ft.Column(spacing=20)
        
        # Create tabs
        tabs = ft.Tabs(
            selected_index=0,
            animation_duration=300,
            tabs=[
                ft.Tab(
                    text="Infill Wells",
                    content=self.pplt.upload_ui(PVT_df)
                ),
                ft.Tab(
                    text="Forecasting",
                    content=self.rfui.uploadUI(RF_data)
                ),
            ],
            expand=1
        )
        
        container.controls.append(tabs)
        return container