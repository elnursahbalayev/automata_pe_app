import flet as ft
import pandas as pd


class ProductionTab:
    def __init__(self, page):
        self.page = page
        self.file_picker = None  # Will be set later
        self.pvt_file_picker = None  # PVT file picker

        # UI elements for production data
        self.result_text = ft.Text(
            "No file selected.\nRequired columns: (Well, NPDCode, On Stream, Oil, Gas, Water, Date)",
            size=12
        )
        self.data_display = ft.Column(scroll=ft.ScrollMode.AUTO, height=200)

        # UI elements for PVT data
        self.pvt_result_text = ft.Text(
            "No PVT file selected.\nRequired columns: (Pressure, Rs, Bo, Bw, Bg)",
            size=12
        )
        self.pvt_data_display = ft.Column(scroll=ft.ScrollMode.AUTO, height=200)

        self.tab = ft.Tab(
            text='Production',
            content=ft.Container(
                content=ft.Column(
                    controls=[
                        # Production Data Section
                        ft.Text("Production Data", size=16, weight=ft.FontWeight.BOLD),
                        ft.ElevatedButton(
                            'Upload Montly Production Data',
                            icon=ft.Icons.UPLOAD_FILE,
                            on_click=self.pick_file
                        ),
                        self.result_text,
                        self.data_display,

                        ft.Divider(height=10, thickness=2),

                        # PVT Data Section
                        ft.Text("PVT Data", size=16, weight=ft.FontWeight.BOLD),
                        ft.ElevatedButton(
                            'Upload PVT Data',
                            icon=ft.Icons.UPLOAD_FILE,
                            on_click=self.pick_pvt_file
                        ),
                        self.pvt_result_text,
                        self.pvt_data_display,
                    ],
                    spacing=10,
                    scroll=ft.ScrollMode.AUTO,
                ),
                padding=10,
                expand=True
            )
        )

    def set_file_picker(self, file_picker):
        """Set the file picker after it's been added to the page"""
        self.file_picker = file_picker

    def set_pvt_file_picker(self, pvt_file_picker):
        """Set the PVT file picker after it's been added to the page"""
        self.pvt_file_picker = pvt_file_picker

    def on_file_selected(self, e: ft.FilePickerResultEvent):
        if e.files:
            file = e.files[0]
            print(f'Production file {file.name}, path: {file.path}')
            file_path = file.path

            self.result_text.value = f"Selected: {file.name}"

            try:
                # Read CSV file with pandas
                df = pd.read_csv(file_path)

                # Display basic info
                info_text = f"\nRows: {len(df)}\nColumns: {len(df.columns)}\n\nColumns: {', '.join(df.columns)}"
                self.result_text.value += info_text

                # Display first few rows
                self.data_display.controls.clear()
                self.data_display.controls.append(
                    ft.Text("First 5 rows:", weight=ft.FontWeight.BOLD, size=14)
                )
                self.data_display.controls.append(
                    ft.Text(df.head().to_string(), font_family="Courier New", size=10)
                )

            except Exception as ex:
                self.result_text.value = f"Error reading CSV: {str(ex)}"
                self.data_display.controls.clear()
        else:
            self.result_text.value = """No file selected.
Required columns: (Well, NPDCode, On Stream, Oil, Gas, Water, Date)"""
            self.data_display.controls.clear()

        self.page.update()

    def on_pvt_file_selected(self, e: ft.FilePickerResultEvent):
        if e.files:
            file = e.files[0]
            print(f'PVT file {file.name}, path: {file.path}')
            file_path = file.path

            self.pvt_result_text.value = f"Selected: {file.name}"

            try:
                # Read CSV file with pandas
                df = pd.read_csv(file_path)

                # Display basic info
                info_text = f"\nRows: {len(df)}\nColumns: {len(df.columns)}\n\nColumns: {', '.join(df.columns)}"
                self.pvt_result_text.value += info_text

                # Display first few rows
                self.pvt_data_display.controls.clear()
                self.pvt_data_display.controls.append(
                    ft.Text("First 5 rows:", weight=ft.FontWeight.BOLD, size=14)
                )
                self.pvt_data_display.controls.append(
                    ft.Text(df.head().to_string(), font_family="Courier New", size=10)
                )

            except Exception as ex:
                self.pvt_result_text.value = f"Error reading CSV: {str(ex)}"
                self.pvt_data_display.controls.clear()
        else:
            self.pvt_result_text.value = """No file selected.
Required columns: (Pressure, Rs, Bo, Bw, Bg)"""
            self.pvt_data_display.controls.clear()

        self.page.update()

    def pick_file(self, e):
        if self.file_picker:
            self.file_picker.pick_files(
                allowed_extensions=['csv'],
                dialog_title='Select a production csv file'
            )

    def pick_pvt_file(self, e):
        if self.pvt_file_picker:
            self.pvt_file_picker.pick_files(
                allowed_extensions=['csv'],
                dialog_title='Select a PVT csv file'
            )


class WellLogTab:
    def __init__(self):
        self.tab = ft.Tab(
            text='Well Log',
            content=ft.Text('Well Log Interpretation')
        )


class DrillingTab:
    def __init__(self):
        self.tab = ft.Tab(
            text='Drilling Risk',
            content=ft.Text('Drilling Risk Prediction and Prevention')
        )


class TabManager:
    def __init__(self, page):
        self.page = page
        self.selected_index = 0
        self.animation_duration = 100
        self.label_color = ft.Colors.RED_500
        self.divider_color = ft.Colors.BLUE_500
        self.overlay_color = ft.Colors.BLACK38
        self.indicator_color = ft.Colors.BLACK
        self.unselected_label_color = ft.Colors.GREY

        self.production_tab = ProductionTab(self.page)
        self.well_log_tab = WellLogTab()
        self.drilling_tab = DrillingTab()

        self.get_tabs()

    def get_tabs(self):
        # Create and add file pickers to overlay FIRST
        file_picker = ft.FilePicker(on_result=self.production_tab.on_file_selected)
        pvt_file_picker = ft.FilePicker(on_result=self.production_tab.on_pvt_file_selected)

        self.page.overlay.append(file_picker)
        self.page.overlay.append(pvt_file_picker)

        self.production_tab.set_file_picker(file_picker)
        self.production_tab.set_pvt_file_picker(pvt_file_picker)

        self.tabs = ft.Tabs(
            selected_index=self.selected_index,
            animation_duration=self.animation_duration,
            label_color=self.label_color,
            divider_color=self.divider_color,
            overlay_color=self.overlay_color,
            indicator_color=self.indicator_color,
            unselected_label_color=self.unselected_label_color,
            tabs=[
                self.production_tab.tab,
                self.well_log_tab.tab,
                self.drilling_tab.tab
            ],
            expand=True
        )

        self.page.add(self.tabs)
        return self.page


def main(page: ft.Page):
    page.title = "AUTOMATA ONG"
    page.vertical_alignment = ft.MainAxisAlignment.START

    TabManager(page)


# Run the app
if __name__ == "__main__":
    ft.app(target=main)