import flet as ft
import pandas as pd
import sys
import os

# Add parent directory to path to import Backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Backend.dataPreprocessing import dataprocess
from Backend.pvtProcessing import pvt_process


class DataLoadingTab:
    def __init__(self, page):
        self.page = page
        self.monthly_prod_file = None
        self.pvt_file = None
        self.pdg_folder = None

        # Initialize file pickers
        self.monthly_prod_picker = ft.FilePicker(on_result=self.on_monthly_prod_selected)
        self.pvt_picker = ft.FilePicker(on_result=self.on_pvt_selected)
        self.folder_picker = ft.FilePicker(on_result=self.on_folder_selected)

        # UI elements
        self.monthly_prod_text = ft.Text(
            "No file selected.\nRequired columns: (Well, NPDCode, On Stream, Oil, Gas, Water, Date)",
            size=12
        )
        self.pvt_text = ft.Text(
            "No file selected.\nRequired columns: (Pressure, Rs, Bo, Bw, Bg)",
            size=12
        )
        self.folder_text = ft.Text("No folder selected", size=12)

        self.tab = ft.Tab(
            text='Data Loading',
            content=ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Text("PDG Data Folder", size=16, weight=ft.FontWeight.BOLD),
                        ft.ElevatedButton(
                            'Select PDG Folder',
                            icon=ft.Icons.FOLDER_OPEN,
                            on_click=lambda _: self.folder_picker.get_directory_path()
                        ),
                        self.folder_text,

                        ft.Divider(height=20, thickness=2),

                        ft.Text("Monthly Production Data", size=16, weight=ft.FontWeight.BOLD),
                        ft.ElevatedButton(
                            'Upload Monthly Production Data',
                            icon=ft.Icons.UPLOAD_FILE,
                            on_click=lambda _: self.monthly_prod_picker.pick_files(
                                allowed_extensions=['csv'],
                                dialog_title='Select monthly production CSV file'
                            )
                        ),
                        self.monthly_prod_text,

                        ft.Divider(height=20, thickness=2),

                        ft.Text("PVT Data", size=16, weight=ft.FontWeight.BOLD),
                        ft.ElevatedButton(
                            'Upload PVT Data',
                            icon=ft.Icons.UPLOAD_FILE,
                            on_click=lambda _: self.pvt_picker.pick_files(
                                allowed_extensions=['csv'],
                                dialog_title='Select PVT CSV file'
                            )
                        ),
                        self.pvt_text,
                    ],
                    spacing=10,
                    scroll=ft.ScrollMode.AUTO,
                ),
                padding=10,
                expand=True
            )
        )

    def on_monthly_prod_selected(self, e: ft.FilePickerResultEvent):
        if e.files:
            file = e.files[0]
            self.monthly_prod_file = file.path
            self.monthly_prod_text.value = f"Selected: {file.name}"

            try:
                df = pd.read_csv(file.path)
                info = f"\nRows: {len(df)}\nColumns: {len(df.columns)}\nColumns: {', '.join(df.columns)}"
                self.monthly_prod_text.value += info
            except Exception as ex:
                self.monthly_prod_text.value = f"Error reading CSV: {str(ex)}"
        else:
            self.monthly_prod_text.value = "No file selected.\nRequired columns: (Well, NPDCode, On Stream, Oil, Gas, Water, Date)"

        self.page.update()

    def on_pvt_selected(self, e: ft.FilePickerResultEvent):
        if e.files:
            file = e.files[0]
            self.pvt_file = file.path
            self.pvt_text.value = f"Selected: {file.name}"

            try:
                df = pd.read_csv(file.path)
                info = f"\nRows: {len(df)}\nColumns: {len(df.columns)}\nColumns: {', '.join(df.columns)}"
                self.pvt_text.value += info
            except Exception as ex:
                self.pvt_text.value = f"Error reading CSV: {str(ex)}"
        else:
            self.pvt_text.value = "No file selected.\nRequired columns: (Pressure, Rs, Bo, Bw, Bg)"

        self.page.update()

    def on_folder_selected(self, e: ft.FilePickerResultEvent):
        if e.path:
            self.pdg_folder = e.path
            self.folder_text.value = f"Selected: {e.path}"
        else:
            self.folder_text.value = "No folder selected"

        self.page.update()

    def get_file_pickers(self):
        return [self.monthly_prod_picker, self.pvt_picker, self.folder_picker]


class PVTAnalysisTab:
    def __init__(self, page, data_loading_tab):
        self.page = page
        self.data_loading_tab = data_loading_tab
        self.pvt = pvt_process()
        self.dp = dataprocess()
        self.PVT_df = None
        self.PDG_df = None
        self.processing_logs = []

        # UI elements
        self.status_text = ft.Text("No data loaded", size=14)
        self.progress_ring = ft.ProgressRing(visible=False)
        self.process_button = ft.ElevatedButton(
            "Process Data",
            icon=ft.Icons.PLAY_ARROW,
            on_click=self.process_data,
            disabled=True
        )

        # Processing logs section
        self.logs_preview = ft.Text(
            "No processing logs yet",
            size=12,
            max_lines=3,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
        self.view_logs_button = ft.ElevatedButton(
            "View Processing Details",
            icon=ft.Icons.VISIBILITY,
            on_click=self.show_logs_dialog,
            disabled=True,
        )

        # Results container
        self.results_container = ft.Column(
            controls=[],
            spacing=10,
            scroll=ft.ScrollMode.AUTO,
        )

        self.tab = ft.Tab(
            text='PVT Analysis',
            content=ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Text("PVT Data Analysis", size=20, weight=ft.FontWeight.BOLD),
                        ft.Row(
                            controls=[
                                self.process_button,
                                self.progress_ring,
                            ],
                            spacing=10
                        ),
                        self.status_text,

                        ft.Divider(height=10, thickness=1),

                        # Processing logs section
                        ft.Container(
                            content=ft.Column([
                                ft.Text("Processing Logs:", size=14, weight=ft.FontWeight.BOLD),
                                self.logs_preview,
                                self.view_logs_button,
                            ]),
                            bgcolor=ft.Colors.GREY_900,
                            padding=10,
                            border_radius=5,
                        ),

                        ft.Divider(height=20, thickness=2),
                        self.results_container,
                    ],
                    spacing=10,
                    scroll=ft.ScrollMode.AUTO,
                ),
                padding=10,
                expand=True
            )
        )

    def add_log(self, message):
        """Add a log message to the processing logs"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.processing_logs.append(log_entry)

        # Update preview with last 3 logs
        preview_text = "\n".join(self.processing_logs[-3:])
        self.logs_preview.value = preview_text
        self.view_logs_button.disabled = False
        self.page.update()

    def show_logs_dialog(self, e):
        """Show full processing logs in an expanded dialog"""
        if not self.processing_logs:
            # If no logs, show a message
            self.page.show_snack_bar(
                ft.SnackBar(content=ft.Text("No processing logs available yet"), duration=2000)
            )
            return

        # Create scrollable log content
        log_content = ft.Column(
            controls=[
                ft.Text(log, size=12, selectable=True, color=ft.Colors.WHITE)
                for log in self.processing_logs
            ],
            spacing=5,
            scroll=ft.ScrollMode.ALWAYS,
        )

        # Create dialog
        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Processing Details", size=18, weight=ft.FontWeight.BOLD),
            content=ft.Container(
                content=log_content,
                width=700,
                height=500,
                padding=10,
                bgcolor=ft.Colors.BLACK87,
                border_radius=5,
            ),
            actions=[
                ft.TextButton("Close", on_click=lambda _: self.close_dialog(dialog)),
                ft.TextButton(
                    "Copy to Clipboard",
                    on_click=lambda _: self.copy_logs_to_clipboard()
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        self.page.dialog = dialog
        dialog.open = True
        self.page.update()

    def close_dialog(self, dialog):
        """Close the logs dialog"""
        dialog.open = False
        self.page.update()

    def copy_logs_to_clipboard(self):
        """Copy all logs to clipboard"""
        logs_text = "\n".join(self.processing_logs)
        self.page.set_clipboard(logs_text)
        # Show a snackbar notification
        self.page.show_snack_bar(
            ft.SnackBar(content=ft.Text("Logs copied to clipboard!"), duration=2000)
        )
        self.page.update()

    def check_data_available(self):
        """Check if required data files are loaded"""
        has_data = (
                self.data_loading_tab.monthly_prod_file is not None and
                self.data_loading_tab.pvt_file is not None and
                self.data_loading_tab.pdg_folder is not None
        )
        self.process_button.disabled = not has_data
        if has_data:
            self.status_text.value = "Ready to process data"
        else:
            self.status_text.value = "Please load all required data files in Data Loading tab"
        self.page.update()

    def process_data(self, e):
        """Process PVT and production data"""
        self.progress_ring.visible = True
        self.process_button.disabled = True
        self.status_text.value = "Processing data..."
        self.results_container.controls.clear()
        self.processing_logs.clear()
        self.logs_preview.value = "Processing started..."
        self.page.update()

        try:
            # Load data
            self.add_log("Loading monthly production data...")
            monthly_prod = pd.read_csv(self.data_loading_tab.monthly_prod_file)
            self.add_log(f"✓ Loaded {len(monthly_prod)} rows from production data")

            self.add_log("Loading PVT data...")
            pvt_data = pd.read_csv(self.data_loading_tab.pvt_file)
            self.add_log(f"✓ Loaded {len(pvt_data)} rows from PVT data")

            self.status_text.value = "Reading PDG files..."
            self.add_log("Reading PDG files from folder...")
            self.page.update()

            # Read PDG files
            self.read_pdg_files(self.data_loading_tab.pdg_folder)

            self.status_text.value = "Processing PVT data..."
            self.add_log("Starting PVT preprocessing...")
            self.page.update()

            # Process PVT data
            self.PVT_df = self.pvt.pvtPreprocessing(monthly_prod, self.PDG_df, pvt_data)
            self.add_log("✓ PVT preprocessing completed")

            self.add_log("Calculating total production days...")
            self.PVT_df = self.pvt.calculate_total_day(self.PVT_df)
            self.add_log("✓ Total days calculated")

            self.add_log("Converting units...")
            self.PVT_df = self.pvt.unitConversion(self.PVT_df)
            self.add_log("✓ Units converted")

            self.add_log("Calculating technical production rates...")
            self.PVT_df = self.pvt.calculate_technical_production(self.PVT_df)
            self.add_log("✓ Technical production rates calculated")

            self.add_log("Calculating GOR (Gas-Oil Ratio)...")
            self.PVT_df = self.pvt.calculate_gor_technical(self.PVT_df)
            self.add_log("✓ GOR calculated")

            self.add_log("Calculating WC (Water Cut)...")
            self.PVT_df = self.pvt.calculate_wc_technical(self.PVT_df)
            self.add_log("✓ WC calculated")

            self.add_log("Calculating free gas...")
            self.PVT_df = self.pvt.calculate_free_gas_technical(self.PVT_df)
            self.add_log("✓ Free gas calculated")

            self.add_log("Calculating fw (water fraction)...")
            self.PVT_df = self.pvt.calculate_fw_technical(self.PVT_df)
            self.add_log("✓ fw calculated")

            self.add_log("Calculating fg (gas fraction)...")
            self.PVT_df = self.pvt.calculate_fg_technical(self.PVT_df)
            self.add_log("✓ fg calculated")

            self.add_log("Calculating fo (oil fraction)...")
            self.PVT_df = self.pvt.calculate_fo_technical(self.PVT_df)
            self.add_log("✓ fo calculated")

            # Display results
            self.add_log("Preparing results display...")
            self.display_results()
            self.add_log("✓ Results displayed")

            self.status_text.value = "✅ Processing completed successfully!"
            self.add_log("=== Processing completed successfully! ===")

        except Exception as ex:
            error_msg = f"❌ Error: {str(ex)}"
            self.status_text.value = error_msg
            self.add_log(error_msg)
            self.results_container.controls.append(
                ft.Text(f"Error details: {str(ex)}", color=ft.Colors.RED)
            )

        finally:
            self.progress_ring.visible = False
            self.process_button.disabled = False
            self.page.update()

    def read_pdg_files(self, folder_path):
        """Read PDG files from folder"""
        import os

        self.add_log(f"Scanning folder: {folder_path}")

        oripath = os.getcwd()
        os.chdir(folder_path)
        file_list = os.listdir()

        self.add_log(f"Found {len(file_list)} files in folder")

        list_csv = [x for x in file_list if 'CSV' in str.upper(x)]
        list_txt = [x for x in file_list if 'TXT' in str.upper(x)]

        self.add_log(f"CSV files: {len(list_csv)}, TXT files: {len(list_txt)}")

        self.PDG_df = pd.DataFrame()
        df_csv = pd.DataFrame()
        df_txt = pd.DataFrame()

        # Process CSV files
        if list_csv:
            self.add_log(f"Processing {len(list_csv)} CSV files...")
            for i, file in enumerate(list_csv, 1):
                self.add_log(f"  Reading CSV file {i}/{len(list_csv)}: {file}")
                filepath = os.path.join(folder_path, file)
                df_temp = self.dp.build_dataframe(filepath, 'csv')
                df_temp = self.dp.formatting(df_temp)

                if df_csv.empty:
                    df_csv = df_temp
                elif df_csv.columns.tolist() == df_temp.columns.tolist():
                    df_csv = pd.concat([df_csv, df_temp], ignore_index=True)
            self.add_log(f"✓ CSV files processed: {len(df_csv)} total rows")

        # Process TXT files
        if list_txt:
            self.add_log(f"Processing {len(list_txt)} TXT files...")
            for i, file in enumerate(list_txt, 1):
                self.add_log(f"  Reading TXT file {i}/{len(list_txt)}: {file}")
                filepath = os.path.join(folder_path, file)
                df_temp = self.dp.build_dataframe(filepath, 'txt')
                df_temp = self.dp.formatting(df_temp)

                if df_txt.empty:
                    df_txt = df_temp
                elif df_txt.columns.tolist() == df_temp.columns.tolist():
                    df_txt = pd.concat([df_txt, df_temp], ignore_index=True)
            self.add_log(f"✓ TXT files processed: {len(df_txt)} total rows")

        # Combine dataframes
        self.add_log("Combining dataframes...")
        if not df_csv.empty and not df_txt.empty:
            if df_csv.columns.tolist() == df_txt.columns.tolist():
                self.PDG_df = pd.concat([df_csv, df_txt], ignore_index=True)
                self.add_log(f"✓ Combined CSV and TXT data: {len(self.PDG_df)} rows")
        elif not df_csv.empty:
            self.PDG_df = df_csv
            self.add_log(f"✓ Using CSV data: {len(self.PDG_df)} rows")
        elif not df_txt.empty:
            self.PDG_df = df_txt
            self.add_log(f"✓ Using TXT data: {len(self.PDG_df)} rows")

        # Sort and handle data
        self.add_log("Sorting data by date...")
        self.PDG_df = self.dp.sorting(self.PDG_df)
        self.add_log("✓ Data sorted")

        self.add_log("Handling null and duplicate values...")
        self.PDG_df, _, _ = self.dp.datahandling(self.PDG_df, '1', '1')
        self.add_log("✓ Data cleaned")

        self.add_log("Resampling data...")
        self.PDG_df = self.dp.dataresampling(self.PDG_df)
        self.add_log(f"✓ Data resampled: {len(self.PDG_df)} rows after resampling")

        num_gauge, gauge_type, _, _, _ = self.dp.num_gaugedetect(self.PDG_df)
        self.add_log(f"✓ Detected {len(gauge_type)} gauge types with {len(num_gauge)} gauges")

        os.chdir(oripath)

    def display_results(self):
        """Display processing results"""
        if self.PVT_df is not None:
            # Statistics
            stats_text = ft.Text("Data Statistics:", size=16, weight=ft.FontWeight.BOLD)
            self.results_container.controls.append(stats_text)

            # Display basic info
            info = ft.Column([
                ft.Text(f"Total rows: {len(self.PVT_df)}"),
                ft.Text(f"Columns: {', '.join(self.PVT_df.columns[:10])}..."),
                ft.Text(f"\nFirst few rows of calculated data:"),
            ])
            self.results_container.controls.append(info)

            # Display sample data
            display_cols = ['Date', 'Oil Technical Production Rate',
                            'Water Technical Production Rate', 'Gas Technical Production Rate',
                            'fo', 'fw', 'fg']
            available_cols = [col for col in display_cols if col in self.PVT_df.columns]

            if available_cols:
                sample_df = self.PVT_df[available_cols].head(10)

                # Create data table
                columns = [ft.DataColumn(ft.Text(col)) for col in available_cols]
                rows = []

                for idx, row in sample_df.iterrows():
                    cells = [ft.DataCell(ft.Text(str(row[col])[:20])) for col in available_cols]
                    rows.append(ft.DataRow(cells=cells))

                data_table = ft.DataTable(
                    columns=columns,
                    rows=rows,
                )

                self.results_container.controls.append(
                    ft.Container(
                        content=ft.Column([data_table], scroll=ft.ScrollMode.AUTO),
                        height=300,
                    )
                )

        self.page.update()


def main(page: ft.Page):
    page.title = "AUTOMATA ONG - Production Data Analysis"
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.scroll = ft.ScrollMode.AUTO

    # Create tabs
    data_loading_tab = DataLoadingTab(page)
    pvt_analysis_tab = PVTAnalysisTab(page, data_loading_tab)

    # Add file pickers to overlay
    for picker in data_loading_tab.get_file_pickers():
        page.overlay.append(picker)

    # Create tabs control
    tabs = ft.Tabs(
        selected_index=0,
        animation_duration=300,
        tabs=[
            data_loading_tab.tab,
            pvt_analysis_tab.tab,
        ],
        expand=True,
        on_change=lambda e: pvt_analysis_tab.check_data_available()
    )

    page.add(tabs)


if __name__ == "__main__":
    ft.app(target=main)