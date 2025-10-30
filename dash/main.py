import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import pandas as pd
import base64
import io
import os
import sys
import datetime

# Add parent directory to path to import Backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Backend.dataPreprocessing import dataprocess
from Backend.pvtProcessing import pvt_process

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "AUTOMATA ONG"

# Initialize processing objects
pvt = pvt_process()
dp = dataprocess()

# Global variables to store data
processing_logs = []
PVT_df = None
PDG_df = None


# Helper function to add logs
def add_log(message):
    global processing_logs
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    processing_logs.append(log_entry)
    return log_entry


# Layout
app.layout = dbc.Container([
    dcc.Store(id='monthly-prod-data', storage_type='memory'),
    dcc.Store(id='pvt-data', storage_type='memory'),
    dcc.Store(id='pdg-folder-path', storage_type='memory'),
    dcc.Store(id='processed-results', storage_type='memory'),

    html.H1("AUTOMATA ONG", className="text-center my-4"),

    dbc.Tabs([
        # Data Loading Tab
        dbc.Tab(label="Data Loading", children=[
            dbc.Container([
                # PDG Data Folder Section
                html.H4("PDG Data Folder", className="mt-4 mb-3 fw-bold"),
                dbc.Input(
                    id="pdg-folder-input",
                    placeholder="Enter PDG folder path...",
                    type="text",
                    className="mb-2"
                ),
                html.Div(id="pdg-folder-status", className="text-muted small"),

                html.Hr(className="my-4"),

                # Monthly Production Data Section
                html.H4("Monthly Production Data", className="mb-3 fw-bold"),
                dcc.Upload(
                    id='upload-monthly-prod',
                    children=dbc.Button(
                        [html.I(className="bi bi-upload me-2"), "Upload Monthly Production Data"],
                        color="primary"
                    ),
                    multiple=False
                ),
                html.Div(id='monthly-prod-status', className="mt-2 text-muted small"),

                html.Hr(className="my-4"),

                # PVT Data Section
                html.H4("PVT Data", className="mb-3 fw-bold"),
                dcc.Upload(
                    id='upload-pvt',
                    children=dbc.Button(
                        [html.I(className="bi bi-upload me-2"), "Upload PVT Data"],
                        color="primary"
                    ),
                    multiple=False
                ),
                html.Div(id='pvt-status', className="mt-2 text-muted small"),
            ], className="p-4")
        ]),

        # PVT Analysis Tab
        dbc.Tab(label="PVT Analysis", children=[
            dbc.Container([
                html.H3("PVT Data Analysis", className="mt-4 mb-3 fw-bold"),

                dbc.Row([
                    dbc.Col([
                        dbc.Button(
                            [html.I(className="bi bi-play-fill me-2"), "Process Data"],
                            id="process-button",
                            color="success",
                            disabled=True,
                            className="me-2"
                        ),
                        dbc.Spinner(
                            html.Div(id="processing-spinner"),
                            color="primary",
                            spinner_style={"display": "none"}
                        ),
                    ], width="auto"),
                ], className="mb-3"),

                html.Div(id="status-text", className="mb-3"),

                html.Hr(),

                # Processing Logs Section
                dbc.Card([
                    dbc.CardHeader(html.H5("Processing Logs", className="mb-0")),
                    dbc.CardBody([
                        html.Div(id="logs-preview", className="mb-2",
                                 style={"maxHeight": "100px", "overflowY": "auto",
                                        "fontFamily": "monospace", "fontSize": "12px"}),
                        dbc.Button(
                            [html.I(className="bi bi-eye me-2"), "View Processing Details"],
                            id="view-logs-button",
                            color="info",
                            size="sm",
                            disabled=True
                        ),
                    ])
                ], className="mb-4 bg-dark text-white"),

                # Logs Modal
                dbc.Modal([
                    dbc.ModalHeader(dbc.ModalTitle("Processing Details")),
                    dbc.ModalBody(
                        html.Div(id="logs-full",
                                 style={"maxHeight": "500px", "overflowY": "auto",
                                        "fontFamily": "monospace", "fontSize": "12px",
                                        "backgroundColor": "#1a1a1a", "color": "white",
                                        "padding": "10px", "borderRadius": "5px"})
                    ),
                    dbc.ModalFooter([
                        dbc.Button("Close", id="close-logs-modal", className="ms-auto")
                    ])
                ], id="logs-modal", size="lg", is_open=False),

                html.Hr(),

                # Results Section
                html.Div(id="results-container"),

            ], className="p-4")
        ])
    ], id="tabs"),

], fluid=True, className="p-3")


# Callback for PDG folder input
@app.callback(
    Output('pdg-folder-status', 'children'),
    Output('pdg-folder-path', 'data'),
    Input('pdg-folder-input', 'value')
)
def update_pdg_folder(folder_path):
    if not folder_path:
        return "No folder selected", None

    if os.path.isdir(folder_path):
        file_count = len([f for f in os.listdir(folder_path)
                          if f.upper().endswith(('.CSV', '.TXT'))])
        return f"Selected: {folder_path} ({file_count} CSV/TXT files found)", folder_path
    else:
        return "Invalid folder path", None


# Callback for monthly production data upload
@app.callback(
    Output('monthly-prod-status', 'children'),
    Output('monthly-prod-data', 'data'),
    Input('upload-monthly-prod', 'contents'),
    State('upload-monthly-prod', 'filename')
)
def update_monthly_prod(contents, filename):
    if contents is None:
        return ("No file selected.\nRequired columns: "
                "(Well, NPDCode, On Stream, Oil, Gas, Water, Date)"), None

    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

        info = (f"Selected: {filename}\n"
                f"Rows: {len(df)}\n"
                f"Columns: {len(df.columns)}\n"
                f"Columns: {', '.join(df.columns)}")

        return info, contents
    except Exception as e:
        return f"Error reading CSV: {str(e)}", None


# Callback for PVT data upload
@app.callback(
    Output('pvt-status', 'children'),
    Output('pvt-data', 'data'),
    Input('upload-pvt', 'contents'),
    State('upload-pvt', 'filename')
)
def update_pvt(contents, filename):
    if contents is None:
        return ("No file selected.\nRequired columns: "
                "(Pressure, Rs, Bo, Bw, Bg)"), None

    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

        info = (f"Selected: {filename}\n"
                f"Rows: {len(df)}\n"
                f"Columns: {len(df.columns)}\n"
                f"Columns: {', '.join(df.columns)}")

        return info, contents
    except Exception as e:
        return f"Error reading CSV: {str(e)}", None


# Callback to enable/disable process button
@app.callback(
    Output('process-button', 'disabled'),
    Output('status-text', 'children'),
    Input('monthly-prod-data', 'data'),
    Input('pvt-data', 'data'),
    Input('pdg-folder-path', 'data'),
    Input('tabs', 'active_tab')
)
def update_process_button(monthly_data, pvt_data, folder_path, active_tab):
    if all([monthly_data, pvt_data, folder_path]):
        return False, "Ready to process data"
    else:
        return True, "Please load all required data files in Data Loading tab"


# Helper function to read PDG files
def read_pdg_files(folder_path):
    global PDG_df, processing_logs

    add_log(f"Scanning folder: {folder_path}")

    oripath = os.getcwd()
    os.chdir(folder_path)
    file_list = os.listdir()

    add_log(f"Found {len(file_list)} files in folder")

    list_csv = [x for x in file_list if 'CSV' in str.upper(x)]
    list_txt = [x for x in file_list if 'TXT' in str.upper(x)]

    add_log(f"CSV files: {len(list_csv)}, TXT files: {len(list_txt)}")

    PDG_df = pd.DataFrame()
    df_csv = pd.DataFrame()
    df_txt = pd.DataFrame()

    # Process CSV files
    if list_csv:
        add_log(f"Processing {len(list_csv)} CSV files...")
        for i, file in enumerate(list_csv, 1):
            add_log(f"  Reading CSV file {i}/{len(list_csv)}: {file}")
            filepath = os.path.join(folder_path, file)
            df_temp = dp.build_dataframe(filepath, 'csv')
            df_temp = dp.formatting(df_temp)

            if df_csv.empty:
                df_csv = df_temp
            elif df_csv.columns.tolist() == df_temp.columns.tolist():
                df_csv = pd.concat([df_csv, df_temp], ignore_index=True)
        add_log(f"✓ CSV files processed: {len(df_csv)} total rows")

    # Process TXT files
    if list_txt:
        add_log(f"Processing {len(list_txt)} TXT files...")
        for i, file in enumerate(list_txt, 1):
            add_log(f"  Reading TXT file {i}/{len(list_txt)}: {file}")
            filepath = os.path.join(folder_path, file)
            df_temp = dp.build_dataframe(filepath, 'txt')
            df_temp = dp.formatting(df_temp)

            if df_txt.empty:
                df_txt = df_temp
            elif df_txt.columns.tolist() == df_temp.columns.tolist():
                df_txt = pd.concat([df_txt, df_temp], ignore_index=True)
        add_log(f"✓ TXT files processed: {len(df_txt)} total rows")

    # Combine dataframes
    add_log("Combining dataframes...")
    if not df_csv.empty and not df_txt.empty:
        if df_csv.columns.tolist() == df_txt.columns.tolist():
            PDG_df = pd.concat([df_csv, df_txt], ignore_index=True)
            add_log(f"✓ Combined CSV and TXT data: {len(PDG_df)} rows")
    elif not df_csv.empty:
        PDG_df = df_csv
        add_log(f"✓ Using CSV data: {len(PDG_df)} rows")
    elif not df_txt.empty:
        PDG_df = df_txt
        add_log(f"✓ Using TXT data: {len(PDG_df)} rows")

    # Sort and handle data
    add_log("Sorting data by date...")
    PDG_df = dp.sorting(PDG_df)
    add_log("✓ Data sorted")

    add_log("Handling null and duplicate values...")
    PDG_df, _, _ = dp.datahandling(PDG_df, '1', '1')
    add_log("✓ Data cleaned")

    add_log("Resampling data...")
    PDG_df = dp.dataresampling(PDG_df)
    add_log(f"✓ Data resampled: {len(PDG_df)} rows after resampling")

    num_gauge, gauge_type, _, _, _ = dp.num_gaugedetect(PDG_df)
    add_log(f"✓ Detected {len(gauge_type)} gauge types with {len(num_gauge)} gauges")

    os.chdir(oripath)


# Callback to process data
@app.callback(
    Output('results-container', 'children'),
    Output('logs-preview', 'children'),
    Output('logs-full', 'children'),
    Output('view-logs-button', 'disabled'),
    Output('processed-results', 'data'),
    Input('process-button', 'n_clicks'),
    State('monthly-prod-data', 'data'),
    State('pvt-data', 'data'),
    State('pdg-folder-path', 'data'),
    prevent_initial_call=True
)
def process_data(n_clicks, monthly_data, pvt_data, folder_path):
    global PVT_df, PDG_df, processing_logs

    if not n_clicks:
        raise PreventUpdate

    processing_logs = []
    results = []

    try:
        # Load data
        add_log("Loading monthly production data...")
        content_type, content_string = monthly_data.split(',')
        decoded = base64.b64decode(content_string)
        monthly_prod = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        add_log(f"✓ Loaded {len(monthly_prod)} rows from production data")

        add_log("Loading PVT data...")
        content_type, content_string = pvt_data.split(',')
        decoded = base64.b64decode(content_string)
        pvt_data_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        add_log(f"✓ Loaded {len(pvt_data_df)} rows from PVT data")

        add_log("Reading PDG files from folder...")
        read_pdg_files(folder_path)

        add_log("Starting PVT preprocessing...")
        PVT_df = pvt.pvtPreprocessing(monthly_prod, PDG_df, pvt_data_df)
        add_log("✓ PVT preprocessing completed")

        add_log("Calculating total production days...")
        PVT_df = pvt.calculate_total_day(PVT_df)
        add_log("✓ Total days calculated")

        add_log("Converting units...")
        PVT_df = pvt.unitConversion(PVT_df)
        add_log("✓ Units converted")

        add_log("Calculating technical production rates...")
        PVT_df = pvt.calculate_technical_production(PVT_df)
        add_log("✓ Technical production rates calculated")

        add_log("Calculating GOR (Gas-Oil Ratio)...")
        PVT_df = pvt.calculate_gor_technical(PVT_df)
        add_log("✓ GOR calculated")

        add_log("Calculating WC (Water Cut)...")
        PVT_df = pvt.calculate_wc_technical(PVT_df)
        add_log("✓ WC calculated")

        add_log("Calculating free gas...")
        PVT_df = pvt.calculate_free_gas_technical(PVT_df)
        add_log("✓ Free gas calculated")

        add_log("Calculating fw (water fraction)...")
        PVT_df = pvt.calculate_fw_technical(PVT_df)
        add_log("✓ fw calculated")

        add_log("Calculating fg (gas fraction)...")
        PVT_df = pvt.calculate_fg_technical(PVT_df)
        add_log("✓ fg calculated")

        add_log("Calculating fo (oil fraction)...")
        PVT_df = pvt.calculate_fo_technical(PVT_df)
        add_log("✓ fo calculated")

        add_log("Preparing results display...")
        add_log("✓ Results displayed")
        add_log("=== Processing completed successfully! ===")

        # Create results display
        results.append(html.H4("Data Statistics", className="fw-bold mb-3"))
        results.append(html.P(f"Total rows: {len(PVT_df)}"))
        results.append(html.P(f"Columns: {', '.join(PVT_df.columns[:10])}..."))
        results.append(html.H5("First few rows of calculated data:", className="mt-3 mb-2"))

        # Display sample data table
        display_cols = ['Date', 'Oil Technical Production Rate',
                        'Water Technical Production Rate', 'Gas Technical Production Rate',
                        'fo', 'fw', 'fg']
        available_cols = [col for col in display_cols if col in PVT_df.columns]

        if available_cols:
            sample_df = PVT_df[available_cols].head(10)

            results.append(
                dash_table.DataTable(
                    data=sample_df.to_dict('records'),
                    columns=[{'name': col, 'id': col} for col in available_cols],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '5px'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                )
            )

        # Prepare logs display
        logs_preview = html.Div([html.Div(log) for log in processing_logs[-3:]])
        logs_full = html.Div([html.Div(log) for log in processing_logs])

        return results, logs_preview, logs_full, False, True

    except Exception as ex:
        error_msg = f"❌ Error: {str(ex)}"
        add_log(error_msg)

        results = [
            dbc.Alert(f"Error details: {str(ex)}", color="danger")
        ]

        logs_preview = html.Div([html.Div(log) for log in processing_logs[-3:]])
        logs_full = html.Div([html.Div(log) for log in processing_logs])

        return results, logs_preview, logs_full, False, None


# Callback to toggle logs modal
@app.callback(
    Output('logs-modal', 'is_open'),
    Input('view-logs-button', 'n_clicks'),
    Input('close-logs-modal', 'n_clicks'),
    State('logs-modal', 'is_open'),
    prevent_initial_call=True
)
def toggle_logs_modal(view_clicks, close_clicks, is_open):
    return not is_open


if __name__ == '__main__':
    app.run(debug=True)