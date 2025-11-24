import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import base64
import datetime

# ---------------------------------------------------------
# 1. Setup & Mock Data
# ---------------------------------------------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True)
app.title = "Automata Intelligence - Corrosion Detect"

# Mock Data for the Map
map_data = pd.DataFrame({
    "Pipeline_ID": ["P-101", "P-102", "P-103", "P-104", "P-105"],
    "Lat": [29.7604, 31.9686, 25.276987, 57.15, 4.71],
    "Lon": [-95.3698, -99.9018, 55.296249, -2.094, -74.07],
    "Status": ["Critical", "Safe", "Warning", "Safe", "Critical"],
    "Corrosion_Level": [85, 12, 45, 5, 92],  # % Severity
    "Location": ["Texas, USA", "West Texas", "Dubai, UAE", "Aberdeen, UK", "Bogota, COL"]
})


# ---------------------------------------------------------
# 2. Visual Components
# ---------------------------------------------------------

def draw_kpi_card(title, value, color):
    return dbc.Card(
        [
            dbc.CardBody(
                [
                    html.H5(title, className="card-title text-muted"),
                    html.H2(value, className=f"text-{color}"),
                ]
            )
        ],
        color="dark", inverse=True, className="mb-4 shadow-sm"
    )


def draw_map():
    fig = px.scatter_mapbox(
        map_data, lat="Lat", lon="Lon", color="Status", size="Corrosion_Level",
        color_discrete_map={"Critical": "red", "Safe": "#00cc96", "Warning": "orange"},
        hover_name="Pipeline_ID", hover_data=["Location", "Corrosion_Level"],
        zoom=1, height=400
    )
    fig.update_layout(mapbox_style="carto-darkmatter", margin={"r": 0, "t": 0, "l": 0, "b": 0},
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def draw_gauge(value=0):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': "Corrosion Severity (%)"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "red" if value > 70 else "orange" if value > 30 else "green"},
            'bgcolor': "white",
        }
    ))
    fig.update_layout(height=300, margin={"r": 10, "t": 40, "l": 10, "b": 10}, paper_bgcolor="rgba(0,0,0,0)",
                      font={'color': "white"})
    return fig


# ---------------------------------------------------------
# 3. App Layout
# ---------------------------------------------------------

# Sidebar Navigation
sidebar = html.Div(
    [
        html.H3("AUTOMATA", className="display-6 text-primary", style={"fontWeight": "bold"}),
        html.H6("INTELLIGENCE", className="text-white mb-5", style={"letterSpacing": "4px"}),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink([html.I(className="fas fa-home me-2"), "Dashboard"], href="/", active="exact"),
                dbc.NavLink([html.I(className="fas fa-camera me-2"), "AI Inspection"], href="/analyze", active="exact"),
                dbc.NavLink([html.I(className="fas fa-cog me-2"), "Settings"], href="/settings", disabled=True),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style={
        "position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "18rem",
        "padding": "2rem 1rem", "backgroundColor": "#111"
    },
)

# Content Container
content = html.Div(id="page-content", style={"marginLeft": "20rem", "marginRight": "2rem", "padding": "2rem 1rem"})

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


# ---------------------------------------------------------
# 4. Pages
# ---------------------------------------------------------

def dashboard_page():
    return html.Div([
        html.H2("Global Asset Overview", className="text-white mb-4"),

        # KPIs
        dbc.Row([
            dbc.Col(draw_kpi_card("Total Pipelines Monitored", "1,240", "primary"), width=3),
            dbc.Col(draw_kpi_card("Critical Corrosion Alerts", "14", "danger"), width=3),
            dbc.Col(draw_kpi_card("Inspections This Month", "342", "success"), width=3),
            dbc.Col(draw_kpi_card("AI Accuracy Rate", "98.4%", "info"), width=3),
        ]),

        # Map and Charts
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Geospatial Risk Map"),
                    dbc.CardBody(dcc.Graph(figure=draw_map()))
                ], color="dark", inverse=True)
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Defect Types Distribution"),
                    dbc.CardBody(dcc.Graph(
                        figure=px.pie(
                            names=["Pitting", "Galvanic", "Crevice", "Uniform"],
                            values=[30, 15, 10, 45],
                            hole=0.4,
                            color_discrete_sequence=px.colors.sequential.RdBu
                        ).update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': 'white'})
                    ))
                ], color="dark", inverse=True)
            ], width=4)
        ])
    ])


def analysis_page():
    return html.Div([
        html.H2("AI Corrosion Analysis", className="text-white mb-4"),
        html.P("Upload drone footage or inspection images to detect anomalies.", className="text-muted"),

        dbc.Row([
            # Left Column: Upload & Controls
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Upload(
                            id='upload-image',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Image', className="text-primary")
                            ]),
                            style={
                                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                                'borderWidth': '1px', 'borderStyle': 'dashed',
                                'borderRadius': '5px', 'textAlign': 'center', 'marginBottom': '20px'
                            },
                            multiple=False
                        ),
                        html.Div(id='upload-info', className="mb-3"),
                        dbc.Button("Run AI Diagnostics", id="btn-analyze", color="primary", className="w-100",
                                   n_clicks=0),
                    ])
                ], color="dark", inverse=True, className="mb-3"),

                # Simulation of Analysis Results
                dbc.Collapse(
                    dbc.Card([
                        dbc.CardHeader("Analysis Report"),
                        dbc.CardBody([
                            html.H5("Pitting Corrosion Detected", className="text-danger"),
                            html.Hr(),
                            html.P("Confidence Score: 94.2%"),
                            html.P("Est. Depth: 2.4mm"),
                            html.P("Recommendation: Immediate Maintenance"),
                        ])
                    ], color="dark", inverse=True),
                    id="collapse-result", is_open=False
                )
            ], width=4),

            # Right Column: Visualization
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Visual Inspection"),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-1",
                            type="cube",
                            children=html.Div(id="image-display-area")
                        )
                    ])
                ], color="dark", inverse=True)
            ], width=8)
        ])
    ])


# ---------------------------------------------------------
# 5. Callbacks (Logic)
# ---------------------------------------------------------

# Navigation Callback
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/analyze":
        return analysis_page()
    return dashboard_page()


# Simulation of AI Analysis
@app.callback(
    [Output("image-display-area", "children"),
     Output("collapse-result", "is_open")],
    [Input("upload-image", "contents"),
     Input("btn-analyze", "n_clicks")],
    [State("upload-image", "filename")]
)
def update_output(contents, n_clicks, filename):
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Default State (Empty)
    empty_graph = html.Div(
        html.I(className="fas fa-image fa-5x text-muted"),
        style={'height': '400px', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center',
               'backgroundColor': '#222'}
    )

    if not contents:
        return empty_graph, False

    # If User clicks analyze, show the "AI Processed" version
    if trigger_id == "btn-analyze" and n_clicks > 0:
        # SIMULATION: In a real app, you would send 'contents' to your ML model here.
        # We will simulate a bounding box using a Plotly Graph with the image as background.

        fig = go.Figure()

        # Add the image (hidden behind grid, but layout logic places it)
        # Note: In a real scenario, use px.imshow or layout images.
        # For this prototype, we simply display the Gauge and a placeholder visual.

        return html.Div([
            html.Div("Processing Complete", className="text-success mb-2 fw-bold"),
            html.Img(src=contents, style={'maxWidth': '100%', 'border': '2px solid red'}),
            dcc.Graph(figure=draw_gauge(88))  # Mock high severity
        ]), True

    # If just uploaded but not analyzed yet
    return html.Div([
        html.Div(f"Preview: {filename}", className="text-muted mb-2"),
        html.Img(src=contents, style={'maxWidth': '100%', 'opacity': '0.6'})
    ]), False


# ---------------------------------------------------------
# 6. Run
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)

    #gemini3