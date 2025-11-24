# app.py
import base64
import io
from datetime import datetime

import dash
from dash import Dash, html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from PIL import Image

# -------------------------------------------------------------------
# Fake / seed data for prototype
# -------------------------------------------------------------------
initial_inspections = [
    {
        "InspectionID": "AI-0001",
        "Asset": "Gas Export Line 14\"",
        "Location": "KP 12+300",
        "Environment": "Offshore",
        "Severity": "Medium",
        "Probability": 0.63,
        "Inspector": "J. Smith",
        "Timestamp": "2025-03-18 10:15",
    },
    {
        "InspectionID": "AI-0002",
        "Asset": "Produced Water Line 6\"",
        "Location": "Module B - Cellar Deck",
        "Environment": "Offshore",
        "Severity": "High",
        "Probability": 0.82,
        "Inspector": "Automata AI",
        "Timestamp": "2025-03-19 14:42",
    },
    {
        "InspectionID": "AI-0003",
        "Asset": "Crude Transfer Line 20\"",
        "Location": "Tank Farm - Bay 3",
        "Environment": "Onshore",
        "Severity": "Low",
        "Probability": 0.28,
        "Inspector": "R. Lee",
        "Timestamp": "2025-03-20 09:02",
    },
]

severity_to_score = {
    "None/Very Low": 0,
    "Low": 1,
    "Medium": 2,
    "High": 3,
}


# -------------------------------------------------------------------
# Simple mock "AI model" – just for prototype/demo
# -------------------------------------------------------------------
def estimate_corrosion_probability(contents: str):
    """
    Very rough heuristic to simulate an AI corrosion model.
    Uses image darkness as a proxy for corrosion probability.
    """
    if contents is None:
        return 0.0, "None/Very Low"

    try:
        header, encoded = contents.split(",", 1)
        decoded = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(decoded)).convert("L")  # grayscale
        arr = np.array(img).astype("float32") / 255.0

        # Darker images -> higher "corrosion probability"
        darkness = 1.0 - arr.mean()
        prob = float(np.clip(0.2 + darkness * 0.8, 0.01, 0.99))
    except Exception:
        # Fallback if anything goes wrong
        prob = 0.5

    if prob < 0.3:
        severity = "None/Very Low"
    elif prob < 0.5:
        severity = "Low"
    elif prob < 0.7:
        severity = "Medium"
    else:
        severity = "High"

    return prob, severity


def make_gauge(probability: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={"text": "Corrosion Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ff4136"},
                "steps": [
                    {"range": [0, 30], "color": "#2ecc71"},
                    {"range": [30, 60], "color": "#f1c40f"},
                    {"range": [60, 100], "color": "#e74c3c"},
                ],
            },
        )
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=0),
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8f9fa"),
    )
    return fig


def make_trend(records) -> go.Figure:
    df = pd.DataFrame(records)
    if df.empty:
        return go.Figure()

    # Map severity to numeric for plotting
    df["SeverityScore"] = df["Severity"].map(severity_to_score)
    df = df.sort_values("Timestamp")

    fig = px.line(
        df,
        x="Timestamp",
        y="SeverityScore",
        color="Asset",
        markers=True,
        title="Corrosion Severity Over Time",
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8f9fa"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_yaxes(
        tickvals=list(severity_to_score.values()),
        ticktext=list(severity_to_score.keys()),
        title="Severity",
    )
    fig.update_xaxes(title="Inspection Time")
    return fig


# -------------------------------------------------------------------
# Dash app setup
# -------------------------------------------------------------------
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Automata Intelligence | Corrosion AI Prototype",
)
server = app.server

trend_figure_initial = make_trend(initial_inspections)

table_columns = [
    {"name": "Inspection ID", "id": "InspectionID"},
    {"name": "Asset", "id": "Asset"},
    {"name": "Location", "id": "Location"},
    {"name": "Env.", "id": "Environment"},
    {"name": "Severity", "id": "Severity"},
    {"name": "Prob.", "id": "Probability"},
    {"name": "Inspector", "id": "Inspector"},
    {"name": "Timestamp", "id": "Timestamp"},
]

app.layout = dbc.Container(
    [
        dcc.Store(id="inspection-store", data=initial_inspections),

        # Header / Navbar
        dbc.Navbar(
            dbc.Container(
                [
                    dbc.NavbarBrand(
                        [
                            html.Span("Automata Intelligence", className="fw-bold"),
                            html.Span("  |  Corrosion AI Prototype", className="ms-2"),
                        ]
                    ),
                    dbc.Nav(
                        [
                            dbc.NavItem(
                                dbc.NavLink(
                                    "Oil & Gas Corrosion Monitoring",
                                    href="#",
                                    active=True,
                                )
                            )
                        ],
                        className="ms-auto",
                        navbar=True,
                    ),
                ]
            ),
            color="primary",
            dark=True,
            sticky="top",
        ),

        html.Br(),

        # Main content
        dbc.Row(
            [
                # LEFT: Input / upload panel
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("New Inspection", className="card-title"),
                                html.P(
                                    "Upload a pipeline / equipment image and run the "
                                    "prototype corrosion analysis.",
                                    className="text-muted",
                                ),
                                dcc.Upload(
                                    id="upload-image",
                                    children=html.Div(
                                        [
                                            "Drag & Drop or ",
                                            html.A("Select inspection image"),
                                        ]
                                    ),
                                    style={
                                        "width": "100%",
                                        "height": "120px",
                                        "lineHeight": "120px",
                                        "borderWidth": "1px",
                                        "borderStyle": "dashed",
                                        "borderRadius": "5px",
                                        "textAlign": "center",
                                        "margin-bottom": "10px",
                                    },
                                    multiple=False,
                                ),
                                html.Div(
                                    [
                                        html.Small(
                                            "Supported: jpg, jpeg, png. "
                                            "This prototype runs a heuristic, "
                                            "not a production model.",
                                            className="text-muted",
                                        )
                                    ],
                                    className="mb-3",
                                ),
                                html.Img(
                                    id="image-preview",
                                    style={
                                        "maxWidth": "100%",
                                        "maxHeight": "260px",
                                        "borderRadius": "4px",
                                        "display": "block",
                                    },
                                ),
                                html.Hr(),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dbc.Label("Asset / Line"),
                                                dbc.Input(
                                                    id="asset-name-input",
                                                    placeholder='e.g., "Gas Export Line 14\""',
                                                    type="text",
                                                    value="Gas Export Line 14\"",
                                                ),
                                            ],
                                            md=6,
                                        ),
                                        dbc.Col(
                                            [
                                                dbc.Label("Location"),
                                                dbc.Input(
                                                    id="location-input",
                                                    placeholder="e.g., KP 12+300",
                                                    type="text",
                                                    value="KP 12+300",
                                                ),
                                            ],
                                            md=6,
                                        ),
                                    ],
                                    className="mb-2",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dbc.Label("Environment"),
                                                dcc.Dropdown(
                                                    id="environment-dropdown",
                                                    options=[
                                                        {
                                                            "label": "Onshore (plant / tank farm)",
                                                            "value": "Onshore",
                                                        },
                                                        {
                                                            "label": "Offshore topside",
                                                            "value": "Offshore",
                                                        },
                                                        {
                                                            "label": "Subsea / splash zone",
                                                            "value": "Subsea",
                                                        },
                                                        {
                                                            "label": "Refinery / processing",
                                                            "value": "Refinery",
                                                        },
                                                    ],
                                                    value="Offshore",
                                                    clearable=False,
                                                    style={"color": "#000"},
                                                ),
                                            ],
                                            md=6,
                                        ),
                                        dbc.Col(
                                            [
                                                dbc.Label("Inspector / Operator"),
                                                dbc.Input(
                                                    id="inspector-input",
                                                    placeholder="e.g., J. Smith",
                                                    type="text",
                                                ),
                                            ],
                                            md=6,
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                dbc.Button(
                                    "Run corrosion analysis",
                                    id="analyze-btn",
                                    color="primary",
                                    className="mt-1",
                                    n_clicks=0,
                                    style={"width": "100%"},
                                ),
                            ]
                        ),
                        className="shadow-sm",
                    ),
                    md=4,
                ),

                # RIGHT: AI analysis + KPI cards
                dbc.Col(
                    [
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("AI Corrosion Assessment", className="card-title"),
                                    html.Div(
                                        [
                                            dbc.Badge(
                                                "No run yet",
                                                id="risk-badge",
                                                color="secondary",
                                                className="me-2",
                                            ),
                                            html.Span(
                                                id="prediction-text",
                                                children="Awaiting image upload & analysis.",
                                            ),
                                        ],
                                        className="mb-3",
                                    ),
                                    dcc.Graph(
                                        id="probability-gauge",
                                        figure=make_gauge(0.0),
                                        config={"displayModeBar": False},
                                    ),
                                    html.Small(
                                        "This is a technology prototype only. "
                                        "Model outputs are illustrative and must not be used "
                                        "for safety‑critical decisions.",
                                        className="text-muted",
                                    ),
                                ]
                            ),
                            className="mb-3 shadow-sm",
                        ),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5(
                                        "Recent Corrosion Risk Overview",
                                        className="card-title",
                                    ),
                                    dcc.Graph(
                                        id="severity-trend",
                                        figure=trend_figure_initial,
                                        config={"displayModeBar": False},
                                    ),
                                ]
                            ),
                            className="shadow-sm",
                        ),
                    ],
                    md=8,
                ),
            ]
        ),

        html.Br(),

        # Inspection table
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5(
                                "Inspection Log (Prototype Data)", className="card-title"
                            ),
                            dash_table.DataTable(
                                id="inspection-table",
                                columns=table_columns,
                                data=initial_inspections,
                                sort_action="native",
                                page_size=7,
                                style_header={
                                    "backgroundColor": "#1f2630",
                                    "color": "white",
                                    "fontWeight": "bold",
                                },
                                style_cell={
                                    "backgroundColor": "#111111",
                                    "color": "#f8f9fa",
                                    "padding": "6px",
                                    "fontSize": 12,
                                    "border": "1px solid #222",
                                },
                                style_table={"overflowX": "auto"},
                                style_data_conditional=[
                                    {
                                        "if": {"filter_query": "{Severity} = 'High'"},
                                        "backgroundColor": "#7f1d1d",
                                    },
                                    {
                                        "if": {"filter_query": "{Severity} = 'Medium'"},
                                        "backgroundColor": "#7f6f1d",
                                    },
                                ],
                            ),
                        ]
                    ),
                    className="shadow-sm",
                ),
                md=12,
            )
        ),

        html.Br(),
    ],
    fluid=True,
)


# -------------------------------------------------------------------
# Callbacks
# -------------------------------------------------------------------
@app.callback(
    Output("image-preview", "src"),
    Input("upload-image", "contents"),
)
def update_preview(contents):
    # Show uploaded image as-is
    return contents


@app.callback(
    Output("inspection-store", "data"),
    Output("prediction-text", "children"),
    Output("risk-badge", "children"),
    Output("risk-badge", "color"),
    Output("probability-gauge", "figure"),
    Output("inspection-table", "data"),
    Output("severity-trend", "figure"),
    Input("analyze-btn", "n_clicks"),
    State("upload-image", "contents"),
    State("upload-image", "filename"),
    State("asset-name-input", "value"),
    State("location-input", "value"),
    State("environment-dropdown", "value"),
    State("inspector-input", "value"),
    State("inspection-store", "data"),
)
def run_analysis(
    n_clicks,
    contents,
    filename,
    asset,
    location,
    environment,
    inspector,
    records,
):
    if records is None:
        records = initial_inspections.copy()

    # If no analysis yet or no image, keep the existing state
    if not n_clicks or contents is None:
        trend_fig = make_trend(records)
        return (
            records,
            "Awaiting image upload & analysis.",
            "No run yet",
            "secondary",
            make_gauge(0.0),
            records,
            trend_fig,
        )

    # Simulated AI model
    prob, severity = estimate_corrosion_probability(contents)

    # Build human-readable text
    prediction_text = (
        f"Predicted corrosion severity: {severity} "
        f"(probability {prob*100:.1f}%)."
    )

    # Map severity to a badge color
    if severity in ["High"]:
        badge_color = "danger"
    elif severity in ["Medium"]:
        badge_color = "warning"
    elif severity in ["Low", "None/Very Low"]:
        badge_color = "success"
    else:
        badge_color = "secondary"

    # Create new inspection record
    new_id = f"AI-{len(records) + 1:04d}"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    asset_val = asset or "Unknown asset"
    loc_val = location or "Unknown location"
    env_val = environment or "Unspecified"
    insp_val = inspector or "Automata AI"

    new_record = {
        "InspectionID": new_id,
        "Asset": asset_val,
        "Location": loc_val,
        "Environment": env_val,
        "Severity": severity,
        "Probability": round(prob, 2),
        "Inspector": insp_val,
        "Timestamp": now_str,
    }

    updated_records = records + [new_record]
    trend_fig = make_trend(updated_records)
    gauge_fig = make_gauge(prob)

    return (
        updated_records,
        prediction_text,
        severity,
        badge_color,
        gauge_fig,
        updated_records,
        trend_fig,
    )


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
    # 5.1 high gpt