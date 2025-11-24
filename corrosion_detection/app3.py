import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime

# =============================================================================
# Fake data generators (looks extremely realistic in demos)
# =============================================================================
def fake_heatmap():
    np.random.seed(42)
    z = np.random.rand(50, 120) * 100
    # Create a few realistic-looking corrosion "hot spots"
    z[15:25, 30:50] += 60
    z[30:40, 80:100] += 80
    z[10:18, 70:85] += 70
    return z

def fake_severity_series():
    sections = [f"Section {i:02d}" for i in range(1, 21)]
    severity = np.random.beta(2, 5, 20) * 100
    severity[[4, 9, 15]] = [78, 92, 86]  # make a few look critical
    return pd.DataFrame({"Section": sections, "Corrosion %": severity.round(1)})

# =============================================================================
# Dash app
# =============================================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.title = "Automata Intelligence – Corrosion Detection"

app.layout = html.Div([
    # Header
    dbc.NavbarSimple(
        brand="Automata Intelligence",
        brand_href="#",
        color="primary",
        dark=True,
        className="mb-5",
        children=[
            html.Img(src="https://via.placeholder.com/80x40/007bff/ffffff?text=AI", height="40px"),
            html.H3("Corrosion Intelligence Platform", className="text-white ms-3")
        ]
    ),

    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Pipeline Corrosion Detector", className="text-center text-primary mb-4"),
                html.P(
                    "Upload an inspection image or click “Run Demo” to see the AI analysis in action.",
                    className="lead text-center"
                ),

                # Upload + Demo button
                dcc.Upload(
                    id='upload-image',
                    children=dbc.Button(
                        [html.I(className="bi bi-cloud-upload me-2"), "Drop Image Here or Click"],
                        color="light",
                        size="lg",
                        className="w-100"
                    ),
                    multiple=False,
                    className="d-grid mb-3"
                ),
                dbc.Button(
                    "Run Demo (no upload needed)",
                    id="demo-btn",
                    color="success",
                    size="lg",
                    className="w-100"
                ),

                html.Hr(),

                # Fake status cards
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H4("127", className="card-title text-danger"),
                            html.P("Pipelines Analyzed", className="card-text")
                        ])
                    ], color="dark"), width=4),
                    dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H4("68 %", className="card-title text-warning"),
                            html.P("Avg. Corrosion Index", className="card-text")
                        ])
                    ], color="dark"), width=4),
                    dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H4("11", className="card-title text-info"),
                            html.P("Critical Alerts Today", className="card-text")
                        ])
                    ], color="dark"), width=4),
                ], className="mb-4"),

            ], width=12)
        ]),

        # Results area (hidden until analysis)
        html.Div(id="results-area", style={"display": "none"}, children=[

            dbc.Row([
                dbc.Col([
                    html.H4("Original Image"),
                    html.Img(id="original-img", style={"width": "100%", "border": "2px solid #444"})
                ], md=6),
                dbc.Col([
                    html.H4("Corrosion Heatmap (AI Detection)"),
                    dcc.Graph(id="heatmap", style={"height": "500px"})
                ], md=6),
            ], className="mb-5"),

            dbc.Row([
                dbc.Col([
                    html.H4("Corrosion Severity by Pipeline Section"),
                    dcc.Graph(id="severity-bar")
                ], md=7),
                dbc.Col([
                    html.H4("Risk Distribution"),
                    dcc.Graph(id="pie-chart")
                ], md=5),
            ]),

            dbc.Alert([
                html.H4("Executive Summary", className="alert-heading text-danger"),
                html.P(id="summary-text"),
                html.Hr(),
                html.P(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            ], color="danger", className="mt-4")
        ])
    ])
])

# =============================================================================
# Callbacks
# =============================================================================
@callback(
    Output("results-area", "style"),
    Output("heatmap", "figure"),
    Output("severity-bar", "figure"),
    Output("pie-chart", "figure"),
    Output("original-img", "src"),
    Output("summary-text", "children"),
    Input("demo-btn", "n_clicks"),
    Input("upload-image", "contents"),
    State("upload-image", "filename")
)
def run_analysis(demo_clicks, contents, filename):
    # Show results
    style = {"display": "block"}

    # --------------------------------------------------
    # Image handling
    # --------------------------------------------------
    if contents is not None:
        img_src = contents
    else:
        # Default demo pipeline image
        img_src = "https://via.placeholder.com/1200x600/333333/ffffff?text=Pipeline+Inspection+Image"

    # --------------------------------------------------
    # Fake analysis results
    # --------------------------------------------------
    z = fake_heatmap()
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=z,
        colorscale="Hot",
        showscale=True,
        colorbar=dict(title="Corrosion Intensity")
    ))
    heatmap_fig.update_layout(
        title="AI-Detected Corrosion Heatmap",
        paper_bgcolor="#222",
        plot_bgcolor="#222",
        font=dict(color="white")
    )

    df = fake_severity_series()
    bar_fig = px.bar(
        df,
        x="Section",
        y="Corrosion %",
        color="Corrosion %",
        color_continuous_scale="Reds",
        title="Corrosion Severity per Section"
    )
    bar_fig.update_layout(paper_bgcolor="#222", plot_bgcolor="#222", font=dict(color="white"))

    pie_fig = px.pie(
        values=[68, 32],
        names=["Corroded Area", "Healthy"],
        color_discrete_sequence=["#ff4444", "#44aa44"],
        title="Overall Pipeline Health"
    )
    pie_fig.update_layout(paper_bgcolor="#222", plot_bgcolor="#222", font=dict(color="white"))

    summary = (
        "Critical corrosion detected in Sections 05, 10, and 16 (severity > 85%). "
        "Immediate maintenance recommended. Overall pipeline risk level: HIGH."
    )

    return style, heatmap_fig, bar_fig, pie_fig, img_src, summary


if __name__ == "__main__":
    app.run(debug=True, port=8050)