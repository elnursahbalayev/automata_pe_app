import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import math


# ------------------------------------------------------------------------------
# 1. MOCK DATA GENERATION
# ------------------------------------------------------------------------------
def generate_mock_data():
    """Generates mock sensor data and 3D pipe coordinates for the prototype."""
    # Simulation parameters
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

    # Generate time-series data for 3 pipeline segments
    segments = ['PIPE-A101', 'PIPE-B205', 'PIPE-C309']
    data_list = []

    for seg in segments:
        base_thickness = 12.0  # mm
        # Random corrosion rate + noise
        decay = np.linspace(0, np.random.uniform(0.5, 2.0), 100)
        thickness = base_thickness - decay + np.random.normal(0, 0.05, 100)

        # Pressure readings (fluctuating)
        pressure = np.random.normal(800, 20, 100)  # PSI

        # Temperature
        temp = np.random.normal(60, 5, 100)  # Celsius

        # AI Anomaly Score (0-100%) - rises as thickness drops
        risk = (decay / 2.0) * 100 + np.random.normal(0, 5, 100)
        risk = np.clip(risk, 0, 100)

        for i, date in enumerate(dates):
            data_list.append({
                'Date': date,
                'Segment_ID': seg,
                'Wall_Thickness_mm': thickness[i],
                'Pressure_PSI': pressure[i],
                'Temperature_C': temp[i],
                'AI_Risk_Score': risk[i]
            })

    return pd.DataFrame(data_list)


df = generate_mock_data()


# Generate 3D Cylinder Data for "Digital Twin" visualization
def generate_3d_pipe(segment_id):
    """Generates points for a 3D cylinder visualization with mock corrosion hotspots."""
    # Cylinder parameters
    radius = 5
    height = 20
    resolution = 50

    z = np.linspace(0, height, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid)
    y_grid = radius * np.sin(theta_grid)

    # Flatten arrays
    x = x_grid.flatten()
    y = y_grid.flatten()
    z = z_grid.flatten()

    # Create mock "surface health" (Corrosion Depth)
    # Add some random "hotspots" based on segment ID to make them look different
    np.random.seed(hash(segment_id) % 2 ** 32)
    health = np.zeros_like(x)

    # Create 2 random corrosion pits
    for _ in range(2):
        center_z = np.random.uniform(0, height)
        center_theta = np.random.uniform(0, 2 * np.pi)

        # Distance calculation on cylinder surface approximation
        # Simple Euclidean distance in 3D for the pit influence
        pit_x = radius * np.cos(center_theta)
        pit_y = radius * np.sin(center_theta)

        dist = np.sqrt((x - pit_x) ** 2 + (y - pit_y) ** 2 + (z - center_z) ** 2)

        # Add corrosion depth based on distance (inverse)
        damage = np.exp(-0.5 * dist ** 2) * np.random.uniform(2, 5)  # Depth in mm
        health += damage

    return x, y, z, health


# ------------------------------------------------------------------------------
# 2. DASH APP SETUP
# ------------------------------------------------------------------------------
# Using the CYBORG theme for a high-tech "Dark Mode" AI look
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "Automata Intelligence | Corrosion Monitor"

# ------------------------------------------------------------------------------
# 3. LAYOUT
# ------------------------------------------------------------------------------
app.layout = dbc.Container([
    # --- Header ---
    dbc.Row([
        dbc.Col([
            html.H2("AUTOMATA INTELLIGENCE", className="text-primary",
                    style={'fontWeight': 'bold', 'letterSpacing': '2px'}),
            html.H5("AI-Driven Corrosion Detection & Predictive Maintenance", className="text-muted"),
        ], width=8),
        dbc.Col([
            html.Div([
                html.Span("SYSTEM STATUS: ", className="text-light"),
                html.Span("ONLINE", className="text-success",
                          style={'fontWeight': 'bold', 'animation': 'blink 1s infinite'})
            ], style={'textAlign': 'right', 'marginTop': '15px'})
        ], width=4)
    ], className="mb-4 mt-4 border-bottom pb-2"),

    # --- Control Panel & KPIs ---
    dbc.Row([
        # Controls
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Control Panel"),
                dbc.CardBody([
                    html.Label("Select Pipeline Segment:"),
                    dcc.Dropdown(
                        id='segment-dropdown',
                        options=[{'label': i, 'value': i} for i in df['Segment_ID'].unique()],
                        value='PIPE-A101',
                        clearable=False,
                        style={'color': '#000'}  # Fix text color for dark theme
                    ),
                    html.Br(),
                    html.Label("Simulation Range:"),
                    dcc.RangeSlider(
                        min=0, max=100, step=10,
                        value=[0, 100],
                        marks={0: 'Start', 50: 'Mid', 100: 'Now'},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                    html.Br(),
                    dbc.Button("Run AI Diagnostics", id="btn-run-ai", color="primary", className="w-100", n_clicks=0),
                    html.Div(id='ai-status-msg', className="mt-2 text-info small")
                ])
            ], className="mb-4 shadow-sm"),

            # KPI Cards
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("Avg Thickness", className="card-title text-muted"),
                        html.H3(id="kpi-thickness", className="text-light")
                    ])
                ], className="mb-3 border-primary")),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("Risk Level", className="card-title text-muted"),
                        html.H3(id="kpi-risk", className="text-danger")
                    ])
                ], className="mb-3 border-danger"))
            ])
        ], width=12, lg=4),

        # 3D Digital Twin Visualization
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Digital Twin: Surface Topology Scan"),
                dbc.CardBody([
                    dcc.Graph(id='3d-pipe-plot', style={'height': '400px'}),
                    html.Small("* Red zones indicate detected structural mass loss (pitting)", className="text-muted")
                ])
            ], className="h-100 shadow-sm")
        ], width=12, lg=8)
    ], className="mb-4"),

    # --- Detailed Charts ---
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Sensor Fusion: Wall Thickness vs. Pressure"),
                dbc.CardBody(dcc.Graph(id='line-chart', style={'height': '350px'}))
            ])
        ], width=12, lg=6),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("AI Anomaly Probability Density"),
                dbc.CardBody(dcc.Graph(id='heatmap-chart', style={'height': '350px'}))
            ])
        ], width=12, lg=6)
    ])

], fluid=True, style={'paddingBottom': '50px'})


# ------------------------------------------------------------------------------
# 4. CALLBACKS
# ------------------------------------------------------------------------------

@app.callback(
    [Output('line-chart', 'figure'),
     Output('3d-pipe-plot', 'figure'),
     Output('heatmap-chart', 'figure'),
     Output('kpi-thickness', 'children'),
     Output('kpi-risk', 'children'),
     Output('ai-status-msg', 'children')],
    [Input('segment-dropdown', 'value'),
     Input('btn-run-ai', 'n_clicks')]
)
def update_dashboard(selected_segment, n_clicks):
    # Filter data
    filtered_df = df[df['Segment_ID'] == selected_segment]

    # 1. Line Chart: Thickness over Time
    fig_line = px.line(filtered_df, x='Date', y='Wall_Thickness_mm',
                       title=f"Trend Analysis: {selected_segment}",
                       labels={'Wall_Thickness_mm': 'Thickness (mm)'})

    # Add a red threshold line
    fig_line.add_hline(y=10.5, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
    fig_line.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

    # 2. 3D Digital Twin Plot
    x, y, z, health = generate_3d_pipe(selected_segment)

    fig_3d = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=4,
            color=health,  # Color by corrosion depth
            colorscale='RdYlGn_r',  # Red = High Corrosion, Green = Low
            opacity=0.8,
            colorbar=dict(title="Corrosion Depth (mm)")
        )
    )])
    fig_3d.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(title="Pipe Length (m)"),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='rgba(0,0,0,0)'
    )

    # 3. Heatmap / Risk Gauge
    # We'll visualize the relationship between Pressure and AI Risk
    fig_heat = px.density_heatmap(filtered_df, x="Pressure_PSI", y="Wall_Thickness_mm", z="AI_Risk_Score",
                                  nbinsx=20, nbinsy=20, color_continuous_scale="Magma",
                                  title="Risk Correlation Heatmap")
    fig_heat.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

    # 4. KPIs
    avg_thickness = f"{filtered_df['Wall_Thickness_mm'].mean():.2f} mm"
    latest_risk = filtered_df['AI_Risk_Score'].iloc[-1]

    if latest_risk > 80:
        risk_text = "CRITICAL"
        risk_color = "text-danger"  # logic handled in layout class, just passing text here
    elif latest_risk > 50:
        risk_text = "WARNING"
    else:
        risk_text = "STABLE"

    kpi_risk_val = f"{risk_text} ({int(latest_risk)}%)"

    # 5. Button Logic
    ctx = callback_context
    ai_msg = ""
    if ctx.triggered:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'btn-run-ai':
            ai_msg = f"✓ AI Analysis complete for {selected_segment}. 2 Micro-fractures detected."

    return fig_line, fig_3d, fig_heat, avg_thickness, kpi_risk_val, ai_msg


# ------------------------------------------------------------------------------
# 5. RUN SERVER
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, port=8050)