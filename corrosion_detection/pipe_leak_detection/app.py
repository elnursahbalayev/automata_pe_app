import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import time
from datetime import datetime
from simulation import simulator

# Initialize App with Dark Theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "Automata Intelligence - Leak Detection"

# --- Layout Components ---

def build_header():
    return dbc.Navbar(
        dbc.Container([
            html.A(
                dbc.Row([
                    dbc.Col(html.Img(src="https://img.icons8.com/color/48/000000/oil-industry.png", height="30px")),
                    dbc.Col(dbc.NavbarBrand("Automata Intelligence // PIPE LEAK DETECTION", className="ms-2")),
                ], align="center", className="g-0"),
                href="#",
                style={"textDecoration": "none"},
            ),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            dbc.Collapse(
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("DASHBOARD", active=True, href="#")),
                    dbc.NavItem(dbc.NavLink("CAMERAS", href="#")),
                    dbc.NavItem(dbc.NavLink("SETTINGS", href="#")),
                ], className="ms-auto", navbar=True),
                id="navbar-collapse",
                navbar=True,
            ),
        ]),
        color="dark",
        dark=True,
        className="mb-4 border-bottom border-secondary"
    )

def build_camera_feed():
    return dbc.Card([
        dbc.CardHeader("Thermal Feed - CAM 04 [LIVE]"),
        dbc.CardBody([
            html.Div([
                # Background Image (Thermal)
                html.Img(src="/assets/thermal_bg.png", style={"width": "100%", "height": "100%", "objectFit": "cover"}),
                
                # Leak Overlay (Hidden by default)
                html.Div(id="leak-overlay", className="leak-overlay"),
                
                # Bounding Box (Hidden by default)
                html.Div([
                    html.Span("LEAK DETECTED", className="badge bg-danger position-absolute top-0 start-0 m-1"),
                    html.Span("CONF: 99%", className="badge bg-warning text-dark position-absolute bottom-0 end-0 m-1")
                ], id="bounding-box", className="bounding-box"),
                
                # Status Overlay
                html.Div("AI SCANNING...", className="position-absolute bottom-0 start-0 m-2 text-success small fw-bold"),
                
            ], className="camera-container position-relative")
        ], className="p-0")
    ])

def build_fusion_gauge():
    return dbc.Card([
        dbc.CardHeader("Sensor Fusion Engine"),
        dbc.CardBody([
            dcc.Graph(id="fusion-gauge", style={"height": "250px"}),
            html.Div([
                html.Span("PRESSURE RISK: ", className="text-muted small"),
                html.Span(id="pressure-risk-text", className="fw-bold small"),
                html.Br(),
                html.Span("VISUAL RISK: ", className="text-muted small"),
                html.Span(id="visual-risk-text", className="fw-bold small"),
            ], className="mt-2 text-center")
        ])
    ])

def build_graphs():
    return dbc.Card([
        dbc.CardHeader("Real-time Telemetry"),
        dbc.CardBody([
            dcc.Graph(id="pressure-graph", style={"height": "200px"}, config={'displayModeBar': False}),
            dcc.Graph(id="flow-graph", style={"height": "200px"}, config={'displayModeBar': False}),
        ])
    ])

def build_control_panel():
    return dbc.Card([
        dbc.CardHeader("System Controls"),
        dbc.CardBody([
            dbc.Button("SIMULATE LEAK", id="btn-leak", color="danger", outline=True, className="w-100 mb-2"),
            dbc.Button("RESET SYSTEM", id="btn-reset", color="success", outline=True, className="w-100"),
            html.Hr(),
            html.Div([
                html.H6("Event Log", className="text-muted"),
                html.Div(id="event-log", style={"height": "150px", "overflowY": "scroll", "fontSize": "0.8rem", "color": "#888"})
            ])
        ])
    ])

# --- Main Layout ---
app.layout = html.Div([
    build_header(),
    dbc.Container([
        dbc.Row([
            dbc.Col([
                build_camera_feed(),
                html.Br(),
                build_graphs()
            ], width=8),
            dbc.Col([
                build_fusion_gauge(),
                html.Br(),
                build_control_panel()
            ], width=4)
        ]),
        
        # Interval for updates (1 second)
        dcc.Interval(id="interval-component", interval=1000, n_intervals=0),
        # Store for data history
        dcc.Store(id="data-store", data={"time": [], "pressure": [], "flow": []}),
        # Dummy output for button callbacks
        html.Div(id="dummy-output", style={"display": "none"})
    ], fluid=True)
])

# --- Callbacks ---

@app.callback(
    [Output("pressure-graph", "figure"),
     Output("flow-graph", "figure"),
     Output("fusion-gauge", "figure"),
     Output("leak-overlay", "className"),
     Output("bounding-box", "className"),
     Output("pressure-risk-text", "children"),
     Output("pressure-risk-text", "className"),
     Output("visual-risk-text", "children"),
     Output("visual-risk-text", "className"),
     Output("event-log", "children"),
     Output("data-store", "data")],
    [Input("interval-component", "n_intervals"),
     Input("btn-leak", "n_clicks"),
     Input("btn-reset", "n_clicks")],
    [State("data-store", "data"),
     State("event-log", "children")]
)
def update_metrics(n, btn_leak, btn_reset, stored_data, log_children):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Handle Buttons
    if triggered_id == "btn-leak":
        simulator.toggle_leak(True)
        new_log = html.P(f"[{datetime.now().strftime('%H:%M:%S')}] MANUAL OVERRIDE: LEAK SIMULATED", className="text-danger")
        log_children = [new_log] + (log_children or [])
    elif triggered_id == "btn-reset":
        simulator.toggle_leak(False)
        new_log = html.P(f"[{datetime.now().strftime('%H:%M:%S')}] SYSTEM RESET", className="text-success")
        log_children = [new_log] + (log_children or [])
        
    # Get Data
    data = simulator.update()
    
    # Update History
    max_points = 60
    stored_data["time"].append(datetime.now().strftime('%H:%M:%S'))
    stored_data["pressure"].append(data["pressure"])
    stored_data["flow"].append(data["flow_rate"])
    
    if len(stored_data["time"]) > max_points:
        stored_data["time"] = stored_data["time"][-max_points:]
        stored_data["pressure"] = stored_data["pressure"][-max_points:]
        stored_data["flow"] = stored_data["flow"][-max_points:]
        
    # 1. Pressure Graph
    pressure_fig = go.Figure()
    pressure_fig.add_trace(go.Scatter(
        x=stored_data["time"], y=stored_data["pressure"],
        mode='lines', line=dict(color='#66fcf1', width=2), fill='tozeroy', name='Pressure'
    ))
    pressure_fig.update_layout(
        template="plotly_dark", margin=dict(l=40, r=10, t=10, b=30),
        yaxis=dict(range=[0, 150], title="PSI"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # 2. Flow Graph
    flow_fig = go.Figure()
    flow_fig.add_trace(go.Scatter(
        x=stored_data["time"], y=stored_data["flow"],
        mode='lines', line=dict(color='#45a29e', width=2), name='Flow'
    ))
    flow_fig.update_layout(
        template="plotly_dark", margin=dict(l=40, r=10, t=10, b=30),
        yaxis=dict(range=[4000, 6000], title="BBL/D"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # 3. Fusion Gauge
    prob = data["probability"]
    gauge_color = "#66fcf1" if prob < 0.5 else "#ff0033"
    
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "LEAK PROBABILITY (%)"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': gauge_color},
               'steps': [{'range': [0, 50], 'color': "#1f2833"}, {'range': [50, 100], 'color': "#2b1111"}]}
    ))
    gauge_fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)')
    
    # 4. Visuals & Text
    leak_class = "leak-overlay active" if data["leak_active"] else "leak-overlay"
    bbox_class = "bounding-box active" if data["leak_active"] else "bounding-box"
    
    p_risk = "LOW" if data["pressure"] > 90 else "CRITICAL"
    p_class = "text-success" if data["pressure"] > 90 else "text-danger neon-text-red"
    
    v_risk = "NONE" if not data["leak_active"] else "DETECTED"
    v_class = "text-success" if not data["leak_active"] else "text-danger neon-text-red"
    
    # Auto-log events
    if data["leak_active"] and (not log_children or "LEAK DETECTED" not in str(log_children[0])):
         new_log = html.P(f"[{datetime.now().strftime('%H:%M:%S')}] ALARM: LEAK PROBABILITY > 90%", className="text-danger fw-bold")
         log_children = [new_log] + (log_children or [])

    return (pressure_fig, flow_fig, gauge_fig, leak_class, bbox_class, 
            p_risk, p_class, v_risk, v_class, log_children, stored_data)

if __name__ == "__main__":
    app.run(debug=False)
