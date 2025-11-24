import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


# =============================================================================
# 1. SOPHISTICATED DATA GENERATION ENGINE (O&G SIMULATION)
# =============================================================================
def generate_comprehensive_well_data():
    """
    Generates a realistic dataset including:
    - Wireline/LWD: GR, RES, NPHI, RHOB, DT, PEF, CAL
    - Drilling (MWD): ROP, WOB, RPM, Torque, SPP (Standpipe Pressure)
    - Gas: Total Gas, C1, C2
    - AI Derived: Lithology Class, Anomaly Score, Permeability
    """
    np.random.seed(42)
    depths = np.arange(3000, 3500, 0.5)  # 500m section
    n_points = len(depths)

    # --- Lithology Generator (Hidden Markov Model simulation) ---
    # 0: Shale, 1: Sandstone (Reservoir), 2: Limestone (Tight)
    litho_prob = [0.6, 0.3, 0.1]
    lithologies = np.random.choice([0, 1, 2], size=n_points, p=litho_prob)

    # Smooth lithology to create "beds" rather than random noise
    for i in range(1, n_points - 1):
        if lithologies[i - 1] == lithologies[i + 1]:
            lithologies[i] = lithologies[i - 1]

    # --- Curve Generation based on Lithology ---
    gr = np.zeros(n_points)
    res_deep = np.zeros(n_points)
    nphi = np.zeros(n_points)
    rhob = np.zeros(n_points)
    rop = np.zeros(n_points)
    gas = np.zeros(n_points)

    for i in range(n_points):
        noise = np.random.normal(0, 1)

        if lithologies[i] == 0:  # Shale
            gr[i] = 110 + noise * 10
            res_deep[i] = 2 + np.abs(noise)
            nphi[i] = 0.35 + noise * 0.02
            rhob[i] = 2.55 + noise * 0.03
            rop[i] = 15 + noise * 5  # Slow drilling
            gas[i] = 50 + noise * 10

        elif lithologies[i] == 1:  # Sandstone (Pay Zone)
            gr[i] = 35 + noise * 8
            res_deep[i] = 50 + np.abs(noise) * 20  # Hydrocarbon spike
            nphi[i] = 0.15 + noise * 0.02
            rhob[i] = 2.15 + noise * 0.03  # Gas crossover effect
            rop[i] = 80 + noise * 10  # Fast drilling
            gas[i] = 2000 + noise * 500  # Gas kick

        else:  # Limestone
            gr[i] = 15 + noise * 5
            res_deep[i] = 200 + np.abs(noise) * 50
            nphi[i] = 0.02 + noise * 0.005
            rhob[i] = 2.7 + noise * 0.02
            rop[i] = 5 + noise * 2  # Very hard drilling
            gas[i] = 20 + noise

    # Create DataFrame
    df = pd.DataFrame({
        'DEPTH': depths,
        'GR': gr,  # Gamma Ray
        'RES_DEEP': res_deep,  # Deep Resistivity
        'RES_MED': res_deep * 0.8,
        'RES_SHAL': res_deep * 0.5,
        'NPHI': nphi,  # Neutron Porosity
        'RHOB': rhob,  # Bulk Density
        'DT': 100 - (rhob * 20),  # Sonic (Simulated)
        'PEF': np.where(lithologies == 1, 2, np.where(lithologies == 2, 5, 3)),
        'ROP': rop,  # Rate of Penetration
        'WOB': 20 + np.random.normal(0, 2, n_points),
        'RPM': 120 + np.random.normal(0, 5, n_points),
        'TORQUE': 5 + (rop / 20) + np.random.normal(0, 0.5, n_points),
        'TOTAL_GAS': gas,
        'LITH_CODE': lithologies,
    })

    # --- AI Predictions Simulation ---
    # 1. Permeability (k) using Timur-Coates simulation
    df['AI_PERM'] = (10 ** 4) * (df['NPHI'] ** 4.4) / (1 - df['NPHI']) ** 2
    df['AI_PERM'] = df['AI_PERM'].fillna(0.1)

    # 2. Anomaly Detection (Looking for washout or bad hole)
    # If Density Correction (DRHO) would be high (simulated here by bad GR/Res combo)
    df['AI_ANOMALY'] = np.where((df['GR'] > 100) & (df['ROP'] > 100), 1, 0)  # Fast drilling in shale = washout risk

    return df


def generate_trajectory_data(df):
    """Simulates 3D survey data for the wellbore"""
    md = df['DEPTH'].values
    # Simulate a build-and-turn well
    inc = np.linspace(0, 90, len(md))  # Vertical to Horizontal
    azi = np.linspace(0, 45, len(md))  # Turning North to North-East

    tvd = []
    north = []
    east = []

    curr_tvd, curr_n, curr_e = 0, 0, 0

    for i in range(1, len(md)):
        course_len = md[i] - md[i - 1]
        avg_inc = np.radians((inc[i] + inc[i - 1]) / 2)
        avg_azi = np.radians((azi[i] + azi[i - 1]) / 2)

        curr_tvd += course_len * np.cos(avg_inc)
        curr_n += course_len * np.sin(avg_inc) * np.cos(avg_azi)
        curr_e += course_len * np.sin(avg_inc) * np.sin(avg_azi)

        tvd.append(curr_tvd)
        north.append(curr_n)
        east.append(curr_e)

    # Pad first values
    tvd.insert(0, 0)
    north.insert(0, 0)
    east.insert(0, 0)

    return pd.DataFrame({'MD': md, 'TVD': tvd, 'NORTH': north, 'EAST': east})


# --- Initialize Data ---
df_logs = generate_comprehensive_well_data()
df_survey = generate_trajectory_data(df_logs)

# =============================================================================
# 2. DASH APP SETUP & LAYOUT
# =============================================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "StrataAI - Intelligent Well Analytics"

# Custom CSS for the "Stunning" dark UI
CUSTOM_STYLE = {
    'card-header': {'backgroundColor': '#1a1a1a', 'borderBottom': '1px solid #333', 'fontWeight': 'bold',
                    'color': '#00bc8c'},
    'card-bg': {'backgroundColor': '#0f0f0f', 'border': '1px solid #333'},
    'graph-bg': {'backgroundColor': '#000000'},
}


# --- Components ---

def build_kpi_card(title, id_value, unit, color):
    return dbc.Card([
        dbc.CardBody([
            html.H6(title, className="card-title text-muted", style={'fontSize': '0.8rem'}),
            html.H3(id=id_value, className=f"text-{color}", style={'fontWeight': 'bold'}),
            html.Span(unit, className="text-muted", style={'fontSize': '0.7rem'})
        ])
    ], style=CUSTOM_STYLE['card-bg'], className="mb-3")


# --- Layout ---
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H2(["Strata", html.Span("AI", style={'color': '#00bc8c'})], className="my-3"),
            html.P("Real-time Formation Evaluation & Drilling Intelligence", className="text-muted")
        ], width=8),
        dbc.Col([
            dbc.Button("Generate PDF Report", color="secondary", size="sm", className="mt-4 me-2"),
            dbc.Button("Export LAS", color="success", size="sm", className="mt-4")
        ], width=4, className="text-end")
    ], className="border-bottom mb-4 border-secondary"),

    # KPI Row (Dynamic)
    dbc.Row([
        dbc.Col(build_kpi_card("Current Depth", "kpi-depth", "mMD", "light"), width=3),
        dbc.Col(build_kpi_card("Rate of Penetration", "kpi-rop", "m/hr", "warning"), width=3),
        dbc.Col(build_kpi_card("Total Gas", "kpi-gas", "units", "danger"), width=3),
        dbc.Col(build_kpi_card("AI Net Pay Est.", "kpi-pay", "meters", "success"), width=3),
    ]),

    # Main Dashboard Grid
    dbc.Row([
        # Left Column: Controls & AI Settings
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("AI Control Center", style=CUSTOM_STYLE['card-header']),
                dbc.CardBody([
                    html.Label("Analysis Interval", className="text-white"),
                    dcc.RangeSlider(
                        id='depth-slider',
                        min=df_logs['DEPTH'].min(),
                        max=df_logs['DEPTH'].max(),
                        step=10,
                        value=[3100, 3300],
                        marks={3000: '3000', 3250: '3250', 3500: '3500'},
                        className="mb-4"
                    ),
                    html.Hr(className="border-secondary"),
                    html.Label("Model Sensitivity", className="text-white"),
                    dbc.RadioItems(
                        options=[
                            {"label": "Conservative (P90)", "value": 1},
                            {"label": "Balanced (P50)", "value": 2},
                            {"label": "Aggressive (P10)", "value": 3},
                        ],
                        value=2,
                        className="text-muted mb-3"
                    ),
                    dbc.Checklist(
                        options=[
                            {"label": "Show LWD/MWD Tracks", "value": "LWD"},
                            {"label": "AI Lithology Overlay", "value": "AI"},
                            {"label": "Show Washout Alerts", "value": "CAL"},
                        ],
                        value=["LWD", "AI"],
                        id="track-toggles",
                        switch=True,
                        className="text-white"
                    ),
                    html.Br(),
                    dbc.Alert("Anomaly Detected at 3150m: Porosity/Resistivity Mismatch", color="danger",
                              className="mt-2", style={'fontSize': '0.8rem'})
                ])
            ], style=CUSTOM_STYLE['card-bg'], className="h-100")
        ], width=2),

        # Center Column: The Mega-Log Viewer (Triple Combo + AI)
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Integrated Well Log Viewer", style=CUSTOM_STYLE['card-header']),
                dbc.CardBody([
                    dcc.Loading(
                        dcc.Graph(id='well-log-graph', style={'height': '800px'}, config={'displayModeBar': False})
                    )
                ], style={'padding': '0'})
            ], style=CUSTOM_STYLE['card-bg'])
        ], width=7),

        # Right Column: Crossplots & 3D Nav
        dbc.Col([
            # Crossplot
            dbc.Card([
                dbc.CardHeader("Rock Physics (NPHI-RHOB)", style=CUSTOM_STYLE['card-header']),
                dbc.CardBody(
                    dcc.Graph(id='crossplot-graph', style={'height': '350px'}, config={'displayModeBar': False}))
            ], style=CUSTOM_STYLE['card-bg'], className="mb-3"),

            # 3D Trajectory
            dbc.Card([
                dbc.CardHeader("Reservoir Navigation (3D)", style=CUSTOM_STYLE['card-header']),
                dbc.CardBody(dcc.Graph(id='3d-trajectory', style={'height': '350px'}, config={'displayModeBar': False}))
            ], style=CUSTOM_STYLE['card-bg'])
        ], width=3),
    ], className="g-3")  # Gap 3

], fluid=True, style={'backgroundColor': '#000', 'minHeight': '100vh', 'padding': '20px'})


# =============================================================================
# 3. CALLBACKS & LOGIC
# =============================================================================
@app.callback(
    [Output('well-log-graph', 'figure'),
     Output('crossplot-graph', 'figure'),
     Output('3d-trajectory', 'figure'),
     Output('kpi-depth', 'children'),
     Output('kpi-rop', 'children'),
     Output('kpi-gas', 'children'),
     Output('kpi-pay', 'children')],
    [Input('depth-slider', 'value'),
     Input('track-toggles', 'value')]
)
def update_dashboard(depth_range, toggles):
    min_d, max_d = depth_range

    # Filter Data
    dff = df_logs[(df_logs['DEPTH'] >= min_d) & (df_logs['DEPTH'] <= max_d)]

    # --- 1. Main Log Generation (Subplots) ---
    show_lwd = "LWD" in toggles
    cols = 5 if show_lwd else 4
    col_widths = [0.2, 0.2, 0.2, 0.1, 0.3] if show_lwd else [0.25, 0.25, 0.25, 0.25]
    column_titles = ["Gamma/Lith", "Resistivity", "Porosity/Dens", "AI Perm", "Drilling Mechanics"] if show_lwd else [
        "Gamma/Lith", "Resistivity", "Porosity/Dens", "AI Perm"]

    fig = make_subplots(
        rows=1, cols=cols, shared_yaxes=True, horizontal_spacing=0.02,
        column_width=col_widths,
        subplot_titles=column_titles
    )

    # Track 1: Gamma Ray (Linear)
    fig.add_trace(
        go.Scatter(x=dff['GR'], y=dff['DEPTH'], name='GR', line=dict(color='#2ecc71', width=1), fill='tozerox',
                   fillcolor='rgba(46, 204, 113, 0.2)'), row=1, col=1)
    fig.update_xaxes(title_text="GR (API)", range=[0, 150], row=1, col=1)

    # Track 2: Resistivity (Logarithmic)
    fig.add_trace(go.Scatter(x=dff['RES_DEEP'], y=dff['DEPTH'], name='R_DEEP', line=dict(color='#e74c3c', width=1.5)),
                  row=1, col=2)
    fig.add_trace(
        go.Scatter(x=dff['RES_MED'], y=dff['DEPTH'], name='R_MED', line=dict(color='#e67e22', width=1, dash='dash')),
        row=1, col=2)
    fig.update_xaxes(type="log", title_text="RES (ohm.m)", range=[0, 3], row=1, col=2)  # range is 10^0 to 10^3

    # Track 3: Neutron/Density (The "Crossover")
    fig.add_trace(go.Scatter(x=dff['NPHI'], y=dff['DEPTH'], name='NPHI', line=dict(color='#3498db', width=1)), row=1,
                  col=3)
    # Scaling RHOB to match NPHI for crossover (standard 1.95-2.95 scaling logic)
    # For viz simplicity, we plot on secondary x or just scale visually. Let's use multi-axis logic or just plot raw for prototype.
    fig.add_trace(
        go.Scatter(x=dff['RHOB'], y=dff['DEPTH'], name='RHOB', line=dict(color='#e74c3c', width=1, dash='dot')), row=1,
        col=3)
    fig.update_xaxes(title_text="NPHI (v/v) | RHOB", range=[0.6, 0], row=1, col=3)  # NPHI is standardly reversed

    # Track 4: AI Permeability & Facies
    # Creating a heatmap strip for Lithology
    # We mock a heatmap by using Bar chart with minimal width
    colors = {0: '#7f8c8d', 1: '#f1c40f', 2: '#3498db'}  # Gray (Shale), Yellow (Sand), Blue (Lime)

    # AI Permeability Curve
    fig.add_trace(go.Scatter(x=dff['AI_PERM'], y=dff['DEPTH'], name='AI Perm', line=dict(color='#9b59b6')), row=1,
                  col=4)
    fig.update_xaxes(type="log", title_text="k (mD)", row=1, col=4)

    # Track 5 (Optional): LWD / Drilling Mechanics
    if show_lwd:
        fig.add_trace(go.Scatter(x=dff['ROP'], y=dff['DEPTH'], name='ROP', line=dict(color='#f39c12')), row=1, col=5)
        fig.add_trace(go.Scatter(x=dff['TOTAL_GAS'], y=dff['DEPTH'], name='GAS', line=dict(color='#e74c3c', width=0.5),
                                 fill='tozerox', opacity=0.5), row=1, col=5)
        fig.update_xaxes(title_text="ROP | GAS", row=1, col=5)

    # Universal Layout Settings
    fig.update_layout(
        template='plotly_dark',
        height=800,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_yaxes(autorange="reversed", title_text="Depth (mMD)", showgrid=True, gridcolor='#333', row=1, col=1)
    for i in range(2, cols + 1):
        fig.update_yaxes(showticklabels=False, showgrid=True, gridcolor='#333', range=[max_d, min_d], row=1, col=i)

    # --- 2. Crossplot Generation ---
    cross = go.Figure()
    # Color by lithology code
    cross.add_trace(go.Scatter(
        x=dff['NPHI'], y=dff['RHOB'],
        mode='markers',
        marker=dict(
            size=5,
            color=dff['LITH_CODE'],  # Color by our AI class
            colorscale=[[0, '#7f8c8d'], [0.5, '#f1c40f'], [1, '#3498db']],
            showscale=False
        ),
        text=dff['DEPTH'],
        name='Data Points'
    ))
    cross.update_layout(
        template='plotly_dark',
        xaxis_title="NPHI (v/v)",
        yaxis_title="RHOB (g/cc)",
        xaxis=dict(range=[0.45, -0.05]),  # Standard reverse axis for NPHI
        yaxis=dict(range=[3.0, 1.8]),  # Standard reverse axis for RHOB
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,20,20,0.5)'
    )

    # --- 3. 3D Trajectory ---
    # Filter survey to roughly match depth
    dfs = df_survey[(df_survey['MD'] >= min_d) & (df_survey['MD'] <= max_d)]
    traj = go.Figure(data=[go.Scatter3d(
        x=dfs['EAST'], y=dfs['NORTH'], z=dfs['TVD'],
        mode='lines+markers',
        marker=dict(size=2, color=dfs['MD'], colorscale='Viridis'),
        line=dict(width=4, color='#00bc8c')
    )])
    traj.update_layout(
        template='plotly_dark',
        scene=dict(
            xaxis_title='East',
            yaxis_title='North',
            zaxis_title='TVD',
            zaxis=dict(autorange='reversed'),  # Depth goes down
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
    )

    # --- 4. KPIs ---
    current_depth = f"{max_d:.1f}"
    avg_rop = f"{dff['ROP'].mean():.1f}"
    max_gas = f"{dff['TOTAL_GAS'].max():.0f}"
    # AI Net Pay Calculation (Logic: Sandstone + Porosity > 12% + Res > 20)
    pay_mask = (dff['LITH_CODE'] == 1) & (dff['NPHI'] > 0.12) & (dff['RES_DEEP'] > 20)
    net_pay_val = f"{(pay_mask.sum() * 0.5):.1f}"  # 0.5m sample rate

    return fig, cross, traj, current_depth, avg_rop, max_gas, net_pay_val


if __name__ == '__main__':
    app.run(debug=True)