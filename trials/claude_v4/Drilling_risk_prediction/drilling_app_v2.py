# drilling_risk_app.py - Complete Drilling Risk Prediction Dashboard
import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc
from scipy import signal
import warnings

warnings.filterwarnings('ignore')

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])
app.title = "AUTOMATA INTELLIGENCE Drilling Risk Prediction Suite"


# ============================================================================
# DATA GENERATION FUNCTIONS
# ============================================================================

def generate_drilling_data(rig_id, hours=72):
    """Generate realistic drilling data with various risk scenarios"""
    np.random.seed(hash(rig_id) % 2 ** 32)

    # Time series
    time_points = pd.date_range(
        start=datetime.now() - timedelta(hours=hours),
        end=datetime.now(),
        freq='10min'
    )

    n_points = len(time_points)
    depth = np.cumsum(np.random.uniform(0.5, 3, n_points))  # Cumulative depth

    # Initialize drilling parameters
    wob = np.zeros(n_points)  # Weight on Bit (klbs)
    rpm = np.zeros(n_points)  # Rotations per minute
    rop = np.zeros(n_points)  # Rate of penetration (ft/hr)
    torque = np.zeros(n_points)  # Torque (klb-ft)
    spp = np.zeros(n_points)  # Stand Pipe Pressure (psi)
    mud_weight = np.zeros(n_points)  # Mud Weight (ppg)
    flow_rate = np.zeros(n_points)  # Flow Rate (gpm)
    hook_load = np.zeros(n_points)  # Hook Load (klbs)
    pit_volume = np.zeros(n_points)  # Pit Volume (bbl)
    gas = np.zeros(n_points)  # Gas readings (units)

    # Create drilling scenarios with risk events
    base_wob = 25 + np.random.normal(0, 2, n_points)
    base_rpm = 120 + np.random.normal(0, 10, n_points)
    base_torque = 15 + np.random.normal(0, 2, n_points)
    base_spp = 2500 + np.random.normal(0, 100, n_points)

    # Add drilling events (stuck pipe, lost circulation, etc.)
    events = []

    # Stuck pipe event
    if np.random.random() > 0.5:
        stuck_start = np.random.randint(n_points // 3, 2 * n_points // 3)
        stuck_duration = np.random.randint(10, 30)
        stuck_end = min(stuck_start + stuck_duration, n_points)
        events.append({
            'type': 'Stuck Pipe Risk',
            'start': stuck_start,
            'end': stuck_end,
            'severity': 'High'
        })
        base_torque[stuck_start:stuck_end] *= 1.8
        base_wob[stuck_start:stuck_end] *= 0.7
        base_rpm[stuck_start:stuck_end] *= 0.5

    # Lost circulation event
    if np.random.random() > 0.6:
        lost_start = np.random.randint(n_points // 4, 3 * n_points // 4)
        lost_duration = np.random.randint(15, 40)
        lost_end = min(lost_start + lost_duration, n_points)
        events.append({
            'type': 'Lost Circulation',
            'start': lost_start,
            'end': lost_end,
            'severity': 'Medium'
        })
        base_spp[lost_start:lost_end] *= 0.6
        pit_volume[lost_start:lost_end] = -np.cumsum(np.random.uniform(0.1, 0.5, lost_end - lost_start))

    # Apply smooth transitions
    wob = signal.savgol_filter(base_wob, window_length=min(11, n_points if n_points % 2 == 1 else n_points - 1),
                               polyorder=3)
    rpm = signal.savgol_filter(base_rpm, window_length=min(11, n_points if n_points % 2 == 1 else n_points - 1),
                               polyorder=3)
    torque = signal.savgol_filter(base_torque, window_length=min(11, n_points if n_points % 2 == 1 else n_points - 1),
                                  polyorder=3)
    spp = signal.savgol_filter(base_spp, window_length=min(11, n_points if n_points % 2 == 1 else n_points - 1),
                               polyorder=3)

    # Calculate ROP based on WOB and RPM (simplified model)
    rop = (wob * rpm) / (100 + depth / 100) + np.random.normal(0, 2, n_points)
    rop = np.clip(rop, 5, 150)

    # Set other parameters
    mud_weight = 10.5 + np.random.normal(0, 0.2, n_points)
    flow_rate = 500 + np.random.normal(0, 20, n_points)
    hook_load = wob + 150 + np.random.normal(0, 5, n_points)
    pit_volume = 500 + np.cumsum(np.random.normal(0, 0.1, n_points))
    gas = np.random.lognormal(2, 0.5, n_points)

    # Apply bounds
    wob = np.clip(wob, 5, 50)
    rpm = np.clip(rpm, 20, 200)
    torque = np.clip(torque, 5, 40)
    spp = np.clip(spp, 500, 5000)
    mud_weight = np.clip(mud_weight, 8, 18)
    flow_rate = np.clip(flow_rate, 200, 800)

    df = pd.DataFrame({
        'TIME': time_points,
        'DEPTH': depth,
        'WOB': wob,
        'RPM': rpm,
        'ROP': rop,
        'TORQUE': torque,
        'SPP': spp,
        'MUD_WEIGHT': mud_weight,
        'FLOW_RATE': flow_rate,
        'HOOK_LOAD': hook_load,
        'PIT_VOLUME': pit_volume,
        'GAS': gas,
        'RIG_ID': rig_id
    })

    return df, events


def calculate_risk_scores(df):
    """Calculate various drilling risk scores based on parameters"""
    risks = {}

    # Stuck Pipe Risk (based on torque, WOB, and RPM patterns)
    torque_anomaly = (df['TORQUE'] > df['TORQUE'].mean() + 2 * df['TORQUE'].std()).astype(int)
    wob_drop = (df['WOB'] < df['WOB'].mean() - 1.5 * df['WOB'].std()).astype(int)
    rpm_drop = (df['RPM'] < df['RPM'].mean() - 1.5 * df['RPM'].std()).astype(int)
    stuck_pipe_risk = (torque_anomaly * 0.4 + wob_drop * 0.3 + rpm_drop * 0.3) * 100
    risks['stuck_pipe'] = np.clip(stuck_pipe_risk, 0, 100).mean()

    # Lost Circulation Risk (based on SPP and pit volume)
    spp_drop = (df['SPP'] < df['SPP'].mean() - 2 * df['SPP'].std()).astype(int)
    pit_loss = (df['PIT_VOLUME'].diff() < -0.5).astype(int)
    lost_circ_risk = (spp_drop * 0.5 + pit_loss * 0.5) * 100
    risks['lost_circulation'] = np.clip(lost_circ_risk, 0, 100).mean()

    # Kick Risk (based on gas readings and flow rate)
    gas_anomaly = (df['GAS'] > df['GAS'].mean() + 2 * df['GAS'].std()).astype(int)
    flow_anomaly = (df['FLOW_RATE'] > df['FLOW_RATE'].mean() + 1.5 * df['FLOW_RATE'].std()).astype(int)
    kick_risk = (gas_anomaly * 0.6 + flow_anomaly * 0.4) * 100
    risks['kick'] = np.clip(kick_risk, 0, 100).mean()

    # Wellbore Instability Risk
    mud_weight_var = df['MUD_WEIGHT'].std() / df['MUD_WEIGHT'].mean()
    instability_risk = min(mud_weight_var * 500, 100)
    risks['wellbore_instability'] = instability_risk

    # Overall Risk Score
    risks['overall'] = np.mean([risks['stuck_pipe'], risks['lost_circulation'],
                                risks['kick'], risks['wellbore_instability']])

    return risks


def generate_rig_metadata(rig_id):
    """Generate metadata for a drilling rig"""
    fields = ['Permian Basin', 'Gulf of Mexico', 'North Sea', 'Eagle Ford', 'Bakken']
    operators = ['DrillTech Solutions', 'Global Drilling Corp', 'Deepwater Operations', 'Continental Drilling']
    well_types = ['Vertical', 'Directional', 'Horizontal', 'Extended Reach']

    np.random.seed(hash(rig_id) % 2 ** 32)

    metadata = {
        'rig_id': rig_id,
        'field': np.random.choice(fields),
        'operator': np.random.choice(operators),
        'well_name': f"WELL-{np.random.randint(100, 999)}",
        'well_type': np.random.choice(well_types),
        'target_depth': 10000 + np.random.randint(0, 5000),
        'current_operation': np.random.choice(['Drilling', 'Tripping', 'Circulating', 'Casing']),
        'days_on_well': np.random.randint(5, 30),
        'last_incident': np.random.randint(0, 100)  # Days since last incident
    }

    return metadata


def generate_predictions(df, hours_ahead=24):
    """Generate risk predictions for the next N hours"""
    future_times = pd.date_range(
        start=df['TIME'].iloc[-1],
        periods=hours_ahead * 6,  # 10-minute intervals
        freq='10min'
    )

    predictions = []
    risk_levels = ['Low', 'Medium', 'High', 'Critical']

    for i, time in enumerate(future_times[:12]):  # Next 2 hours detailed
        base_prob = 20 + i * 2  # Increasing risk over time for demo
        predictions.append({
            'time': time,
            'stuck_pipe_prob': min(base_prob + np.random.randint(-10, 20), 100),
            'lost_circ_prob': min(base_prob + np.random.randint(-15, 15), 100),
            'kick_prob': min(base_prob / 2 + np.random.randint(-5, 10), 100),
            'overall_risk': np.random.choice(risk_levels, p=[0.5, 0.3, 0.15, 0.05])
        })

    return pd.DataFrame(predictions)


# ============================================================================
# GENERATE DATA FOR MULTIPLE RIGS
# ============================================================================

rigs_data = {}
rigs_metadata = {}
rigs_events = {}
for rig_id in ['RIG-001', 'RIG-002', 'RIG-003', 'RIG-004', 'RIG-005']:
    rigs_data[rig_id], rigs_events[rig_id] = generate_drilling_data(rig_id)
    rigs_metadata[rig_id] = generate_rig_metadata(rig_id)

# Define color scheme (consistent with well log app)
colors = {
    'background': '#0a0e27',
    'surface': '#1a1f3a',
    'primary': '#00d4ff',
    'secondary': '#ff6b35',
    'success': '#00ff88',
    'warning': '#ffaa00',
    'danger': '#ff3366',
    'text': '#ffffff',
    'text-secondary': '#8892b0'
}

# ============================================================================
# DASH LAYOUT
# ============================================================================

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("AUTOMATA INTELLIGENCE",
                        style={'color': colors['text'], 'fontWeight': 'bold'}),
                html.H1("⚙️ Drilling Risk Prediction Suite",
                        style={'color': colors['text'], 'fontWeight': 'bold'}),
                html.P("Real-Time Risk Analytics & Predictive Drilling Optimization",
                       style={'color': colors['text-secondary'], 'fontSize': '18px'})
            ], style={'textAlign': 'center', 'padding': '30px 0'})
        ])
    ]),

    # Control Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Rig Selection & Monitoring Controls", style={'color': colors['primary']}),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Rig", style={'color': colors['text']}),
                            dcc.Dropdown(
                                id='rig-selector',
                                options=[{'label': f'{rig} - {rigs_metadata[rig]["well_name"]}',
                                          'value': rig} for rig in rigs_data.keys()],
                                value='RIG-001',
                                style={'color': '#000'}
                            )
                        ], md=3),
                        dbc.Col([
                            html.Label("Time Window", style={'color': colors['text']}),
                            dcc.Dropdown(
                                id='time-window',
                                options=[
                                    {'label': 'Last 6 Hours', 'value': 6},
                                    {'label': 'Last 12 Hours', 'value': 12},
                                    {'label': 'Last 24 Hours', 'value': 24},
                                    {'label': 'Last 48 Hours', 'value': 48},
                                    {'label': 'Last 72 Hours', 'value': 72}
                                ],
                                value=24,
                                style={'color': '#000'}
                            )
                        ], md=2),
                        dbc.Col([
                            html.Label("Risk Threshold", style={'color': colors['text']}),
                            dcc.Slider(
                                id='risk-threshold',
                                min=0,
                                max=100,
                                value=60,
                                marks={0: '0%', 25: '25%', 50: '50%', 75: '75%', 100: '100%'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], md=3),
                        dbc.Col([
                            html.Label("Update Frequency", style={'color': colors['text']}),
                            dcc.RadioItems(
                                id='update-freq',
                                options=[
                                    {'label': ' Live', 'value': 'live'},
                                    {'label': ' Manual', 'value': 'manual'}
                                ],
                                value='manual',
                                inline=True,
                                style={'color': colors['text']}
                            )
                        ], md=2),
                        dbc.Col([
                            dbc.Button("Analyze Risks", id='analyze-btn', color='primary',
                                       className='w-100', style={'marginTop': '25px'}, n_clicks=0)
                        ], md=2)
                    ])
                ])
            ], style={'backgroundColor': colors['surface'], 'marginBottom': '20px'})
        ])
    ]),

    # Alert Banner
    dbc.Row([
        dbc.Col([
            html.Div(id='alert-banner')
        ])
    ], className='mb-3'),

    # Key Risk Metrics
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='overall-risk', children='0%', style={'color': colors['danger']}),
                    html.P('Overall Risk', style={'color': colors['text-secondary']}),
                    html.Div(id='risk-trend', style={'fontSize': '12px'})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='stuck-pipe-risk', children='0%', style={'color': colors['warning']}),
                    html.P('Stuck Pipe Risk', style={'color': colors['text-secondary']}),
                    html.Small('Next 2 hrs', style={'color': colors['text-secondary']})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='lost-circ-risk', children='0%', style={'color': colors['primary']}),
                    html.P('Lost Circulation', style={'color': colors['text-secondary']}),
                    html.Small('Next 2 hrs', style={'color': colors['text-secondary']})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='kick-risk', children='0%', style={'color': colors['secondary']}),
                    html.P('Kick Risk', style={'color': colors['text-secondary']}),
                    html.Small('Next 2 hrs', style={'color': colors['text-secondary']})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='current-rop', children='0 ft/hr', style={'color': colors['success']}),
                    html.P('Current ROP', style={'color': colors['text-secondary']}),
                    html.Small('Avg: 0 ft/hr', id='avg-rop', style={'color': colors['text-secondary']})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='current-depth', children='0 ft', style={'color': colors['primary']}),
                    html.P('Current Depth', style={'color': colors['text-secondary']}),
                    html.Small('Target: 0 ft', id='target-depth', style={'color': colors['text-secondary']})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2)
    ], className='mb-4'),

    # Main Visualization Area
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Tabs(id='main-tabs', value='real-time', children=[
                        dcc.Tab(label='Real-Time Parameters', value='real-time',
                                style={'backgroundColor': colors['surface'], 'color': colors['text']},
                                selected_style={'backgroundColor': colors['primary'], 'color': colors['background']}),
                        dcc.Tab(label='Risk Timeline', value='risk-timeline',
                                style={'backgroundColor': colors['surface'], 'color': colors['text']},
                                selected_style={'backgroundColor': colors['primary'], 'color': colors['background']}),
                        dcc.Tab(label='Predictive Analysis', value='predictive',
                                style={'backgroundColor': colors['surface'], 'color': colors['text']},
                                selected_style={'backgroundColor': colors['primary'], 'color': colors['background']})
                    ]),
                    html.Div(id='tab-content', style={'marginTop': '20px'})
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=8),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Risk Heat Map", style={'color': colors['primary'], 'marginBottom': '20px'}),
                    dcc.Graph(id='risk-heatmap', style={'height': '250px'}),
                    html.Hr(),
                    html.H5("AI Confidence", style={'color': colors['primary'], 'marginTop': '20px'}),
                    dcc.Graph(id='ai-confidence', style={'height': '200px'}),
                    html.Hr(),
                    html.H5("Parameter Correlations", style={'color': colors['primary'], 'marginTop': '20px'}),
                    dcc.Graph(id='correlation-plot', style={'height': '200px'})
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=4)
    ], className='mb-4'),

    # Detailed Analysis Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Risk Mitigation Recommendations", style={'color': colors['primary']}),
                    html.Div(id='recommendations-table')
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Historical Events Analysis", style={'color': colors['primary']}),
                    dcc.Graph(id='events-timeline', style={'height': '300px'})
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=6)
    ]),

    # Store components and interval for updates
    dcc.Store(id='processed-data-store'),
    dcc.Interval(id='interval-component', interval=5000, n_intervals=0, disabled=True)

], fluid=True, style={'backgroundColor': colors['background'], 'minHeight': '100vh', 'padding': '20px'})


# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    [Output('processed-data-store', 'data'),
     Output('overall-risk', 'children'),
     Output('stuck-pipe-risk', 'children'),
     Output('lost-circ-risk', 'children'),
     Output('kick-risk', 'children'),
     Output('current-rop', 'children'),
     Output('avg-rop', 'children'),
     Output('current-depth', 'children'),
     Output('target-depth', 'children'),
     Output('risk-trend', 'children'),
     Output('alert-banner', 'children')],
    [Input('analyze-btn', 'n_clicks'),
     Input('interval-component', 'n_intervals')],
    [State('rig-selector', 'value'),
     State('time-window', 'value'),
     State('risk-threshold', 'value')]
)
def process_drilling_data(n_clicks, n_intervals, rig_id, time_window, threshold):
    if n_clicks == 0 and n_intervals == 0:
        return None, '0%', '0%', '0%', '0%', '0 ft/hr', 'Avg: 0 ft/hr', '0 ft', 'Target: 0 ft', '', None

    # Get rig data
    df = rigs_data[rig_id].copy()
    metadata = rigs_metadata[rig_id]

    # Filter by time window
    cutoff_time = df['TIME'].max() - timedelta(hours=time_window)
    df_window = df[df['TIME'] >= cutoff_time].copy()

    # Calculate risk scores
    risks = calculate_risk_scores(df_window)

    # Get predictions
    predictions = generate_predictions(df_window)

    # Current values
    current_rop = df_window['ROP'].iloc[-1]
    avg_rop = df_window['ROP'].mean()
    current_depth = df_window['DEPTH'].iloc[-1]

    # Risk trend
    recent_risk = risks['overall']
    older_window = df[df['TIME'] >= cutoff_time - timedelta(hours=time_window)].copy()
    older_window = older_window[older_window['TIME'] < cutoff_time]
    if len(older_window) > 0:
        older_risk = calculate_risk_scores(older_window)['overall']
        if recent_risk > older_risk:
            trend = html.Span(['📈 Increasing ', f'+{recent_risk - older_risk:.1f}%'],
                              style={'color': colors['danger']})
        else:
            trend = html.Span(['📉 Decreasing ', f'{recent_risk - older_risk:.1f}%'],
                              style={'color': colors['success']})
    else:
        trend = html.Span('➡️ Stable', style={'color': colors['text-secondary']})

    # Generate alerts
    alerts = []
    if risks['overall'] > threshold:
        alerts.append(
            dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                f"HIGH RISK ALERT: Overall risk ({risks['overall']:.0f}%) exceeds threshold ({threshold}%)",
            ], color="danger", dismissable=True)
        )

    if risks['stuck_pipe'] > 70:
        alerts.append(
            dbc.Alert([
                html.I(className="fas fa-tools me-2"),
                f"STUCK PIPE WARNING: Risk level at {risks['stuck_pipe']:.0f}%. Consider reducing WOB and increasing circulation.",
            ], color="warning", dismissable=True)
        )

    # Prepare data for storage
    processed_data = {
        'df': df_window.to_dict('records'),
        'risks': risks,
        'predictions': predictions.to_dict('records'),
        'metadata': metadata,
        'events': rigs_events[rig_id]
    }

    return (
        processed_data,
        f"{risks['overall']:.0f}%",
        f"{predictions['stuck_pipe_prob'].iloc[0]:.0f}%",
        f"{predictions['lost_circ_prob'].iloc[0]:.0f}%",
        f"{predictions['kick_prob'].iloc[0]:.0f}%",
        f"{current_rop:.1f} ft/hr",
        f"Avg: {avg_rop:.1f} ft/hr",
        f"{current_depth:.0f} ft",
        f"Target: {metadata['target_depth']:,} ft",
        trend,
        html.Div(alerts) if alerts else None
    )


@app.callback(
    Output('tab-content', 'children'),
    [Input('main-tabs', 'value'),
     Input('processed-data-store', 'data')]
)
def update_tab_content(active_tab, processed_data):
    if not processed_data:
        return html.Div("No data available. Click 'Analyze Risks' to start.",
                        style={'color': colors['text-secondary'], 'textAlign': 'center', 'padding': '50px'})

    df = pd.DataFrame(processed_data['df'])

    if active_tab == 'real-time':
        # Create real-time parameters plot
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            subplot_titles=('Weight on Bit & RPM', 'Rate of Penetration',
                            'Torque & Stand Pipe Pressure', 'Mud Weight & Gas'),
            vertical_spacing=0.05,
            specs=[[{"secondary_y": True}], [{"secondary_y": False}],
                   [{"secondary_y": True}], [{"secondary_y": True}]]
        )

        # WOB and RPM
        fig.add_trace(
            go.Scatter(x=df['TIME'], y=df['WOB'], name='WOB (klbs)',
                       line=dict(color=colors['primary'], width=2)),
            row=1, col=1, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=df['TIME'], y=df['RPM'], name='RPM',
                       line=dict(color=colors['secondary'], width=2)),
            row=1, col=1, secondary_y=True
        )

        # ROP
        fig.add_trace(
            go.Scatter(x=df['TIME'], y=df['ROP'], name='ROP (ft/hr)',
                       line=dict(color=colors['success'], width=2),
                       fill='tozeroy', fillcolor='rgba(0, 255, 136, 0.1)'),
            row=2, col=1
        )

        # Torque and SPP
        fig.add_trace(
            go.Scatter(x=df['TIME'], y=df['TORQUE'], name='Torque (klb-ft)',
                       line=dict(color=colors['warning'], width=2)),
            row=3, col=1, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=df['TIME'], y=df['SPP'], name='SPP (psi)',
                       line=dict(color=colors['danger'], width=2)),
            row=3, col=1, secondary_y=True
        )

        # Mud Weight and Gas
        fig.add_trace(
            go.Scatter(x=df['TIME'], y=df['MUD_WEIGHT'], name='Mud Weight (ppg)',
                       line=dict(color=colors['primary'], width=2)),
            row=4, col=1, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=df['TIME'], y=df['GAS'], name='Gas (units)',
                       line=dict(color=colors['danger'], width=2, dash='dot')),
            row=4, col=1, secondary_y=True
        )

        fig.update_layout(
            height=600,
            showlegend=True,
            plot_bgcolor=colors['surface'],
            paper_bgcolor=colors['surface'],
            font=dict(color=colors['text']),
            hovermode='x unified'
        )

        fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')

        return dcc.Graph(figure=fig)

    elif active_tab == 'risk-timeline':
        # Create risk timeline
        risks_over_time = []
        window_size = max(len(df) // 20, 1)

        for i in range(0, len(df) - window_size, window_size):
            window = df.iloc[i:i + window_size]
            window_risks = calculate_risk_scores(window)
            risks_over_time.append({
                'time': window['TIME'].iloc[-1],
                'stuck_pipe': window_risks['stuck_pipe'],
                'lost_circulation': window_risks['lost_circulation'],
                'kick': window_risks['kick'],
                'overall': window_risks['overall']
            })

        risk_df = pd.DataFrame(risks_over_time)

        fig = go.Figure()

        for risk_type in ['stuck_pipe', 'lost_circulation', 'kick', 'overall']:
            fig.add_trace(
                go.Scatter(
                    x=risk_df['time'],
                    y=risk_df[risk_type],
                    name=risk_type.replace('_', ' ').title(),
                    mode='lines+markers',
                    line=dict(width=2)
                )
            )

        # Add threshold line
        fig.add_hline(y=60, line_dash="dash", line_color=colors['danger'],
                      annotation_text="Risk Threshold")

        fig.update_layout(
            height=600,
            title="Risk Evolution Over Time",
            xaxis_title="Time",
            yaxis_title="Risk Level (%)",
            showlegend=True,
            plot_bgcolor=colors['surface'],
            paper_bgcolor=colors['surface'],
            font=dict(color=colors['text']),
            hovermode='x unified'
        )

        return dcc.Graph(figure=fig)

    elif active_tab == 'predictive':
        # Create predictive analysis
        predictions = pd.DataFrame(processed_data['predictions'])

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Stuck Pipe Probability', 'Lost Circulation Probability',
                            'Kick Probability', 'Risk Matrix'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )

        # Stuck Pipe
        fig.add_trace(
            go.Scatter(x=predictions['time'], y=predictions['stuck_pipe_prob'],
                       mode='lines+markers', name='Stuck Pipe',
                       line=dict(color=colors['warning'], width=2),
                       fill='tozeroy', fillcolor='rgba(255, 170, 0, 0.1)'),
            row=1, col=1
        )

        # Lost Circulation
        fig.add_trace(
            go.Scatter(x=predictions['time'], y=predictions['lost_circ_prob'],
                       mode='lines+markers', name='Lost Circulation',
                       line=dict(color=colors['primary'], width=2),
                       fill='tozeroy', fillcolor='rgba(0, 212, 255, 0.1)'),
            row=1, col=2
        )

        # Kick
        fig.add_trace(
            go.Scatter(x=predictions['time'], y=predictions['kick_prob'],
                       mode='lines+markers', name='Kick',
                       line=dict(color=colors['danger'], width=2),
                       fill='tozeroy', fillcolor='rgba(255, 51, 102, 0.1)'),
            row=2, col=1
        )

        # Risk Matrix
        risk_matrix = np.random.rand(5, 5) * 100
        fig.add_trace(
            go.Heatmap(
                z=risk_matrix,
                colorscale=[[0, colors['success']], [0.5, colors['warning']], [1, colors['danger']]],
                showscale=True,
                text=np.round(risk_matrix, 0),
                texttemplate='%{text}',
                textfont={"size": 10}
            ),
            row=2, col=2
        )

        fig.update_layout(
            height=600,
            showlegend=False,
            plot_bgcolor=colors['surface'],
            paper_bgcolor=colors['surface'],
            font=dict(color=colors['text'])
        )

        fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')

        return dcc.Graph(figure=fig)


@app.callback(
    Output('risk-heatmap', 'figure'),
    [Input('processed-data-store', 'data')]
)
def update_risk_heatmap(processed_data):
    if not processed_data:
        return go.Figure()

    risks = processed_data['risks']

    # Create risk matrix
    risk_categories = ['Stuck Pipe', 'Lost Circ.', 'Kick', 'Instability', 'Overall']
    risk_values = [
        [risks['stuck_pipe'], 0, 0, 0, 0],
        [0, risks['lost_circulation'], 0, 0, 0],
        [0, 0, risks['kick'], 0, 0],
        [0, 0, 0, risks['wellbore_instability'], 0],
        [0, 0, 0, 0, risks['overall']]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=risk_values,
        x=['Current'] * 5,
        y=risk_categories,
        colorscale=[[0, colors['success']], [0.5, colors['warning']], [1, colors['danger']]],
        showscale=False,
        text=[[f"{risks['stuck_pipe']:.0f}%"],
              [f"{risks['lost_circulation']:.0f}%"],
              [f"{risks['kick']:.0f}%"],
              [f"{risks['wellbore_instability']:.0f}%"],
              [f"{risks['overall']:.0f}%"]],
        texttemplate='%{text}',
        textfont={"size": 14, "color": "white"}
    ))

    fig.update_layout(
        margin=dict(t=10, b=10, l=80, r=10),
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['surface'],
        font=dict(color=colors['text'])
    )

    return fig


@app.callback(
    Output('ai-confidence', 'figure'),
    [Input('processed-data-store', 'data')]
)
def update_ai_confidence(processed_data):
    if not processed_data:
        return go.Figure()

    # Simulated AI confidence scores
    categories = ['Data Quality', 'Model Accuracy', 'Prediction Confidence']
    values = [85 + np.random.randint(-5, 10),
              90 + np.random.randint(-5, 5),
              88 + np.random.randint(-3, 7)]

    fig = go.Figure(data=[
        go.Bar(
            x=values,
            y=categories,
            orientation='h',
            marker=dict(
                color=values,
                colorscale=[[0, colors['danger']], [0.7, colors['warning']], [1, colors['success']]],
                cmin=60,
                cmax=100
            ),
            text=[f'{v}%' for v in values],
            textposition='inside',
            textfont=dict(color='white', size=12)
        )
    ])

    fig.update_layout(
        xaxis=dict(range=[0, 100], showgrid=False),
        yaxis=dict(showgrid=False),
        margin=dict(t=10, b=10, l=100, r=10),
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['surface'],
        font=dict(color=colors['text']),
        height=200
    )

    return fig


@app.callback(
    Output('correlation-plot', 'figure'),
    [Input('processed-data-store', 'data')]
)
def update_correlation_plot(processed_data):
    if not processed_data:
        return go.Figure()

    df = pd.DataFrame(processed_data['df'])

    # Create correlation scatter
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df['WOB'],
            y=df['ROP'],
            mode='markers',
            marker=dict(
                color=df['TORQUE'],
                colorscale='Viridis',
                size=5,
                colorbar=dict(title="Torque", len=0.5, y=0.5)
            ),
            text=[f"Depth: {d:.0f}ft" for d in df['DEPTH']],
            hovertemplate="WOB: %{x:.1f}<br>ROP: %{y:.1f}<br>%{text}<extra></extra>"
        )
    )

    fig.update_layout(
        xaxis_title="WOB (klbs)",
        yaxis_title="ROP (ft/hr)",
        margin=dict(t=10, b=40, l=40, r=10),
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['surface'],
        font=dict(color=colors['text']),
        height=200
    )

    return fig


@app.callback(
    Output('recommendations-table', 'children'),
    [Input('processed-data-store', 'data')]
)
def update_recommendations(processed_data):
    if not processed_data:
        return html.Div()

    risks = processed_data['risks']

    recommendations = []

    if risks['stuck_pipe'] > 60:
        recommendations.append({
            'Risk Type': 'Stuck Pipe',
            'Severity': 'High' if risks['stuck_pipe'] > 80 else 'Medium',
            'Action': 'Reduce WOB to 15-20 klbs',
            'Secondary': 'Increase RPM to 140-160',
            'Priority': 1
        })

    if risks['lost_circulation'] > 50:
        recommendations.append({
            'Risk Type': 'Lost Circulation',
            'Severity': 'High' if risks['lost_circulation'] > 75 else 'Medium',
            'Action': 'Add LCM to mud system',
            'Secondary': 'Reduce mud weight by 0.5 ppg',
            'Priority': 2
        })

    if risks['kick'] > 40:
        recommendations.append({
            'Risk Type': 'Kick',
            'Severity': 'Critical' if risks['kick'] > 70 else 'Medium',
            'Action': 'Increase mud weight',
            'Secondary': 'Monitor gas readings closely',
            'Priority': 1
        })

    if not recommendations:
        recommendations.append({
            'Risk Type': 'All Systems',
            'Severity': 'Low',
            'Action': 'Continue current operations',
            'Secondary': 'Maintain monitoring',
            'Priority': 3
        })

    return dash_table.DataTable(
        data=recommendations,
        columns=[{'name': col, 'id': col} for col in recommendations[0].keys()],
        style_cell={
            'textAlign': 'left',
            'backgroundColor': colors['surface'],
            'color': colors['text'],
            'border': f'1px solid {colors["primary"]}40'
        },
        style_header={
            'backgroundColor': colors['primary'],
            'color': colors['background'],
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'column_id': 'Severity', 'filter_query': '{Severity} = "Critical"'},
                'backgroundColor': colors['danger'],
                'color': 'white',
            },
            {
                'if': {'column_id': 'Severity', 'filter_query': '{Severity} = "High"'},
                'backgroundColor': colors['warning'],
                'color': 'white',
            }
        ],
        sort_action="native"
    )


@app.callback(
    Output('events-timeline', 'figure'),
    [Input('processed-data-store', 'data')]
)
def update_events_timeline(processed_data):
    if not processed_data:
        return go.Figure()

    # Create a Gantt-like chart for events
    events = processed_data['events']
    df = pd.DataFrame(processed_data['df'])

    fig = go.Figure()

    # Add drilling parameters background
    fig.add_trace(
        go.Scatter(
            x=df['TIME'],
            y=df['DEPTH'],
            mode='lines',
            name='Depth',
            line=dict(color=colors['primary'], width=2),
            yaxis='y2'
        )
    )

    # Add event markers
    colors_map = {
        'High': colors['danger'],
        'Medium': colors['warning'],
        'Low': colors['success']
    }

    for event in events:
        start_time = df['TIME'].iloc[event['start']] if event['start'] < len(df) else df['TIME'].iloc[-1]
        end_time = df['TIME'].iloc[event['end'] - 1] if event['end'] <= len(df) else df['TIME'].iloc[-1]

        fig.add_vrect(
            x0=start_time, x1=end_time,
            fillcolor=colors_map.get(event['severity'], colors['warning']),
            opacity=0.2,
            line_width=0,
            annotation_text=event['type'],
            annotation_position="top left"
        )

    fig.update_layout(
        title="Historical Events & Depth Progress",
        xaxis_title="Time",
        yaxis=dict(title="Events", showticklabels=False),
        yaxis2=dict(title="Depth (ft)", overlaying='y', side='right'),
        height=300,
        showlegend=True,
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['surface'],
        font=dict(color=colors['text']),
        hovermode='x unified'
    )

    return fig


@app.callback(
    Output('interval-component', 'disabled'),
    [Input('update-freq', 'value')]
)
def toggle_interval(freq):
    return freq != 'live'


# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == '__main__':
    app.run(debug=False, port=8051, host='127.0.0.1')