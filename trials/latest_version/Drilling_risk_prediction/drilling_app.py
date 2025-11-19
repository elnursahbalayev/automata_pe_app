# drilling_risk_app.py - Complete Drilling Risk Prediction Dashboard
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc
from scipy.ndimage import gaussian_filter1d

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])
app.title = "AUTOMATA INTELLIGENCE - Drilling Risk Prediction"


# ============================================================================
# DATA GENERATION FUNCTIONS
# ============================================================================

def generate_drilling_data(well_id, num_points=1000):
    """Generate realistic drilling data with various risk scenarios"""
    np.random.seed(hash(well_id) % 2 ** 32)

    # Time series data (hourly data over several days)
    start_time = datetime.now() - timedelta(hours=num_points)
    time = [start_time + timedelta(hours=i) for i in range(num_points)]

    # Depth progression (0 to 15000 ft)
    target_depth = 15000
    depth = np.linspace(0, target_depth, num_points)

    # Add realistic depth variation (stops, slower drilling in hard formations)
    depth_variation = np.zeros(num_points)
    for i in range(1, num_points):
        if np.random.random() > 0.95:  # 5% chance of stop
            depth_variation[i] = depth_variation[i - 1]
        else:
            depth_variation[i] = depth[i] + np.random.normal(0, 10)

    depth = gaussian_filter1d(depth_variation, sigma=5)
    depth = np.clip(depth, 0, target_depth)

    # Rate of Penetration (ROP) - ft/hr
    base_rop = 50 + np.random.normal(0, 5, num_points)

    # Simulate different formations affecting ROP
    formation_zones = [
        (0, 3000, 60),  # Soft formation
        (3000, 6000, 40),  # Medium formation
        (6000, 9000, 25),  # Hard formation
        (9000, 12000, 45),  # Medium formation
        (12000, 15000, 20)  # Very hard formation
    ]

    rop = np.zeros(num_points)
    for i, d in enumerate(depth):
        for start, end, base in formation_zones:
            if start <= d < end:
                rop[i] = base + np.random.normal(0, 5)
                break

    rop = gaussian_filter1d(rop, sigma=3)
    rop = np.clip(rop, 0, 100)

    # Weight on Bit (WOB) - klbs
    wob = 20 + np.random.normal(0, 3, num_points)
    wob = gaussian_filter1d(wob, sigma=2)
    wob = np.clip(wob, 10, 40)

    # Rotary Speed (RPM)
    rpm = 120 + np.random.normal(0, 10, num_points)
    rpm = gaussian_filter1d(rpm, sigma=2)
    rpm = np.clip(rpm, 80, 180)

    # Torque - ft-lbs
    torque = 5000 + wob * 150 + np.random.normal(0, 500, num_points)
    torque = gaussian_filter1d(torque, sigma=2)

    # Standpipe Pressure (SPP) - psi
    spp = 2500 + depth * 0.1 + np.random.normal(0, 100, num_points)
    spp = gaussian_filter1d(spp, sigma=2)

    # Flow Rate - gpm
    flow_rate = 350 + np.random.normal(0, 20, num_points)
    flow_rate = gaussian_filter1d(flow_rate, sigma=2)
    flow_rate = np.clip(flow_rate, 300, 450)

    # Mud Weight - ppg
    mud_weight = 9.5 + (depth / 15000) * 2.5 + np.random.normal(0, 0.1, num_points)
    mud_weight = gaussian_filter1d(mud_weight, sigma=10)

    # Temperature - F
    temperature = 70 + (depth / 100) + np.random.normal(0, 2, num_points)
    temperature = gaussian_filter1d(temperature, sigma=5)

    # Formation Pressure - psi
    formation_pressure = depth * 0.465 + np.random.normal(0, 50, num_points)  # Normal pressure gradient

    # Add overpressure zones
    overpressure_zones = [(8000, 9000), (13000, 14000)]
    for start_d, end_d in overpressure_zones:
        mask = (depth >= start_d) & (depth <= end_d)
        formation_pressure[mask] = depth[mask] * 0.65 + np.random.normal(0, 50, np.sum(mask))

    formation_pressure = gaussian_filter1d(formation_pressure, sigma=5)

    # ECD (Equivalent Circulating Density) - ppg
    ecd = mud_weight + (spp / (depth + 1)) * 0.002

    # Hook Load - klbs
    hook_load = 200 + depth * 0.02 + np.random.normal(0, 10, num_points)
    hook_load = gaussian_filter1d(hook_load, sigma=3)

    # Differential Pressure
    hydrostatic = mud_weight * 0.052 * depth
    diff_pressure = hydrostatic - formation_pressure

    # Create risk events/anomalies
    stuck_pipe_events = np.zeros(num_points)
    kick_events = np.zeros(num_points)
    lost_circulation_events = np.zeros(num_points)

    # Simulate stuck pipe conditions
    for i in range(num_points):
        if diff_pressure[i] > 500 and np.random.random() > 0.98:
            stuck_pipe_events[i] = 1
            # Reduce ROP during stuck pipe
            if i < num_points - 10:
                rop[i:i + 10] *= 0.1

    # Simulate kick indicators
    for i in range(num_points):
        if diff_pressure[i] < -200 and np.random.random() > 0.99:
            kick_events[i] = 1
            if i < num_points - 5:
                spp[i:i + 5] *= 1.2
                flow_rate[i:i + 5] *= 1.15

    # Simulate lost circulation
    for i in range(num_points):
        if diff_pressure[i] > 800 and np.random.random() > 0.98:
            lost_circulation_events[i] = 1
            if i < num_points - 8:
                flow_rate[i:i + 8] *= 0.7
                spp[i:i + 8] *= 0.8

    df = pd.DataFrame({
        'TIMESTAMP': time,
        'DEPTH': depth,
        'ROP': rop,
        'WOB': wob,
        'RPM': rpm,
        'TORQUE': torque,
        'SPP': spp,
        'FLOW_RATE': flow_rate,
        'MUD_WEIGHT': mud_weight,
        'TEMPERATURE': temperature,
        'FORMATION_PRESSURE': formation_pressure,
        'ECD': ecd,
        'HOOK_LOAD': hook_load,
        'DIFF_PRESSURE': diff_pressure,
        'STUCK_PIPE_EVENT': stuck_pipe_events,
        'KICK_EVENT': kick_events,
        'LOST_CIRC_EVENT': lost_circulation_events,
        'WELL_ID': well_id
    })

    return df


def generate_well_info(well_id):
    """Generate well information"""
    np.random.seed(hash(well_id) % 2 ** 32)

    wells_info = {
        'well_id': well_id,
        'well_name': f'{well_id}',
        'field': np.random.choice(['Permian Basin', 'Eagle Ford', 'Bakken', 'Marcellus', 'Haynesville']),
        'rig': f'RIG-{np.random.randint(100, 999)}',
        'operator': np.random.choice(
            ['Global Drilling Inc.', 'DeepDrill Corp', 'PetroTech Drilling', 'Advanced Well Solutions']),
        'spud_date': datetime.now() - timedelta(days=np.random.randint(5, 30)),
        'target_depth': np.random.randint(12000, 18000),
        'well_type': np.random.choice(['Vertical', 'Directional', 'Horizontal']),
        'formation': np.random.choice(['Sandstone', 'Shale', 'Limestone', 'Mixed'])
    }

    return wells_info


# ============================================================================
# RISK CALCULATION FUNCTIONS
# ============================================================================

def calculate_stuck_pipe_risk(df):
    """Calculate stuck pipe risk score (0-100)"""
    risk_scores = np.zeros(len(df))

    # Factors contributing to stuck pipe risk
    # 1. High differential pressure
    diff_pressure_risk = np.clip((df['DIFF_PRESSURE'] - 200) / 10, 0, 40)

    # 2. Low ROP indicates potential problems
    rop_risk = np.clip((50 - df['ROP']) / 2, 0, 25)

    # 3. High torque
    torque_risk = np.clip((df['TORQUE'] - 6000) / 100, 0, 20)

    # 4. Overpull on hook load
    hook_load_risk = np.clip((df['HOOK_LOAD'] - 250) / 10, 0, 15)

    risk_scores = diff_pressure_risk + rop_risk + torque_risk + hook_load_risk

    # Apply smoothing
    risk_scores = gaussian_filter1d(risk_scores, sigma=3)

    return np.clip(risk_scores, 0, 100)


def calculate_kick_risk(df):
    """Calculate kick/well control risk score (0-100)"""
    risk_scores = np.zeros(len(df))

    # Factors contributing to kick risk
    # 1. Negative differential pressure (underbalanced)
    diff_pressure_risk = np.clip((-df['DIFF_PRESSURE']) / 5, 0, 40)

    # 2. Flow rate increase
    flow_increase = df['FLOW_RATE'] - df['FLOW_RATE'].rolling(10, min_periods=1).mean()
    flow_risk = np.clip(flow_increase * 2, 0, 30)

    # 3. Low mud weight relative to formation pressure
    mud_gradient = df['MUD_WEIGHT'] * 0.052 * df['DEPTH']
    pressure_risk = np.clip((df['FORMATION_PRESSURE'] - mud_gradient) / 50, 0, 30)

    risk_scores = diff_pressure_risk + flow_risk + pressure_risk

    risk_scores = gaussian_filter1d(risk_scores, sigma=3)

    return np.clip(risk_scores, 0, 100)


def calculate_lost_circulation_risk(df):
    """Calculate lost circulation risk score (0-100)"""
    risk_scores = np.zeros(len(df))

    # Factors contributing to lost circulation
    # 1. High ECD (overbalanced)
    ecd_risk = np.clip((df['ECD'] - df['MUD_WEIGHT'] - 0.5) * 50, 0, 35)

    # 2. Flow rate decrease
    flow_decrease = df['FLOW_RATE'].rolling(10, min_periods=1).mean() - df['FLOW_RATE']
    flow_risk = np.clip(flow_decrease * 1.5, 0, 35)

    # 3. High differential pressure (overbalanced)
    diff_pressure_risk = np.clip(df['DIFF_PRESSURE'] / 20, 0, 30)

    risk_scores = ecd_risk + flow_risk + diff_pressure_risk

    risk_scores = gaussian_filter1d(risk_scores, sigma=3)

    return np.clip(risk_scores, 0, 100)


def calculate_equipment_risk(df):
    """Calculate equipment failure risk score (0-100)"""
    risk_scores = np.zeros(len(df))

    # Factors contributing to equipment failure
    # 1. High torque variation
    torque_std = df['TORQUE'].rolling(20, min_periods=1).std()
    torque_risk = np.clip(torque_std / 50, 0, 30)

    # 2. High vibration proxy (RPM variation)
    rpm_std = df['RPM'].rolling(20, min_periods=1).std()
    vibration_risk = np.clip(rpm_std * 2, 0, 25)

    # 3. High pressure
    pressure_risk = np.clip((df['SPP'] - 3000) / 100, 0, 25)

    # 4. Temperature
    temp_risk = np.clip((df['TEMPERATURE'] - 200) / 10, 0, 20)

    risk_scores = torque_risk + vibration_risk + pressure_risk + temp_risk

    risk_scores = gaussian_filter1d(risk_scores, sigma=3)

    return np.clip(risk_scores, 0, 100)


def calculate_overall_risk(df):
    """Calculate overall drilling risk"""
    stuck_risk = calculate_stuck_pipe_risk(df)
    kick_risk = calculate_kick_risk(df)
    lost_circ_risk = calculate_lost_circulation_risk(df)
    equip_risk = calculate_equipment_risk(df)

    # Weighted average
    overall = (stuck_risk * 0.3 + kick_risk * 0.35 + lost_circ_risk * 0.2 + equip_risk * 0.15)

    return overall


def generate_recommendations(current_risks):
    """Generate AI-powered recommendations based on risk levels"""
    recommendations = []

    if current_risks['stuck_pipe'] > 60:
        recommendations.append({
            'type': 'CRITICAL',
            'category': 'Stuck Pipe',
            'message': 'HIGH STUCK PIPE RISK: Reduce differential pressure. Consider reducing mud weight or increasing circulation.',
            'action': 'Immediate Action Required'
        })
    elif current_risks['stuck_pipe'] > 40:
        recommendations.append({
            'type': 'WARNING',
            'category': 'Stuck Pipe',
            'message': 'Elevated stuck pipe risk detected. Monitor differential pressure closely.',
            'action': 'Increase Monitoring'
        })

    if current_risks['kick'] > 60:
        recommendations.append({
            'type': 'CRITICAL',
            'category': 'Well Control',
            'message': 'KICK RISK CRITICAL: Potential underbalanced condition. Increase mud weight immediately.',
            'action': 'Stop Drilling - Assess'
        })
    elif current_risks['kick'] > 40:
        recommendations.append({
            'type': 'WARNING',
            'category': 'Well Control',
            'message': 'Monitor for kick indicators. Check flow rates and pit volumes.',
            'action': 'Prepare BOP'
        })

    if current_risks['lost_circulation'] > 60:
        recommendations.append({
            'type': 'CRITICAL',
            'category': 'Lost Circulation',
            'message': 'HIGH LOST CIRCULATION RISK: Reduce ECD. Consider LCM or reducing flow rate.',
            'action': 'Prepare LCM Pills'
        })

    if current_risks['equipment'] > 70:
        recommendations.append({
            'type': 'CRITICAL',
            'category': 'Equipment',
            'message': 'Equipment failure risk high. Excessive vibration or torque detected.',
            'action': 'Equipment Inspection'
        })

    if not recommendations:
        recommendations.append({
            'type': 'NORMAL',
            'category': 'Operations',
            'message': 'All parameters within normal operating ranges.',
            'action': 'Continue Operations'
        })

    return recommendations


# ============================================================================
# GENERATE DATA
# ============================================================================

wells_drilling_data = {}
wells_info = {}
for well_id in ['WELL-A1', 'WELL-B2', 'WELL-C3', 'WELL-D4', 'WELL-E5']:
    wells_drilling_data[well_id] = generate_drilling_data(well_id)
    wells_info[well_id] = generate_well_info(well_id)

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
                html.H1("⚠️ Drilling Risk Prediction & Monitoring",
                        style={'color': colors['text'], 'fontWeight': 'bold'}),
                html.P("AI-Powered Real-Time Risk Assessment & Anomaly Detection",
                       style={'color': colors['text-secondary'], 'fontSize': '18px'})
            ], style={'textAlign': 'center', 'padding': '30px 0'})
        ])
    ]),

    # Control Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Well Selection & Monitoring Controls", style={'color': colors['primary']}),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Well", style={'color': colors['text']}),
                            dcc.Dropdown(
                                id='well-selector',
                                options=[{'label': f"{well} - {wells_info[well]['field']}", 'value': well}
                                         for well in wells_drilling_data.keys()],
                                value='WELL-A1',
                                style={'color': '#000'}
                            )
                        ], md=3),
                        dbc.Col([
                            html.Label("Time Window (hours)", style={'color': colors['text']}),
                            dcc.Slider(
                                id='time-window',
                                min=24,
                                max=500,
                                value=200,
                                marks={24: '24h', 100: '100h', 200: '200h', 500: '500h'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], md=4),
                        dbc.Col([
                            html.Label("Risk Threshold", style={'color': colors['text']}),
                            dcc.Slider(
                                id='risk-threshold',
                                min=0,
                                max=100,
                                value=60,
                                marks={0: 'Low', 50: 'Medium', 100: 'High'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], md=3),
                        dbc.Col([
                            dbc.Button("Update Analysis", id='update-analysis', color='primary',
                                       className='w-100', style={'marginTop': '25px'}, n_clicks=0),
                        ], md=2)
                    ])
                ])
            ], style={'backgroundColor': colors['surface'], 'marginBottom': '20px'})
        ])
    ]),

    # Risk Gauges Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Stuck Pipe Risk", style={'color': colors['text-secondary'], 'textAlign': 'center'}),
                    dcc.Graph(id='stuck-pipe-gauge', style={'height': '200px'}, config={'displayModeBar': False})
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Kick/Well Control Risk", style={'color': colors['text-secondary'], 'textAlign': 'center'}),
                    dcc.Graph(id='kick-risk-gauge', style={'height': '200px'}, config={'displayModeBar': False})
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Lost Circulation Risk", style={'color': colors['text-secondary'], 'textAlign': 'center'}),
                    dcc.Graph(id='lost-circ-gauge', style={'height': '200px'}, config={'displayModeBar': False})
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Equipment Risk", style={'color': colors['text-secondary'], 'textAlign': 'center'}),
                    dcc.Graph(id='equipment-gauge', style={'height': '200px'}, config={'displayModeBar': False})
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=3)
    ], className='mb-4'),

    # Key Metrics Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(id='current-depth-metric', children='0 ft', style={'color': colors['primary']}),
                    html.P('Current Depth', style={'color': colors['text-secondary']})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(id='rop-metric', children='0 ft/hr', style={'color': colors['success']}),
                    html.P('Avg ROP', style={'color': colors['text-secondary']})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(id='npt-metric', children='0 hrs', style={'color': colors['warning']}),
                    html.P('NPT Predicted', style={'color': colors['text-secondary']})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(id='incidents-metric', children='0', style={'color': colors['danger']}),
                    html.P('Risk Events', style={'color': colors['text-secondary']})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(id='overall-risk-metric', children='0%', style={'color': colors['secondary']}),
                    html.P('Overall Risk', style={'color': colors['text-secondary']})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(id='confidence-metric', children='0%', style={'color': colors['primary']}),
                    html.P('AI Confidence', style={'color': colors['text-secondary']})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2)
    ], className='mb-4'),

    # Main Drilling Parameters Visualization
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Real-Time Drilling Parameters", style={'color': colors['primary']}),
                    dcc.Loading(
                        id="loading-drilling-params",
                        children=[dcc.Graph(id='drilling-params-plot', style={'height': '500px'})],
                        type="default"
                    )
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=12)
    ], className='mb-4'),

    # Risk Trends and Pressure Analysis
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Risk Trends Over Time", style={'color': colors['primary']}),
                    dcc.Graph(id='risk-trends-plot', style={'height': '400px'})
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=8),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Pressure Analysis", style={'color': colors['primary']}),
                    dcc.Graph(id='pressure-plot', style={'height': '400px'})
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=4)
    ], className='mb-4'),

    # AI Recommendations and Events
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("AI-Powered Recommendations", style={'color': colors['primary']}),
                    html.Div(id='recommendations-panel')
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Risk Events Log", style={'color': colors['primary']}),
                    html.Div(id='events-log')
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=6)
    ], className='mb-4'),

    # Anomaly Detection
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Anomaly Detection Dashboard", style={'color': colors['primary']}),
                    dcc.Graph(id='anomaly-plot', style={'height': '350px'})
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=12)
    ]),

    # Store component
    dcc.Store(id='processed-drilling-data')

], fluid=True, style={'backgroundColor': colors['background'], 'minHeight': '100vh', 'padding': '20px'})


# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    [Output('processed-drilling-data', 'data'),
     Output('current-depth-metric', 'children'),
     Output('rop-metric', 'children'),
     Output('npt-metric', 'children'),
     Output('incidents-metric', 'children'),
     Output('overall-risk-metric', 'children'),
     Output('confidence-metric', 'children')],
    [Input('update-analysis', 'n_clicks')],
    [State('well-selector', 'value'),
     State('time-window', 'value')]
)
def process_drilling_data(n_clicks, well_id, time_window):
    if n_clicks == 0:
        return None, '0 ft', '0 ft/hr', '0 hrs', '0', '0%', '0%'

    df = wells_drilling_data[well_id].copy()

    # Filter by time window
    df = df.tail(time_window)

    # Calculate risk scores
    df['STUCK_PIPE_RISK'] = calculate_stuck_pipe_risk(df)
    df['KICK_RISK'] = calculate_kick_risk(df)
    df['LOST_CIRC_RISK'] = calculate_lost_circulation_risk(df)
    df['EQUIPMENT_RISK'] = calculate_equipment_risk(df)
    df['OVERALL_RISK'] = calculate_overall_risk(df)

    # Calculate metrics
    current_depth = df['DEPTH'].iloc[-1]
    avg_rop = df['ROP'].mean()

    # NPT prediction based on risk levels
    high_risk_hours = len(df[df['OVERALL_RISK'] > 60])
    npt_predicted = high_risk_hours * 0.5  # Simplified NPT calculation

    # Count risk events
    total_events = (df['STUCK_PIPE_EVENT'].sum() +
                    df['KICK_EVENT'].sum() +
                    df['LOST_CIRC_EVENT'].sum())

    overall_risk = df['OVERALL_RISK'].iloc[-1]

    # AI confidence (simulated based on data quality and consistency)
    confidence = 85 + np.random.normal(0, 5)
    confidence = np.clip(confidence, 0, 100)

    processed_data = {
        'data': df.to_dict('records'),
        'well_id': well_id,
        'well_info': wells_info[well_id]
    }

    return (processed_data,
            f'{current_depth:.0f} ft',
            f'{avg_rop:.1f} ft/hr',
            f'{npt_predicted:.1f} hrs',
            str(int(total_events)),
            f'{overall_risk:.0f}%',
            f'{confidence:.0f}%')


@app.callback(
    [Output('stuck-pipe-gauge', 'figure'),
     Output('kick-risk-gauge', 'figure'),
     Output('lost-circ-gauge', 'figure'),
     Output('equipment-gauge', 'figure')],
    [Input('processed-drilling-data', 'data')]
)
def update_risk_gauges(processed_data):
    if not processed_data:
        empty_fig = go.Figure()
        return empty_fig, empty_fig, empty_fig, empty_fig

    df = pd.DataFrame(processed_data['data'])

    # Get current risk values (last point)
    stuck_risk = df['STUCK_PIPE_RISK'].iloc[-1]
    kick_risk = df['KICK_RISK'].iloc[-1]
    lost_circ_risk = df['LOST_CIRC_RISK'].iloc[-1]
    equipment_risk = df['EQUIPMENT_RISK'].iloc[-1]

    def create_gauge(value, title):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': title, 'font': {'color': colors['text'], 'size': 14}},
            number={'font': {'size': 40, 'color': colors['text']}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': colors['text']},
                'bar': {'color': colors['primary']},
                'bgcolor': colors['surface'],
                'borderwidth': 2,
                'bordercolor': colors['text-secondary'],
                'steps': [
                    {'range': [0, 40], 'color': colors['success'] + '40'},
                    {'range': [40, 70], 'color': colors['warning'] + '40'},
                    {'range': [70, 100], 'color': colors['danger'] + '40'}
                ],
                'threshold': {
                    'line': {'color': colors['danger'], 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))

        fig.update_layout(
            paper_bgcolor=colors['surface'],
            plot_bgcolor=colors['surface'],
            font={'color': colors['text']},
            margin=dict(l=20, r=20, t=40, b=20),
            height=200
        )

        return fig

    return (create_gauge(stuck_risk, ""),
            create_gauge(kick_risk, ""),
            create_gauge(lost_circ_risk, ""),
            create_gauge(equipment_risk, ""))


@app.callback(
    Output('drilling-params-plot', 'figure'),
    [Input('processed-drilling-data', 'data')]
)
def update_drilling_params(processed_data):
    if not processed_data:
        return go.Figure()

    df = pd.DataFrame(processed_data['data'])

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=('Rate of Penetration', 'Weight on Bit',
                        'Rotary Speed', 'Torque',
                        'Standpipe Pressure', 'Flow Rate',
                        'Mud Weight', 'Hook Load'),
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )

    # ROP
    fig.add_trace(
        go.Scatter(x=df['DEPTH'], y=df['ROP'], mode='lines', name='ROP',
                   line=dict(color=colors['success'], width=2)),
        row=1, col=1
    )

    # WOB
    fig.add_trace(
        go.Scatter(x=df['DEPTH'], y=df['WOB'], mode='lines', name='WOB',
                   line=dict(color=colors['primary'], width=2)),
        row=1, col=2
    )

    # RPM
    fig.add_trace(
        go.Scatter(x=df['DEPTH'], y=df['RPM'], mode='lines', name='RPM',
                   line=dict(color=colors['warning'], width=2)),
        row=2, col=1
    )

    # Torque
    fig.add_trace(
        go.Scatter(x=df['DEPTH'], y=df['TORQUE'], mode='lines', name='Torque',
                   line=dict(color=colors['secondary'], width=2)),
        row=2, col=2
    )

    # SPP
    fig.add_trace(
        go.Scatter(x=df['DEPTH'], y=df['SPP'], mode='lines', name='SPP',
                   line=dict(color=colors['danger'], width=2)),
        row=3, col=1
    )

    # Flow Rate
    fig.add_trace(
        go.Scatter(x=df['DEPTH'], y=df['FLOW_RATE'], mode='lines', name='Flow',
                   line=dict(color='#00d4ff', width=2)),
        row=3, col=2
    )

    # Mud Weight
    fig.add_trace(
        go.Scatter(x=df['DEPTH'], y=df['MUD_WEIGHT'], mode='lines', name='MW',
                   line=dict(color='#9c27b0', width=2)),
        row=4, col=1
    )

    # Hook Load
    fig.add_trace(
        go.Scatter(x=df['DEPTH'], y=df['HOOK_LOAD'], mode='lines', name='Hook Load',
                   line=dict(color='#ff6b35', width=2)),
        row=4, col=2
    )

    # Update axes labels
    fig.update_xaxes(title_text="Depth (ft)", row=4, col=1)
    fig.update_xaxes(title_text="Depth (ft)", row=4, col=2)

    fig.update_yaxes(title_text="ft/hr", row=1, col=1)
    fig.update_yaxes(title_text="klbs", row=1, col=2)
    fig.update_yaxes(title_text="RPM", row=2, col=1)
    fig.update_yaxes(title_text="ft-lbs", row=2, col=2)
    fig.update_yaxes(title_text="psi", row=3, col=1)
    fig.update_yaxes(title_text="gpm", row=3, col=2)
    fig.update_yaxes(title_text="ppg", row=4, col=1)
    fig.update_yaxes(title_text="klbs", row=4, col=2)

    fig.update_layout(
        height=500,
        showlegend=False,
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['background'],
        font=dict(color=colors['text'], size=10),
        title=dict(
            text=f"Drilling Parameters vs Depth - {processed_data['well_id']}",
            font=dict(size=18, color=colors['primary'])
        )
    )

    return fig


@app.callback(
    Output('risk-trends-plot', 'figure'),
    [Input('processed-drilling-data', 'data')]
)
def update_risk_trends(processed_data):
    if not processed_data:
        return go.Figure()

    df = pd.DataFrame(processed_data['data'])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['DEPTH'], y=df['STUCK_PIPE_RISK'],
        mode='lines', name='Stuck Pipe',
        line=dict(color='#ff6b35', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df['DEPTH'], y=df['KICK_RISK'],
        mode='lines', name='Kick Risk',
        line=dict(color='#ff3366', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df['DEPTH'], y=df['LOST_CIRC_RISK'],
        mode='lines', name='Lost Circulation',
        line=dict(color='#ffaa00', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df['DEPTH'], y=df['EQUIPMENT_RISK'],
        mode='lines', name='Equipment',
        line=dict(color='#9c27b0', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df['DEPTH'], y=df['OVERALL_RISK'],
        mode='lines', name='Overall Risk',
        line=dict(color=colors['primary'], width=3)
    ))

    # Add threshold line
    fig.add_hline(y=60, line_dash="dash", line_color=colors['danger'],
                  annotation_text="High Risk Threshold")

    fig.update_layout(
        xaxis_title="Depth (ft)",
        yaxis_title="Risk Score (0-100)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['surface'],
        font=dict(color=colors['text']),
        hovermode='x unified'
    )

    return fig


@app.callback(
    Output('pressure-plot', 'figure'),
    [Input('processed-drilling-data', 'data')]
)
def update_pressure_plot(processed_data):
    if not processed_data:
        return go.Figure()

    df = pd.DataFrame(processed_data['data'])

    fig = go.Figure()

    # Formation Pressure
    fig.add_trace(go.Scatter(
        y=df['DEPTH'], x=df['FORMATION_PRESSURE'],
        mode='lines', name='Formation Pressure',
        line=dict(color=colors['danger'], width=2)
    ))

    # Hydrostatic Pressure
    hydrostatic = df['MUD_WEIGHT'] * 0.052 * df['DEPTH']
    fig.add_trace(go.Scatter(
        y=df['DEPTH'], x=hydrostatic,
        mode='lines', name='Hydrostatic',
        line=dict(color=colors['success'], width=2)
    ))

    # ECD Pressure
    ecd_pressure = df['ECD'] * 0.052 * df['DEPTH']
    fig.add_trace(go.Scatter(
        y=df['DEPTH'], x=ecd_pressure,
        mode='lines', name='ECD',
        line=dict(color=colors['warning'], width=2, dash='dash')
    ))

    # Highlight overpressure zones
    overpressure_mask = df['FORMATION_PRESSURE'] > (df['DEPTH'] * 0.5)
    if overpressure_mask.any():
        fig.add_trace(go.Scatter(
            y=df[overpressure_mask]['DEPTH'],
            x=df[overpressure_mask]['FORMATION_PRESSURE'],
            mode='markers',
            name='Overpressure Zone',
            marker=dict(color=colors['danger'], size=8, symbol='diamond')
        ))

    fig.update_layout(
        xaxis_title="Pressure (psi)",
        yaxis_title="Depth (ft)",
        yaxis_autorange='reversed',
        showlegend=True,
        legend=dict(x=0.7, y=0.98),
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['surface'],
        font=dict(color=colors['text'])
    )

    return fig


@app.callback(
    Output('recommendations-panel', 'children'),
    [Input('processed-drilling-data', 'data'),
     Input('risk-threshold', 'value')]
)
def update_recommendations(processed_data, threshold):
    if not processed_data:
        return html.Div()

    df = pd.DataFrame(processed_data['data'])

    # Get current risks
    current_risks = {
        'stuck_pipe': df['STUCK_PIPE_RISK'].iloc[-1],
        'kick': df['KICK_RISK'].iloc[-1],
        'lost_circulation': df['LOST_CIRC_RISK'].iloc[-1],
        'equipment': df['EQUIPMENT_RISK'].iloc[-1]
    }

    recommendations = generate_recommendations(current_risks)

    # Create alert cards
    alert_cards = []
    for rec in recommendations:
        if rec['type'] == 'CRITICAL':
            color = 'danger'
            icon = '🚨'
        elif rec['type'] == 'WARNING':
            color = 'warning'
            icon = '⚠️'
        else:
            color = 'success'
            icon = '✓'

        alert_cards.append(
            dbc.Alert([
                html.H6([icon, f"  {rec['category']}"], className="alert-heading"),
                html.P(rec['message'], className="mb-2"),
                html.Hr(),
                html.P(f"Recommended Action: {rec['action']}",
                       className="mb-0", style={'fontWeight': 'bold'})
            ], color=color, className="mb-3")
        )

    return html.Div(alert_cards)


@app.callback(
    Output('events-log', 'children'),
    [Input('processed-drilling-data', 'data')]
)
def update_events_log(processed_data):
    if not processed_data:
        return html.Div()

    df = pd.DataFrame(processed_data['data'])

    # Find events
    events = []

    stuck_events = df[df['STUCK_PIPE_EVENT'] == 1]
    for _, row in stuck_events.iterrows():
        events.append({
            'Timestamp': row['TIMESTAMP'].strftime('%Y-%m-%d %H:%M'),
            'Depth (ft)': f"{row['DEPTH']:.0f}",
            'Event Type': 'Stuck Pipe',
            'Severity': 'High'
        })

    kick_events = df[df['KICK_EVENT'] == 1]
    for _, row in kick_events.iterrows():
        events.append({
            'Timestamp': row['TIMESTAMP'].strftime('%Y-%m-%d %H:%M'),
            'Depth (ft)': f"{row['DEPTH']:.0f}",
            'Event Type': 'Kick Warning',
            'Severity': 'Critical'
        })

    lost_circ_events = df[df['LOST_CIRC_EVENT'] == 1]
    for _, row in lost_circ_events.iterrows():
        events.append({
            'Timestamp': row['TIMESTAMP'].strftime('%Y-%m-%d %H:%M'),
            'Depth (ft)': f"{row['DEPTH']:.0f}",
            'Event Type': 'Lost Circulation',
            'Severity': 'High'
        })

    if events:
        # Sort by timestamp (most recent first)
        events_df = pd.DataFrame(events).sort_values('Timestamp', ascending=False).head(10)

        return dash_table.DataTable(
            data=events_df.to_dict('records'),
            columns=[{'name': col, 'id': col} for col in events_df.columns],
            style_cell={
                'textAlign': 'left',
                'backgroundColor': colors['surface'],
                'color': colors['text'],
                'border': f'1px solid {colors["primary"]}40',
                'fontSize': '12px'
            },
            style_header={
                'backgroundColor': colors['primary'],
                'color': colors['background'],
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Severity} = "Critical"'},
                    'backgroundColor': colors['danger'] + '20',
                    'color': colors['danger']
                },
                {
                    'if': {'filter_query': '{Severity} = "High"'},
                    'backgroundColor': colors['warning'] + '20',
                    'color': colors['warning']
                }
            ]
        )

    return html.P("No risk events detected", style={'color': colors['text-secondary']})


@app.callback(
    Output('anomaly-plot', 'figure'),
    [Input('processed-drilling-data', 'data')]
)
def update_anomaly_plot(processed_data):
    if not processed_data:
        return go.Figure()

    df = pd.DataFrame(processed_data['data'])

    # Create anomaly detection using statistical methods
    # Calculate rolling statistics
    window = 50
    df['ROP_MEAN'] = df['ROP'].rolling(window, min_periods=1).mean()
    df['ROP_STD'] = df['ROP'].rolling(window, min_periods=1).std()
    df['ROP_ANOMALY'] = np.abs(df['ROP'] - df['ROP_MEAN']) > (2 * df['ROP_STD'])

    df['TORQUE_MEAN'] = df['TORQUE'].rolling(window, min_periods=1).mean()
    df['TORQUE_STD'] = df['TORQUE'].rolling(window, min_periods=1).std()
    df['TORQUE_ANOMALY'] = np.abs(df['TORQUE'] - df['TORQUE_MEAN']) > (2 * df['TORQUE_STD'])

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('ROP Anomaly Detection', 'Torque Anomaly Detection'),
        vertical_spacing=0.15
    )

    # ROP plot
    fig.add_trace(
        go.Scatter(x=df['DEPTH'], y=df['ROP'], mode='lines', name='ROP',
                   line=dict(color=colors['primary'], width=1)),
        row=1, col=1
    )

    # Highlight anomalies
    anomalies = df[df['ROP_ANOMALY']]
    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(x=anomalies['DEPTH'], y=anomalies['ROP'], mode='markers',
                       name='ROP Anomaly',
                       marker=dict(color=colors['danger'], size=10, symbol='x')),
            row=1, col=1
        )

    # Torque plot
    fig.add_trace(
        go.Scatter(x=df['DEPTH'], y=df['TORQUE'], mode='lines', name='Torque',
                   line=dict(color=colors['warning'], width=1)),
        row=2, col=1
    )

    # Highlight anomalies
    torque_anomalies = df[df['TORQUE_ANOMALY']]
    if not torque_anomalies.empty:
        fig.add_trace(
            go.Scatter(x=torque_anomalies['DEPTH'], y=torque_anomalies['TORQUE'],
                       mode='markers', name='Torque Anomaly',
                       marker=dict(color=colors['danger'], size=10, symbol='x')),
            row=2, col=1
        )

    fig.update_xaxes(title_text="Depth (ft)", row=2, col=1)
    fig.update_yaxes(title_text="ft/hr", row=1, col=1)
    fig.update_yaxes(title_text="ft-lbs", row=2, col=1)

    fig.update_layout(
        height=350,
        showlegend=True,
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['surface'],
        font=dict(color=colors['text'])
    )

    return fig


# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == '__main__':
    app.run(debug=False, port=8051, host='127.0.0.1')