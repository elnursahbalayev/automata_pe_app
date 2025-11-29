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
import random

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
# AI CHATBOT RESPONSES FOR DRILLING
# ============================================================================

AI_RESPONSES = {
    'default': [
        "Based on my analysis of the current drilling parameters, I'm monitoring several key indicators. The torque and WOB patterns are within normal ranges, but I recommend maintaining vigilance on the SPP trends.",
        "I've analyzed the real-time data from this rig. Current ROP is optimal for the formation being drilled. The mud properties appear stable, but consider monitoring gas readings more closely.",
        "Looking at the drilling dynamics, I can see the bit is performing well. However, I've detected some minor fluctuations in torque that could indicate formation changes ahead.",
        "The current drilling window appears safe. Mud weight is balanced well against pore pressure estimates. Continue monitoring for any signs of wellbore instability.",
    ],
    'stuck_pipe': [
        "Stuck pipe prevention recommendations:\n• Maintain continuous pipe rotation (>40 RPM)\n• Avoid prolonged static periods\n• Keep hole clean with adequate flow rate\n• Consider short trips every 500ft\n• Monitor torque trends for early warning signs",
        "Based on current torque patterns, stuck pipe risk is moderate. Key actions:\n1. Increase circulation rate by 10%\n2. Reduce WOB to 20 klbs\n3. Consider pumping a pill before next connection\n4. Ensure proper mud conditioning",
    ],
    'lost_circulation': [
        "Lost circulation management strategy:\n• Monitor pit levels continuously\n• Have LCM ready (fiber, flake, granular mix)\n• Consider reducing mud weight by 0.3-0.5 ppg\n• Identify loss zone depth from drilling breaks\n• Be prepared to set a plug if losses are severe",
        "Current indicators suggest potential loss zone ahead. Recommendations:\n1. Pre-treat mud with 15-20 ppb fine LCM\n2. Reduce ECD by lowering flow rate\n3. Avoid surge pressures during trips\n4. Monitor SPP carefully for sudden drops",
    ],
    'kick': [
        "Kick detection and prevention:\n• Monitor flow rate in vs out continuously\n• Watch for drilling breaks and increased ROP\n• Track pit volume gains (>5 bbl is significant)\n• Be prepared to shut in immediately\n• Maintain proper mud weight for formation pressure",
        "Gas readings are elevated. Immediate actions:\n1. Increase mud weight by 0.2 ppg\n2. Flow check on next connection\n3. Alert well control crew\n4. Verify BOP equipment readiness\n5. Monitor trip tank closely",
    ],
    'optimization': [
        "Drilling optimization recommendations:\n• Current WOB/RPM ratio can be improved\n• Suggest increasing RPM to 130-140 for better ROP\n• Bit appears slightly under-weighted, consider +3 klbs\n• Flow rate is optimal for hole cleaning\n• Monitor MSE for bit wear indicators",
        "To improve drilling efficiency:\n1. Optimize WOB: Currently at 25 klbs, try 28-30 klbs\n2. RPM can be increased to 135 safely\n3. Consider shorter bit runs with current formation\n4. Mud rheology is good, maintain current properties",
    ],
    'help': [
        "I can help you with:\n• Stuck pipe risk analysis & prevention\n• Lost circulation detection & management\n• Kick warning signs & well control\n• Drilling parameter optimization\n• Real-time risk assessment\n• Equipment recommendations\n\nJust ask me anything about your drilling operations!",
    ]
}


def get_ai_response(user_message):
    """Generate contextual AI response based on user message"""
    message_lower = user_message.lower()
    
    if any(word in message_lower for word in ['stuck', 'pipe', 'differential', 'pack off']):
        return random.choice(AI_RESPONSES['stuck_pipe'])
    elif any(word in message_lower for word in ['lost', 'circulation', 'loss', 'lcm', 'losses']):
        return random.choice(AI_RESPONSES['lost_circulation'])
    elif any(word in message_lower for word in ['kick', 'gas', 'influx', 'well control', 'blowout']):
        return random.choice(AI_RESPONSES['kick'])
    elif any(word in message_lower for word in ['optimize', 'rop', 'faster', 'improve', 'efficiency', 'wob', 'rpm']):
        return random.choice(AI_RESPONSES['optimization'])
    elif any(word in message_lower for word in ['help', 'what can you', 'how to', 'guide']):
        return random.choice(AI_RESPONSES['help'])
    else:
        return random.choice(AI_RESPONSES['default'])


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
    'surface-light': '#252b4d',
    'primary': '#00d4ff',
    'secondary': '#ff6b35',
    'success': '#00ff88',
    'warning': '#ffaa00',
    'danger': '#ff3366',
    'text': '#ffffff',
    'text-secondary': '#8892b0',
    'chat-user': '#1e3a5f',
    'chat-ai': '#2d1f3d'
}


# ============================================================================
# CHAT COMPONENT
# ============================================================================

def create_chat_message(text, is_user=False):
    """Create a chat message bubble"""
    return html.Div([
        html.Div([
            html.I(className="fas fa-user" if is_user else "fas fa-robot", 
                   style={'marginRight': '8px'}),
            html.Span("You" if is_user else "AUTOMATA AI", 
                     style={'fontWeight': 'bold', 'fontSize': '12px'})
        ], style={'marginBottom': '5px', 'color': colors['primary'] if not is_user else colors['secondary']}),
        html.P(text, style={'margin': '0', 'lineHeight': '1.5', 'whiteSpace': 'pre-line'})
    ], style={
        'backgroundColor': colors['chat-user'] if is_user else colors['chat-ai'],
        'padding': '12px 15px',
        'borderRadius': '15px',
        'marginBottom': '10px',
        'marginLeft': '40px' if is_user else '0',
        'marginRight': '0' if is_user else '40px',
        'color': colors['text']
    })


# Initial chat messages
initial_chat = [
    {'text': "Hello! I'm AUTOMATA AI, your drilling risk prediction assistant. I'm monitoring real-time drilling parameters and can help you with stuck pipe prevention, lost circulation management, kick detection, and drilling optimization. What would you like to know?", 'is_user': False}
]


# ============================================================================
# DASH LAYOUT
# ============================================================================

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("AUTOMATA INTELLIGENCE",
                        style={'color': colors['text'], 'fontWeight': 'bold', 'marginBottom': '0'}),
                html.H2("⚙️ Drilling Risk Prediction Suite",
                        style={'color': colors['primary'], 'fontWeight': 'bold', 'marginTop': '5px'}),
                html.P("Real-Time Risk Analytics & Predictive Drilling Optimization",
                       style={'color': colors['text-secondary'], 'fontSize': '18px'})
            ], style={'textAlign': 'center', 'padding': '20px 0'})
        ])
    ]),

    # Upload Section - NEW
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.I(className="fas fa-cloud-upload-alt fa-2x", 
                                       style={'color': colors['primary'], 'marginBottom': '10px'}),
                                html.H5("Upload Drilling Data", style={'color': colors['text'], 'marginBottom': '5px'}),
                                html.P("Import drilling reports, WITSML data, or survey files", 
                                       style={'color': colors['text-secondary'], 'fontSize': '13px', 'margin': '0'})
                            ], style={'textAlign': 'center'})
                        ], md=4),
                        dbc.Col([
                            dcc.Upload(
                                id='upload-drilling-files',
                                children=html.Div([
                                    html.I(className="fas fa-file-upload", style={'fontSize': '24px', 'marginRight': '10px'}),
                                    'Drop files here or ',
                                    html.A('Browse', style={'color': colors['primary'], 'textDecoration': 'underline', 'cursor': 'pointer'})
                                ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
                                style={
                                    'width': '100%',
                                    'height': '80px',
                                    'lineHeight': '60px',
                                    'borderWidth': '2px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '10px',
                                    'borderColor': colors['primary'],
                                    'textAlign': 'center',
                                    'backgroundColor': colors['surface-light'],
                                    'color': colors['text'],
                                    'cursor': 'pointer',
                                    'transition': 'all 0.3s ease'
                                },
                                multiple=True,
                                accept='.csv,.xlsx,.las,.xml,.witsml,.pdf'
                            )
                        ], md=5),
                        dbc.Col([
                            html.Div(id='upload-status', children=[
                                html.Div([
                                    html.I(className="fas fa-info-circle", style={'marginRight': '8px', 'color': colors['primary']}),
                                    html.Span("Supported formats:", style={'fontWeight': 'bold'})
                                ]),
                                html.Ul([
                                    html.Li("Daily Drilling Reports (DDR)", style={'fontSize': '12px'}),
                                    html.Li("WITSML / Real-time Data", style={'fontSize': '12px'}),
                                    html.Li("Survey & Trajectory Files", style={'fontSize': '12px'}),
                                    html.Li("Mud Reports / BHA Data", style={'fontSize': '12px'})
                                ], style={'marginBottom': '0', 'paddingLeft': '20px'})
                            ], style={'color': colors['text-secondary'], 'fontSize': '13px'})
                        ], md=3)
                    ], align='center')
                ])
            ], style={'backgroundColor': colors['surface'], 'marginBottom': '15px', 'border': f'1px solid {colors["surface-light"]}'})
        ])
    ]),
    
    # Uploaded Files Display
    dbc.Row([
        dbc.Col([
            html.Div(id='uploaded-files-list', children=[], style={'marginBottom': '15px'})
        ])
    ]),

    # Control Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5([
                        html.I(className="fas fa-sliders-h", style={'marginRight': '10px'}),
                        "Rig Selection & Monitoring Controls"
                    ], style={'color': colors['primary'], 'marginBottom': '15px'}),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Rig", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '13px'}),
                            dcc.Dropdown(
                                id='rig-selector',
                                options=[{'label': f'{rig} - {rigs_metadata[rig]["well_name"]}',
                                          'value': rig} for rig in rigs_data.keys()],
                                value='RIG-001',
                                style={'color': '#000'}
                            )
                        ], md=3),
                        dbc.Col([
                            html.Label("Time Window", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '13px'}),
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
                            html.Label("Risk Threshold", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '13px'}),
                            dcc.Slider(
                                id='risk-threshold',
                                min=0,
                                max=100,
                                value=60,
                                marks={0: {'label': '0%', 'style': {'color': colors['text-secondary']}}, 
                                       50: {'label': '50%', 'style': {'color': colors['text-secondary']}}, 
                                       100: {'label': '100%', 'style': {'color': colors['text-secondary']}}},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], md=3),
                        dbc.Col([
                            html.Label("Update Frequency", style={'color': colors['text'], 'fontWeight': 'bold', 'fontSize': '13px'}),
                            dcc.RadioItems(
                                id='update-freq',
                                options=[
                                    {'label': ' Live', 'value': 'live'},
                                    {'label': ' Manual', 'value': 'manual'}
                                ],
                                value='manual',
                                inline=True,
                                style={'color': colors['text'], 'marginTop': '8px'},
                                inputStyle={'marginRight': '5px'},
                                labelStyle={'marginRight': '15px'}
                            )
                        ], md=2),
                        dbc.Col([
                            dbc.Button([
                                html.I(className="fas fa-search", style={'marginRight': '8px'}),
                                "Analyze Risks"
                            ], id='analyze-btn', color='primary',
                               className='w-100', style={'marginTop': '20px', 'fontWeight': 'bold'}, n_clicks=0)
                        ], md=2)
                    ])
                ])
            ], style={'backgroundColor': colors['surface'], 'marginBottom': '20px', 'border': f'1px solid {colors["surface-light"]}'})
        ])
    ]),

    # Rig Information Card - NEW
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.I(className="fas fa-oil-well fa-2x", style={'color': colors['primary'], 'marginRight': '15px'}),
                            ], style={'display': 'inline-block', 'verticalAlign': 'middle'}),
                            html.Div([
                                html.Span(id='rig-info-name', children='RIG-001', style={'color': colors['text'], 'fontSize': '20px', 'fontWeight': 'bold'}),
                                html.Span(' | ', style={'color': colors['text-secondary'], 'margin': '0 10px'}),
                                html.Span(id='rig-info-well', children='WELL-XXX', style={'color': colors['primary'], 'fontSize': '16px'})
                            ], style={'display': 'inline-block', 'verticalAlign': 'middle'})
                        ], md=3),
                        dbc.Col([
                            html.Div([
                                html.I(className="fas fa-map-marker-alt", style={'color': colors['secondary'], 'marginRight': '8px'}),
                                html.Span(id='rig-info-field', children='Field Name', style={'color': colors['text']})
                            ])
                        ], md=2),
                        dbc.Col([
                            html.Div([
                                html.I(className="fas fa-cog", style={'color': colors['warning'], 'marginRight': '8px'}),
                                html.Span(id='rig-info-operation', children='Operation', style={'color': colors['text']})
                            ])
                        ], md=2),
                        dbc.Col([
                            html.Div([
                                html.I(className="fas fa-calendar-alt", style={'color': colors['success'], 'marginRight': '8px'}),
                                html.Span(id='rig-info-days', children='0 Days', style={'color': colors['text']})
                            ])
                        ], md=2),
                        dbc.Col([
                            html.Div([
                                html.I(className="fas fa-shield-alt", style={'color': colors['success'], 'marginRight': '8px'}),
                                html.Span(id='rig-info-incident', children='0 Days Since Incident', style={'color': colors['text']})
                            ])
                        ], md=3)
                    ], align='center')
                ], style={'padding': '15px 20px'})
            ], style={'backgroundColor': colors['surface'], 'marginBottom': '15px', 'border': f'1px solid {colors["surface-light"]}'})
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
                    html.Div([
                        html.I(className="fas fa-exclamation-triangle fa-2x", style={'color': colors['danger'], 'marginBottom': '10px'})
                    ]),
                    html.H4(id='overall-risk', children='0%', style={'color': colors['danger'], 'marginBottom': '5px'}),
                    html.P('Overall Risk', style={'color': colors['text-secondary'], 'margin': '0', 'fontWeight': 'bold'}),
                    html.Div(id='risk-trend', style={'fontSize': '12px', 'marginTop': '5px'})
                ], style={'textAlign': 'center', 'padding': '15px'})
            ], style={'backgroundColor': colors['surface'], 'border': f'1px solid {colors["surface-light"]}'})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-link fa-2x", style={'color': colors['warning'], 'marginBottom': '10px'})
                    ]),
                    html.H4(id='stuck-pipe-risk', children='0%', style={'color': colors['warning'], 'marginBottom': '5px'}),
                    html.P('Stuck Pipe', style={'color': colors['text-secondary'], 'margin': '0', 'fontWeight': 'bold'}),
                    html.Small('Next 2 hrs', style={'color': colors['text-secondary']})
                ], style={'textAlign': 'center', 'padding': '15px'})
            ], style={'backgroundColor': colors['surface'], 'border': f'1px solid {colors["surface-light"]}'})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-tint-slash fa-2x", style={'color': colors['primary'], 'marginBottom': '10px'})
                    ]),
                    html.H4(id='lost-circ-risk', children='0%', style={'color': colors['primary'], 'marginBottom': '5px'}),
                    html.P('Lost Circulation', style={'color': colors['text-secondary'], 'margin': '0', 'fontWeight': 'bold'}),
                    html.Small('Next 2 hrs', style={'color': colors['text-secondary']})
                ], style={'textAlign': 'center', 'padding': '15px'})
            ], style={'backgroundColor': colors['surface'], 'border': f'1px solid {colors["surface-light"]}'})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-arrow-up fa-2x", style={'color': colors['secondary'], 'marginBottom': '10px'})
                    ]),
                    html.H4(id='kick-risk', children='0%', style={'color': colors['secondary'], 'marginBottom': '5px'}),
                    html.P('Kick Risk', style={'color': colors['text-secondary'], 'margin': '0', 'fontWeight': 'bold'}),
                    html.Small('Next 2 hrs', style={'color': colors['text-secondary']})
                ], style={'textAlign': 'center', 'padding': '15px'})
            ], style={'backgroundColor': colors['surface'], 'border': f'1px solid {colors["surface-light"]}'})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-tachometer-alt fa-2x", style={'color': colors['success'], 'marginBottom': '10px'})
                    ]),
                    html.H4(id='current-rop', children='0 ft/hr', style={'color': colors['success'], 'marginBottom': '5px'}),
                    html.P('Current ROP', style={'color': colors['text-secondary'], 'margin': '0', 'fontWeight': 'bold'}),
                    html.Small('Avg: 0 ft/hr', id='avg-rop', style={'color': colors['text-secondary']})
                ], style={'textAlign': 'center', 'padding': '15px'})
            ], style={'backgroundColor': colors['surface'], 'border': f'1px solid {colors["surface-light"]}'})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className="fas fa-arrows-alt-v fa-2x", style={'color': colors['primary'], 'marginBottom': '10px'})
                    ]),
                    html.H4(id='current-depth', children='0 ft', style={'color': colors['primary'], 'marginBottom': '5px'}),
                    html.P('Current Depth', style={'color': colors['text-secondary'], 'margin': '0', 'fontWeight': 'bold'}),
                    html.Small('Target: 0 ft', id='target-depth', style={'color': colors['text-secondary']})
                ], style={'textAlign': 'center', 'padding': '15px'})
            ], style={'backgroundColor': colors['surface'], 'border': f'1px solid {colors["surface-light"]}'})
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
            ], style={'backgroundColor': colors['surface'], 'border': f'1px solid {colors["surface-light"]}'})
        ], md=8),
        dbc.Col([
            # Risk Heat Map
            dbc.Card([
                dbc.CardBody([
                    html.H5([
                        html.I(className="fas fa-th", style={'marginRight': '10px'}),
                        "Risk Heat Map"
                    ], style={'color': colors['primary'], 'marginBottom': '15px'}),
                    dcc.Graph(id='risk-heatmap', style={'height': '220px'})
                ])
            ], style={'backgroundColor': colors['surface'], 'marginBottom': '15px', 'border': f'1px solid {colors["surface-light"]}'}),
            
            # AI Chat Interface - NEW
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H5([
                            html.I(className="fas fa-robot", style={'marginRight': '10px', 'color': colors['success']}),
                            "AUTOMATA AI Assistant"
                        ], style={'color': colors['primary'], 'marginBottom': '0'}),
                        html.Span([
                            html.I(className="fas fa-circle", style={'fontSize': '8px', 'marginRight': '5px'}),
                            "Online"
                        ], style={
                            'color': colors['success'], 
                            'fontSize': '12px',
                            'display': 'flex',
                            'alignItems': 'center'
                        })
                    ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '15px'}),
                    
                    # Chat Messages Container
                    html.Div(
                        id='chat-messages',
                        children=[create_chat_message(msg['text'], msg['is_user']) for msg in initial_chat],
                        style={
                            'height': '250px',
                            'overflowY': 'auto',
                            'padding': '10px',
                            'backgroundColor': colors['background'],
                            'borderRadius': '10px',
                            'marginBottom': '15px'
                        }
                    ),
                    
                    # Chat Input
                    dbc.InputGroup([
                        dbc.Input(
                            id='chat-input',
                            placeholder='Ask about drilling risks...',
                            type='text',
                            style={
                                'backgroundColor': colors['surface-light'],
                                'border': 'none',
                                'color': colors['text']
                            }
                        ),
                        dbc.Button(
                            html.I(className="fas fa-paper-plane"),
                            id='send-chat',
                            color='primary',
                            n_clicks=0
                        )
                    ]),
                    
                    # Quick Actions
                    html.Div([
                        html.P("Quick questions:", style={'color': colors['text-secondary'], 'fontSize': '12px', 'marginTop': '10px', 'marginBottom': '5px'}),
                        html.Div([
                            dbc.Badge("Stuck pipe", color="warning", className="me-1", id='quick-stuck', style={'cursor': 'pointer'}),
                            dbc.Badge("Lost circulation", color="primary", className="me-1", id='quick-lost', style={'cursor': 'pointer'}),
                            dbc.Badge("Optimize ROP", color="success", className="me-1", id='quick-optimize', style={'cursor': 'pointer'}),
                        ])
                    ])
                ])
            ], style={'backgroundColor': colors['surface'], 'border': f'1px solid {colors["surface-light"]}'})
        ], md=4)
    ], className='mb-4'),

    # Detailed Analysis Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5([
                        html.I(className="fas fa-clipboard-list", style={'marginRight': '10px'}),
                        "Risk Mitigation Recommendations"
                    ], style={'color': colors['primary'], 'marginBottom': '15px'}),
                    html.Div(id='recommendations-table', style={'overflowX': 'auto'})
                ])
            ], style={'backgroundColor': colors['surface'], 'border': f'1px solid {colors["surface-light"]}', 'height': '100%'})
        ], md=7),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5([
                        html.I(className="fas fa-history", style={'marginRight': '10px'}),
                        "Historical Events"
                    ], style={'color': colors['primary'], 'marginBottom': '15px'}),
                    dcc.Graph(id='events-timeline', style={'height': '280px'})
                ])
            ], style={'backgroundColor': colors['surface'], 'border': f'1px solid {colors["surface-light"]}', 'height': '100%'})
        ], md=5)
    ], className='mb-4'),

    # Equipment Status Row - NEW
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5([
                        html.I(className="fas fa-tools", style={'marginRight': '10px'}),
                        "Equipment Status"
                    ], style={'color': colors['primary'], 'marginBottom': '15px'}),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.I(className="fas fa-check-circle fa-2x", style={'color': colors['success']}),
                                html.P("Top Drive", style={'color': colors['text'], 'margin': '5px 0 0 0', 'fontSize': '13px'}),
                                html.Small("Operational", style={'color': colors['success']})
                            ], style={'textAlign': 'center'})
                        ], md=2),
                        dbc.Col([
                            html.Div([
                                html.I(className="fas fa-check-circle fa-2x", style={'color': colors['success']}),
                                html.P("Mud Pumps", style={'color': colors['text'], 'margin': '5px 0 0 0', 'fontSize': '13px'}),
                                html.Small("3/3 Active", style={'color': colors['success']})
                            ], style={'textAlign': 'center'})
                        ], md=2),
                        dbc.Col([
                            html.Div([
                                html.I(className="fas fa-check-circle fa-2x", style={'color': colors['success']}),
                                html.P("BOP", style={'color': colors['text'], 'margin': '5px 0 0 0', 'fontSize': '13px'}),
                                html.Small("Tested OK", style={'color': colors['success']})
                            ], style={'textAlign': 'center'})
                        ], md=2),
                        dbc.Col([
                            html.Div([
                                html.I(className="fas fa-exclamation-circle fa-2x", style={'color': colors['warning']}),
                                html.P("Shakers", style={'color': colors['text'], 'margin': '5px 0 0 0', 'fontSize': '13px'}),
                                html.Small("1 Screen Low", style={'color': colors['warning']})
                            ], style={'textAlign': 'center'})
                        ], md=2),
                        dbc.Col([
                            html.Div([
                                html.I(className="fas fa-check-circle fa-2x", style={'color': colors['success']}),
                                html.P("Drawworks", style={'color': colors['text'], 'margin': '5px 0 0 0', 'fontSize': '13px'}),
                                html.Small("Operational", style={'color': colors['success']})
                            ], style={'textAlign': 'center'})
                        ], md=2),
                        dbc.Col([
                            html.Div([
                                html.I(className="fas fa-check-circle fa-2x", style={'color': colors['success']}),
                                html.P("Generators", style={'color': colors['text'], 'margin': '5px 0 0 0', 'fontSize': '13px'}),
                                html.Small("4/4 Online", style={'color': colors['success']})
                            ], style={'textAlign': 'center'})
                        ], md=2)
                    ])
                ])
            ], style={'backgroundColor': colors['surface'], 'border': f'1px solid {colors["surface-light"]}'})
        ])
    ]),

    # Store components and interval for updates
    dcc.Store(id='processed-data-store'),
    dcc.Store(id='chat-history-store', data=initial_chat),
    dcc.Store(id='uploaded-files-store', data=[]),
    dcc.Interval(id='interval-component', interval=5000, n_intervals=0, disabled=True)

], fluid=True, style={'backgroundColor': colors['background'], 'minHeight': '100vh', 'padding': '20px'})


# ============================================================================
# CALLBACKS
# ============================================================================

# File Upload Callback
@app.callback(
    [Output('uploaded-files-list', 'children'),
     Output('uploaded-files-store', 'data')],
    [Input('upload-drilling-files', 'contents')],
    [State('upload-drilling-files', 'filename'),
     State('uploaded-files-store', 'data')]
)
def handle_file_upload(contents, filenames, existing_files):
    if contents is None:
        return [], existing_files or []
    
    if existing_files is None:
        existing_files = []
    
    new_files = []
    file_types = {
        '.csv': 'Drilling Data',
        '.xlsx': 'Report',
        '.las': 'LAS File',
        '.xml': 'WITSML Data',
        '.witsml': 'WITSML Data',
        '.pdf': 'DDR Report'
    }
    
    for filename in filenames:
        ext = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
        file_info = {
            'name': filename,
            'type': file_types.get(ext, 'Data File'),
            'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'status': 'Processed'
        }
        new_files.append(file_info)
    
    all_files = existing_files + new_files
    
    # Create file badges/cards
    file_badges = []
    for f in all_files:
        file_badges.append(
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.I(className="fas fa-file-alt fa-2x", style={'color': colors['primary']})
                        ], width='auto'),
                        dbc.Col([
                            html.Div(f['name'], style={'color': colors['text'], 'fontWeight': 'bold'}),
                            html.Small(f"{f['type']} | Uploaded: {f['upload_time']}", style={'color': colors['text-secondary']})
                        ]),
                        dbc.Col([
                            dbc.Badge(f['status'], color='success', className='me-1'),
                            html.I(className="fas fa-check-circle", style={'color': colors['success'], 'marginLeft': '5px'})
                        ], width='auto', style={'display': 'flex', 'alignItems': 'center'})
                    ], align='center')
                ], style={'padding': '10px 15px'})
            ], style={'backgroundColor': colors['surface'], 'marginBottom': '10px', 'border': f'1px solid {colors["success"]}'})
        )
    
    return file_badges, all_files


# Chat Callback
@app.callback(
    [Output('chat-messages', 'children'),
     Output('chat-history-store', 'data'),
     Output('chat-input', 'value')],
    [Input('send-chat', 'n_clicks'),
     Input('chat-input', 'n_submit'),
     Input('quick-stuck', 'n_clicks'),
     Input('quick-lost', 'n_clicks'),
     Input('quick-optimize', 'n_clicks')],
    [State('chat-input', 'value'),
     State('chat-history-store', 'data')]
)
def update_chat(send_clicks, input_submit, q_stuck, q_lost, q_optimize, user_message, chat_history):
    ctx = callback_context
    
    if not ctx.triggered:
        messages = [create_chat_message(msg['text'], msg['is_user']) for msg in chat_history]
        return messages, chat_history, ''
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Handle quick action buttons
    if triggered_id == 'quick-stuck':
        user_message = "How can I prevent stuck pipe?"
    elif triggered_id == 'quick-lost':
        user_message = "What should I do about lost circulation?"
    elif triggered_id == 'quick-optimize':
        user_message = "How can I optimize ROP?"
    
    if not user_message or user_message.strip() == '':
        messages = [create_chat_message(msg['text'], msg['is_user']) for msg in chat_history]
        return messages, chat_history, ''
    
    # Add user message
    chat_history.append({'text': user_message, 'is_user': True})
    
    # Generate AI response
    ai_response = get_ai_response(user_message)
    chat_history.append({'text': ai_response, 'is_user': False})
    
    # Create message elements
    messages = [create_chat_message(msg['text'], msg['is_user']) for msg in chat_history]
    
    return messages, chat_history, ''


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
     Output('alert-banner', 'children'),
     Output('rig-info-name', 'children'),
     Output('rig-info-well', 'children'),
     Output('rig-info-field', 'children'),
     Output('rig-info-operation', 'children'),
     Output('rig-info-days', 'children'),
     Output('rig-info-incident', 'children')],
    [Input('analyze-btn', 'n_clicks'),
     Input('interval-component', 'n_intervals')],
    [State('rig-selector', 'value'),
     State('time-window', 'value'),
     State('risk-threshold', 'value')]
)
def process_drilling_data(n_clicks, n_intervals, rig_id, time_window, threshold):
    if n_clicks == 0 and n_intervals == 0:
        return (None, '0%', '0%', '0%', '0%', '0 ft/hr', 'Avg: 0 ft/hr', '0 ft', 'Target: 0 ft', '', None,
                'RIG-001', 'WELL-XXX', 'Select a rig', 'N/A', '0 Days', '0 Days Since Incident')

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
            trend = html.Span(['📈 ', f'+{recent_risk - older_risk:.1f}%'],
                              style={'color': colors['danger']})
        else:
            trend = html.Span(['📉 ', f'{recent_risk - older_risk:.1f}%'],
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
        html.Div(alerts) if alerts else None,
        rig_id,
        metadata['well_name'],
        metadata['field'],
        metadata['current_operation'],
        f"{metadata['days_on_well']} Days",
        f"{metadata['last_incident']} Days Safe"
    )


@app.callback(
    Output('tab-content', 'children'),
    [Input('main-tabs', 'value'),
     Input('processed-data-store', 'data')]
)
def update_tab_content(active_tab, processed_data):
    if not processed_data:
        return html.Div([
            html.I(className="fas fa-chart-line fa-3x", style={'color': colors['text-secondary'], 'marginBottom': '20px'}),
            html.P("Click 'Analyze Risks' to start monitoring", style={'color': colors['text-secondary']})
        ], style={'textAlign': 'center', 'padding': '100px'})

    df = pd.DataFrame(processed_data['df'])

    if active_tab == 'real-time':
        # Create real-time parameters plot
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            subplot_titles=('Weight on Bit & RPM', 'Rate of Penetration',
                            'Torque & Stand Pipe Pressure', 'Mud Weight & Gas'),
            vertical_spacing=0.08,
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
            legend=dict(
                orientation='v',
                yanchor='middle',
                y=0.5,
                xanchor='left',
                x=1.02
            ),
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

        risk_colors = {
            'stuck_pipe': colors['warning'],
            'lost_circulation': colors['primary'],
            'kick': colors['danger'],
            'overall': colors['secondary']
        }

        for risk_type in ['stuck_pipe', 'lost_circulation', 'kick', 'overall']:
            fig.add_trace(
                go.Scatter(
                    x=risk_df['time'],
                    y=risk_df[risk_type],
                    name=risk_type.replace('_', ' ').title(),
                    mode='lines+markers',
                    line=dict(width=2, color=risk_colors[risk_type])
                )
            )

        # Add threshold line
        fig.add_hline(y=60, line_dash="dash", line_color=colors['danger'],
                      annotation_text="Risk Threshold", annotation_position="right")

        fig.update_layout(
            height=600,
            title=dict(text="Risk Evolution Over Time", font=dict(color=colors['primary'])),
            xaxis_title="Time",
            yaxis_title="Risk Level (%)",
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            plot_bgcolor=colors['surface'],
            paper_bgcolor=colors['surface'],
            font=dict(color=colors['text']),
            hovermode='x unified',
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)', range=[0, 100])
        )

        return dcc.Graph(figure=fig)

    elif active_tab == 'predictive':
        # Create predictive analysis
        predictions = pd.DataFrame(processed_data['predictions'])

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Stuck Pipe Probability', 'Lost Circulation Probability',
                            'Kick Probability', 'Combined Risk Forecast'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
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

        # Combined Risk (all risks together)
        fig.add_trace(
            go.Scatter(x=predictions['time'], y=predictions['stuck_pipe_prob'],
                       mode='lines', name='Stuck Pipe', line=dict(color=colors['warning'], width=2)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=predictions['time'], y=predictions['lost_circ_prob'],
                       mode='lines', name='Lost Circ', line=dict(color=colors['primary'], width=2)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=predictions['time'], y=predictions['kick_prob'],
                       mode='lines', name='Kick', line=dict(color=colors['danger'], width=2)),
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
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', range=[0, 100])

        return dcc.Graph(figure=fig)


# Updated risk heatmap callback with text always visible outside the bars:

@app.callback(
    Output('risk-heatmap', 'figure'),
    [Input('processed-data-store', 'data')]
)
def update_risk_heatmap(processed_data):
    if not processed_data:
        fig = go.Figure()
        fig.update_layout(
            plot_bgcolor=colors['surface'],
            paper_bgcolor=colors['surface'],
            font=dict(color=colors['text']),
            annotations=[dict(text="No data", x=0.5, y=0.5, showarrow=False, font=dict(color=colors['text-secondary']))]
        )
        return fig

    risks = processed_data['risks']

    categories = ['Stuck Pipe', 'Lost Circ.', 'Kick', 'Instability', 'Overall']
    values = [risks['stuck_pipe'], risks['lost_circulation'], risks['kick'], 
              risks['wellbore_instability'], risks['overall']]

    # Determine text position based on value - outside for low values, inside for high values
    text_positions = ['outside' if v < 25 else 'inside' for v in values]
    text_colors = [colors['text'] if v < 25 else 'white' for v in values]

    fig = go.Figure(data=[
        go.Bar(
            y=categories,
            x=values,
            orientation='h',
            marker=dict(
                color=values,
                colorscale=[[0, colors['success']], [0.5, colors['warning']], [1, colors['danger']]],
                cmin=0,
                cmax=100
            ),
            text=[f'{v:.0f}%' for v in values],
            textposition=text_positions,
            textfont=dict(size=12, family='Arial Black'),
            # Add cliponaxis=False to ensure text outside bars is visible
            cliponaxis=False
        )
    ])

    # Add text color for each bar using annotations for better control
    fig.update_layout(
        xaxis=dict(range=[0, 110], showgrid=False, title='Risk %'),  # Extended range to fit outside text
        yaxis=dict(showgrid=False),
        margin=dict(t=10, b=30, l=80, r=40),  # Added right margin for outside text
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['surface'],
        font=dict(color=colors['text']),
        height=220
    )

    # Update text colors individually using a workaround
    for i, (val, pos) in enumerate(zip(values, text_positions)):
        if pos == 'outside':
            fig.add_annotation(
                x=val + 2,
                y=categories[i],
                text=f'{val:.0f}%',
                showarrow=False,
                font=dict(size=12, color=colors['text'], family='Arial Black'),
                xanchor='left'
            )
    
    # Remove the default text for items with outside position
    fig.update_traces(
        text=[f'{v:.0f}%' if v >= 25 else '' for v in values]
    )

    return fig


@app.callback(
    Output('recommendations-table', 'children'),
    [Input('processed-data-store', 'data')]
)
def update_recommendations(processed_data):
    if not processed_data:
        return html.Div([
            html.I(className="fas fa-info-circle", style={'marginRight': '10px', 'color': colors['primary']}),
            html.Span("Run analysis to view recommendations", style={'color': colors['text-secondary']})
        ], style={'textAlign': 'center', 'padding': '30px'})

    risks = processed_data['risks']

    recommendations = []

    if risks['stuck_pipe'] > 40:
        recommendations.append({
            'Risk': 'Stuck Pipe',
            'Level': 'High' if risks['stuck_pipe'] > 70 else 'Medium',
            'Primary Action': 'Reduce WOB to 15-20 klbs, increase RPM',
            'Secondary': 'Pump viscous sweep, short trip',
            'Priority': '1'
        })

    if risks['lost_circulation'] > 35:
        recommendations.append({
            'Risk': 'Lost Circulation',
            'Level': 'High' if risks['lost_circulation'] > 65 else 'Medium',
            'Primary Action': 'Add LCM (15-25 ppb), reduce flow rate',
            'Secondary': 'Reduce mud weight by 0.3-0.5 ppg',
            'Priority': '2'
        })

    if risks['kick'] > 30:
        recommendations.append({
            'Risk': 'Kick',
            'Level': 'Critical' if risks['kick'] > 60 else 'Medium',
            'Primary Action': 'Increase mud weight, flow check',
            'Secondary': 'Alert well control, verify BOP',
            'Priority': '1'
        })

    if risks['wellbore_instability'] > 40:
        recommendations.append({
            'Risk': 'Wellbore',
            'Level': 'Medium',
            'Primary Action': 'Optimize mud weight, check inhibition',
            'Secondary': 'Monitor cavings, adjust rheology',
            'Priority': '3'
        })

    if not recommendations:
        recommendations.append({
            'Risk': 'All Systems',
            'Level': 'Low',
            'Primary Action': 'Continue current operations',
            'Secondary': 'Maintain standard monitoring',
            'Priority': '4'
        })

    return dash_table.DataTable(
        data=recommendations,
        columns=[{'name': col, 'id': col} for col in recommendations[0].keys()],
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'backgroundColor': colors['surface'],
            'color': colors['text'],
            'border': f'1px solid {colors["surface-light"]}',
            'padding': '10px',
            'fontSize': '13px',
            'whiteSpace': 'normal',
            'height': 'auto'
        },
        style_header={
            'backgroundColor': colors['primary'],
            'color': colors['background'],
            'fontWeight': 'bold',
            'fontSize': '12px'
        },
        style_data_conditional=[
            {
                'if': {'column_id': 'Level', 'filter_query': '{Level} = "Critical"'},
                'backgroundColor': colors['danger'],
                'color': 'white',
                'fontWeight': 'bold'
            },
            {
                'if': {'column_id': 'Level', 'filter_query': '{Level} = "High"'},
                'backgroundColor': colors['warning'],
                'color': 'white',
                'fontWeight': 'bold'
            },
            {
                'if': {'column_id': 'Level', 'filter_query': '{Level} = "Medium"'},
                'backgroundColor': colors['secondary'],
                'color': 'white'
            },
            {
                'if': {'column_id': 'Level', 'filter_query': '{Level} = "Low"'},
                'backgroundColor': colors['success'],
                'color': 'white'
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
        fig = go.Figure()
        fig.update_layout(
            plot_bgcolor=colors['surface'],
            paper_bgcolor=colors['surface'],
            font=dict(color=colors['text']),
            annotations=[dict(text="No data", x=0.5, y=0.5, showarrow=False)]
        )
        return fig

    events = processed_data['events']
    df = pd.DataFrame(processed_data['df'])

    fig = go.Figure()

    # Add depth trace
    fig.add_trace(
        go.Scatter(
            x=df['TIME'],
            y=df['DEPTH'],
            mode='lines',
            name='Depth',
            line=dict(color=colors['primary'], width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 255, 0.1)'
        )
    )

    # Add event markers
    colors_map = {
        'High': colors['danger'],
        'Medium': colors['warning'],
        'Low': colors['success']
    }

    for event in events:
        if event['start'] < len(df) and event['end'] <= len(df):
            start_time = df['TIME'].iloc[event['start']]
            end_time = df['TIME'].iloc[event['end'] - 1]
            mid_depth = df['DEPTH'].iloc[(event['start'] + event['end']) // 2]

            fig.add_vrect(
                x0=start_time, x1=end_time,
                fillcolor=colors_map.get(event['severity'], colors['warning']),
                opacity=0.3,
                line_width=2,
                line_color=colors_map.get(event['severity'], colors['warning'])
            )

            fig.add_annotation(
                x=start_time,
                y=mid_depth,
                text=event['type'],
                showarrow=True,
                arrowhead=2,
                arrowcolor=colors['text'],
                font=dict(size=10, color=colors['text']),
                bgcolor=colors['surface'],
                bordercolor=colors_map.get(event['severity'], colors['warning'])
            )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Depth (ft)",
        height=280,
        showlegend=False,
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['surface'],
        font=dict(color=colors['text']),
        margin=dict(t=10, b=40, l=50, r=10),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', autorange='reversed')
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
    app.run(debug=False, port=8052, host='127.0.0.1')