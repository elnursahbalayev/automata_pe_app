# geosteering_app.py - Complete Reservoir Navigation with Geosteering & Optimization Dashboard
import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])
app.title = "AUTOMATA INTELLIGENCE Reservoir Navigation with Geosteering & Optimization"

# ============================================================================
# DATA GENERATION FUNCTIONS
# ============================================================================

def generate_formation_model(well_id):
    """Generate 3D formation model for geosteering"""
    np.random.seed(hash(well_id) % 2**32)
    
    # Lateral distance (measured depth along horizontal)
    md_points = 500
    md = np.linspace(0, 5000, md_points)  # feet
    
    # Generate formation tops (wavy layers)
    base_frequency = 0.003
    
    # Shale cap (top boundary)
    shale_top = 10000 + 5 * np.sin(md * base_frequency) + np.random.normal(0, 0.5, md_points)
    
    # Reservoir top (landing target)
    reservoir_top = shale_top + 15 + 8 * np.sin(md * base_frequency * 1.5) + np.random.normal(0, 1, md_points)
    
    # Reservoir bottom
    reservoir_thickness = 40 + 10 * np.sin(md * base_frequency * 0.8)
    reservoir_bottom = reservoir_top + reservoir_thickness
    
    # Water contact
    water_contact = reservoir_bottom + 5 + 3 * np.sin(md * base_frequency * 2)
    
    # Smooth the curves
    shale_top = gaussian_filter1d(shale_top, sigma=3)
    reservoir_top = gaussian_filter1d(reservoir_top, sigma=3)
    reservoir_bottom = gaussian_filter1d(reservoir_bottom, sigma=3)
    water_contact = gaussian_filter1d(water_contact, sigma=3)
    
    return {
        'md': md,
        'shale_top': shale_top,
        'reservoir_top': reservoir_top,
        'reservoir_bottom': reservoir_bottom,
        'water_contact': water_contact,
        'reservoir_thickness': reservoir_thickness
    }

def generate_planned_trajectory(formation_model):
    """Generate planned well trajectory"""
    md = formation_model['md']
    
    # Plan to stay in middle of reservoir
    reservoir_mid = (formation_model['reservoir_top'] + formation_model['reservoir_bottom']) / 2
    
    # Add slight variations for realistic planning
    planned_tvd = reservoir_mid + np.random.normal(0, 2, len(md))
    planned_tvd = gaussian_filter1d(planned_tvd, sigma=5)
    
    return planned_tvd

def generate_actual_trajectory(formation_model, planned_tvd, current_md_index):
    """Generate actual drilled trajectory with some deviation"""
    md = formation_model['md'][:current_md_index]
    
    # Actual follows plan but with deviations
    actual_tvd = planned_tvd[:current_md_index].copy()
    
    # Add realistic drilling variations
    for i in range(20, len(actual_tvd)):
        deviation = np.random.normal(0, 0.5)
        actual_tvd[i] = actual_tvd[i-1] + (planned_tvd[i] - planned_tvd[i-1]) + deviation
    
    actual_tvd = gaussian_filter1d(actual_tvd, sigma=2)
    
    return actual_tvd

def generate_lwd_data(formation_model, actual_tvd, current_md_index):
    """Generate Logging While Drilling (LWD) data"""
    md = formation_model['md'][:current_md_index]
    
    # Initialize arrays
    gamma_ray = np.zeros(len(md))
    resistivity = np.zeros(len(md))
    porosity = np.zeros(len(md))
    
    for i, (m, tvd) in enumerate(zip(md, actual_tvd)):
        # Determine which formation we're in
        res_top = np.interp(m, formation_model['md'], formation_model['reservoir_top'])
        res_bottom = np.interp(m, formation_model['md'], formation_model['reservoir_bottom'])
        shale_top = np.interp(m, formation_model['md'], formation_model['shale_top'])
        
        if tvd < shale_top:
            # Above reservoir (shale)
            gamma_ray[i] = np.random.normal(90, 10)
            resistivity[i] = np.random.lognormal(1.0, 0.3)
            porosity[i] = np.random.normal(0.08, 0.02)
        elif tvd < res_top:
            # Upper shale/transition
            gamma_ray[i] = np.random.normal(75, 12)
            resistivity[i] = np.random.lognormal(1.5, 0.4)
            porosity[i] = np.random.normal(0.12, 0.03)
        elif tvd <= res_bottom:
            # In reservoir (target zone)
            distance_from_top = tvd - res_top
            distance_from_bottom = res_bottom - tvd
            
            # Quality decreases near boundaries
            quality_factor = min(distance_from_top, distance_from_bottom) / 10
            quality_factor = np.clip(quality_factor, 0.3, 1.0)
            
            gamma_ray[i] = np.random.normal(25, 8) / quality_factor
            resistivity[i] = np.random.lognormal(3.0, 0.5) * quality_factor
            porosity[i] = np.random.normal(0.22, 0.03) * quality_factor
        else:
            # Below reservoir (wet zone or shale)
            gamma_ray[i] = np.random.normal(60, 15)
            resistivity[i] = np.random.lognormal(1.2, 0.3)
            porosity[i] = np.random.normal(0.15, 0.04)
    
    # Smooth the logs
    gamma_ray = gaussian_filter1d(np.clip(gamma_ray, 0, 150), sigma=2)
    resistivity = gaussian_filter1d(np.clip(resistivity, 0.5, 1000), sigma=2)
    porosity = gaussian_filter1d(np.clip(porosity, 0, 0.35), sigma=2)
    
    return {
        'md': md,
        'gamma_ray': gamma_ray,
        'resistivity': resistivity,
        'porosity': porosity
    }

def calculate_directional_survey(md_array, actual_tvd):
    """Calculate directional survey data"""
    # Inclination calculation
    inclination = np.zeros(len(md_array))
    azimuth = np.zeros(len(md_array))
    
    for i in range(1, len(md_array)):
        delta_md = md_array[i] - md_array[i-1]
        delta_tvd = actual_tvd[i] - actual_tvd[i-1]
        
        if delta_md > 0:
            inclination[i] = np.arccos(delta_tvd / delta_md) * 180 / np.pi
        else:
            inclination[i] = inclination[i-1]
        
        # Azimuth (simplified - assume roughly eastward)
        azimuth[i] = 90 + np.random.normal(0, 2)
    
    inclination = gaussian_filter1d(np.clip(inclination, 0, 90), sigma=3)
    azimuth = gaussian_filter1d(azimuth, sigma=3)
    
    # Dogleg Severity (DLS)
    dls = np.zeros(len(md_array))
    for i in range(1, len(md_array)):
        delta_inc = inclination[i] - inclination[i-1]
        delta_md = md_array[i] - md_array[i-1]
        if delta_md > 0:
            dls[i] = abs(delta_inc) / delta_md * 100  # degrees per 100 ft
    
    dls = gaussian_filter1d(dls, sigma=2)
    
    return inclination, azimuth, dls

def calculate_distance_to_boundaries(formation_model, actual_tvd, current_md):
    """Calculate distance to formation boundaries"""
    res_top = np.interp(current_md, formation_model['md'], formation_model['reservoir_top'])
    res_bottom = np.interp(current_md, formation_model['md'], formation_model['reservoir_bottom'])
    current_tvd = actual_tvd[-1]
    
    distance_to_top = current_tvd - res_top
    distance_to_bottom = res_bottom - current_tvd
    
    return {
        'to_top': distance_to_top,
        'to_bottom': distance_to_bottom,
        'in_zone': res_top <= current_tvd <= res_bottom
    }

def generate_geosteering_recommendations(formation_model, actual_tvd, lwd_data, current_md_index):
    """AI-powered geosteering recommendations"""
    current_md = formation_model['md'][current_md_index-1]
    current_tvd = actual_tvd[-1]
    
    # Get formation boundaries ahead
    look_ahead_md = min(current_md + 200, formation_model['md'][-1])
    look_ahead_indices = (formation_model['md'] >= current_md) & (formation_model['md'] <= look_ahead_md)
    
    avg_res_top = np.mean(formation_model['reservoir_top'][look_ahead_indices])
    avg_res_bottom = np.mean(formation_model['reservoir_bottom'][look_ahead_indices])
    
    # Calculate optimal target TVD
    target_tvd = (avg_res_top + avg_res_bottom) / 2
    
    # Current position relative to target
    position_error = current_tvd - target_tvd
    
    # Recent log quality
    recent_gr = lwd_data['gamma_ray'][-20:].mean() if len(lwd_data['gamma_ray']) > 20 else 50
    recent_res = lwd_data['resistivity'][-20:].mean() if len(lwd_data['resistivity']) > 20 else 10
    
    recommendations = []
    
    # Generate steering advice
    if position_error > 5:
        recommendations.append({
            'action': 'Build Angle',
            'magnitude': 'Moderate',
            'reason': f'Trajectory {position_error:.1f} ft too deep',
            'target_inc_change': '+0.5°/100ft',
            'priority': 'High',
            'confidence': 92
        })
    elif position_error < -5:
        recommendations.append({
            'action': 'Drop Angle',
            'magnitude': 'Moderate',
            'reason': f'Trajectory {abs(position_error):.1f} ft too shallow',
            'target_inc_change': '-0.5°/100ft',
            'priority': 'High',
            'confidence': 92
        })
    else:
        recommendations.append({
            'action': 'Hold Angle',
            'magnitude': 'Maintain',
            'reason': 'On optimal trajectory',
            'target_inc_change': '0°/100ft',
            'priority': 'Normal',
            'confidence': 95
        })
    
    # Log quality assessment
    if recent_gr > 60:
        recommendations.append({
            'action': 'Monitor GR',
            'magnitude': 'High',
            'reason': 'Approaching shale boundary',
            'target_inc_change': 'Build if increasing',
            'priority': 'Medium',
            'confidence': 88
        })
    
    if recent_res < 10:
        recommendations.append({
            'action': 'Check Water Contact',
            'magnitude': 'Critical',
            'reason': 'Low resistivity detected',
            'target_inc_change': 'Build immediately',
            'priority': 'Critical',
            'confidence': 85
        })
    
    return recommendations

def generate_offset_wells(formation_model):
    """Generate offset well data for correlation"""
    offset_wells = []
    
    for i in range(3):
        md_offset = formation_model['md'].copy()
        
        # Offset wells have slightly different formation depths
        depth_offset = np.random.uniform(-10, 10)
        
        offset_wells.append({
            'name': f'OFFSET-{i+1}',
            'md': md_offset,
            'reservoir_top': formation_model['reservoir_top'] + depth_offset,
            'reservoir_bottom': formation_model['reservoir_bottom'] + depth_offset,
            'production_rate': np.random.uniform(800, 1500),
            'lateral_distance': np.random.uniform(500, 2000)
        })
    
    return offset_wells

def generate_well_metadata(well_id):
    """Generate well metadata"""
    np.random.seed(hash(well_id) % 2**32)
    
    metadata = {
        'well_id': well_id,
        'well_name': f'HORIZONTAL-{np.random.randint(100, 999)}',
        'field': np.random.choice(['Eagle Ford', 'Permian Basin', 'Bakken', 'STACK', 'Marcellus']),
        'operator': 'AUTOMATA Drilling',
        'rig': f'RIG-{np.random.randint(10, 50)}',
        'target_formation': 'Wolfcamp A',
        'planned_lateral_length': 5000,
        'spud_date': datetime.now() - timedelta(days=np.random.randint(5, 15)),
        'current_operation': 'Drilling Lateral',
        'target_azimuth': 90,  # East
        'landing_point_md': 10000,
        'kickoff_point_md': 8500
    }
    
    return metadata

# ============================================================================
# GENERATE DATA FOR MULTIPLE WELLS
# ============================================================================

print("Generating geosteering data... This may take a moment...")

wells_data = {}
wells_metadata = {}

for well_id in ['WELL-GS-001', 'WELL-GS-002', 'WELL-GS-003']:
    print(f"Generating {well_id}...")
    
    formation = generate_formation_model(well_id)
    planned_traj = generate_planned_trajectory(formation)
    
    # Simulate current drilling progress (random between 40% and 80%)
    progress = np.random.uniform(0.4, 0.8)
    current_md_index = int(len(formation['md']) * progress)
    
    actual_traj = generate_actual_trajectory(formation, planned_traj, current_md_index)
    lwd_data = generate_lwd_data(formation, actual_traj, current_md_index)
    
    inclination, azimuth, dls = calculate_directional_survey(lwd_data['md'], actual_traj)
    
    offset_wells = generate_offset_wells(formation)
    
    wells_data[well_id] = {
        'formation': formation,
        'planned_trajectory': planned_traj,
        'actual_trajectory': actual_traj,
        'lwd_data': lwd_data,
        'current_md_index': current_md_index,
        'inclination': inclination,
        'azimuth': azimuth,
        'dls': dls,
        'offset_wells': offset_wells
    }
    
    wells_metadata[well_id] = generate_well_metadata(well_id)

print("Data generation complete!")

# Define color scheme
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
                html.H1("🎯 Reservoir Navigation with Geosteering & Optimization",
                        style={'color': colors['text'], 'fontWeight': 'bold'}),
                html.P("Real-Time Directional Drilling & Formation Navigation",
                       style={'color': colors['text-secondary'], 'fontSize': '18px'})
            ], style={'textAlign': 'center', 'padding': '30px 0'})
        ])
    ]),
    
    # Control Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Well Selection & Geosteering Controls", style={'color': colors['primary']}),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Well", style={'color': colors['text']}),
                            dcc.Dropdown(
                                id='well-selector',
                                options=[{'label': f'{well} - {wells_metadata[well]["well_name"]}', 
                                         'value': well} for well in wells_metadata.keys()],
                                value='WELL-GS-001',
                                style={'color': '#000'}
                            )
                        ], md=3),
                        dbc.Col([
                            html.Label("Display Mode", style={'color': colors['text']}),
                            dcc.Dropdown(
                                id='display-mode',
                                options=[
                                    {'label': 'Trajectory View', 'value': 'trajectory'},
                                    {'label': 'LWD Logs', 'value': 'lwd'},
                                    {'label': 'Formation Correlation', 'value': 'correlation'}
                                ],
                                value='trajectory',
                                style={'color': '#000'}
                            )
                        ], md=2),
                        dbc.Col([
                            html.Label("Steering Sensitivity", style={'color': colors['text']}),
                            dcc.Slider(
                                id='steering-sensitivity',
                                min=1,
                                max=5,
                                value=3,
                                marks={1: 'Low', 3: 'Med', 5: 'High'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], md=3),
                        dbc.Col([
                            html.Label("Update Mode", style={'color': colors['text']}),
                            dcc.RadioItems(
                                id='update-mode',
                                options=[
                                    {'label': ' Real-Time', 'value': 'realtime'},
                                    {'label': ' Manual', 'value': 'manual'}
                                ],
                                value='manual',
                                inline=True,
                                style={'color': colors['text']}
                            )
                        ], md=2),
                        dbc.Col([
                            dbc.Button("Update View", id='update-btn', color='primary',
                                      className='w-100', style={'marginTop': '25px'}, n_clicks=0)
                        ], md=2)
                    ])
                ])
            ], style={'backgroundColor': colors['surface'], 'marginBottom': '20px'})
        ])
    ]),
    
    # Alert Banner for Critical Steering Recommendations
    dbc.Row([
        dbc.Col([
            html.Div(id='steering-alerts')
        ])
    ], className='mb-3'),
    
    # Key Real-Time Metrics
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='current-md', children='0 ft', style={'color': colors['primary']}),
                    html.P('Current MD', style={'color': colors['text-secondary']}),
                    html.Small(id='lateral-progress', style={'fontSize': '12px'})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='current-tvd', children='0 ft', style={'color': colors['success']}),
                    html.P('Current TVD', style={'color': colors['text-secondary']}),
                    html.Small(id='tvd-target', style={'fontSize': '12px'})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='dist-to-top', children='0 ft', style={'color': colors['warning']}),
                    html.P('Distance to Top', style={'color': colors['text-secondary']}),
                    html.Small(id='zone-status', style={'fontSize': '12px'})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='dist-to-bottom', children='0 ft', style={'color': colors['warning']}),
                    html.P('Distance to Bottom', style={'color': colors['text-secondary']}),
                    html.Small('Safe margin', style={'fontSize': '12px', 'color': colors['text-secondary']})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='current-inc', children='0°', style={'color': colors['secondary']}),
                    html.P('Inclination', style={'color': colors['text-secondary']}),
                    html.Small(id='inc-trend', style={'fontSize': '12px'})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='current-dls', children='0°/100ft', style={'color': colors['primary']}),
                    html.P('Dogleg Severity', style={'color': colors['text-secondary']}),
                    html.Small(id='dls-status', style={'fontSize': '12px'})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2)
    ], className='mb-4'),
    
    # Main Geosteering Canvas
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Geosteering Canvas", style={'color': colors['primary'], 'marginBottom': '20px'}),
                    dcc.Graph(id='main-geosteering-plot', style={'height': '500px'})
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=9),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("LWD Real-Time", style={'color': colors['primary'], 'marginBottom': '20px'}),
                    dcc.Graph(id='lwd-strip-log', style={'height': '500px'})
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=3)
    ], className='mb-4'),
    
    # Detailed Analysis Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Formation Correlation", style={'color': colors['primary']}),
                    dcc.Graph(id='formation-correlation', style={'height': '300px'})
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Directional Survey", style={'color': colors['primary']}),
                    dcc.Graph(id='directional-survey', style={'height': '300px'})
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=6)
    ], className='mb-4'),
    
    # Recommendations Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("AI Steering Recommendations", style={'color': colors['primary']}),
                    html.Div(id='recommendations-table')
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=7),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Offset Well Comparison", style={'color': colors['primary']}),
                    html.Div(id='offset-wells-table')
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=5)
    ]),
    
    # Store components
    dcc.Store(id='processed-data-store'),
    dcc.Interval(id='interval-component', interval=3000, n_intervals=0, disabled=True)
    
], fluid=True, style={'backgroundColor': colors['background'], 'minHeight': '100vh', 'padding': '20px'})

# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    [Output('processed-data-store', 'data'),
     Output('current-md', 'children'),
     Output('lateral-progress', 'children'),
     Output('current-tvd', 'children'),
     Output('tvd-target', 'children'),
     Output('dist-to-top', 'children'),
     Output('zone-status', 'children'),
     Output('dist-to-bottom', 'children'),
     Output('current-inc', 'children'),
     Output('inc-trend', 'children'),
     Output('current-dls', 'children'),
     Output('dls-status', 'children'),
     Output('steering-alerts', 'children')],
    [Input('update-btn', 'n_clicks'),
     Input('interval-component', 'n_intervals')],
    [State('well-selector', 'value'),
     State('steering-sensitivity', 'value')]
)
def process_geosteering_data(n_clicks, n_intervals, well_id, sensitivity):
    if n_clicks == 0 and n_intervals == 0:
        return (None, '0 ft', '', '0 ft', '', '0 ft', '', '0 ft', '0°', '', '0°/100ft', '', None)
    
    # Get well data
    well_data = wells_data[well_id]
    metadata = wells_metadata[well_id]
    
    current_md_idx = well_data['current_md_index']
    current_md = well_data['lwd_data']['md'][-1]
    current_tvd = well_data['actual_trajectory'][-1]
    
    # Calculate distances to boundaries
    boundaries = calculate_distance_to_boundaries(
        well_data['formation'],
        well_data['actual_trajectory'],
        current_md
    )
    
    # Current directional parameters
    current_inc = well_data['inclination'][-1]
    current_dls = well_data['dls'][-1]
    
    # Inclination trend
    if len(well_data['inclination']) > 10:
        inc_trend_val = well_data['inclination'][-1] - well_data['inclination'][-10]
        if inc_trend_val > 0.5:
            inc_trend = html.Span(['📈 Building'], style={'color': colors['success']})
        elif inc_trend_val < -0.5:
            inc_trend = html.Span(['📉 Dropping'], style={'color': colors['warning']})
        else:
            inc_trend = html.Span(['➡️ Holding'], style={'color': colors['text-secondary']})
    else:
        inc_trend = ''
    
    # DLS status
    if current_dls > 8:
        dls_status = html.Span('High', style={'color': colors['danger']})
    elif current_dls > 5:
        dls_status = html.Span('Moderate', style={'color': colors['warning']})
    else:
        dls_status = html.Span('Normal', style={'color': colors['success']})
    
    # Zone status
    if boundaries['in_zone']:
        zone_status = html.Span('✓ In Zone', style={'color': colors['success']})
    else:
        zone_status = html.Span('✗ Out of Zone', style={'color': colors['danger']})
    
    # Progress
    progress_pct = (current_md / metadata['planned_lateral_length']) * 100
    lateral_progress = f"{progress_pct:.1f}% Complete"
    
    # Target TVD
    target_tvd_val = (np.interp(current_md, well_data['formation']['md'], well_data['formation']['reservoir_top']) + 
                      np.interp(current_md, well_data['formation']['md'], well_data['formation']['reservoir_bottom'])) / 2
    tvd_target = f"Target: {target_tvd_val:.0f} ft"
    
    # Generate recommendations
    recommendations = generate_geosteering_recommendations(
        well_data['formation'],
        well_data['actual_trajectory'],
        well_data['lwd_data'],
        current_md_idx
    )
    
    # Generate alerts
    alerts = []
    for rec in recommendations:
        if rec['priority'] == 'Critical':
            alerts.append(
                dbc.Alert([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    html.Strong("CRITICAL: "),
                    f"{rec['action']} - {rec['reason']}"
                ], color="danger", dismissable=True)
            )
        elif rec['priority'] == 'High':
            alerts.append(
                dbc.Alert([
                    html.I(className="fas fa-info-circle me-2"),
                    html.Strong("STEERING: "),
                    f"{rec['action']} - {rec['reason']}"
                ], color="warning", dismissable=True)
            )
    
    # Prepare data for storage
    processed_data = {
        'well_id': well_id,
        'formation': {
            'md': well_data['formation']['md'].tolist(),
            'shale_top': well_data['formation']['shale_top'].tolist(),
            'reservoir_top': well_data['formation']['reservoir_top'].tolist(),
            'reservoir_bottom': well_data['formation']['reservoir_bottom'].tolist(),
            'water_contact': well_data['formation']['water_contact'].tolist()
        },
        'planned_trajectory': well_data['planned_trajectory'].tolist(),
        'actual_trajectory': well_data['actual_trajectory'].tolist(),
        'lwd_data': {
            'md': well_data['lwd_data']['md'].tolist(),
            'gamma_ray': well_data['lwd_data']['gamma_ray'].tolist(),
            'resistivity': well_data['lwd_data']['resistivity'].tolist(),
            'porosity': well_data['lwd_data']['porosity'].tolist()
        },
        'inclination': well_data['inclination'].tolist(),
        'azimuth': well_data['azimuth'].tolist(),
        'dls': well_data['dls'].tolist(),
        'recommendations': recommendations,
        'offset_wells': well_data['offset_wells'],
        'metadata': {k: str(v) if isinstance(v, datetime) else v for k, v in metadata.items()}
    }
    
    return (
        processed_data,
        f"{current_md:.0f} ft",
        lateral_progress,
        f"{current_tvd:.0f} ft",
        tvd_target,
        f"{boundaries['to_top']:.1f} ft",
        zone_status,
        f"{boundaries['to_bottom']:.1f} ft",
        f"{current_inc:.1f}°",
        inc_trend,
        f"{current_dls:.2f}°/100ft",
        dls_status,
        html.Div(alerts) if alerts else None
    )

@app.callback(
    Output('main-geosteering-plot', 'figure'),
    [Input('processed-data-store', 'data'),
     Input('display-mode', 'value')]
)
def update_main_plot(processed_data, display_mode):
    if not processed_data:
        return go.Figure()
    
    formation = processed_data['formation']
    
    if display_mode == 'trajectory':
        # Main trajectory view with formation layers
        fig = go.Figure()
        
        # Add formation layers
        fig.add_trace(go.Scatter(
            x=formation['md'],
            y=formation['shale_top'],
            mode='lines',
            name='Shale Top',
            line=dict(color='#795548', width=2),
            fill=None
        ))
        
        fig.add_trace(go.Scatter(
            x=formation['md'],
            y=formation['reservoir_top'],
            mode='lines',
            name='Reservoir Top',
            line=dict(color=colors['warning'], width=3),
            fill=None
        ))
        
        fig.add_trace(go.Scatter(
            x=formation['md'],
            y=formation['reservoir_bottom'],
            mode='lines',
            name='Reservoir Bottom',
            line=dict(color=colors['warning'], width=3),
            fill='tonexty',
            fillcolor='rgba(255, 170, 0, 0.2)'
        ))
        
        fig.add_trace(go.Scatter(
            x=formation['md'],
            y=formation['water_contact'],
            mode='lines',
            name='Water Contact',
            line=dict(color=colors['primary'], width=2, dash='dash')
        ))
        
        # Add planned trajectory
        fig.add_trace(go.Scatter(
            x=formation['md'],
            y=processed_data['planned_trajectory'],
            mode='lines',
            name='Planned Path',
            line=dict(color=colors['text-secondary'], width=2, dash='dot')
        ))
        
        # Add actual trajectory
        fig.add_trace(go.Scatter(
            x=processed_data['lwd_data']['md'],
            y=processed_data['actual_trajectory'],
            mode='lines',
            name='Actual Path',
            line=dict(color=colors['success'], width=3)
        ))
        
        # Add current bit position
        fig.add_trace(go.Scatter(
            x=[processed_data['lwd_data']['md'][-1]],
            y=[processed_data['actual_trajectory'][-1]],
            mode='markers',
            name='Current Bit Position',
            marker=dict(size=15, color=colors['danger'], symbol='diamond')
        ))
        
        fig.update_layout(
            title="Geosteering Trajectory View",
            xaxis_title="Measured Depth (ft)",
            yaxis_title="True Vertical Depth (ft)",
            yaxis=dict(autorange='reversed'),
            height=500,
            showlegend=True,
            legend=dict(x=0.01, y=0.99),
            plot_bgcolor=colors['surface'],
            paper_bgcolor=colors['surface'],
            font=dict(color=colors['text']),
            hovermode='x unified'
        )
        
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
        
    elif display_mode == 'lwd':
        # LWD logs view
        fig = make_subplots(
            rows=1, cols=3,
            shared_yaxes=True,
            subplot_titles=['Gamma Ray', 'Resistivity', 'Porosity'],
            horizontal_spacing=0.05
        )
        
        lwd = processed_data['lwd_data']
        
        # Gamma Ray
        fig.add_trace(
            go.Scatter(x=lwd['gamma_ray'], y=lwd['md'],
                      mode='lines', name='GR',
                      line=dict(color=colors['success'], width=2)),
            row=1, col=1
        )
        
        # Resistivity
        fig.add_trace(
            go.Scatter(x=lwd['resistivity'], y=lwd['md'],
                      mode='lines', name='Resistivity',
                      line=dict(color=colors['secondary'], width=2)),
            row=1, col=2
        )
        
        # Porosity
        fig.add_trace(
            go.Scatter(x=lwd['porosity'], y=lwd['md'],
                      mode='lines', name='Porosity',
                      line=dict(color=colors['primary'], width=2),
                      fill='tozerox', fillcolor='rgba(0, 212, 255, 0.2)'),
            row=1, col=3
        )
        
        fig.update_xaxes(title='API', range=[0, 150], row=1, col=1)
        fig.update_xaxes(title='Ohm.m', type='log', row=1, col=2)
        fig.update_xaxes(title='v/v', range=[0, 0.35], row=1, col=3)
        
        fig.update_yaxes(title='Measured Depth (ft)', autorange='reversed', row=1, col=1)
        
        fig.update_layout(
            height=500,
            showlegend=False,
            plot_bgcolor=colors['surface'],
            paper_bgcolor=colors['surface'],
            font=dict(color=colors['text'])
        )
        
    elif display_mode == 'correlation':
        # Formation correlation with offset wells
        fig = go.Figure()
        
        # Add target well formations
        fig.add_trace(go.Scatter(
            x=formation['md'],
            y=formation['reservoir_top'],
            mode='lines',
            name='Target Well - Top',
            line=dict(color=colors['primary'], width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=formation['md'],
            y=formation['reservoir_bottom'],
            mode='lines',
            name='Target Well - Bottom',
            line=dict(color=colors['primary'], width=3, dash='dash')
        ))
        
        # Add offset wells
        offset_colors = [colors['success'], colors['warning'], colors['secondary']]
        for i, offset in enumerate(processed_data['offset_wells'][:3]):
            fig.add_trace(go.Scatter(
                x=offset['md'],
                y=offset['reservoir_top'],
                mode='lines',
                name=f"{offset['name']} - Top",
                line=dict(color=offset_colors[i], width=2, dash='dot')
            ))
        
        # Add actual trajectory
        fig.add_trace(go.Scatter(
            x=processed_data['lwd_data']['md'],
            y=processed_data['actual_trajectory'],
            mode='lines',
            name='Current Well Path',
            line=dict(color=colors['danger'], width=3)
        ))
        
        fig.update_layout(
            title="Formation Correlation with Offset Wells",
            xaxis_title="Measured Depth (ft)",
            yaxis_title="True Vertical Depth (ft)",
            yaxis=dict(autorange='reversed'),
            height=500,
            showlegend=True,
            plot_bgcolor=colors['surface'],
            paper_bgcolor=colors['surface'],
            font=dict(color=colors['text'])
        )
    
    return fig

@app.callback(
    Output('lwd-strip-log', 'figure'),
    [Input('processed-data-store', 'data')]
)
def update_lwd_strip(processed_data):
    if not processed_data:
        return go.Figure()
    
    lwd = processed_data['lwd_data']
    
    # Show only last 500 ft for real-time view
    window_size = min(500, len(lwd['md']))
    
    fig = make_subplots(
        rows=1, cols=3,
        shared_yaxes=True,
        subplot_titles=['GR', 'Res', 'Por'],
        horizontal_spacing=0.1,
        column_widths=[0.33, 0.33, 0.34]
    )
    
    # Gamma Ray
    fig.add_trace(
        go.Scatter(
            x=lwd['gamma_ray'][-window_size:],
            y=lwd['md'][-window_size:],
            mode='lines',
            line=dict(color=colors['success'], width=2),
            fill='tozerox',
            fillcolor='rgba(0, 255, 136, 0.3)',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Resistivity
    fig.add_trace(
        go.Scatter(
            x=lwd['resistivity'][-window_size:],
            y=lwd['md'][-window_size:],
            mode='lines',
            line=dict(color=colors['secondary'], width=2),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Porosity
    fig.add_trace(
        go.Scatter(
            x=lwd['porosity'][-window_size:],
            y=lwd['md'][-window_size:],
            mode='lines',
            line=dict(color=colors['primary'], width=2),
            fill='tozerox',
            fillcolor='rgba(0, 212, 255, 0.3)',
            showlegend=False
        ),
        row=1, col=3
    )
    
    fig.update_xaxes(range=[0, 150], row=1, col=1)
    fig.update_xaxes(type='log', range=[-1, 3], row=1, col=2)
    fig.update_xaxes(range=[0, 0.35], row=1, col=3)
    
    fig.update_yaxes(autorange='reversed', row=1, col=1)
    
    fig.update_layout(
        height=500,
        showlegend=False,
        margin=dict(t=40, b=20, l=20, r=20),
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['surface'],
        font=dict(color=colors['text'], size=10)
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig

@app.callback(
    Output('formation-correlation', 'figure'),
    [Input('processed-data-store', 'data')]
)
def update_formation_correlation(processed_data):
    if not processed_data:
        return go.Figure()
    
    lwd = processed_data['lwd_data']
    
    # Crossplot: GR vs Resistivity colored by depth
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=lwd['gamma_ray'],
        y=lwd['resistivity'],
        mode='markers',
        marker=dict(
            size=6,
            color=lwd['md'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="MD (ft)", len=0.7)
        ),
        text=[f"MD: {md:.0f} ft" for md in lwd['md']],
        hovertemplate="GR: %{x:.1f}<br>Res: %{y:.1f}<br>%{text}<extra></extra>"
    ))
    
    # Add zone indicators
    fig.add_hline(y=20, line_dash="dash", line_color=colors['success'],
                 annotation_text="HC Zone Threshold")
    fig.add_vline(x=60, line_dash="dash", line_color=colors['warning'],
                 annotation_text="Shale Cutoff")
    
    fig.update_layout(
        title="Formation Crossplot (GR vs Resistivity)",
        xaxis_title="Gamma Ray (API)",
        yaxis_title="Resistivity (Ohm.m)",
        yaxis_type='log',
        height=300,
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['surface'],
        font=dict(color=colors['text'])
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig

@app.callback(
    Output('directional-survey', 'figure'),
    [Input('processed-data-store', 'data')]
)
def update_directional_survey(processed_data):
    if not processed_data:
        return go.Figure()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Inclination', 'Dogleg Severity'],
        specs=[[{'secondary_y': False}, {'secondary_y': False}]]
    )
    
    lwd_md = processed_data['lwd_data']['md']
    
    # Inclination
    fig.add_trace(
        go.Scatter(
            x=lwd_md,
            y=processed_data['inclination'],
            mode='lines',
            line=dict(color=colors['primary'], width=2),
            name='Inclination'
        ),
        row=1, col=1
    )
    
    # DLS
    fig.add_trace(
        go.Scatter(
            x=lwd_md,
            y=processed_data['dls'],
            mode='lines',
            line=dict(color=colors['secondary'], width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 53, 0.2)',
            name='DLS'
        ),
        row=1, col=2
    )
    
    # Add DLS limit line
    fig.add_hline(y=8, line_dash="dash", line_color=colors['danger'],
                 annotation_text="Max DLS", row=1, col=2)
    
    fig.update_xaxes(title="Measured Depth (ft)", gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(title="Degrees", gridcolor='rgba(255,255,255,0.1)', row=1, col=1)
    fig.update_yaxes(title="°/100ft", gridcolor='rgba(255,255,255,0.1)', row=1, col=2)
    
    fig.update_layout(
        height=300,
        showlegend=False,
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['surface'],
        font=dict(color=colors['text'])
    )
    
    return fig

@app.callback(
    Output('recommendations-table', 'children'),
    [Input('processed-data-store', 'data')]
)
def update_recommendations(processed_data):
    if not processed_data:
        return html.Div()
    
    recommendations = processed_data['recommendations']
    
    return dash_table.DataTable(
        data=recommendations,
        columns=[{'name': col, 'id': col} for col in recommendations[0].keys()],
        style_cell={
            'textAlign': 'left',
            'backgroundColor': colors['surface'],
            'color': colors['text'],
            'border': f'1px solid {colors["primary"]}40',
            'padding': '12px'
        },
        style_header={
            'backgroundColor': colors['primary'],
            'color': colors['background'],
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'column_id': 'priority', 'filter_query': '{priority} = "Critical"'},
                'backgroundColor': colors['danger'],
                'color': 'white',
                'fontWeight': 'bold'
            },
            {
                'if': {'column_id': 'priority', 'filter_query': '{priority} = "High"'},
                'backgroundColor': colors['warning'],
                'color': 'white',
            }
        ],
        sort_action="native",
        sort_by=[{'column_id': 'confidence', 'direction': 'desc'}]
    )

@app.callback(
    Output('offset-wells-table', 'children'),
    [Input('processed-data-store', 'data')]
)
def update_offset_wells(processed_data):
    if not processed_data:
        return html.Div()
    
    offset_data = []
    for well in processed_data['offset_wells']:
        offset_data.append({
            'Well': well['name'],
            'Distance (ft)': f"{well['lateral_distance']:.0f}",
            'Production (bbl/d)': f"{well['production_rate']:.0f}",
            'Status': 'Producing'
        })
    
    return dash_table.DataTable(
        data=offset_data,
        columns=[{'name': col, 'id': col} for col in offset_data[0].keys()],
        style_cell={
            'textAlign': 'center',
            'backgroundColor': colors['surface'],
            'color': colors['text'],
            'border': f'1px solid {colors["primary"]}40',
            'padding': '10px'
        },
        style_header={
            'backgroundColor': colors['primary'],
            'color': colors['background'],
            'fontWeight': 'bold'
        }
    )

@app.callback(
    Output('interval-component', 'disabled'),
    [Input('update-mode', 'value')]
)
def toggle_realtime(mode):
    return mode != 'realtime'

# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("AUTOMATA INTELLIGENCE")
    print("Reservoir Navigation with Geosteering & Optimization Suite")
    print("="*70)
    print("\nStarting server on http://127.0.0.1:8052")
    print("Press CTRL+C to stop\n")
    
    app.run(debug=False, port=8052, host='127.0.0.1')