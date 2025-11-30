# reservoir_geosteering_app.py - Fixed version with consistent colors and UI improvements
import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc
from scipy.interpolate import griddata, interp1d
from scipy.ndimage import gaussian_filter
import warnings

warnings.filterwarnings('ignore')

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])
app.title = "AUTOMATA INTELLIGENCE Reservoir Navigation with Geosteering & Optimization"

# ============================================================================
# DEFINE CONSISTENT FORMATION COLORS
# ============================================================================

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

# CONSISTENT FORMATION COLORS FOR ALL VISUALIZATIONS
formation_colors = {
    'Overburden': '#8B7355',  # Brown
    'Cap Rock': '#696969',  # Dark Gray
    'Upper Sand': '#FFD700',  # Gold
    'Middle Shale': '#8B7355',  # Brown
    'Target Zone': '#00ff88',  # Bright Green (success color)
    'Lower Sand': '#F4A460',  # Sandy Brown
    'Base Shale': '#654321',  # Dark Brown
    'Unknown': '#808080'  # Gray
}


# ============================================================================
# DATA GENERATION FUNCTIONS (Updated with better DLS calculation)
# ============================================================================

def calculate_dogleg_severity(inc1, inc2, azi1, azi2, md_interval):
    """Calculate dogleg severity between two survey points"""
    if md_interval == 0:
        return 0

    # Convert to radians
    inc1_rad = np.radians(inc1)
    inc2_rad = np.radians(inc2)
    azi1_rad = np.radians(azi1)
    azi2_rad = np.radians(azi2)

    # Calculate dogleg angle using minimum curvature method
    dogleg = np.arccos(
        np.cos(inc1_rad) * np.cos(inc2_rad) +
        np.sin(inc1_rad) * np.sin(inc2_rad) * np.cos(azi2_rad - azi1_rad)
    )

    # Convert to degrees per 100 ft
    dls = np.degrees(dogleg) * (100 / md_interval)

    return dls


def generate_stratigraphic_model(field_id):
    """Generate stratigraphic layers and structure for geosteering"""
    np.random.seed(hash(field_id) % 2 ** 32)

    # Define stratigraphic layers
    layers = {
        'Overburden': {'top': 0, 'bottom': 2500, 'type': 'shale'},
        'Cap Rock': {'top': 2500, 'bottom': 2700, 'type': 'shale'},
        'Upper Sand': {'top': 2700, 'bottom': 2850, 'type': 'sand', 'quality': 'moderate'},
        'Middle Shale': {'top': 2850, 'bottom': 2950, 'type': 'shale'},
        'Target Zone': {'top': 2950, 'bottom': 3100, 'type': 'sand', 'quality': 'excellent'},
        'Lower Sand': {'top': 3100, 'bottom': 3200, 'type': 'sand', 'quality': 'good'},
        'Base Shale': {'top': 3200, 'bottom': 3500, 'type': 'shale'}
    }

    # Add structural variation (anticline/syncline)
    x = np.linspace(0, 10000, 100)
    y = np.linspace(0, 10000, 100)
    X, Y = np.meshgrid(x, y)

    # Create structural surface with dip and azimuth
    structure_map = 2950 + 100 * np.sin(X / 2000) * np.cos(Y / 2000) + \
                    50 * np.sin(X / 1000) + np.random.normal(0, 10, X.shape)

    return layers, X, Y, structure_map


def generate_wellbore_trajectory(well_id, target_depth=3000, lateral_length=5000):
    """Generate realistic wellbore trajectory for horizontal/directional well"""
    np.random.seed(hash(well_id) % 2 ** 32)

    # Vertical section
    vertical_depth = 2500
    kickoff_depth = 2200

    # Build section (curve)
    build_rate = 3  # degrees per 100 ft
    max_inclination = 90  # horizontal

    # Landing point
    landing_depth = target_depth

    # Generate trajectory points
    trajectory_points = []

    # Vertical section
    for d in np.linspace(0, kickoff_depth, 50):
        trajectory_points.append({
            'md': d,
            'tvd': d,
            'inclination': 0,
            'azimuth': 0,
            'north': 0,
            'east': 0,
            'dls': 0
        })

    # Build section
    current_inc = 0
    current_azi = np.random.uniform(0, 360)
    current_north = 0
    current_east = 0
    current_tvd = kickoff_depth
    current_md = kickoff_depth

    while current_inc < max_inclination and current_tvd < landing_depth:
        prev_inc = current_inc
        prev_azi = current_azi
        prev_md = current_md

        current_inc = min(current_inc + build_rate, max_inclination)
        step_length = 30  # ft
        current_md += step_length

        # Calculate position change
        delta_tvd = step_length * np.cos(np.radians(current_inc))
        delta_horizontal = step_length * np.sin(np.radians(current_inc))

        current_tvd += delta_tvd
        current_north += delta_horizontal * np.cos(np.radians(current_azi))
        current_east += delta_horizontal * np.sin(np.radians(current_azi))

        # Calculate DLS
        dls = calculate_dogleg_severity(prev_inc, current_inc, prev_azi, current_azi, step_length)

        trajectory_points.append({
            'md': current_md,
            'tvd': current_tvd,
            'inclination': current_inc,
            'azimuth': current_azi,
            'north': current_north,
            'east': current_east,
            'dls': dls
        })

    # Lateral section
    for i in range(int(lateral_length / 30)):
        prev_inc = current_inc
        prev_azi = current_azi

        current_md += 30

        # Add some tortuosity
        current_inc = 90 + np.random.normal(0, 2)
        current_azi += np.random.normal(0, 1)

        # Stay in target zone with small variations
        current_tvd += np.random.normal(0, 0.5)
        current_north += 30 * np.cos(np.radians(current_azi))
        current_east += 30 * np.sin(np.radians(current_azi))

        # Calculate DLS
        dls = calculate_dogleg_severity(prev_inc, current_inc, prev_azi, current_azi, 30)

        trajectory_points.append({
            'md': current_md,
            'tvd': current_tvd,
            'inclination': current_inc,
            'azimuth': current_azi % 360,
            'north': current_north,
            'east': current_east,
            'dls': dls
        })

    return pd.DataFrame(trajectory_points)


def generate_geosteering_data(trajectory_df, layers, structure_map):
    """Generate real-time geosteering measurements along trajectory"""
    geosteering_data = []

    # Generate time series for drilling data
    start_time = datetime.now() - timedelta(hours=len(trajectory_df) * 0.1)

    for idx, point in trajectory_df.iterrows():
        # Calculate drilling time
        drilling_time = start_time + timedelta(hours=idx * 0.1)

        # Determine which layer we're in
        current_layer = None
        for layer_name, layer_props in layers.items():
            if layer_props['top'] <= point['tvd'] <= layer_props['bottom']:
                current_layer = layer_name
                break

        # Generate formation properties based on layer
        if current_layer == 'Target Zone':
            gamma_ray = np.random.normal(35, 5)
            resistivity = np.random.lognormal(3.5, 0.3)
            porosity = np.random.normal(0.22, 0.02)
            oil_sat = np.random.normal(0.75, 0.05)
            base_rop = np.random.uniform(80, 150)
        elif 'Sand' in current_layer:
            gamma_ray = np.random.normal(45, 8)
            resistivity = np.random.lognormal(2.8, 0.3)
            porosity = np.random.normal(0.18, 0.03)
            oil_sat = np.random.normal(0.60, 0.08)
            base_rop = np.random.uniform(60, 120)
        else:  # Shale
            gamma_ray = np.random.normal(85, 10)
            resistivity = np.random.lognormal(1.5, 0.2)
            porosity = np.random.normal(0.08, 0.02)
            oil_sat = np.random.normal(0.15, 0.05)
            base_rop = np.random.uniform(30, 80)

        # Distance to boundaries (for geosteering decisions)
        if current_layer == 'Target Zone':
            dist_to_top = point['tvd'] - layers['Target Zone']['top']
            dist_to_bottom = layers['Target Zone']['bottom'] - point['tvd']
        else:
            dist_to_top = 999
            dist_to_bottom = 999

        # Generate drilling parameters with some correlation to formation and DLS
        dls_factor = 1 - min(point['dls'] / 10, 0.5)
        rop = base_rop * dls_factor + np.random.normal(0, 10)

        # WOB and RPM affected by formation and trajectory
        if point['inclination'] > 85:  # Horizontal section
            wob = np.random.uniform(20, 35)
            rpm = np.random.uniform(100, 140)
        else:  # Build section
            wob = np.random.uniform(15, 25)
            rpm = np.random.uniform(80, 120)

        geosteering_data.append({
            'md': point['md'],
            'tvd': point['tvd'],
            'inclination': point['inclination'],
            'azimuth': point['azimuth'],
            'north': point['north'],
            'east': point['east'],
            'dls': point['dls'],
            'time': drilling_time,
            'current_layer': current_layer,
            'gamma_ray': gamma_ray,
            'resistivity': resistivity,
            'porosity': porosity,
            'oil_saturation': oil_sat,
            'dist_to_top': dist_to_top,
            'dist_to_bottom': dist_to_bottom,
            'in_target': current_layer == 'Target Zone',
            'rop': rop,
            'wob': wob,
            'rpm': rpm,
            'torque': wob * rpm / 100 + np.random.normal(0, 2)
        })

    return pd.DataFrame(geosteering_data)


def calculate_geosteering_kpis(geosteering_df):
    """Calculate key performance indicators for geosteering"""
    kpis = {
        'total_md': geosteering_df['md'].max(),
        'lateral_length': geosteering_df[geosteering_df['inclination'] > 85]['md'].max() -
                          geosteering_df[geosteering_df['inclination'] > 85]['md'].min() if any(
            geosteering_df['inclination'] > 85) else 0,
        'net_pay': len(geosteering_df[geosteering_df['in_target']]) * 30,
        'target_zone_percentage': (geosteering_df['in_target'].sum() / len(geosteering_df)) * 100,
        'avg_porosity_in_target': geosteering_df[geosteering_df['in_target']]['porosity'].mean() * 100,
        'avg_oil_sat_in_target': geosteering_df[geosteering_df['in_target']]['oil_saturation'].mean() * 100,
        'tortuosity_index': geosteering_df['inclination'].std(),
        'avg_dls': geosteering_df['dls'].mean(),
        'max_dls': geosteering_df['dls'].max(),
        'drilling_efficiency': geosteering_df['rop'].mean()
    }

    return kpis


def generate_look_ahead_prediction(current_point, layers, distance_ahead=500):
    """Generate look-ahead predictions for geosteering decisions"""
    predictions = []

    for dist in np.linspace(0, distance_ahead, 10):
        # Project ahead based on current trajectory
        projected_tvd = current_point['tvd'] + dist * np.sin(np.radians(90 - current_point['inclination']))
        projected_north = current_point['north'] + dist * np.cos(np.radians(current_point['azimuth']))
        projected_east = current_point['east'] + dist * np.sin(np.radians(current_point['azimuth']))

        # Predict formation
        predicted_layer = None
        for layer_name, layer_props in layers.items():
            if layer_props['top'] <= projected_tvd <= layer_props['bottom']:
                predicted_layer = layer_name
                break

        predictions.append({
            'distance_ahead': dist,
            'projected_tvd': projected_tvd,
            'projected_north': projected_north,
            'projected_east': projected_east,
            'predicted_layer': predicted_layer,
            'in_target': predicted_layer == 'Target Zone'
        })

    return pd.DataFrame(predictions)


def optimize_trajectory_adjustment(current_point, target_tvd, max_dls=3):
    """Calculate optimal trajectory adjustment to stay in target zone"""
    current_tvd = current_point['tvd']
    current_inc = current_point['inclination']

    # Calculate required inclination change
    tvd_error = target_tvd - current_tvd

    if abs(tvd_error) < 5:  # Within tolerance
        recommended_inc = current_inc
        adjustment = "MAINTAIN"
    elif tvd_error > 0:  # Too shallow, need to drop
        recommended_inc = min(current_inc + max_dls, 92)
        adjustment = "DROP"
    else:  # Too deep, need to build
        recommended_inc = max(current_inc - max_dls, 88)
        adjustment = "BUILD"

    return {
        'current_inclination': current_inc,
        'recommended_inclination': recommended_inc,
        'adjustment': adjustment,
        'tvd_error': tvd_error,
        'urgency': 'HIGH' if abs(tvd_error) > 20 else 'MEDIUM' if abs(tvd_error) > 10 else 'LOW'
    }


# ============================================================================
# GENERATE DATA FOR MULTIPLE WELLS/FIELDS
# ============================================================================

wells_data = {}
geosteering_data = {}
stratigraphic_models = {}
trajectory_data = {}

for well_id in ['WELL-H001', 'WELL-H002', 'WELL-H003', 'WELL-H004']:
    # Generate stratigraphic model
    layers, X, Y, structure_map = generate_stratigraphic_model(well_id)
    stratigraphic_models[well_id] = {'layers': layers, 'X': X, 'Y': Y, 'structure': structure_map}

    # Generate wellbore trajectory
    trajectory = generate_wellbore_trajectory(well_id)
    trajectory_data[well_id] = trajectory

    # Generate geosteering data
    geo_data = generate_geosteering_data(trajectory, layers, structure_map)
    geosteering_data[well_id] = geo_data

# ============================================================================
# DASH LAYOUT - UPDATED WITH DRILLING PARAMETERS AND DLS
# ============================================================================

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("AUTOMATA INTELLIGENCE",
                        style={'color': colors['text'], 'fontWeight': 'bold'}),
                html.H1("🧭 Reservoir Navigation with Geosteering & Optimization",
                        style={'color': colors['text'], 'fontWeight': 'bold'}),
                html.P("Real-Time Wellbore Placement • Formation Evaluation • Trajectory Optimization",
                       style={'color': colors['text-secondary'], 'fontSize': '18px'})
            ], style={'textAlign': 'center', 'padding': '30px 0'})
        ])
    ]),

    # Control Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Geosteering Control Center", style={'color': colors['primary']}),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Active Well", style={'color': colors['text']}),
                            dcc.Dropdown(
                                id='well-selector',
                                options=[{'label': well, 'value': well} for well in trajectory_data.keys()],
                                value='WELL-H001',
                                style={'color': '#000'}
                            )
                        ], md=2),
                        dbc.Col([
                            html.Label("Current MD (ft)", style={'color': colors['text']}),
                            dcc.Slider(
                                id='md-slider',
                                min=0,
                                max=8000,
                                value=5000,
                                marks={i: f'{i}' for i in range(0, 8001, 1000)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], md=4),
                        dbc.Col([
                            html.Label("View Mode", style={'color': colors['text']}),
                            dcc.RadioItems(
                                id='view-mode',
                                options=[
                                    {'label': ' 3D', 'value': '3d'},
                                    {'label': ' Section', 'value': 'section'},
                                    {'label': ' Plan', 'value': 'plan'}
                                ],
                                value='3d',
                                inline=True,
                                style={'color': colors['text']}
                            )
                        ], md=2),
                        dbc.Col([
                            html.Label("Steering Mode", style={'color': colors['text']}),
                            dcc.RadioItems(
                                id='steering-mode',
                                options=[
                                    {'label': ' Auto', 'value': 'auto'},
                                    {'label': ' Manual', 'value': 'manual'}
                                ],
                                value='auto',
                                inline=True,
                                style={'color': colors['text']}
                            )
                        ], md=2),
                        dbc.Col([
                            dbc.Button("Update Trajectory", id='update-btn', color='primary',
                                       className='w-100', style={'marginTop': '25px'}, n_clicks=0)
                        ], md=2)
                    ])
                ])
            ], style={'backgroundColor': colors['surface'], 'marginBottom': '20px'})
        ])
    ]),

    # Alert Section
    dbc.Row([
        dbc.Col([
            html.Div(id='steering-alerts')
        ])
    ], className='mb-3'),

    # Key Geosteering Metrics - Including DLS
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='current-tvd', children='0 ft', style={'color': colors['primary']}),
                    html.P('Current TVD', style={'color': colors['text-secondary']}),
                    html.Small(id='current-formation', style={'fontSize': '12px', 'color': colors['text-secondary']})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='inclination-azimuth', children='0°/0°', style={'color': colors['warning']}),
                    html.P('Inc/Azi', style={'color': colors['text-secondary']}),
                    html.Small(id='dls-current', style={'fontSize': '12px', 'color': colors['text-secondary']})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='target-percentage', children='0%', style={'color': colors['success']}),
                    html.P('In Target Zone', style={'color': colors['text-secondary']}),
                    html.Small(id='net-pay', style={'fontSize': '12px', 'color': colors['text-secondary']})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='distance-to-boundary', children='0 ft', style={'color': colors['secondary']}),
                    html.P('To Boundary', style={'color': colors['text-secondary']}),
                    html.Small(id='boundary-type', style={'fontSize': '12px', 'color': colors['text-secondary']})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='dogleg-severity', children='0°/100ft', style={'color': colors['danger']}),
                    html.P('Dogleg Severity', style={'color': colors['text-secondary']}),
                    html.Small(id='dls-status', style={'fontSize': '12px'})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='current-rop', children='0 ft/hr', style={'color': colors['primary']}),
                    html.P('ROP', style={'color': colors['text-secondary']}),
                    html.Small(id='drilling-time', style={'fontSize': '12px', 'color': colors['text-secondary']})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2)
    ], className='mb-4'),

    # Main Visualization Area
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Tabs(id='main-tabs', value='trajectory', children=[
                        dcc.Tab(label='Trajectory Visualization', value='trajectory',
                                style={'backgroundColor': colors['surface'], 'color': colors['text']},
                                selected_style={'backgroundColor': colors['primary'], 'color': colors['background']}),
                        dcc.Tab(label='Formation Evaluation', value='formation',
                                style={'backgroundColor': colors['surface'], 'color': colors['text']},
                                selected_style={'backgroundColor': colors['primary'], 'color': colors['background']}),
                        dcc.Tab(label='Drilling Parameters', value='drilling',
                                style={'backgroundColor': colors['surface'], 'color': colors['text']},
                                selected_style={'backgroundColor': colors['primary'], 'color': colors['background']}),
                        dcc.Tab(label='Look-Ahead Prediction', value='lookahead',
                                style={'backgroundColor': colors['surface'], 'color': colors['text']},
                                selected_style={'backgroundColor': colors['primary'], 'color': colors['background']}),
                        dcc.Tab(label='Optimization', value='optimization',
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
                    html.H5("Stratigraphic Column", style={'color': colors['primary'], 'marginBottom': '20px'}),
                    dcc.Graph(id='strat-column', style={'height': '200px'}),
                    html.Hr(),
                    html.H5("DLS Analysis", style={'color': colors['primary'], 'marginTop': '20px'}),
                    dcc.Graph(id='dls-gauge', style={'height': '250px'}),  # INCREASED HEIGHT
                    html.Hr(),
                    html.H5("Steering Compass", style={'color': colors['primary'], 'marginTop': '20px'}),
                    dcc.Graph(id='steering-compass', style={'height': '200px'})
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=4)
    ], className='mb-4'),

    # Bottom Section - Recommendations only (removed Performance Metrics)
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Geosteering Recommendations", style={'color': colors['primary']}),
                    html.Div(id='steering-recommendations')
                ])
            ], style={'backgroundColor': colors['surface']})
        ])
    ]),

    # Store components
    dcc.Store(id='processed-data-store'),
    dcc.Interval(id='interval-component', interval=5000, n_intervals=0, disabled=True)

], fluid=True, style={'backgroundColor': colors['background'], 'minHeight': '100vh', 'padding': '20px'})


# ============================================================================
# CALLBACKS - UPDATED
# ============================================================================

# [Keep all the process_geosteering_data callback the same]

@app.callback(
    [Output('processed-data-store', 'data'),
     Output('current-tvd', 'children'),
     Output('current-formation', 'children'),
     Output('inclination-azimuth', 'children'),
     Output('dls-current', 'children'),
     Output('target-percentage', 'children'),
     Output('net-pay', 'children'),
     Output('distance-to-boundary', 'children'),
     Output('boundary-type', 'children'),
     Output('dogleg-severity', 'children'),
     Output('dls-status', 'children'),
     Output('current-rop', 'children'),
     Output('drilling-time', 'children'),
     Output('steering-alerts', 'children')],
    [Input('update-btn', 'n_clicks')],
    [State('well-selector', 'value'),
     State('md-slider', 'value'),
     State('view-mode', 'value'),
     State('steering-mode', 'value')]
)
def process_geosteering_data(n_clicks, well_id, current_md, view_mode, steering_mode):
    if n_clicks == 0 and current_md == 0:
        return (None, '0 ft', '', '0°/0°', '', '0%', '', '0 ft', '', '0°/100ft', '', '0 ft/hr', '', None)

    # Get data for selected well
    geo_df = geosteering_data[well_id]
    traj_df = trajectory_data[well_id]
    strat_model = stratigraphic_models[well_id]

    # Find current position
    current_idx = np.argmin(np.abs(geo_df['md'] - current_md))
    current_point = geo_df.iloc[current_idx]

    # Calculate KPIs
    kpis = calculate_geosteering_kpis(geo_df.iloc[:current_idx + 1])

    # Get look-ahead predictions
    predictions = generate_look_ahead_prediction(current_point, strat_model['layers'])

    # Get trajectory optimization recommendations
    target_tvd = (strat_model['layers']['Target Zone']['top'] +
                  strat_model['layers']['Target Zone']['bottom']) / 2
    optimization = optimize_trajectory_adjustment(current_point, target_tvd)

    # Generate alerts based on DLS and position
    alerts = []

    # DLS alert
    if current_point['dls'] > 6:
        alerts.append(
            dbc.Alert([
                html.I(className="fas fa-exclamation-circle me-2"),
                f"HIGH DOGLEG SEVERITY: {current_point['dls']:.1f}°/100ft exceeds recommended limit of 6°/100ft"
            ], color="danger", dismissable=True)
        )
    elif current_point['dls'] > 4:
        alerts.append(
            dbc.Alert([
                html.I(className="fas fa-warning me-2"),
                f"MODERATE DOGLEG SEVERITY: {current_point['dls']:.1f}°/100ft - Monitor torque and drag"
            ], color="warning", dismissable=True)
        )

    # Formation alert
    if not current_point['in_target']:
        alerts.append(
            dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                f"OUT OF TARGET ZONE: Currently in {current_point['current_layer']}. Recommend {optimization['adjustment']} to {optimization['recommended_inclination']:.1f}°"
            ], color="danger", dismissable=True)
        )

    if current_point['dist_to_top'] < 10 or current_point['dist_to_bottom'] < 10:
        alerts.append(
            dbc.Alert([
                html.I(className="fas fa-warning me-2"),
                f"APPROACHING BOUNDARY: {min(current_point['dist_to_top'], current_point['dist_to_bottom']):.1f} ft to nearest boundary"
            ], color="warning", dismissable=True)
        )

    # Prepare processed data
    processed_data = {
        'well_id': well_id,
        'current_point': current_point.to_dict(),
        'trajectory': traj_df.to_dict('records'),
        'geosteering': geo_df.iloc[:current_idx + 1].to_dict('records'),
        'strat_model': strat_model,
        'kpis': kpis,
        'predictions': predictions.to_dict('records'),
        'optimization': optimization,
        'current_md': current_md
    }

    # Format outputs
    tvd_text = f"{current_point['tvd']:.0f} ft"
    formation_text = current_point['current_layer']
    inc_azi_text = f"{current_point['inclination']:.1f}°/{current_point['azimuth']:.1f}°"
    dls_current_text = f"Current: {current_point['dls']:.2f}°/100ft"
    target_pct_text = f"{kpis['target_zone_percentage']:.1f}%"
    net_pay_text = f"Net Pay: {kpis['net_pay']:.0f} ft"

    if current_point['in_target']:
        dist_boundary = min(current_point['dist_to_top'], current_point['dist_to_bottom'])
        boundary_text = "Top" if current_point['dist_to_top'] < current_point['dist_to_bottom'] else "Bottom"
    else:
        dist_boundary = abs(current_point['tvd'] - target_tvd)
        boundary_text = "To Target"

    dist_boundary_text = f"{dist_boundary:.1f} ft"

    # DLS status
    dls_text = f"{current_point['dls']:.2f}°/100ft"
    if current_point['dls'] > 6:
        dls_status = html.Span("⚠️ Critical", style={'color': colors['danger']})
    elif current_point['dls'] > 4:
        dls_status = html.Span("⚠️ Moderate", style={'color': colors['warning']})
    else:
        dls_status = html.Span("✓ Normal", style={'color': colors['success']})

    rop_text = f"{current_point['rop']:.0f} ft/hr"

    # Calculate drilling time
    if 'time' in current_point:
        time_elapsed = (datetime.now() - current_point['time']).total_seconds() / 3600
        drilling_time_text = f"Time: {time_elapsed:.1f} hrs"
    else:
        drilling_time_text = "Time: N/A"

    return (
        processed_data,
        tvd_text,
        formation_text,
        inc_azi_text,
        dls_current_text,
        target_pct_text,
        net_pay_text,
        dist_boundary_text,
        boundary_text,
        dls_text,
        dls_status,
        rop_text,
        drilling_time_text,
        html.Div(alerts) if alerts else None
    )


@app.callback(
    Output('tab-content', 'children'),
    [Input('main-tabs', 'value'),
     Input('processed-data-store', 'data')]
)
def update_tab_content(active_tab, processed_data):
    if not processed_data:
        return html.Div("Select a well and click 'Update Trajectory' to start.",
                        style={'color': colors['text-secondary'], 'textAlign': 'center', 'padding': '50px'})

    if active_tab == 'trajectory':
        # Create 3D trajectory visualization with CONSISTENT COLORS
        traj = pd.DataFrame(processed_data['trajectory'])
        geo = pd.DataFrame(processed_data['geosteering'])
        current_idx = np.argmin(np.abs(geo['md'] - processed_data['current_md']))

        fig = go.Figure()

        # Add trajectory colored by formation using CONSISTENT COLORS
        for formation in geo['current_layer'].unique():
            formation_data = geo[geo['current_layer'] == formation]
            fig.add_trace(go.Scatter3d(
                x=formation_data['east'],
                y=formation_data['north'],
                z=-formation_data['tvd'],
                mode='markers+lines',
                marker=dict(
                    size=4,
                    color=formation_colors.get(formation, formation_colors['Unknown'])  # Use consistent colors
                ),
                line=dict(
                    color=formation_colors.get(formation, formation_colors['Unknown']),  # Use consistent colors
                    width=4
                ),
                name=formation
            ))

        # Add current position marker
        current = geo.iloc[current_idx]
        fig.add_trace(go.Scatter3d(
            x=[current['east']],
            y=[current['north']],
            z=[-current['tvd']],
            mode='markers',
            marker=dict(
                size=10,
                color=colors['danger'],
                symbol='diamond'
            ),
            name='Current Position'
        ))

        # Add target zone boundaries
        layers = processed_data['strat_model']['layers']
        target_top = -layers['Target Zone']['top']
        target_bottom = -layers['Target Zone']['bottom']

        x_range = [geo['east'].min() - 500, geo['east'].max() + 500]
        y_range = [geo['north'].min() - 500, geo['north'].max() + 500]

        xx, yy = np.meshgrid(x_range, y_range)

        fig.add_trace(go.Surface(
            x=xx,
            y=yy,
            z=np.full_like(xx, target_top),
            opacity=0.3,
            colorscale=[[0, formation_colors['Target Zone']], [1, formation_colors['Target Zone']]],
            showscale=False,
            name='Target Top'
        ))

        fig.add_trace(go.Surface(
            x=xx,
            y=yy,
            z=np.full_like(xx, target_bottom),
            opacity=0.3,
            colorscale=[[0, colors['danger']], [1, colors['danger']]],
            showscale=False,
            name='Target Bottom'
        ))

        fig.update_layout(
            scene=dict(
                xaxis_title='East (ft)',
                yaxis_title='North (ft)',
                zaxis_title='Depth (ft)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=600,
            title="3D Wellbore Trajectory with Stratigraphic Layers",
            plot_bgcolor=colors['surface'],
            paper_bgcolor=colors['surface'],
            font=dict(color=colors['text'])
        )

        return dcc.Graph(figure=fig)

    elif active_tab == 'drilling':
        # Drilling parameters vs time
        geo = pd.DataFrame(processed_data['geosteering'])
        current_idx = np.argmin(np.abs(geo['md'] - processed_data['current_md']))
        window_data = geo.iloc[:current_idx + 1]

        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            subplot_titles=('Rate of Penetration (ROP)', 'Weight on Bit (WOB)',
                            'Rotations Per Minute (RPM)', 'Dogleg Severity'),
            vertical_spacing=0.05
        )

        # ROP vs MD
        fig.add_trace(
            go.Scatter(x=window_data['md'], y=window_data['rop'],
                       mode='lines', line=dict(color=colors['success'], width=2),
                       fill='tozeroy', fillcolor='rgba(0, 255, 136, 0.1)',
                       name='ROP'),
            row=1, col=1
        )

        # Add average line
        fig.add_hline(y=window_data['rop'].mean(), line_dash="dash",
                      line_color=colors['text-secondary'], line_width=1,
                      annotation_text=f"Avg: {window_data['rop'].mean():.0f} ft/hr",
                      row=1, col=1)

        # WOB vs MD
        fig.add_trace(
            go.Scatter(x=window_data['md'], y=window_data['wob'],
                       mode='lines', line=dict(color=colors['warning'], width=2),
                       fill='tozeroy', fillcolor='rgba(255, 170, 0, 0.1)',
                       name='WOB'),
            row=2, col=1
        )

        # RPM vs MD
        fig.add_trace(
            go.Scatter(x=window_data['md'], y=window_data['rpm'],
                       mode='lines', line=dict(color=colors['primary'], width=2),
                       fill='tozeroy', fillcolor='rgba(0, 212, 255, 0.1)',
                       name='RPM'),
            row=3, col=1
        )

        # DLS vs MD with critical zones
        fig.add_trace(
            go.Scatter(x=window_data['md'], y=window_data['dls'],
                       mode='lines', line=dict(color=colors['secondary'], width=2),
                       name='DLS'),
            row=4, col=1
        )

        # Add DLS limit lines
        fig.add_hline(y=6, line_dash="dash", line_color=colors['danger'],
                      line_width=2, annotation_text="Critical (6°/100ft)",
                      row=4, col=1)
        fig.add_hline(y=4, line_dash="dash", line_color=colors['warning'],
                      line_width=1, annotation_text="Warning (4°/100ft)",
                      row=4, col=1)

        # Highlight zones
        for _, row in window_data.iterrows():
            if row['in_target']:
                for i in range(1, 5):
                    fig.add_vrect(
                        x0=row['md'] - 15, x1=row['md'] + 15,
                        fillcolor="green",
                        opacity=0.05,
                        layer="below",
                        line_width=0,
                        row=i, col=1
                    )

        # Add current position line
        fig.add_vline(x=processed_data['current_md'], line_dash="dash",
                      line_color=colors['danger'], line_width=2)

        fig.update_xaxes(title='Measured Depth (ft)', row=4, col=1)
        fig.update_yaxes(title='ft/hr', row=1, col=1)
        fig.update_yaxes(title='klbs', row=2, col=1)
        fig.update_yaxes(title='RPM', row=3, col=1)
        fig.update_yaxes(title='°/100ft', row=4, col=1)

        fig.update_layout(
            height=600,
            showlegend=False,
            title="Drilling Parameters Analysis",
            plot_bgcolor=colors['surface'],
            paper_bgcolor=colors['surface'],
            font=dict(color=colors['text'])
        )

        return dcc.Graph(figure=fig)

    elif active_tab == 'formation':
        # Formation evaluation display
        geo = pd.DataFrame(processed_data['geosteering'])
        current_idx = np.argmin(np.abs(geo['md'] - processed_data['current_md']))

        window_start = max(0, current_idx - 100)
        window_data = geo.iloc[window_start:current_idx + 1]

        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            subplot_titles=('Gamma Ray', 'Resistivity', 'Porosity', 'Oil Saturation'),
            vertical_spacing=0.05
        )

        # Gamma Ray
        fig.add_trace(
            go.Scatter(x=window_data['md'], y=window_data['gamma_ray'],
                       mode='lines', line=dict(color=colors['success'], width=2),
                       fill='tozeroy', fillcolor='rgba(0, 255, 136, 0.1)'),
            row=1, col=1
        )

        # Resistivity
        fig.add_trace(
            go.Scatter(x=window_data['md'], y=window_data['resistivity'],
                       mode='lines', line=dict(color=colors['warning'], width=2),
                       fill='tozeroy', fillcolor='rgba(255, 170, 0, 0.1)'),
            row=2, col=1
        )

        # Porosity
        fig.add_trace(
            go.Scatter(x=window_data['md'], y=window_data['porosity'] * 100,
                       mode='lines', line=dict(color=colors['primary'], width=2),
                       fill='tozeroy', fillcolor='rgba(0, 212, 255, 0.1)'),
            row=3, col=1
        )

        # Oil Saturation
        fig.add_trace(
            go.Scatter(x=window_data['md'], y=window_data['oil_saturation'] * 100,
                       mode='lines', line=dict(color=colors['secondary'], width=2),
                       fill='tozeroy', fillcolor='rgba(255, 107, 53, 0.1)'),
            row=4, col=1
        )

        # Add formation boundaries
        for _, row in window_data.iterrows():
            if row['current_layer'] == 'Target Zone':
                for i in range(1, 5):
                    fig.add_vrect(
                        x0=row['md'] - 15, x1=row['md'] + 15,
                        fillcolor="green",
                        opacity=0.1,
                        layer="below",
                        line_width=0,
                        row=i, col=1
                    )

        fig.add_vline(x=processed_data['current_md'], line_dash="dash",
                      line_color=colors['danger'], line_width=2)

        fig.update_xaxes(title='Measured Depth (ft)', row=4, col=1)
        fig.update_yaxes(title='API', row=1, col=1)
        fig.update_yaxes(title='Ohm.m', row=2, col=1)
        fig.update_yaxes(title='%', row=3, col=1)
        fig.update_yaxes(title='%', row=4, col=1)

        fig.update_layout(
            height=600,
            showlegend=False,
            title="Formation Evaluation While Drilling",
            plot_bgcolor=colors['surface'],
            paper_bgcolor=colors['surface'],
            font=dict(color=colors['text'])
        )

        return dcc.Graph(figure=fig)

    elif active_tab == 'lookahead':
        # Look-ahead prediction visualization - FIXED: NO LEGEND
        predictions = pd.DataFrame(processed_data['predictions'])
        current_point = processed_data['current_point']

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Projected Path', 'Formation Prediction',
                            'Uncertainty Cone', 'Decision Matrix')
        )

        # Projected path
        fig.add_trace(
            go.Scatter(x=predictions['projected_east'], y=predictions['projected_north'],
                       mode='lines+markers', line=dict(color=colors['primary'], width=2),
                       marker=dict(size=8)),
            row=1, col=1
        )

        # Add current position
        fig.add_trace(
            go.Scatter(x=[current_point['east']], y=[current_point['north']],
                       mode='markers', marker=dict(size=12, color=colors['danger'], symbol='star')),
            row=1, col=1
        )

        # Formation prediction with consistent colors
        for formation in predictions['predicted_layer'].unique():
            if formation:
                form_data = predictions[predictions['predicted_layer'] == formation]
                fig.add_trace(
                    go.Bar(x=form_data['distance_ahead'],
                           y=[1] * len(form_data),
                           marker=dict(color=formation_colors.get(formation, formation_colors['Unknown']))),
                    row=1, col=2
                )

        # Uncertainty cone
        distances = predictions['distance_ahead']
        uncertainty = distances * 0.05

        fig.add_trace(
            go.Scatter(x=distances, y=predictions['projected_tvd'],
                       mode='lines', line=dict(color=colors['primary'], width=2)),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=distances, y=predictions['projected_tvd'] + uncertainty,
                       mode='lines', line=dict(width=0)),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=distances, y=predictions['projected_tvd'] - uncertainty,
                       fill='tonexty', fillcolor='rgba(0, 212, 255, 0.2)',
                       mode='lines', line=dict(width=0)),
            row=2, col=1
        )

        # Decision matrix
        decision_matrix = np.random.rand(3, 3) * 100
        fig.add_trace(
            go.Heatmap(
                z=decision_matrix,
                x=['Build', 'Hold', 'Drop'],
                y=['High Risk', 'Medium Risk', 'Low Risk'],
                colorscale=[[0, colors['danger']], [0.5, colors['warning']], [1, colors['success']]],
                text=np.round(decision_matrix, 0),
                texttemplate='%{text}%',
                textfont={"size": 12}
            ),
            row=2, col=2
        )

        fig.update_xaxes(title='East (ft)', row=1, col=1)
        fig.update_yaxes(title='North (ft)', row=1, col=1)
        fig.update_xaxes(title='Distance Ahead (ft)', row=1, col=2)
        fig.update_xaxes(title='Distance Ahead (ft)', row=2, col=1)
        fig.update_yaxes(title='TVD (ft)', row=2, col=1)

        fig.update_layout(
            height=600,
            title="Look-Ahead Predictions & Uncertainty Analysis",
            plot_bgcolor=colors['surface'],
            paper_bgcolor=colors['surface'],
            font=dict(color=colors['text']),
            showlegend=False  # FIXED: Remove legend
        )

        return dcc.Graph(figure=fig)

    elif active_tab == 'optimization':
        # Optimization dashboard with DLS focus
        geo = pd.DataFrame(processed_data['geosteering'])
        kpis = processed_data['kpis']

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Target Zone Performance', 'DLS Distribution',
                            'Trajectory Tortuosity vs DLS', 'Drilling Efficiency'),
            specs=[[{'type': 'scatter'}, {'type': 'histogram'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )

        # Target zone performance over distance
        target_pct = []
        for i in range(10, len(geo), 10):
            window = geo.iloc[:i]
            pct = (window['in_target'].sum() / len(window)) * 100
            target_pct.append({'md': window['md'].iloc[-1], 'pct': pct})

        if target_pct:
            target_df = pd.DataFrame(target_pct)
            fig.add_trace(
                go.Scatter(x=target_df['md'], y=target_df['pct'],
                           mode='lines+markers', line=dict(color=colors['success'], width=2)),
                row=1, col=1
            )

        # DLS Distribution
        fig.add_trace(
            go.Histogram(x=geo['dls'], nbinsx=30,
                         marker=dict(color=colors['warning']),
                         name='DLS Distribution'),
            row=1, col=2
        )

        # Add vertical lines for limits
        fig.add_vline(x=4, line_dash="dash", line_color=colors['warning'],
                      annotation_text="Warning", row=1, col=2)
        fig.add_vline(x=6, line_dash="dash", line_color=colors['danger'],
                      annotation_text="Critical", row=1, col=2)

        # Tortuosity vs DLS scatter
        fig.add_trace(
            go.Scatter(x=geo['inclination'], y=geo['dls'],
                       mode='markers',
                       marker=dict(
                           size=5,
                           color=geo['md'],
                           colorscale='Viridis',
                           showscale=True,
                           colorbar=dict(title="MD (ft)", x=0.45, len=0.4)
                       ),
                       text=[f"MD: {md:.0f}" for md in geo['md']],
                       hovertemplate="Inc: %{x:.1f}°<br>DLS: %{y:.2f}°/100ft<br>%{text}<extra></extra>"),
            row=2, col=1
        )

        # Drilling efficiency vs DLS
        fig.add_trace(
            go.Scatter(x=geo['dls'], y=geo['rop'],
                       mode='markers',
                       marker=dict(
                           size=8,
                           color=np.where(geo['in_target'], colors['success'], colors['danger']),
                           opacity=0.6
                       ),
                       text=['In Target' if t else 'Out of Target' for t in geo['in_target']],
                       hovertemplate="DLS: %{x:.2f}°/100ft<br>ROP: %{y:.1f} ft/hr<br>%{text}<extra></extra>"),
            row=2, col=2
        )

        # Add trend line for ROP vs DLS
        z = np.polyfit(geo['dls'], geo['rop'], 1)
        p = np.poly1d(z)
        fig.add_trace(
            go.Scatter(x=geo['dls'], y=p(geo['dls']),
                       mode='lines', line=dict(color=colors['text-secondary'], width=1, dash='dash'),
                       name='Trend'),
            row=2, col=2
        )

        fig.update_xaxes(title='Measured Depth (ft)', row=1, col=1)
        fig.update_yaxes(title='In Target (%)', row=1, col=1)
        fig.update_xaxes(title='DLS (°/100ft)', row=1, col=2)
        fig.update_yaxes(title='Frequency', row=1, col=2)
        fig.update_xaxes(title='Inclination (°)', row=2, col=1)
        fig.update_yaxes(title='DLS (°/100ft)', row=2, col=1)
        fig.update_xaxes(title='DLS (°/100ft)', row=2, col=2)
        fig.update_yaxes(title='ROP (ft/hr)', row=2, col=2)

        fig.update_layout(
            height=600,
            title="Geosteering Optimization with DLS Analysis",
            plot_bgcolor=colors['surface'],
            paper_bgcolor=colors['surface'],
            font=dict(color=colors['text']),
            showlegend=False
        )

        return dcc.Graph(figure=fig)


# Updated callbacks for side panels

@app.callback(
    Output('dls-gauge', 'figure'),
    [Input('processed-data-store', 'data')]
)
def update_dls_gauge(processed_data):
    """Create DLS gauge showing current, average, and maximum DLS"""
    if not processed_data:
        return go.Figure()

    current_dls = processed_data['current_point']['dls']
    kpis = processed_data['kpis']

    fig = go.Figure()

    # Main gauge for current DLS
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=current_dls,
        domain={'x': [0, 1], 'y': [0.25, 1]},  # Adjusted to make room for title
        title={'text': "Current DLS (°/100ft)", 'font': {'size': 14}},
        delta={'reference': kpis['avg_dls']},
        gauge={
            'axis': {'range': [None, 10], 'tickwidth': 1},
            'bar': {'color': colors['primary']},
            'steps': [
                {'range': [0, 4], 'color': colors['success']},
                {'range': [4, 6], 'color': colors['warning']},
                {'range': [6, 10], 'color': colors['danger']}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 2},
                'thickness': 0.75,
                'value': kpis['max_dls']
            }
        }
    ))

    # Add text annotations for avg and max
    fig.add_annotation(
        text=f"Avg: {kpis['avg_dls']:.2f}°",
        x=0.3, y=0.1,
        showarrow=False,
        font=dict(size=12, color=colors['text-secondary'])
    )

    fig.add_annotation(
        text=f"Max: {kpis['max_dls']:.2f}°",
        x=0.7, y=0.1,
        showarrow=False,
        font=dict(size=12, color=colors['danger'] if kpis['max_dls'] > 6 else colors['warning'])
    )

    fig.update_layout(
        height=250,  # INCREASED HEIGHT
        margin=dict(t=40, b=30, l=20, r=20),  # Added more top margin
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['surface'],
        font=dict(color=colors['text'])
    )

    return fig


@app.callback(
    Output('strat-column', 'figure'),
    [Input('processed-data-store', 'data')]
)
def update_strat_column(processed_data):
    if not processed_data:
        return go.Figure()

    layers = processed_data['strat_model']['layers']
    current_tvd = processed_data['current_point']['tvd']

    fig = go.Figure()

    # Use consistent formation colors
    for layer_name, props in layers.items():
        fig.add_trace(go.Bar(
            x=[1],
            y=[props['bottom'] - props['top']],
            base=[props['top']],
            marker=dict(
                color=formation_colors.get(layer_name, formation_colors['Unknown']),  # Use consistent colors
                line=dict(color='white', width=1)
            ),
            text=[layer_name],
            textposition='inside',
            name=layer_name,
            hovertemplate=f"{layer_name}<br>Top: {props['top']} ft<br>Bottom: {props['bottom']} ft<extra></extra>"
        ))

    fig.add_hline(y=current_tvd, line_color=colors['danger'], line_width=3,
                  annotation_text=f"Current: {current_tvd:.0f} ft")

    fig.update_layout(
        height=200,
        showlegend=False,
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(autorange='reversed', title='TVD (ft)'),
        margin=dict(t=10, b=40, l=60, r=10),
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['surface'],
        font=dict(color=colors['text'])
    )

    return fig


@app.callback(
    Output('steering-compass', 'figure'),
    [Input('processed-data-store', 'data')]
)
def update_steering_compass(processed_data):
    if not processed_data:
        return go.Figure()

    current = processed_data['current_point']
    optimization = processed_data['optimization']

    fig = go.Figure()

    # Create compass rose
    angles = np.linspace(0, 360, 36)
    r = np.ones_like(angles)

    fig.add_trace(go.Scatterpolar(
        r=r,
        theta=angles,
        mode='lines',
        line=dict(color=colors['text-secondary'], width=1),
        showlegend=False
    ))

    # Add current azimuth
    fig.add_trace(go.Scatterpolar(
        r=[0, 1],
        theta=[current['azimuth'], current['azimuth']],
        mode='lines+markers',
        line=dict(color=colors['primary'], width=3),
        marker=dict(size=10),
        name='Current'
    ))

    # Add recommended adjustment arrow
    if optimization['adjustment'] != 'MAINTAIN':
        recommended_azi = current['azimuth']
        fig.add_trace(go.Scatterpolar(
            r=[0.5, 0.9],
            theta=[recommended_azi, recommended_azi],
            mode='lines+markers',
            line=dict(color=colors['warning'], width=3, dash='dash'),
            marker=dict(size=8, symbol='triangle-up'),
            name='Recommended'
        ))

    # Add inclination indicator
    inc_normalized = current['inclination'] / 90
    fig.add_trace(go.Scatterpolar(
        r=[inc_normalized],
        theta=[current['azimuth']],
        mode='markers',
        marker=dict(size=20, color=colors['success']),
        name=f"Inc: {current['inclination']:.1f}°"
    ))

    # Add DLS indicator as text
    fig.add_annotation(
        text=f"DLS: {current['dls']:.1f}°/100ft",
        x=0.5, y=-0.1,
        showarrow=False,
        font=dict(size=10, color=colors['warning'] if current['dls'] > 4 else colors['text-secondary'])
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            angularaxis=dict(direction='clockwise', rotation=90)
        ),
        height=200,
        margin=dict(t=20, b=20, l=20, r=20),
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['surface'],
        font=dict(color=colors['text']),
        showlegend=True,
        legend=dict(x=0, y=1, font=dict(size=10))
    )

    return fig


@app.callback(
    Output('steering-recommendations', 'children'),
    [Input('processed-data-store', 'data')]
)
def update_steering_recommendations(processed_data):
    if not processed_data:
        return html.Div()

    optimization = processed_data['optimization']
    current = processed_data['current_point']
    kpis = processed_data['kpis']

    recommendations = []

    # DLS-based recommendations
    if current['dls'] > 6:
        recommendations.append({
            'Priority': 1,
            'Category': 'DLS Management',
            'Action': 'Reduce Dogleg Severity',
            'Current': f"{current['dls']:.2f}°/100ft",
            'Target': '<4°/100ft',
            'Urgency': 'HIGH',
            'Impact': 'Prevent tool failure and casing wear'
        })

    # Trajectory recommendations
    recommendations.append({
        'Priority': 2,
        'Category': 'Trajectory',
        'Action': f'{optimization["adjustment"]} Trajectory',
        'Current': f'{optimization["current_inclination"]:.1f}°',
        'Target': f'{optimization["recommended_inclination"]:.1f}°',
        'Urgency': optimization['urgency'],
        'Impact': 'Maintain target zone contact'
    })

    # Formation-based recommendations
    if not current['in_target']:
        recommendations.append({
            'Priority': 3,
            'Category': 'Formation',
            'Action': 'Return to Target',
            'Current': current['current_layer'],
            'Target': 'Target Zone',
            'Urgency': 'HIGH',
            'Impact': f'Improve net pay by {100 - kpis["target_zone_percentage"]:.0f}%'
        })

    # Drilling optimization
    if current['rop'] < 50:
        recommendations.append({
            'Priority': 4,
            'Category': 'Drilling',
            'Action': 'Optimize Parameters',
            'Current': f'{current["rop"]:.0f} ft/hr',
            'Target': '>80 ft/hr',
            'Urgency': 'LOW',
            'Impact': 'Reduce drilling time and cost'
        })

    # Boundary proximity
    if current['in_target'] and min(current['dist_to_top'], current['dist_to_bottom']) < 20:
        recommendations.append({
            'Priority': 5,
            'Category': 'Boundary',
            'Action': 'Adjust for Boundary',
            'Current': f'{min(current["dist_to_top"], current["dist_to_bottom"]):.1f} ft',
            'Target': '>30 ft clearance',
            'Urgency': 'MEDIUM',
            'Impact': 'Avoid exiting target zone'
        })

    return dash_table.DataTable(
        data=recommendations,
        columns=[{'name': col, 'id': col} for col in recommendations[0].keys()],
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
                'if': {'column_id': 'Urgency', 'filter_query': '{Urgency} = "HIGH"'},
                'backgroundColor': colors['danger'],
                'color': 'white',
            },
            {
                'if': {'column_id': 'Urgency', 'filter_query': '{Urgency} = "MEDIUM"'},
                'backgroundColor': colors['warning'],
                'color': 'white',
            },
            {
                'if': {'column_id': 'Category', 'filter_query': '{Category} = "DLS Management"'},
                'fontWeight': 'bold',
            }
        ],
        sort_action="native"
    )


# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == '__main__':
    app.run(debug=False, port=8053, host='127.0.0.1')