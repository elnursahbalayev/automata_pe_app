# reservoir_geosteering_app.py - Complete Reservoir Navigation with Geosteering & Optimization Dashboard
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
# GEOSTEERING DATA GENERATION FUNCTIONS
# ============================================================================

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
            'md': d,  # Measured depth
            'tvd': d,  # True vertical depth
            'inclination': 0,
            'azimuth': 0,
            'north': 0,
            'east': 0,
            'dls': 0  # Dogleg severity
        })

    # Build section
    current_inc = 0
    current_azi = np.random.uniform(0, 360)
    current_north = 0
    current_east = 0
    current_tvd = kickoff_depth
    current_md = kickoff_depth

    while current_inc < max_inclination and current_tvd < landing_depth:
        current_inc = min(current_inc + build_rate, max_inclination)
        step_length = 30  # ft
        current_md += step_length

        # Calculate position change
        delta_tvd = step_length * np.cos(np.radians(current_inc))
        delta_horizontal = step_length * np.sin(np.radians(current_inc))

        current_tvd += delta_tvd
        current_north += delta_horizontal * np.cos(np.radians(current_azi))
        current_east += delta_horizontal * np.sin(np.radians(current_azi))

        trajectory_points.append({
            'md': current_md,
            'tvd': current_tvd,
            'inclination': current_inc,
            'azimuth': current_azi,
            'north': current_north,
            'east': current_east,
            'dls': build_rate
        })

    # Lateral section
    for i in range(int(lateral_length / 30)):
        current_md += 30

        # Add some tortuosity
        current_inc = 90 + np.random.normal(0, 2)
        current_azi += np.random.normal(0, 1)

        # Stay in target zone with small variations
        current_tvd += np.random.normal(0, 0.5)
        current_north += 30 * np.cos(np.radians(current_azi))
        current_east += 30 * np.sin(np.radians(current_azi))

        trajectory_points.append({
            'md': current_md,
            'tvd': current_tvd,
            'inclination': current_inc,
            'azimuth': current_azi % 360,
            'north': current_north,
            'east': current_east,
            'dls': abs(np.random.normal(0, 0.5))
        })

    return pd.DataFrame(trajectory_points)


def generate_geosteering_data(trajectory_df, layers, structure_map):
    """Generate real-time geosteering measurements along trajectory"""
    geosteering_data = []

    for idx, point in trajectory_df.iterrows():
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
        elif 'Sand' in current_layer:
            gamma_ray = np.random.normal(45, 8)
            resistivity = np.random.lognormal(2.8, 0.3)
            porosity = np.random.normal(0.18, 0.03)
            oil_sat = np.random.normal(0.60, 0.08)
        else:  # Shale
            gamma_ray = np.random.normal(85, 10)
            resistivity = np.random.lognormal(1.5, 0.2)
            porosity = np.random.normal(0.08, 0.02)
            oil_sat = np.random.normal(0.15, 0.05)

        # Distance to boundaries (for geosteering decisions)
        if current_layer == 'Target Zone':
            dist_to_top = point['tvd'] - layers['Target Zone']['top']
            dist_to_bottom = layers['Target Zone']['bottom'] - point['tvd']
        else:
            dist_to_top = 999
            dist_to_bottom = 999

        geosteering_data.append({
            'md': point['md'],
            'tvd': point['tvd'],
            'inclination': point['inclination'],
            'azimuth': point['azimuth'],
            'north': point['north'],
            'east': point['east'],
            'current_layer': current_layer,
            'gamma_ray': gamma_ray,
            'resistivity': resistivity,
            'porosity': porosity,
            'oil_saturation': oil_sat,
            'dist_to_top': dist_to_top,
            'dist_to_bottom': dist_to_bottom,
            'in_target': current_layer == 'Target Zone',
            'rop': np.random.uniform(30, 150),  # Rate of penetration
            'wob': np.random.uniform(15, 35),  # Weight on bit
            'rpm': np.random.uniform(80, 140)  # Rotations per minute
        })

    return pd.DataFrame(geosteering_data)


def calculate_geosteering_kpis(geosteering_df):
    """Calculate key performance indicators for geosteering"""
    kpis = {
        'total_md': geosteering_df['md'].max(),
        'lateral_length': geosteering_df[geosteering_df['inclination'] > 85]['md'].max() -
                          geosteering_df[geosteering_df['inclination'] > 85]['md'].min() if any(
            geosteering_df['inclination'] > 85) else 0,
        'net_pay': len(geosteering_df[geosteering_df['in_target']]) * 30,  # Assuming 30 ft intervals
        'target_zone_percentage': (geosteering_df['in_target'].sum() / len(geosteering_df)) * 100,
        'avg_porosity_in_target': geosteering_df[geosteering_df['in_target']]['porosity'].mean() * 100,
        'avg_oil_sat_in_target': geosteering_df[geosteering_df['in_target']]['oil_saturation'].mean() * 100,
        'tortuosity_index': geosteering_df['inclination'].std(),
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

print("Generating geosteering data...")

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

    # Key Geosteering Metrics (REMOVED Res. Quality - now 5 metrics instead of 6)
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='current-tvd', children='0 ft', style={'color': colors['primary']}),
                    html.P('Current TVD', style={'color': colors['text-secondary']}),
                    html.Small(id='current-formation', style={'fontSize': '12px'})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='inclination-azimuth', children='0°/0°', style={'color': colors['warning']}),
                    html.P('Inc/Azi', style={'color': colors['text-secondary']}),
                    html.Small(id='dls-value', style={'fontSize': '12px'})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='target-percentage', children='0%', style={'color': colors['success']}),
                    html.P('In Target Zone', style={'color': colors['text-secondary']}),
                    html.Small(id='net-pay', style={'fontSize': '12px'})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='distance-to-boundary', children='0 ft', style={'color': colors['secondary']}),
                    html.P('To Boundary', style={'color': colors['text-secondary']}),
                    html.Small(id='boundary-type', style={'fontSize': '12px'})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='current-rop', children='0 ft/hr', style={'color': colors['primary']}),
                    html.P('ROP', style={'color': colors['text-secondary']}),
                    html.Small(id='drilling-params', style={'fontSize': '12px'})
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
                    html.H5("Steering Compass", style={'color': colors['primary'], 'marginTop': '20px'}),
                    dcc.Graph(id='steering-compass', style={'height': '200px'}),
                    html.Hr(),
                    html.H5("Real-Time Logs", style={'color': colors['primary'], 'marginTop': '20px'}),
                    dcc.Graph(id='realtime-logs', style={'height': '250px'})
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=4)
    ], className='mb-4'),

    # Decision Support Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Geosteering Recommendations", style={'color': colors['primary']}),
                    html.Div(id='steering-recommendations')
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Performance Metrics", style={'color': colors['primary']}),
                    dcc.Graph(id='performance-chart', style={'height': '300px'})
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=6)
    ]),

    # Store components
    dcc.Store(id='processed-data-store'),
    dcc.Interval(id='interval-component', interval=5000, n_intervals=0, disabled=True)

], fluid=True, style={'backgroundColor': colors['background'], 'minHeight': '100vh', 'padding': '20px'})


# ============================================================================
# CALLBACKS (REMOVED reservoir-quality output)
# ============================================================================

@app.callback(
    [Output('processed-data-store', 'data'),
     Output('current-tvd', 'children'),
     Output('current-formation', 'children'),
     Output('inclination-azimuth', 'children'),
     Output('dls-value', 'children'),
     Output('target-percentage', 'children'),
     Output('net-pay', 'children'),
     Output('distance-to-boundary', 'children'),
     Output('boundary-type', 'children'),
     Output('current-rop', 'children'),
     Output('drilling-params', 'children'),
     Output('steering-alerts', 'children')],
    [Input('update-btn', 'n_clicks')],
    [State('well-selector', 'value'),
     State('md-slider', 'value'),
     State('view-mode', 'value'),
     State('steering-mode', 'value')]
)
def process_geosteering_data(n_clicks, well_id, current_md, view_mode, steering_mode):
    if n_clicks == 0 and current_md == 0:
        return (None, '0 ft', '', '0°/0°', '', '0%', '', '0 ft', '', '0 ft/hr', '', None)

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

    # Generate alerts
    alerts = []
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
        'strat_model': {
            'layers': strat_model['layers']
        },
        'kpis': kpis,
        'predictions': predictions.to_dict('records'),
        'optimization': optimization,
        'current_md': current_md
    }

    # Format outputs
    tvd_text = f"{current_point['tvd']:.0f} ft"
    formation_text = current_point['current_layer']
    inc_azi_text = f"{current_point['inclination']:.1f}°/{current_point['azimuth']:.1f}°"
    dls_text = f"DLS: {geo_df.iloc[max(0, current_idx - 10):current_idx + 1]['inclination'].std():.2f}°/100ft"
    target_pct_text = f"{kpis['target_zone_percentage']:.1f}%"
    net_pay_text = f"Net Pay: {kpis['net_pay']:.0f} ft"

    if current_point['in_target']:
        dist_boundary = min(current_point['dist_to_top'], current_point['dist_to_bottom'])
        boundary_text = "Top" if current_point['dist_to_top'] < current_point['dist_to_bottom'] else "Bottom"
    else:
        dist_boundary = abs(current_point['tvd'] - target_tvd)
        boundary_text = "To Target"

    dist_boundary_text = f"{dist_boundary:.1f} ft"
    rop_text = f"{current_point['rop']:.0f} ft/hr"
    drilling_params_text = f"WOB: {current_point['wob']:.0f} klbs"

    return (
        processed_data,
        tvd_text,
        formation_text,
        inc_azi_text,
        dls_text,
        target_pct_text,
        net_pay_text,
        dist_boundary_text,
        boundary_text,
        rop_text,
        drilling_params_text,
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
        # Create 3D trajectory visualization
        traj = pd.DataFrame(processed_data['trajectory'])
        geo = pd.DataFrame(processed_data['geosteering'])
        current_idx = np.argmin(np.abs(geo['md'] - processed_data['current_md']))

        fig = go.Figure()

        # Add trajectory colored by formation
        formation_colors = {
            'Target Zone': colors['success'],
            'Upper Sand': colors['warning'],
            'Lower Sand': colors['warning'],
            'Cap Rock': colors['text-secondary'],
            'Middle Shale': colors['text-secondary'],
            'Base Shale': colors['text-secondary']
        }

        for formation in geo['current_layer'].unique():
            formation_data = geo[geo['current_layer'] == formation]
            fig.add_trace(go.Scatter3d(
                x=formation_data['east'],
                y=formation_data['north'],
                z=-formation_data['tvd'],  # Negative for depth
                mode='markers+lines',
                marker=dict(
                    size=4,
                    color=formation_colors.get(formation, colors['primary'])
                ),
                line=dict(
                    color=formation_colors.get(formation, colors['primary']),
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

        # Create a mesh for target zone
        x_range = [geo['east'].min() - 500, geo['east'].max() + 500]
        y_range = [geo['north'].min() - 500, geo['north'].max() + 500]

        xx, yy = np.meshgrid(x_range, y_range)

        # Add top and bottom surfaces
        fig.add_trace(go.Surface(
            x=xx,
            y=yy,
            z=np.full_like(xx, target_top),
            opacity=0.3,
            colorscale=[[0, 'green'], [1, 'green']],
            showscale=False,
            name='Target Top'
        ))

        fig.add_trace(go.Surface(
            x=xx,
            y=yy,
            z=np.full_like(xx, target_bottom),
            opacity=0.3,
            colorscale=[[0, 'red'], [1, 'red']],
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

    elif active_tab == 'formation':
        # Create formation evaluation display
        geo = pd.DataFrame(processed_data['geosteering'])
        current_idx = np.argmin(np.abs(geo['md'] - processed_data['current_md']))

        # Get recent data window
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

        # Add current position line
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
        # Create look-ahead prediction visualization
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
                       mode='markers', marker=dict(size=12, color=colors['danger'], symbol='star'),
                       name='Current'),
            row=1, col=1
        )

        # Formation prediction
        formation_colors_map = {'Target Zone': colors['success'],
                                'Upper Sand': colors['warning'],
                                'Lower Sand': colors['warning']}

        for formation in predictions['predicted_layer'].unique():
            if formation:
                form_data = predictions[predictions['predicted_layer'] == formation]
                fig.add_trace(
                    go.Bar(x=form_data['distance_ahead'],
                           y=[1] * len(form_data),
                           marker=dict(color=formation_colors_map.get(formation, colors['text-secondary'])),
                           name=formation),
                    row=1, col=2
                )

        # Uncertainty cone
        distances = predictions['distance_ahead']
        uncertainty = distances * 0.05  # 5% uncertainty

        fig.add_trace(
            go.Scatter(x=distances, y=predictions['projected_tvd'],
                       mode='lines', line=dict(color=colors['primary'], width=2),
                       name='Expected'),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=distances, y=predictions['projected_tvd'] + uncertainty,
                       mode='lines', line=dict(width=0),
                       showlegend=False),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=distances, y=predictions['projected_tvd'] - uncertainty,
                       fill='tonexty', fillcolor='rgba(0, 212, 255, 0.2)',
                       mode='lines', line=dict(width=0),
                       name='Uncertainty'),
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
            font=dict(color=colors['text'])
        )

        return dcc.Graph(figure=fig)

    elif active_tab == 'optimization':
        # Create optimization dashboard
        geo = pd.DataFrame(processed_data['geosteering'])
        kpis = processed_data['kpis']

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Target Zone Performance', 'Drilling Efficiency',
                            'Trajectory Tortuosity', 'Economic Impact'),
            specs=[[{'type': 'scatter'}, {'type': 'bar'}],
                   [{'type': 'scatter'}, {'type': 'indicator'}]]
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

        # Drilling efficiency metrics
        metrics = ['ROP', 'WOB', 'RPM']
        values = [kpis['drilling_efficiency'],
                  geo['wob'].mean(),
                  geo['rpm'].mean()]

        fig.add_trace(
            go.Bar(x=metrics, y=values,
                   marker=dict(color=[colors['primary'], colors['warning'], colors['secondary']]),
                   text=[f'{v:.1f}' for v in values],
                   textposition='outside'),
            row=1, col=2
        )

        # Tortuosity analysis
        fig.add_trace(
            go.Scatter(x=geo['md'], y=geo['inclination'],
                       mode='lines', line=dict(color=colors['warning'], width=1),
                       fill='tozeroy', fillcolor='rgba(255, 170, 0, 0.1)'),
            row=2, col=1
        )

        # Economic indicator
        economic_score = kpis['target_zone_percentage'] * 0.7 + \
                         (100 - kpis['tortuosity_index']) * 0.3

        fig.add_trace(
            go.Indicator(
                mode='gauge+number+delta',
                value=economic_score,
                title={'text': 'Economic Score'},
                delta={'reference': 75},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': colors['success']},
                    'steps': [
                        {'range': [0, 50], 'color': colors['danger']},
                        {'range': [50, 75], 'color': colors['warning']},
                        {'range': [75, 100], 'color': colors['success']}
                    ],
                    'threshold': {
                        'line': {'color': 'white', 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=2, col=2
        )

        fig.update_xaxes(title='Measured Depth (ft)', row=1, col=1)
        fig.update_yaxes(title='In Target (%)', row=1, col=1)
        fig.update_xaxes(title='Measured Depth (ft)', row=2, col=1)
        fig.update_yaxes(title='Inclination (°)', row=2, col=1)

        fig.update_layout(
            height=600,
            title="Geosteering Optimization Metrics",
            plot_bgcolor=colors['surface'],
            paper_bgcolor=colors['surface'],
            font=dict(color=colors['text'])
        )

        return dcc.Graph(figure=fig)


# Add callbacks for the side panel visualizations...

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

    # Add stratigraphic layers
    colors_map = {
        'shale': '#8B7355',
        'sand': '#FFD700'
    }

    for layer_name, props in layers.items():
        layer_type = props.get('type', 'shale')
        quality = props.get('quality', '')

        # Determine color based on quality
        if layer_name == 'Target Zone':
            color = colors['success']
        elif quality == 'good':
            color = colors['warning']
        else:
            color = colors_map.get(layer_type, colors['text-secondary'])

        fig.add_trace(go.Bar(
            x=[1],
            y=[props['bottom'] - props['top']],
            base=[props['top']],
            marker=dict(color=color, line=dict(color='white', width=1)),
            text=[layer_name],
            textposition='inside',
            name=layer_name,
            hovertemplate=f"{layer_name}<br>Top: {props['top']} ft<br>Bottom: {props['bottom']} ft<extra></extra>"
        ))

    # Add current position marker
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
        recommended_azi = current['azimuth']  # Could be adjusted based on optimization
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
    Output('realtime-logs', 'figure'),
    [Input('processed-data-store', 'data')]
)
def update_realtime_logs(processed_data):
    if not processed_data:
        return go.Figure()

    geo = pd.DataFrame(processed_data['geosteering'])
    current_idx = np.argmin(np.abs(geo['md'] - processed_data['current_md']))

    # Get last 50 points
    window_start = max(0, current_idx - 50)
    window_data = geo.iloc[window_start:current_idx + 1]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=('Gamma Ray & Resistivity', 'ROP & Torque')
    )

    # GR and Resistivity
    fig.add_trace(
        go.Scatter(x=window_data['md'], y=window_data['gamma_ray'],
                   mode='lines', line=dict(color=colors['success'], width=2),
                   name='GR', yaxis='y'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=window_data['md'], y=window_data['resistivity'],
                   mode='lines', line=dict(color=colors['warning'], width=2, dash='dash'),
                   name='Res', yaxis='y2'),
        row=1, col=1
    )

    # ROP
    fig.add_trace(
        go.Scatter(x=window_data['md'], y=window_data['rop'],
                   mode='lines', line=dict(color=colors['primary'], width=2),
                   fill='tozeroy', fillcolor='rgba(0, 212, 255, 0.1)',
                   name='ROP'),
        row=2, col=1
    )

    # Add target zone markers
    for _, row in window_data.iterrows():
        if row['in_target']:
            fig.add_vrect(
                x0=row['md'] - 5, x1=row['md'] + 5,
                fillcolor="green",
                opacity=0.2,
                layer="below",
                line_width=0
            )

    fig.update_xaxes(title='MD (ft)', row=2, col=1)
    fig.update_yaxes(title='GR (API)', row=1, col=1)
    fig.update_yaxes(title='ROP (ft/hr)', row=2, col=1)

    fig.update_layout(
        height=250,
        showlegend=True,
        legend=dict(x=0, y=1, font=dict(size=10)),
        margin=dict(t=30, b=40, l=60, r=10),
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['surface'],
        font=dict(color=colors['text'])
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

    # Primary steering recommendation
    recommendations.append({
        'Priority': 1,
        'Action': f'{optimization["adjustment"]} Trajectory',
        'Current': f'{optimization["current_inclination"]:.1f}°',
        'Target': f'{optimization["recommended_inclination"]:.1f}°',
        'Urgency': optimization['urgency'],
        'Impact': 'Stay in target zone'
    })

    # Formation-based recommendations
    if not current['in_target']:
        recommendations.append({
            'Priority': 2,
            'Action': 'Return to Target',
            'Current': current['current_layer'],
            'Target': 'Target Zone',
            'Urgency': 'HIGH',
            'Impact': f'Improve net pay by {100 - kpis["target_zone_percentage"]:.0f}%'
        })

    # Boundary proximity warning
    if current['in_target'] and min(current['dist_to_top'], current['dist_to_bottom']) < 20:
        recommendations.append({
            'Priority': 3,
            'Action': 'Adjust for Boundary',
            'Current': f'{min(current["dist_to_top"], current["dist_to_bottom"]):.1f} ft',
            'Target': '>30 ft clearance',
            'Urgency': 'MEDIUM',
            'Impact': 'Avoid exit from target'
        })

    # Drilling optimization
    if current['rop'] < 50:
        recommendations.append({
            'Priority': 4,
            'Action': 'Optimize Drilling',
            'Current': f'{current["rop"]:.0f} ft/hr',
            'Target': '>80 ft/hr',
            'Urgency': 'LOW',
            'Impact': 'Reduce drilling time'
        })

    return dash_table.DataTable(
        data=recommendations,
        columns=[{'name': col, 'id': col} for col in recommendations[0].keys()],
        style_cell={
            'textAlign': 'center',
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
                'if': {'column_id': 'Urgency', 'filter_query': '{Urgency} = "HIGH"'},
                'backgroundColor': colors['danger'],
                'color': 'white',
            },
            {
                'if': {'column_id': 'Urgency', 'filter_query': '{Urgency} = "MEDIUM"'},
                'backgroundColor': colors['warning'],
                'color': 'white',
            }
        ],
        sort_action="native"
    )


@app.callback(
    Output('performance-chart', 'figure'),
    [Input('processed-data-store', 'data')]
)
def update_performance_chart(processed_data):
    if not processed_data:
        return go.Figure()

    kpis = processed_data['kpis']

    # Create KPI gauge charts
    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=('Target Zone %', 'Net Pay', 'Avg Porosity', 'Drilling Efficiency'),
        specs=[[{'type': 'indicator'}, {'type': 'indicator'},
                {'type': 'indicator'}, {'type': 'indicator'}]]
    )

    # Target Zone Percentage
    fig.add_trace(
        go.Indicator(
            mode='gauge+number',
            value=kpis['target_zone_percentage'],
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': colors['success']},
                   'steps': [
                       {'range': [0, 50], 'color': colors['danger']},
                       {'range': [50, 80], 'color': colors['warning']},
                       {'range': [80, 100], 'color': colors['success']}]},
            domain={'x': [0, 0.25], 'y': [0, 1]}
        ),
        row=1, col=1
    )

    # Net Pay
    fig.add_trace(
        go.Indicator(
            mode='gauge+number',
            value=kpis['net_pay'],
            gauge={'axis': {'range': [None, kpis['lateral_length']]},
                   'bar': {'color': colors['primary']}},
            domain={'x': [0.25, 0.5], 'y': [0, 1]}
        ),
        row=1, col=2
    )

    # Average Porosity
    fig.add_trace(
        go.Indicator(
            mode='gauge+number',
            value=kpis.get('avg_porosity_in_target', 0),
            gauge={'axis': {'range': [None, 30]},
                   'bar': {'color': colors['warning']}},
            domain={'x': [0.5, 0.75], 'y': [0, 1]}
        ),
        row=1, col=3
    )

    # Drilling Efficiency
    fig.add_trace(
        go.Indicator(
            mode='gauge+number',
            value=kpis['drilling_efficiency'],
            gauge={'axis': {'range': [None, 200]},
                   'bar': {'color': colors['secondary']}},
            domain={'x': [0.75, 1], 'y': [0, 1]}
        ),
        row=1, col=4
    )

    fig.update_layout(
        height=300,
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['surface'],
        font=dict(color=colors['text'])
    )

    return fig


# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("AUTOMATA INTELLIGENCE")
    print("Reservoir Navigation with Geosteering & Optimization Suite")
    print("=" * 70)
    print("\nStarting server on http://127.0.0.1:8052")
    print("Press CTRL+C to stop\n")

    app.run(debug=False, port=8052, host='127.0.0.1')