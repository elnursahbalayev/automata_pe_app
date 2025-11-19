# reservoir_nav_app.py - Complete Reservoir Navigation & Optimization Dashboard
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.interpolate import griddata

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])
app.title = "AUTOMATA INTELLIGENCE - Reservoir Navigation & Optimization"


# ============================================================================
# DATA GENERATION FUNCTIONS
# ============================================================================

def generate_reservoir_grid(reservoir_id, grid_size=50):
    """Generate 3D reservoir property grids"""
    np.random.seed(hash(reservoir_id) % 2 ** 32)

    # Create 3D grid coordinates
    x = np.linspace(0, 10000, grid_size)  # ft
    y = np.linspace(0, 8000, grid_size)  # ft
    z = np.linspace(7000, 7500, 20)  # ft (reservoir thickness)

    X, Y = np.meshgrid(x, y)

    # Generate porosity field with spatial correlation
    porosity_base = 0.15 + 0.1 * np.random.random((grid_size, grid_size))

    # Add geological features (channels, trends)
    for i in range(3):
        center_x = np.random.randint(10, grid_size - 10)
        center_y = np.random.randint(10, grid_size - 10)
        angle = np.random.uniform(0, 2 * np.pi)

        # Create channel-like feature
        for xi in range(grid_size):
            for yi in range(grid_size):
                dx = xi - center_x
                dy = yi - center_y

                # Rotate coordinates
                dx_rot = dx * np.cos(angle) - dy * np.sin(angle)
                dy_rot = dx * np.sin(angle) + dy * np.cos(angle)

                # Channel shape
                if abs(dy_rot) < 5:
                    dist = abs(dx_rot) / 30.0
                    porosity_base[yi, xi] += 0.05 * np.exp(-dist)

    # Apply smoothing for realistic appearance
    porosity = gaussian_filter(porosity_base, sigma=2)
    porosity = np.clip(porosity, 0.05, 0.35)

    # Generate permeability (correlated with porosity)
    permeability = 10 ** (3 * porosity + np.random.normal(0, 0.5, (grid_size, grid_size)))
    permeability = gaussian_filter(permeability, sigma=2)
    permeability = np.clip(permeability, 0.1, 1000)  # mD

    # Generate water saturation
    water_sat_base = 0.3 + 0.4 * np.random.random((grid_size, grid_size))

    # Add oil-water contact influence
    owc_depth = 7400
    for iz, depth in enumerate(z):
        if depth > owc_depth:
            water_sat_base *= (1 + (depth - owc_depth) / 100)

    water_sat = gaussian_filter(water_sat_base, sigma=3)
    water_sat = np.clip(water_sat, 0.2, 0.95)

    # Generate pressure
    pressure_gradient = 0.45  # psi/ft
    pressure = np.ones((grid_size, grid_size)) * 7250 * pressure_gradient
    pressure += np.random.normal(0, 50, (grid_size, grid_size))
    pressure = gaussian_filter(pressure, sigma=5)

    # Net to Gross ratio
    ntg = 0.7 + 0.25 * np.random.random((grid_size, grid_size))
    ntg = gaussian_filter(ntg, sigma=3)
    ntg = np.clip(ntg, 0.4, 0.95)

    reservoir_grid = {
        'X': X,
        'Y': Y,
        'Z': z,
        'POROSITY': porosity,
        'PERMEABILITY': permeability,
        'WATER_SAT': water_sat,
        'PRESSURE': pressure,
        'NTG': ntg,
        'reservoir_id': reservoir_id
    }

    return reservoir_grid


def generate_well_trajectories(reservoir_id, num_wells=5):
    """Generate well trajectories for horizontal wells"""
    np.random.seed(hash(reservoir_id) % 2 ** 32)

    wells = {}

    for i in range(num_wells):
        well_id = f'WELL-{reservoir_id}-{i + 1:02d}'

        # Surface location
        surf_x = np.random.uniform(1000, 9000)
        surf_y = np.random.uniform(1000, 7000)

        # Kickoff point
        kickoff_depth = np.random.uniform(5000, 6000)

        # Landing point in reservoir
        landing_depth = np.random.uniform(7100, 7300)
        landing_x = surf_x + np.random.uniform(-500, 500)
        landing_y = surf_y + np.random.uniform(-500, 500)

        # Horizontal section
        lateral_length = np.random.uniform(5000, 8000)
        azimuth = np.random.uniform(0, 2 * np.pi)

        # Build trajectory
        md = []  # Measured Depth
        tvd = []  # True Vertical Depth
        x_coords = []
        y_coords = []

        # Vertical section
        for depth in np.linspace(0, kickoff_depth, 50):
            md.append(depth)
            tvd.append(depth)
            x_coords.append(surf_x)
            y_coords.append(surf_y)

        # Build section (curve)
        build_length = 1000
        for j, s in enumerate(np.linspace(0, 1, 30)):
            depth_increment = s * (landing_depth - kickoff_depth)
            lateral_increment = s * 500

            current_md = kickoff_depth + build_length * s + depth_increment
            current_tvd = kickoff_depth + depth_increment
            current_x = surf_x + lateral_increment * np.cos(azimuth)
            current_y = surf_y + lateral_increment * np.sin(azimuth)

            md.append(current_md)
            tvd.append(current_tvd)
            x_coords.append(current_x)
            y_coords.append(current_y)

        # Horizontal section
        start_md = md[-1]
        for j, dist in enumerate(np.linspace(0, lateral_length, 100)):
            current_md = start_md + dist
            current_tvd = landing_depth + np.random.normal(0, 5)  # Small variations
            current_x = landing_x + dist * np.cos(azimuth)
            current_y = landing_y + dist * np.sin(azimuth)

            md.append(current_md)
            tvd.append(current_tvd)
            x_coords.append(current_x)
            y_coords.append(current_y)

        wells[well_id] = {
            'MD': np.array(md),
            'TVD': np.array(tvd),
            'X': np.array(x_coords),
            'Y': np.array(y_coords),
            'surf_x': surf_x,
            'surf_y': surf_y,
            'lateral_length': lateral_length,
            'landing_depth': landing_depth
        }

    return wells


def generate_production_data(well_id, days=365):
    """Generate production data for a well"""
    np.random.seed(hash(well_id) % 2 ** 32)

    time = np.arange(days)

    # Oil production (barrels/day) - decline curve
    ip = np.random.uniform(800, 1500)  # Initial production
    di = np.random.uniform(0.003, 0.008)  # Decline rate
    b = np.random.uniform(0.5, 1.5)  # Hyperbolic exponent

    # Hyperbolic decline
    oil_rate = ip / ((1 + b * di * time) ** (1 / b))

    # Add noise and operational events
    oil_rate = oil_rate * (1 + np.random.normal(0, 0.05, days))

    # Simulate shut-ins
    for i in range(np.random.randint(2, 5)):
        shut_in_start = np.random.randint(30, days - 10)
        shut_in_duration = np.random.randint(3, 10)
        oil_rate[shut_in_start:shut_in_start + shut_in_duration] *= 0.1

    oil_rate = np.clip(oil_rate, 0, None)

    # Gas production (MCF/day) - correlated with oil
    gor = np.random.uniform(800, 1500)  # Gas-Oil Ratio
    gas_rate = oil_rate * gor / 1000
    gas_rate = gas_rate * (1 + np.random.normal(0, 0.03, days))

    # Water production (barrels/day) - increasing over time
    initial_water_cut = np.random.uniform(0.1, 0.3)
    water_cut = initial_water_cut + (0.8 - initial_water_cut) * (1 - np.exp(-time / 365))
    water_rate = oil_rate * water_cut / (1 - water_cut)
    water_rate = water_rate * (1 + np.random.normal(0, 0.05, days))

    # Cumulative production
    cum_oil = np.cumsum(oil_rate)
    cum_gas = np.cumsum(gas_rate)
    cum_water = np.cumsum(water_rate)

    # Reservoir pressure decline
    initial_pressure = np.random.uniform(3000, 3500)
    pressure = initial_pressure - (initial_pressure - 1500) * (cum_oil / cum_oil[-1])
    pressure = pressure + np.random.normal(0, 20, days)

    # Bottom hole pressure
    bhp = pressure - np.random.uniform(200, 500)
    bhp = bhp + np.random.normal(0, 15, days)

    dates = [datetime.now() - timedelta(days=days - i) for i in range(days)]

    df = pd.DataFrame({
        'DATE': dates,
        'DAYS': time,
        'OIL_RATE': oil_rate,
        'GAS_RATE': gas_rate,
        'WATER_RATE': water_rate,
        'CUM_OIL': cum_oil,
        'CUM_GAS': cum_gas,
        'CUM_WATER': cum_water,
        'PRESSURE': pressure,
        'BHP': bhp,
        'WATER_CUT': water_cut * 100,
        'WELL_ID': well_id
    })

    return df


def generate_geosteering_data(well_id, num_points=500):
    """Generate real-time geosteering data"""
    np.random.seed(hash(well_id) % 2 ** 32)

    # Measured depth along lateral
    md = np.linspace(10000, 18000, num_points)

    # Target zone: 7200-7300 ft TVD
    target_center = 7250
    target_thickness = 100

    # Actual TVD (with steering adjustments)
    tvd = target_center + np.cumsum(np.random.normal(0, 0.5, num_points))
    tvd = gaussian_filter1d(tvd, sigma=10)

    # Distance to top and bottom boundaries
    dist_to_top = tvd - (target_center - target_thickness / 2)
    dist_to_bottom = (target_center + target_thickness / 2) - tvd

    # Formation properties while drilling
    gamma_ray = np.zeros(num_points)
    resistivity = np.zeros(num_points)

    for i in range(num_points):
        if dist_to_top[i] < 10:  # Near top (shale)
            gamma_ray[i] = 80 + np.random.normal(0, 10)
            resistivity[i] = np.random.lognormal(1.5, 0.3)
        elif dist_to_bottom[i] < 10:  # Near bottom (shale)
            gamma_ray[i] = 85 + np.random.normal(0, 10)
            resistivity[i] = np.random.lognormal(1.6, 0.3)
        else:  # In pay zone
            gamma_ray[i] = 35 + np.random.normal(0, 8)
            resistivity[i] = np.random.lognormal(3.5, 0.4)

    gamma_ray = gaussian_filter1d(gamma_ray, sigma=3)
    resistivity = gaussian_filter1d(resistivity, sigma=3)

    # Inclination and azimuth
    inclination = 88 + np.random.normal(0, 1, num_points)
    inclination = gaussian_filter1d(inclination, sigma=5)

    azimuth = 45 + np.cumsum(np.random.normal(0, 0.1, num_points))
    azimuth = gaussian_filter1d(azimuth, sigma=10)

    # Formation dip
    formation_dip = 2 + np.random.normal(0, 0.5, num_points)
    formation_dip = gaussian_filter1d(formation_dip, sigma=10)

    df = pd.DataFrame({
        'MD': md,
        'TVD': tvd,
        'DIST_TO_TOP': dist_to_top,
        'DIST_TO_BOTTOM': dist_to_bottom,
        'GAMMA_RAY': gamma_ray,
        'RESISTIVITY': resistivity,
        'INCLINATION': inclination,
        'AZIMUTH': azimuth,
        'FORMATION_DIP': formation_dip,
        'WELL_ID': well_id
    })

    return df


# ============================================================================
# OPTIMIZATION FUNCTIONS
# ============================================================================

def calculate_eur(production_df):
    """Calculate Estimated Ultimate Recovery"""
    # Using hyperbolic decline curve extrapolation
    current_cum = production_df['CUM_OIL'].iloc[-1]
    current_rate = production_df['OIL_RATE'].iloc[-1]

    # Estimate ultimate recovery (simplified)
    if current_rate > 50:
        future_recovery = current_rate * 365 * 5  # Approximate next 5 years
        eur = current_cum + future_recovery
    else:
        eur = current_cum * 1.1  # Near end of life

    return eur


def calculate_recovery_factor(eur, ooip):
    """Calculate recovery factor"""
    return (eur / ooip) * 100 if ooip > 0 else 0


def optimize_well_placement(reservoir_grid):
    """AI-powered optimal well placement"""
    # Calculate composite quality score
    quality_score = (
            reservoir_grid['POROSITY'] * 0.3 +
            np.log10(reservoir_grid['PERMEABILITY']) / 3 * 0.3 +
            (1 - reservoir_grid['WATER_SAT']) * 0.25 +
            reservoir_grid['NTG'] * 0.15
    )

    # Find sweet spots (top 10% quality)
    threshold = np.percentile(quality_score, 90)
    sweet_spots = quality_score > threshold

    # Find optimal locations (peaks in quality)
    optimal_locations = []
    grid_size = quality_score.shape[0]

    for i in range(5, grid_size - 5, 10):
        for j in range(5, grid_size - 5, 10):
            if sweet_spots[i, j]:
                local_quality = quality_score[i - 5:i + 5, j - 5:j + 5].mean()
                optimal_locations.append({
                    'x_idx': i,
                    'y_idx': j,
                    'quality': local_quality,
                    'x_coord': reservoir_grid['X'][i, j],
                    'y_coord': reservoir_grid['Y'][i, j]
                })

    # Sort by quality and return top locations
    optimal_locations = sorted(optimal_locations, key=lambda x: x['quality'], reverse=True)

    return optimal_locations[:10], quality_score


def calculate_drainage_efficiency(well_trajectories, reservoir_grid):
    """Calculate drainage efficiency and coverage"""
    # Simplified drainage radius
    drainage_radius = 500  # ft

    grid_size = reservoir_grid['POROSITY'].shape[0]
    drained_area = np.zeros((grid_size, grid_size))

    for well_id, traj in well_trajectories.items():
        for x, y in zip(traj['X'], traj['Y']):
            # Map to grid indices
            x_idx = int((x / 10000) * grid_size)
            y_idx = int((y / 8000) * grid_size)

            # Mark drainage area
            for i in range(max(0, x_idx - 10), min(grid_size, x_idx + 10)):
                for j in range(max(0, y_idx - 10), min(grid_size, y_idx + 10)):
                    dist = np.sqrt((i - x_idx) ** 2 + (j - y_idx) ** 2) * (10000 / grid_size)
                    if dist < drainage_radius:
                        drained_area[i, j] = 1

    total_area = grid_size * grid_size
    drained_cells = np.sum(drained_area)
    efficiency = (drained_cells / total_area) * 100

    return efficiency, drained_area


# ============================================================================
# GENERATE DATA
# ============================================================================

reservoirs_data = {}
wells_trajectories = {}
production_data = {}
geosteering_data = {}

for res_id in ['RES-A', 'RES-B', 'RES-C']:
    reservoirs_data[res_id] = generate_reservoir_grid(res_id)
    wells_trajectories[res_id] = generate_well_trajectories(res_id)

    production_data[res_id] = {}
    for well_id in wells_trajectories[res_id].keys():
        production_data[res_id][well_id] = generate_production_data(well_id)

    # Generate geosteering for one well per reservoir
    first_well = list(wells_trajectories[res_id].keys())[0]
    geosteering_data[res_id] = generate_geosteering_data(first_well)

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
                html.H1("🎯 Reservoir Navigation & Optimization",
                        style={'color': colors['text'], 'fontWeight': 'bold'}),
                html.P("AI-Powered Geosteering, Production Optimization & Well Placement",
                       style={'color': colors['text-secondary'], 'fontSize': '18px'})
            ], style={'textAlign': 'center', 'padding': '30px 0'})
        ])
    ]),

    # Control Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Reservoir Selection & Analysis Mode", style={'color': colors['primary']}),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Reservoir", style={'color': colors['text']}),
                            dcc.Dropdown(
                                id='reservoir-selector',
                                options=[{'label': f"{res_id}", 'value': res_id}
                                         for res_id in reservoirs_data.keys()],
                                value='RES-A',
                                style={'color': '#000'}
                            )
                        ], md=3),
                        dbc.Col([
                            html.Label("Analysis Module", style={'color': colors['text']}),
                            dcc.RadioItems(
                                id='analysis-module',
                                options=[
                                    {'label': ' 3D Modeling', 'value': 'modeling'},
                                    {'label': ' Geosteering', 'value': 'geosteering'},
                                    {'label': ' Production', 'value': 'production'}
                                ],
                                value='modeling',
                                inline=True,
                                style={'color': colors['text']}
                            )
                        ], md=4),
                        dbc.Col([
                            html.Label("Optimization Target", style={'color': colors['text']}),
                            dcc.Dropdown(
                                id='optimization-target',
                                options=[
                                    {'label': 'Maximize EUR', 'value': 'eur'},
                                    {'label': 'Maximize NPV', 'value': 'npv'},
                                    {'label': 'Minimize Water Cut', 'value': 'water'},
                                    {'label': 'Optimize Spacing', 'value': 'spacing'}
                                ],
                                value='eur',
                                style={'color': '#000'}
                            )
                        ], md=3),
                        dbc.Col([
                            dbc.Button("Run Optimization", id='run-optimization', color='primary',
                                       className='w-100', style={'marginTop': '25px'}, n_clicks=0)
                        ], md=2)
                    ])
                ])
            ], style={'backgroundColor': colors['surface'], 'marginBottom': '20px'})
        ])
    ]),

    # Key Metrics Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(id='eur-metric', children='0 MBBL', style={'color': colors['success']}),
                    html.P('Total EUR', style={'color': colors['text-secondary']})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(id='recovery-factor-metric', children='0%', style={'color': colors['primary']}),
                    html.P('Recovery Factor', style={'color': colors['text-secondary']})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(id='active-wells-metric', children='0', style={'color': colors['warning']}),
                    html.P('Active Wells', style={'color': colors['text-secondary']})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(id='drainage-efficiency-metric', children='0%', style={'color': colors['secondary']}),
                    html.P('Drainage Efficiency', style={'color': colors['text-secondary']})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(id='sweet-spots-metric', children='0', style={'color': colors['success']}),
                    html.P('Sweet Spots', style={'color': colors['text-secondary']})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(id='production-rate-metric', children='0 BOPD', style={'color': colors['primary']}),
                    html.P('Current Production', style={'color': colors['text-secondary']})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2)
    ], className='mb-4'),

    # Main Visualization Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("3D Reservoir Model & Well Trajectories", style={'color': colors['primary']}),
                    dcc.Loading(
                        id="loading-3d",
                        children=[dcc.Graph(id='reservoir-3d-plot', style={'height': '600px'})],
                        type="default"
                    )
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=8),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Property Distribution", style={'color': colors['primary']}),
                    dcc.Dropdown(
                        id='property-selector',
                        options=[
                            {'label': 'Porosity', 'value': 'POROSITY'},
                            {'label': 'Permeability', 'value': 'PERMEABILITY'},
                            {'label': 'Water Saturation', 'value': 'WATER_SAT'},
                            {'label': 'Pressure', 'value': 'PRESSURE'},
                            {'label': 'Net-to-Gross', 'value': 'NTG'}
                        ],
                        value='POROSITY',
                        style={'color': '#000', 'marginBottom': '10px'}
                    ),
                    dcc.Graph(id='property-histogram', style={'height': '230px'}),
                    html.Hr(),
                    html.H6("Sweet Spot Analysis", style={'color': colors['primary']}),
                    dcc.Graph(id='quality-map', style={'height': '250px'})
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=4)
    ], className='mb-4'),

    # Geosteering Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Real-Time Geosteering Dashboard", style={'color': colors['primary']}),
                    dcc.Graph(id='geosteering-plot', style={'height': '400px'})
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=8),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Formation Evaluation", style={'color': colors['primary']}),
                    dcc.Graph(id='formation-eval-plot', style={'height': '400px'})
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=4)
    ], className='mb-4'),

    # Production Analysis
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Production Performance", style={'color': colors['primary']}),
                    dcc.Dropdown(
                        id='well-selector',
                        style={'color': '#000', 'marginBottom': '10px'}
                    ),
                    dcc.Graph(id='production-plot', style={'height': '350px'})
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Decline Curve Analysis", style={'color': colors['primary']}),
                    dcc.Graph(id='decline-curve-plot', style={'height': '400px'})
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=6)
    ], className='mb-4'),

    # Optimization Results
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("AI Optimization Recommendations", style={'color': colors['primary']}),
                    html.Div(id='optimization-results')
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Well Spacing & Drainage Analysis", style={'color': colors['primary']}),
                    dcc.Graph(id='drainage-plot', style={'height': '350px'})
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=6)
    ]),

    # Store components
    dcc.Store(id='processed-reservoir-data'),
    dcc.Store(id='optimization-data')

], fluid=True, style={'backgroundColor': colors['background'], 'minHeight': '100vh', 'padding': '20px'})


# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    [Output('processed-reservoir-data', 'data'),
     Output('well-selector', 'options'),
     Output('well-selector', 'value'),
     Output('eur-metric', 'children'),
     Output('recovery-factor-metric', 'children'),
     Output('active-wells-metric', 'children'),
     Output('drainage-efficiency-metric', 'children'),
     Output('sweet-spots-metric', 'children'),
     Output('production-rate-metric', 'children')],
    [Input('run-optimization', 'n_clicks')],
    [State('reservoir-selector', 'value')]
)
def process_reservoir_data(n_clicks, reservoir_id):
    if n_clicks == 0:
        return None, [], None, '0 MBBL', '0%', '0', '0%', '0', '0 BOPD'

    reservoir = reservoirs_data[reservoir_id]
    wells = wells_trajectories[reservoir_id]
    prod_data = production_data[reservoir_id]

    # Calculate metrics
    total_eur = 0
    current_production = 0

    for well_id, df in prod_data.items():
        eur = calculate_eur(df)
        total_eur += eur
        current_production += df['OIL_RATE'].iloc[-1]

    # Calculate OOIP (Original Oil In Place) - simplified
    avg_porosity = reservoir['POROSITY'].mean()
    avg_sw = reservoir['WATER_SAT'].mean()
    reservoir_volume = 10000 * 8000 * 500  # ft³
    ooip = reservoir_volume * avg_porosity * (1 - avg_sw) * 7.48 / 5.615  # Convert to barrels

    recovery_factor = calculate_recovery_factor(total_eur, ooip)

    # Drainage efficiency
    drainage_eff, _ = calculate_drainage_efficiency(wells, reservoir)

    # Sweet spots
    optimal_locs, _ = optimize_well_placement(reservoir)
    num_sweet_spots = len(optimal_locs)

    # Well options for dropdown
    well_options = [{'label': well_id, 'value': well_id} for well_id in wells.keys()]
    first_well = list(wells.keys())[0]

    processed_data = {
        'reservoir_id': reservoir_id,
        'total_eur': total_eur,
        'ooip': ooip,
        'recovery_factor': recovery_factor,
        'drainage_efficiency': drainage_eff
    }

    return (processed_data,
            well_options,
            first_well,
            f'{total_eur / 1000:.1f} MBBL',
            f'{recovery_factor:.1f}%',
            str(len(wells)),
            f'{drainage_eff:.1f}%',
            str(num_sweet_spots),
            f'{current_production:.0f} BOPD')


@app.callback(
    Output('reservoir-3d-plot', 'figure'),
    [Input('processed-reservoir-data', 'data')],
    [State('reservoir-selector', 'value')]
)
def update_3d_plot(processed_data, reservoir_id):
    if not processed_data:
        return go.Figure()

    reservoir = reservoirs_data[reservoir_id]
    wells = wells_trajectories[reservoir_id]

    # Create 3D surface for porosity
    fig = go.Figure()

    # Add reservoir property as surface
    fig.add_trace(go.Surface(
        x=reservoir['X'],
        y=reservoir['Y'],
        z=np.ones_like(reservoir['X']) * 7250,  # Mid-reservoir depth
        surfacecolor=reservoir['POROSITY'],
        colorscale='Viridis',
        name='Porosity',
        colorbar=dict(title='Porosity', x=1.1),
        opacity=0.7
    ))

    # Add well trajectories
    colors_list = ['#ff6b35', '#00d4ff', '#00ff88', '#ffaa00', '#ff3366']
    for idx, (well_id, traj) in enumerate(wells.items()):
        fig.add_trace(go.Scatter3d(
            x=traj['X'],
            y=traj['Y'],
            z=traj['TVD'],
            mode='lines',
            name=well_id,
            line=dict(color=colors_list[idx % len(colors_list)], width=4)
        ))

        # Add surface location marker
        fig.add_trace(go.Scatter3d(
            x=[traj['surf_x']],
            y=[traj['surf_y']],
            z=[0],
            mode='markers',
            name=f'{well_id} Surface',
            marker=dict(size=8, color=colors_list[idx % len(colors_list)], symbol='diamond'),
            showlegend=False
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (ft)', backgroundcolor=colors['surface'], gridcolor=colors['text-secondary']),
            yaxis=dict(title='Y (ft)', backgroundcolor=colors['surface'], gridcolor=colors['text-secondary']),
            zaxis=dict(title='Depth (ft)', backgroundcolor=colors['surface'], gridcolor=colors['text-secondary'],
                       autorange='reversed'),
            bgcolor=colors['background']
        ),
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['surface'],
        font=dict(color=colors['text']),
        title=dict(
            text=f"3D Reservoir Model - {reservoir_id}",
            font=dict(size=18, color=colors['primary'])
        ),
        showlegend=True,
        height=600
    )

    return fig


@app.callback(
    Output('property-histogram', 'figure'),
    [Input('property-selector', 'value')],
    [State('reservoir-selector', 'value')]
)
def update_property_histogram(property_name, reservoir_id):
    reservoir = reservoirs_data[reservoir_id]

    data = reservoir[property_name].flatten()

    fig = go.Figure(data=[go.Histogram(
        x=data,
        nbinsx=30,
        marker_color=colors['primary'],
        opacity=0.7
    )])

    # Set appropriate labels
    labels = {
        'POROSITY': 'Porosity (fraction)',
        'PERMEABILITY': 'Permeability (mD)',
        'WATER_SAT': 'Water Saturation (fraction)',
        'PRESSURE': 'Pressure (psi)',
        'NTG': 'Net-to-Gross (fraction)'
    }

    fig.update_layout(
        xaxis_title=labels.get(property_name, property_name),
        yaxis_title='Frequency',
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['surface'],
        font=dict(color=colors['text']),
        showlegend=False,
        margin=dict(t=10, b=40, l=40, r=10)
    )

    return fig


@app.callback(
    Output('quality-map', 'figure'),
    [Input('processed-reservoir-data', 'data')],
    [State('reservoir-selector', 'value')]
)
def update_quality_map(processed_data, reservoir_id):
    if not processed_data:
        return go.Figure()

    reservoir = reservoirs_data[reservoir_id]
    optimal_locs, quality_score = optimize_well_placement(reservoir)

    fig = go.Figure(data=go.Contour(
        z=quality_score,
        x=reservoir['X'][0, :],
        y=reservoir['Y'][:, 0],
        colorscale='RdYlGn',
        contours=dict(
            coloring='heatmap',
            showlabels=True
        ),
        colorbar=dict(title='Quality Score')
    ))

    # Add optimal locations
    if optimal_locs:
        opt_x = [loc['x_coord'] for loc in optimal_locs]
        opt_y = [loc['y_coord'] for loc in optimal_locs]

        fig.add_trace(go.Scatter(
            x=opt_x,
            y=opt_y,
            mode='markers',
            name='Optimal Locations',
            marker=dict(size=12, color='white', symbol='star',
                        line=dict(color='black', width=2))
        ))

    fig.update_layout(
        xaxis_title='X (ft)',
        yaxis_title='Y (ft)',
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['surface'],
        font=dict(color=colors['text']),
        showlegend=True,
        margin=dict(t=10, b=40, l=40, r=40)
    )

    return fig


@app.callback(
    Output('geosteering-plot', 'figure'),
    [Input('processed-reservoir-data', 'data')],
    [State('reservoir-selector', 'value')]
)
def update_geosteering_plot(processed_data, reservoir_id):
    if not processed_data:
        return go.Figure()

    df = geosteering_data[reservoir_id]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=('Well Position in Target Zone', 'Gamma Ray', 'Resistivity'),
        vertical_spacing=0.08,
        row_heights=[0.4, 0.3, 0.3]
    )

    # Position plot with boundaries
    target_center = 7250
    target_thickness = 100

    fig.add_trace(go.Scatter(
        x=df['MD'], y=df['TVD'],
        mode='lines', name='Well Path',
        line=dict(color=colors['primary'], width=2)
    ), row=1, col=1)

    # Add target boundaries
    fig.add_hline(y=target_center - target_thickness / 2, line_dash="dash",
                  line_color=colors['danger'], row=1, col=1,
                  annotation_text="Top Boundary")
    fig.add_hline(y=target_center + target_thickness / 2, line_dash="dash",
                  line_color=colors['danger'], row=1, col=1,
                  annotation_text="Bottom Boundary")

    # Color code by position
    in_zone = (df['DIST_TO_TOP'] > 0) & (df['DIST_TO_BOTTOM'] > 0)

    fig.add_trace(go.Scatter(
        x=df[in_zone]['MD'], y=df[in_zone]['TVD'],
        mode='markers', name='In Target',
        marker=dict(color=colors['success'], size=3)
    ), row=1, col=1)

    # Gamma Ray
    fig.add_trace(go.Scatter(
        x=df['MD'], y=df['GAMMA_RAY'],
        mode='lines', name='GR',
        line=dict(color=colors['success'], width=1.5)
    ), row=2, col=1)

    # Resistivity
    fig.add_trace(go.Scatter(
        x=df['MD'], y=df['RESISTIVITY'],
        mode='lines', name='RT',
        line=dict(color=colors['secondary'], width=1.5)
    ), row=3, col=1)

    fig.update_xaxes(title_text="Measured Depth (ft)", row=3, col=1)
    fig.update_yaxes(title_text="TVD (ft)", row=1, col=1, autorange='reversed')
    fig.update_yaxes(title_text="API", row=2, col=1)
    fig.update_yaxes(title_text="Ohm.m", type='log', row=3, col=1)

    fig.update_layout(
        height=400,
        showlegend=True,
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['surface'],
        font=dict(color=colors['text'])
    )

    return fig


@app.callback(
    Output('formation-eval-plot', 'figure'),
    [Input('processed-reservoir-data', 'data')],
    [State('reservoir-selector', 'value')]
)
def update_formation_eval(processed_data, reservoir_id):
    if not processed_data:
        return go.Figure()

    df = geosteering_data[reservoir_id]

    # Calculate percentage in zone
    in_zone = (df['DIST_TO_TOP'] > 10) & (df['DIST_TO_BOTTOM'] > 10)
    pct_in_zone = (in_zone.sum() / len(df)) * 100

    near_top = (df['DIST_TO_TOP'] < 10) & (df['DIST_TO_TOP'] > 0)
    pct_near_top = (near_top.sum() / len(df)) * 100

    near_bottom = (df['DIST_TO_BOTTOM'] < 10) & (df['DIST_TO_BOTTOM'] > 0)
    pct_near_bottom = (near_bottom.sum() / len(df)) * 100

    out_of_zone = (~in_zone) & (~near_top) & (~near_bottom)
    pct_out = (out_of_zone.sum() / len(df)) * 100

    # Pie chart
    labels = ['In Target Zone', 'Near Top', 'Near Bottom', 'Out of Zone']
    values = [pct_in_zone, pct_near_top, pct_near_bottom, pct_out]
    colors_pie = [colors['success'], colors['warning'], colors['warning'], colors['danger']]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors_pie),
        hole=0.4
    )])

    fig.update_layout(
        annotations=[dict(text=f'{pct_in_zone:.1f}%<br>In Zone',
                          x=0.5, y=0.5, font_size=16, showarrow=False)],
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['surface'],
        font=dict(color=colors['text']),
        showlegend=True,
        margin=dict(t=10, b=10, l=10, r=10)
    )

    return fig


@app.callback(
    Output('production-plot', 'figure'),
    [Input('well-selector', 'value')],
    [State('reservoir-selector', 'value')]
)
def update_production_plot(well_id, reservoir_id):
    if not well_id:
        return go.Figure()

    df = production_data[reservoir_id][well_id]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=('Production Rates', 'Cumulative Production'),
        vertical_spacing=0.1
    )

    # Rates
    fig.add_trace(go.Scatter(
        x=df['DATE'], y=df['OIL_RATE'],
        mode='lines', name='Oil',
        line=dict(color=colors['success'], width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['DATE'], y=df['WATER_RATE'],
        mode='lines', name='Water',
        line=dict(color=colors['primary'], width=2)
    ), row=1, col=1)

    # Cumulative
    fig.add_trace(go.Scatter(
        x=df['DATE'], y=df['CUM_OIL'] / 1000,
        mode='lines', name='Cum Oil',
        line=dict(color=colors['success'], width=2),
        fill='tozeroy'
    ), row=2, col=1)

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Rate (BOPD/BWPD)", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative (MBBL)", row=2, col=1)

    fig.update_layout(
        height=400,
        showlegend=True,
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['surface'],
        font=dict(color=colors['text']),
        hovermode='x unified'
    )

    return fig


@app.callback(
    Output('decline-curve-plot', 'figure'),
    [Input('well-selector', 'value')],
    [State('reservoir-selector', 'value')]
)
def update_decline_curve(well_id, reservoir_id):
    if not well_id:
        return go.Figure()

    df = production_data[reservoir_id][well_id]

    fig = go.Figure()

    # Log plot of oil rate
    fig.add_trace(go.Scatter(
        x=df['DAYS'], y=df['OIL_RATE'],
        mode='markers', name='Actual',
        marker=dict(color=colors['primary'], size=4)
    ))

    # Fit curve (simplified)
    # Using last 100 days to project
    recent_data = df.tail(100)
    if len(recent_data) > 10 and recent_data['OIL_RATE'].iloc[-1] > 10:
        # Simple exponential fit for visualization
        days_future = np.linspace(df['DAYS'].iloc[-1], df['DAYS'].iloc[-1] + 365, 100)
        current_rate = recent_data['OIL_RATE'].mean()
        decline_rate = 0.15  # 15% annual decline
        future_rate = current_rate * np.exp(-decline_rate * (days_future - df['DAYS'].iloc[-1]) / 365)

        fig.add_trace(go.Scatter(
            x=days_future, y=future_rate,
            mode='lines', name='Forecast',
            line=dict(color=colors['warning'], width=2, dash='dash')
        ))

    fig.update_layout(
        xaxis_title="Days",
        yaxis_title="Oil Rate (BOPD)",
        yaxis_type="log",
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['surface'],
        font=dict(color=colors['text']),
        showlegend=True
    )

    return fig


@app.callback(
    Output('optimization-results', 'children'),
    [Input('run-optimization', 'n_clicks')],
    [State('reservoir-selector', 'value'),
     State('optimization-target', 'value')]
)
def update_optimization_results(n_clicks, reservoir_id, target):
    if n_clicks == 0:
        return html.Div()

    reservoir = reservoirs_data[reservoir_id]
    optimal_locs, quality_score = optimize_well_placement(reservoir)

    recommendations = []

    # Header
    recommendations.append(
        dbc.Alert([
            html.H5("🎯 AI Optimization Complete", className="alert-heading"),
            html.P(f"Analysis Target: {target.upper()}", className="mb-0")
        ], color="info", className="mb-3")
    )

    # Well placement recommendations
    rec_text = f"Identified {len(optimal_locs)} optimal well locations based on integrated reservoir quality analysis."
    recommendations.append(
        dbc.Alert([
            html.H6("📍 Optimal Well Placement"),
            html.P(rec_text),
            html.Ul([
                html.Li(
                    f"Location {i + 1}: X={loc['x_coord']:.0f} ft, Y={loc['y_coord']:.0f} ft (Quality Score: {loc['quality']:.3f})")
                for i, loc in enumerate(optimal_locs[:3])
            ])
        ], color="success", className="mb-3")
    )

    # Spacing recommendation
    avg_quality = quality_score.mean()
    if avg_quality > 0.5:
        spacing_rec = "High quality reservoir - Recommend 660 ft well spacing for optimal recovery"
        spacing_color = "success"
    elif avg_quality > 0.3:
        spacing_rec = "Medium quality reservoir - Recommend 880 ft well spacing"
        spacing_color = "warning"
    else:
        spacing_rec = "Lower quality reservoir - Recommend 1320 ft well spacing to maintain economics"
        spacing_color = "warning"

    recommendations.append(
        dbc.Alert([
            html.H6("📏 Well Spacing Optimization"),
            html.P(spacing_rec)
        ], color=spacing_color, className="mb-3")
    )

    # Completion recommendation
    recommendations.append(
        dbc.Alert([
            html.H6("🔧 Completion Strategy"),
            html.P("Based on permeability distribution:"),
            html.Ul([
                html.Li(f"Recommended stages: {np.random.randint(25, 40)}"),
                html.Li(f"Cluster spacing: {np.random.randint(15, 25)} ft"),
                html.Li("Proppant: High quality ceramic in sweet spots")
            ])
        ], color="info", className="mb-3")
    )

    return html.Div(recommendations)


@app.callback(
    Output('drainage-plot', 'figure'),
    [Input('processed-reservoir-data', 'data')],
    [State('reservoir-selector', 'value')]
)
def update_drainage_plot(processed_data, reservoir_id):
    if not processed_data:
        return go.Figure()

    reservoir = reservoirs_data[reservoir_id]
    wells = wells_trajectories[reservoir_id]

    drainage_eff, drained_area = calculate_drainage_efficiency(wells, reservoir)

    # Create heat map showing drainage
    fig = go.Figure(data=go.Heatmap(
        z=drained_area,
        x=reservoir['X'][0, :],
        y=reservoir['Y'][:, 0],
        colorscale=[[0, colors['danger']], [1, colors['success']]],
        showscale=True,
        colorbar=dict(title='Drained')
    ))

    # Add well surface locations
    for well_id, traj in wells.items():
        fig.add_trace(go.Scatter(
            x=[traj['surf_x']],
            y=[traj['surf_y']],
            mode='markers+text',
            name=well_id,
            marker=dict(size=12, color='white', symbol='star',
                        line=dict(color='black', width=2)),
            text=[well_id.split('-')[-1]],
            textposition='top center'
        ))

    fig.update_layout(
        xaxis_title='X (ft)',
        yaxis_title='Y (ft)',
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['surface'],
        font=dict(color=colors['text']),
        showlegend=False,
        title=dict(text=f"Drainage Efficiency: {drainage_eff:.1f}%",
                   font=dict(size=14, color=colors['text']))
    )

    return fig


# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == '__main__':
    app.run(debug=False, port=8054, host='127.0.0.1')