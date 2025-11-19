# reservoir_optimization_app.py - Complete Reservoir Navigation & Optimization Dashboard (FIXED)
import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])
app.title = "AUTOMATA INTELLIGENCE Reservoir Navigation & Optimization Suite"

# ============================================================================
# DATA GENERATION FUNCTIONS
# ============================================================================

def generate_reservoir_grid(field_id, nx=50, ny=50, nz=10):
    """Generate 3D reservoir grid with properties"""
    np.random.seed(hash(field_id) % 2**32)
    
    # Create grid coordinates
    x = np.linspace(0, 5000, nx)  # meters
    y = np.linspace(0, 5000, ny)  # meters
    z = np.linspace(2800, 3200, nz)  # meters depth
    
    # Generate reservoir properties
    properties = {}
    
    # Create base structure with anticline
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Porosity distribution (higher at crest)
    distance_from_center = np.sqrt((X - 2500)**2 + (Y - 2500)**2)
    base_porosity = 0.25 - distance_from_center/50000
    porosity = base_porosity + np.random.normal(0, 0.02, X.shape)
    properties['porosity'] = np.clip(porosity, 0.05, 0.35)
    
    # Permeability (correlated with porosity)
    properties['permeability'] = 100 * np.exp(10 * properties['porosity']) + np.random.normal(0, 50, X.shape)
    properties['permeability'] = np.clip(properties['permeability'], 1, 2000)
    
    # Oil saturation (higher at top of structure)
    base_so = 0.8 - (Z - Z.min())/(Z.max() - Z.min()) * 0.3
    properties['oil_saturation'] = base_so + np.random.normal(0, 0.05, X.shape)
    properties['oil_saturation'] = np.clip(properties['oil_saturation'], 0, 0.85)
    
    # Pressure distribution (hydrostatic + depletion)
    properties['pressure'] = 3000 + Z * 0.433 + np.random.normal(0, 50, X.shape)  # psi
    
    # Net to Gross
    properties['ntg'] = 0.8 + np.random.normal(0, 0.1, X.shape)
    properties['ntg'] = np.clip(properties['ntg'], 0.3, 1.0)
    
    return X, Y, Z, properties

def generate_well_data(field_id, num_wells=12):
    """Generate well locations and production data"""
    np.random.seed(hash(field_id) % 2**32 + 1)
    
    wells = []
    well_types = ['Producer', 'Injector', 'Observer']
    
    for i in range(num_wells):
        well_type = np.random.choice(well_types, p=[0.6, 0.3, 0.1])
        
        # Generate well trajectory
        if well_type == 'Producer':
            # Producers at structural highs
            x = 2500 + np.random.normal(0, 800)
            y = 2500 + np.random.normal(0, 800)
            z_top = 2850 + np.random.normal(0, 20)
            z_bottom = 3150 + np.random.normal(0, 20)
        else:
            # Injectors at flanks
            angle = np.random.uniform(0, 2*np.pi)
            radius = np.random.uniform(1500, 2000)
            x = 2500 + radius * np.cos(angle)
            y = 2500 + radius * np.sin(angle)
            z_top = 2900 + np.random.normal(0, 30)
            z_bottom = 3150 + np.random.normal(0, 30)
        
        # Production/Injection history
        days = 365
        time_series = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        if well_type == 'Producer':
            # Oil production decline curve
            initial_rate = np.random.uniform(500, 2000)  # bbl/day
            decline_rate = np.random.uniform(0.001, 0.003)
            oil_rate = initial_rate * np.exp(-decline_rate * np.arange(days))
            oil_rate += np.random.normal(0, 50, days)
            oil_rate = np.clip(oil_rate, 0, None)
            
            # Water cut increase
            initial_wc = np.random.uniform(0.05, 0.2)
            wc_increase = np.random.uniform(0.0001, 0.001)
            water_cut = initial_wc + wc_increase * np.arange(days)
            water_cut = np.clip(water_cut, 0, 0.95)
            
            # GOR trend
            gor = 500 + np.random.normal(0, 50, days) + np.arange(days) * 0.5
            
            water_rate = oil_rate * water_cut / (1 - water_cut)
            gas_rate = oil_rate * gor / 1000
            
        elif well_type == 'Injector':
            # Water injection
            water_rate = np.random.uniform(1000, 3000) + np.random.normal(0, 100, days)
            water_rate = np.clip(water_rate, 0, None)
            oil_rate = np.zeros(days)
            water_cut = np.ones(days)
            gas_rate = np.zeros(days)
            gor = np.zeros(days)
        else:
            # Observer well
            oil_rate = np.zeros(days)
            water_rate = np.zeros(days)
            water_cut = np.zeros(days)
            gas_rate = np.zeros(days)
            gor = np.zeros(days)
        
        wells.append({
            'well_id': f'WELL-{i+1:03d}',
            'type': well_type,
            'x': x,
            'y': y,
            'z_top': z_top,
            'z_bottom': z_bottom,
            'status': 'Active' if np.random.random() > 0.1 else 'Shut-in',
            'completion_date': datetime.now() - timedelta(days=np.random.randint(100, 1000)),
            'time_series': time_series,
            'oil_rate': oil_rate,
            'water_rate': water_rate,
            'gas_rate': gas_rate,
            'water_cut': water_cut,
            'gor': gor,
            'cum_oil': np.cumsum(oil_rate),
            'cum_water': np.cumsum(water_rate),
            'cum_gas': np.cumsum(gas_rate)
        })
    
    return wells

def generate_field_metadata(field_id):
    """Generate field metadata"""
    fields = ['North Field', 'South Basin', 'East Platform', 'West Prospect', 'Central Hub']
    operators = ['ReservoirTech Corp', 'Global Production', 'Integrated Energy', 'Field Solutions']
    
    np.random.seed(hash(field_id) % 2**32)
    
    metadata = {
        'field_id': field_id,
        'field_name': np.random.choice(fields),
        'operator': np.random.choice(operators),
        'discovery_date': datetime.now() - timedelta(days=np.random.randint(3650, 7300)),
        'first_production': datetime.now() - timedelta(days=np.random.randint(1825, 3650)),
        'area_acres': np.random.randint(5000, 20000),
        'ooip': np.random.randint(100, 500),  # Million barrels
        'current_rf': np.random.uniform(0.15, 0.35),
        'target_rf': np.random.uniform(0.40, 0.60),
        'reservoir_type': np.random.choice(['Sandstone', 'Carbonate', 'Fractured']),
        'drive_mechanism': np.random.choice(['Water Drive', 'Gas Cap', 'Solution Gas', 'Combination']),
        'api_gravity': np.random.uniform(28, 42),
        'reservoir_temp': np.random.uniform(150, 250),  # Fahrenheit
        'reservoir_pressure': np.random.uniform(2500, 4500)  # psi
    }
    
    return metadata

def calculate_recovery_metrics(wells, metadata):
    """Calculate field recovery metrics"""
    total_oil = sum([w['cum_oil'][-1] if len(w['cum_oil']) > 0 else 0 for w in wells])
    total_water = sum([w['cum_water'][-1] if len(w['cum_water']) > 0 else 0 for w in wells])
    total_gas = sum([w['cum_gas'][-1] if len(w['cum_gas']) > 0 else 0 for w in wells])
    
    current_oil_rate = sum([w['oil_rate'][-1] if len(w['oil_rate']) > 0 else 0 for w in wells])
    current_water_rate = sum([w['water_rate'][-1] if len(w['water_rate']) > 0 else 0 for w in wells])
    current_gas_rate = sum([w['gas_rate'][-1] if len(w['gas_rate']) > 0 else 0 for w in wells])
    
    active_producers = len([w for w in wells if w['type'] == 'Producer' and w['status'] == 'Active'])
    active_injectors = len([w for w in wells if w['type'] == 'Injector' and w['status'] == 'Active'])
    
    metrics = {
        'cumulative_oil': total_oil,
        'cumulative_water': total_water,
        'cumulative_gas': total_gas,
        'current_oil_rate': current_oil_rate,
        'current_water_rate': current_water_rate,
        'current_gas_rate': current_gas_rate,
        'field_water_cut': current_water_rate / (current_oil_rate + current_water_rate) * 100 if (current_oil_rate + current_water_rate) > 0 else 0,
        'field_gor': current_gas_rate / current_oil_rate * 1000 if current_oil_rate > 0 else 0,
        'active_producers': active_producers,
        'active_injectors': active_injectors,
        'voidage_replacement': current_water_rate / (current_oil_rate + current_water_rate) if (current_oil_rate + current_water_rate) > 0 else 0
    }
    
    return metrics

def optimize_well_placement(X, Y, Z, properties, num_new_wells=3):
    """AI-based optimal well placement recommendations"""
    # Find sweet spots based on porosity, permeability, and oil saturation
    quality_index = (properties['porosity'] * 
                    properties['oil_saturation'] * 
                    np.log10(properties['permeability'] + 1) / 10)
    
    # Average over depth
    quality_map = np.mean(quality_index, axis=2)
    
    # Apply Gaussian filter for smoothing
    quality_map = gaussian_filter(quality_map, sigma=2)
    
    # Find peaks
    recommendations = []
    for _ in range(num_new_wells):
        # Find maximum location
        max_idx = np.unravel_index(quality_map.argmax(), quality_map.shape)
        x_opt = X[max_idx[0], max_idx[1], 0]
        y_opt = Y[max_idx[0], max_idx[1], 0]
        
        recommendations.append({
            'x': x_opt,
            'y': y_opt,
            'quality_score': quality_map[max_idx],
            'estimated_rate': 500 + quality_map[max_idx] * 1000,
            'confidence': 85 + np.random.randint(-5, 10)
        })
        
        # Zero out nearby area to find next spot
        mask_x = (X[:,:,0] - x_opt)**2 + (Y[:,:,0] - y_opt)**2 < 500000
        quality_map[mask_x] = 0
    
    return recommendations

# ============================================================================
# GENERATE DATA FOR MULTIPLE FIELDS - FIXED
# ============================================================================

fields_metadata = {}
fields_wells = {}
fields_grid = {}

# Generate data for each field
for field_id in ['FIELD-001', 'FIELD-002', 'FIELD-003']:
    X, Y, Z, properties = generate_reservoir_grid(field_id)
    fields_grid[field_id] = {'X': X, 'Y': Y, 'Z': Z, 'properties': properties}
    fields_wells[field_id] = generate_well_data(field_id)
    fields_metadata[field_id] = generate_field_metadata(field_id)

# Define color scheme (consistent with other apps)
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
# DASH LAYOUT - FIXED DROPDOWN
# ============================================================================

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("AUTOMATA INTELLIGENCE",
                        style={'color': colors['text'], 'fontWeight': 'bold'}),
                html.H1("🗺️ Reservoir Navigation & Optimization Suite",
                        style={'color': colors['text'], 'fontWeight': 'bold'}),
                html.P("AI-Powered Field Development & Production Optimization",
                       style={'color': colors['text-secondary'], 'fontSize': '18px'})
            ], style={'textAlign': 'center', 'padding': '30px 0'})
        ])
    ]),
    
    # Control Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Field Selection & Analysis Controls", style={'color': colors['primary']}),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Field", style={'color': colors['text']}),
                            dcc.Dropdown(
                                id='field-selector',
                                options=[{'label': f'{field} - {fields_metadata[field]["field_name"]}', 
                                         'value': field} for field in fields_metadata.keys()],  # FIXED: Using fields_metadata instead of fields_data
                                value='FIELD-001',
                                style={'color': '#000'}
                            )
                        ], md=3),
                        dbc.Col([
                            html.Label("Display Property", style={'color': colors['text']}),
                            dcc.Dropdown(
                                id='property-selector',
                                options=[
                                    {'label': 'Porosity', 'value': 'porosity'},
                                    {'label': 'Permeability', 'value': 'permeability'},
                                    {'label': 'Oil Saturation', 'value': 'oil_saturation'},
                                    {'label': 'Pressure', 'value': 'pressure'},
                                    {'label': 'Net to Gross', 'value': 'ntg'}
                                ],
                                value='oil_saturation',
                                style={'color': '#000'}
                            )
                        ], md=2),
                        dbc.Col([
                            html.Label("Optimization Mode", style={'color': colors['text']}),
                            dcc.RadioItems(
                                id='optimization-mode',
                                options=[
                                    {'label': ' Recovery', 'value': 'recovery'},
                                    {'label': ' Economics', 'value': 'economics'},
                                    {'label': ' Risk', 'value': 'risk'}
                                ],
                                value='recovery',
                                inline=True,
                                style={'color': colors['text']}
                            )
                        ], md=3),
                        dbc.Col([
                            html.Label("Time Range", style={'color': colors['text']}),
                            dcc.Dropdown(
                                id='time-range',
                                options=[
                                    {'label': 'Last 30 Days', 'value': 30},
                                    {'label': 'Last 90 Days', 'value': 90},
                                    {'label': 'Last 180 Days', 'value': 180},
                                    {'label': 'Last Year', 'value': 365},
                                    {'label': 'All Time', 'value': 9999}
                                ],
                                value=180,
                                style={'color': '#000'}
                            )
                        ], md=2),
                        dbc.Col([
                            dbc.Button("Run Optimization", id='optimize-btn', color='primary',
                                      className='w-100', style={'marginTop': '25px'}, n_clicks=0)
                        ], md=2)
                    ])
                ])
            ], style={'backgroundColor': colors['surface'], 'marginBottom': '20px'})
        ])
    ]),
    
    # Key Performance Metrics
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='current-production', children='0 bbl/d', 
                            style={'color': colors['success']}),
                    html.P('Current Oil Rate', style={'color': colors['text-secondary']}),
                    html.Small(id='production-change', style={'fontSize': '12px'})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='recovery-factor', children='0%', 
                            style={'color': colors['primary']}),
                    html.P('Recovery Factor', style={'color': colors['text-secondary']}),
                    html.Small(id='rf-target', style={'fontSize': '12px', 'color': colors['text-secondary']})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='water-cut', children='0%', 
                            style={'color': colors['warning']}),
                    html.P('Field Water Cut', style={'color': colors['text-secondary']}),
                    html.Small('Trend', id='wc-trend', style={'color': colors['warning']})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='active-wells', children='0/0', 
                            style={'color': colors['secondary']}),
                    html.P('Producers/Injectors', style={'color': colors['text-secondary']}),
                    html.Small(id='well-efficiency', style={'fontSize': '12px', 'color': colors['text-secondary']})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='voidage-ratio', children='0.0', 
                            style={'color': colors['primary']}),
                    html.P('Voidage Ratio', style={'color': colors['text-secondary']}),
                    html.Small(id='pressure-support', style={'fontSize': '12px'})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id='optimization-score', children='0%', 
                            style={'color': colors['success']}),
                    html.P('Optimization', style={'color': colors['text-secondary']}),
                    html.Small('AI Score', style={'color': colors['text-secondary']})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=2)
    ], className='mb-4'),
    
    # Main Visualization Area
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Tabs(id='main-tabs', value='3d-view', children=[
                        dcc.Tab(label='3D Reservoir View', value='3d-view',
                               style={'backgroundColor': colors['surface'], 'color': colors['text']},
                               selected_style={'backgroundColor': colors['primary'], 'color': colors['background']}),
                        dcc.Tab(label='Production Analysis', value='production',
                               style={'backgroundColor': colors['surface'], 'color': colors['text']},
                               selected_style={'backgroundColor': colors['primary'], 'color': colors['background']}),
                        dcc.Tab(label='Well Performance', value='wells',
                               style={'backgroundColor': colors['surface'], 'color': colors['text']},
                               selected_style={'backgroundColor': colors['primary'], 'color': colors['background']}),
                        dcc.Tab(label='Optimization Plan', value='optimization',
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
                    html.H5("Property Distribution", style={'color': colors['primary'], 'marginBottom': '20px'}),
                    dcc.Graph(id='property-histogram', style={'height': '200px'}),
                    html.Hr(),
                    html.H5("Recovery Forecast", style={'color': colors['primary'], 'marginTop': '20px'}),
                    dcc.Graph(id='recovery-forecast', style={'height': '200px'}),
                    html.Hr(),
                    html.H5("Field Map", style={'color': colors['primary'], 'marginTop': '20px'}),
                    dcc.Graph(id='field-map', style={'height': '250px'})
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=4)
    ], className='mb-4'),
    
    # Optimization Results Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Optimization Recommendations", style={'color': colors['primary']}),
                    html.Div(id='recommendations-table')
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Economic Analysis", style={'color': colors['primary']}),
                    dcc.Graph(id='economics-chart', style={'height': '300px'})
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=6)
    ]),
    
    # Store components
    dcc.Store(id='processed-data-store'),
    dcc.Store(id='optimization-results-store')
    
], fluid=True, style={'backgroundColor': colors['background'], 'minHeight': '100vh', 'padding': '20px'})

# ============================================================================
# CALLBACKS (unchanged from original, just included for completeness)
# ============================================================================

@app.callback(
    [Output('processed-data-store', 'data'),
     Output('current-production', 'children'),
     Output('production-change', 'children'),
     Output('recovery-factor', 'children'),
     Output('rf-target', 'children'),
     Output('water-cut', 'children'),
     Output('active-wells', 'children'),
     Output('well-efficiency', 'children'),
     Output('voidage-ratio', 'children'),
     Output('pressure-support', 'children'),
     Output('optimization-score', 'children')],
    [Input('optimize-btn', 'n_clicks')],
    [State('field-selector', 'value'),
     State('property-selector', 'value'),
     State('time-range', 'value'),
     State('optimization-mode', 'value')]
)
def process_field_data(n_clicks, field_id, property_name, time_range, opt_mode):
    if n_clicks == 0:
        return (None, '0 bbl/d', '', '0%', '', '0%', '0/0', '', '0.0', '', '0%')
    
    # Get field data
    grid_data = fields_grid[field_id]
    wells = fields_wells[field_id]
    metadata = fields_metadata[field_id]
    
    # Calculate metrics
    metrics = calculate_recovery_metrics(wells, metadata)
    
    # Production change
    if metrics['current_oil_rate'] > 0:
        last_month_rate = np.mean([w['oil_rate'][-30:].mean() for w in wells if w['type'] == 'Producer'])
        change = (metrics['current_oil_rate'] - last_month_rate) / last_month_rate * 100
        if change > 0:
            prod_change = html.Span(['📈 ', f'+{change:.1f}%'], style={'color': colors['success']})
        else:
            prod_change = html.Span(['📉 ', f'{change:.1f}%'], style={'color': colors['danger']})
    else:
        prod_change = ''
    
    # Recovery factor
    rf_current = metadata['current_rf'] * 100
    rf_target = metadata['target_rf'] * 100
    rf_target_text = f"Target: {rf_target:.1f}%"
    
    # Well efficiency
    avg_rate_per_well = metrics['current_oil_rate'] / metrics['active_producers'] if metrics['active_producers'] > 0 else 0
    efficiency_text = f"{avg_rate_per_well:.0f} bbl/d per well"
    
    # Pressure support
    if metrics['voidage_replacement'] > 1.1:
        pressure_text = "Over-injecting"
        pressure_color = colors['warning']
    elif metrics['voidage_replacement'] > 0.9:
        pressure_text = "Balanced"
        pressure_color = colors['success']
    else:
        pressure_text = "Under-injecting"
        pressure_color = colors['danger']
    
    pressure_support = html.Span(pressure_text, style={'color': pressure_color})
    
    # Calculate optimization score
    opt_score = 75 + np.random.randint(-10, 15)
    
    # Get optimal well locations
    recommendations = optimize_well_placement(
        grid_data['X'], 
        grid_data['Y'], 
        grid_data['Z'],
        grid_data['properties']
    )
    
    # Prepare data for storage
    processed_data = {
        'field_id': field_id,
        'grid': {
            'X': grid_data['X'].tolist(),
            'Y': grid_data['Y'].tolist(),
            'Z': grid_data['Z'].tolist(),
            'property_values': grid_data['properties'][property_name].tolist()
        },
        'wells': [{
            'well_id': w['well_id'],
            'type': w['type'],
            'x': w['x'],
            'y': w['y'],
            'z_top': w['z_top'],
            'z_bottom': w['z_bottom'],
            'status': w['status'],
            'oil_rate': w['oil_rate'][-time_range:].tolist() if time_range < len(w['oil_rate']) else w['oil_rate'].tolist(),
            'water_rate': w['water_rate'][-time_range:].tolist() if time_range < len(w['water_rate']) else w['water_rate'].tolist(),
            'gas_rate': w['gas_rate'][-time_range:].tolist() if time_range < len(w['gas_rate']) else w['gas_rate'].tolist(),
            'time_series': [str(t) for t in w['time_series'][-time_range:]] if time_range < len(w['time_series']) else [str(t) for t in w['time_series']]
        } for w in wells],
        'metadata': metadata,
        'metrics': metrics,
        'recommendations': recommendations,
        'property_name': property_name
    }
    
    return (
        processed_data,
        f"{metrics['current_oil_rate']:,.0f} bbl/d",
        prod_change,
        f"{rf_current:.1f}%",
        rf_target_text,
        f"{metrics['field_water_cut']:.1f}%",
        f"{metrics['active_producers']}/{metrics['active_injectors']}",
        efficiency_text,
        f"{metrics['voidage_replacement']:.2f}",
        pressure_support,
        f"{opt_score}%"
    )

# ... [Rest of the callbacks remain the same as in the original code]
# Including all other callbacks for completeness

@app.callback(
    Output('tab-content', 'children'),
    [Input('main-tabs', 'value'),
     Input('processed-data-store', 'data')]
)
def update_tab_content(active_tab, processed_data):
    if not processed_data:
        return html.Div("No data available. Click 'Run Optimization' to start.", 
                        style={'color': colors['text-secondary'], 'textAlign': 'center', 'padding': '50px'})
    
    if active_tab == '3d-view':
        # Create 3D reservoir visualization
        X = np.array(processed_data['grid']['X'])
        Y = np.array(processed_data['grid']['Y'])
        Z = np.array(processed_data['grid']['Z'])
        values = np.array(processed_data['grid']['property_values'])
        
        # Create a slice at middle depth
        mid_z = Z.shape[2] // 2
        
        fig = go.Figure()
        
        # Add surface plot
        fig.add_trace(go.Surface(
            x=X[:,:,mid_z],
            y=Y[:,:,mid_z],
            z=Z[:,:,mid_z],
            surfacecolor=values[:,:,mid_z],
            colorscale='Viridis',
            colorbar=dict(title=processed_data['property_name'].replace('_', ' ').title()),
            name='Reservoir Property'
        ))
        
        # Add wells
        for well in processed_data['wells']:
            color = colors['success'] if well['type'] == 'Producer' else colors['primary'] if well['type'] == 'Injector' else colors['text-secondary']
            
            fig.add_trace(go.Scatter3d(
                x=[well['x'], well['x']],
                y=[well['y'], well['y']],
                z=[well['z_top'], well['z_bottom']],
                mode='lines+markers',
                line=dict(color=color, width=5),
                marker=dict(size=8),
                name=well['well_id'],
                text=[f"{well['well_id']} - {well['type']}", ''],
                hovertemplate='%{text}<extra></extra>'
            ))
        
        # Add recommended well locations
        for i, rec in enumerate(processed_data['recommendations']):
            fig.add_trace(go.Scatter3d(
                x=[rec['x']],
                y=[rec['y']],
                z=[2900],  # Average depth
                mode='markers',
                marker=dict(size=10, color=colors['warning'], symbol='diamond'),
                name=f'Recommended #{i+1}',
                text=[f"Score: {rec['quality_score']:.2f}"],
                hovertemplate='Recommended Location<br>%{text}<extra></extra>'
            ))
        
        fig.update_layout(
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Depth (m)',
                zaxis=dict(autorange='reversed'),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=600,
            plot_bgcolor=colors['surface'],
            paper_bgcolor=colors['surface'],
            font=dict(color=colors['text']),
            title="3D Reservoir Model with Well Locations"
        )
        
        return dcc.Graph(figure=fig)
    
    elif active_tab == 'production':
        # Create production analysis plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Field Oil Production', 'Water Cut Trend', 
                          'GOR Evolution', 'Cumulative Production'),
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': True}]]
        )
        
        # Aggregate production data
        producer_wells = [w for w in processed_data['wells'] if w['type'] == 'Producer']
        
        if producer_wells:
            time_series = pd.to_datetime(producer_wells[0]['time_series'])
            
            # Field oil production
            total_oil = np.sum([w['oil_rate'] for w in producer_wells], axis=0)
            fig.add_trace(
                go.Scatter(x=time_series, y=total_oil, name='Oil Rate',
                          line=dict(color=colors['success'], width=2),
                          fill='tozeroy', fillcolor='rgba(0, 255, 136, 0.1)'),
                row=1, col=1
            )
            
            # Water cut trend
            total_water = np.sum([w['water_rate'] for w in producer_wells], axis=0)
            field_wc = total_water / (total_oil + total_water) * 100
            fig.add_trace(
                go.Scatter(x=time_series, y=field_wc, name='Water Cut',
                          line=dict(color=colors['warning'], width=2)),
                row=1, col=2
            )
            
            # GOR evolution
            total_gas = np.sum([w['gas_rate'] for w in producer_wells], axis=0)
            field_gor = total_gas / np.maximum(total_oil, 1) * 1000  # Avoid division by zero
            fig.add_trace(
                go.Scatter(x=time_series, y=field_gor, name='GOR',
                          line=dict(color=colors['danger'], width=2)),
                row=2, col=1
            )
            
            # Cumulative production
            cum_oil = np.cumsum(total_oil)
            cum_water = np.cumsum(total_water)
            
            fig.add_trace(
                go.Scatter(x=time_series, y=cum_oil/1000, name='Cum Oil (Mbbl)',
                          line=dict(color=colors['success'], width=2)),
                row=2, col=2, secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=time_series, y=cum_water/1000, name='Cum Water (Mbbl)',
                          line=dict(color=colors['primary'], width=2, dash='dash')),
                row=2, col=2, secondary_y=True
            )
        
        fig.update_xaxes(title='Date', gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
        
        fig.update_yaxes(title='Oil Rate (bbl/d)', row=1, col=1)
        fig.update_yaxes(title='Water Cut (%)', row=1, col=2)
        fig.update_yaxes(title='GOR (scf/bbl)', row=2, col=1)
        fig.update_yaxes(title='Oil (Mbbl)', row=2, col=2, secondary_y=False)
        fig.update_yaxes(title='Water (Mbbl)', row=2, col=2, secondary_y=True)
        
        fig.update_layout(
            height=600,
            showlegend=True,
            plot_bgcolor=colors['surface'],
            paper_bgcolor=colors['surface'],
            font=dict(color=colors['text'])
        )
        
        return dcc.Graph(figure=fig)
    
    elif active_tab == 'wells':
        # Create well performance bubble plot
        wells = processed_data['wells']
        
        fig = go.Figure()
        
        for well in wells:
            if well['type'] == 'Producer' and len(well['oil_rate']) > 0:
                current_rate = well['oil_rate'][-1] if well['oil_rate'] else 0
                avg_rate = np.mean(well['oil_rate']) if well['oil_rate'] else 0
                
                color = colors['success'] if current_rate > avg_rate else colors['warning']
                
                fig.add_trace(go.Scatter(
                    x=[well['x']],
                    y=[well['y']],
                    mode='markers+text',
                    marker=dict(
                        size=max(current_rate/10, 5),  # Minimum size of 5
                        color=color,
                        opacity=0.6,
                        line=dict(color='white', width=1)
                    ),
                    text=[well['well_id']],
                    textposition='top center',
                    name=well['well_id'],
                    hovertemplate=f"{well['well_id']}<br>Rate: {current_rate:.0f} bbl/d<br>Type: {well['type']}<extra></extra>"
                ))
            elif well['type'] == 'Injector':
                fig.add_trace(go.Scatter(
                    x=[well['x']],
                    y=[well['y']],
                    mode='markers+text',
                    marker=dict(
                        size=20,
                        color=colors['primary'],
                        symbol='square',
                        opacity=0.6,
                        line=dict(color='white', width=1)
                    ),
                    text=[well['well_id']],
                    textposition='top center',
                    name=well['well_id'],
                    hovertemplate=f"{well['well_id']}<br>Type: {well['type']}<extra></extra>"
                ))
        
        # Add recommended locations
        for i, rec in enumerate(processed_data['recommendations']):
            fig.add_trace(go.Scatter(
                x=[rec['x']],
                y=[rec['y']],
                mode='markers',
                marker=dict(
                    size=25,
                    color=colors['warning'],
                    symbol='star',
                    line=dict(color='white', width=2)
                ),
                name=f'Recommended #{i+1}',
                hovertemplate=f"Recommended Location #{i+1}<br>Est. Rate: {rec['estimated_rate']:.0f} bbl/d<extra></extra>"
            ))
        
        fig.update_layout(
            title="Well Performance Map (Bubble size = Oil Rate)",
            xaxis_title='X Coordinate (m)',
            yaxis_title='Y Coordinate (m)',
            height=600,
            showlegend=False,
            plot_bgcolor=colors['surface'],
            paper_bgcolor=colors['surface'],
            font=dict(color=colors['text'])
        )
        
        return dcc.Graph(figure=fig)
    
    elif active_tab == 'optimization':
        # Create optimization plan visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Recovery Factor Scenarios', 'NPV Analysis',
                          'Risk Matrix', 'Development Timeline')
        )
        
        # Recovery factor scenarios
        years = np.arange(0, 11)
        base_rf = processed_data['metadata']['current_rf']
        
        scenarios = {
            'Base Case': base_rf + years * 0.01,
            'Optimized': base_rf + years * 0.025,
            'Aggressive': base_rf + years * 0.035
        }
        
        for scenario, values in scenarios.items():
            fig.add_trace(
                go.Scatter(x=years, y=values*100, name=scenario,
                          mode='lines+markers'),
                row=1, col=1
            )
        
        # NPV Analysis
        npv_categories = ['Base Case', 'Add 3 Wells', 'Add 5 Wells', 'EOR']
        npv_values = [100, 150, 180, 220]
        
        fig.add_trace(
            go.Bar(x=npv_categories, y=npv_values,
                  marker=dict(color=[colors['primary'], colors['success'], 
                                   colors['warning'], colors['secondary']]),
                  text=[f'${v}M' for v in npv_values],
                  textposition='outside'),
            row=1, col=2
        )
        
        # Risk Matrix
        risk_matrix = np.random.rand(4, 4) * 100
        fig.add_trace(
            go.Heatmap(
                z=risk_matrix,
                colorscale=[[0, colors['success']], [0.5, colors['warning']], [1, colors['danger']]],
                showscale=False,
                text=np.round(risk_matrix, 0),
                texttemplate='%{text}',
                textfont={"size": 10}
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            plot_bgcolor=colors['surface'],
            paper_bgcolor=colors['surface'],
            font=dict(color=colors['text'])
        )
        
        return dcc.Graph(figure=fig)

# Include all other callbacks from the original code...
# [Rest of callbacks remain identical]

@app.callback(
    Output('property-histogram', 'figure'),
    [Input('processed-data-store', 'data')]
)
def update_property_histogram(processed_data):
    if not processed_data:
        return go.Figure()
    
    values = np.array(processed_data['grid']['property_values']).flatten()
    
    fig = go.Figure(data=[
        go.Histogram(
            x=values,
            nbinsx=30,
            marker=dict(color=colors['primary'], line=dict(color=colors['text'], width=1))
        )
    ])
    
    fig.update_layout(
        xaxis_title=processed_data['property_name'].replace('_', ' ').title(),
        yaxis_title='Frequency',
        height=200,
        margin=dict(t=10, b=40, l=40, r=10),
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['surface'],
        font=dict(color=colors['text'])
    )
    
    return fig

@app.callback(
    Output('recovery-forecast', 'figure'),
    [Input('processed-data-store', 'data')]
)
def update_recovery_forecast(processed_data):
    if not processed_data:
        return go.Figure()
    
    # Generate forecast
    months = np.arange(0, 25)
    current_rf = processed_data['metadata']['current_rf']
    
    # Different decline scenarios
    optimistic = current_rf + months * 0.003
    expected = current_rf + months * 0.002
    pessimistic = current_rf + months * 0.001
    
    fig = go.Figure()
    
    # Add confidence bands
    fig.add_trace(go.Scatter(
        x=months, y=optimistic*100,
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=months, y=pessimistic*100,
        fill='tonexty',
        fillcolor='rgba(0, 212, 255, 0.2)',
        line=dict(width=0),
        name='Confidence Band'
    ))
    
    fig.add_trace(go.Scatter(
        x=months, y=expected*100,
        mode='lines',
        line=dict(color=colors['primary'], width=2),
        name='Expected RF'
    ))
    
    # Add target line
    fig.add_hline(
        y=processed_data['metadata']['target_rf']*100,
        line_dash="dash",
        line_color=colors['success'],
        annotation_text="Target RF"
    )
    
    fig.update_layout(
        xaxis_title='Months Ahead',
        yaxis_title='Recovery Factor (%)',
        height=200,
        margin=dict(t=10, b=40, l=40, r=10),
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['surface'],
        font=dict(color=colors['text']),
        showlegend=True,
        legend=dict(x=0, y=1)
    )
    
    return fig

@app.callback(
    Output('field-map', 'figure'),
    [Input('processed-data-store', 'data')]
)
def update_field_map(processed_data):
    if not processed_data:
        return go.Figure()
    
    # Create 2D field map with contours
    X = np.array(processed_data['grid']['X'])[:,:,0]
    Y = np.array(processed_data['grid']['Y'])[:,:,0]
    values = np.mean(np.array(processed_data['grid']['property_values']), axis=2)
    
    fig = go.Figure()
    
    # Add contour plot
    fig.add_trace(go.Contour(
        x=X[0,:],
        y=Y[:,0],
        z=values,
        colorscale='Viridis',
        showscale=False,
        contours=dict(showlabels=True, labelfont=dict(size=9, color='white'))
    ))
    
    # Add wells
    for well in processed_data['wells']:
        if well['type'] == 'Producer':
            color = colors['success']
            symbol = 'circle'
        elif well['type'] == 'Injector':
            color = colors['primary']
            symbol = 'square'
        else:
            color = colors['text-secondary']
            symbol = 'diamond'
        
        fig.add_trace(go.Scatter(
            x=[well['x']],
            y=[well['y']],
            mode='markers',
            marker=dict(size=10, color=color, symbol=symbol, line=dict(color='white', width=1)),
            name=well['well_id'],
            showlegend=False,
            hovertemplate=f"{well['well_id']}<extra></extra>"
        ))
    
    fig.update_layout(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        height=250,
        margin=dict(t=10, b=40, l=40, r=10),
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['surface'],
        font=dict(color=colors['text'])
    )
    
    return fig

@app.callback(
    Output('recommendations-table', 'children'),
    [Input('processed-data-store', 'data')]
)
def update_recommendations_table(processed_data):
    if not processed_data:
        return html.Div()
    
    recommendations = []
    
    # Well placement recommendations
    for i, rec in enumerate(processed_data['recommendations'][:3]):
        recommendations.append({
            'Action': f'Drill Producer #{i+1}',
            'Location': f"({rec['x']:.0f}, {rec['y']:.0f})",
            'Est. Rate': f"{rec['estimated_rate']:.0f} bbl/d",
            'Confidence': f"{rec['confidence']}%",
            'Priority': 'High' if rec['confidence'] > 90 else 'Medium',
            'Timeline': f'Q{i+1} 2024'
        })
    
    # Operational recommendations
    if processed_data['metrics']['field_water_cut'] > 70:
        recommendations.append({
            'Action': 'Workover High WC Wells',
            'Location': 'Field-wide',
            'Est. Rate': '+200 bbl/d',
            'Confidence': '75%',
            'Priority': 'High',
            'Timeline': 'Q1 2024'
        })
    
    if processed_data['metrics']['voidage_replacement'] < 0.9:
        recommendations.append({
            'Action': 'Increase Injection',
            'Location': 'Flank Injectors',
            'Est. Rate': 'Pressure Support',
            'Confidence': '85%',
            'Priority': 'Medium',
            'Timeline': 'Immediate'
        })
    
    if not recommendations:
        return html.P("No recommendations available", style={'color': colors['text-secondary']})
    
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
                'if': {'column_id': 'Priority', 'filter_query': '{Priority} = "High"'},
                'backgroundColor': colors['danger'],
                'color': 'white',
            },
            {
                'if': {'column_id': 'Priority', 'filter_query': '{Priority} = "Medium"'},
                'backgroundColor': colors['warning'],
                'color': 'white',
            }
        ],
        sort_action="native"
    )

@app.callback(
    Output('economics-chart', 'figure'),
    [Input('processed-data-store', 'data')]
)
def update_economics_chart(processed_data):
    if not processed_data:
        return go.Figure()
    
    # Create economics waterfall chart
    categories = ['Current Revenue', 'New Wells', 'Workovers', 'Operating Costs', 
                  'Capital Costs', 'Net Revenue']
    values = [150, 50, 20, -40, -30, 150]
    
    fig = go.Figure(go.Waterfall(
        name="Economics",
        orientation="v",
        measure=["relative", "relative", "relative", "relative", "relative", "total"],
        x=categories,
        text=[f"${v}M" for v in values],
        y=values,
        connector={"line": {"color": colors['text-secondary']}},
        increasing={"marker": {"color": colors['success']}},
        decreasing={"marker": {"color": colors['danger']}},
        totals={"marker": {"color": colors['primary']}}
    ))
    
    fig.update_layout(
        title="Project Economics (Next 5 Years)",
        yaxis_title="Value ($MM)",
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
    app.run(debug=False, port=8055, host='127.0.0.1')