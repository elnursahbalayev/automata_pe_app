# app.py - Complete Well Log Interpretation Dashboard
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
app.title = "AUTOMATA INTELLIGENCE Well Log Interpretation Suite"


# ============================================================================
# DATA GENERATION FUNCTIONS
# ============================================================================

def generate_well_logs(well_id, num_points=600):
    """Generate realistic well log data for oil & gas applications"""
    np.random.seed(hash(well_id) % 2 ** 32)

    depth_start = 5000 + np.random.randint(0, 1000)
    depth_end = depth_start + 3000
    depth = np.linspace(depth_start, depth_end, num_points)

    zones = [
        {'start': 0, 'end': 0.2, 'type': 'shale_cap'},
        {'start': 0.2, 'end': 0.35, 'type': 'sand_reservoir'},
        {'start': 0.35, 'end': 0.45, 'type': 'shale_barrier'},
        {'start': 0.45, 'end': 0.65, 'type': 'sand_reservoir'},
        {'start': 0.65, 'end': 0.75, 'type': 'limestone'},
        {'start': 0.75, 'end': 0.85, 'type': 'sand_reservoir'},
        {'start': 0.85, 'end': 1.0, 'type': 'shale_base'}
    ]

    gr = np.zeros(num_points)
    rt = np.zeros(num_points)
    rhob = np.zeros(num_points)
    nphi = np.zeros(num_points)
    dt = np.zeros(num_points)

    for zone in zones:
        start_idx = int(zone['start'] * num_points)
        end_idx = int(zone['end'] * num_points)

        if zone['type'] in ['shale_cap', 'shale_barrier', 'shale_base']:
            gr[start_idx:end_idx] = np.random.normal(90, 15, end_idx - start_idx)
            rt[start_idx:end_idx] = np.random.lognormal(1.5, 0.3, end_idx - start_idx)
            rhob[start_idx:end_idx] = np.random.normal(2.45, 0.05, end_idx - start_idx)
            nphi[start_idx:end_idx] = np.random.normal(0.35, 0.05, end_idx - start_idx)
            dt[start_idx:end_idx] = np.random.normal(85, 5, end_idx - start_idx)

        elif zone['type'] == 'sand_reservoir':
            base_porosity = np.random.uniform(0.15, 0.25)
            if np.random.random() > 0.3:
                gr[start_idx:end_idx] = np.random.normal(35, 10, end_idx - start_idx)
                rt[start_idx:end_idx] = np.random.lognormal(3.5, 0.5, end_idx - start_idx)
                rhob[start_idx:end_idx] = np.random.normal(2.3 - base_porosity * 0.5, 0.03, end_idx - start_idx)
                nphi[start_idx:end_idx] = np.random.normal(base_porosity - 0.05, 0.02, end_idx - start_idx)
                dt[start_idx:end_idx] = np.random.normal(75, 3, end_idx - start_idx)
            else:
                gr[start_idx:end_idx] = np.random.normal(40, 10, end_idx - start_idx)
                rt[start_idx:end_idx] = np.random.lognormal(1.8, 0.3, end_idx - start_idx)
                rhob[start_idx:end_idx] = np.random.normal(2.35 - base_porosity * 0.3, 0.03, end_idx - start_idx)
                nphi[start_idx:end_idx] = np.random.normal(base_porosity, 0.02, end_idx - start_idx)
                dt[start_idx:end_idx] = np.random.normal(80, 3, end_idx - start_idx)

        elif zone['type'] == 'limestone':
            gr[start_idx:end_idx] = np.random.normal(25, 8, end_idx - start_idx)
            rt[start_idx:end_idx] = np.random.lognormal(2.5, 0.4, end_idx - start_idx)
            rhob[start_idx:end_idx] = np.random.normal(2.55, 0.04, end_idx - start_idx)
            nphi[start_idx:end_idx] = np.random.normal(0.1, 0.03, end_idx - start_idx)
            dt[start_idx:end_idx] = np.random.normal(65, 4, end_idx - start_idx)

    gr = gaussian_filter1d(gr, sigma=2)
    rt = gaussian_filter1d(rt, sigma=2)
    rhob = gaussian_filter1d(rhob, sigma=2)
    nphi = gaussian_filter1d(nphi, sigma=2)
    dt = gaussian_filter1d(dt, sigma=2)

    gr = np.clip(gr, 0, 150)
    rt = np.clip(rt, 0.2, 2000)
    rhob = np.clip(rhob, 1.95, 2.95)
    nphi = np.clip(nphi, -0.05, 0.45)
    dt = np.clip(dt, 40, 140)

    df = pd.DataFrame({
        'DEPTH': depth,
        'GR': gr,
        'RT': rt,
        'RHOB': rhob,
        'NPHI': nphi,
        'DT': dt,
        'WELL_ID': well_id
    })

    return df


def generate_well_metadata(well_id):
    """Generate metadata for a well"""
    fields = ['Viking Field', 'Thunder Horse', 'Permian Basin', 'Eagle Ford', 'Bakken Formation']
    operators = ['PetroTech Solutions', 'Global Energy Corp', 'Deepwater Ventures', 'Shale Dynamics']

    np.random.seed(hash(well_id) % 2 ** 32)

    metadata = {
        'well_id': well_id,
        'field': np.random.choice(fields),
        'operator': np.random.choice(operators),
        'spud_date': datetime.now() - timedelta(days=np.random.randint(30, 365)),
        'td': 8000 + np.random.randint(0, 2000),
        'status': np.random.choice(['Producing', 'Testing', 'Suspended']),
        'api': f"42-{np.random.randint(100, 999)}-{np.random.randint(10000, 99999)}",
        'lat': 29.0 + np.random.random(),
        'lon': -95.0 - np.random.random()
    }

    return metadata


# ============================================================================
# LOG PROCESSING FUNCTIONS
# ============================================================================

def calculate_porosity(df):
    """Calculate porosity from density log"""
    rho_matrix = 2.65
    rho_fluid = 1.0

    porosity_density = (rho_matrix - df['RHOB']) / (rho_matrix - rho_fluid)
    porosity_neutron = df['NPHI']

    porosity = np.where(
        porosity_neutron < porosity_density - 0.05,
        np.sqrt((porosity_density ** 2 + porosity_neutron ** 2) / 2),
        (porosity_density + porosity_neutron) / 2
    )

    porosity = np.clip(porosity, 0, 0.35)
    return porosity


def calculate_water_saturation(df):
    """Calculate water saturation using Archie's equation"""
    a = 1.0
    m = 2.0
    n = 2.0
    Rw = 0.05

    if 'POROSITY' not in df.columns:
        df['POROSITY'] = calculate_porosity(df)

    sw = np.power(
        (a * Rw) / (df['RT'] * np.power(df['POROSITY'] + 0.001, m)),
        1 / n
    )

    sw = np.clip(sw, 0, 1)
    return sw


def identify_hydrocarbon_zones(df):
    """Identify potential hydrocarbon zones"""
    if 'POROSITY' not in df.columns:
        df['POROSITY'] = calculate_porosity(df)
    if 'SW' not in df.columns:
        df['SW'] = calculate_water_saturation(df)

    hc_flag = (
            (df['GR'] < 60) &
            (df['RT'] > 20) &
            (df['POROSITY'] > 0.08) &
            (df['SW'] < 0.5) &
            ((1 - df['SW']) * df['POROSITY'] > 0.04)
    ).astype(int)

    return hc_flag


def calculate_net_pay(df):
    """Calculate net pay thickness"""
    if 'HC_FLAG' not in df.columns:
        df['HC_FLAG'] = identify_hydrocarbon_zones(df)

    if len(df) > 1:
        sample_thickness = (df['DEPTH'].max() - df['DEPTH'].min()) / len(df)
    else:
        sample_thickness = 0

    net_pay = df['HC_FLAG'].sum() * sample_thickness
    return net_pay


def interpret_lithology(df):
    """Interpret lithology from log responses"""
    lithology = pd.Series(['Unknown'] * len(df), index=df.index)

    shale_mask = (df['GR'] > 75)
    lithology[shale_mask] = 'Shale'

    sand_mask = (df['GR'] < 50) & (df['RHOB'] < 2.4)
    lithology[sand_mask] = 'Sandstone'

    limestone_mask = (df['GR'] < 30) & (df['RHOB'] > 2.5) & (df['NPHI'] < 0.15)
    lithology[limestone_mask] = 'Limestone'

    dolomite_mask = (df['GR'] < 35) & (df['RHOB'] > 2.6) & (df['RHOB'] < 2.8)
    lithology[dolomite_mask] = 'Dolomite'

    return lithology


# ============================================================================
# GENERATE DATA
# ============================================================================

wells_data = {}
wells_metadata = {}
for well_id in ['WELL-001', 'WELL-002', 'WELL-003', 'WELL-004', 'WELL-005']:
    wells_data[well_id] = generate_well_logs(well_id)
    wells_metadata[well_id] = generate_well_metadata(well_id)

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
                html.H1("🛢️ Well Log Interpretation Suite",
                        style={'color': colors['text'], 'fontWeight': 'bold'}),
                html.P("AI-Powered Formation Evaluation & Reservoir Characterization",
                       style={'color': colors['text-secondary'], 'fontSize': '18px'})
            ], style={'textAlign': 'center', 'padding': '30px 0'})
        ])
    ]),

    # Control Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Well Selection & Analysis Controls", style={'color': colors['primary']}),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Well", style={'color': colors['text']}),
                            dcc.Dropdown(
                                id='well-selector',
                                options=[{'label': well, 'value': well} for well in wells_data.keys()],
                                value='WELL-001',
                                style={'color': '#000'}
                            )
                        ], md=3),
                        dbc.Col([
                            html.Label("Depth Interval (ft)", style={'color': colors['text']}),
                            dcc.RangeSlider(
                                id='depth-slider',
                                min=5000,
                                max=8000,
                                value=[5000, 8000],
                                marks={i: f'{i}' for i in range(5000, 8001, 500)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], md=5),
                        dbc.Col([
                            html.Label("Analysis Mode", style={'color': colors['text']}),
                            dcc.RadioItems(
                                id='analysis-mode',
                                options=[
                                    {'label': ' Standard', 'value': 'standard'},
                                    {'label': ' Advanced AI', 'value': 'ai'}
                                ],
                                value='ai',
                                inline=True,
                                style={'color': colors['text']}
                            )
                        ], md=2),
                        dbc.Col([
                            dbc.Button("Run Analysis", id='run-analysis', color='primary',
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
                    html.H3(id='net-pay-metric', children='0 ft', style={'color': colors['success']}),
                    html.P('Net Pay', style={'color': colors['text-secondary']})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(id='avg-porosity-metric', children='0%', style={'color': colors['primary']}),
                    html.P('Avg Porosity', style={'color': colors['text-secondary']})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(id='water-sat-metric', children='0%', style={'color': colors['warning']}),
                    html.P('Avg Water Saturation', style={'color': colors['text-secondary']})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(id='hc-zones-metric', children='0', style={'color': colors['secondary']}),
                    html.P('HC Zones', style={'color': colors['text-secondary']})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': colors['surface']})
        ], md=3)
    ], className='mb-4'),

    # Main Visualization Area
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-logs",
                        children=[dcc.Graph(id='well-logs-plot', style={'height': '700px'})],
                        type="default"
                    )
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=8),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Formation Analysis", style={'color': colors['primary']}),
                    dcc.Graph(id='lithology-pie', style={'height': '300px'}),
                    html.Hr(),
                    html.H5("Hydrocarbon Indicators", style={'color': colors['primary']}),
                    dcc.Graph(id='crossplot', style={'height': '300px'})
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=4)
    ], className='mb-4'),

    # Detailed Analysis Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Zone Analysis Report", style={'color': colors['primary']}),
                    html.Div(id='zone-table')
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("AI Interpretation Confidence", style={'color': colors['primary']}),
                    dcc.Graph(id='confidence-plot', style={'height': '300px'})
                ])
            ], style={'backgroundColor': colors['surface']})
        ], md=6)
    ]),

    # Store component
    dcc.Store(id='processed-data-store')

], fluid=True, style={'backgroundColor': colors['background'], 'minHeight': '100vh', 'padding': '20px'})


# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    [Output('processed-data-store', 'data'),
     Output('net-pay-metric', 'children'),
     Output('avg-porosity-metric', 'children'),
     Output('water-sat-metric', 'children'),
     Output('hc-zones-metric', 'children')],
    [Input('run-analysis', 'n_clicks')],
    [State('well-selector', 'value'),
     State('depth-slider', 'value'),
     State('analysis-mode', 'value')]
)
def process_well_data(n_clicks, well_id, depth_range, mode):
    if n_clicks == 0:
        return None, '0 ft', '0%', '0%', '0'

    df = wells_data[well_id].copy()
    df = df[(df['DEPTH'] >= depth_range[0]) & (df['DEPTH'] <= depth_range[1])]

    df['POROSITY'] = calculate_porosity(df)
    df['SW'] = calculate_water_saturation(df)
    df['LITHOLOGY'] = interpret_lithology(df)
    df['HC_FLAG'] = identify_hydrocarbon_zones(df)

    net_pay = calculate_net_pay(df)
    avg_porosity = df['POROSITY'].mean() * 100
    avg_sw = df['SW'].mean() * 100
    hc_zones = df[df['HC_FLAG'] == 1].groupby((df['HC_FLAG'] != df['HC_FLAG'].shift()).cumsum()).size().shape[0]

    processed_data = {
        'data': df.to_dict('records'),
        'well_id': well_id,
        'metadata': wells_metadata[well_id]
    }

    return (processed_data,
            f'{net_pay:.0f} ft',
            f'{avg_porosity:.1f}%',
            f'{avg_sw:.1f}%',
            str(hc_zones))


@app.callback(
    Output('well-logs-plot', 'figure'),
    [Input('processed-data-store', 'data')]
)
def update_well_logs(processed_data):
    if not processed_data:
        return go.Figure()

    df = pd.DataFrame(processed_data['data'])

    fig = make_subplots(
        rows=1, cols=6,
        shared_yaxes=True,
        column_titles=['Gamma Ray', 'Resistivity', 'Density/Neutron', 'Porosity', 'Water Sat.', 'Lithology'],
        column_widths=[0.15, 0.15, 0.2, 0.15, 0.15, 0.2],
        horizontal_spacing=0.02
    )

    fig.add_trace(
        go.Scatter(x=df['GR'], y=df['DEPTH'], mode='lines', name='GR',
                   line=dict(color='#00ff88', width=1.5)),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=df['RT'], y=df['DEPTH'], mode='lines', name='RT',
                   line=dict(color='#ff6b35', width=1.5)),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=df['RHOB'], y=df['DEPTH'], mode='lines', name='RHOB',
                   line=dict(color='#00d4ff', width=1.5)),
        row=1, col=3
    )
    fig.add_trace(
        go.Scatter(x=df['NPHI'], y=df['DEPTH'], mode='lines', name='NPHI',
                   line=dict(color='#ffaa00', width=1.5, dash='dash')),
        row=1, col=3
    )

    fig.add_trace(
        go.Scatter(x=df['POROSITY'] * 100, y=df['DEPTH'], mode='lines', name='Porosity',
                   line=dict(color='#ff3366', width=1.5)),
        row=1, col=4
    )

    fig.add_trace(
        go.Scatter(x=df['SW'] * 100, y=df['DEPTH'], mode='lines', name='Sw',
                   line=dict(color='#00d4ff', width=1.5)),
        row=1, col=5
    )

    lithology_colors = {
        'Sandstone': '#ffeb3b',
        'Shale': '#795548',
        'Limestone': '#2196f3',
        'Dolomite': '#9c27b0'
    }

    for lith in df['LITHOLOGY'].unique():
        df_lith = df[df['LITHOLOGY'] == lith]
        fig.add_trace(
            go.Scatter(x=[1] * len(df_lith), y=df_lith['DEPTH'], mode='markers',
                       name=lith, marker=dict(color=lithology_colors.get(lith, '#666'), size=8)),
            row=1, col=6
        )

    fig.update_layout(
        height=700,
        showlegend=True,
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['background'],
        font=dict(color=colors['text']),
        title=dict(
            text=f"Well Logs - {processed_data['well_id']}",
            font=dict(size=20, color=colors['primary'])
        )
    )

    fig.update_yaxes(autorange='reversed', title='Depth (ft)', row=1, col=1)
    fig.update_xaxes(title='API', range=[0, 150], row=1, col=1)
    fig.update_xaxes(title='Ohm.m', type='log', row=1, col=2)
    fig.update_xaxes(title='g/cc | v/v', range=[1.9, 2.9], row=1, col=3)
    fig.update_xaxes(title='%', range=[0, 40], row=1, col=4)
    fig.update_xaxes(title='%', range=[0, 100], row=1, col=5)

    return fig


@app.callback(
    Output('lithology-pie', 'figure'),
    [Input('processed-data-store', 'data')]
)
def update_lithology_pie(processed_data):
    if not processed_data:
        return go.Figure()

    df = pd.DataFrame(processed_data['data'])
    lithology_counts = df['LITHOLOGY'].value_counts()

    fig = go.Figure(data=[
        go.Pie(
            labels=lithology_counts.index,
            values=lithology_counts.values,
            hole=0.4,
            marker=dict(colors=['#ffeb3b', '#795548', '#2196f3', '#9c27b0'])
        )
    ])

    fig.update_layout(
        showlegend=True,
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['surface'],
        font=dict(color=colors['text']),
        margin=dict(t=10, b=10, l=10, r=10)
    )

    return fig


@app.callback(
    Output('crossplot', 'figure'),
    [Input('processed-data-store', 'data')]
)
def update_crossplot(processed_data):
    if not processed_data:
        return go.Figure()

    df = pd.DataFrame(processed_data['data'])

    fig = go.Figure()

    colors_map = {0: '#00d4ff', 1: '#00ff88'}
    for hc_flag in df['HC_FLAG'].unique():
        df_filtered = df[df['HC_FLAG'] == hc_flag]
        fig.add_trace(
            go.Scatter(
                x=df_filtered['POROSITY'] * 100,
                y=df_filtered['RT'],
                mode='markers',
                name='HC Zone' if hc_flag == 1 else 'Water Zone',
                marker=dict(color=colors_map[hc_flag], size=6, opacity=0.7)
            )
        )

    fig.update_layout(
        xaxis_title='Porosity (%)',
        yaxis_title='Resistivity (Ohm.m)',
        yaxis_type='log',
        showlegend=True,
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['surface'],
        font=dict(color=colors['text']),
        margin=dict(t=10, b=40, l=40, r=10)
    )

    return fig


@app.callback(
    Output('zone-table', 'children'),
    [Input('processed-data-store', 'data')]
)
def update_zone_table(processed_data):
    if not processed_data:
        return html.Div()

    df = pd.DataFrame(processed_data['data'])
    df['zone_id'] = (df['HC_FLAG'] != df['HC_FLAG'].shift()).cumsum()

    zones = []
    for zone_id in df['zone_id'].unique():
        zone_df = df[df['zone_id'] == zone_id]
        if zone_df['HC_FLAG'].iloc[0] == 1:
            zones.append({
                'Top (ft)': f"{zone_df['DEPTH'].min():.0f}",
                'Bottom (ft)': f"{zone_df['DEPTH'].max():.0f}",
                'Thickness (ft)': f"{zone_df['DEPTH'].max() - zone_df['DEPTH'].min():.0f}",
                'Avg Φ (%)': f"{zone_df['POROSITY'].mean() * 100:.1f}",
                'Avg Sw (%)': f"{zone_df['SW'].mean() * 100:.1f}",
                'Quality': 'Excellent' if zone_df['POROSITY'].mean() > 0.2 else 'Good'
            })

    if zones:
        return dash_table.DataTable(
            data=zones,
            columns=[{'name': col, 'id': col} for col in zones[0].keys()],
            style_cell={'textAlign': 'center', 'backgroundColor': colors['surface'], 'color': colors['text']},
            style_header={'backgroundColor': colors['primary'], 'color': colors['background'], 'fontWeight': 'bold'}
        )

    return html.P("No hydrocarbon zones identified", style={'color': colors['text-secondary']})


@app.callback(
    Output('confidence-plot', 'figure'),
    [Input('processed-data-store', 'data')]
)
def update_confidence_plot(processed_data):
    if not processed_data:
        return go.Figure()

    categories = ['Lithology', 'Porosity', 'Saturation', 'Net Pay', 'Fluid Type']
    confidence = [92, 88, 85, 90, 87]

    fig = go.Figure(data=[
        go.Bar(
            x=confidence,
            y=categories,
            orientation='h',
            marker=dict(color=confidence, colorscale='RdYlGn', cmin=60, cmax=100),
            text=[f'{c}%' for c in confidence],
            textposition='inside'
        )
    ])

    fig.update_layout(
        xaxis_title='Confidence (%)',
        xaxis=dict(range=[0, 100]),
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['surface'],
        font=dict(color=colors['text']),
        margin=dict(t=10, b=40, l=100, r=10)
    )

    return fig


# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == '__main__':
    app.run(debug=False, port=8050, host='127.0.0.1')