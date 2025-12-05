# app.py - Complete Well Log Interpretation Dashboard with Upload & Chat (Mantine Version)
import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import dash_mantine_components as dmc
from dash_iconify import DashIconify
from scipy.ndimage import gaussian_filter1d
import base64
import random

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
    ]
)
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
# AI CHATBOT RESPONSES
# ============================================================================

AI_RESPONSES = {
    'default': [
        "Based on my analysis of the well logs, I can see several interesting features. The gamma ray log shows clear sand-shale sequences with good reservoir quality in the sand intervals.",
        "I've identified multiple pay zones in this well. The primary zone between 5,400-5,650 ft shows excellent porosity (18-22%) and low water saturation (<35%).",
        "The resistivity patterns suggest hydrocarbon presence in at least 3 distinct intervals. Would you like me to provide a detailed breakdown of each zone?",
        "Looking at the density-neutron crossover, I can confirm gas effect in the upper reservoir section. This is a positive indicator for production potential.",
    ],
    'porosity': [
        "The average porosity in the pay zones is approximately 19.5%. This is calculated using a density-neutron combination to account for gas effect and shale content.",
        "Porosity distribution shows: Zone 1: 18.2%, Zone 2: 21.4%, Zone 3: 17.8%. The higher porosity in Zone 2 correlates with cleaner sand facies.",
    ],
    'saturation': [
        "Water saturation analysis using Archie's equation (a=1, m=2, n=2) indicates Sw ranging from 25-45% in the reservoir intervals.",
        "The hydrocarbon saturation (1-Sw) averages 62% across the pay zones, suggesting excellent movable oil/gas volumes.",
    ],
    'lithology': [
        "Lithology interpretation shows a classic deltaic sequence: shale cap → channel sands → marine shale → stacked bar sands → carbonate platform → basal shale.",
        "The GR and density logs indicate predominantly clean sandstone reservoirs with minimal clay content (<15% Vshale).",
    ],
    'recommendation': [
        "Based on my analysis, I recommend: 1) Perforate zones at 5,420-5,480 ft and 5,680-5,750 ft, 2) Consider acid stimulation for the limestone interval, 3) Run production logs to confirm fluid contacts.",
        "This well shows strong production potential. Estimated recoverable reserves: 450-600 MBOE. Recommended completion: Multi-stage frac with 4-6 stages targeting the main sand bodies.",
    ],
    'help': [
        "I can help you with: \n• Well log interpretation\n• Porosity & saturation calculations\n• Lithology identification\n• Pay zone analysis\n• Production recommendations\n\nJust ask me anything about your well data!",
    ]
}


def get_ai_response(user_message):
    """Generate contextual AI response based on user message"""
    message_lower = user_message.lower()

    if any(word in message_lower for word in ['porosity', 'phi', 'pore']):
        return random.choice(AI_RESPONSES['porosity'])
    elif any(word in message_lower for word in ['saturation', 'water', 'sw', 'hydrocarbon']):
        return random.choice(AI_RESPONSES['saturation'])
    elif any(word in message_lower for word in ['lithology', 'rock', 'sand', 'shale', 'formation']):
        return random.choice(AI_RESPONSES['lithology'])
    elif any(word in message_lower for word in ['recommend', 'suggest', 'should', 'what to do', 'next step']):
        return random.choice(AI_RESPONSES['recommendation'])
    elif any(word in message_lower for word in ['help', 'what can you', 'how to', 'guide']):
        return random.choice(AI_RESPONSES['help'])
    else:
        return random.choice(AI_RESPONSES['default'])


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
# HELPER FUNCTIONS FOR ICONS
# ============================================================================

def get_icon(icon_name, size=20, color=None):
    """Helper function to create DashIconify icons"""
    return DashIconify(icon=icon_name, width=size, height=size, color=color)


# ============================================================================
# CHAT COMPONENT
# ============================================================================

def create_chat_message(text, is_user=False):
    """Create a chat message bubble using Mantine components"""
    return dmc.Paper(
        children=[
            dmc.Group(
                [
                    dmc.ThemeIcon(
                        get_icon("mdi:account" if is_user else "mdi:robot", 16),
                        size="sm",
                        radius="xl",
                        color="orange" if is_user else "cyan",
                        variant="light"
                    ),
                    dmc.Text(
                        "You" if is_user else "AUTOMATA AI",
                        size="xs",
                        fw=700,
                        c="orange" if is_user else "cyan"
                    )
                ],
                gap="xs",
                mb=5
            ),
            dmc.Text(text, size="sm", style={'lineHeight': '1.5'})
        ],
        p="sm",
        radius="md",
        mb="sm",
        ml=40 if is_user else 0,
        mr=0 if is_user else 40,
        style={
            'backgroundColor': colors['chat-user'] if is_user else colors['chat-ai'],
            'color': colors['text']
        }
    )


# Initial chat messages
initial_chat = [
    {
        'text': "Hello! I'm AUTOMATA AI, your intelligent well log interpretation assistant. I've analyzed the current well data and I'm ready to help you understand the formation characteristics, identify pay zones, and provide recommendations. What would you like to know?",
        'is_user': False}
]


# ============================================================================
# METRIC CARD COMPONENT
# ============================================================================

def create_metric_card(icon, icon_color, value_id, label):
    """Create a metric card component"""
    return dmc.Paper(
        children=[
            dmc.Stack(
                [
                    dmc.ThemeIcon(
                        get_icon(icon, 28),
                        size=50,
                        radius="md",
                        color=icon_color,
                        variant="light"
                    ),
                    dmc.Title(
                        id=value_id,
                        children='0',
                        order=3,
                        c=icon_color
                    ),
                    dmc.Text(
                        label,
                        c="dimmed",
                        fw=600,
                        size="sm"
                    )
                ],
                align="center",
                gap="xs",
                p="md"
            )
        ],
        p="md",
        radius="md",
        withBorder=True,
        style={'backgroundColor': colors['surface'], 'borderColor': colors['surface-light']}
    )


# ============================================================================
# DASH LAYOUT
# ============================================================================

app.layout = dmc.MantineProvider(
    theme={
        "colorScheme": "dark",
        "fontFamily": "Inter, sans-serif",
        "primaryColor": "cyan",
        "colors": {
            "dark": [
                "#C1C2C5", "#A6A7AB", "#909296", "#5C5F66",
                "#373A40", "#2C2E33", "#25262B", "#1A1B1E",
                "#141517", "#101113"
            ]
        }
    },
    children=[
        dmc.Container(
            [
                # Header
                dmc.Stack(
                    [
                        dmc.Title(
                            "AUTOMATA INTELLIGENCE",
                            order=1,
                            c="white",
                            ta="center"
                        ),
                        dmc.Group(
                            [
                                get_icon("mdi:oil", 32, colors['primary']),
                                dmc.Title(
                                    "Well Log Interpretation Suite",
                                    order=2,
                                    c="cyan",
                                    ta="center"
                                )
                            ],
                            justify="center",
                            gap="sm"
                        ),
                        dmc.Text(
                            "AI-Powered Formation Evaluation & Reservoir Characterization",
                            c="dimmed",
                            size="lg",
                            ta="center"
                        )
                    ],
                    gap="xs",
                    py="xl"
                ),

                # Upload Section
                dmc.Paper(
                    children=[
                        dmc.Grid(
                            [
                                dmc.GridCol(
                                    [
                                        dmc.Stack(
                                            [
                                                dmc.ThemeIcon(
                                                    get_icon("mdi:cloud-upload", 32),
                                                    size=60,
                                                    radius="md",
                                                    color="cyan",
                                                    variant="light"
                                                ),
                                                dmc.Title("Upload Well Log Data", order=5, c="white"),
                                                dmc.Text(
                                                    "Drag & drop or click to upload LAS, DLIS, or CSV files",
                                                    c="dimmed",
                                                    size="sm"
                                                )
                                            ],
                                            align="center",
                                            gap="xs"
                                        )
                                    ],
                                    span={"base": 12, "md": 4}
                                ),
                                dmc.GridCol(
                                    [
                                        dcc.Upload(
                                            id='upload-las-files',
                                            children=dmc.Group(
                                                [
                                                    get_icon("mdi:file-upload", 24, colors['primary']),
                                                    dmc.Text("Drop files here or ", c="white"),
                                                    dmc.Text("Browse", c="cyan", td='underline', style={'cursor': 'pointer'})
                                                ],
                                                justify="center"
                                            ),
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
                                                'cursor': 'pointer',
                                                'display': 'flex',
                                                'alignItems': 'center',
                                                'justifyContent': 'center'
                                            },
                                            multiple=True,
                                            accept='.las,.dlis,.csv,.xlsx'
                                        )
                                    ],
                                    span={"base": 12, "md": 5}
                                ),
                                dmc.GridCol(
                                    [
                                        html.Div(
                                            id='upload-status',
                                            children=[
                                                dmc.Group(
                                                    [
                                                        get_icon("mdi:information", 16, colors['primary']),
                                                        dmc.Text("Supported formats:", fw=600, size="sm")
                                                    ],
                                                    gap="xs"
                                                ),
                                                dmc.List(
                                                    [
                                                        dmc.ListItem(dmc.Text(".LAS (Log ASCII Standard)", size="xs")),
                                                        dmc.ListItem(
                                                            dmc.Text(".DLIS (Digital Log Interchange)", size="xs")),
                                                        dmc.ListItem(dmc.Text(".CSV / .XLSX", size="xs"))
                                                    ],
                                                    size="sm",
                                                    c="dimmed"
                                                )
                                            ]
                                        )
                                    ],
                                    span={"base": 12, "md": 3}
                                )
                            ],
                            align="center",
                            gutter="lg"
                        )
                    ],
                    p="lg",
                    radius="md",
                    withBorder=True,
                    mb="md",
                    style={'backgroundColor': colors['surface'], 'borderColor': colors['surface-light']}
                ),

                # Uploaded Files Display
                html.Div(id='uploaded-files-list', children=[], style={'marginBottom': '15px'}),

                # Control Panel
                dmc.Paper(
                    children=[
                        dmc.Title(
                            "Well Selection & Analysis Controls",
                            order=5,
                            c="cyan",
                            mb="md"
                        ),
                        dmc.Grid(
                            [
                                dmc.GridCol(
                                    [
                                        dmc.Select(
                                            id='well-selector',
                                            label="Select Well",
                                            data=[{'label': well, 'value': well} for well in wells_data.keys()],
                                            value='WELL-001',
                                            styles={
                                                "input": {"backgroundColor": colors['surface-light'], "color": "white",
                                                          "borderColor": colors['surface-light']},
                                                "dropdown": {"backgroundColor": colors['surface'],
                                                             "borderColor": colors['surface-light']}
                                            }
                                        )
                                    ],
                                    span={"base": 12, "md": 3}
                                ),
                                dmc.GridCol(
                                    [
                                        dmc.Text("Depth Interval (ft)", size="sm", fw=500, mb="xs"),
                                        dcc.RangeSlider(
                                            id='depth-slider',
                                            min=5000,
                                            max=8000,
                                            value=[5000, 8000],
                                            marks={i: {'label': f'{i}', 'style': {'color': colors['text-secondary']}}
                                                   for i
                                                   in range(5000, 8001, 500)},
                                            tooltip={"placement": "bottom", "always_visible": True}
                                        )
                                    ],
                                    span={"base": 12, "md": 5}
                                ),
                                dmc.GridCol(
                                    [
                                        dmc.Text("Analysis Mode", size="sm", fw=500, mb="xs"),
                                        dmc.SegmentedControl(
                                            id='analysis-mode',
                                            data=[
                                                {'label': 'Standard', 'value': 'standard'},
                                                {'label': 'Advanced AI', 'value': 'ai'}
                                            ],
                                            value='ai',
                                            fullWidth=True,
                                            color="cyan"
                                        )
                                    ],
                                    span={"base": 12, "md": 2}
                                ),
                                dmc.GridCol(
                                    [
                                        dmc.Button(
                                            "Run Analysis",
                                            id='run-analysis',
                                            leftSection=get_icon("mdi:play", 18),
                                            fullWidth=True,
                                            color="cyan",
                                            variant="filled",
                                            size="md",
                                            mt="lg",
                                            n_clicks=0
                                        )
                                    ],
                                    span={"base": 12, "md": 2}
                                )
                            ],
                            gutter="lg"
                        )
                    ],
                    p="lg",
                    radius="md",
                    withBorder=True,
                    mb="lg",
                    style={'backgroundColor': colors['surface'], 'borderColor': colors['surface-light']}
                ),

                # Key Metrics Row
                dmc.SimpleGrid(
                    cols={"base": 2, "md": 4},
                    spacing="lg",
                    mb="lg",
                    children=[
                        create_metric_card("mdi:ruler", "green", 'net-pay-metric', 'Net Pay'),
                        create_metric_card("mdi:percent", "cyan", 'avg-porosity-metric', 'Avg Porosity'),
                        create_metric_card("mdi:water", "yellow", 'water-sat-metric', 'Avg Water Saturation'),
                        create_metric_card("mdi:layers-triple", "orange", 'hc-zones-metric', 'HC Zones')
                    ]
                ),

                # Main Visualization Area with Chat
                dmc.Grid(
                    [
                        # Well Logs Plot
                        dmc.GridCol(
                            [
                                dmc.Paper(
                                    # 1. CRITICAL: Set position relative so the overlay stays inside this Paper
                                    pos="relative",
                                    children=[
                                        # 2. LoadingOverlay is now a sibling (no children argument)
                                        dmc.LoadingOverlay(
                                            visible=False,
                                            # You must now control this via a callback (Output -> visible)
                                            zIndex=1000,
                                            overlayProps={"radius": "sm", "blur": 2}
                                        ),
                                        # 3. The Graph is now a direct child of the Paper
                                        dcc.Graph(id='well-logs-plot', style={'height': '700px'})
                                    ],
                                    # 4. Your existing styling
                                    p="md",
                                    radius="md",
                                    withBorder=True,
                                    style={'backgroundColor': colors['surface'], 'borderColor': colors['surface-light']}
                                )
                            ],
                            span={"base": 12, "lg": 8}
                        ),

                        # Right Panel with Analysis & Chat
                        dmc.GridCol(
                            [
                                dmc.Stack(
                                    [
                                        # Formation Analysis
                                        dmc.Paper(
                                            children=[
                                                dmc.Group(
                                                    [
                                                        get_icon("mdi:chart-pie", 20, colors['primary']),
                                                        dmc.Title("Formation Analysis", order=5, c="cyan")
                                                    ],
                                                    gap="xs",
                                                    mb="sm"
                                                ),
                                                dcc.Graph(id='lithology-pie', style={'height': '250px'})
                                            ],
                                            p="md",
                                            radius="md",
                                            withBorder=True,
                                            style={'backgroundColor': colors['surface'],
                                                   'borderColor': colors['surface-light']}
                                        ),

                                        # AI Chat Interface
                                        dmc.Paper(
                                            children=[
                                                dmc.Group(
                                                    [
                                                        dmc.Group(
                                                            [
                                                                dmc.ThemeIcon(
                                                                    get_icon("mdi:robot", 20),
                                                                    size="md",
                                                                    radius="xl",
                                                                    color="green",
                                                                    variant="light"
                                                                ),
                                                                dmc.Title("AUTOMATA AI Assistant", order=5, c="cyan")
                                                            ],
                                                            gap="xs"
                                                        ),
                                                        dmc.Badge("Online", color="green", variant="dot", size="sm")
                                                    ],
                                                    justify="space-between",
                                                    mb="md"
                                                ),

                                                # Chat Messages Container
                                                html.Div(
                                                    id='chat-messages',
                                                    children=[create_chat_message(msg['text'], msg['is_user']) for msg
                                                              in initial_chat],
                                                    style={
                                                        'height': '280px',
                                                        'overflowY': 'auto',
                                                        'padding': '10px',
                                                        'backgroundColor': colors['background'],
                                                        'borderRadius': '10px',
                                                        'marginBottom': '15px'
                                                    }
                                                ),

                                                # Chat Input
                                                dmc.Group(
                                                    [
                                                        dmc.TextInput(
                                                            id='chat-input',
                                                            placeholder='Ask about well logs, porosity, zones...',
                                                            style={'flex': 1},
                                                            styles={
                                                                "input": {
                                                                    "backgroundColor": colors['surface-light'],
                                                                    "border": "none",
                                                                    "color": colors['text']
                                                                }
                                                            }
                                                        ),
                                                        dmc.ActionIcon(
                                                            get_icon("mdi:send", 20),
                                                            id='send-chat',
                                                            size="lg",
                                                            variant="filled",
                                                            color="cyan",
                                                            n_clicks=0
                                                        )
                                                    ],
                                                    gap="xs"
                                                ),

                                                # Quick Actions
                                                dmc.Stack(
                                                    [
                                                        dmc.Text("Quick questions:", c="dimmed", size="xs", mt="sm"),
                                                        dmc.Group(
                                                            [
                                                                dmc.Badge(
                                                                    "Porosity analysis",
                                                                    id='quick-porosity',
                                                                    color="cyan",
                                                                    variant="light",
                                                                    style={'cursor': 'pointer'}
                                                                ),
                                                                dmc.Badge(
                                                                    "Pay zones",
                                                                    id='quick-zones',
                                                                    color="green",
                                                                    variant="light",
                                                                    style={'cursor': 'pointer'}
                                                                ),
                                                                dmc.Badge(
                                                                    "Recommendations",
                                                                    id='quick-recommend',
                                                                    color="yellow",
                                                                    variant="light",
                                                                    style={'cursor': 'pointer'}
                                                                )
                                                            ],
                                                            gap="xs"
                                                        )
                                                    ],
                                                    gap=5
                                                )
                                            ],
                                            p="md",
                                            radius="md",
                                            withBorder=True,
                                            style={'backgroundColor': colors['surface'],
                                                   'borderColor': colors['surface-light']}
                                        )
                                    ],
                                    gap="md"
                                )
                            ],
                            span={"base": 12, "lg": 4}
                        )
                    ],
                    gutter="lg",
                    mb="lg"
                ),

                # Detailed Analysis Section
                dmc.Grid(
                    [
                        dmc.GridCol(
                            [
                                dmc.Paper(
                                    children=[
                                        dmc.Group(
                                            [
                                                get_icon("mdi:table", 20, colors['primary']),
                                                dmc.Title("Zone Analysis Report", order=5, c="cyan")
                                            ],
                                            gap="xs",
                                            mb="md"
                                        ),
                                        html.Div(id='zone-table', style={'overflowX': 'auto'})
                                    ],
                                    p="md",
                                    radius="md",
                                    withBorder=True,
                                    style={'backgroundColor': colors['surface'], 'borderColor': colors['surface-light']}
                                )
                            ],
                            span={"base": 12, "md": 6}
                        ),
                        dmc.GridCol(
                            [
                                dmc.Paper(
                                    children=[
                                        dmc.Group(
                                            [
                                                get_icon("mdi:crosshairs", 20, colors['primary']),
                                                dmc.Title("Porosity vs Resistivity Crossplot", order=5, c="cyan")
                                            ],
                                            gap="xs",
                                            mb="md"
                                        ),
                                        dcc.Graph(id='crossplot', style={'height': '350px'})
                                    ],
                                    p="md",
                                    radius="md",
                                    withBorder=True,
                                    style={'backgroundColor': colors['surface'], 'borderColor': colors['surface-light']}
                                )
                            ],
                            span={"base": 12, "md": 6}
                        )
                    ],
                    gutter="lg"
                ),

                # Store components
                dcc.Store(id='processed-data-store'),
                dcc.Store(id='chat-history-store', data=initial_chat),
                dcc.Store(id='uploaded-files-store', data=[])

            ],
            fluid=True,
            style={'backgroundColor': colors['background'], 'minHeight': '100vh', 'padding': '20px'}
        )
    ]
)


# ============================================================================
# CALLBACKS
# ============================================================================

# File Upload Callback
@app.callback(
    [Output('uploaded-files-list', 'children'),
     Output('uploaded-files-store', 'data')],
    [Input('upload-las-files', 'contents')],
    [State('upload-las-files', 'filename'),
     State('uploaded-files-store', 'data')]
)
def handle_file_upload(contents, filenames, existing_files):
    if contents is None:
        return [], existing_files or []

    if existing_files is None:
        existing_files = []

    new_files = []
    for filename in filenames:
        file_info = {
            'name': filename,
            'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'status': 'Processed'
        }
        new_files.append(file_info)

    all_files = existing_files + new_files

    # Create file cards using Mantine
    file_badges = []
    for f in all_files:
        file_badges.append(
            dmc.Paper(
                children=[
                    dmc.Group(
                        [
                            dmc.ThemeIcon(
                                get_icon("mdi:file-document", 24),
                                size="lg",
                                radius="md",
                                color="cyan",
                                variant="light"
                            ),
                            dmc.Stack(
                                [
                                    dmc.Text(f['name'], fw=600, c="white"),
                                    dmc.Text(f"Uploaded: {f['upload_time']}", size="xs", c="dimmed")
                                ],
                                gap=0,
                                style={'flex': 1}
                            ),
                            dmc.Group(
                                [
                                    dmc.Badge(f['status'], color='green', variant='light'),
                                    dmc.ThemeIcon(
                                        get_icon("mdi:check-circle", 16),
                                        size="sm",
                                        radius="xl",
                                        color="green",
                                        variant="light"
                                    )
                                ],
                                gap="xs"
                            )
                        ],
                        gap="md"
                    )
                ],
                p="sm",
                radius="md",
                withBorder=True,
                mb="sm",
                style={'backgroundColor': colors['surface'], 'borderColor': colors['success']}
            )
        )

    return file_badges, all_files


# Chat Callback
@app.callback(
    [Output('chat-messages', 'children'),
     Output('chat-history-store', 'data'),
     Output('chat-input', 'value')],
    [Input('send-chat', 'n_clicks'),
     Input('chat-input', 'n_submit'),
     Input('quick-porosity', 'n_clicks'),
     Input('quick-zones', 'n_clicks'),
     Input('quick-recommend', 'n_clicks')],
    [State('chat-input', 'value'),
     State('chat-history-store', 'data')]
)
def update_chat(send_clicks, input_submit, q_porosity, q_zones, q_recommend, user_message, chat_history):
    ctx = callback_context

    if not ctx.triggered:
        messages = [create_chat_message(msg['text'], msg['is_user']) for msg in chat_history]
        return messages, chat_history, ''

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Handle quick action buttons
    if triggered_id == 'quick-porosity':
        user_message = "Can you analyze the porosity in this well?"
    elif triggered_id == 'quick-zones':
        user_message = "What are the identified pay zones?"
    elif triggered_id == 'quick-recommend':
        user_message = "What are your recommendations for this well?"

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
        # Return empty figure with styling
        fig = go.Figure()
        fig.update_layout(
            plot_bgcolor=colors['surface'],
            paper_bgcolor=colors['background'],
            font=dict(color=colors['text']),
            annotations=[
                dict(
                    text="Click 'Run Analysis' to visualize well logs",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=18, color=colors['text-secondary'])
                )
            ]
        )
        return fig

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
                   line=dict(color='#00ff88', width=1.5),
                   fill='tozerox', fillcolor='rgba(0, 255, 136, 0.1)'),
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
                   line=dict(color='#ff3366', width=1.5),
                   fill='tozerox', fillcolor='rgba(255, 51, 102, 0.1)'),
        row=1, col=4
    )

    fig.add_trace(
        go.Scatter(x=df['SW'] * 100, y=df['DEPTH'], mode='lines', name='Sw',
                   line=dict(color='#00d4ff', width=1.5),
                   fill='tozerox', fillcolor='rgba(0, 212, 255, 0.1)'),
        row=1, col=5
    )

    lithology_colors = {
        'Sandstone': '#ffeb3b',
        'Shale': '#795548',
        'Limestone': '#2196f3',
        'Dolomite': '#9c27b0',
        'Unknown': '#666666'
    }

    for lith in df['LITHOLOGY'].unique():
        df_lith = df[df['LITHOLOGY'] == lith]
        fig.add_trace(
            go.Scatter(x=[1] * len(df_lith), y=df_lith['DEPTH'], mode='markers',
                       name=lith, marker=dict(color=lithology_colors.get(lith, '#666'), size=8, symbol='square')),
            row=1, col=6
        )

    fig.update_layout(
        height=700,
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='middle',
            y=0.5,
            xanchor='left',
            x=1.02
        ),
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['background'],
        font=dict(color=colors['text']),
        title=dict(
            text=f"Well Logs - {processed_data['well_id']}",
            font=dict(size=20, color=colors['primary'])
        )
    )

    fig.update_yaxes(autorange='reversed', title='Depth (ft)', row=1, col=1, gridcolor=colors['surface-light'])
    fig.update_xaxes(title='API', range=[0, 150], row=1, col=1, gridcolor=colors['surface-light'])
    fig.update_xaxes(title='Ohm.m', type='log', row=1, col=2, gridcolor=colors['surface-light'])
    fig.update_xaxes(title='g/cc | v/v', range=[1.9, 2.9], row=1, col=3, gridcolor=colors['surface-light'])
    fig.update_xaxes(title='%', range=[0, 40], row=1, col=4, gridcolor=colors['surface-light'])
    fig.update_xaxes(title='%', range=[0, 100], row=1, col=5, gridcolor=colors['surface-light'])

    return fig


@app.callback(
    Output('lithology-pie', 'figure'),
    [Input('processed-data-store', 'data')]
)
def update_lithology_pie(processed_data):
    if not processed_data:
        fig = go.Figure()
        fig.update_layout(
            plot_bgcolor=colors['surface'],
            paper_bgcolor=colors['surface'],
            font=dict(color=colors['text']),
            annotations=[dict(text="No data", x=0.5, y=0.5, showarrow=False)]
        )
        return fig

    df = pd.DataFrame(processed_data['data'])
    lithology_counts = df['LITHOLOGY'].value_counts()

    lith_colors = {
        'Sandstone': '#ffeb3b',
        'Shale': '#795548',
        'Limestone': '#2196f3',
        'Dolomite': '#9c27b0',
        'Unknown': '#666666'
    }

    pie_colors = [lith_colors.get(lith, '#666666') for lith in lithology_counts.index]

    fig = go.Figure(data=[
        go.Pie(
            labels=lithology_counts.index,
            values=lithology_counts.values,
            hole=0.4,
            marker=dict(colors=pie_colors),
            textinfo='percent+label',
            textposition='outside'
        )
    ])

    fig.update_layout(
        showlegend=False,
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['surface'],
        font=dict(color=colors['text'], size=11),
        margin=dict(t=20, b=20, l=20, r=20)
    )

    return fig


@app.callback(
    Output('crossplot', 'figure'),
    [Input('processed-data-store', 'data')]
)
def update_crossplot(processed_data):
    if not processed_data:
        fig = go.Figure()
        fig.update_layout(
            plot_bgcolor=colors['surface'],
            paper_bgcolor=colors['surface'],
            font=dict(color=colors['text']),
            annotations=[dict(text="No data", x=0.5, y=0.5, showarrow=False)]
        )
        return fig

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
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        plot_bgcolor=colors['surface'],
        paper_bgcolor=colors['surface'],
        font=dict(color=colors['text']),
        margin=dict(t=30, b=40, l=50, r=10),
        xaxis=dict(gridcolor=colors['surface-light']),
        yaxis=dict(gridcolor=colors['surface-light'])
    )

    return fig


@app.callback(
    Output('zone-table', 'children'),
    [Input('processed-data-store', 'data')]
)
def update_zone_table(processed_data):
    if not processed_data:
        return dmc.Group(
            [
                get_icon("mdi:information", 20, colors['primary']),
                dmc.Text("Run analysis to view zone data", c="dimmed")
            ],
            justify="center",
            p="xl"
        )

    df = pd.DataFrame(processed_data['data'])
    df['zone_id'] = (df['HC_FLAG'] != df['HC_FLAG'].shift()).cumsum()

    zones = []
    for zone_id in df['zone_id'].unique():
        zone_df = df[df['zone_id'] == zone_id]
        if zone_df['HC_FLAG'].iloc[0] == 1:
            quality = 'Excellent' if zone_df['POROSITY'].mean() > 0.2 else (
                'Good' if zone_df['POROSITY'].mean() > 0.15 else 'Fair')
            zones.append({
                'Zone': f'Z{len(zones) + 1}',
                'Top': f"{zone_df['DEPTH'].min():.0f}",
                'Bottom': f"{zone_df['DEPTH'].max():.0f}",
                'Thick': f"{zone_df['DEPTH'].max() - zone_df['DEPTH'].min():.0f}",
                'Φ%': f"{zone_df['POROSITY'].mean() * 100:.1f}",
                'Sw%': f"{zone_df['SW'].mean() * 100:.1f}",
                'Quality': quality
            })

    if zones:
        # Create table header
        header = dmc.TableThead(
            dmc.TableTr([
                dmc.TableTh(col, style={'backgroundColor': colors['primary'], 'color': colors['background'],
                                        'padding': '10px', 'textAlign': 'center'})
                for col in zones[0].keys()
            ])
        )

        # Create table rows
        rows = []
        for zone in zones:
            quality_color = colors['success'] if zone['Quality'] == 'Excellent' else (
                colors['warning'] if zone['Quality'] == 'Good' else colors['secondary'])
            cells = []
            for key, value in zone.items():
                if key == 'Quality':
                    cells.append(dmc.TableTd(
                        dmc.Badge(value, color="green" if value == "Excellent" else (
                            "yellow" if value == "Good" else "orange"), variant="light"),
                        style={'textAlign': 'center', 'padding': '8px'}
                    ))
                else:
                    cells.append(
                        dmc.TableTd(value, style={'textAlign': 'center', 'padding': '8px', 'color': colors['text']}))
            rows.append(dmc.TableTr(cells, style={'backgroundColor': colors['surface']}))

        body = dmc.TableTbody(rows)

        return dmc.Table(
            [header, body],
            striped=True,
            highlightOnHover=True,
            withTableBorder=True,
            withColumnBorders=True,
            style={'backgroundColor': colors['surface']}
        )

    return dmc.Group(
        [
            get_icon("mdi:alert", 20, colors['warning']),
            dmc.Text("No hydrocarbon zones identified in selected interval", c="dimmed")
        ],
        justify="center",
        p="xl"
    )


# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == '__main__':
    app.run(debug=False, port=8051, host='127.0.0.1')