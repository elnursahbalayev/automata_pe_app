import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc


# ==============================================================================
# 1. BACKGROUND LOGIC: DUMMY DATA GENERATION
# ==============================================================================
# This function creates a realistic-looking well log dataset.
# The "AI" part is simulated by assigning a facies based on logical rules
# applied to the log values.

def create_dummy_data(wells=['Well-A', 'Well-B', 'Well-C'], depth_range=(1000, 2000), step=0.5):
    """Generates a DataFrame with synthetic well log data."""

    facies_properties = {
        'Sandstone': {'GR': (20, 45), 'RT': (10, 200), 'NPHI': (0.10, 0.25), 'RHOB': (2.1, 2.4)},
        'Shale': {'GR': (80, 150), 'RT': (1, 10), 'NPHI': (0.30, 0.50), 'RHOB': (2.3, 2.6)},
        'Limestone': {'GR': (10, 30), 'RT': (50, 5000), 'NPHI': (0.02, 0.10), 'RHOB': (2.6, 2.75)},
        'Shaly Sand': {'GR': (50, 75), 'RT': (5, 50), 'NPHI': (0.20, 0.35), 'RHOB': (2.2, 2.5)}
    }

    dfs = []
    for well_name in wells:
        depth = np.arange(depth_range[0], depth_range[1], step)
        num_points = len(depth)

        # Create zones of different facies
        facies = np.random.choice(list(facies_properties.keys()), size=num_points // 100)
        facies = np.repeat(facies, 100)
        np.random.shuffle(facies)  # Mix it up a bit
        facies = facies[:num_points]

        # Generate log data based on facies properties
        gr = np.zeros(num_points)
        rt = np.zeros(num_points)
        nphi = np.zeros(num_points)
        rhob = np.zeros(num_points)

        for f in np.unique(facies):
            mask = (facies == f)
            props = facies_properties[f]
            gr[mask] = np.random.normal(np.mean(props['GR']), 5, size=np.sum(mask))
            rt[mask] = np.random.normal(np.mean(props['RT']), 10, size=np.sum(mask))
            nphi[mask] = np.random.normal(np.mean(props['NPHI']), 0.03, size=np.sum(mask))
            rhob[mask] = np.random.normal(np.mean(props['RHOB']), 0.05, size=np.sum(mask))

        # Add some random noise and smoothing
        gr = np.convolve(gr, np.ones(5) / 5, mode='same')

        well_df = pd.DataFrame({
            'DEPTH': depth,
            'WELL': well_name,
            'GR': gr,
            'RT': rt,
            'NPHI': nphi,
            'RHOB': rhob,
            'FACIES': facies  # This is our "AI-Predicted Lithofacies"
        })
        dfs.append(well_df)

    df = pd.concat(dfs, ignore_index=True)
    return df


# Create the data
df = create_dummy_data()

# Define colors for our predicted facies for consistency across plots
FACIES_COLORS = {
    'Sandstone': '#f4d03f',
    'Shaly Sand': '#7d6608',
    'Shale': '#707b7c',
    'Limestone': '#3498db'
}

# ==============================================================================
# 2. STUNNING UI: APP LAYOUT
# ==============================================================================
# We use Dash Bootstrap Components for a professional and responsive layout.
# A dark theme (CYBORG) is used for a modern "tech" feel.

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server

# --- Sidebar Controls ---
sidebar = dbc.Card(
    [
        html.H3("Controls", className="card-title p-3"),
        dbc.CardBody(
            [
                html.H5("Select Well:", className="card-title"),
                dcc.Dropdown(
                    id='well-selector',
                    options=[{'label': well, 'value': well} for well in df['WELL'].unique()],
                    value=df['WELL'].unique()[0],  # Default value
                    clearable=False,
                    className="mb-4"
                ),
                html.H5("About This Module", className="card-title"),
                dcc.Markdown(
                    """
                    This module demonstrates AI-powered lithofacies classification from well log data.

                    **Logs:**
                    - **GR:** Gamma Ray (API)
                    - **RT:** Resistivity (ohm.m)
                    - **NPHI:** Neutron Porosity (v/v)
                    - **RHOB:** Bulk Density (g/cm³)

                    The colored track shows the predicted rock type (lithofacies) at each depth, enabling rapid geological interpretation.
                    """
                ),
            ]
        ),
    ],
    className="h-100",
)

# --- Main Content Area ---
main_content = dbc.Card(
    [
        html.H3("Well Log Interpretation Dashboard", className="card-title p-3"),
        dbc.CardBody(
            dcc.Loading(  # Adds a spinner while the plots are updating
                id="loading-main",
                type="default",
                children=[
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(id='well-log-plot', style={'height': '80vh'}), width=8),
                            dbc.Col(dcc.Graph(id='cross-plot', style={'height': '80vh'}), width=4),
                        ]
                    )
                ]
            )
        )
    ],
)

# --- Final App Layout ---
app.layout = dbc.Container(
    [
        html.H1("AI-Driven Oil & Gas Intelligence Platform", className="text-center my-4"),
        dbc.Row(
            [
                dbc.Col(sidebar, width=3),
                dbc.Col(main_content, width=9),
            ],
            className="g-0",  # No gutters
        ),
    ],
    fluid=True,
    className="dbc"  # Use .dbc class for CYBORG theme styles
)


# ==============================================================================
# 3. INTERACTIVITY: APP CALLBACKS
# ==============================================================================
# This callback connects the sidebar controls to the main content plots.
# When a new well is selected, this function re-generates the figures.

@app.callback(
    [Output('well-log-plot', 'figure'),
     Output('cross-plot', 'figure')],
    [Input('well-selector', 'value')]
)
def update_graphs(selected_well):
    if not selected_well:
        return go.Figure(), go.Figure()

    # Filter data for the selected well
    dff = df[df['WELL'] == selected_well].copy()

    # --- 1. Create the Well Log Plot ---
    fig = make_subplots(
        rows=1, cols=5,
        shared_yaxes=True,
        column_widths=[0.8, 0.8, 0.8, 0.8, 0.5],
        horizontal_spacing=0.02
    )

    # GR Track
    fig.add_trace(go.Scatter(x=dff['GR'], y=dff['DEPTH'], name='GR', line=dict(color='green')), row=1, col=1)
    # RT Track (Logarithmic scale is standard for resistivity)
    fig.add_trace(go.Scatter(x=dff['RT'], y=dff['DEPTH'], name='RT', line=dict(color='red')), row=1, col=2)
    # NPHI Track
    fig.add_trace(go.Scatter(x=dff['NPHI'], y=dff['DEPTH'], name='NPHI', line=dict(color='blue')), row=1, col=3)
    # RHOB Track
    fig.add_trace(go.Scatter(x=dff['RHOB'], y=dff['DEPTH'], name='RHOB', line=dict(color='purple')), row=1, col=4)

    # Facies Track (The "AI" output)
    dff['FACIES_ID'] = dff['FACIES'].astype('category').cat.codes
    facies_labels = list(dff['FACIES'].astype('category').cat.categories)
    facies_codes = list(range(len(facies_labels)))

    custom_colorscale = []
    # Sort labels to match FACIES_COLORS order if needed, but this works fine
    sorted_labels = sorted(facies_labels, key=lambda x: facies_codes[facies_labels.index(x)])
    for i, label in enumerate(sorted_labels):
        color = FACIES_COLORS.get(label, 'white')
        norm_val_start = i / len(facies_labels)
        norm_val_end = (i + 1) / len(facies_labels)
        custom_colorscale.append([norm_val_start, color])
        custom_colorscale.append([norm_val_end, color])

    fig.add_trace(go.Heatmap(
        z=dff['FACIES_ID'], y=dff['DEPTH'], x=['FACIES'],
        colorscale=custom_colorscale, showscale=True,
        colorbar=dict(tickvals=[c + 0.5 for c in facies_codes], ticktext=facies_labels, title='Facies')
    ), row=1, col=5)

    # --- Update Layout for the Log Plot ---
    fig.update_layout(
        title=f'Well Log Plot: {selected_well}', template='plotly_dark',
        showlegend=False, yaxis=dict(autorange='reversed', title='Depth (m)')
    )
    fig.update_xaxes(title_text="GR (API)", row=1, col=1)
    fig.update_xaxes(title_text="RT (ohm.m)", type='log', row=1, col=2)
    fig.update_xaxes(title_text="NPHI (v/v)", row=1, col=3)
    fig.update_xaxes(title_text="RHOB (g/cm³)", row=1, col=4)
    fig.update_xaxes(title_text="AI Facies", showticklabels=False, row=1, col=5)

    # --- 2. Create the Cross-Plot ---
    cross_plot_fig = px.scatter(
        dff, x='RHOB', y='NPHI', color='FACIES',
        color_discrete_map=FACIES_COLORS,
        title=f'NPHI vs. RHOB Cross-Plot: {selected_well}',
        template='plotly_dark',
        labels={'RHOB': 'Bulk Density (g/cm³)', 'NPHI': 'Neutron Porosity (v/v)'}
    )
    cross_plot_fig.update_yaxes(autorange='reversed')

    return fig, cross_plot_fig


# ==============================================================================
# 4. RUN THE APP
# ==============================================================================
if __name__ == '__main__':
    app.run(debug=True)