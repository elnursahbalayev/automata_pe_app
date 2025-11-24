import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)


# Generate mock data for demonstration
def generate_mock_data():
    dates = pd.date_range(start='2024-01-01', end='2024-11-24', freq='W')

    # Pipeline inspection data
    pipeline_data = pd.DataFrame({
        'Date': dates,
        'Corrosion_Rate': np.random.uniform(0.5, 4.5, len(dates)),
        'Thickness_Loss': np.cumsum(np.random.uniform(0.01, 0.15, len(dates))),
        'Risk_Score': np.random.uniform(20, 95, len(dates))
    })

    # Asset locations with risk levels
    assets = pd.DataFrame({
        'Asset_ID': ['P-001', 'P-002', 'P-003', 'P-004', 'P-005', 'T-001', 'T-002', 'V-001', 'V-002', 'P-006'],
        'Asset_Type': ['Pipeline', 'Pipeline', 'Pipeline', 'Pipeline', 'Pipeline', 'Tank', 'Tank', 'Valve', 'Valve',
                       'Pipeline'],
        'Location': ['Sector A', 'Sector A', 'Sector B', 'Sector B', 'Sector C', 'Sector A', 'Sector C', 'Sector B',
                     'Sector C', 'Sector A'],
        'Risk_Level': ['High', 'Medium', 'Critical', 'Low', 'Medium', 'High', 'Low', 'Medium', 'Critical', 'High'],
        'Corrosion_Rate': [3.2, 2.1, 4.5, 1.2, 2.8, 3.5, 1.5, 2.3, 4.2, 3.8],
        'Last_Inspection': ['2024-11-20', '2024-11-18', '2024-11-15', '2024-11-22', '2024-11-19',
                            '2024-11-17', '2024-11-21', '2024-11-16', '2024-11-14', '2024-11-20']
    })

    return pipeline_data, assets


pipeline_data, assets = generate_mock_data()

# Color scheme
colors = {
    'background': '#0a0e27',
    'card': '#151b3d',
    'text': '#ffffff',
    'primary': '#00d4ff',
    'secondary': '#7c3aed',
    'critical': '#ef4444',
    'high': '#f59e0b',
    'medium': '#fbbf24',
    'low': '#10b981'
}

# App layout
app.layout = html.Div(style={'backgroundColor': colors['background'], 'minHeight': '100vh', 'padding': '20px',
                             'fontFamily': 'Arial, sans-serif'}, children=[
    # Header
    html.Div(style={'marginBottom': '30px'}, children=[
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}, children=[
            html.Div(children=[
                html.H1('AUTOMATA INTELLIGENCE',
                        style={'color': colors['primary'], 'margin': '0', 'fontSize': '28px', 'fontWeight': '700',
                               'letterSpacing': '2px'}),
                html.P('Corrosion Detection & Asset Integrity System',
                       style={'color': colors['text'], 'margin': '5px 0 0 0', 'fontSize': '14px', 'opacity': '0.8'})
            ]),
            html.Div(style={'textAlign': 'right'}, children=[
                html.P(f'Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                       style={'color': colors['text'], 'margin': '0', 'fontSize': '12px', 'opacity': '0.6'})
            ])
        ])
    ]),

    # KPI Cards
    html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(250px, 1fr))', 'gap': '20px',
                    'marginBottom': '30px'}, children=[
        # Total Assets
        html.Div(style={'backgroundColor': colors['card'], 'borderRadius': '12px', 'padding': '25px',
                        'border': f'1px solid {colors["primary"]}', 'boxShadow': f'0 4px 20px rgba(0, 212, 255, 0.1)'},
                 children=[
                     html.P('Total Assets Monitored',
                            style={'color': colors['text'], 'margin': '0 0 10px 0', 'fontSize': '14px',
                                   'opacity': '0.8'}),
                     html.H2(str(len(assets)),
                             style={'color': colors['primary'], 'margin': '0', 'fontSize': '36px', 'fontWeight': '700'})
                 ]),
        # Critical Assets
        html.Div(style={'backgroundColor': colors['card'], 'borderRadius': '12px', 'padding': '25px',
                        'border': f'1px solid {colors["critical"]}', 'boxShadow': f'0 4px 20px rgba(239, 68, 68, 0.1)'},
                 children=[
                     html.P('Critical Risk Assets',
                            style={'color': colors['text'], 'margin': '0 0 10px 0', 'fontSize': '14px',
                                   'opacity': '0.8'}),
                     html.H2(str(len(assets[assets['Risk_Level'] == 'Critical'])),
                             style={'color': colors['critical'], 'margin': '0', 'fontSize': '36px',
                                    'fontWeight': '700'})
                 ]),
        # High Risk
        html.Div(style={'backgroundColor': colors['card'], 'borderRadius': '12px', 'padding': '25px',
                        'border': f'1px solid {colors["high"]}', 'boxShadow': f'0 4px 20px rgba(245, 158, 11, 0.1)'},
                 children=[
                     html.P('High Risk Assets',
                            style={'color': colors['text'], 'margin': '0 0 10px 0', 'fontSize': '14px',
                                   'opacity': '0.8'}),
                     html.H2(str(len(assets[assets['Risk_Level'] == 'High'])),
                             style={'color': colors['high'], 'margin': '0', 'fontSize': '36px', 'fontWeight': '700'})
                 ]),
        # Avg Corrosion Rate
        html.Div(style={'backgroundColor': colors['card'], 'borderRadius': '12px', 'padding': '25px',
                        'border': f'1px solid {colors["secondary"]}',
                        'boxShadow': f'0 4px 20px rgba(124, 58, 237, 0.1)'}, children=[
            html.P('Avg Corrosion Rate',
                   style={'color': colors['text'], 'margin': '0 0 10px 0', 'fontSize': '14px', 'opacity': '0.8'}),
            html.H2(f'{assets["Corrosion_Rate"].mean():.2f} mm/yr',
                    style={'color': colors['secondary'], 'margin': '0', 'fontSize': '32px', 'fontWeight': '700'})
        ])
    ]),

    # Main Content Area
    html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px', 'marginBottom': '30px'},
             children=[
                 # Corrosion Rate Trend
                 html.Div(style={'backgroundColor': colors['card'], 'borderRadius': '12px', 'padding': '25px',
                                 'border': '1px solid rgba(255, 255, 255, 0.1)'}, children=[
                     html.H3('Corrosion Rate Trend Analysis',
                             style={'color': colors['text'], 'margin': '0 0 20px 0', 'fontSize': '18px'}),
                     dcc.Graph(id='corrosion-trend', config={'displayModeBar': False})
                 ]),

                 # Risk Score Over Time
                 html.Div(style={'backgroundColor': colors['card'], 'borderRadius': '12px', 'padding': '25px',
                                 'border': '1px solid rgba(255, 255, 255, 0.1)'}, children=[
                     html.H3('Asset Risk Score Timeline',
                             style={'color': colors['text'], 'margin': '0 0 20px 0', 'fontSize': '18px'}),
                     dcc.Graph(id='risk-trend', config={'displayModeBar': False})
                 ])
             ]),

    # Asset Analysis Section
    html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px', 'marginBottom': '30px'},
             children=[
                 # Risk Distribution
                 html.Div(style={'backgroundColor': colors['card'], 'borderRadius': '12px', 'padding': '25px',
                                 'border': '1px solid rgba(255, 255, 255, 0.1)'}, children=[
                     html.H3('Risk Level Distribution',
                             style={'color': colors['text'], 'margin': '0 0 20px 0', 'fontSize': '18px'}),
                     dcc.Graph(id='risk-distribution', config={'displayModeBar': False})
                 ]),

                 # Asset Type Analysis
                 html.Div(style={'backgroundColor': colors['card'], 'borderRadius': '12px', 'padding': '25px',
                                 'border': '1px solid rgba(255, 255, 255, 0.1)'}, children=[
                     html.H3('Corrosion by Asset Type',
                             style={'color': colors['text'], 'margin': '0 0 20px 0', 'fontSize': '18px'}),
                     dcc.Graph(id='asset-type-analysis', config={'displayModeBar': False})
                 ])
             ]),

    # Asset Table
    html.Div(style={'backgroundColor': colors['card'], 'borderRadius': '12px', 'padding': '25px',
                    'border': '1px solid rgba(255, 255, 255, 0.1)'}, children=[
        html.H3('Asset Monitoring Dashboard',
                style={'color': colors['text'], 'margin': '0 0 20px 0', 'fontSize': '18px'}),
        html.Div(id='asset-table')
    ])
])


# Callbacks for graphs
@app.callback(
    Output('corrosion-trend', 'figure'),
    Input('corrosion-trend', 'id')
)
def update_corrosion_trend(_):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pipeline_data['Date'],
        y=pipeline_data['Corrosion_Rate'],
        mode='lines+markers',
        name='Corrosion Rate',
        line=dict(color=colors['primary'], width=3),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor=f'rgba(0, 212, 255, 0.1)'
    ))
    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['card'],
        font=dict(color=colors['text']),
        xaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)', title='mm/year'),
        margin=dict(l=40, r=20, t=20, b=40),
        hovermode='x unified'
    )
    return fig


@app.callback(
    Output('risk-trend', 'figure'),
    Input('risk-trend', 'id')
)
def update_risk_trend(_):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pipeline_data['Date'],
        y=pipeline_data['Risk_Score'],
        mode='lines+markers',
        name='Risk Score',
        line=dict(color=colors['high'], width=3),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor=f'rgba(245, 158, 11, 0.1)'
    ))
    fig.add_hline(y=75, line_dash="dash", line_color=colors['critical'], annotation_text="Critical Threshold")
    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['card'],
        font=dict(color=colors['text']),
        xaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)', title='Risk Score', range=[0, 100]),
        margin=dict(l=40, r=20, t=20, b=40),
        hovermode='x unified'
    )
    return fig


@app.callback(
    Output('risk-distribution', 'figure'),
    Input('risk-distribution', 'id')
)
def update_risk_distribution(_):
    risk_counts = assets['Risk_Level'].value_counts()
    color_map = {'Critical': colors['critical'], 'High': colors['high'], 'Medium': colors['medium'],
                 'Low': colors['low']}

    fig = go.Figure(data=[go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        hole=0.4,
        marker=dict(colors=[color_map.get(level, colors['primary']) for level in risk_counts.index]),
        textfont=dict(size=14, color='white')
    )])
    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['card'],
        font=dict(color=colors['text']),
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=True,
        legend=dict(orientation="v", x=1.1, y=0.5)
    )
    return fig


@app.callback(
    Output('asset-type-analysis', 'figure'),
    Input('asset-type-analysis', 'id')
)
def update_asset_type(_):
    type_data = assets.groupby('Asset_Type')['Corrosion_Rate'].mean().reset_index()

    fig = go.Figure(data=[go.Bar(
        x=type_data['Asset_Type'],
        y=type_data['Corrosion_Rate'],
        marker=dict(color=colors['secondary'], line=dict(color=colors['primary'], width=2)),
        text=type_data['Corrosion_Rate'].round(2),
        textposition='outside',
        textfont=dict(color=colors['text'])
    )])
    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['card'],
        font=dict(color=colors['text']),
        xaxis=dict(showgrid=False, title='Asset Type'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)', title='Avg Corrosion Rate (mm/yr)'),
        margin=dict(l=40, r=20, t=20, b=40)
    )
    return fig


@app.callback(
    Output('asset-table', 'children'),
    Input('asset-table', 'id')
)
def update_table(_):
    color_map = {'Critical': colors['critical'], 'High': colors['high'], 'Medium': colors['medium'],
                 'Low': colors['low']}

    table_rows = []
    # Header
    table_rows.append(html.Tr([
        html.Th('Asset ID',
                style={'padding': '15px', 'textAlign': 'left', 'borderBottom': '2px solid rgba(255, 255, 255, 0.2)',
                       'color': colors['primary']}),
        html.Th('Type',
                style={'padding': '15px', 'textAlign': 'left', 'borderBottom': '2px solid rgba(255, 255, 255, 0.2)',
                       'color': colors['primary']}),
        html.Th('Location',
                style={'padding': '15px', 'textAlign': 'left', 'borderBottom': '2px solid rgba(255, 255, 255, 0.2)',
                       'color': colors['primary']}),
        html.Th('Risk Level',
                style={'padding': '15px', 'textAlign': 'center', 'borderBottom': '2px solid rgba(255, 255, 255, 0.2)',
                       'color': colors['primary']}),
        html.Th('Corrosion Rate',
                style={'padding': '15px', 'textAlign': 'right', 'borderBottom': '2px solid rgba(255, 255, 255, 0.2)',
                       'color': colors['primary']}),
        html.Th('Last Inspection',
                style={'padding': '15px', 'textAlign': 'right', 'borderBottom': '2px solid rgba(255, 255, 255, 0.2)',
                       'color': colors['primary']})
    ], style={'backgroundColor': 'rgba(255, 255, 255, 0.05)'}))

    # Data rows
    for _, row in assets.iterrows():
        table_rows.append(html.Tr([
            html.Td(row['Asset_ID'], style={'padding': '12px', 'borderBottom': '1px solid rgba(255, 255, 255, 0.1)',
                                            'color': colors['text']}),
            html.Td(row['Asset_Type'], style={'padding': '12px', 'borderBottom': '1px solid rgba(255, 255, 255, 0.1)',
                                              'color': colors['text']}),
            html.Td(row['Location'], style={'padding': '12px', 'borderBottom': '1px solid rgba(255, 255, 255, 0.1)',
                                            'color': colors['text']}),
            html.Td(html.Span(row['Risk_Level'], style={
                'padding': '6px 12px',
                'borderRadius': '20px',
                'backgroundColor': color_map.get(row['Risk_Level'], colors['primary']),
                'color': 'white',
                'fontSize': '12px',
                'fontWeight': '600'
            }), style={'padding': '12px', 'borderBottom': '1px solid rgba(255, 255, 255, 0.1)', 'textAlign': 'center'}),
            html.Td(f"{row['Corrosion_Rate']} mm/yr",
                    style={'padding': '12px', 'borderBottom': '1px solid rgba(255, 255, 255, 0.1)',
                           'color': colors['text'], 'textAlign': 'right'}),
            html.Td(row['Last_Inspection'],
                    style={'padding': '12px', 'borderBottom': '1px solid rgba(255, 255, 255, 0.1)',
                           'color': colors['text'], 'textAlign': 'right'})
        ]))

    return html.Table(table_rows, style={'width': '100%', 'borderCollapse': 'collapse'})


if __name__ == '__main__':
    app.run(debug=False, port=8050)