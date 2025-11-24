# app.py
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import base64
import io
from PIL import Image

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Automata Intelligence - Corrosion Detection System"

# Generate sample data
np.random.seed(42)

# Sample pipeline data
pipelines = ['Pipeline A', 'Pipeline B', 'Pipeline C', 'Pipeline D', 'Pipeline E']
dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')

# Generate corrosion data
corrosion_data = pd.DataFrame({
    'Date': dates,
    'Pipeline A': np.random.normal(2.5, 0.5, len(dates)) + np.sin(np.arange(len(dates)) / 30) * 0.5,
    'Pipeline B': np.random.normal(3.0, 0.3, len(dates)) + np.cos(np.arange(len(dates)) / 25) * 0.3,
    'Pipeline C': np.random.normal(1.8, 0.4, len(dates)),
    'Pipeline D': np.random.normal(2.2, 0.6, len(dates)),
    'Pipeline E': np.random.normal(4.0, 0.7, len(dates)) + np.sin(np.arange(len(dates)) / 40) * 0.8,
})

# Alert data
alerts_data = pd.DataFrame({
    'Alert ID': ['ALT-001', 'ALT-002', 'ALT-003', 'ALT-004', 'ALT-005'],
    'Pipeline': ['Pipeline E', 'Pipeline B', 'Pipeline E', 'Pipeline A', 'Pipeline C'],
    'Severity': ['Critical', 'High', 'Medium', 'Low', 'Medium'],
    'Location': ['Section 12', 'Section 7', 'Section 3', 'Section 9', 'Section 15'],
    'Date': ['2024-01-01', '2023-12-30', '2023-12-28', '2023-12-27', '2023-12-25'],
    'Status': ['Active', 'Active', 'Under Review', 'Resolved', 'Active']
})

# Equipment status
equipment_data = pd.DataFrame({
    'Equipment': ['Sensor Unit 1', 'Sensor Unit 2', 'Sensor Unit 3', 'Drone Inspector A', 'Drone Inspector B'],
    'Status': ['Online', 'Online', 'Maintenance', 'Online', 'Online'],
    'Battery': [95, 87, 0, 78, 92],
    'Last Check': ['2 hours ago', '1 hour ago', '5 days ago', '30 mins ago', '45 mins ago']
})

# Custom CSS colors
colors = {
    'background': '#0f1214',
    'text': '#ffffff',
    'primary': '#00d4ff',
    'secondary': '#ff6b6b',
    'success': '#4ecdc4',
    'warning': '#ffd93d',
    'danger': '#ff6b6b',
    'card_bg': '#1a1f25',
    'border': '#2a3239'
}

# Define app layout
app.layout = html.Div(style={'backgroundColor': colors['background'], 'minHeight': '100vh'}, children=[
    # Header
    html.Div([
        html.Div([
            html.H1('AUTOMATA INTELLIGENCE',
                    style={'color': colors['primary'], 'margin': '0', 'fontSize': '28px', 'fontWeight': 'bold'}),
            html.P('Corrosion Detection & Monitoring System',
                   style={'color': colors['text'], 'margin': '0', 'fontSize': '14px', 'opacity': '0.8'})
        ], style={'display': 'inline-block', 'verticalAlign': 'middle'}),

        html.Div([
            html.Span('System Status: ', style={'color': colors['text'], 'marginRight': '10px'}),
            html.Span('● ONLINE', style={'color': colors['success'], 'fontWeight': 'bold'}),
            html.Span(' | ', style={'color': colors['text'], 'margin': '0 10px'}),
            html.Span(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                      style={'color': colors['text'], 'opacity': '0.8'})
        ], style={'display': 'inline-block', 'float': 'right', 'marginTop': '15px'})
    ], style={'padding': '20px', 'backgroundColor': colors['card_bg'],
              'borderBottom': f'2px solid {colors["primary"]}'}),

    # Main Dashboard Tabs
    dcc.Tabs(id='main-tabs', value='dashboard', children=[
        dcc.Tab(label='Dashboard Overview', value='dashboard',
                style={'backgroundColor': colors['card_bg'], 'color': colors['text']},
                selected_style={'backgroundColor': colors['primary'], 'color': colors['background']}),
        dcc.Tab(label='Corrosion Analysis', value='analysis',
                style={'backgroundColor': colors['card_bg'], 'color': colors['text']},
                selected_style={'backgroundColor': colors['primary'], 'color': colors['background']}),
        dcc.Tab(label='Image Detection', value='detection',
                style={'backgroundColor': colors['card_bg'], 'color': colors['text']},
                selected_style={'backgroundColor': colors['primary'], 'color': colors['background']}),
        dcc.Tab(label='Alerts & Reports', value='alerts',
                style={'backgroundColor': colors['card_bg'], 'color': colors['text']},
                selected_style={'backgroundColor': colors['primary'], 'color': colors['background']}),
    ], style={'margin': '20px'}),

    # Tab Content
    html.Div(id='tab-content', style={'padding': '20px'}),

    # Auto-refresh interval
    dcc.Interval(id='interval-component', interval=5 * 1000, n_intervals=0)
])


# Callback for tab content
@app.callback(Output('tab-content', 'children'),
              Input('main-tabs', 'value'))
def render_tab_content(tab):
    if tab == 'dashboard':
        return render_dashboard()
    elif tab == 'analysis':
        return render_analysis()
    elif tab == 'detection':
        return render_detection()
    elif tab == 'alerts':
        return render_alerts()


def render_dashboard():
    return html.Div([
        # KPI Cards Row
        html.Div([
            # Card 1: Total Pipelines
            html.Div([
                html.H3('Total Pipelines', style={'color': colors['text'], 'fontSize': '14px', 'opacity': '0.8'}),
                html.H2('5', style={'color': colors['primary'], 'fontSize': '36px', 'margin': '10px 0'}),
                html.P('↑ 100% Monitored', style={'color': colors['success'], 'fontSize': '12px'})
            ], style={'backgroundColor': colors['card_bg'], 'padding': '20px', 'borderRadius': '10px',
                      'border': f'1px solid {colors["border"]}', 'width': '22%', 'display': 'inline-block',
                      'marginRight': '2%'}),

            # Card 2: Critical Alerts
            html.Div([
                html.H3('Critical Alerts', style={'color': colors['text'], 'fontSize': '14px', 'opacity': '0.8'}),
                html.H2('1', style={'color': colors['danger'], 'fontSize': '36px', 'margin': '10px 0'}),
                html.P('Immediate attention required', style={'color': colors['danger'], 'fontSize': '12px'})
            ], style={'backgroundColor': colors['card_bg'], 'padding': '20px', 'borderRadius': '10px',
                      'border': f'1px solid {colors["border"]}', 'width': '22%', 'display': 'inline-block',
                      'marginRight': '2%'}),

            # Card 3: Average Corrosion Rate
            html.Div([
                html.H3('Avg Corrosion Rate', style={'color': colors['text'], 'fontSize': '14px', 'opacity': '0.8'}),
                html.H2('2.7 mm/y', style={'color': colors['warning'], 'fontSize': '36px', 'margin': '10px 0'}),
                html.P('↓ 5% from last month', style={'color': colors['success'], 'fontSize': '12px'})
            ], style={'backgroundColor': colors['card_bg'], 'padding': '20px', 'borderRadius': '10px',
                      'border': f'1px solid {colors["border"]}', 'width': '22%', 'display': 'inline-block',
                      'marginRight': '2%'}),

            # Card 4: System Health
            html.Div([
                html.H3('System Health', style={'color': colors['text'], 'fontSize': '14px', 'opacity': '0.8'}),
                html.H2('98%', style={'color': colors['success'], 'fontSize': '36px', 'margin': '10px 0'}),
                html.P('All systems operational', style={'color': colors['success'], 'fontSize': '12px'})
            ], style={'backgroundColor': colors['card_bg'], 'padding': '20px', 'borderRadius': '10px',
                      'border': f'1px solid {colors["border"]}', 'width': '22%', 'display': 'inline-block'})
        ], style={'marginBottom': '20px'}),

        # Charts Row
        html.Div([
            # Real-time Corrosion Monitor
            html.Div([
                dcc.Graph(
                    id='realtime-corrosion',
                    figure={
                        'data': [
                            go.Scatter(
                                x=corrosion_data['Date'][-30:],
                                y=corrosion_data[pipeline][-30:],
                                mode='lines',
                                name=pipeline,
                                line=dict(width=2)
                            ) for pipeline in pipelines
                        ],
                        'layout': go.Layout(
                            title='Real-time Corrosion Monitoring (Last 30 Days)',
                            titlefont={'color': colors['text']},
                            plot_bgcolor=colors['card_bg'],
                            paper_bgcolor=colors['card_bg'],
                            xaxis={'gridcolor': colors['border'], 'color': colors['text']},
                            yaxis={'title': 'Corrosion Rate (mm/year)', 'gridcolor': colors['border'],
                                   'color': colors['text']},
                            legend={'font': {'color': colors['text']}},
                            hovermode='x unified'
                        )
                    }
                )
            ], style={'width': '65%', 'display': 'inline-block', 'backgroundColor': colors['card_bg'],
                      'borderRadius': '10px', 'padding': '10px', 'marginRight': '2%'}),

            # Pipeline Health Status
            html.Div([
                dcc.Graph(
                    id='pipeline-health',
                    figure={
                        'data': [
                            go.Bar(
                                x=pipelines,
                                y=[85, 72, 90, 88, 65],
                                marker_color=[colors['success'], colors['warning'], colors['success'],
                                              colors['success'], colors['danger']],
                                text=[85, 72, 90, 88, 65],
                                textposition='auto',
                            )
                        ],
                        'layout': go.Layout(
                            title='Pipeline Health Score',
                            titlefont={'color': colors['text']},
                            plot_bgcolor=colors['card_bg'],
                            paper_bgcolor=colors['card_bg'],
                            xaxis={'gridcolor': colors['border'], 'color': colors['text']},
                            yaxis={'title': 'Health Score (%)', 'gridcolor': colors['border'], 'color': colors['text']},
                            showlegend=False
                        )
                    }
                )
            ], style={'width': '33%', 'display': 'inline-block', 'backgroundColor': colors['card_bg'],
                      'borderRadius': '10px', 'padding': '10px'})
        ], style={'marginBottom': '20px'}),

        # Equipment Status Table
        html.Div([
            html.H3('Equipment Status', style={'color': colors['text'], 'marginBottom': '15px'}),
            dash_table.DataTable(
                data=equipment_data.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in equipment_data.columns],
                style_cell={
                    'backgroundColor': colors['card_bg'],
                    'color': colors['text'],
                    'border': f'1px solid {colors["border"]}'
                },
                style_header={
                    'backgroundColor': colors['primary'],
                    'color': colors['background'],
                    'fontWeight': 'bold'
                },
                style_data_conditional=[
                    {
                        'if': {'column_id': 'Status', 'filter_query': '{Status} = "Online"'},
                        'color': colors['success']
                    },
                    {
                        'if': {'column_id': 'Status', 'filter_query': '{Status} = "Maintenance"'},
                        'color': colors['warning']
                    }
                ]
            )
        ], style={'backgroundColor': colors['card_bg'], 'padding': '20px', 'borderRadius': '10px'})
    ])


def render_analysis():
    return html.Div([
        # Analysis Controls
        html.Div([
            html.Label('Select Pipeline:', style={'color': colors['text'], 'marginRight': '10px'}),
            dcc.Dropdown(
                id='pipeline-select',
                options=[{'label': p, 'value': p} for p in pipelines],
                value='Pipeline A',
                style={'width': '200px', 'display': 'inline-block', 'marginRight': '20px'}
            ),
            html.Label('Time Range:', style={'color': colors['text'], 'marginRight': '10px', 'marginLeft': '20px'}),
            dcc.DatePickerRange(
                id='date-range',
                start_date='2023-10-01',
                end_date='2024-01-01',
                display_format='YYYY-MM-DD',
                style={'display': 'inline-block'}
            )
        ], style={'backgroundColor': colors['card_bg'], 'padding': '20px', 'borderRadius': '10px',
                  'marginBottom': '20px'}),

        # Analysis Charts
        html.Div([
            # Corrosion Trend Analysis
            html.Div([
                dcc.Graph(
                    id='trend-analysis',
                    figure={
                        'data': [
                            go.Scatter(
                                x=corrosion_data['Date'],
                                y=corrosion_data['Pipeline A'],
                                mode='lines+markers',
                                name='Actual',
                                line=dict(color=colors['primary'], width=2)
                            ),
                            go.Scatter(
                                x=corrosion_data['Date'],
                                y=corrosion_data['Pipeline A'].rolling(window=30).mean(),
                                mode='lines',
                                name='30-Day Moving Avg',
                                line=dict(color=colors['warning'], width=2, dash='dash')
                            ),
                            go.Scatter(
                                x=corrosion_data['Date'],
                                y=[3.5] * len(corrosion_data),
                                mode='lines',
                                name='Critical Threshold',
                                line=dict(color=colors['danger'], width=2, dash='dot')
                            )
                        ],
                        'layout': go.Layout(
                            title='Corrosion Trend Analysis - Pipeline A',
                            titlefont={'color': colors['text']},
                            plot_bgcolor=colors['card_bg'],
                            paper_bgcolor=colors['card_bg'],
                            xaxis={'gridcolor': colors['border'], 'color': colors['text']},
                            yaxis={'title': 'Corrosion Rate (mm/year)', 'gridcolor': colors['border'],
                                   'color': colors['text']},
                            legend={'font': {'color': colors['text']}},
                            hovermode='x unified'
                        )
                    }
                )
            ], style={'width': '100%', 'backgroundColor': colors['card_bg'], 'borderRadius': '10px',
                      'padding': '10px', 'marginBottom': '20px'}),
        ]),

        # Statistical Analysis
        html.Div([
            html.Div([
                html.H4('Statistical Summary', style={'color': colors['primary'], 'marginBottom': '15px'}),
                html.Div([
                    html.P(f"Mean Corrosion Rate: 2.51 mm/year", style={'color': colors['text']}),
                    html.P(f"Standard Deviation: 0.48 mm/year", style={'color': colors['text']}),
                    html.P(f"Maximum Rate: 4.12 mm/year", style={'color': colors['text']}),
                    html.P(f"Minimum Rate: 1.23 mm/year", style={'color': colors['text']}),
                    html.P(f"Predicted Next Month: 2.73 mm/year",
                           style={'color': colors['warning'], 'fontWeight': 'bold'})
                ])
            ], style={'width': '48%', 'display': 'inline-block', 'backgroundColor': colors['card_bg'],
                      'borderRadius': '10px', 'padding': '20px', 'marginRight': '2%'}),

            # Risk Assessment
            html.Div([
                html.H4('Risk Assessment', style={'color': colors['primary'], 'marginBottom': '15px'}),
                dcc.Graph(
                    id='risk-gauge',
                    figure={
                        'data': [go.Indicator(
                            mode="gauge+number+delta",
                            value=72,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Risk Score", 'font': {'color': colors['text']}},
                            delta={'reference': 68, 'increasing': {'color': colors['danger']}},
                            gauge={
                                'axis': {'range': [None, 100], 'tickcolor': colors['text']},
                                'bar': {'color': colors['warning']},
                                'steps': [
                                    {'range': [0, 50], 'color': colors['success']},
                                    {'range': [50, 80], 'color': colors['warning']},
                                    {'range': [80, 100], 'color': colors['danger']}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        )],
                        'layout': go.Layout(
                            paper_bgcolor=colors['card_bg'],
                            font={'color': colors['text']},
                            height=250
                        )
                    }
                )
            ], style={'width': '48%', 'display': 'inline-block', 'backgroundColor': colors['card_bg'],
                      'borderRadius': '10px', 'padding': '20px'})
        ])
    ])


def render_detection():
    return html.Div([
        # Image Upload Section
        html.Div([
            html.H3('AI-Powered Corrosion Detection', style={'color': colors['primary'], 'marginBottom': '20px'}),
            dcc.Upload(
                id='upload-image',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files', style={'color': colors['primary'], 'textDecoration': 'underline'})
                ]),
                style={
                    'width': '100%',
                    'height': '100px',
                    'lineHeight': '100px',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '10px',
                    'borderColor': colors['primary'],
                    'textAlign': 'center',
                    'backgroundColor': colors['card_bg'],
                    'color': colors['text'],
                    'cursor': 'pointer'
                },
                multiple=False
            ),
            html.Div(id='output-image-upload', style={'marginTop': '20px'})
        ], style={'backgroundColor': colors['card_bg'], 'padding': '20px', 'borderRadius': '10px',
                  'marginBottom': '20px'}),

        # Sample Detection Results
        html.Div([
            html.H3('Recent Detection Results', style={'color': colors['primary'], 'marginBottom': '20px'}),
            html.Div([
                # Sample Result 1
                html.Div([
                    html.Div(style={
                        'width': '200px',
                        'height': '150px',
                        'backgroundColor': colors['border'],
                        'borderRadius': '10px',
                        'marginBottom': '10px'
                    }),
                    html.P('Pipeline B - Section 7', style={'color': colors['text'], 'fontWeight': 'bold'}),
                    html.P('Detected: Surface Corrosion', style={'color': colors['warning'], 'fontSize': '14px'}),
                    html.P('Confidence: 94.2%', style={'color': colors['text'], 'fontSize': '14px'}),
                    html.P('Severity: Medium', style={'color': colors['warning'], 'fontSize': '14px'})
                ], style={'width': '23%', 'display': 'inline-block', 'backgroundColor': colors['card_bg'],
                          'padding': '15px', 'borderRadius': '10px', 'marginRight': '2%',
                          'border': f'1px solid {colors["border"]}'}),

                # Sample Result 2
                html.Div([
                    html.Div(style={
                        'width': '200px',
                        'height': '150px',
                        'backgroundColor': colors['border'],
                        'borderRadius': '10px',
                        'marginBottom': '10px'
                    }),
                    html.P('Pipeline E - Section 12', style={'color': colors['text'], 'fontWeight': 'bold'}),
                    html.P('Detected: Pitting Corrosion', style={'color': colors['danger'], 'fontSize': '14px'}),
                    html.P('Confidence: 97.8%', style={'color': colors['text'], 'fontSize': '14px'}),
                    html.P('Severity: Critical', style={'color': colors['danger'], 'fontSize': '14px'})
                ], style={'width': '23%', 'display': 'inline-block', 'backgroundColor': colors['card_bg'],
                          'padding': '15px', 'borderRadius': '10px', 'marginRight': '2%',
                          'border': f'1px solid {colors["danger"]}'}),

                # Sample Result 3
                html.Div([
                    html.Div(style={
                        'width': '200px',
                        'height': '150px',
                        'backgroundColor': colors['border'],
                        'borderRadius': '10px',
                        'marginBottom': '10px'
                    }),
                    html.P('Pipeline A - Section 3', style={'color': colors['text'], 'fontWeight': 'bold'}),
                    html.P('Detected: Minor Rust', style={'color': colors['success'], 'fontSize': '14px'}),
                    html.P('Confidence: 89.1%', style={'color': colors['text'], 'fontSize': '14px'}),
                    html.P('Severity: Low', style={'color': colors['success'], 'fontSize': '14px'})
                ], style={'width': '23%', 'display': 'inline-block', 'backgroundColor': colors['card_bg'],
                          'padding': '15px', 'borderRadius': '10px', 'marginRight': '2%',
                          'border': f'1px solid {colors["border"]}'}),

                # Sample Result 4
                html.Div([
                    html.Div(style={
                        'width': '200px',
                        'height': '150px',
                        'backgroundColor': colors['border'],
                        'borderRadius': '10px',
                        'marginBottom': '10px'
                    }),
                    html.P('Pipeline C - Section 9', style={'color': colors['text'], 'fontWeight': 'bold'}),
                    html.P('Detected: No Corrosion', style={'color': colors['success'], 'fontSize': '14px'}),
                    html.P('Confidence: 99.2%', style={'color': colors['text'], 'fontSize': '14px'}),
                    html.P('Severity: None', style={'color': colors['success'], 'fontSize': '14px'})
                ], style={'width': '23%', 'display': 'inline-block', 'backgroundColor': colors['card_bg'],
                          'padding': '15px', 'borderRadius': '10px',
                          'border': f'1px solid {colors["border"]}'})
            ])
        ], style={'backgroundColor': colors['card_bg'], 'padding': '20px', 'borderRadius': '10px'}),

        # AI Model Performance
        html.Div([
            html.H3('AI Model Performance', style={'color': colors['primary'], 'marginBottom': '20px'}),
            html.Div([
                html.Div([
                    html.P('Model Accuracy', style={'color': colors['text'], 'fontSize': '14px'}),
                    html.H2('96.5%', style={'color': colors['success'], 'margin': '5px 0'})
                ], style={'width': '24%', 'display': 'inline-block', 'textAlign': 'center'}),
                html.Div([
                    html.P('Images Processed', style={'color': colors['text'], 'fontSize': '14px'}),
                    html.H2('12,847', style={'color': colors['primary'], 'margin': '5px 0'})
                ], style={'width': '24%', 'display': 'inline-block', 'textAlign': 'center'}),
                html.Div([
                    html.P('Avg Processing Time', style={'color': colors['text'], 'fontSize': '14px'}),
                    html.H2('0.3s', style={'color': colors['primary'], 'margin': '5px 0'})
                ], style={'width': '24%', 'display': 'inline-block', 'textAlign': 'center'}),
                html.Div([
                    html.P('False Positive Rate', style={'color': colors['text'], 'fontSize': '14px'}),
                    html.H2('2.1%', style={'color': colors['warning'], 'margin': '5px 0'})
                ], style={'width': '24%', 'display': 'inline-block', 'textAlign': 'center'})
            ])
        ], style={'backgroundColor': colors['card_bg'], 'padding': '20px', 'borderRadius': '10px', 'marginTop': '20px'})
    ])


def render_alerts():
    return html.Div([
        # Active Alerts
        html.Div([
            html.H3('Active Alerts', style={'color': colors['primary'], 'marginBottom': '20px'}),
            dash_table.DataTable(
                data=alerts_data.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in alerts_data.columns],
                style_cell={
                    'backgroundColor': colors['card_bg'],
                    'color': colors['text'],
                    'border': f'1px solid {colors["border"]}'
                },
                style_header={
                    'backgroundColor': colors['primary'],
                    'color': colors['background'],
                    'fontWeight': 'bold'
                },
                style_data_conditional=[
                    {
                        'if': {'column_id': 'Severity', 'filter_query': '{Severity} = "Critical"'},
                        'backgroundColor': colors['danger'],
                        'color': colors['text']
                    },
                    {
                        'if': {'column_id': 'Severity', 'filter_query': '{Severity} = "High"'},
                        'backgroundColor': colors['warning'],
                        'color': colors['background']
                    },
                    {
                        'if': {'column_id': 'Status', 'filter_query': '{Status} = "Active"'},
                        'fontWeight': 'bold'
                    }
                ],
                page_size=10
            )
        ], style={'backgroundColor': colors['card_bg'], 'padding': '20px', 'borderRadius': '10px',
                  'marginBottom': '20px'}),

        # Reports Section
        html.Div([
            html.H3('Generate Reports', style={'color': colors['primary'], 'marginBottom': '20px'}),
            html.Div([
                html.Label('Report Type:', style={'color': colors['text'], 'marginRight': '10px'}),
                dcc.Dropdown(
                    id='report-type',
                    options=[
                        {'label': 'Daily Summary', 'value': 'daily'},
                        {'label': 'Weekly Analysis', 'value': 'weekly'},
                        {'label': 'Monthly Report', 'value': 'monthly'},
                        {'label': 'Custom Range', 'value': 'custom'}
                    ],
                    value='weekly',
                    style={'width': '200px', 'display': 'inline-block', 'marginRight': '20px'}
                ),
                html.Button('Generate Report',
                            style={
                                'backgroundColor': colors['primary'],
                                'color': colors['background'],
                                'border': 'none',
                                'padding': '10px 20px',
                                'borderRadius': '5px',
                                'cursor': 'pointer',
                                'fontWeight': 'bold',
                                'marginLeft': '20px'
                            }),
                html.Button('Export to PDF',
                            style={
                                'backgroundColor': colors['success'],
                                'color': colors['background'],
                                'border': 'none',
                                'padding': '10px 20px',
                                'borderRadius': '5px',
                                'cursor': 'pointer',
                                'fontWeight': 'bold',
                                'marginLeft': '10px'
                            })
            ], style={'marginBottom': '20px'}),

            # Recent Reports
            html.H4('Recent Reports', style={'color': colors['text'], 'marginTop': '30px', 'marginBottom': '15px'}),
            html.Div([
                html.Div([
                    html.P('📄 Monthly Report - December 2023', style={'color': colors['text']}),
                    html.P('Generated: 2024-01-01 09:00',
                           style={'color': colors['text'], 'opacity': '0.7', 'fontSize': '12px'})
                ], style={'borderBottom': f'1px solid {colors["border"]}', 'paddingBottom': '10px',
                          'marginBottom': '10px'}),
                html.Div([
                    html.P('📄 Weekly Analysis - Week 52', style={'color': colors['text']}),
                    html.P('Generated: 2023-12-31 18:00',
                           style={'color': colors['text'], 'opacity': '0.7', 'fontSize': '12px'})
                ], style={'borderBottom': f'1px solid {colors["border"]}', 'paddingBottom': '10px',
                          'marginBottom': '10px'}),
                html.Div([
                    html.P('📄 Critical Alert Report - Pipeline E', style={'color': colors['text']}),
                    html.P('Generated: 2023-12-30 14:30',
                           style={'color': colors['text'], 'opacity': '0.7', 'fontSize': '12px'})
                ], style={'borderBottom': f'1px solid {colors["border"]}', 'paddingBottom': '10px',
                          'marginBottom': '10px'})
            ])
        ], style={'backgroundColor': colors['card_bg'], 'padding': '20px', 'borderRadius': '10px'})
    ])


# Callback for image upload
@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'))
def update_output(contents, filename):
    if contents is not None:
        # Simulate AI processing
        import time
        import random

        # Mock detection results
        confidence = random.uniform(85, 99)
        corrosion_types = ['Surface Corrosion', 'Pitting Corrosion', 'Crevice Corrosion', 'Minor Rust']
        detected_type = random.choice(corrosion_types)
        severity = random.choice(['Low', 'Medium', 'High', 'Critical'])

        severity_colors = {
            'Low': colors['success'],
            'Medium': colors['warning'],
            'High': colors['warning'],
            'Critical': colors['danger']
        }

        return html.Div([
            html.H4(f'Analysis Results for {filename}', style={'color': colors['primary'], 'marginBottom': '15px'}),
            html.Img(src=contents, style={'maxWidth': '500px', 'borderRadius': '10px', 'marginBottom': '20px'}),
            html.Div([
                html.P(f'✅ Analysis Complete',
                       style={'color': colors['success'], 'fontWeight': 'bold', 'fontSize': '16px'}),
                html.P(f'Detected: {detected_type}', style={'color': colors['text'], 'marginTop': '10px'}),
                html.P(f'Confidence: {confidence:.1f}%', style={'color': colors['text']}),
                html.P(f'Severity: {severity}', style={'color': severity_colors[severity], 'fontWeight': 'bold'}),
                html.P(
                    f'Recommended Action: Schedule maintenance within {"24 hours" if severity == "Critical" else "1 week" if severity == "High" else "1 month"}',
                    style={'color': colors['text'], 'marginTop': '10px', 'padding': '10px',
                           'backgroundColor': colors['border'], 'borderRadius': '5px'})
            ])
        ])


if __name__ == '__main__':
    app.run(debug=True)

    # opus 4.1