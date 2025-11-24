import base64
import io
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go

# =============================================================================
# Automata Intelligence – Corrosion Detection Prototype (Plotly Dash)
# This is a fully functional, professional-looking prototype you can run locally
# and demo to your oil & gas clients. It does NOT need real ML – everything is
# simulated realistically with random but believable corrosion spots.
# =============================================================================

app = dash.Dash(__name__,
                external_stylesheets=["https://cdn.jsdelivr.net/npm/bootswatch@5.3.3/dist/lumen/bootstrap.min.css"])
app.title = "Automata CorrosionAI"

# Random seed so the same image always gives the same (realistic) result during demo
random.seed(42)
np.random.seed(42)

app.layout = html.Div([
    html.Div([
        html.H1("Automata CorrosionAI", className="text-center mb-4 text-primary fw-bold"),
        html.H4("AI-Powered Corrosion Detection for Oil & Gas Assets", className="text-center text-secondary mb-5"),
    ], className="p-4 bg-dark text-white"),

    html.Div([
        html.Div([
            html.H3("1. Upload Inspection Image", className="mb-3"),
            dcc.Upload(
                id='upload-image',
                children=html.Div([
                    html.A('Drag & drop or click to select a pipeline / vessel / flange photo (JPG, PNG)',
                           className="text-decoration-none")
                ]),
                style={
                    'width': '100%', 'height': '200px', 'lineHeight': '200px',
                    'borderWidth': '3px', 'borderStyle': 'dashed', 'borderRadius': '15px',
                    'textAlign': 'center', 'backgroundColor': '#f8f9fa'
                },
                multiple=False
            ),
            html.Div(id='uploaded-image-display', className="mt-4 text-center")
        ], className="col-lg-6"),

        html.Div([
            html.H3("2. Run Detection", className="mb-4"),
            html.Button('🔍 Detect Corrosion', id='detect-button', n_clicks=0,
                        className="btn btn-danger btn-lg px-5"),
            html.Div(id='detection-results')
        ], className="col-lg-6 d-flex flex-column align-items-start justify-content-center")
    ], className="row mt-5 mx-3"),

    html.Hr(className="my-5"),

    html.Div(id='dashboard-section', className="mx-5")
], className="container-fluid")


# =============================================================================
# Callback 1: Show uploaded image immediately
# =============================================================================
@callback(
    Output('uploaded-image-display', 'children'),
    Input('upload-image', 'contents'),
    State('upload-image', 'filename')
)
def show_uploaded_image(contents, filename):
    if contents is None:
        return html.Div("No image uploaded yet", className="text-muted")

    return html.Div([
        html.H5(f"Uploaded: {filename}", className="text-success"),
        html.Img(src=contents,
                 style={'maxWidth': '100%', 'borderRadius': '10px', 'boxShadow': '0 4px 12px rgba(0,0,0,0.2)'})
    ])


# =============================================================================
# Callback 2: Run fake but realistic corrosion detection
# =============================================================================
@callback(
    Output('detection-results', 'children'),
    Output('dashboard-section', 'children'),
    Input('detect-button', 'n_clicks'),
    State('upload-image', 'contents'),
    State('upload-image', 'filename'),
    prevent_initial_call=True
)
def detect_corrosion(n_clicks, contents, filename):
    if contents is None:
        return html.Div("Please upload an image first", className="text-danger"), dash.no_update

    # Decode image
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    img = Image.open(io.BytesIO(decoded)).convert("RGBA")
    img_processed = img.copy()
    draw = ImageDraw.Draw(img_processed, "RGBA")

    width, height = img.size

    # Simulate realistic corrosion spots (3–9 spots, typical for real inspections)
    num_spots = random.randint(3, 9)
    corrosion_levels = []

    # Load a nice bold font if available, otherwise default
    try:
        font = ImageFont.truetype("arial.ttf", max(40, int(height / 15)))
        small_font = ImageFont.truetype("arial.ttf", max(28, int(height / 25)))
    except:
        font = ImageFont.load_default()
        small_font = font

    for _ in range(num_spots):
        # Random location and size
        cx = random.randint(int(width * 0.15), int(width * 0.85))
        cy = random.randint(int(height * 0.15), int(height * 0.85))
        radius = random.randint(int(height * 0.05), int(height * 0.18))

        # Random severity 15–98%
        severity = round(random.uniform(15, 98), 1)
        corrosion_levels.append(severity)

        # Red semi-transparent circle + dark border
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                     fill=(255, 0, 0, 140), outline=(180, 0, 0, 255), width=8)

        # Percentage label
        label = f"{severity}%"
        bbox = draw.textbbox((cx, cy), label, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        draw.rectangle([cx - w // 2 - 15, cy - h // 2 - 10, cx + w // 2 + 15, cy + h // 2 + 10], fill=(0, 0, 0, 200))
        draw.text((cx - w // 2, cy - h // 2), label, fill=(255, 255, 255, 255), font=font)

    # Convert processed image to base64
    buffered = io.BytesIO()
    img_processed.save(buffered, format="PNG")
    img_str = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()

    avg_corrosion = round(np.mean(corrosion_levels), 1)
    max_corrosion = round(max(corrosion_levels), 1)
    severity_level = "LOW" if avg_corrosion < 30 else "MEDIUM" if avg_corrosion < 60 else "HIGH"
    color = "green" if severity_level == "LOW" else "orange" if severity_level == "MEDIUM" else "red"

    # =============================================================================
    # Results panel (right side)
    # =============================================================================
    results_panel = html.Div([
        html.H4("Detection Complete ✓", className="text-success mb-4"),
        html.Img(src=img_str,
                 style={'maxWidth': '100%', 'borderRadius': '12px', 'boxShadow': '0 8px 20px rgba(0,0,0,0.3)'}),
        html.Div([
            html.Div([
                html.H5("Corrosion Sites", className="mb-0"),
                html.H2(f"{num_spots}", className="text-primary mb-0")
            ], className="col text-center"),
            html.Div([
                html.H5("Average Severity", className="mb-0"),
                html.H2(f"{avg_corrosion}%", className="text-danger mb-0")
            ], className="col text-center"),
            html.Div([
                html.H5("Highest Severity", className="mb-0"),
                html.H2(f"{max_corrosion}%", className="text-danger mb-0")
            ], className="col text-center"),
        ], className="row g-4 mt-3 text-center"),

        html.Hr(),
        html.H5("Risk Assessment", className="mt-3"),
        html.H3(severity_level, style={'color': color, 'fontWeight': 'bold'}),
        html.P([
            "Recommended action: ",
            html.Strong("Immediate intervention required – schedule pigging/repair within 30 days."
                        if severity_level == "HIGH" else
                        "Plan inspection within 90 days."
                        if severity_level == "MEDIUM" else
                        "Continue routine monitoring.")
        ], className="lead")
    ])

    # =============================================================================
    # Full dashboard (bottom section) – appears after detection
    # =============================================================================
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    historical = [random.uniform(5, 35) for _ in range(8)] + [avg_corrosion - 10, avg_corrosion - 5, avg_corrosion]

    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(x=months, y=historical, mode='lines+markers', name='Avg Corrosion %',
                                   line=dict(color='#dc3545', width=4), marker=dict(size=10)))
    trend_fig.add_hline(y=60, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
    trend_fig.update_layout(
        title=f"Corrosion Trend – Asset {filename[:15]}...",
        template="plotly_white",
        height=400,
        plot_bgcolor="#f8f9fa"
    )

    dashboard = html.Div([
        html.H2("Asset Dashboard", className="mb-4 text-primary"),
        html.Div([
            html.Div([
                html.H5("Asset ID", className="text-muted mb-0"),
                html.H4(filename[:20] if filename else "Unknown", className="mb-3"),
                html.H5("Inspection Date", className="text-muted mb-0"),
                html.H4("2025-04-05", className="mb-3"),
                html.H5("Location", className="text-muted mb-0"),
                html.H4("Gulf of Mexico – Platform Alpha", className="mb-0"),
            ], className="col-lg-3 bg-light p-4 rounded shadow"),

            html.Div([
                dcc.Graph(figure=trend_fig, className="shadow")
            ], className="col-lg-9")
        ], className="row")
    ], className="mt-5 p-4 bg-light rounded shadow")

    return results_panel, dashboard


# =============================================================================
# Run the app
# =============================================================================
if __name__ == '__main__':
    print("\n🚀 Automata CorrosionAI prototype is ready!")
    print("👉 Open http://127.0.0.1:8050 in your browser after starting.\n")
    app.run(debug=False, port=8050)


    # grok