# app.py
# MVP: AI-driven Well Log Interpretation (Plotly Dash)
# Features: synthetic well logs, Vsh, porosity, Sw (Archie), permeability, ML facies, tops detection, multi-track plots, crossplots, KPIs, export.
# Install: pip install dash dash-bootstrap-components plotly scikit-learn pandas numpy

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.cluster import KMeans

from dash import Dash, dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
from dash.dash_table import DataTable
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------
# ---- Synthetic data model ----
# ------------------------------

@dataclass
class ArchieParams:
    a: float = 1.0
    m: float = 2.0
    n: float = 2.0
    Rw: float = 0.08  # ohm-m

@dataclass
class MatrixParams:
    name: str = "Limestone"
    rho_ma: float = 2.71  # g/cc
    rho_f: float = 1.0    # g/cc freshwater
    phi_shale: float = 0.3

def seed_all(seed=42):
    np.random.seed(seed)

def generate_synthetic_wells(seed=42):
    """
    Create 3 wells with realistic log behavior across 600 m interval.
    Logs: GR(API), RT(ohm-m), RHOB(g/cc), NPHI(v/v), DT(us/ft), PEF(b/e), CALI(in)
    Also returns "TRUE_FACIES", "SW_TRUE", "PHI_TRUE", "VSH_TRUE" for validation/QC.
    """
    seed_all(seed)
    wells = {}
    settings = [
        # name, top, base, step, Rw, style
        ("Well_A_Fluvial", 2400, 3000, 0.5, 0.08, "sand_prone"),
        ("Well_B_Carbonate", 2500, 3100, 0.5, 0.05, "carbonate"),
        ("Well_C_ShalyCh", 2350, 2950, 0.5, 0.10, "shaly_channel"),
    ]

    for name, top, base, step, rw, style in settings:
        depth = np.arange(top, base + step, step)
        n = len(depth)

        # Define facies sequence via bed thicknesses and Markov-style transitions
        facies_list = ["Shale", "Sandstone", "Limestone"]
        facies_idx = {"Shale": 0, "Sandstone": 1, "Limestone": 2}

        # Style-dependent facies probabilities
        if style == "sand_prone":
            p = [0.25, 0.55, 0.20]
        elif style == "carbonate":
            p = [0.20, 0.30, 0.50]
        else:  # shaly_channel
            p = [0.50, 0.35, 0.15]

        # Generate beds
        facies = []
        i = 0
        while i < n:
            chosen = np.random.choice(facies_list, p=p)
            thickness = int(np.clip(np.random.gamma(shape=2, scale=6), 4, 60) / step)
            facies.extend([chosen] * min(thickness, n - i))
            i += thickness
        facies = np.array(facies[:n])

        # Create a couple of HC-enriched zones to emulate pay (lower Sw)
        hc_mask = np.zeros(n, dtype=bool)
        for _ in range(2):
            start = np.random.randint(int(n * 0.15), int(n * 0.85))
            width = int(np.clip(np.random.normal(80/step, 40/step), 40/step, 160/step))
            hc_mask[start:start+width] = True

        # Baseline parameters by facies
        rho_ma_map = {"Sandstone": 2.65, "Limestone": 2.71, "Shale": 2.5}
        dt_ma_map = {"Sandstone": 55.5, "Limestone": 47.6, "Shale": 80.0}  # us/ft
        pef_map   = {"Sandstone": 1.8, "Limestone": 5.1, "Shale": 3.5}
        gr_clean_map = {"Sandstone": 40, "Limestone": 20, "Shale": 120}

        # True porosity distribution
        phi_true = np.zeros(n)
        vsh_true = np.zeros(n)
        sw_true = np.zeros(n)
        rho_ma_vec = np.zeros(n)
        dt_ma_vec = np.zeros(n)
        pef_mat = np.zeros(n)
        gr = np.zeros(n)

        # Depth trend: slight porosity decrease with depth
        trend = (depth - depth.min()) / (depth.max() - depth.min())  # 0 to 1

        for i in range(n):
            f = facies[i]
            rho_ma_vec[i] = rho_ma_map[f]
            dt_ma_vec[i] = dt_ma_map[f]
            pef_mat[i] = pef_map[f]

            if f == "Shale":
                phi_true[i] = np.clip(np.random.normal(0.07, 0.025), 0.02, 0.14)
                vsh_true[i] = np.clip(np.random.normal(0.8, 0.1), 0.5, 1.0)
            elif f == "Sandstone":
                base_phi = np.clip(np.random.normal(0.22, 0.05), 0.1, 0.32)
                phi_true[i] = np.clip(base_phi * (1 - 0.15*trend[i]) + np.random.normal(0, 0.01), 0.06, 0.34)
                vsh_true[i] = np.clip(np.random.normal(0.18, 0.1), 0.0, 0.45)
            else:  # Limestone
                base_phi = np.clip(np.random.normal(0.12, 0.05), 0.02, 0.22)
                phi_true[i] = np.clip(base_phi * (1 - 0.10*trend[i]) + np.random.normal(0, 0.01), 0.02, 0.25)
                vsh_true[i] = np.clip(np.random.normal(0.10, 0.08), 0.0, 0.35)

            # True Sw: high in shale, variable in sands/carbs; lower in HC zones
            if f == "Shale":
                sw_true[i] = np.clip(np.random.normal(0.98, 0.03), 0.8, 1.0)
            else:
                if hc_mask[i]:
                    sw_true[i] = np.clip(np.random.normal(0.35, 0.12), 0.05, 0.7)
                else:
                    sw_true[i] = np.clip(np.random.normal(0.7 if style != "carbonate" else 0.6, 0.15), 0.3, 1.0)

            # GR weighted by shale fraction
            gr_matrix = gr_clean_map[f]
            gr[i] = np.clip(gr_matrix*(1 - vsh_true[i]) + 140*vsh_true[i] + np.random.normal(0, 3), 5, 180)

        # Fluid density for density calculation
        rho_f = 1.0

        # Compute logs
        rhob = phi_true * rho_f + (1 - phi_true) * rho_ma_vec + np.random.normal(0, 0.015, n)
        # Neutron porosity in limestone units (approx), with shale effect
        nphi = np.clip(phi_true + (vsh_true * np.random.uniform(0.05, 0.12)) + np.random.normal(0, 0.01, n), 0.0, 0.6)
        # Sonic slowness via Wyllie time-average (approx)
        dt_f = 189.0  # us/ft
        dt = phi_true * dt_f + (1 - phi_true) * dt_ma_vec + np.random.normal(0, 1.2, n)
        # Photoelectric factor weighted
        pef = np.clip(pef_mat*(1 - vsh_true) + 3.8*vsh_true + np.random.normal(0, 0.2, n), 1.2, 6.5)

        # Resistivity from Archie (invert Sw)
        archie_a, archie_m, archie_n = 1.0, 2.0, 2.0
        phi_eff_for_rt = np.clip(phi_true, 0.02, 0.35)
        rt = (archie_a * rw) / (np.power(phi_eff_for_rt, archie_m) * np.power(np.clip(sw_true, 0.05, 1.0), archie_n))
        # Add realistic noise and clamp
        rt = np.clip(rt * np.exp(np.random.normal(0, 0.15, n)), 0.2, 500)

        # Caliper: washouts mainly in shale
        cali = np.clip(8.5 + (vsh_true * np.random.uniform(0.5, 1.5, n)) + np.random.normal(0, 0.1, n), 8.0, 12.0)

        df = pd.DataFrame({
            "WELL": name,
            "DEPTH": depth,
            "GR": gr,
            "RT": rt,
            "RHOB": rhob,
            "NPHI": nphi,
            "DT": dt,
            "PEF": pef,
            "CALI": cali,
            "TRUE_FACIES": facies,
            "PHI_TRUE": phi_true,
            "VSH_TRUE": vsh_true,
            "SW_TRUE": sw_true,
            "HC_ZONE": hc_mask
        })
        df["STEP"] = np.gradient(df["DEPTH"])
        df["STEP"] = np.where(df["STEP"] <= 0, np.median(np.diff(depth)), df["STEP"])
        wells[name] = df

    return wells

# ------------------------------
# ---- Petrophysical logic -----
# ------------------------------

def larionov_tertiary_vsh(gr, gr_min, gr_max):
    igr = np.clip((gr - gr_min) / max(gr_max - gr_min, 1e-6), 0, 1.5)
    vsh = 0.083 * (np.power(2, 3.7 * igr) - 1)
    return np.clip(vsh, 0, 1.0)

def linear_vsh(gr, gr_min, gr_max):
    igr = np.clip((gr - gr_min) / max(gr_max - gr_min, 1e-6), 0, 1.0)
    return igr

def density_porosity(rhob, rho_ma=2.71, rho_f=1.0):
    denom = max(rho_ma - rho_f, 1e-6)
    phi_d = (rho_ma - rhob) / denom
    return np.clip(phi_d, 0.0, 0.5)

def combined_porosity(phi_d, nphi, vsh, phi_shale=0.3):
    # Simple shale correction on neutron, then combine
    nphi_corr = np.clip(nphi - vsh * phi_shale, 0.0, 0.6)
    phi = np.clip(0.5 * (phi_d + nphi_corr), 0.0, 0.5)
    # Effective porosity after shale correction on total
    phi_eff = np.clip(phi - vsh * phi_shale, 0.0, 0.5)
    return phi, phi_eff, nphi_corr

def archie_sw(rt, phi_eff, params: ArchieParams):
    phi_eff = np.clip(phi_eff, 0.02, 0.5)
    rt = np.clip(rt, 0.2, 1000.0)
    sw = ((params.a * params.Rw) / (rt * (phi_eff ** params.m))) ** (1.0 / params.n)
    return np.clip(sw, 0.0, 1.2)

def estimate_perm(phi_eff, sw):
    # Simple Timur-like trend (MVP). Avoid blow-ups at low Sw.
    return np.clip(10000.0 * (phi_eff ** 3) / (np.maximum(sw, 0.1) ** 2), 0.01, 5000.0)

def smooth_series(series, window=5):
    if window <= 1:
        return series.values
    return pd.Series(series).rolling(window=window, min_periods=1, center=True).mean().values

def detect_tops(depth, gr, rt, window=9, gr_threshold=18, sep_m=8):
    # Basic change-point heuristic: large GR changes coupled with RT contrast.
    dgr = pd.Series(gr).diff().rolling(window=window, center=True).mean().abs()
    drt = pd.Series(np.log10(np.clip(rt, 0.2, 1000))).diff().rolling(window=window, center=True).mean().abs()
    score = dgr / (np.nanmax(dgr)+1e-6) + drt / (np.nanmax(drt)+1e-6)
    idxs = np.where(score > 0.9)[0]  # high contrast
    tops = []
    last_depth = -1e9
    for i in idxs:
        z = depth[i]
        if z - last_depth > sep_m:
            tops.append(float(z))
            last_depth = z
    return tops[:12]  # cap

def ml_facies(df, n_clusters=3, random_state=8):
    feats = df[["GR", "RT", "RHOB", "NPHI", "PEF"]].copy()
    # log-scale RT for clustering
    feats["LOGRT"] = np.log10(np.clip(df["RT"].values, 0.2, 1000))
    feats = feats.drop(columns=["RT"])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(feats.values)
    # Name clusters by heuristic
    facies_names = []
    for c in range(n_clusters):
        mask = labels == c
        g = df.loc[mask, "GR"].median()
        rh = df.loc[mask, "RHOB"].median()
        pe = df.loc[mask, "PEF"].median()
        # Rough rules
        if g > 85:
            facies_names.append("Shale")
        elif pe > 3.7 and rh > 2.65:
            facies_names.append("Limestone")
        else:
            facies_names.append("Sandstone")
    pred_names = [facies_names[l] for l in labels]
    return labels, pred_names

def compute_interpretation(df, gr_min, gr_max, vsh_method, archie: ArchieParams, matrix: MatrixParams, smooth_win=5,
                           ml_on=True, ml_k=3):
    # Vsh
    if vsh_method == "Larionov (Tertiary)":
        vsh = larionov_tertiary_vsh(df["GR"].values, gr_min, gr_max)
    else:
        vsh = linear_vsh(df["GR"].values, gr_min, gr_max)

    # Porosity
    phi_d = density_porosity(df["RHOB"].values, rho_ma=matrix.rho_ma, rho_f=matrix.rho_f)
    phi, phi_eff, nphi_corr = combined_porosity(phi_d, df["NPHI"].values, vsh, phi_shale=matrix.phi_shale)

    # Sw, k
    sw = archie_sw(df["RT"].values, phi_eff, archie)
    k = estimate_perm(phi_eff, sw)

    # Optional smoothing
    vsh_s = smooth_series(vsh, smooth_win)
    phi_s = smooth_series(phi, smooth_win)
    phi_eff_s = smooth_series(phi_eff, smooth_win)
    sw_s = smooth_series(sw, smooth_win)
    k_s = smooth_series(k, smooth_win)

    out = df.copy()
    out["VSH"] = vsh_s
    out["PHI_T"] = phi_s
    out["PHI_E"] = phi_eff_s
    out["SW"] = sw_s
    out["K_MDK"] = k_s
    out["NPHI_CORR"] = nphi_corr

    if ml_on:
        _, facies_ml = ml_facies(out, n_clusters=ml_k)
        out["FACIES"] = facies_ml
    else:
        out["FACIES"] = "Unknown"

    # Tops detection
    tops = detect_tops(out["DEPTH"].values, out["GR"].values, out["RT"].values)

    return out, tops

def compute_pay_flags(df, vsh_cut=0.35, phi_cut=0.08, sw_cut=0.6, rt_cut=2.0):
    pay = (df["VSH"] <= vsh_cut) & (df["PHI_E"] >= phi_cut) & (df["SW"] <= sw_cut) & (df["RT"] >= rt_cut)
    return pay

def pay_metrics(df, pay_mask):
    dz = df["STEP"].values
    gross = float(np.nansum(dz))
    net = float(np.nansum(dz[pay_mask]))
    ntg = float(net / gross) if gross > 0 else 0.0
    phi_pay = float(np.nanmean(df.loc[pay_mask, "PHI_E"])) if pay_mask.any() else np.nan
    sw_pay = float(np.nanmean(df.loc[pay_mask, "SW"])) if pay_mask.any() else np.nan
    # HCPVI (index): sum(phi*(1-Sw)*dz)
    hcpvi = float(np.nansum((df["PHI_E"].values * (1 - df["SW"].values)) * dz))
    return gross, net, ntg, phi_pay, sw_pay, hcpvi

# ------------------------------
# --------- Plotting -----------
# ------------------------------

def main_logs_figure(df, depth_range, vsh_cut, rt_cut, tops):
    # Filter range
    zmin, zmax = depth_range
    view = df[(df["DEPTH"] >= zmin) & (df["DEPTH"] <= zmax)].copy()
    if view.empty:
        view = df.copy()

    # Build subplots: GR/VSH, RT, PORO/SW, Facies
    fig = make_subplots(
        rows=1, cols=4, shared_yaxes=True,
        horizontal_spacing=0.03,
        subplot_titles=("Gamma Ray & Vsh", "Resistivity (log)", "Porosity & Sw", "Lithology / Pay"),
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}, {"type": "xy"}]]
    )

    y = view["DEPTH"].values

    # Track 1: GR and VSH
    fig.add_trace(go.Scatter(x=view["GR"], y=y, mode="lines", name="GR [API]", line=dict(color="#2ecc71", width=2),
                             hovertemplate="Depth: %{y:.1f} m<br>GR: %{x:.1f} API<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=view["VSH"]*150, y=y, mode="lines", name="Vsh (scaled)", line=dict(color="#27ae60", dash="dash"),
                             hovertemplate="Depth: %{y:.1f} m<br>Vsh: %{text:.2f}<extra></extra>",
                             text=np.round(view["VSH"], 2)), row=1, col=1)
    # Fill to emphasize shale-rich zones
    fig.add_trace(go.Scatter(x=np.where(view["VSH"] > vsh_cut, view["GR"], np.nan),
                             y=y, mode="lines", line=dict(color="rgba(231, 76, 60,0.5)", width=6),
                             name=f"Vsh > {vsh_cut:g}", hoverinfo="skip"), row=1, col=1)
    fig.update_xaxes(title="GR / Vsh", range=[0, 150], row=1, col=1)

    # Track 2: Resistivity (log)
    fig.add_trace(go.Scatter(x=view["RT"], y=y, mode="lines", name="RT [ohm·m]",
                             line=dict(color="#f39c12", width=2),
                             hovertemplate="Depth: %{y:.1f} m<br>RT: %{x:.2f} ohm·m<extra></extra>"), row=1, col=2)
    # Shade high-RT potential pay (visual cue)
    fig.add_trace(go.Scatter(x=np.where(view["RT"] > rt_cut, view["RT"], np.nan), y=y,
                             mode="lines", line=dict(color="rgba(243,156,18,0.5)", width=6),
                             name=f"RT > {rt_cut:g}", hoverinfo="skip"), row=1, col=2)
    fig.update_xaxes(type="log", title="RT", range=[-0.7, 2.7], row=1, col=2)  # ~0.2 to 500

    # Track 3: Porosity & Sw
    fig.add_trace(go.Scatter(x=view["PHI_E"]*100, y=y, mode="lines", name="Phi_eff [%]",
                             line=dict(color="#3498db", width=2),
                             hovertemplate="Depth: %{y:.1f} m<br>Phi_e: %{text:.2f}<extra></extra>",
                             text=np.round(view["PHI_E"], 2)), row=1, col=3)
    fig.add_trace(go.Scatter(x=view["SW"]*100, y=y, mode="lines", name="Sw [%]",
                             line=dict(color="#9b59b6", width=2, dash="dash"),
                             hovertemplate="Depth: %{y:.1f} m<br>Sw: %{text:.2f}<extra></extra>",
                             text=np.round(view["SW"], 2)), row=1, col=3)
    fig.update_xaxes(title="% (Phi_e / Sw)", range=[0, 100], row=1, col=3)

    # Track 4: Lithology/Pay track using colored markers
    color_map = {"Sandstone": "#F4D03F", "Limestone": "#85C1E9", "Shale": "#7DCEA0", "Unknown": "#95A5A6"}
    colors = [color_map.get(v, "#95A5A6") for v in view["FACIES"]]
    pay_mask = view.get("PAY", pd.Series(False, index=view.index))
    pay_color = np.where(pay_mask, "#e74c3c", "#2c3e50")
    fig.add_trace(go.Scatter(x=np.where(pay_mask, 1.0, 0.0), y=y, mode="markers",
                             marker=dict(color=pay_color, size=6, symbol="square"),
                             name="Pay flag", hovertemplate="Depth: %{y:.1f} m<br>Pay: %{text}<extra></extra>",
                             text=np.where(pay_mask, "Yes", "No")), row=1, col=4)
    fig.add_trace(go.Scatter(x=np.ones_like(y)*0.5, y=y, mode="markers",
                             marker=dict(color=colors, size=6, symbol="square"),
                             name="Facies", hovertemplate="Depth: %{y:.1f} m<br>Facies: %{text}<extra></extra>",
                             text=view["FACIES"]), row=1, col=4)
    fig.update_xaxes(title="Facies/Pay", showticklabels=False, range=[-0.5, 1.5], row=1, col=4)

    # Global axes
    for c in [1,2,3,4]:
        fig.update_yaxes(autorange="reversed", title="Depth [m]" if c==1 else None, row=1, col=c)

    # Draw tops
    for t in tops:
        if zmin <= t <= zmax:
            for c in [1,2,3,4]:
                fig.add_hline(y=t, line=dict(color="rgba(255,255,255,0.35)", width=1, dash="dot"),
                              row=1, col=c)

    fig.update_layout(template="plotly_dark", height=900, legend_orientation="h", margin=dict(l=10, r=10, t=40, b=10))
    return fig

def crossplots_figure(df, archie: ArchieParams):
    # RHOB-NPHI crossplot and Pickett-style RT vs Phi
    fig = make_subplots(rows=1, cols=2, subplot_titles=("RHOB vs NPHI", "RT vs Phi_eff (Pickett-style)"),
                        specs=[[{"type": "xy"}, {"type": "xy"}]], horizontal_spacing=0.12)

    # Left: RHOB vs NPHI colored by facies
    color_map = {"Sandstone": "#F4D03F", "Limestone": "#85C1E9", "Shale": "#7DCEA0", "Unknown": "#95A5A6"}
    for fac in df["FACIES"].unique():
        sub = df[df["FACIES"] == fac]
        fig.add_trace(go.Scatter(
            x=sub["RHOB"], y=sub["NPHI"], mode="markers", name=str(fac),
            marker=dict(color=color_map.get(fac, "#95A5A6"), size=4, opacity=0.7),
            hovertemplate="RHOB: %{x:.2f} g/cc<br>NPHI: %{y:.2f}<br>Depth: %{text:.1f} m<extra></extra>",
            text=sub["DEPTH"]
        ), row=1, col=1)
    fig.update_xaxes(title="RHOB [g/cc]", range=[1.95, 2.95], row=1, col=1)
    fig.update_yaxes(title="NPHI [v/v]", range=[0.6, -0.05], row=1, col=1)  # reverse to mimic log style

    # Right: Pickett-style, RT vs Phi_eff
    x = np.clip(df["PHI_E"].values, 0.01, 0.45)
    y = np.clip(df["RT"].values, 0.2, 1000)
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers", name="Data",
        marker=dict(color="#f39c12", size=4, opacity=0.65),
        hovertemplate="Phi_e: %{x:.2f}<br>RT: %{y:.2f} ohm·m<extra></extra>"
    ), row=1, col=2)
    # Iso-Sw curves
    phi_grid = np.linspace(0.04, 0.35, 100)
    for sw_level, col in zip([0.2, 0.4, 0.6, 0.8], ["#2ecc71", "#27ae60", "#16a085", "#1abc9c"]):
        rt_curve = (archie.a * archie.Rw) / (np.power(phi_grid, archie.m) * np.power(sw_level, archie.n))
        fig.add_trace(go.Scatter(
            x=phi_grid, y=rt_curve, mode="lines", line=dict(color=col, width=1.5),
            name=f"Sw={sw_level:.1f}", hoverinfo="skip"
        ), row=1, col=2)
    fig.update_xaxes(title="Phi_eff [v/v]", range=[0, 0.4], row=1, col=2)
    fig.update_yaxes(title="RT [ohm·m]", type="log", range=[-0.7, 3], row=1, col=2)

    fig.update_layout(template="plotly_dark", height=500, legend_orientation="h", margin=dict(l=10, r=10, t=40, b=10))
    return fig

# ------------------------------
# -------- Dash layout ---------
# ------------------------------

wells_data = generate_synthetic_wells()

app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, dbc.icons.BOOTSTRAP])
app.title = "AI Well Log Interpreter"

def kpi_card(title, value, subtitle=None, color="primary", icon="bi bi-speedometer2"):
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.I(className=icon, style={"fontSize": "1.4rem", "marginRight": "6px"}),
                html.Span(title, className="text-muted")
            ]),
            html.H4(value, className="card-title mt-2"),
            html.Small(subtitle or "", className="text-muted")
        ]),
        className=f"border-0 bg-{color} bg-opacity-25",
        style={"height": "100%"}
    )

controls = dbc.Accordion([
    dbc.AccordionItem([
        dbc.Row([
            dbc.Col(dbc.Select(id="well-select", options=[{"label": k, "value": k} for k in wells_data.keys()],
                               value=list(wells_data.keys())[0], className="mb-2"), width=12),
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("Depth min [m]"),
                dbc.Input(id="depth-min", type="number"),
            ]), width=6),
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("Depth max [m]"),
                dbc.Input(id="depth-max", type="number"),
            ]), width=6),
        ], className="g-2"),
        html.Div("Tip: You can set precise depth limits here."),
    ], title="Well & Interval"),
    dbc.AccordionItem([
        dbc.Row([
            dbc.Col(dbc.RadioItems(
                id="vsh-method",
                options=[{"label": "Larionov (Tertiary)", "value": "Larionov (Tertiary)"},
                         {"label": "Linear", "value": "Linear"}],
                value="Larionov (Tertiary)", inline=True
            ), width=12),
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("GR clean [API]"),
                dbc.Input(id="gr-min", type="number", value=None, placeholder="auto (p5)"),
            ]), width=6),
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("GR shale [API]"),
                dbc.Input(id="gr-max", type="number", value=None, placeholder="auto (p95)"),
            ]), width=6),
            html.Hr(className="mt-3"),
            dbc.Col(dbc.Select(
                id="matrix-type",
                options=[
                    {"label": "Sandstone (2.65 g/cc)", "value": "Sandstone"},
                    {"label": "Limestone (2.71 g/cc)", "value": "Limestone"},
                    {"label": "Dolomite (2.87 g/cc)", "value": "Dolomite"},
                ],
                value="Limestone"
            ), width=6),
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("Shale porosity"),
                dbc.Input(id="phi-shale", type="number", step=0.01, value=0.30),
            ]), width=6),
            html.Hr(className="mt-3"),
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("Archie a"),
                dbc.Input(id="archie-a", type="number", step=0.1, value=1.0),
            ]), width=3),
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("m"),
                dbc.Input(id="archie-m", type="number", step=0.1, value=2.0),
            ]), width=3),
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("n"),
                dbc.Input(id="archie-n", type="number", step=0.1, value=2.0),
            ]), width=3),
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("Rw [ohm·m]"),
                dbc.Input(id="archie-rw", type="number", step=0.01, value=0.08),
            ]), width=3),
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("Smoothing window"),
                dbc.Input(id="smooth-win", type="number", min=1, max=21, step=2, value=5),
            ]), width=4, className="mt-2"),
        ], className="g-2"),
    ], title="Petrophysical Parameters"),
    dbc.AccordionItem([
        dbc.Row([
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("Vsh max"),
                dbc.Input(id="cut-vsh", type="number", step=0.01, value=0.35),
            ]), width=3),
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("Phi min"),
                dbc.Input(id="cut-phi", type="number", step=0.01, value=0.08),
            ]), width=3),
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("Sw max"),
                dbc.Input(id="cut-sw", type="number", step=0.01, value=0.6),
            ]), width=3),
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("RT min"),
                dbc.Input(id="cut-rt", type="number", step=0.1, value=2.0),
            ]), width=3),
        ], className="g-2"),
        html.Small("These cutoffs define the pay flag and Net Pay/NTG.", className="text-muted"),
    ], title="Pay Cutoffs"),
    dbc.AccordionItem([
        dbc.Row([
            dbc.Col(dbc.Checklist(
                id="ml-on", options=[{"label": "Enable ML facies", "value": "on"}], value=["on"], switch=True
            ), width=6),
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("Clusters"),
                dbc.Input(id="ml-k", type="number", min=2, max=6, value=3),
            ]), width=6),
            dbc.Col(dbc.Checklist(
                id="qc-smooth", options=[{"label": "Apply smoothing", "value": "on"}], value=["on"], switch=True
            ), width=6, className="mt-2"),
        ], className="g-2"),
    ], title="ML & QC"),
    dbc.AccordionItem([
        dbc.Button("Download interpreted CSV", id="btn-download", color="info", className="w-100"),
        dcc.Download(id="download-data"),
        html.Small("Exports the visible well interval with computed Vsh, Phi, Sw, k, pay flags, and facies.", className="text-muted")
    ], title="Export"),
], start_collapsed=False, always_open=True, className="mb-3")

app.layout = dbc.Container(fluid=True, children=[
    dbc.Navbar(dbc.Container([
        dbc.NavbarBrand("AI Well Log Interpreter (MVP)", className="ms-2"),
        dbc.Badge("Well Log Interpretation", color="success", className="ms-2")
    ]), dark=True, color="dark", className="mb-3"),
    dbc.Row([
        dbc.Col([
            controls
        ], width=3),
        dbc.Col([
            dbc.Row([
                dbc.Col(id="kpi-gross", width=2),
                dbc.Col(id="kpi-net", width=2),
                dbc.Col(id="kpi-ntg", width=2),
                dbc.Col(id="kpi-phi", width=3),
                dbc.Col(id="kpi-hcpv", width=3),
            ], className="g-2 mb-2"),
            dcc.Tabs(id="tabs", value="logs", children=[
                dcc.Tab(label="Logs", value="logs", children=[
                    dcc.Loading(dcc.Graph(id="log-fig"), type="dot")
                ]),
                dcc.Tab(label="Crossplots", value="xplots", children=[
                    dcc.Loading(dcc.Graph(id="xplot-fig"), type="dot", className="mt-2")
                ]),
                dcc.Tab(label="ML & Tops", value="ml", children=[
                    html.Div(id="tops-list", className="mt-3"),
                    html.Div(id="facies-summary", className="mt-2"),
                ]),
                dcc.Tab(label="Data", value="data", children=[
                    html.Div(id="table-div", className="mt-3")
                ]),
            ]),
        ], width=9)
    ]),
    dcc.Store(id="store-well-data"),
    dcc.Store(id="store-interp-data"),
])

# ------------------------------
# --------- Callbacks ----------
# ------------------------------

@app.callback(
    Output("store-well-data", "data"),
    Output("depth-min", "value"),
    Output("depth-max", "value"),
    Output("archie-rw", "value"),
    Input("well-select", "value"),
)
def on_well_change(well):
    df = wells_data[well].copy()
    zmin, zmax = float(df["DEPTH"].min()), float(df["DEPTH"].max())
    # default Rw per well (from generator)
    if "Fluvial" in well:
        rw = 0.08
    elif "Carbonate" in well:
        rw = 0.05
    else:
        rw = 0.10
    return df.to_dict("records"), zmin, zmax, rw

@app.callback(
    Output("store-interp-data", "data"),
    Output("log-fig", "figure"),
    Output("xplot-fig", "figure"),
    Output("kpi-gross", "children"),
    Output("kpi-net", "children"),
    Output("kpi-ntg", "children"),
    Output("kpi-phi", "children"),
    Output("kpi-hcpv", "children"),
    Output("tops-list", "children"),
    Output("facies-summary", "children"),
    Input("store-well-data", "data"),
    Input("depth-min", "value"),
    Input("depth-max", "value"),
    Input("vsh-method", "value"),
    Input("gr-min", "value"),
    Input("gr-max", "value"),
    Input("matrix-type", "value"),
    Input("phi-shale", "value"),
    Input("archie-a", "value"),
    Input("archie-m", "value"),
    Input("archie-n", "value"),
    Input("archie-rw", "value"),
    Input("smooth-win", "value"),
    Input("cut-vsh", "value"),
    Input("cut-phi", "value"),
    Input("cut-sw", "value"),
    Input("cut-rt", "value"),
    Input("ml-on", "value"),
    Input("ml-k", "value"),
    Input("qc-smooth", "value"),
)
def run_interpretation(records, zmin, zmax, vsh_method, gr_min_in, gr_max_in,
                       matrix_type, phi_shale, a, m, n, rw, smooth_win,
                       cut_vsh, cut_phi, cut_sw, cut_rt, ml_on_list, ml_k, qc_smooth_list):
    if not records:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    df = pd.DataFrame.from_records(records)
    # Auto GR bounds if not provided (percentiles within selected interval)
    df_range = df[(df["DEPTH"] >= zmin) & (df["DEPTH"] <= zmax)]
    if df_range.empty:
        df_range = df
    gr_min = float(gr_min_in) if gr_min_in is not None else float(np.nanpercentile(df_range["GR"], 5))
    gr_max = float(gr_max_in) if gr_max_in is not None else float(np.nanpercentile(df_range["GR"], 95))
    gr_min, gr_max = min(gr_min, gr_max-1e-3), max(gr_max, gr_min+1e-3)

    # Matrix params
    rho_ma_map = {"Sandstone": 2.65, "Limestone": 2.71, "Dolomite": 2.87}
    matrix = MatrixParams(name=matrix_type, rho_ma=rho_ma_map.get(matrix_type, 2.71), rho_f=1.0,
                          phi_shale=float(phi_shale or 0.3))
    archie = ArchieParams(a=float(a or 1.0), m=float(m or 2.0), n=float(n or 2.0), Rw=float(rw or 0.08))
    smooth = int(smooth_win or 1)
    if (qc_smooth_list is None) or ("on" not in qc_smooth_list):
        smooth = 1
    ml_on = (ml_on_list is not None) and ("on" in ml_on_list)
    ml_k = int(ml_k or 3)

    # Compute interpretation
    interp, tops = compute_interpretation(
        df, gr_min, gr_max, vsh_method, archie, matrix, smooth_win=smooth, ml_on=ml_on, ml_k=ml_k
    )

    # Pay flags and metrics in range
    view = interp[(interp["DEPTH"] >= zmin) & (interp["DEPTH"] <= zmax)].copy()
    pay_mask = compute_pay_flags(view, vsh_cut=float(cut_vsh), phi_cut=float(cut_phi),
                                 sw_cut=float(cut_sw), rt_cut=float(cut_rt))
    view["PAY"] = pay_mask
    gross, net, ntg, phi_pay, sw_pay, hcpvi = pay_metrics(view, pay_mask)

    # Merge pay back for plotting
    interp["PAY"] = False
    interp.loc[view.index, "PAY"] = pay_mask.values

    # Figures
    fig_logs = main_logs_figure(interp, (zmin, zmax), float(cut_vsh), float(cut_rt), tops)
    fig_xplots = crossplots_figure(view, archie)

    # KPIs
    kpi_g = kpi_card("Gross [m]", f"{gross:,.1f}", "Interval thickness", color="secondary", icon="bi bi-arrows-collapse")
    kpi_n = kpi_card("Net Pay [m]", f"{net:,.1f}", f"Cutoffs: Vsh≤{cut_vsh}, Phi≥{cut_phi}, Sw≤{cut_sw}, RT≥{cut_rt}", color="success", icon="bi bi-bar-chart-fill")
    kpi_ntg = kpi_card("NTG [-]", f"{ntg:.2f}", "Net/Gross", color="info", icon="bi bi-pie-chart-fill")
    phi_txt = "—" if np.isnan(phi_pay) else f"{phi_pay:.02f}"
    kpi_phi = kpi_card("Avg Phi in Pay", phi_txt, color="primary", icon="bi bi-droplet-fill")
    kpi_h = kpi_card("HCPVI [index]", f"{hcpvi:,.1f}", "Sum(phi*(1-Sw)*dz)", color="warning", icon="bi bi-fire")

    # Tops list and facies summary
    tops_div = html.Div([
        html.H5("Detected Formation Tops"),
        html.Ul([html.Li(f"{t:.1f} m") for t in tops]) if len(tops) else html.Small("No distinct tops detected.")
    ])
    facies_counts = view["FACIES"].value_counts(normalize=True).sort_values(ascending=False)
    facies_div = html.Div([
        html.H5("Facies distribution (interval)"),
        html.Ul([html.Li(f"{k}: {v*100:.1f}%") for k, v in facies_counts.items()]) if not facies_counts.empty else html.Small("No facies.")
    ])

    return (
        interp.to_dict("records"),
        fig_logs, fig_xplots,
        kpi_g, kpi_n, kpi_ntg, kpi_phi, kpi_h,
        tops_div, facies_div
    )

@app.callback(
    Output("download-data", "data"),
    Input("btn-download", "n_clicks"),
    State("store-interp-data", "data"),
    prevent_initial_call=True
)
def on_download(n, records):
    if not records:
        return no_update
    df = pd.DataFrame.from_records(records)
    cols = ["WELL","DEPTH","GR","RT","RHOB","NPHI","DT","PEF","CALI",
            "VSH","PHI_T","PHI_E","SW","K_MDK","FACIES","PAY"]
    cols = [c for c in cols if c in df.columns]
    csv = df[cols].to_csv(index=False)
    return dict(content=csv, filename="interpreted_logs.csv")

if __name__ == "__main__":
    app.run(debug=True)