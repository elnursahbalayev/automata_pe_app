# log_processor.py - Well Log Processing and Interpretation Functions
import numpy as np
import pandas as pd


def calculate_porosity(df):
    """
    Calculate porosity from density log using standard equation
    """
    # Matrix density for different lithologies
    rho_matrix = 2.65  # g/cc for quartz sandstone
    rho_fluid = 1.0  # g/cc for water

    # Density porosity calculation
    porosity_density = (rho_matrix - df['RHOB']) / (rho_matrix - rho_fluid)

    # Neutron-Density average (corrected for gas effect)
    porosity_neutron = df['NPHI']

    # Average porosity with gas correction
    porosity = np.where(
        porosity_neutron < porosity_density - 0.05,  # Gas flag
        np.sqrt((porosity_density ** 2 + porosity_neutron ** 2) / 2),  # Gas correction
        (porosity_density + porosity_neutron) / 2  # Normal average
    )

    # Apply realistic bounds
    porosity = np.clip(porosity, 0, 0.35)

    return porosity


def calculate_water_saturation(df):
    """
    Calculate water saturation using Archie's equation
    """
    # Archie's parameters
    a = 1.0  # Tortuosity factor
    m = 2.0  # Cementation exponent
    n = 2.0  # Saturation exponent
    Rw = 0.05  # Formation water resistivity at formation temperature

    # Get porosity
    if 'POROSITY' not in df.columns:
        df['POROSITY'] = calculate_porosity(df)

    # Archie's equation: Sw = [(a * Rw) / (Rt * φ^m)]^(1/n)
    sw = np.power(
        (a * Rw) / (df['RT'] * np.power(df['POROSITY'] + 0.001, m)),
        1 / n
    )

    # Apply bounds
    sw = np.clip(sw, 0, 1)

    return sw


def identify_hydrocarbon_zones(df):
    """
    Identify potential hydrocarbon zones based on log responses
    """
    # Ensure required calculations are done
    if 'POROSITY' not in df.columns:
        df['POROSITY'] = calculate_porosity(df)
    if 'SW' not in df.columns:
        df['SW'] = calculate_water_saturation(df)

    # HC zone criteria:
    # 1. Low gamma ray (clean formation)
    # 2. High resistivity
    # 3. Good porosity
    # 4. Low water saturation
    # 5. Crossover effect (NPHI < RHOB in gas zones)

    hc_flag = (
            (df['GR'] < 60) &  # Clean formation
            (df['RT'] > 20) &  # High resistivity
            (df['POROSITY'] > 0.08) &  # Minimum porosity cutoff
            (df['SW'] < 0.5) &  # Low water saturation
            ((1 - df['SW']) * df['POROSITY'] > 0.04)  # Minimum hydrocarbon volume
    ).astype(int)

    return hc_flag


def calculate_net_pay(df):
    """
    Calculate net pay thickness
    """
    if 'HC_FLAG' not in df.columns:
        df['HC_FLAG'] = identify_hydrocarbon_zones(df)

    # Calculate thickness per sample
    if len(df) > 1:
        sample_thickness = (df['DEPTH'].max() - df['DEPTH'].min()) / len(df)
    else:
        sample_thickness = 0

    # Net pay is sum of all hydrocarbon-bearing intervals
    net_pay = df['HC_FLAG'].sum() * sample_thickness

    return net_pay


def interpret_lithology(df):
    """
    Interpret lithology from log responses using simple cutoffs
    """
    lithology = pd.Series(['Unknown'] * len(df), index=df.index)

    # Shale: High GR, High NPHI
    shale_mask = (df['GR'] > 75)
    lithology[shale_mask] = 'Shale'

    # Sandstone: Low GR, moderate porosity
    sand_mask = (df['GR'] < 50) & (df['RHOB'] < 2.4)
    lithology[sand_mask] = 'Sandstone'

    # Limestone: Very low GR, high density, low NPHI
    limestone_mask = (df['GR'] < 30) & (df['RHOB'] > 2.5) & (df['NPHI'] < 0.15)
    lithology[limestone_mask] = 'Limestone'

    # Dolomite: Similar to limestone but slightly different density
    dolomite_mask = (df['GR'] < 35) & (df['RHOB'] > 2.6) & (df['RHOB'] < 2.8)
    lithology[dolomite_mask] = 'Dolomite'

    return lithology


def calculate_hydrocarbon_volume(df):
    """
    Calculate hydrocarbon pore volume
    """
    if 'POROSITY' not in df.columns:
        df['POROSITY'] = calculate_porosity(df)
    if 'SW' not in df.columns:
        df['SW'] = calculate_water_saturation(df)

    # Hydrocarbon saturation
    shc = 1 - df['SW']

    # Hydrocarbon pore volume
    hcpv = df['POROSITY'] * shc

    return hcpv