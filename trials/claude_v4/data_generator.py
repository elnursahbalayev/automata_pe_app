# data_generator.py - Realistic Oil & Gas Well Log Data Generator
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_well_logs(well_id, num_points=600):
    """
    Generate realistic well log data for oil & gas applications
    """
    np.random.seed(hash(well_id) % 2 ** 32)

    # Depth array
    depth_start = 5000 + np.random.randint(0, 1000)
    depth_end = depth_start + 3000
    depth = np.linspace(depth_start, depth_end, num_points)

    # Define zone boundaries (for creating realistic variations)
    zones = [
        {'start': 0, 'end': 0.2, 'type': 'shale_cap'},
        {'start': 0.2, 'end': 0.35, 'type': 'sand_reservoir'},
        {'start': 0.35, 'end': 0.45, 'type': 'shale_barrier'},
        {'start': 0.45, 'end': 0.65, 'type': 'sand_reservoir'},
        {'start': 0.65, 'end': 0.75, 'type': 'limestone'},
        {'start': 0.75, 'end': 0.85, 'type': 'sand_reservoir'},
        {'start': 0.85, 'end': 1.0, 'type': 'shale_base'}
    ]

    # Initialize logs
    gr = np.zeros(num_points)
    rt = np.zeros(num_points)
    rhob = np.zeros(num_points)
    nphi = np.zeros(num_points)
    dt = np.zeros(num_points)

    for zone in zones:
        start_idx = int(zone['start'] * num_points)
        end_idx = int(zone['end'] * num_points)

        if zone['type'] == 'shale_cap' or zone['type'] == 'shale_barrier' or zone['type'] == 'shale_base':
            # Shale properties
            gr[start_idx:end_idx] = np.random.normal(90, 15, end_idx - start_idx)
            rt[start_idx:end_idx] = np.random.lognormal(1.5, 0.3, end_idx - start_idx)
            rhob[start_idx:end_idx] = np.random.normal(2.45, 0.05, end_idx - start_idx)
            nphi[start_idx:end_idx] = np.random.normal(0.35, 0.05, end_idx - start_idx)
            dt[start_idx:end_idx] = np.random.normal(85, 5, end_idx - start_idx)

        elif zone['type'] == 'sand_reservoir':
            # Sandstone reservoir properties
            base_porosity = np.random.uniform(0.15, 0.25)

            # Add hydrocarbon effect randomly
            if np.random.random() > 0.3:  # 70% chance of hydrocarbons
                # Hydrocarbon-bearing sandstone
                gr[start_idx:end_idx] = np.random.normal(35, 10, end_idx - start_idx)
                rt[start_idx:end_idx] = np.random.lognormal(3.5, 0.5, end_idx - start_idx)  # Higher resistivity
                rhob[start_idx:end_idx] = np.random.normal(2.3 - base_porosity * 0.5, 0.03, end_idx - start_idx)
                nphi[start_idx:end_idx] = np.random.normal(base_porosity - 0.05, 0.02, end_idx - start_idx)
                dt[start_idx:end_idx] = np.random.normal(75, 3, end_idx - start_idx)
            else:
                # Water-bearing sandstone
                gr[start_idx:end_idx] = np.random.normal(40, 10, end_idx - start_idx)
                rt[start_idx:end_idx] = np.random.lognormal(1.8, 0.3, end_idx - start_idx)  # Lower resistivity
                rhob[start_idx:end_idx] = np.random.normal(2.35 - base_porosity * 0.3, 0.03, end_idx - start_idx)
                nphi[start_idx:end_idx] = np.random.normal(base_porosity, 0.02, end_idx - start_idx)
                dt[start_idx:end_idx] = np.random.normal(80, 3, end_idx - start_idx)

        elif zone['type'] == 'limestone':
            # Limestone properties
            gr[start_idx:end_idx] = np.random.normal(25, 8, end_idx - start_idx)
            rt[start_idx:end_idx] = np.random.lognormal(2.5, 0.4, end_idx - start_idx)
            rhob[start_idx:end_idx] = np.random.normal(2.55, 0.04, end_idx - start_idx)
            nphi[start_idx:end_idx] = np.random.normal(0.1, 0.03, end_idx - start_idx)
            dt[start_idx:end_idx] = np.random.normal(65, 4, end_idx - start_idx)

    # Apply smoothing for more realistic appearance
    from scipy.ndimage import gaussian_filter1d
    gr = gaussian_filter1d(gr, sigma=2)
    rt = gaussian_filter1d(rt, sigma=2)
    rhob = gaussian_filter1d(rhob, sigma=2)
    nphi = gaussian_filter1d(nphi, sigma=2)
    dt = gaussian_filter1d(dt, sigma=2)

    # Ensure realistic bounds
    gr = np.clip(gr, 0, 150)
    rt = np.clip(rt, 0.2, 2000)
    rhob = np.clip(rhob, 1.95, 2.95)
    nphi = np.clip(nphi, -0.05, 0.45)
    dt = np.clip(dt, 40, 140)

    # Create DataFrame
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
    """
    Generate metadata for a well
    """
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