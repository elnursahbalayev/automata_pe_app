# Production Enhancement Dashboard

This application provides a dashboard for production enhancement analysis. It includes tools for data loading, quality control, analysis, and reporting.

## Running the Application

The application can be run in two different modes: Streamlit or Flet (Flutter).

### Streamlit Version

To run the Streamlit version of the application, navigate to the `UI files` directory and run:

```bash
streamlit run main.py
```

### Flet Version

There are two ways to run the Flet version of the application:

1. Using the main.py file with the --flet flag:

```bash
python main.py --flet
```

2. Using the dedicated Flet main file:

```bash
python main_flet.py
```

## Features

The application includes the following features:

- Data Loading: Upload and manage production data, PVT data, survey data, and more.
- Data QC/Analysis: Analyze the loaded data, including PDG data, production data, gas lift optimization, reservoir management, and well trajectory.
- Data Export and Reporting: Export data and generate reports (coming soon).

## Requirements

- Python 3.7+
- Streamlit (for Streamlit version)
- Flet (for Flet version)
- Pandas
- Plotly
- PIL (Python Imaging Library)
- Other dependencies as specified in the code

## File Structure

- `UI files/`: Root directory for the UI components
  - `Streamlit/`: Contains the Streamlit implementation of the UI components
  - `Flet/`: Contains the Flet implementation of the UI components
  - `Backend/`: Contains the backend processing code
  - `images/`: Contains images used in the application
  - `main.py`: Main entry point for the application (supports both Streamlit and Flet)
  - `main_flet.py`: Dedicated entry point for the Flet version

## Development

To add new features or modify existing ones, update the corresponding files in the `Streamlit/` or `Flet/` directories. The backend processing code is shared between both implementations and is located in the `Backend/` directory.