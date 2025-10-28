# AUTOMATA ONG Application

This is a modular Flet application for AUTOMATA ONG that provides tools for oil and gas production data analysis.

## Application Structure

The application is organized in a modular structure:

- `main_app.py`: The entry point of the application that sets up the page and imports the modular components.
- `modules/`: Directory containing the modular components of the application:
  - `file_handling.py`: Handles file selection and CSV data processing.
  - `ui_components.py`: Manages the UI tabs and their content.
  - `__init__.py`: Makes the modules directory a Python package.
  - `README.md`: Documentation for the modules.
- `assets/`: Directory containing application assets like images.

## Running the Application

To run the application, execute the following command from the project root:

```bash
python app/src/main_app.py
```

## Features

The application provides the following features:

- Upload and analyze monthly production data
- Upload and analyze PVT data
- Well Log Interpretation (placeholder)
- Drilling Risk Prediction and Prevention (placeholder)

## Modular Design Benefits

The modular design of this application provides several benefits:

1. **Maintainability**: Each module has a single responsibility, making the code easier to maintain.
2. **Testability**: Modules can be tested independently, simplifying the testing process.
3. **Extensibility**: New features can be added by creating new modules without modifying existing code.
4. **Readability**: The code is organized in a logical structure, making it easier to understand.
5. **Reusability**: Modules can be reused in other parts of the application or in other projects.