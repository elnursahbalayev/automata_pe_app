import sys
import os

# Check if the user wants to run the Flet version
if len(sys.argv) > 1 and sys.argv[1] == "--flet":
    print("Running Flet version...")
    # Add the current directory to the path so we can import modules
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    # Import the Flet main module
    from Flet.main import main
    import flet as ft

    if __name__ == "__main__":
        # Run the Flet app
        ft.app(target=main)
else:
    # Run the Streamlit version
    import streamlit as st
    from PIL import Image
    from Streamlit.wellpath_ui import WellpathUi
    from Streamlit.pda_ui import PdaUi
    from Streamlit.data_management_ui import DataManagementUi

    im = Image.open("images/logo.png")
    st.set_page_config(
        page_title="Production Enhancement Dashboard",
        page_icon=im,
        layout="wide",
    )

    st.image('images/logo.png', width=200)
    st.text('')

    tab_data_man, tab_pda, tab_export_report = st.tabs(["Data Loading", "Data QC/Analysis", "Data Export and Reporting"])

    with tab_data_man:
        dmu = DataManagementUi()
        dmu.upload_ui()

    with tab_pda:
        pdaui=PdaUi()
        pdaui.monthly_prod_data, pdaui.pvt_data, pdaui.dirname = dmu.return_data()
        rf_data = dmu.return_rf_data()
        buildup_pressure_df, buildup_rate_df = dmu.return_buildup_data()
        survey_file = dmu.return_survey_data()
        if pdaui.dirname is not None and rf_data is not None and buildup_pressure_df is not None and buildup_rate_df is not None and survey_file is not None:
            pdaui.upload_ui(rf_data, buildup_pressure_df, buildup_rate_df, survey_file)
