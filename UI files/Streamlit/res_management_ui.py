import streamlit as st
from Streamlit.infill_well_ui import PePlotsUi
from Streamlit.rf_ui import RFUI

class Res_Management():
    def __int__(self):
        pass

    def upload_ui(self, PVT_df, RF_data):
        self.infill_wells_tab, self.forecasting_tab = st.tabs(['Infill Wells', 'Forecasting'])

        with self.infill_wells_tab:
            self.pplt = PePlotsUi()
            self.pplt.upload_ui(PVT_df)

        with self.forecasting_tab:
            self.rfui = RFUI()
            self.rfui.uploadUI(RF_data)


