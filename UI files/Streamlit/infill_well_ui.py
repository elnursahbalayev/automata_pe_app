from Backend.pePlots import Plot as pplt
import pandas as pd
import streamlit as st


class PePlotsUi:
    def __init__(self):
        self.pplt = pplt()

    def upload_ui(self, PVT_df):
        chan_tab, hall_tab, vrr_tab = st.tabs(['Chan Plot', 'Hall Plot', 'VRR Plot'])
        
        with chan_tab:
            self.PVT_df, fig_chan = pplt.chanplot(PVT_df)
            st.plotly_chart(fig_chan, use_container_width=True)

        with hall_tab:
            p_avg_res = st.number_input('Please enter average pressure of the reservoir: ')
            WI_volume = [7000]
            # WI_volume = [8000, 7000, 6000, 4000] # This has to be changed later as file input

            self.PVT_df, fig_hall = pplt.hallplot(self.PVT_df, WI_volume, p_avg_res)
            st.plotly_chart(fig_chan, use_container_width=True)

        with vrr_tab:
            self.PVT_df, fig_vrr = pplt.plot_VRR(self.PVT_df)
            st.plotly_chart(fig_vrr, use_container_width=True)