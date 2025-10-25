from Backend.gasLiftOptimization import GasLiftOptimization
import streamlit as st
class GLOUi:
    def __init__(self):
        self.glo = GasLiftOptimization()

    def upload_ui(self):

        self.glo.check_dir_and_retrieve_values(1.5, 0.05)
        self.glo.simulation()
        self.glo.wells()
        self.glo.generate_well_list()
        self.glo.testing()
        st.plotly_chart(self.glo.visualize(), use_container_width=True)


