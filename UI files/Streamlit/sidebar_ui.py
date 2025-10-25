import streamlit as st

class SidebarUi:
    def __init__(self):
        pass

    def upload_ui(self):
        self.choice_null = st.sidebar.radio('Keep or remove null values', ('Keep', 'Remove'))
        self.choice_duplicate = st.sidebar.radio('Keep or remove duplicate values', ('Keep', 'Remove'))