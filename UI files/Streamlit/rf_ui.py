import streamlit as st
import pandas as pd
from Backend.rfBenchmark import RFB
import numpy as np
from PIL import Image

class RFUI:
    def __init__(self):
        pass

    def uploadUI(self, RF_data):
        rfb = RFB()
        rfb2 = RFB()

        if len(RF_data) != 0:
            df_rf_1 = rfb.load_data(RF_data[0])
            df_rf_2 = rfb2.load_data(RF_data[1])

            df_rf_1, rfb_df_grouped = rfb.dataformat()
            df_rf_2, rfb_df_grouped2 = rfb2.dataformat()

            plot_p50_tab, outlier_detection_tab, plot_margin_tab, bubble_map_tab = st.tabs(['P50 Plot', 'Outliers Detection', 'Margin Plot', 'Bubble map'])

            with plot_p50_tab:
                fig = rfb.plot_p50()
                st.plotly_chart(fig, use_container_width=True)

            with outlier_detection_tab:
                fig = rfb.detect_outliers_iso_forest()
                st.plotly_chart(fig, use_container_width=True)

            with plot_margin_tab:
                fig = rfb.plot_margins()
                st.plotly_chart(fig, use_container_width=True)

            with bubble_map_tab:
                source = Image.open(r"C:\Users\aliyu\OneDrive\İş masası\Risehill\Desktop-PDA\pda-gui\images\MicrosoftTeams-image.png")
                coef, intercept = rfb.get_coef_and_intercept()


                # creating comparison dataset for 2 wells
                df_compare = pd.concat([rfb_df_grouped[['fault block', 'Kh']], rfb_df_grouped2[['fault block', 'Kh']]],
                                       axis=0)
                df_compare = df_compare.groupby('fault block', axis=0, as_index=False).sum()
                df_compare.rename(columns={'Kh': 'exp Kh'}, inplace=True)
                df_compare = df_compare.merge(rfb_df_grouped[['fault block', 'Kh', 'OOIP, MMSTB']], on=['fault block'])

                # calculating RF & exponential RF
                df_compare['RF'] = np.power(10, coef * np.log10(df_compare['Kh']) + intercept)
                df_compare['exp RF'] = np.power(10, coef * np.log10(df_compare['exp Kh']) + intercept)
                # calculating RF_10 & exponential RF_!0
                df_compare['RF_10'] = df_compare['RF'] + (
                            rfb.lr.intercept_ - df_compare['RF'].std() * rfb.margin_multiplier)
                df_compare['exp RF_10'] = df_compare['exp RF'] + (
                            rfb.lr.intercept_ - df_compare['exp RF'].std() * rfb.margin_multiplier)
                # calculating RF_90 & exponential RF_90
                df_compare['RF_90'] = df_compare['RF'] - (
                            rfb.lr.intercept_ - df_compare['RF'].std() * rfb.margin_multiplier)
                df_compare['exp RF_90'] = df_compare['exp RF'] - (
                            rfb.lr.intercept_ - df_compare['exp RF'].std() * rfb.margin_multiplier)
                # calculating incrementals for RF, RF_10 & RF_90
                df_compare['incremental RF'] = df_compare['exp RF'] - df_compare['RF']
                df_compare['incremental RF_10'] = df_compare['exp RF_10'] - df_compare['RF_10']
                df_compare['incremental RF_90'] = df_compare['exp RF_90'] - df_compare['RF_90']
                # calculating EUR
                df_compare['incremental EUR, MMSTB'] = df_compare['incremental RF'] / 100 * df_compare['OOIP, MMSTB']
                df_compare['incremental EUR_10, MMSTB'] = df_compare['incremental RF_10'] / 100 * df_compare['OOIP, MMSTB']
                df_compare['incremental EUR_90, MMSTB'] = df_compare['incremental RF_90'] / 100 * df_compare['OOIP, MMSTB']
                # validation of incremental RF
                if (df_compare['incremental RF'] < 100).all() == False:
                    print('Warning!!! RF is bigger than 100% which is impossible!!!')
                # calculating total EUR increase
                eur_increase_total = df_compare['incremental EUR, MMSTB'].sum()
                eur_increase_total_10 = df_compare['incremental EUR_10, MMSTB'].sum()
                eur_increase_total_90 = df_compare['incremental EUR_90, MMSTB'].sum()

                # calculating radius of investigation
                rfb_df_grouped['Boi'] = 1.23
                rfb_df_grouped['radius, ft'] = np.sqrt(
                    ((rfb_df_grouped['EUR, MMSTB'] * 1_000_000) / (rfb_df_grouped['RF'] / 100) * rfb_df_grouped['Boi']) / (
                                (rfb_df_grouped['poro %'] / 100) * (rfb_df_grouped['perf NTG'] / 100) * (
                                    1 - (rfb_df_grouped['Sw %'] / 100)) * rfb_df_grouped['block thickness, ft'] * np.pi))
                rfb_df_grouped['radius, km'] = rfb_df_grouped['radius, ft'] * 0.0003048

                st.write(f'Estimated Ultimate Recovery(Total): {eur_increase_total}')
                st.write(f'Estimated Ultimate Recovery(10): {eur_increase_total_10}')
                st.write(f'Estimated Ultimate Recovery(90): {eur_increase_total_90}')

                rfb_df_grouped, fig = rfb.scatterplot()

                fig = rfb.mapplot(source)
                st.plotly_chart(fig, use_container_width=True)
