import plotly.graph_objects as go
import pandas as pd
import numpy as np    
from plotly.subplots import make_subplots
from Backend.dataPreprocessing import dataprocess as datap
    
    
class pdg_process():
    def __init__(self):
        pass

    @staticmethod
    def visualize_well(dataframe):
        """Takes in the dataframe, plots the distributions using Histogram Plotly interactive plots."""
        col = [x for x in dataframe.columns if x not in ['Date', 'Time','Date, Time']]
        # print(col)
        fig = make_subplots(rows=int(len(col)/2), cols=2)
        k = 0
        for i in range(int(len(col)/2)):
            for j in range(2):
                fig.add_trace(
                    go.Histogram(x=dataframe[col[k]], name=col[k]),
                    row=i+1, col=j+1
                )
                # fig.update_layout(xaxis_title_text='Value')
                k+=1
        fig.update_xaxes(showgrid=True)
        fig.update_layout(height=720, width=1200, title_text="PDG Pressure and Temperature distribution")
        # fig.show()
        return fig

    @staticmethod
    def plot_date_vs_pressure_difference(dataframe):
        """Takes in the dataframe, plots the date vs pressure difference"""

        num_gauges, type_gauge, tp_reading, ap_reading, tt_reading = datap.num_gaugedetect(dataframe)

        pressure_differences = pd.DataFrame()
        for i in range(len(num_gauges)):
            diff = dataframe[tp_reading[i]] - dataframe[ap_reading[i]]
            pressure_differences[f'diff_{i+1}'] = diff
        pressure_differences['Date'] = dataframe['Date']
        fig = make_subplots(rows=len(num_gauges), cols=1)
        k=1
        for i in range(len(num_gauges)):
            fig.add_trace(
            go.Scatter(x=pressure_differences['Date'], y=pressure_differences[f'diff_{k}'], name=f'difference {k}', showlegend=True)
            ,row=k, col=1)
            k+=1
        
        fig.update_xaxes(showgrid=True)
        fig.update_layout(height=720, width=720, title_text="Pressure difference over time")
        # fig.show()
        return fig
    
    @staticmethod
    def calculate_fluid_density(dataframe, depth_gauges):
        """This function  is used calculate the fluid density for n  number of gauges. Input: dataframe and  self for determine the number
        gauges from class instance."""
        # depth_gauges=[1000,2000]
        num_gauges, type_gauge, tp_reading, ap_reading, tt_reading = datap.num_gaugedetect(dataframe)
        print("There are {} type of reading detected with {} number of gauge! ".format(len(type_gauge),len(num_gauges)))

        print('1')
        # for i in range(len(num_gauges)):
        #     dp = float(input('Please enter the gauge depth for gauge {} '.format(i+1)))
        #     print('123')
        #     depth_gauges.append(dp)
        print('2')
        for i in range(1,len(num_gauges)):
            array_1 = dataframe['{}'.format(tp_reading[i-1])].to_numpy() 
            array_2 = dataframe['{}'.format(tp_reading[i])].to_numpy()
            result = np.array((array_2 - array_1)/(depth_gauges[i]-depth_gauges[i-1]))
            dataframe['The Fluid density between Gauge {} and Gauge {}'.format(i,i+1)] = result.tolist()
        print('3')
        return dataframe

    @staticmethod
    def plot_fluid_density(dataframe):
        """This function  is used to plot fluid density vs time using plotly library. Input: dataframe  and self for determine the  number 
        of gauges from class instance"""
        # init_notebook_mode(connected=True)
        # cf.go_offline()
        fig = go.Figure()
        num_gauges,type_gauge, tp_reading, ap_reading, tt_reading = datap.num_gaugedetect(dataframe)
        for i in range(1, len(num_gauges)):
            fig.add_trace(go.Scatter(x=dataframe['Date'], 
                        y=dataframe['The Fluid density between Gauge {} and Gauge {}'.format(i,i+1)], 
                        mode='lines', 
                        name='The Fluid density between Gauge {} and Gauge {}'.format(i,i+1)
                        ))
        
        fig.update_layout(title='Fluid density vs Time',
                   xaxis_title='Date', yaxis_title='Fluid Density')

        # fig.show()
        return fig

