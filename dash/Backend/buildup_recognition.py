import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

class BuildUp:

    def __init__(self) -> None:
        pass

    def read_data(self,df_pressure, df_rate):
        self.df_pressure = df_pressure
        self.df_rate = df_rate

        self.df_pressure.drop(0, axis=0, inplace=True)
        self.df_rate.drop(0,axis=0, inplace=True)

    def format_data(self):
        # Change time column from object to datetime
        self.df_pressure['Test Design 2_Date Time'] =pd.to_datetime(self.df_pressure['Test Design 2_Date Time'])
        self.df_pressure['Test Design 2_Pressure'] = self.df_pressure['Test Design 2_Pressure'].astype(float)

        # Change time column from object to datetime
        self.df_rate['Oil rate_Date Time'] =pd.to_datetime(self.df_rate['Oil rate_Date Time'])
        self.df_rate['Oil rate_Liquid rate'] = self.df_rate['Oil rate_Liquid rate'].astype(float)

        # Drop duplicates
        self.df_rate = self.df_rate.drop_duplicates()

        # Create a new column 'y_shifted' that contains the value of the next row
        self.df_rate['y_shifted'] = self.df_rate['Oil rate_Liquid rate'].shift(-1)

        df_new_rates = self.df_rate[['Oil rate_Date Time', 'y_shifted']]
        df_new_rates =  df_new_rates.iloc[:-1]

        self.df_rate.drop('y_shifted', axis=1, inplace=True)

        df_new_rates = df_new_rates.rename(columns={'y_shifted':'Oil rate_Liquid rate'})

        self.df_rate = pd.concat([self.df_rate, df_new_rates])
        self.df_rate.reset_index(inplace=True)
        self.df_rate = self.df_rate.sort_values(by=['Oil rate_Date Time', 'index'])

    def plot_pressure_rate(self):
        
        fig = make_subplots(rows=2, cols=1, subplot_titles=['Time (hr) vs Pressure (psia)', 'Time (hr) vs Rate (STB/D)'])  # Repeating the graph above with plotly

        # Plotting time vs Pressure
        fig.add_trace(go.Scatter(x=self.df_pressure['Test Design 2_Date Time'], y=self.df_pressure['Test Design 2_Pressure']), row=1, col=1) 
        # Plotting time vs Rate
        fig.add_trace(go.Scatter(x=self.df_rate['Oil rate_Date Time'], y=self.df_rate['Oil rate_Liquid rate']), row=2, col=1) 

        fig.update_layout(height=600, width=800)

        fig.update_xaxes(title_text='Time') # adding x axis title
        fig.update_yaxes(title_text='Pressure', row=1, col=1) # adding y axis title
        fig.update_yaxes(title_text='Rate', row=2, col=1) # adding y axis title

        return fig

    def plot_pressure_der_rate(self):
        # Finding the first derivative of pressure
        self.df_pressure['pressure_1der'] = self.df_pressure['Test Design 2_Pressure'].diff()

        fig = make_subplots(rows=2, cols=1, subplot_titles=['Time (hr) vs First derivative of pressure (psi)', 'Time (hr) vs Rate (STB/D)'])  # Repeating the graph above with plotly

        # Plotting time vs Pressure
        fig.add_trace(go.Scatter(x=self.df_pressure['Test Design 2_Date Time'], y=self.df_pressure['pressure_1der']), row=1, col=1) 
        # Plotting time vs Rate
        fig.add_trace(go.Scatter(x=self.df_rate['Oil rate_Date Time'], y=self.df_rate['Oil rate_Liquid rate']), row=2, col=1) 

        fig.update_layout(height=600, width=800)

        fig.update_xaxes(title_text='Time') # adding x axis title
        fig.update_yaxes(title_text='First derivative of pressure', row=1, col=1) # adding y axis title
        fig.update_yaxes(title_text='Rate', row=2, col=1) # adding y axis title

        return fig

    def change_small_values_to_zero(self, df, column, multiplier=0.5):
        """Takes the dataframe column and removes small fluctuations in the data (filters noise) by standard deviation method"""
        mean = df[column].mean()
        std = df[column].std()
        threshold = mean + multiplier * std
        df.loc[np.abs(df[column]) < threshold, column] = 0
        return df

    def plot_pressure_der_cleaned_rate(self):

        self.df_pressure = self.change_small_values_to_zero(self.df_pressure, 'pressure_1der')

        fig = make_subplots(rows=2, cols=1, subplot_titles=['Time (hr) vs First derivative of pressure (psi)', 'Time (hr) vs Rate (STB/D)'])  # Repeating the graph above with plotly

        # Plotting time vs Pressure
        fig.add_trace(go.Scatter(x=self.df_pressure['Test Design 2_Date Time'], y=self.df_pressure['pressure_1der']), row=1, col=1) 
        # Plotting time vs Rate
        fig.add_trace(go.Scatter(x=self.df_rate['Oil rate_Date Time'], y=self.df_rate['Oil rate_Liquid rate']), row=2, col=1) 

        fig.update_layout(height=600, width=800)

        fig.update_xaxes(title_text='Time') # adding x axis title
        fig.update_yaxes(title_text='First derivative of pressure', row=1, col=1) # adding y axis title
        fig.update_yaxes(title_text='Rate', row=2, col=1) # adding y axis title

        return fig

    def cluster_spikes(self,df, column):
        df['is_non_zero'] = df[column] != 0
        df['cluster'] = df['is_non_zero'].ne(df['is_non_zero'].shift()).cumsum()
        df['cluster'] = df['cluster'].where(df['is_non_zero'], None)
        df['cluster'] = df['cluster'].ffill().bfill()
        df['cluster'] = df['cluster'].astype(int)
        df.drop('is_non_zero',axis=1, inplace=True)
        return df

    def plot_buildup_zones(self):

        self.df_pressure = self.cluster_spikes(self.df_pressure, 'pressure_1der')
        self.df_pressure['cluster'] = self.df_pressure['cluster'] - 1
        # Create a list of tuples representing the start and end indices of each cluster
        clusters = [(self.df_pressure[self.df_pressure['cluster'] == c].index[0], self.df_pressure[self.df_pressure['cluster'] == c].index[-1]) for c in self.df_pressure['cluster'].unique()]

        # Create the line plot
        fig = px.line(self.df_pressure, x='Test Design 2_Date Time', y='Test Design 2_Pressure', title='Time vs Pressure (Multi-rate)')
        fig.update_traces(line_color='red')

        # Create a list of shapes and annotations to be added to the plot
        shapes = []
        annotations = []

        # get the max pressure
        max_pressure = self.df_pressure['Test Design 2_Pressure'].max()

        # Add shaded areas under the clusters and cluster labels
        for i, (start, end) in enumerate(clusters):
            if i != 0: #skipping the first cluster
                # Generate a random color for the cluster
                color = '#' + ''.join([random.choice('123456789') for j in range(6)])
                # Add the shape
                shapes.append(
                    dict(
                        type='rect',
                        xref='x',
                        yref='paper',
                        x0=self.df_pressure['Test Design 2_Date Time'].iloc[start],
                        x1=self.df_pressure['Test Design 2_Date Time'].iloc[end-1],
                        y0=0,
                        y1=1,
                        fillcolor=color,
                        opacity=0.5,
                        layer='below',
                        line_width=0
                    )
                )
                # Add the annotation
                if (max_pressure - (max_pressure*0.03) <= self.df_pressure['Test Design 2_Pressure'].iloc[start:end].mean() <= max_pressure + (max_pressure*0.03)):

                    annotations.append(
                        dict(        x=self.df_pressure['Test Design 2_Date Time'].iloc[(start+end)//2],
                y=1.2,
                xref='x',
                yref='paper',
                text='buildup',
                textangle = -45,
                showarrow=False,
                font=dict(color=color)
            ))
                else:
                    annotations.append(
                        dict(
                            x=self.df_pressure['Test Design 2_Date Time'].iloc[(start+end)//2],
                            y=1.2,
                            xref='x',
                            yref='paper',
                            text=f'Cluster {i+1}',
                            textangle = -45,
                            showarrow=False,
                            font=dict(color=color)
                        )
                    )

        # Update the layout
        fig.update_layout(
            shapes=shapes,
            annotations=annotations,
            title_x=0.5
        )

        # Show the plot
        return fig
