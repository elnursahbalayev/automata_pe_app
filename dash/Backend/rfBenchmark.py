import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import cufflinks as cf
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.subplots import make_subplots


class RFB:
    def __init__(self):
        pass

    def dataformat(self):
        """
        Create unique labels and calculate Kh & Rf
        Return: RF dataset and grouped RF dataset(by block)
        """
        self.data = self.create_unique_label()
        self.data = self.calculate_kh()
        self.data_grouped = self.group_by_block()
        self.data_grouped = self.calculate_rf()  

        return self.data, self.data_grouped  


    def load_data(self, file_name):
        """
        Load data and preprocess it
        Parameter: Path to dataset
        Return: Processed dataset
        """
        self.data = pd.read_csv(file_name)
        self.data.dropna(inplace=True)
        self.data.drop_duplicates(inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        self.format_colnames()
        return self.data

    def format_colnames(self):
        """
        Format column names for consistency
        Parameter: Dataset
        """
        columns = self.data.columns
        for i in range(len(columns)):
            if columns[i].lower() in ['thickness', 'lock thickness', 'h']:
                self.data.rename(columns={columns[i]: 'block thickness, ft'}, inplace=True)
            if columns[i].lower() in ['saturation', 'water saturation', 'sw','saturation %', 'water saturation %']:
                self.data.rename(columns={columns[i]: 'Sw %'}, inplace=True)
            if columns[i].lower() in ['porosity', 'porosity %', 'poro']:
                self.data.rename(columns={columns[i]: 'poro %'}, inplace=True)
            if columns[i].lower() in ['ooip']:
                self.data.rename(columns={columns[i]: 'OOIP, MMSTB'}, inplace=True)
            if columns[i].lower() in ['lat']:
                self.data.rename(columns={columns[i]: 'latitude'}, inplace=True)
            if columns[i].lower() in ['lon', 'long']:
                self.data.rename(columns={columns[i]: 'longitude'}, inplace=True)
            if columns[i].lower() in ['perforation length', 'perforation len', 'perf len', 'perforation len, ft']:
                self.data.rename(columns={columns[i]: 'perforation length, ft'}, inplace=True)
            if columns[i].lower() in ['perforation permeability', 'perforation permeability, md','perf perm']:
                self.data.rename(columns={columns[i]: 'perf perm, md'}, inplace=True)
            if columns[i].lower() in ['perforation ntg', 'ntg']:
                self.data.rename(columns={columns[i]: 'perf NTG'}, inplace=True)
            if columns[i].lower() in ['eur', 'ultimate recovery', 'ultimate recovery, mmstb']:
                self.data.rename(columns={columns[i]: 'EUR, MMSTB'}, inplace=True)
    
    def create_unique_label(self):
        """
        Create unique label based on sand and fault block names
        Return : Dataset with new column(label)
        """
        self.data['block'] = self.data['sand'] + '-' + self.data['fault block'].astype(str)
        return self.data
    
    def calculate_kh(self):
        """
        Calculate Kh values
        Return: Updated dataset with Kh values
        """
        self.data['Kh'] = self.data['perforation length, ft'] * self.data['perf perm, md']*\
        self.data['perf NTG'] / 100
        return self.data
    
    def group_by_block(self):
        """
        Group data by fault block
        Return: Grouped dataset
        """
        self.data_grouped_sum = self.data.groupby(['fault block'], as_index=False).sum()
        self.data_grouped_mean = self.data.groupby(['fault block'], as_index=False).mean()
        self.data_grouped_mean = self.data_grouped_mean[self.data_grouped_mean.columns.difference(['Kh', 'EUR, MMSTB'])]
        self.data_grouped_mean[['Kh', 'EUR, MMSTB']] = self.data_grouped_sum[['Kh', 'EUR, MMSTB']]
        self.data_grouped = self.data_grouped_mean
        return self.data_grouped
    
    def calculate_rf(self):
        """
        Calculate RF values
        Return: Updated dataset with RF values
        """
        self.data_grouped['RF'] =  self.data_grouped['EUR, MMSTB'] / \
         self.data_grouped['OOIP, MMSTB'] * 100
        return self.data_grouped
    
    def plot_p50(self, type='loglog'):
        """
        Plot P50 graph based on scale choice(loglog / semilog)
        Parameter:
        type:str
        Choice of scale(loglog / semilog)
        default - loglog        
        """
        if type == 'loglog':
            #create linear regression model
            self.lr = LinearRegression()
            #get log values of Kh and RF
            x = np.log10(self.data_grouped['Kh'])
            y = np.log10(self.data_grouped['RF'])
            #fit the data
            self.lr.fit(x.values.reshape(-1, 1), y)
            #predict RF values
            self.data_grouped['RF pred'] = np.power(10, self.lr.predict(x.values.reshape(-1, 1)))
            self.data_grouped.head()
            #sort the data by Kh values
            self.data_grouped.sort_values(by='Kh',inplace=True)
            #create P50 plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.data_grouped['Kh'],\
                        y=self.data_grouped['RF pred'],
                                mode='lines',
                                name='P50'))
            fig.add_trace(go.Scatter(x=self.data_grouped['Kh'], \
                        y=self.data_grouped['RF'],
                                mode='markers', name='Actual RF'))
                
            #change scale of x and y axis
            fig.update_xaxes(type="log")
            fig.update_yaxes(type="log")
            fig.update_layout(
                title="Log-log graph of Kh vs Rf",
                xaxis_title="Kh",
                yaxis_title="Recovery factor (Rf)",
)
            # fig.show()
            return fig
        elif type == 'semilog':
            #create linear regression model
            self.lr = LinearRegression()
            #get log values of Kh 
            x = np.log10(self.data_grouped['Kh'])
            y = self.data_grouped['RF']
            #fit the data
            self.lr.fit(x.values.reshape(-1, 1), y)
            #predict RF
            self.data_grouped['RF pred'] = self.lr.predict(x.values.reshape(-1, 1))
            self.data_grouped.head()
            #sort the values based on Kh
            self.data_grouped.sort_values(by='Kh',inplace=True)
            #create P50 graph
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.data_grouped['Kh'],\
                        y=self.data_grouped['RF pred'],
                                mode='lines',
                                name='P50'))
            fig.add_trace(go.Scatter(x=self.data_grouped['Kh'], \
                        y=self.data_grouped['RF'],
                                mode='markers', name='Actual RF'))
            fig.update_xaxes(type="log")
            fig.update_layout(
                title="Semi-log graph of Kh vs Rf",
                xaxis_title="Kh",
                yaxis_title="Recovery factor (Rf)",
)
            # fig.show()
            return fig

    def get_coef_and_intercept(self):
        """
        Get coefficient and intercept from linear regression models
        Return:  coefficient and intercept
        """
        return self.lr.coef_, self.lr.intercept_

    def detect_outliers_iso_forest(self, contamination=0.01):
        """
        Detect outliers based on the dataframe provided
        Parameter:
        contamination:float
        Contamination of Isolation Forest
        """
        #find the outliers
        indeces = IsolationForest(contamination=contamination, random_state=1234). \
            fit_predict(self.data_grouped[['RF']])
        outliers = indeces.tolist().count(-1)
        #plot outliers
        cf.go_offline()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data_grouped['Kh'][indeces==1], y=self.data_grouped['RF'][indeces==1],
                            mode='markers', name='Normal values', line=dict(color='Blue')))
        fig.add_trace(go.Scatter(x=self.data_grouped['Kh'][indeces==-1], y=self.data_grouped['RF'][indeces==-1],
                            mode='markers', name='Outliers',line=dict(color='Red')))
        fig.update_layout(title='Outlier detection', xaxis_title='Kh', yaxis_title='Rf')
        #update scale of y and x axes
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
        # fig.show()
        #ask user to keep or remove the outliers detected
        # remove = input(f'There are {outliers} outliers in RF column. Do you want to remove or keep them? Yes/No: ').lower()
        remove='yes'
        if remove=='yes':
            self.data_grouped.drop(self.data_grouped.index[indeces == -1], inplace=True)
            self.plot_p50()
        else:
            pass
        return fig
    
    def plot_margins(self):
        """
        Plot margins for P90 and P10
        """
        #create linear regression model
        self.lr = LinearRegression()
        #get the log of Kh 
        x = np.log10(self.data_grouped['Kh'])
        y = self.data_grouped['RF']
        #fit the data
        self.lr.fit(x.values.reshape(-1, 1), y)
        #predict Rf
        self.data_grouped['RF pred'] = self.lr.predict(x.values.reshape(-1, 1))
        self.data_grouped.head()
        #sort the values based on Kh
        self.data_grouped.sort_values(by='Kh',inplace=True)
        #ask user to input margin multiplier
        # self.margin_multiplier = float(input('Enter the margin multiplier: (As a rule it will be multiplied to standard deviation)'))
        self.margin_multiplier = 1
        #calculate p10 and p90
        self.data_grouped['P90'] = self.data_grouped['RF pred'] - (self.lr.intercept_ \
            - self.data_grouped['RF'].std()*self.margin_multiplier)
        self.data_grouped['P10'] = self.data_grouped['RF pred'] + (self.lr.intercept_ \
            - self.data_grouped['RF'].std()*self.margin_multiplier)
        
        #plot p10 and p90
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data_grouped['Kh'], y=self.data_grouped['RF pred'],
                            mode='lines',
                            name='P50'))
        fig.add_trace(go.Scatter(x=self.data_grouped['Kh'], y=self.data_grouped['RF'],
                            mode='markers', name='Actual RF'))
        fig.add_trace(go.Scatter(x=self.data_grouped['Kh'], y=self.data_grouped['P10'],
                            mode='lines', name='P10'))
        fig.add_trace(go.Scatter(x=self.data_grouped['Kh'], y=self.data_grouped['P90'],
                            mode='lines', name='P90'))
        fig.update_xaxes(type="log")
        fig.update_layout(
                title="Semi-log graph of Kh vs Rf",
                xaxis_title="Kh",
                yaxis_title="Recovery factor (Rf)",
)
        # fig.show()
        return fig
    def scatterplot(self):
        """
        Create scatter plot of Latitude and Longitude 
        Return: updated dataset
        """
        #ask user to input scale value to enlarge the map
        # scale = float(input('How many times the map should be enlarged? '))
        scale = 2
        #create new radius based on scale value
        self.data_grouped['radius, ft'] = self.data_grouped['radius, ft']* scale
        #plot the latitude and longitude
        fig = px.scatter(self.data_grouped, x='latitude', y='longitude ', size='radius, ft')
        # fig.show()

        return self.data_grouped, fig
    
    def mapplot(self,source_image):
        """
        Create bubble plot on the map provided by user
        Parameter:
        source_image:str
        path to source image
        Return:
        Dataset
        """
        #create bubble plot
        fig=go.Figure()
        fig.add_scatter(x=self.data_grouped['latitude'], y=self.data_grouped['longitude '], mode='markers', marker=dict(size=self.data_grouped['radius, ft']/150))
        fig.update_layout(template="plotly_white", width=700, height=700,
                        xaxis_showgrid=False, yaxis_showgrid=False)
        fig.add_layout_image(
        source=source_image,
        x=0,
        y=1,
        xanchor="left",
        yanchor="top",
        layer="below",
        sizing="stretch",
        sizex=1.0,
        sizey=1.0
        )
        # fig.show()
        return  fig





