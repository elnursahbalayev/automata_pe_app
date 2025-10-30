import pandas as pd
import numpy as np
from scipy import interpolate
import re
import plotly.graph_objects as go
from Backend.dataPreprocessing import dataprocess as datap
import cufflinks as cf
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest, RandomForestRegressor
import plotly.express as px
import os



class pvt_process():
    def __init__(self):
        pass

    @staticmethod
    def remove_zero_and_negatives(dataframe):
        """This function  is used remove any zeros and negative values in the dataframe. Input: dataframe"""
        dataframe = dataframe[(dataframe != 0).all(axis=1)]
        # dataframe = dataframe[(dataframe != 0).all(axis=1)]
        bad_index = dataframe[dataframe[dataframe._get_numeric_data()<0].sum(axis=1)<0].index
        print('There were removed {} of negative numbers in the data'.format(len(bad_index)))
        dataframe.drop(bad_index, inplace=True)
        dataframe.reset_index(drop=True, inplace=True)
        return dataframe
    

    def pvtPreprocessing(self, monthly_prod, df_csv_temp, pvt_data ):

        prod_df = monthly_prod
        dataframe = df_csv_temp

        for i in dataframe.columns:
            if re.search(r"TT", i): dataframe.drop(i, axis=1, inplace=True)
            if re.search(r"AT", i): dataframe.drop(i, axis=1, inplace=True)

        _,_,_,ap_reading,_ = datap.num_gaugedetect(dataframe)

        length_before = (len(dataframe.columns)-1)//2
        for i in range(length_before):
            dataframe[f'AP{i+1}-TP{i+1}'] = dataframe[dataframe.columns[(i+1)*2]]-dataframe[dataframe.columns[(i+1)*2-1]]

        DIFFERENCE_DROPPING_CRITERIA = 25
        for i in range (length_before):
            dataframe.drop(dataframe[(np.abs(dataframe[f'AP{i+1}-TP{i+1}']) > DIFFERENCE_DROPPING_CRITERIA)].index, inplace=True)

        dataframe.reset_index(inplace=True, drop=True)

        dataframe['year'] = dataframe['Date'].dt.year
        dataframe['month'] = dataframe['Date'].dt.month


        df_new = dataframe.groupby(['year', 'month']).mean()

        df_new['AP_AVERAGE'] = 0
        for i in ap_reading:
            df_new['AP_AVERAGE'] = df_new['AP_AVERAGE'] + df_new[i]

        df_new['AP_AVERAGE'] = df_new['AP_AVERAGE']/len(ap_reading)
        prod_df['AP_AVERAGE'] = pd.DataFrame(df_new['AP_AVERAGE'].to_numpy())

        df_new_ap = pd.DataFrame(df_new['AP_AVERAGE'].to_numpy(), columns=['AP_AVERAGE'])
        df_new_ap.sort_values(['AP_AVERAGE'],inplace=True, ignore_index=True)

        
        """## Preprocessing RS, Bo, Bw, Bg data"""

        # print(pvt_data.columns)
        pvt_data.sort_values(by=['pressure'], inplace=True, ignore_index=True)
        x = pvt_data['pressure']
        y = pvt_data['Rs']
        tck = interpolate.splrep(x, y, s=0)

        xnew = df_new_ap['AP_AVERAGE']
        ynew = interpolate.splev(xnew, tck, der=0)
        df_new_ap['Rs'] = pd.DataFrame(ynew)

        x = pvt_data['pressure']
        y = pvt_data['Bo']
        tck = interpolate.splrep(x, y, s=0)

        xnew = df_new_ap['AP_AVERAGE']
        ynew = interpolate.splev(xnew, tck, der=0)

        df_new_ap['Bo'] = pd.DataFrame(ynew)
        pvt_data.dropna(subset=['Bg'], inplace=True)

        x = pvt_data['pressure']
        y = pvt_data['Bg']
        tck = interpolate.splrep(x, y, s=0)

        xnew = df_new_ap['AP_AVERAGE']
        ynew = interpolate.splev(xnew, tck, der=0)
        df_new_ap['Bg'] = pd.DataFrame(ynew)
        df_new_ap['Bw'] = pd.DataFrame(np.ones(len(df_new_ap)))


        df_final = pd.merge(prod_df, df_new_ap, on="AP_AVERAGE")
        df_final.rename(columns={'AP_AVERAGE':'AVG_PRESSURE'}, inplace=True)

        try:
            df_final.drop(['Unnamed: 0'], axis=1, inplace=True)
        except:
            df_final = df_final
        # df_final['Date'] = pd.to_datetime(df_final['Date'])

        return df_final

    ############################### TOTAL IN PRODUCTION CALCULATION #############################################
    
    @staticmethod
    def calculate_total_day(dataframe):
        """
        Calculate the total day in production based on the dataframe provided
        """
        #unit for time
        dataframe['Total Day'] = round(dataframe['On Stream']/24, 3)
        return dataframe
    
    ############################### TECHNICAL PRODUCTION CALCULATION #############################################
    @staticmethod
    def calculate_technical_production(dataframe):
        """
        Calculate the water technical production based on the dataframe provided
        """
        dataframe['Water Technical Production Rate'] = round(dataframe['Water'] / dataframe['Total Day'],3)

        """
        Calculate the oil technical production based on the dataframe provided
        """
        dataframe['Oil Technical Production Rate'] = round(dataframe['Oil'] / dataframe['Total Day'], 3)

        """
        Calculate the gas technical production based on the dataframe provided
        """
        dataframe['Gas Technical Production Rate'] = round(dataframe['Gas'] / dataframe['Total Day'], 3)
        
        return dataframe
    
    ############################### WC, GOR, Free gas, fw, fo, fg  CALCULATION #############################################
    
    @staticmethod
    def calculate_wc_technical(dataframe):
        """
        Calculate the WC based on Water and Total rate data from dataframe provided
        """
        dataframe['Total'] = dataframe['Water Technical Production Rate']+dataframe['Oil Technical Production Rate']
        dataframe['WC_Technical'] = dataframe['Water Technical Production Rate'] / dataframe['Total']
        dataframe.drop(['Total'], axis=1, inplace=True)
        return dataframe
    
    @staticmethod
    def calculate_gor_technical(dataframe):
        """
        Calculate the GOR based on Gas and Oil rate from dataframe provided
        """
        #the unit of the data 
        dataframe['GOR_Technical'] = dataframe['Gas Technical Production Rate']*1000 / dataframe['Oil Technical Production Rate']
        return dataframe

    @staticmethod 
    def calculate_free_gas_technical(dataframe):
        """Calculate the free gas based on the dataframe provided"""
        dataframe['Free Gas Technical'] = round(dataframe['Oil Technical Production Rate']*(dataframe['GOR_Technical'] - dataframe['Rs']), 3)
        return dataframe

    @staticmethod
    def calculate_fg_technical(dataframe):
        """Calculate the fg based on the dataframe provided
        Fg= Free Gas (sc) * Bg / (Free Gas (sc) * Bg + Qo * Bo + Qw*Bw)"""
        dataframe['fg'] = round(dataframe['Free Gas Technical']*dataframe['Bg']/(dataframe['Free Gas Technical']*dataframe['Bg']+dataframe['Oil Technical Production Rate']*dataframe['Bo']+dataframe['Water Technical Production Rate']*dataframe['Bw']), 3)
        return dataframe
    
    @staticmethod
    def calculate_fw_technical(dataframe):
        """Calculate the fw based on the dataframe provided
        fw= Qw*Bw / (Free Gas (sc) * Bg  + Qo * Bo + Qw*Bw)"""
        dataframe['fw']= round(dataframe['Water Technical Production Rate']*dataframe['Bw']/(dataframe['Free Gas Technical']*dataframe['Bg']+dataframe['Oil Technical Production Rate']*dataframe['Bo']+dataframe['Water Technical Production Rate']*dataframe['Bw']), 3)
        return dataframe
    
    @staticmethod
    def calculate_fo_technical(dataframe):
        """Calculate the fo based on the dataframe provided
        fo = 1 – fg – fw"""
        dataframe['fo'] = round(1-dataframe['fg']-dataframe['fw'], 3)
        return dataframe

    ############################### DATASET ADJUSTMENT #############################################   
    
    @staticmethod 
    def remove_commas(dataframe, labels):
        """Remove commas in numeric data"""
        for i in labels:
            dataframe[i] = dataframe[i].str.replace(',', '')
        return dataframe

    
    ############################### UNIT CONVERSION #############################################
    #conversion 
    @staticmethod
    def unitConversion(dataframe):
        # dataframe = cls.convert_to_stb_per_day(dataframe, 'Water')
        dataframe['Water'] = round(dataframe['Water'] * 6.28981,3)
        # dataframe = cls.convert_to_stb_per_day(dataframe, 'Oil')
        dataframe['Oil'] = round(dataframe['Oil'] * 6.28981,3)
        # dataframe = cls.convert_to_mscf_per_day(dataframe, 'Gas')
        dataframe['Gas'] = round(dataframe['Gas'] * 35.3146/1000, 3)

        return dataframe

    @staticmethod
    def convert_to_stb_per_day(dataframe, column_label):
        """
        Convert the column label to stb per day
        """
        dataframe[column_label] = round(dataframe[column_label] * 6.28981,3)
        return dataframe

    @staticmethod
    def convert_to_mscf_per_day(dataframe, column_label):
        """
        Convert the column label to stb per day
        """
        dataframe[column_label] = round(dataframe[column_label] * 35.3146/1000, 3)
        return dataframe
    
    def plot_fg_fw_fo_vs_time_calendar_rate(self,dataframe):
        """Plot the fg, fw and fo vs time based on the dataframe provided"""
        cf.go_offline()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['fg'], name='fg', mode='lines+markers', line=dict(color='Red')))
        fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['fw'], name='fw', mode='lines+markers', line=dict(color='Blue')))
        fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['fo'], name='fo', mode='lines+markers', line=dict(color='Green')))
        fig.update_layout(title='fg & fw & fo vs Date', xaxis_title='Date', yaxis_title='fg & fw & fo')
        # fig.show()
        return fig

    @staticmethod
    def plot_production_data_technical_rate(dataframe):
        """Plot the production data vs date based on the dataframe provided"""
        cf.go_offline()
        # fig = go.Figure()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['Oil Technical Production Rate'], name='Oil Technical Production Rate', mode='lines+markers', line=dict(color='Green')), secondary_y=False)
        fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['Water Technical Production Rate'], name='Water Technical Production Rate', mode='lines+markers', line=dict(color='Blue')), secondary_y=False)                                                            
        fig.add_trace(go.Scatter(x=dataframe['Date'], y=dataframe['Gas Technical Production Rate'], name='Gas Technical Production Rate', mode='lines+markers', line=dict(color='Red')),secondary_y=True)
        fig.update_yaxes(title_text="<b>Water and Oil </b> production rate", secondary_y=False)
        fig.update_yaxes(title_text="<b>Gas</b> Production Rate", secondary_y=True)       
        fig.update_layout(title='Production rate vs Date', xaxis_title='Date')
        # fig.show()
        return fig

#water is blue, oil is green, gas is red
#add the markers

    ############################### PRODUCTION RATE #############################################
    
    @staticmethod
    def get_days_in_month(data, column):
        """gets the date column in the date and adds back to the dataframe how many days there were in given months"""
        data[column] = pd.to_datetime(data[column])
        data['days in month'] = data[column].dt.days_in_month
        return data

    @staticmethod
    def get_flow_rates_calendar_rate(data, columns):
        """calculates production rate from production data"""
        for column in columns:
            data[f'{column} calendar rate'] = data[column] / data['days in month']
        return data



    ############################### PREDICTION #######################################################    
    def fit(self, dataframe, x='Date', y='fo'):
        self.rf = RandomForestRegressor(random_state=1234)
        print(dataframe)
        self.rf.fit(dataframe[x].index.to_numpy().reshape(-1, 1), dataframe[y])
        print(dataframe)

        

    def predict(self, dataframe, x='Date', y='fo'):
        """Predict fo, fg or fw based on the dataframe provided and plot the result"""
        # rf = RandomForestRegressor(random_state=1234)
        # rf.fit(dataframe[x].index.to_numpy().reshape(-1, 1), dataframe[y])

        self.y_pred = self.rf.predict(dataframe[x].index.to_numpy().reshape(-1, 1))

        dataframe[f'predicted {y}'] = pd.DataFrame(self.y_pred)


        cf.go_offline()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dataframe[x], y=dataframe[y], name=f'actual {y} vs time ', mode='lines+markers', line=dict(color='Blue')))
        fig.add_trace(go.Scatter(x=dataframe[x], y=self.y_pred, name=f'predicted {y} vs time', mode='lines+markers', line=dict(color='Red')))
        fig.update_layout(title=f'Actual vs Predicted {y}', xaxis_title=x, yaxis_title=y)
        # fig.show()


        df, fig_res = self.plot_residuals(dataframe, y)

        return dataframe, fig, fig_res

    ############################### Outliers Detection #######################################################           
    
    @staticmethod
    def detect_outliers_iso_forest(dataframe,residuals, column, contamination=0.005):
        """Detect outliers based on the dataframe provided"""
        dataframe['Date'] = pd.to_datetime(dataframe['Date'])
        indeces = IsolationForest(contamination=contamination, random_state=1234).fit_predict(dataframe[[residuals]])
       
        outliers = indeces.tolist().count(-1)

        cf.go_offline()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=dataframe['Date'][indeces == 1], y=dataframe[column][indeces == 1], name='Normal values',
                       mode='markers', line=dict(color='Blue')))
        fig.add_trace(go.Scatter(x=dataframe['Date'][indeces == -1], y=dataframe[column][indeces == -1],
                                 name='Outliers', mode='markers', line=dict(color='Red')))
        fig.update_layout(title='Outlier detection', xaxis_title='Date', yaxis_title=f'{column}')
        # fig.show()

        # remove = input(f'There are {outliers} outliers in {column} column. Do you want to remove or keep them? Yes/No: ').lower()
        remove = 'yes'
        if remove=='yes':
            dataframe.drop(dataframe.index[indeces == -1], inplace=True)
        else:
            pass
        return dataframe, fig
    
    def detect_outliers_by_percent(self, dataframe,residuals,column, percent=.2):
        """Detect outliers based on the dataframe provided"""
        dataframe['Date'] = pd.to_datetime(dataframe['Date'])
        residuals = np.abs(residuals).sort_values(ascending=True)
        normal_values = residuals[:int(len(residuals) * percent)].index
        outliers = residuals[int(len(residuals) * percent):].index

        cf.go_offline()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=dataframe['Date'][normal_values], y=dataframe[column][normal_values], name='Normal values',
                       mode='markers', line=dict(color='Blue')))
        fig.add_trace(go.Scatter(x=dataframe['Date'][outliers], y=dataframe[column][outliers],
                                 name='Outliers', mode='markers', line=dict(color='Red')))
        fig.update_layout(title='Outlier detection', xaxis_title='Date', yaxis_title=f'{column}')
        fig.show()

        remove = input(f'There are {outliers} outliers in {column} column. Do you want to remove or keep them? Yes/No: ').lower()
        if remove=='yes':
            dataframe.drop(dataframe.index[indeces == -1], inplace=True)
        else:
            pass
        return dataframe

    def detect_outliers(self, dataframe, column, contamination=0.01):
        try:
            self.fit(dataframe)
            dataframe, _, _ = self.predict(dataframe)
            dataframe['residuals'] = ((dataframe[column] - self.y_pred))
            _, fig = self.detect_outliers_iso_forest(dataframe, residuals='residuals',column=column , contamination=contamination)
        except:
            print('Inconsistency with PVT data. Make sure the data used are correct')
            fig = None
            # os._exit(0) 
        return dataframe, fig

    def detect_outliers_v2(self, dataframe, column, percent=0.2):
        try:
            self.fit(dataframe)
            dataframe = self.predict(dataframe)
            dataframe['residuals'] = ((dataframe[column] - self.y_pred))
            self.detect_outliers_by_percent(dataframe, residuals='residuals',column=column, percent = percent)
        except:
            print('Inconsistency with PVT data. Make sure the data used are correct')
            # os._exit(0) 
        return dataframe

    def plot_residuals(self, dataframe, column):
        """Plot the residuals based on the dataframe provided in histogram"""
        dataframe['residuals'] = ((dataframe[column] - self.y_pred))
        cf.go_offline()
        fig = px.histogram(dataframe, x='residuals')
        # fig.show()
        return dataframe, fig