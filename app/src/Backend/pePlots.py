import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go



class Plot():
    def __init__(self):
        pass

    @staticmethod
    def chanplot(dataframe):
        """
        Creating chanplot using PVT data
        Input: PVT dataset
        Return: Updated PVT dataset 
        """
        #resetting index
        dataframe.reset_index(drop=True, inplace=True)
        #calculating WOR
        dataframe['WOR'] = dataframe['Water Technical Production Rate'] / dataframe['Oil Technical Production Rate']
        #calculating incremental WOR
        for i in range(len(dataframe)):
            dataframe.loc[i,'dWOR'] = (dataframe.loc[i,'WOR'] - dataframe.loc[0,'WOR'])
            dataframe['Date'] = pd.to_datetime(dataframe['Date'])
        #compressing timeframe to month
        for i in range(len(dataframe)):
            dataframe.loc[i,'nb_months'] = ((dataframe.loc[i,'Date'] - dataframe.loc[0,'Date'])/np.timedelta64(1, 'M'))
        #converting to integer
        dataframe['nb_months'] = np.round(dataframe['nb_months']).astype(int)
        dataframe['dWOR/dT'] = dataframe['dWOR'] / dataframe['nb_months']
        #creating chan plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(mode="markers", x=dataframe["nb_months"], y=dataframe["dWOR/dT"], name = 'dWOR/dT'))
        fig.add_trace(go.Scatter(mode="markers", x=dataframe["nb_months"], y=dataframe["WOR"], name = 'WOR'))
        fig.update_layout(
            title="Chan Plot",
            xaxis_title="Elapsed time(months)",
            yaxis_title="WOR or WOR'",
        )
        #change x and y axis to log scale
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
        # fig.show()
        
        return dataframe, fig

    @staticmethod
    def hallplot(dataframe, Volume_WI, Average_p):
        """
        Creating hallplot using PVT data
        Input: PVT dataset, Volume of Water Injected, Average reservoir Pressure
        Return: Updated PVT dataset 
        """
        #resetting index
        dataframe.reset_index(drop=True, inplace=True)
        #caclulating pressure difference
        dataframe['delta_p'] = dataframe['AVG_PRESSURE'] - Average_p
        #generating number of injection days using random module
        dataframe['inj_days'] = np.random.randint(15, 30, size=len(dataframe))
        dataframe['Dp*Dt'] = dataframe['delta_p'] * dataframe['inj_days']
        # try:
        dataframe['WI_volume'] = np.array(Volume_WI)
        # except:
            # print('Inconsistent values detected. Please provide adequate number of values for Water Injection')
        dataframe['Cumulative_WI'] = np.cumsum(dataframe['WI_volume'])
        #creating hall plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(mode="markers", x= dataframe["Cumulative_WI"], y= dataframe["Dp*Dt"]))

        fig.update_layout(
            title="Hall Plot",
            xaxis_title="Cumulative Water Injection",
            yaxis_title="Dp*Dt'",
        )
        # fig.show()
        
        return dataframe, fig


    @staticmethod
    def plot_VRR(dataframe):
        """
        Creating VRR and IVRR plots using PVT data
        Input: PVT dataset
        Return: Updated PVT dataset 
        """
        #calculating Voidage Replacement Ratio(VRR) and Instantaneous VRR
        dataframe['IVRR'] = dataframe['WI_volume'] / (dataframe['Oil']+ dataframe['Water']+ dataframe['Free Gas Technical'])
        dataframe['VRR'] = np.cumsum(dataframe['IVRR'])
        #creating VRR and IVRR plot
        fig = go.Figure()

        fig.add_trace(go.Scatter(mode="markers", x=dataframe["Date"], y=dataframe["VRR"], name = 'VRR'))
        fig.add_trace(go.Scatter(mode="markers", x=dataframe["Date"], y=dataframe["IVRR"], name = 'IVRR'))

        fig.update_layout(
            title="Voidage Replacement Ratio",
            xaxis_title="Date",
            yaxis_title="IVRR and VRR",
        )
        # fig.show()

        return dataframe, fig