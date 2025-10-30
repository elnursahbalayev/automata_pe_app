import plotly.graph_objects as go
import cufflinks as cf
from plotly.subplots import make_subplots
import wellpathpy as wp #pip install wellpathpy
import plotly.express as px

class Wellpath():
    #initiate the class
    def __init__(self, name):
        self.name = name
      
    def plot_wellpath(dataframe):
        """
        Create well path based on wellpath dataframe
        Parameter:
        dataframe:dataframe
        dataset of wellpath
        """
        #create subplots
        fig = make_subplots(rows=1, cols=3)
        #add trace for easting and depth
        fig.add_trace(
            go.Scatter(x=dataframe['easting'], y=dataframe['depth']),
            row=1, col=1
        )
        #add trace for northing and depth
        fig.add_trace(
            go.Scatter(x=dataframe['northing'], y=dataframe['depth']),
            row=1, col=2
        )
        #add trace for easting and northing
        fig.add_trace(
            go.Scatter(x=dataframe['easting'], y=dataframe['northing']),
            row=1, col=3
        )

        # Update xaxis properties
        fig.update_xaxes(title_text="Easting", row=1, col=1)
        fig.update_xaxes(title_text="Northing", row=1, col=2)
        fig.update_xaxes(title_text="Easting", row=1, col=3)
        

        # Update yaxis properties
        fig.update_yaxes(title_text="Depth", row=1, col=1)
        fig.update_yaxes(title_text="Depth", row=1, col=2)
        fig.update_yaxes(title_text="Northing", row=1, col=3)
        

        fig.update_layout(showlegend = False , title_text="Minimum curvature plots")
        # fig.show()
        return fig
    
    
    
    def plot_wellpath_3d(dataframe):
        """
        Create 3d map based on wellpath dataframe
        Parameter:
        dataframe:dataframe
        Dataset of wellpath
        """
        #add trace of northing and depth
        fig = go.Figure(data=[go.Scatter3d(x=dataframe['easting'].values,
           y=dataframe['northing'].values,
           z=dataframe['depth'].values,
            mode='lines'
          )])
        
        #update layout of plot
        fig.update_layout(
            scene = dict(
                xaxis_title="X Location",
                yaxis_title="Y Location",
                zaxis_title="TVD"),
                title_text="3D Plot"
            )
        

        # fig.show()

        return fig
    def wellpath_deviation(dataframe):
        """
        Calculate easting, northing and depth from MD(measured depth),
        INC(inclination) and AZI(azimuth)
        Parameter:
        dataframe : dataframe
        dataset of wellpath
        Return:
        updated dataframe
        """
        #extract md, inc and azi
        survey_subset = dataframe[['MD', 'INC', 'AZI']]
        md = survey_subset.MD
        inc = survey_subset.INC
        azi = survey_subset.AZI
        #calculate deviation
        dev = wp.deviation(md, inc, azi)
        minimum_curvature = dev.minimum_curvature()
        #add values to dataset
        dataframe['easting'] = minimum_curvature.easting
        dataframe['northing'] = minimum_curvature.northing
        dataframe['depth'] = minimum_curvature.depth
        return dataframe