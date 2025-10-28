######################################################################################
############################ importing library and resources #########################
######################################################################################
    
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from csv import reader
import dask.dataframe as dd

######################################################################################
############################ importing and initiating class ##########################
######################################################################################

class dataprocess():
    def __init__(self):
        pass
    
    # def detectdatecolumn
    @staticmethod    
    def dataresampling(df):
        """This function  is used to resample the dataframe. Input: dataframe, time_range"""
        df = pd.DataFrame(data=df)
        time_range='5T'
        df = df.resample(time_range, on='Date').mean()
        df.reset_index(inplace=True)
        
        print("Data resampled")
        print('------------------------------------------------------------------------------------------------')
        return df
    
    @staticmethod
    def datahandling(df,choice,choice_duplicate):
        for i in df.columns:
            if df[i].isnull().sum() != 0:
                print(f'Note: column {i} has {df[i].isnull().sum()} null values. Either remove or keep them.')
        print(f'\nThere are total of {df.isnull().sum().sum()} null values.')
        print('------------------------------------------------------------------------------------------------')

        null_value = df.isnull().sum() 
        null_values_table = pd.DataFrame({'Columns': df.columns,
                                'Number of Null Values': [int(i) for i in null_value]})
        fig1 = px.bar(null_values_table,x='Columns', y='Number of Null Values', title=' Null values in columns')
        # fig.show()

        print('------------------------------------------------------------------------------------------------')
        if choice == '1':
            df.dropna(axis=0, how='any', subset=None, inplace=True)
            print('DONE. All null values are removed')
        elif choice == '2':
            df = df
            print('DONE. All null values are preserved. Warning is raised for further operation')
        else:
            print('Wrong choice!')

        null_value = df.isnull().sum() 
        null_values_table = pd.DataFrame({'Columns': df.columns,
                                'Number of Null Values': [int(i) for i in null_value]})
        fig2 = px.bar(null_values_table,x='Columns', y='Number of Null Values', title=' Null values in columns')
        # fig.show()
            
        print('------------------------------------------------------------------------------------------------')
        
        
        """This function  is used to detect and  remove the duplicates in the dataframe. USer has option to remove on
        Date column.  """
        if choice_duplicate == '1':
            total_duplicates_date = df.duplicated(subset=['Date']).sum()
            total_duplicates_all = df.duplicated().sum()

            print('Overall duplicates')
            df[df.duplicated(keep=False)].head()
            print('------------------------------------------------------------------------------------------------')
            print('Duplicates in Date column')
            df[df.duplicated('Date',keep=False)].head()


            print(f'There are {total_duplicates_all} duplicate rows and number of duplicates in Date column is {total_duplicates_date}')
            if total_duplicates_date>0:
                df.drop_duplicates(subset = ['Date'], inplace = True)
                print('All the duplicates in Date column has been removed successfully')

            if total_duplicates_all>0:
                df.drop_duplicates(inplace = True)
                print('All the other duplicates has been removed successfully!')
            
        elif choice == '2':
            print('DONE. All duplicate values are preserved. Warning is raised for further operation')
        else:
            print('Wrong choice!')
        
        print('------------------------------------------------------------------------------------------------')
            
        return df, fig1, fig2
    
    def findskiprow_text(self,filepath,type):
        #another way of reading file in case of header present 
        with open(filepath, 'r') as read_obj:
            line = []
            date_regex = [r'(\d{1,2})(\/|-|\s*)?((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)|\d{2})(\/|-|\s*)?(\d{4})',r'^(?:(?:(?:0?[13578]|1[02])(\/|-|\.)31)\1|(?:(?:0?[1,3-9]|1[0-2])(\/|-|\.)(?:29|30)\2))(?:(?:1[6-9]|[2-9]\d)?\d{2})$|^(?:0?2(\/|-|\.)29\3(?:(?:(?:1[6-9]|[2-9]\d)?(?:0[48]|[2468][048]|[13579][26])|(?:(?:16|[2468][048]|[3579][26])00))))$|^(?:(?:0?[1-9])|(?:1[0-2]))(\/|-|\.)(?:0?[1-9]|1\d|2[0-8])\4(?:(?:1[6-9]|[2-9]\d)?\d{2})$']
            stop_number = 0
            row_number = 0
            col = list() 


            file_reader = reader(read_obj, delimiter='\t')

            for row in file_reader:
                line.append(row)

                if re.search(r"Date", ''.join(row)) and not re.search(r"Date:", ''.join(row)) and not col :
                    col = row
                    # print(col)
                    skip_n = row_number
                    # print(skip_n)

                    
                if re.search(r"AP|TP", ' '.join(row)) and not col :
                    col = row
                    # print(col)
                    skip_n = row_number
                    # print(skip_n)


                #detecting 'Date' column according to the position of date started
                for i in date_regex:
                    if re.search(i, ''.join(row)) and stop_number == 0 and not re.search(r"Date:", ''.join(row)):
                        col_row_number = row_number - 1
                        if not col:
                            col = line[col_row_number]
                            skip_n = row_number
                        stop_number += 1

                        if col[0] == '':
                            col[0] = 'Date' # changing the column name into 'Date'

                        del line[col_row_number]
                        del line[0:col_row_number]
                        # print(col)
                        del line[0]
                        return skip_n,col

                row_number += 1   

 
    def findskiprow_csv(self,filepath,type):
        #another way of reading file in case of header present 
        with open(filepath, 'r') as read_obj:
            line = []
            date_regex = [r'(\d{1,2})(\/|-|\s*)?((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)|\d{2})(\/|-|\s*)?(\d{4})',r'^(?:(?:(?:0?[13578]|1[02])(\/|-|\.)31)\1|(?:(?:0?[1,3-9]|1[0-2])(\/|-|\.)(?:29|30)\2))(?:(?:1[6-9]|[2-9]\d)?\d{2})$|^(?:0?2(\/|-|\.)29\3(?:(?:(?:1[6-9]|[2-9]\d)?(?:0[48]|[2468][048]|[13579][26])|(?:(?:16|[2468][048]|[3579][26])00))))$|^(?:(?:0?[1-9])|(?:1[0-2]))(\/|-|\.)(?:0?[1-9]|1\d|2[0-8])\4(?:(?:1[6-9]|[2-9]\d)?\d{2})$']
            stop_number = 0
            row_number = 0


            file_reader = reader(read_obj)
            

            for row in file_reader:
                line.append(row)

                if re.search(r"Date", ''.join(row)) and not re.search(r"Date:", ''.join(row)):
                    col = row
                    # print(col)

                #detecting 'Date' column according to the position of date started
                for i in date_regex:
                        if re.search(i, ''.join(row)) and stop_number == 0 and not re.search(r"Date:", ''.join(row)):
                            col_row_number = row_number - 1
                            col = line[col_row_number]
                            stop_number += 1
        
                            col[0] = 'Date' # changing the column name into 'Date'
                            del line[col_row_number]
                            del line[0:col_row_number]
                            skip_n = row_number
                            # print(skip_n)
                            return skip_n,col

                row_number += 1    

    def build_dataframe(self,filepath,type):
        # running each line of code to read every csv with different format according to 'Date' position in the row

        if type == 'txt': skip_n, colnames = self.findskiprow_text(filepath,type)
        if type == 'csv': skip_n, colnames = self.findskiprow_csv(filepath,type)
        
        print('File reading is starting')
        print(skip_n)
        try:
            if type == 'csv': df_temp = dd.read_csv(filepath, skiprows = skip_n,names=colnames,dtype = 'object', assume_missing=True,low_memory=False) 
            if type == 'txt': df_temp = dd.read_csv(filepath, skiprows = skip_n,dtype = 'object',delimiter='\t', assume_missing=True,low_memory=False)
        except:
            if type == 'csv': df_temp = pd.read_csv(filepath, skiprows=skip_n,names=colnames, dtype = 'object', low_memory=False) 
            if type == 'txt': df_temp = pd.read_csv(filepath, skiprows=skip_n,names=colnames,delimiter='\t', dtype = 'object',low_memory=False) 

        try:    
            #detecting null column and drop the entire column
            col = []
            for i in df_temp.columns: col.append(str(i))
            for x in range(len(col)):
                if col[x] == 'nan' or col[x] == '':
                    df_temp = df_temp.drop(df_temp.columns[[x]],axis = 1)
        except:
            df_temp = df_temp
        
        return df_temp


    @staticmethod
    def formatting(df_temp):

        print('File reading is done')
        print('--------------------Converting invalid data into null values--------------------------------------')
        df_temp.replace(['No Data','no data','Comm Fail','Error','Arc Off-line','Not Connect','I/O Timeout','Scan Timeout','AccessDenied','Pt Created',''], np.nan)
        
        print("Invalid data handled successfully")
        print('-------------------------------------------------------------------------------------------------')
        print('------------------------Formatting datatype for each column--------------------------------------')
        print(df_temp.head())

        if 'Date' not in df_temp.columns:
            print('Date column is not detected. Kindly provide Date data')

        if 'Date' and 'Time' in df_temp.columns:
            df_temp['Date'] = df_temp['Date'] + " " +df_temp['Time']
        else:
            pass
        """Takes in the dataframe, converts necessary datatypes, and sorts the values by date. Returns the dataframe"""
        try:
            df_temp['Date'] = dd.to_datetime(df_temp['Date'])
        except:
            df_temp['Date'] = pd.to_datetime(df_temp['Date'])

        try:
            df_temp = df_temp.drop(columns=['Time'])
        except:
            pass
        
        for i in df_temp.columns:
            if i != 'Date':
                try:
                    df_temp[i] = dd.to_numeric(df_temp[i], errors='coerce')  
                except:
                    df_temp[i] = pd.to_numeric(df_temp[i], errors='coerce') 

        print(df_temp.head())
        print("Date and Columns are converted into their respective data type")
        print('-------------------------------------------------------------------------------------------------')
        
        try:
            df_temp = df_temp.compute()
        except:
            pass
        print(df_temp.head())
        return df_temp
    
    @staticmethod
    def sorting(df):
        print('--------------------Checking the date column and uniformity--------------------------------------')
        
        "check the date index in the dataframe"
        date_indeces = df.filter(like='-', axis=1).index
        print("Does the loaded file has Date as index?: {}".format(len(date_indeces) > 0))
        if len(date_indeces) > 0 : print("Is this uniform throughout the dataframe?: {}".format(len(date_indeces) == df.shape[0]))
        if len(date_indeces) > 0 : df.reset_index(inplace = True)
        if 'index' in df.columns : df.drop('index', inplace=True, axis=1)
            
        print('-------------------------------------------------------------------------------------------------') 
        df.sort_values('Date' , inplace=True)
        print("All values are sorted according to " + 'Date' + ' column')
        print('-------------------------------------------------------------------------------------------------')
        
        print('------------------------Removing zero and negative numbers---------------------------------------')
        df = df[(df != 0).all(axis=1)]
        bad_index = df[df[df._get_numeric_data()<0].sum(axis=1)<0].index
        print('There were removed {} of negative numbers in the data'.format(len(bad_index)))
        df.drop(bad_index, inplace=True)
        df.reset_index(drop=True, inplace=True)
        print('-------------------------------------------------------------------------------------------------')
        

        return df
    
    # @staticmethod
    # def format_sorting(df):
    #     label = 'Date'
    #     print('--------------------Coverting invalid data into null values--------------------------------------')
    #     df = df[~df.index.isna()]
    #     df.replace(['No Data','no data','Comm Fail','Error','Arc Off-line','Not Connect','I/O Timeout','Scan Timeout','AccessDenied','Pt Created'], np.nan, inplace= True)
    #     try:
    #             df.rename(columns = {(df == '-').any().idxmax():'Date'}, inplace = True)
    #     except:
    #             print('Error with the column name "Date". Kindly check the column name')
    #             # os._exit(0) 
                
    #     print("Invalid data handled successfully")
    #     print('-------------------------------------------------------------------------------------------------')
    #     print('--------------------Checking the date column and uniformity--------------------------------------')
        
    #     "check the date index in the dataframe"
    #     date_indeces = df.filter(like='-', axis=1).index
    #     print("Does the loaded file has Date as index?: {}".format(len(date_indeces) > 0))
    #     if len(date_indeces) > 0 : print("Is this uniform throughout the dataframe?: {}".format(len(date_indeces) == df.shape[0]))
    #     if len(date_indeces) > 0 : df.reset_index(inplace = True)
    #     if 'index' in df.columns : df.drop('index', inplace=True, axis=1)
            
    #     print('-------------------------------------------------------------------------------------------------') 
    #     print('------------------------Formatting datatype for each column--------------------------------------')
        
    #     if 'Date' not in df.columns:
    #         print('Date column is not detected. Kindly provide Date data')
    #         os._exit(0) 

    #     if 'Date' and 'Time' in df.columns:
    #         df['Date'] = df['Date'] + " " +df['Time']
    #     else:
    #         pass
    #     """Takes in the dataframe, converts necessary datatypes, and sorts the values by date. Returns the dataframe"""
    #     df['Date'] = pd.to_datetime(df['Date'])
    
    #     #conversion of numeric values
    #     cols = df.columns.drop(['Date'])
    #     df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

    #     print("Date and Columns are converted into their respective data type")
    #     print('-------------------------------------------------------------------------------------------------')

    #     #sorting
    #     df.sort_values(label , inplace=True)
    #     try:
    #         df = df.drop(columns=['Time'])
    #     except:
    #         pass
        
    #     print("All values are sorted according to " + label + ' column')
    #     print('-------------------------------------------------------------------------------------------------')
    #     print('------------------------Removing zero and negative numbers---------------------------------------')
    #     df = df[(df != 0).all(axis=1)]
    #     bad_index = df[df[df._get_numeric_data()<0].sum(axis=1)<0].index
    #     print('There were removed {} of negative numbers in the data'.format(len(bad_index)))
    #     df.drop(bad_index, inplace=True)
    #     df.reset_index(drop=True, inplace=True)
    #     print('-------------------------------------------------------------------------------------------------')
        
    #     return df

    @staticmethod
    def reduce_memory(df):
        """ iterate through all the columns of a dataframe and modify the data type
            to reduce memory usage.        
        """  
        subtype_int = ['uint8','uint16','uint32','uint64','int8','int16','int32','int64']
        subtype_float = ['float16','float32','float64']
        for col in df.columns:
            col_type = str(df[col].dtypes)
            mx_col = df[col].max()
            mn_col = df[col].min()
            if 'int'in col_type:
                for ele in subtype_int:
                    if mn_col>np.iinfo(ele).min and mx_col<np.iinfo(ele).max:
                        df[col] = df[col].astype(ele)
                        break
            
            elif 'float' in col_type:
                for ele in subtype_float:
                    if mn_col>np.finfo(ele).min and mx_col<np.finfo(ele).max:
                        df[col] = df[col].astype(ele)
                        break  
            elif 'object' in col_type:
                    if col=='date':
                        df['date'] = pd.to_datetime(df['date'],format='%Y-%m-%d')
                    else:
                        numbr_of_unique = len(df[col].unique())
                        numbr_total = len(df[col])
                        if numbr_of_unique/numbr_total<0.5:
                            
                            df[col] = df[col].astype('category')

        return df

    @staticmethod
    def num_gaugedetect(df):
        "Automatically detect each column for number of gauge and type of gauge reading"
        column = df.columns

        tp_reading, ap_reading, tt_reading, at_reading = [],[],[],[]
        gauge_type = [] #to find multiple gauge reading type
        for i in column:
            if re.search(r"TP", i): tp_reading.append(i) 
            if re.search(r"AP", i): ap_reading.append(i)
            if re.search(r"TT", i): tt_reading.append(i)
            if re.search(r"AT", i): at_reading.append(i)

        if tp_reading : gauge_type.append(tp_reading) 
        if ap_reading : gauge_type.append(ap_reading) 
        if tt_reading : gauge_type.append(tt_reading) 
        if at_reading : gauge_type.append(at_reading) 

        num_gauge = max(gauge_type, key=len)
        print("There are {} type of reading detected with {} number of gauge! ".format(len(gauge_type),len(num_gauge)))
        
        return num_gauge,gauge_type,tp_reading, ap_reading, tt_reading

