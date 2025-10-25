import win32com.client
import sys
import time
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import pythoncom
import math
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import random


class OpenServer():
    "Class for holding ActiveX reference. Allows license disconnectio"

    def __init__(self):
        self.status = "Disconnected"
        self.OSReference = None

    def Connect(self):
        self.OSReference = win32com.client.Dispatch("PX32.OpenServer.1", pythoncom.CoInitialize())
        self.status = "Connected"
        print("OpenServer connected")

    def Disconnect(self):
        self.OSReference = None
        self.status = "Disconnected"
        print("OpenServer disconnected")


class GasLiftOptimization:

    def __init__(self):
        self.status = "Disconnected"
        self.OSReference = None

    def Connect(self):
        self.OSReference = win32com.client.Dispatch("PX32.OpenServer.1", pythoncom.CoInitialize())
        self.status = "Connected"
        print("OpenServer connected")

    def Disconnect(self):
        self.OSReference = None
        self.status = "Disconnected"
        print("OpenServer disconnected")

    def GetAppName(self, sv):
        # function for returning app name from tag string
        pos = sv.find(".")
        if pos < 2:
            sys.exit("GetAppName: Badly formed tag string")
        app_name = sv[:pos]
        if app_name.lower() not in ["prosper", "mbal", "gap", "pvt", "resolve",
                                    "reveal"]:
            sys.exit("GetAppName: Unrecognised application name in tag string")
        return app_name

    def DoCmd(self, OpenServe, cmd):
        # perform a command and check for errors
        lerr = OpenServe.OSReference.DoCommand(cmd)
        if lerr > 0:
            err = OpenServe.OSReference.GetErrorDescription(lerr)
            OpenServe.Disconnect()
            sys.exit("DoCmd: " + err)

    def DoSet(self, OpenServe, sv, val):
        # set a value and check for errors
        lerr = OpenServe.OSReference.SetValue(sv, val)
        app_name = self.GetAppName(sv)
        lerr = OpenServe.OSReference.GetLastError(app_name)
        if lerr > 0:
            err = OpenServe.OSReference.GetErrorDescription(lerr)
            OpenServe.Disconnect()
            sys.exit("DoSet: " + err)

    def DoGet(self, OpenServe, gv):
        # get a value and check for errors
        get_value = OpenServe.OSReference.GetValue(gv)
        app_name = self.GetAppName(gv)
        lerr = OpenServe.OSReference.GetLastError(app_name)
        if lerr > 0:
            err = OpenServe.OSReference.GetErrorDescription(lerr)
            OpenServe.Disconnect()
            sys.exit("DoGet: " + err)
        return get_value

    def DoSlowCmd(self, OpenServe, cmd):
        # perform a command then wait for command to exit and check for errors
        step = 0.001
        app_name = self.GetAppName(cmd)
        lerr = OpenServe.OSReference.DoCommandAsync(cmd)
        if lerr > 0:
            err = OpenServe.OSReference.GetErrorDescription(lerr)
            OpenServe.Disconnect()
            sys.exit("DoSlowCmd: " + err)
        while OpenServe.OSReference.IsBusy(app_name) > 0:
            if step < 2:
                step = step * 2
            time.sleep(step)
        lerr = OpenServe.OSReference.GetLastError(app_name)
        if lerr > 0:
            err = OpenServe.OSReference.GetErrorDescription(lerr)
            OpenServe.Disconnect()
            sys.exit("DoSlowCmd: " + err)

    def DoGAPFunc(self, OpenServe, gv):
        self.DoSlowCmd(gv)
        DoGAPFunc = self.DoGet(OpenServe, "GAP.LASTCMDRET")
        lerr = OpenServe.OSReference.GetLastError("GAP")
        if lerr > 0:
            err = OpenServe.OSReference.GetErrorDescription(lerr)
            OpenServe.Disconnect()
            sys.exit("DoGAPFunc: " + err)
        return DoGAPFunc

    def OSOpenFile(self, OpenServe, theModel, appname):
        self.DoSlowCmd(OpenServe, appname + '.OPENFILE ("' + theModel + '")')
        lerr = OpenServe.OSReference.GetLastError(appname)
        if lerr > 0:
            err = OpenServe.OSReference.GetErrorDescription(lerr)
            OpenServe.Disconnect()
            sys.exit("OSOpenFile: " + err)

    def OSSaveFile(self, OpenServe, theModel, appname):
        self.DoSlowCmd(OpenServe, appname + '.SAVEFILE ("' + theModel + '")')
        lerr = OpenServe.OSReference.GetLastError(appname)
        if lerr > 0:
            err = OpenServe.OSReference.GetErrorDescription(lerr)
            OpenServe.Disconnect()
            sys.exit("OSSaveFile: " + err)

    def check_dir_and_retrieve_values(self, gas_available, increment):
        # self.number_of_wells = int(input('Please enter the number of wells:'))
        self.number_of_wells = 10
        self.gas_available = gas_available
        self.increment = increment
        self.gas_injection_array = np.arange(0, gas_available + increment, increment)
        self.df = pd.DataFrame()
        self.df['gas_injection_array'] = self.gas_injection_array
        print(self.gas_available)



    def simulation(self):
        # Initialises an 'OpenServer' class

        petex = OpenServer()

        # Creates ActiveX reference and holds a license

        petex.Connect()
        oil_rates_for_wells = {}

        # Perform functions

        # cwd = os.getcwd()
        cwd = r'C:\Users\aliyu\PycharmProjects\openserver_with_python'
        for j in range(1, self.number_of_wells + 1):
            # opening well file
            self.OSOpenFile(petex, cwd + f'\models\well_{j}.Out', 'PROSPER')
            print(f'Well {j} opened')
            # oil rates calculation
            for i in range(0, len(self.gas_injection_array)):
                command = f'PROSPER.ANL.SYS.Sens.SensDB.Sens[138].Vals[{i}]'
                self.DoSet(petex, command, self.gas_injection_array[i])
            self.DoCmd(petex, 'PROSPER.ANL.SYS.CALC')
            oil_rates = []  # list of oil rates for 1 well
            for i in range(0, len(self.gas_injection_array)):
                value = f'PROSPER.OUT.SYS.Results[{i}].Sol.OilRate'
                oil_rates.append(np.round(float(self.DoGet(petex, value)), 2))
            oil_rates_for_wells[f'well_{j}'] = oil_rates
            # closing file
            self.OSSaveFile(petex, cwd + f'\well_{j}.Out', 'PROSPER')
            print(f'Well {j} closed')

        # merging to main dataframe
        for i in range(1, self.number_of_wells + 1):
            self.df[f'Well_{i}'] = oil_rates_for_wells[f'well_{i}']
        return self.df


    def wells(self):
        self.df_new = self.df.drop('gas_injection_array', axis=1)
        self.wells = []
        i = 1
        for element in self.df_new.idxmax():
            oil_rate = self.df_new[f'Well_{i}'][element]
            gas_lift = self.df['gas_injection_array'][element]
            self.wells.append({'production_rate': oil_rate, 'gas_lift': gas_lift})
            i += 1
        return self.wells

    def generate_well_list(self):
        self.well_list = [(d['production_rate'], d['gas_lift']) for d in self.wells]
        return self.well_list

    def optimize_gas_allocation_PSO(self, wells, gas_capacity, iteration_count, num_particles):
        # Initialize the particles and their velocities
        particles = [np.random.rand(len(wells)) for _ in range(num_particles)]
        velocities = [np.zeros(len(wells)) for _ in range(num_particles)]

        # Initialize the best position and best fitness for each particle
        personal_best_positions = particles.copy()
        personal_best_fitness = [np.inf for _ in range(num_particles)]

        # Initialize the global best position and global best fitness
        global_best_position = personal_best_positions[np.argmin(personal_best_fitness)]
        global_best_fitness = np.inf

        # Run the algorithm for the specified number of iterations
        for i in range(iteration_count):
            # Update the position and fitness of each particle
            for j, particle in enumerate(particles):
                # Update the velocity of the particle
                rp = random.uniform(0, 1)
                rg = random.uniform(0, 1)
                velocities[j] = 0.5 * velocities[j] + rp * (personal_best_positions[j] - particle) + rg * (
                            global_best_position - particle)
                # Update the position of the particle
                particles[j] += velocities[j]
                # Ensure that the particle's position is within the constraints
                particles[j] = np.maximum(np.zeros(len(wells)), particles[j])

                # Calculate the fitness of the particle
                fitness = -1 * np.sum(particles[j] * [well['production_rate'] for well in wells])
                # Update the personal best position and personal best fitness of the particle
                if fitness < personal_best_fitness[j]:
                    personal_best_positions[j] = particles[j]
                    personal_best_fitness[j] = fitness

                # Check if the total gas allocated to all wells is less than or equal to the total available gas capacity
                total_gas_allocated = sum(particles[j])
                if total_gas_allocated > gas_capacity:
                    particles[j] = particles[j] * (gas_capacity / total_gas_allocated)
                    total_gas_allocated = gas_capacity

                # Update the global best position and global best fitness
                if fitness < global_best_fitness:
                    global_best_position = particles[j]
                    global_best_fitness = fitness
        return [{'well_id': j, 'allocation': global_best_position[j]} for j in range(len(wells))]

    def testing(self):
        gas_capacity = 1.5
        iteration_count = 50
        num_part = 240
        self.allocations = self.optimize_gas_allocation_PSO(self.wells, gas_capacity, iteration_count, num_part)
        # test the accuracy of method-Matt
        petex = OpenServer()

        petex.Connect()
        self.oil_rates = []

        # Perform functions
        cwd = r'C:\Users\aliyu\PycharmProjects\openserver_with_python'

        for j in range(len(self.allocations)):
            # opening well file
            self.OSOpenFile(petex, cwd + f'\models\well_{j + 1}.Out', 'PROSPER')
            print(f'Well {j + 1} opened')

            self.DoSet(petex, 'PROSPER.ANL.SYS.Sens.SensDB.Clear', 0)
            self.DoSet(petex, "PROSPER.ANL.SYS.Sens.SensDB.Vars[0]", 22)

            self.DoSet(petex, "PROSPER.ANL.SYS.Sens.SensDB.Sens[138].Vals[" + str(0) + "]", self.allocations[j]['allocation'])

            self.DoCmd(petex, 'PROSPER.ANL.SYS.CALC')

            self.oil_rates.append((np.round(float(self.DoGet(petex, 'PROSPER.OUT.SYS.Results[0].Sol.OilRate')), 2),
                              self.allocations[j]['allocation']))

        petex.Disconnect()

        return self.oil_rates

    def visualize(self):
        cols = 2
        rows = math.ceil(len(self.allocations) / cols)
        subplot_titles = ['Well ' + str(i) for i in range(1, 11)]
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)


        for i in range(self.number_of_wells):
            k = i // 2 + 1
            j = i % 2 + 1

            x = self.df['gas_injection_array']
            y = self.df[f'Well_{i + 1}']
            label = f'Well_{i + 1}'

            fig.add_trace(
                go.Scatter(x=x, y=y, name=subplot_titles[i]),
                row=k, col=j)
            # axs[k, j].axvline(self.oil_rates[i][1], c='r', dashes=(5, 2, 1, 2))
            fig.add_vline(x=self.oil_rates[i][1], line_width=1, line_dash="dash", line_color="red", row=k, col=j)
            fig.update_layout(height=600, width=800)

        return fig