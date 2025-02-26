import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

class Material:
    def __init__(self, name, dir_name, tube_name, tube_directory, mass_density, diameter):
        self.name = name
        self.dir_name = dir_name
        self.tube_name = tube_name
        self.tube_dir = tube_directory
        self.data_was_set = False
        self.rho = mass_density
        self.r = diameter / 2 # radius (micro meters)
        self.set_data()
        self.Qs_calculated = False

    @property
    def M(self):
        return self._M
    
    @M.setter
    def M(self, value):
        self._M = value

    def set_data(self):
        file_names = [os.path.join(self.tube_dir, self.dir_name, file_name) for file_name in os.listdir(os.path.join(self.tube_dir, self.dir_name)) if file_name.endswith('.CSV')]
        column_names = ['TIME', 'CH1', 'CH2']
        runs = []
        for i, run in enumerate(file_names):
            runs.append(pd.read_csv(file_names[i], engine='python', usecols=column_names, on_bad_lines='skip'))
        self.num_runs = len(runs)
        self.times = [run['TIME'] for run in runs]
        self.CH1 = [run['CH1'] for run in runs]
        self.data_was_set = True

    @property
    def Qs(self):
        integration_results = []
        R = 5*10**6 # in Ohms
        for idx, (time_val, ch1_series) in enumerate(zip(self.times, self.CH1)):
            integral =  - np.trapezoid(ch1_series, time_val) / R # in Coulombs
            integration_results.append(integral) # in Coulombs
        self.Qs_calculated = True
        self.Q_values = np.array(integration_results) 
        return np.array(integration_results) # in Coulombs
    
    # Q is the charge in Coulombs
    # rho is the mass density in g/cm^3
    # r is the radius in micro meters
    # M is the injected mass in grams
    # returns the charge per surface area in e/microm^2
    def charge_per_surface_area(self):
        e_charge = 1.602176634e-19 # elementary charge in Coulombs
        if self.Qs_calculated == False:
            self.Qs
        rho = self.rho * (10000)**-3 # convert to g/microm^3
        sigma = (self.Q_values * rho * self.r) / (3 * self.M)
        return sigma / e_charge # in e/microm^2
    
    def plot_ch1(self, ax, specific_run = None):
        ax.grid()
        ax.set_title(f'Oscilloscope data of {self.name} in {self.tube_name}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage (V)')
        if self.data_was_set:
            if specific_run is None:
                for idx, (time_val, ch1_series) in enumerate(zip(self.times, self.CH1)):
                    ax.plot(time_val, ch1_series, label=f'{self.tube_name} Run {idx+1}')
            else:
                ax.plot(self.times[specific_run], self.CH1[specific_run], label=f'{self.tube_name} Run {specific_run+1}')
            ax.legend()

#measure_scooped_mass: mass of the material scooped in the measure cup (g)
#num_measure_scoops: number of scoops of material in the measure cup (unitless)
#h_measure: height of the measure cup (cm)
#d_measure: diameter of the measure cup (cm)
#h: height of the actual cup (must be in the same units as h_measure)
#d: diameter of the actual cup (must be in the same units as d_measure)
#returns the mass of the material in the actual cup (g)
def mass_calc(measure_scooped_mass, num_measure_scoops=3, h_measure=0.9, d_measure=1.2, h=0.1, d=0.2, get_rho=False):
    V_measure = np.pi * (d_measure / 2) ** 2 * h_measure #(cm^3)
    rho = measure_scooped_mass / (num_measure_scoops * V_measure) #(g/cm^3)
    V = np.pi * (d / 2) ** 2 * h #(cm^3)
    M = rho * V
    if get_rho:
        return M, rho
    return M #(g)