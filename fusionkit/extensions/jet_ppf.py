'''
The JET_PPF class is meant to extract and store data from the JET PPF system, for now limited
to reading text files exported through 'inspect values' option in JETDSP. Also includes as reader function for
ProfileMaker data exports.
'''

import pandas as pd
import numpy as np
from scipy import interpolate
import os

## JET_PPF
class JET_PPF:
    def __init__(self):
        self.data = {}
    
    def read_file(f_loc, data_only=True, verbose=False):
        # Ingest all the lines in the file
        f = open(f_loc, 'r')
        lines = f.readlines()
        f.close()

        # Check the dimensionality of the PPF data
        if [value for value in lines[1].split()][-1] == 'Scalar':
            if verbose:
                print('PPF scalar data detected')
            for i, line in enumerate(lines):
                if line.strip() and [value for value in line.split()][0] == 'Value:':
                    value = float(''.join(ch for ch in [value for value in line.split()][-1] if ch in '1234567890.'))
                    #print(value)
                    return value
        elif [value for value in lines[1].split()][-1] == '2D':
            if verbose:
                print('PPF vector data detected')
            data = pd.read_csv(f_loc, skiprows=5, sep='\\s{1,}', names=['x','data'], engine='python')
            if data_only:
                return np.array(data['data'].astype('float64')) # outputs numpy array with just the values of the quantity
            else:
                return data.astype('float64') # outputs pandas dataframe with the radial/time and quantity data combined

    def read_sertoli(data_loc, fname, t_slices=None, type=None, value_only=False, interp=True, header=0):
        times = list(map(str,np.linspace(1,t_slices,t_slices)))
        # read and time average the ppf data file
        data = pd.read_csv(data_loc+fname, skiprows=header, sep='\\s{2,}', names=['rho_pol']+times, engine='python')
        data_avg = data.loc[:, times[0]:times[-1]].mean(axis=1)
        #print(data)
        if interp:
            # import rho toroidal and time average
            rho_tor = pd.read_csv(data_loc+'dataRHOT.dat', skiprows=header, sep='\\s{2,}', names=['rho_pol']+times, engine='python').loc[:, times[0]:times[-1]].mean(axis=1)
            #print(rho_tor)
            # high resolution rho toroidal basis
            rho_tor_int = np.linspace(0.,1.,161)
            #print(rho_tor_int)
            # interpolate the time averaged data onto this basis
            data_int = interpolate.interp1d(rho_tor,data_avg,kind='quadratic',fill_value='extrapolate')(rho_tor_int)
            #print(data_int)
            # return either the interpolated data or a dataframe containing
            if value_only:
                return data_int
            elif not value_only:
                # put it into a dataframe
                df = pd.concat([pd.Series(rho_tor_int),pd.Series(data_int)],axis=1,keys=['rho_tor',type])
                return df.astype('float64')
        else:
            if value_only:
                return np.array(data_avg.astype('float64'))
            elif not value_only:
                return data.astype('float64')
    
    def read_profilemaker(data_loc=None, shot=None, time_start=None,time_end=None, return_yest=False):
        quantities = []
        raw_data = {}
        fit_data = {}

        # Loop over all files in the data location
        for fname in sorted(os.listdir(data_loc)):
            # Filter for the files 
            if shot in fname and '_Data' in fname:
                ftime = ((fname.split('-')[-1]).split('.csv')[0]).split('s')[0]
                if float(ftime) <= time_end and float(ftime)>= time_start:
                    #print(fname)
                    # Generate the list with quantities for which there is profile maker data in data_loc
                    quantity = fname.split('_Data')[0]
                    if quantity == 'T_PL':
                        quantity = 'T_I'
                    if quantity not in quantities:
                        #print(quantity)
                        quantities.append(quantity)
                        raw_data[quantity] = {}
                        fit_data[quantity] = []
                    print('\t '+data_loc+fname)
                    df = pd.read_csv(data_loc+fname,header=1)
                    for source in list(sorted(set(df['diagnostic']))):
                        if source not in raw_data[quantity]:
                            #print(source)
                            raw_data[quantity][source] = []
                        raw_data[quantity][source].append(df[df['diagnostic']==source].sort_values('x').reset_index(drop=True))
                    fit_data[quantity].append(df[['x','y_estimated']])
        #print("\t Added custom data for: "+quantity)
        if return_yest:
            return raw_data, sorted(quantities), fit_data
        else:
            return raw_data, sorted(quantities)
