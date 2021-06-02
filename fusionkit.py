# fusionkit framework
# created by gsnoep at 10 May 2021
# fusionkit is a toolkit of processing tools for fusion experimental and simulation data

# IMPORT
import sys, os
from datetime import datetime
import re, copy, pathlib, json, codecs
import numpy as np
import pandas as pd
from scipy import interpolate,integrate

# Utilities
# general numerical or pythonics utilities
def number(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

def find(val, arr,n=1):
    if isinstance(arr,list):
        arr_ = np.array(arr)
    else:
        arr_ = arr
    if n == 1:
        return np.argsort(np.abs(arr_-val))[0]
    else:
        return list(np.argsort(np.abs(arr_-val)))[:n]

# CORE CLASSES
# core fusionkit framework classes
## DATASPINE
class DataSpine:
    def __init__(self,dataspine=None):
        if dataspine is None:
            self.dataspine = {}

    def create(self,author=None):
        self.author = author
        self.add_metadata()
        return self

    def add_metadata(self,projectname=None):
        if 'metadata' not in self.dataspine:
            self.dataspine['metadata'] = {}
        today = datetime.now()
        created = "{}/{}/{} @ {}:{}".format(today.day,today.month,today.year,today.strftime("%H"),today.strftime("%M"))
        self.dataspine['metadata'].update({'created':created})
        if self.author is not None:
            self.dataspine['metadata'].update({'author':self.author})
        else:
            self.dataspine['metadata'].update({'author':'Unknown'})

        return self

## PLASMA
class Plasma:
    def __init__(self):
        self.metadata = {}
        self.species = {}
        self.num_species = len(self.species)
        self.equilibrium = Equilibrium()
        self.diagnostics = {}
        self.dataset = {}
    
    def construct_plasma(self,tokamak=None,shot=None,imp_rescale=False,imp_composite=False,imp_sertoli=False,miller=False,gyro=False):
        ## Data location variables
        ppf_loc = "../Data/"+tokamak+"_"+shot+"/PPF/"
        gpr_loc = "../Data/"+tokamak+"_"+shot+"/GPR/"
        sertoli_loc = "../Data/"+tokamak+"_"+shot+"/PPF/Sertoli/"

        dataset = self.dataset

        ## GPR data import (EFTP equilibrium)
        # data import
        gpr_raw = EX2GK().read_file(data_loc=gpr_loc, fname=shot+"_eftp_raw.txt", quantities=['NE','TE','TI','TIMP'])
        gpr_fit = EX2GK().read_file(data_loc=gpr_loc, fname=shot+"_eftp_fit.txt", quantities=['NE','TE','TI1','TIMP'])
        gpr_proc = EX2GK().read_file(data_loc=gpr_loc, fname=shot+"_eftp_qlk.txt", quantities=['ALPHATOT','ANE','ATE','ATI1','BETATOT'])

        # quantity definition
        dataset['ne_raw'] = gpr_raw['NE']
        dataset['Te_raw'] = gpr_raw['TE']
        dataset['Ti_raw'] = gpr_raw['TI']

        dataset['rho'] = np.array(gpr_fit['NE']['x'])
        dataset['ne'] = np.array(gpr_fit['NE']['y'])
        #print('<ne>:'+str(sum(dataset['ne'])/len(dataset['ne'])))
        dataset['Te'] = np.array(gpr_fit['TE']['y'])
        dataset['Ti'] = np.array(gpr_fit['TI1']['y'])

        dataset['ne_sigma'] = np.array(gpr_fit['NE']['y_sigma'])
        dataset['Te_sigma'] = np.array(gpr_fit['TE']['y_sigma'])
        dataset['Ti_sigma'] = np.array(gpr_fit['TI1']['y_sigma'])

        #ANE = np.array(gpr_proc['ANE']['y'])
        #ATE = np.array(gpr_proc['ATE']['y'])
        #ATI = np.array(gpr_proc['ATI1']['y'])

        dataset['RLne_sigma'] = np.array(gpr_proc['ANE']['y_sigma'])
        dataset['RLTe_sigma'] = np.array(gpr_proc['ATE']['y_sigma'])
        dataset['RLTi_sigma'] = np.array(gpr_proc['ATI1']['y_sigma'])

        ## EFTP data import
        #rho_eftp = JET_PPF.read_sertoli(ppf_loc, 'dataQ_EFTP.dat', ms_list, interp=False, value_only=False, header=2)['rho_pol']
        #q_eftp = interpolate.interp1d(rho_eftp,JET_PPF.read_sertoli(ppf_loc, 'dataQ_EFTP.dat', ms_list, interp=False, value_only=True, header=2),kind='quadratic',fill_value='extrapolate')(rho)
        #sh_eftp = interpolate.interp1d(rho_eftp,JET_PPF.read_sertoli(ppf_loc, 'dataSH_EFTP.dat', ms_list, interp=False, value_only=True, header=2),kind='quadratic',fill_value='extrapolate')(rho)

        ## PPF data import (curdiff equilibrium)
        rho_esco = JET_PPF.read_file(ppf_loc+"dataXRHO.dat")
        #rtor = JET_PPF.read_file(ppf_loc+"dataRHO.dat")
        Rlfs = JET_PPF.read_file(ppf_loc+"dataR.dat")
        Rhfs = JET_PPF.read_file(ppf_loc+"dataRI.dat")

        dataset['q'] = interpolate.interp1d(rho_esco,JET_PPF.read_file(ppf_loc+"dataQ.dat"),kind='quadratic',fill_value='extrapolate')(dataset['rho'])
        #sh = JET_PPF.read_file(ppf_loc+"dataSH.dat",data_only=False)
        if miller:
            if self.equilibrium and self.equilibrium.fluxsurfaces:
                dataset['q'] = interpolate.interp1d(self.equilibrium.derived['rho_tor'],self.equilibrium.raw['qpsi'],bounds_error=False)(dataset['rho'])
                dataset['B0'] = interpolate.interp1d(self.equilibrium.derived['rho_tor'],self.equilibrium.derived['Bref_miller'],bounds_error=False)(dataset['rho'])
                dataset['kappa'] = interpolate.interp1d(self.equilibrium.derived['rho_tor'],self.equilibrium.derived['kappa'],bounds_error=False)(dataset['rho'])
                dataset['delta'] = interpolate.interp1d(self.equilibrium.derived['rho_tor'],self.equilibrium.derived['delta'],bounds_error=False)(dataset['rho'])
                dataset['zeta'] = interpolate.interp1d(self.equilibrium.derived['rho_tor'],self.equilibrium.derived['zeta'],bounds_error=False)(dataset['rho'])
                dataset['s_kappa'] = interpolate.interp1d(self.equilibrium.derived['rho_tor'],self.equilibrium.derived['s_kappa'],bounds_error=False)(dataset['rho'])
                dataset['s_delta'] = interpolate.interp1d(self.equilibrium.derived['rho_tor'],self.equilibrium.derived['s_delta'],bounds_error=False)(dataset['rho'])
                dataset['s_zeta'] = interpolate.interp1d(self.equilibrium.derived['rho_tor'],self.equilibrium.derived['s_zeta'],bounds_error=False)(dataset['rho'])
                dataset['dRodr'] = interpolate.interp1d(self.equilibrium.derived['rho_tor'],self.equilibrium.derived['dRodr'],bounds_error=False)(dataset['rho'])
                dataset['dZodr'] = interpolate.interp1d(self.equilibrium.derived['rho_tor'],self.equilibrium.derived['dZodr'],bounds_error=False)(dataset['rho'])
                dataset['zeta'] = interpolate.interp1d(self.equilibrium.derived['rho_tor'],self.equilibrium.derived['zeta'],bounds_error=False)(dataset['rho'])
        elif gyro:
            if self.equilibrium and self.equilibrium.fluxsurfaces:
                dataset['B0'] = interpolate.interp1d(self.equilibrium.derived['rho_tor'],self.equilibrium.derived['B_unit'])(dataset['rho'])
        else:
            #dataset['B0'] = JET_PPF.read_file(ppf_loc+"dataBTOR.dat")
            dataset['B0'] = self.equilibrium.derived['Bref_eqdsk']

        #rho_pm = JET_PPF.read_sertoli(ppf_loc, 'dataRHO_PM.dat', t_slices=5, interp=False, value_only=True, header=2)
        #rho_pm_filter = [not bool for bool in np.isinf(rho_pm)]
        #ne_pm = interpolate.interp1d(rho_pm[rho_pm_filter],JET_PPF.read_sertoli(ppf_loc, 'dataNE_PM.dat', t_slices=5, interp=False, value_only=True, header=2)[rho_pm_filter],kind='quadratic',fill_value='extrapolate')(rho)
        #Te_pm = interpolate.interp1d(rho_pm[rho_pm_filter],JET_PPF.read_sertoli(ppf_loc, 'dataTE_PM.dat', t_slices=5, interp=False, value_only=True, header=2)[rho_pm_filter],kind='quadratic',fill_value='extrapolate')(rho)
        #Ti_pm = interpolate.interp1d(rho_pm[rho_pm_filter],JET_PPF.read_sertoli(ppf_loc, 'dataTI_PM.dat', t_slices=5, interp=False, value_only=True, header=2)[rho_pm_filter],kind='quadratic',fill_value='extrapolate')(rho)

        ## Sertoli data import (EFTP equilibrium)
        LZ_scale = 1
        MZ_scale = 1
        HZ_scale = 1
        if imp_rescale:
            if shot == '83157':
                LZ_scale = 4.28777E+00
                MZ_scale = 3.00391E-01
                HZ_scale = 4.29612E-01
            elif shot == '83160':
                LZ_scale = 5.92230E+00
                MZ_scale = 6.92019E-02
                HZ_scale = 3.77182E-01
                '''
                LZ_scale = 4.86144E+00
                MZ_scale = 1.39226E-01
                HZ_scale = 3.42878E-01
                '''
        elif not imp_rescale:
            if shot == '83157':
                sertoli_loc = "../Data/"+tokamak+"_94123/PPF/Sertoli/"
            elif shot == '83160':
                sertoli_loc = "../Data/"+tokamak+"_94119/PPF/Sertoli/"

        rho_sertoli = JET_PPF.read_sertoli(sertoli_loc, 'dataRHOT.dat', t_slices=5, value_only=True, header=2)
        dataset['n_LZ'] = LZ_scale*interpolate.interp1d(rho_sertoli,JET_PPF.read_sertoli(sertoli_loc, 'dataLZAV.dat', t_slices=5, value_only=True, header=2),kind='quadratic',fill_value='extrapolate')(dataset['rho'])
        n_MZ = MZ_scale*interpolate.interp1d(rho_sertoli,JET_PPF.read_sertoli(sertoli_loc, 'dataOZAV.dat', t_slices=5, value_only=True, header=2),kind='quadratic',fill_value='extrapolate')(dataset['rho'])
        n_HZ = HZ_scale*interpolate.interp1d(rho_sertoli,JET_PPF.read_sertoli(sertoli_loc, 'dataHZAV.dat', t_slices=5, value_only=True, header=2),kind='quadratic',fill_value='extrapolate')(dataset['rho'])
        Z_M = interpolate.interp1d(rho_sertoli,JET_PPF.read_sertoli(sertoli_loc, 'dataZ_M.dat', t_slices=5, value_only=True, header=2),kind='quadratic',fill_value='extrapolate')(dataset['rho'])
        Z_H = interpolate.interp1d(rho_sertoli,JET_PPF.read_sertoli(sertoli_loc, 'dataZ_H.dat', t_slices=5, value_only=True, header=2),kind='quadratic',fill_value='extrapolate')(dataset['rho'])
        #Z_EFF = interpolate.interp1d(rho_sertoli,JET_PPF.read_sertoli(sertoli_loc, 'dataZEFF.dat', t_slices=5, value_only=True, header=2),kind='quadratic',fill_value='extrapolate')(dataset['rho'])

        ## Physical constants
        e = 1.602176E-19                        # electron charge
        mu0 = 4*np.pi*1E-7                      # vacuum magnetic permeability

        A_M = 58.6934                           # atomic mass medium mass impurity (Nickel) 
        A_H = 183.84                            # atomic mass high mass impurity (Tungsten)
        dataset['Z_L'] = 4

        ## Computed quantities
        # Geometry
        r = (Rlfs-Rhfs)/2                       # midplane-averaged minor plasma radius
        a = r[-1]                               # midplane-averaged minor lcfs radius
        x = r/a                                 # qualikiz normalised radial coordinate
        Ro = (Rlfs+Rhfs)/2                      # midplane-averaged flux surface major radius
        R0 = Ro[-1]                             # midplane-averaged lcfs major radius
        trpeps = r/Ro                           # normalised GENE radial coordinate

        dataset['r'] = interpolate.interp1d(rho_esco,r,kind='quadratic',fill_value='extrapolate')(dataset['rho'])
        dataset['a'] = a
        dataset['x'] = interpolate.interp1d(rho_esco,x,kind='quadratic',fill_value='extrapolate')(dataset['rho'])
        dataset['Ro'] = interpolate.interp1d(rho_esco,Ro,kind='quadratic',fill_value='extrapolate')(dataset['rho'])
        dataset['R0'] = R0
        dataset['trpeps'] = interpolate.interp1d(rho_esco,trpeps,kind='quadratic',fill_value='extrapolate')(dataset['rho'])

        # Physics quantities
        dataset['s'] = interpolate.interp1d(rho_esco,r*np.gradient(np.log(JET_PPF.read_file(ppf_loc+"dataQ.dat")),r,edge_order=2),fill_value='extrapolate')(dataset['rho'])
        #dataset['sh'] = interpolate.interp1d(rho_esco,rtor*np.gradient(np.log(JET_PPF.read_file(ppf_loc+"dataQ.dat")),rtor,edge_order=2),kind='quadratic',fill_value='extrapolate')(dataset['rho'])                                                                               # midplane-averaged magnetic shear
        
        dataset['Z_comp'] = (np.round(((n_MZ*Z_M**2)+(n_HZ*Z_H**2))/(n_MZ*Z_M+n_HZ*Z_H))).astype(int)                                       # atomic number composite impurity species
        dataset['n_comp'] = (n_MZ*Z_M+n_HZ*Z_H)/dataset['Z_comp']                                                                           # density composite impurity species
        dataset['A_comp'] = (n_MZ*Z_M/(dataset['n_comp']*dataset['Z_comp']))*A_M + (n_HZ*Z_H/(dataset['n_comp']*dataset['Z_comp']))*A_H     # atomic mass composite impurity species

        if imp_composite:
            dataset['ni'] = dataset['ne']-(dataset['n_LZ']*dataset['Z_L']+dataset['n_comp']*dataset['Z_comp'])
            dataset['n_comp'] = (dataset['ne']-dataset['ni']-dataset['n_LZ']*dataset['Z_L'])/dataset['Z_comp']
            ZEFF = (dataset['ni']+(dataset['n_LZ']*dataset['Z_L']**2)+(dataset['n_comp']*dataset['Z_comp']**2))/dataset['ne']
            #print("quasi-neutrality check: "+str(dataset['ne']-dataset['ni']-(dataset['n_LZ']*dataset['Z_L']+dataset['n_comp']*dataset['Z_comp'])))
        elif imp_sertoli:
            dataset['n_MZ'] = n_MZ
            dataset['n_HZ'] = n_HZ
            dataset['Z_M'] = Z_M
            dataset['Z_H'] = Z_H
            #print('<n_LZ>:'+str(sum(dataset['n_LZ'])/len(dataset['n_LZ'])))
            #print('<n_MZ>:'+str(sum(dataset['n_MZ'])/len(dataset['n_MZ'])))
            #print('<n_HZ>:'+str(sum(dataset['n_HZ'])/len(dataset['n_HZ'])))
            dataset['ni'] = dataset['ne']-(dataset['n_LZ']*dataset['Z_L']+dataset['n_MZ']*dataset['Z_M']+dataset['n_HZ']*dataset['Z_H'])
            dataset['ZEFF'] = (dataset['ni']+(dataset['n_LZ']*dataset['Z_L']**2)+(dataset['n_MZ']*dataset['Z_M']**2)+(dataset['n_HZ']*dataset['Z_H']**2))/dataset['ne']
        else:
            dataset['ni'] = dataset['ne']-(dataset['n_LZ']*dataset['Z_L'])
            dataset['n_LZ'] = (dataset['ne']-dataset['ni'])/dataset['Z_L']
            ZEFF = (dataset['ni']+(dataset['n_LZ']*dataset['Z_L']**2))/dataset['ne']
            #print("quasi-neutrality check: "+str(dataset['ne']-dataset['ni']-dataset['n_LZ']*dataset['Z_L']))

        pe = dataset['ne']*e*dataset['Te']
        pi = dataset['ni']*e*dataset['Ti']
        pLZ = dataset['n_LZ']*e*dataset['Ti']
        if imp_composite:
            pcomp = dataset['n_comp']*e*dataset['Ti']

        beta_e = 2*pe*mu0/(dataset['B0']**2)
        beta_i = 2*pi*mu0/(dataset['B0']**2)
        beta_LZ = 2*pLZ*mu0/(dataset['B0']**2)
        dataset['beta'] = beta_e+beta_i+beta_LZ
        dataset['beta_e'] = beta_e

        if imp_composite:
            beta_comp = 2*pcomp*mu0/(dataset['B0']**2)
            dataset['beta'] = beta_e+beta_i+beta_LZ+beta_comp

        # Normalised logarithmic gradients
        dataset['RLTe'] = -(R0/dataset['Te'])*np.gradient(dataset['Te'],dataset['r'])
        dataset['RLTi'] = -(R0/dataset['Ti'])*np.gradient(dataset['Ti'],dataset['r'])
        dataset['RLne'] = -(R0/dataset['ne'])*np.gradient(dataset['ne'],dataset['r'])
        dataset['RLni'] = -(R0/dataset['ni'])*np.gradient(dataset['ni'],dataset['r'])
        dataset['RLn_LZ'] = -(R0/dataset['n_LZ'])*np.gradient(dataset['n_LZ'],dataset['r'])

        # Normalised logarithmic gradients
        dataset['aLTe'] = -(a/dataset['Te'])*np.gradient(dataset['Te'],dataset['r'])
        dataset['aLTi'] = -(a/dataset['Ti'])*np.gradient(dataset['Ti'],dataset['r'])
        dataset['aLne'] = -(a/dataset['ne'])*np.gradient(dataset['ne'],dataset['r'])
        dataset['aLni'] = -(a/dataset['ni'])*np.gradient(dataset['ni'],dataset['r'])
        dataset['aLn_LZ'] = -(a/dataset['n_LZ'])*np.gradient(dataset['n_LZ'],dataset['r'])

        if imp_composite:
            dataset['RLn_comp'] = ((dataset['ne']*dataset['RLne'])-(dataset['ni']*dataset['RLni'])-(dataset['n_LZ']*dataset['Z_L']*dataset['RLn_LZ']))/(dataset['n_comp']*dataset['Z_comp'])
            dataset['aLn_comp'] = ((dataset['ne']*dataset['aLne'])-(dataset['ni']*dataset['aLni'])-(dataset['n_LZ']*dataset['Z_L']*dataset['aLn_LZ']))/(dataset['n_comp']*dataset['Z_comp'])
            #print("quasi-neutrality gradient check: "+str(dataset['ne']*dataset['RLne']-(dataset['ni']*dataset['RLni'])-(dataset['n_LZ']*dataset['Z_L']*dataset['RLn_LZ'])-(dataset['n_comp']*dataset['Z_comp']*dataset['RLn_comp'])))
        else:
            dataset['RLn_LZ'] = ((dataset['ne']*dataset['RLne'])-(dataset['ni']*dataset['RLni']))/(dataset['n_LZ']*dataset['Z_L'])
            dataset['aLn_LZ'] = ((dataset['ne']*dataset['aLne'])-(dataset['ni']*dataset['aLni']))/(dataset['n_LZ']*dataset['Z_L'])
            #print("quasi-neutrality gradient check: "+str(dataset['ne']*dataset['RLne']-(dataset['ni']*dataset['RLni'])-(dataset['n_LZ']*dataset['Z_L']*dataset['RLn_LZ'])))
        
        if imp_composite:
            dataset['alpha'] = dataset['q']**2*(beta_e*(dataset['RLne']+dataset['RLTe'])+beta_i*(dataset['RLni']+dataset['RLTi'])+beta_LZ*(dataset['RLn_LZ']+dataset['RLTi'])+beta_comp*(dataset['RLn_comp']+dataset['RLTi']))
        else:
            dataset['alpha'] = dataset['q']**2*(beta_e*(dataset['RLne']+dataset['RLTe'])+beta_i*(dataset['RLni']+dataset['RLTi'])+beta_LZ*(dataset['RLn_LZ']+dataset['RLTi']))

        return dataset

## EQUILIBRIUM
class Equilibrium:
    '''
    Class to handle any and all data related to the magnetic equilibrium in a magnetic confinement fusion device
    '''
    def __init__(self):
        self.raw = {} # storage for all raw eqdsk data
        self.derived = {} # storage for all data derived from eqdsk data
        self.fluxsurfaces = {} # storage for all data related to flux surfaces

    ## I/O functions
    def read_geqdsk(self,f_path=None,just_raw=False,add_derived=False):
        '''
        Function to convert an eqdsk g-file from file to Equilibrium() object

        :param f_path: string containing the path to the eqdsk g-file, including the file name (!)

        :param just_raw: boolean to return only the raw dictionary (True) or [default] return the Equilibrium() object (False)

        :param add_derived: boolean to directly add derived quantities (e.g. phi, rho_tor) to the Equilibrium() object upon reading the g-file

        :return: self or dict if just_raw
        '''

        print('Reading eqdsk file to Equilibrium...')

        # check if eqdsk file path is provided and if it exists
        if f_path is None or not os.path.isfile(f_path):
            print('Invalid file or path provided!')
            return

        # specify the eqdsk file formate, based on 'G EQDSK FORMAT - L Lao 2/7/97'
        self.eqdsk_format = {
            0:{'vars':['code','case','idum','nw','nh'],'size':[5]},
            1:{'vars':['Rdim', 'Zdim', 'Rcentr', 'Rmin', 'Zmid'],'size':[5]},
            2:{'vars':['Rmag', 'Zmag', 'psimag', 'psisep', 'Bcentr'],'size':[5]},
            3:{'vars':['current', 'psimag2', 'xdum', 'Rmag2', 'xdum'],'size':[5]},
            4:{'vars':['Zmag2', 'xdum', 'psisep2', 'xdum', 'xdum'],'size':[5]},
            5:{'vars':['fpol'],'size':['nw']},
            6:{'vars':['pressure'],'size':['nw']},
            7:{'vars':['ffprime'],'size':['nw']},
            8:{'vars':['pprime'],'size':['nw']},
            9:{'vars':['psiRZ'],'size':['nw','nh']},
            10:{'vars':['qpsi'],'size':['nw']},
            11:{'vars':['nbbbs','limitr'],'size':[2]},
            12:{'vars':['Rbbbs','Zbbbs'],'size':['nbbbs']},
            13:{'vars':['Rlim','Zlim'],'size':['limitr']},
        }

        # specify the sanity values used for consistency check of eqdsk file
        self.sanity_values = ['Rmag','Zmag','psimag','psisep']
        self.max_values = 5 # maximum number of values per line
        
        # read the g-file
        with open(f_path,'r') as file:
            lines = file.readlines()
        
        # convert the line strings in the values list to lists of numerical values, while retaining potential character strings at the start of the file
        for i,line in enumerate(lines):
            # split the line string into separate values by ' ' as delimiter, adding a space before a minus sign if it is the delimiter
            values = list(filter(None,re.sub(r'(?<!E)-',' -',line).rstrip('\n').split(' ')))
            #print('values: '+str(values))
            # select all the numerical values in the list of sub-strings of the current line, but keep them as strings so the fortran formatting remains
            numbers = [j for i in [number for number in (re.findall(r'^(?![A-Z]).*-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?', value) for value in values)] for j in i]
            #print('numbers: '+str(numbers))
            # select all the remaining sub-strings and store them in a separate list
            strings = [value for value in values if value not in numbers]
            # if there is a list of strings, this means it is the first line of the eqdsk file, so split it in the code and case strings
            if len(strings) > 0:
                strings = [strings[0],' '.join([string for string in strings[1:]])]
            #print('strings: '+str(strings))
            # convert the list of numerical sub-strings to their actual int or float value
            numbers = [number(value) for value in numbers]
            #print('numbers: '+str(numbers))
            lines[i] = strings+numbers
            #print('line after: '+str(lines[i]))

        # start at the top of the file
        current_row = 0
        # go through the eqdsk format line by line and collect all the values for the vars in each format line
        for key in self.eqdsk_format:
            # check if the var size is a string refering to a value to be read from the eqdsk file and backfill it, for loop for multidimensional vars
            for i,size in enumerate(self.eqdsk_format[key]['size']):
                if isinstance(size,str):
                    self.eqdsk_format[key]['size'][i] = self.raw[size]

            # compute the row the current eqdsk format line ends
            if len(self.eqdsk_format[key]['vars']) != np.prod(self.eqdsk_format[key]['size']):
                end_row = current_row + int(np.ceil(len(self.eqdsk_format[key]['vars'])*np.prod(self.eqdsk_format[key]['size'])/self.max_values))
            else:
                end_row = current_row + int(np.ceil(np.prod(self.eqdsk_format[key]['size'])/self.max_values))

            # check if there are values to be collected
            if end_row > current_row:
                # collect all the values between current_row and end_row in the eqdsk file and flatten the resulting list of lists to a list
                values = [j for i in lines[current_row:end_row] for j in i]
                # handle the exception of len(eqdsk_format[key]['vars']) > 1 and the data being stored in value pairs 
                if len(self.eqdsk_format[key]['vars']) > 1 and len(self.eqdsk_format[key]['vars']) != self.eqdsk_format[key]['size'][0]:
                    # make a shadow copy of values
                    values_ = copy.deepcopy(values)
                    # empty the values list
                    values = []
                    # collect all the values belonging to the n-th variable in the format list and remove them from the shadow value list until empty
                    for j in range(len(self.eqdsk_format[key]['vars']),0,-1):
                        values.append(np.array(values_[0::j]))
                        values_ = [value for value in values_ if value not in values[-1]]
                # store and reshape the values in a np.array() in case eqdsk_format[key]['size'] > max_values
                elif self.eqdsk_format[key]['size'][0] > self.max_values:
                    values = [np.array(values).reshape(self.eqdsk_format[key]['size'])]
                # store the var value pairs in the eqdsk dict
                self.raw.update({var:values[k] for k,var in enumerate(self.eqdsk_format[key]['vars'])})
            # update the current position in the 
            current_row = end_row

        # sanity check the eqdsk values
        for key in self.sanity_values:
            # find the matching sanity key in eqdsk
            sanity_pair = [keypair for keypair in self.raw.keys() if keypair.startswith(key)][1]
            #print(sanity_pair)
            if self.raw[key]!=self.raw[sanity_pair]:
                exit('Inconsistent '+key+': %7.4g, %7.4g'%(self.raw[key], self.raw[sanity_pair])+'. CHECK YOUR EQDSK FILE!')
        if add_derived:
            self.add_derived()
        if just_raw:
            return self.raw
        else:
            return self
    
    def read_json(self,f_path=None):
        '''
        Function to read an Equilibrium object stored on disk in json into a callable Equilibrium object

        :param f_path: string path to the location the desired file, including the desired file name (!)

        :return: Equilibrium object containing the data from the json
        '''
        with open(f_path,'r') as file:
            equilibrium_json = json.load(file)
        
        if 'raw' in equilibrium_json:
            self.raw = equilibrium_json['raw']
            # convert lists back to ndarrays
            for key in self.raw:
                if isinstance(self.raw[key],list):
                    self.raw[key] = np.array(self.raw[key])
        if 'derived' in equilibrium_json:
            self.derived = equilibrium_json['derived']
            for key in self.derived:
                if isinstance(self.derived[key],list):
                    if all(isinstance(element,list) for element in self.derived[key]):
                        for fs in range(0,len(self.derived[key])):
                            self.derived[key][fs] = np.array(self.derived[key][fs])
                    else:
                        self.derived[key] = np.array(self.derived[key])
        if 'fluxsurfaces' in equilibrium_json:
            self.fluxsurfaces = equilibrium_json['fluxsurfaces']
            for key in self.fluxsurfaces:
                if isinstance(self.fluxsurfaces[key],list):
                    if all(isinstance(element,list) for element in self.fluxsurfaces[key]):
                        for fs in range(0,len(self.fluxsurfaces[key])):
                            self.fluxsurfaces[key][fs] = np.array(self.fluxsurfaces[key][fs])
                    else:
                        self.fluxsurfaces[key] = np.array(self.fluxsurfaces[key])

        if 'metadata' in equilibrium_json:
            self.metadata = equilibrium_json['metadata']

        return self

    def write_json(self,f_path='./Equilibrium',metadata=None):
        '''
        Function to write the Equilibrium object to a json file on disk

        :param f_path: string path to the location the desired file, including the desired file name (!) [default] the current folder '.' with the file called 'Equilibrium'  (optional) 

        :param metadata: dict contain relevant metadata for the Equilibrium file 

        :return: 
        '''

        if metadata is not None and 'author' in metadata:
            author = metadata['author']
        else:
            author = 'fusionkit.Equilibrium class'

        # create a dict to store all the equilibrium data in to write to json
        equilbrium = DataSpine().create(author=author)
        equilbrium.add_metadata()
        equilbrium = equilbrium.dataspine

        if metadata:
            for key in metadata:
                equilbrium['metadata'].update({key:metadata[key]})

        if self.raw:
            equilbrium['raw'] = copy.deepcopy(self.raw)
            for key in equilbrium['raw']:
                if isinstance(equilbrium['raw'][key],np.ndarray):
                    equilbrium['raw'][key] = equilbrium['raw'][key].tolist()
        if self.derived:
            equilbrium['derived'] = copy.deepcopy(self.derived)
            for key in equilbrium['derived']:
                if isinstance(equilbrium['derived'][key],np.ndarray):
                    equilbrium['derived'][key] = equilbrium['derived'][key].tolist()
                elif isinstance(equilbrium['derived'][key],list):
                    for fs in range(0,len(equilbrium['derived'][key])):
                        if isinstance(equilbrium['derived'][key][fs],np.ndarray):
                            equilbrium['derived'][key][fs] = equilbrium['derived'][key][fs].tolist()
        if self.fluxsurfaces:
            equilbrium['fluxsurfaces'] = copy.deepcopy(self.fluxsurfaces)
            for key in equilbrium['fluxsurfaces']:
                if isinstance(equilbrium['fluxsurfaces'][key],np.ndarray):
                    equilbrium['fluxsurfaces'][key] = equilbrium['fluxsurfaces'][key].tolist()
                else:
                    for fs in range(0,len(equilbrium['fluxsurfaces'][key])):
                        if isinstance(equilbrium['fluxsurfaces'][key][fs],np.ndarray):
                            equilbrium['fluxsurfaces'][key][fs] = equilbrium['fluxsurfaces'][key][fs].tolist()


        json.dump(equilbrium, codecs.open(f_path+'.json', 'w', encoding='utf-8'), separators=(',', ':'), indent=4)

        print('Generated fusionkit.Equilibrium file at: '+f_path+'.json')

        return

    ## physics functions
    def add_derived(self,f_path=None,just_derived=False,incl_fluxsurfaces=False,incl_miller_geo=False):
        '''
        Function to add quantities derived from the raw Equilibrium.read_geqdsk output, such as phi, rho_pol, rho_tor to the Equilibrium()

        Can also be called standalone if f_path is defined

        :param f_path: string containing the path to the eqdsk g-file, including the file name (!)

        :param just_derived: boolean to return only the derived quantities dictionary (True) or [default] return the Equilibrium() object (False)

        :param incl_fluxsurfaces: boolean to also return 

        :param incl_miller_geo: boolean to include the symmetrised flux surface Miller shaping parameters delta, kappa and zeta (True) or [default] not (False)

        :return: self or dict if just_derived
        '''

        print('Adding derived quantities to Equilibrium...')

        if self.raw == {}:
            try:
                self.raw= self.read_eqdsk(f_path=f_path,just_data=True)
            except:
                print('Unable to read provided EQDSK file, check file and/or path')

        # introduce shorthands for data and derived locations for increased readability
        raw = self.raw
        derived = self.derived
        fluxsurfaces = self.fluxsurfaces

        # compute R and Z grid vectors
        derived['R'] = np.array([raw['Rmin'] + i*(raw['Rdim']/(raw['nw']-1)) for i in range(raw['nw'])])
        derived['Z'] = np.array([raw['Zmid'] - 0.5*raw['Zdim'] + i*(raw['Zdim']/(raw['nh']-1)) for i in range(raw['nh'])])

        # find the indexes of 'zmag' on the high field side (hfs) and low field side (lfs) of the separatrix
        i_Zmag_hfs, i_Zmag_lfs = sorted(find(raw['Zmag'],raw['Zbbbs'],n=2))
        
        # find the index of 'zmag' in the R,Z grid
        i_Zmag = find(raw['Zmag'],derived['Z'])

        # find indexes of separatrix on HFS, magnetic axis, separatrix on LFS in R
        i_R_hfs, i_Rmag, i_R_lfs = sorted([find(raw['Rbbbs'][i_Zmag_hfs],derived['R']),find(raw['Rmag'],derived['R']),find(raw['Rbbbs'][i_Zmag_lfs],derived['R'])])

        # HFS and LFS R and psirz
        R_hfs = derived['R'][i_R_hfs:i_Rmag]
        R_lfs = derived['R'][i_Rmag:i_R_lfs]
        psiRZmag_hfs = raw['psiRZ'][i_Zmag,i_R_hfs:i_Rmag]
        psiRZmag_lfs = raw['psiRZ'][i_Zmag,i_Rmag:i_R_lfs]

        # equidistant psi grid
        derived['psi'] = np.linspace(raw['psimag'],raw['psisep'],raw['nw'])

        # corresponding rho_pol grid
        psi_norm = (derived['psi'] - raw['psimag'])/(raw['psisep'] - raw['psimag'])
        derived['rho_pol'] = np.sqrt(psi_norm)

        # nonlinear R grid at 'Zmag' based on equidistant psi grid for 'fpol', 'pres', 'ffprime', 'pprime' and 'qpsi'
        derived['R_psi_hfs'] = interpolate.interp1d(psiRZmag_hfs,R_hfs,fill_value='extrapolate')(derived['psi'][::-1])
        derived['R_psi_lfs'] = interpolate.interp1d(psiRZmag_lfs,R_lfs,fill_value='extrapolate')(derived['psi'])

        # compute LFS phi (toroidal flux in W/rad) grid from integrating q = d psi/d phi
        derived['phi'] = integrate.cumtrapz(raw['qpsi'],derived['psi'],initial=0)

        # construct the corresponding rho_tor grid
        if derived['phi'][-1] !=0:
            phi_norm = derived['phi']/derived['phi'][-1] # as phi[0] = 0 this term is dropped
        else:
            phi_norm = np.ones_like(derived['phi'])*np.NaN
            print('Could not construct valid rho_tor')
        derived['rho_tor']  = np.sqrt(phi_norm)

        # compute the rho_pol and rho_tor grids corresponding to the R,Z grid
        psiRZ_norm = abs(raw['psiRZ'] - raw['psimag'])/(raw['psisep'] - raw['psimag'])
        derived['rhoRZ_pol'] = np.sqrt(psiRZ_norm)

        derived['phiRZ'] = interpolate.interp1d(derived['psi'],derived['phi'],bounds_error=False)(raw['psiRZ'])
        # repair nan values in phiRZ, first find the indexes of the nan values
        ij_nan = np.argwhere(np.isnan(derived['phiRZ']))
        for _nan in ij_nan:
            i_nan = _nan[0]
            j_nan = _nan[1]
            if j_nan !=0:
                j_nan_min = j_nan-1
            else:
                j_nan_min = j_nan+1
            if j_nan != raw['nw']-1:
                j_nan_plus = j_nan+1
            else:
                j_nan_plus = j_nan-1
            # cycle through the nan values and compute a weighted sum of the last and earliest non-nan values
            derived['phiRZ'][i_nan,j_nan] = 0.5*(derived['phiRZ'][i_nan,j_nan_min]+derived['phiRZ'][i_nan,j_nan_plus]) 

        phiRZ_norm = abs(derived['phiRZ'])/(derived['phi'][-1])
        derived['rhoRZ_tor'] = np.sqrt(phiRZ_norm)

        # find the R,Z values of the x-point, !TODO: should add check for second x-point in case of double-null equilibrium
        i_xpoint_Z = find(np.min(raw['Zbbbs']),raw['Zbbbs']) # assuming lower null, JET-ILW shape for now
        derived['R_x'] = raw['Rbbbs'][i_xpoint_Z]
        derived['Z_x'] = raw['Zbbbs'][i_xpoint_Z]

        # compute the toroidal magnetic field and current density
        derived['B_tor'] = raw['ffprime']/derived['R']
        derived['j_tor'] = derived['R']*raw['pprime']+derived['B_tor']

        if incl_fluxsurfaces:
            self.add_fluxsurfaces(raw=raw,derived=derived,fluxsurfaces=fluxsurfaces,incl_miller_geo=incl_miller_geo)
              
        if just_derived:
            return self.raw['derived']
        else:
            return self

    def add_fluxsurfaces(self,raw=None,derived=None,fluxsurfaces=None,incl_miller_geo=False):
        '''
        Function to add fluxsurfaces to an Equilibrium()
        
        :param raw: dict containing the raw Equilibrium data, [default] if None is set to self.raw

        :param derived: dict containing the derived Equilibrium quantities, [default] if None is set to self.derived

        :param fluxsurfaces: dict to store the Equilibrium flux surface data, [default] if None is set to self.fluxsurfaces

        :param incl_miller_geo: boolean to include the symmetrised flux surface Miller shaping parameters delta, kappa and zeta (True) or [default] not (False)

        :return: self
        '''
        print('Adding fluxsurfaces to Equilibrium...')

        # check if self.fluxsurfaces contains all the fluxsurfaces specified by derived['rho_tor'] already
        if self.fluxsurfaces and self.derived and len(self.fluxsurfaces['rho_tor']) == len(self.derived['rho_tor']):
            # skip
            print('Skipped adding fluxsurfaces to Equilibrium as it already contains fluxsurfaces')
        else:
            # set the default locations if None is specified
            if raw is None:
                raw = self.raw
            if derived is None:
                derived = self.derived
            if fluxsurfaces is None:
                fluxsurfaces = self.fluxsurfaces
            
            # add the flux surface data for rho_tor > 0
            for rho_fs in derived['rho_tor'][1:]:
                sys.stdout.write('\r {}% completed'.format(round(100*(find(rho_fs,derived['rho_tor'][1:])+1)/len(derived['rho_tor'][1:]))))
                sys.stdout.flush()
                # check that rho stays inside the lcfs
                if rho_fs < 0.999:
                    self.fluxsurface_find(x_fs=rho_fs,psiRZ=raw['psiRZ'],R=derived['R'],Z=derived['Z'],incl_miller_geo=incl_miller_geo,return_self=True)
            sys.stdout.write('\n')

            # find the geometric center, minor radius and extrema of the lcfs manually
            lcfs = self.fluxsurface_center(psi_fs=raw['psisep'],R_fs=raw['Rbbbs'],Z_fs=raw['Zbbbs'],psiRZ=raw['psiRZ'],R=derived['R'],Z=derived['Z'],incl_extrema=True)
            lcfs.update({'R':raw['Rbbbs'],'Z':raw['Zbbbs']})
            if incl_miller_geo:
                lcfs = self.fluxsurface_miller_geo(fs=lcfs)
            
            # add a zero at the start of all fluxsurface quantities and append the lcfs values to the end of the flux surface data
            for key in fluxsurfaces:
                fluxsurfaces[key].insert(0,0.*fluxsurfaces[key][-1])

            for key in lcfs:
                if key in fluxsurfaces:
                    fluxsurfaces[key].append(lcfs[key])
            fluxsurfaces['rho_tor'].append(1.)
            fluxsurfaces['psi'].append(derived['psi'][-1])

            # add the midplane average geometric flux surface quantities to derived
            derived['Ro'] = np.array(fluxsurfaces['R0'])
            derived['Ro'][0] = derived['Ro'][1] # clear the starting zero
            derived['R0'] = derived['Ro'][-1] # midplane average major radius of the lcfs
            derived['Zo'] = np.array(fluxsurfaces['Z0'])
            derived['Z0'] = derived['Zo'][-1] # average elevation of the lcfs
            derived['r'] = np.array(fluxsurfaces['r'])
            derived['a'] = derived['r'][-1] # midplane average minor radius of the lcfs
            derived['epsilon'] = derived['r']/derived['Ro']

            # add the midplane average major radius and elevation derivatives to derived
            derived['dRodr'] = np.gradient(derived['Ro'],derived['r'])
            derived['dZodr'] = np.gradient(derived['Zo'],derived['r'])

            # add the magnetic shear to derived
            derived['s'] = derived['r']*np.gradient(np.log(raw['qpsi']),derived['r'],edge_order=2)

            # add several magnetic field quantities to derived
            derived['Bref_eqdsk'] = raw['fpol'][0]/raw['Rmag']
            derived['Bref_miller'] = raw['fpol']/derived['Ro']
            #derived['B_unit'] = interpolate.interp1d(derived['r'],(1/derived['r'])*np.gradient(derived['phi'],derived['r'],edge_order=2))(derived['r'])
            derived['B_unit'] = interpolate.interp1d(derived['r'],(raw['qpsi']/derived['r'])*np.gradient(derived['psi'],derived['r']))(derived['r'])
            
            if incl_miller_geo:
                # add the symmetrised flux surface trace arrays to derived
                derived['R_sym'] = fluxsurfaces['R_sym']
                derived['Z_sym'] = fluxsurfaces['Z_sym']

                # add the Miller shaping parameters to derived
                derived['kappa'] = np.array(fluxsurfaces['kappa'])
                derived['delta'] = np.array(fluxsurfaces['delta'])
                derived['zeta'] = np.array(fluxsurfaces['zeta'])

                # compute the shear of the Miller shaping parameters
                derived['s_kappa'] = derived['r']*np.gradient(np.log(derived['kappa']),derived['r'],edge_order=2)
                derived['s_delta'] = (derived['r']/np.sqrt(1-derived['delta']**2))*np.gradient(derived['delta'],derived['r'],edge_order=2)
                derived['s_delta_ga'] = derived['r']*np.gradient(derived['delta'],derived['r'],edge_order=2)
                derived['s_zeta'] = derived['r']*np.gradient(derived['zeta'],derived['r'],edge_order=2)
            
            return self
    
    def fluxsurface_find(self,psi_fs=None,psi=None,x_fs=None,x=None,x_label='rho_tor',psiRZ=None,R=None,Z=None,incl_miller_geo=False,return_self=False):
        '''
        #Function to find the R,Z trace of a flux surface 

        :param psi_fs: (optional) float of the poloidal flux value of the flux surface

        :param psi: (optional) array vector containing the poloidal flux psi from axis to separatrix

        :param x_fs: float of the radial flux label of the flux surface, by default assumed to be in rho_tor

        :param x: (optional) array vector of the radial flux surface label on the same grid as psi, by default assume to be rho_tor

        :param x_label: string of the radial flux label, options (for now) are [default] 'rho_tor', 'rho_pol', 'psi' and 'r'

        :param psiRZ: array containing the R,Z map of the poloidal flux psi of the magnetic equilibrium

        :param R: array vector of R grid mesh

        :param Z: array vector of Z grid mesh

        :param incl_miller_geo: boolean to include the symmetrised flux surface Miller shaping parameters delta, kappa and zeta (True) or [default] not (False)

        :param return_self: boolean to return the result to the Equilibrium() object (True) or [default] as a standalone dictionary (False)

        :return: dict with the flux surface [default] or add the fluxsurface data to Equilibrium.fluxsurfaces

        '''

        fs = {}

        if x_fs != None:
            if psi_fs == None:
                if x_label in self.derived  and 'psi' in self.derived:
                    # find the flux of the selected flux surface
                    x = self.derived[x_label]
                    psi = self.derived['psi']
                else:
                    exit('Equilibrium.fluxsurface_find error: Did not receive enough inputs to determine psi of the flux surface, check your inputs!')
                psi_fs = interpolate.interp1d(x,psi,kind='cubic')(x_fs)
        else:
            exit('Equilibrium.fluxsurface_find error: No radial position of the flux surface was specified, check your inputs!')
        
        fs['psi'] = float(psi_fs)

        refine=None
        # refine the R,Z and psiRZ grids if the eqdsk resolution is below 512x512
        if self.raw['nw'] < 512:
            refine = int(512/self.raw['nw'])

        # refine the psi R,Z grid if refine
        if refine!=None:
            R_fine = np.linspace(R[0],R[-1],refine*len(R))
            Z_fine = np.linspace(Z[0],Z[-1],refine*len(Z))
            psiRZ = interpolate.interp2d(R,Z,psiRZ)(R_fine,Z_fine)
            R=R_fine
            Z=Z_fine

        # find the approximate magnetic axis in psiRZ
        i_Rmag = np.where(psiRZ == np.min(psiRZ))[1][0]
        Zmag = Z[np.where(psiRZ == np.min(psiRZ))[0][0]]

        # find the R values of the flux surface at 'Zmag' and the corresponding closest indexes in derived['R']
        psiRZmag = interpolate.interp2d(R,Z,psiRZ)(R,Zmag)

        R_fs_hfs = interpolate.interp1d(psiRZmag[:i_Rmag],R[:i_Rmag])(psi_fs)
        i_R_hfs = find(R_fs_hfs,R)

        R_fs_lfs = interpolate.interp1d(psiRZmag[i_Rmag:],R[i_Rmag:])(psi_fs)
        i_R_lfs = find(R_fs_lfs,R)

        # setup arrays to store R,Z coordinates of the flux surface
        R_fs = R[i_R_hfs:i_R_lfs]
        Z_fs = np.zeros((len(R_fs),2))

        # find the top and bottom Z of the flux surface by slicing psiRZ by R between R_fs_hfs and R_fs_lfs to ensure max(Z_fs) and min(Z_fs) are included
        for i_R in range(i_R_hfs,i_R_lfs):
            i = i_R - i_R_hfs

            # take a slice of psiRZ
            psiZ = np.array(psiRZ[:,i_R])

            # find the minimum of psi in the slice to split the R,Z plane
            i_psiZ_min = find(np.min(psiZ),psiZ)

            # find the maximum of psi in the split slice to ensure not interpolating to a match in psi at a too high/low Z
            i_psiZ_upper_max = find(np.max(psiZ[i_psiZ_min:]),psiZ[i_psiZ_min:])
            i_psiZ_lower_max = find(np.max(psiZ[:i_psiZ_min]),psiZ[:i_psiZ_min])

            # find the upper and lower Z corresponding to the flux surface
            Z_fs[i,0] = interpolate.interp1d(psiZ[i_psiZ_min:i_psiZ_min+i_psiZ_upper_max],Z[i_psiZ_min:i_psiZ_min+i_psiZ_upper_max],bounds_error=False)(psi_fs)
            Z_fs[i,1] = interpolate.interp1d(psiZ[i_psiZ_lower_max:i_psiZ_min],Z[i_psiZ_lower_max:i_psiZ_min],bounds_error=False)(psi_fs)
        
        # find the top and bottom of the Z gap at the inner and outer sides as a consequence of assuming min(R_fs) and max(R_fs) to be on Zmag
        i_Z_upper = find(np.max([Z_fs[np.where(~np.isnan(Z_fs[:,0]))[0][0],0],Z_fs[np.where(~np.isnan(Z_fs[:,0]))[0][-1],0]]),Z)
        i_Z_lower = find(np.min([Z_fs[np.where(~np.isnan(Z_fs[:,1]))[0][0],1],Z_fs[np.where(~np.isnan(Z_fs[:,1]))[0][-1],1]]),Z)

        # setup arrays to store R,Z coordinates of the missing slices of the flux surface
        Z_fs_ = Z[i_Z_lower:i_Z_upper]
        R_fs_ = np.zeros((len(Z_fs_),2))

        # find the inner and outer R of the flux surface by slicing psiRZ by Z between Z[i_Z_lower] and Z[i_Z_upper] to ensure min(R_fs) and max(R_fs) are included
        for i_Z in range(i_Z_lower,i_Z_upper):
            i = i_Z - i_Z_lower
            # take a slice of psiRZ
            psiR = np.array(psiRZ[i_Z,:])

            # find the minimum of psi in the slice to split the R,Z plane
            i_psiR_min = find(np.min(psiR),psiR)

            # find the inner and outer R corresponding to the flux surface
            R_fs_[i,0] = interpolate.interp1d(psiR[:i_psiR_min],R[:i_psiR_min],bounds_error=False)(psi_fs)
            R_fs_[i,1] = interpolate.interp1d(psiR[i_psiR_min:],R[i_psiR_min:],bounds_error=False)(psi_fs)

        # find the glue edges
        i_upper = sorted(find(Z[i_Z_upper],Z_fs[:,0],n=4))
        i_lower = sorted(find(Z[i_Z_lower],Z_fs[:,1],n=4))
        i_Zmag_fs_ = find(Zmag,Z_fs_)

        # merge the upper and lower halves of the flux surface coordinates with the side slices such that the trace starts and ends at the lfs mid-plane
        fs['R'] = np.hstack((R_fs_[i_Zmag_fs_:,1],R_fs[i_upper[0]:i_upper[-1]][::-1],R_fs_[:-2,0][::-1],R_fs[i_lower[0]+1:i_lower[-1]],R_fs_[:i_Zmag_fs_+1,1]))
        fs['Z'] = np.hstack((Z_fs_[i_Zmag_fs_:],Z_fs[i_upper[0]:i_upper[-1],0][::-1],Z_fs_[:-2][::-1],Z_fs[i_lower[0]+1:i_lower[-1],1],Z_fs_[:i_Zmag_fs_+1]))
        
        # find the flux surface center quantities and add them to the flux surface dict
        fs.update(self.fluxsurface_center(psi_fs=psi_fs,R_fs=fs['R'],Z_fs=fs['Z'],psiRZ=psiRZ,R=R,Z=Z,incl_extrema=True))

        if incl_miller_geo:
            fs = self.fluxsurface_miller_geo(fs=fs)

        append_values = False
        if return_self:
            if x_label in self.fluxsurfaces:
                # if there is no entry for the flux surface in self.fluxsurfaces yet append it
                if x_fs not in self.fluxsurfaces[x_label]:
                    self.fluxsurfaces[x_label].append(x_fs)
                    append_values = True
                # there already is an entry for the flux surface
                else:
                    # find the index of the existing flux surface data
                    i_x_fs = find(x_fs,self.fluxsurfaces[x_label])
            else:
                self.fluxsurfaces.update({x_label:[x_fs]})
                i_x_fs=0
            
            # cycle through all the flux surface quantities
            for key in fs.keys():
                # if there is no previous data for this quantity somehome, create a dummy array to ensure the current data is at least stored at the correct spot
                if key not in self.fluxsurfaces:
                    self.fluxsurfaces.update({key:list(np.zeros_like(self.fluxsurfaces[x_label]))})
                # insert the flux surface quantity data in self.fluxsurfaces at i_x_fs
                if append_values:
                    self.fluxsurfaces[key].append(fs[key])
                else:
                    self.fluxsurfaces[key][i_x_fs] = fs[key]

            return self
        else:
            # return the bare flux surface dict
            return fs

    def fluxsurface_center(self,psi_fs=None,R_fs=None,Z_fs=None,psiRZ=None,R=None,Z=None,incl_extrema=False,return_self=False):
        '''
        Function to find the geometric center of a flux surface trace defined by R_fs,Z_fs and psi_fs

        :param psi_fs: float of the poloidal flux value of the flux surface

        :param R_fs: array containing the horizontal coordinates of the flux surface trace

        :param Z_fs: array containing the vertical coordinates of the flux surface trace

        :param psiRZ: array containing the R,Z map of the poloidal flux psi of the magnetic equilibrium

        :param R: array vector of R grid mesh

        :param Z: array vector of Z grid mesh

        :param incl_extrema: boolean to include the extrema data in the returned dict (True) or [default] to leave it separate (False)

        :param return_self: boolean to return the result to the Equilibrium() object (True) or [default] as a standalone dictionary (False)

        :return: dict with the flux surface [default] or add the fluxsurface data to Equilibrium.fluxsurfaces
        '''

        # create temporary flux surface storage dict
        fs = {}

        # find the average elevation (midplane) of the flux surface [Candy PPCF 51 (2009) 105009]
        fs['Z0'] = integrate.trapz(R_fs*Z_fs,Z_fs)/integrate.trapz(R_fs,Z_fs)
        #print(fs['Z0'])

        # find the extrema of the flux surface in the radial direction at the average elevation
        fs_extrema = self.fluxsurface_extrema(psi_fs=psi_fs,R_fs=R_fs,Z_fs=Z_fs,Z0_fs=fs['Z0'],psiRZ=psiRZ,R=R,Z=Z)
        R_out = fs_extrema['R_out']
        R_in = fs_extrema['R_in']

        # compute the minor and major radii of the flux surface
        fs['r'] = (R_out-R_in)/2
        fs['R0'] = (R_out+R_in)/2

        if incl_extrema:
            fs.update(fs_extrema)

        if return_self:
            # check if there is already an entry for the flux surface in Equilibrium.fluxsurfaces
            if str(psi_fs) in self.fluxsurfaces:
                # append the extrema information to the flux surface entry
                self.fluxsurfaces[str(psi_fs)].update({key:fs[key] for key in fs.keys()})
            else:
                # append the flux surface to Equilibrium.fluxsurfaces
                self.fluxsurfaces.update({str(psi_fs):fs})
            return self
        else:
            # return the bare flux surface dict
            return fs

    def fluxsurface_extrema(self,psi_fs=None,R_fs=None,Z_fs=None,Z0_fs=None,psiRZ=None,R=None,Z=None,return_self=False):
        '''
        Function to find the extrema in R and Z of a flux surface trace defined by R_fs,Z_fs and psi_fs

        :param psi_fs: float of the poloidal flux value of the flux surface

        :param R_fs: array containing the horizontal coordinates of the flux surface trace

        :param Z_fs: array containing the vertical coordinates of the flux surface trace

        :param Z0_fs: float of the average elevation of the flux surface

        :param psiRZ: array containing the R,Z map of the poloidal flux psi of the magnetic equilibrium

        :param R: array vector of R grid mesh

        :param Z: array vector of Z grid mesh

        :param return_self: boolean to return the result to the Equilibrium object (True) or [default] as a standalone dict (False)

        :return: dict with the flux surface [default] or append the fluxsurface data to Equilibrium.fluxsurfaces
        '''

        # create temporary flux surface storage dict
        fs = {}

        # check if the poloidal flux value of the flux surface is provided
        if psi_fs != None:
            # check if the midplane of the flux surface is provided
            if Z0_fs != None:
                # find the flux as function of the horizontal coordinate at the midplane of the flux surface
                psiRZ0 = interpolate.interp2d(R,Z,psiRZ)(R,Z0_fs)
                #print(psiRZ0)

                # find the extrema in R of the flux surface at the midplane
                fs['R_out'] = float(interpolate.interp1d(psiRZ0[int(len(psiRZ0)/2):],R[int(len(psiRZ0)/2):],bounds_error=False)(psi_fs))
                fs['R_in'] = float(interpolate.interp1d(psiRZ0[:int(len(psiRZ0)/2)],R[:int(len(psiRZ0)/2)],bounds_error=False)(psi_fs))

                # find the extrema in Z of the flux surface
                fs['Z_top'] = np.max(Z_fs)
                fs['R_top'] = R_fs[find(fs['Z_top'],Z_fs)]
                fs['Z_bottom'] = np.min(Z_fs)
                fs['R_bottom'] = R_fs[find(fs['Z_bottom'],Z_fs)]
            else:
                exit('Equilibibrium.fluxsurface_extrema error: No average elevation provided for the target flux surface, check your inputs!')
        else:
            exit('Equilibibrium.fluxsurface_extrema error: No poloidal flux value for target flux surface was provided, check your inputs!')

        if return_self:
            # check if there is already an entry for the flux surface in Equilibrium.fluxsurfaces
            if str(psi_fs) in self.fluxsurfaces:
                # append the extrema information to the flux surface entry
                self.fluxsurfaces[str(psi_fs)].update({key:fs[key] for key in fs.keys()})
            else:
                # append the flux surface to Equilibrium.fluxsurfaces
                self.fluxsurfaces.update({str(psi_fs):fs})
            return self
        else:
            # return the bare flux surface dict
            return fs
    
    def fluxsurface_miller_geo(self,fs=None,symmetrise=True):
        '''
        Function to extract Miller geometry parameters from (symmetrised) flux surface parameterisation [Turnbull PoP 6 1113 (1999)]

        :param fs: dict of flux surface data containing R, Z, R0, Z0, r, Z_top, Z_bottom, R_out, R_in

        :param symmetrise: boolean to set whether to [default] symmetrise the provided flux surface trace (True) or not (False)

        :return: returns an updated 
        '''

        if symmetrise:
            fs['R_sym'] = (fs['R']+fs['R'][::-1])/2
            fs['Z_sym'] = (fs['Z']-fs['Z'][::-1])/2+fs['Z0']
            R_fs = fs['R_sym']
            Z_fs = fs['Z_sym']
        else:
            R_fs = fs['R']
            Z_fs = fs['Z']
    
        # find the R,Z coordinates of the top and bottom of the flux surface
        Z_bottom = np.min(Z_fs)
        R_bottom = R_fs[find(Z_bottom,Z_fs)]
        Z_top = np.max(Z_fs)
        R_top = R_fs[find(Z_top,Z_fs)]

        # compute triangularity (delta) and elongation (kappa) of flux surface
        delta_top = (fs['R0'] - R_top)/fs['r']
        delta_bottom = (fs['R0'] - R_bottom)/fs['r']
        fs['delta'] = (delta_top+delta_bottom)/2
        x = np.arcsin(fs['delta'])
        fs['kappa'] = (Z_top - Z_bottom)/(2*fs['r'])

        # generate theta grid and interpolate the flux surface trace to the Miller parameterisation
        fs['theta'] = np.linspace(0,2*np.pi,3600)
        R_miller = fs['R0'] + fs['r']*np.cos(fs['theta']+x*np.sin(fs['theta']))
        Z_miller = np.hstack((interpolate.interp1d(R_fs[:int(len(R_fs)/2)],Z_fs[:int(len(R_fs)/2)],bounds_error=False)(R_miller[:int(len(fs['theta'])/2)]),interpolate.interp1d(R_fs[int(len(R_fs)/2):],Z_fs[int(len(R_fs)/2):],bounds_error=False)(R_miller[int(len(fs['theta'])/2):])))

        # derive zeta from the Miller parametrisation
        theta_zeta = np.array([0.25*np.pi,0.75*np.pi,1.25*np.pi,1.75*np.pi])
        R_zeta = np.zeros_like(theta_zeta)
        Z_zeta = np.zeros_like(theta_zeta)
        for i,quadrant in enumerate(theta_zeta):
            R_zeta[i] = interpolate.interp1d(fs['theta'][find(quadrant-0.25*np.pi,fs['theta']):find(quadrant+0.25*np.pi,fs['theta'])],R_miller[find(quadrant-0.25*np.pi,fs['theta']):find(quadrant+0.25*np.pi,fs['theta'])])(quadrant)
            Z_zeta[i] = interpolate.interp1d(fs['theta'][find(quadrant-0.25*np.pi,fs['theta']):find(quadrant+0.25*np.pi,fs['theta'])],Z_miller[find(quadrant-0.25*np.pi,fs['theta']):find(quadrant+0.25*np.pi,fs['theta'])])(quadrant)
        
        # invert the Miller parametrisation of Z, holding off on subtracting theta/sin(2*theta)
        zeta_ = np.arcsin((Z_zeta-fs['Z0'])/(fs['kappa']*fs['r']))/np.sin(2*theta_zeta)

        # applying a periodic correction for the arcsin of the hfs quadrants
        zeta_ = np.array([1,-1,-1,1])*zeta_+np.array([0,-np.pi,-np.pi,0])

        # compute the hfs and lfs average zeta
        zeta_lfs = 0.5*(zeta_[0]+zeta_[3])-(theta_zeta[0]/np.sin(2*theta_zeta[0]))
        zeta_hfs = 0.5*(zeta_[1]+zeta_[2])-(theta_zeta[1]/np.sin(2*theta_zeta[1]))

        # compute the average zeta of the flux surface
        fs['zeta'] = 0.5*(zeta_lfs+zeta_hfs)

        fs['R_miller'] = R_miller
        fs['Z_miller'] = fs['Z0']+fs['kappa']*fs['r']*np.sin(fs['theta']+fs['zeta']*np.sin(2*fs['theta']))
        
        return fs

# EXTENSION CLASSES
# extensions with tools for external codes
## EX2GK
class EX2GK:
    def __init__(self):
        self.gpr_data = {}
        self.gpr_data['metadata'] = {}
    
    # I/O functions
    def read_file(self, data_loc=None, fname=None, quantities=None):
        gpr_data = self.gpr_data

        empty_lines = []
        table_lines = {}
        line_count = 0
        if data_loc!=None and os.path.isdir(data_loc):
            with open(data_loc+fname, 'r') as file:
                for line in file.readlines():
                    line_count += 1
                    # If the line_count = 1 note the data type
                    if line_count == 1:
                        gpr_data['metadata']['type'] = line.split()[3]
                    # Add the header contents to the metadata section of gpr_data
                    if 'Shot' in line:
                        gpr_data['metadata']['shot'] = str([int(s) for s in line.split() if s.isdigit()][0])
                    elif 'Radial' in line:
                        gpr_data['metadata']['x'] = line.split()[-1]
                    elif 'Time' in line:
                        if 'time' not in gpr_data['metadata']:
                            gpr_data['metadata']['time'] = []
                        gpr_data['metadata']['time'].append(np.round(float(line.split()[-2]),2))
                    # If the line is empty add it to the empty_lines list
                    if not line.strip():
                        empty_lines.append(line_count)
                    # If there have been any empty lines, check if the table header on the next line contains one of the requested quantities 
                    if len(empty_lines)>=1 and (line_count)-1 == empty_lines[-1]:
                        for quantity in quantities:
                            # If the quantity was not already found and the table header contains it, record the line counter
                            if quantity not in table_lines:
                                if gpr_data['metadata']['type'] == 'Processed':
                                    qstring = 'QLK_'+quantity+' Proc.'
                                else:
                                    qstring = quantity+' '+gpr_data['metadata']['type']
                                if qstring in line:
                                    table_lines[quantity] = line_count-1
                                    gpr_data[quantity] = {}
                                    #print(line)
            #print(empty_lines)
            #print(table_lines)

            for quantity in quantities:
                if quantity in table_lines:
                    line = table_lines[quantity]
                    line_index = empty_lines.index(line)
                    df = pd.read_csv(data_loc+fname, delimiter='\\s{2,}', skiprows=line, nrows=(empty_lines[line_index+1]-empty_lines[line_index])-1, engine='python')
                    if gpr_data['metadata']['type'] == 'Raw':
                        df = df.iloc[:,0:5]
                        df.columns = ['x', 'y', 'y_sigma', 'x_err', 'diagnostic']
                        #print(df)
                        for source in list(sorted(set(df['diagnostic']))):
                            if 'BC' not in source:
                                if source not in gpr_data[quantity]:
                                    #print(source)
                                    gpr_data[quantity][source] = []
                                gpr_data[quantity][source] = df[df['diagnostic']==source].sort_values('x').reset_index(drop=True)
                    elif gpr_data['metadata']['type'] == 'Fit':
                        df = df.iloc[:,0:5]
                        df.columns = ['x', 'y', 'y_sigma', 'dydx', 'dydx_err']
                        #print(df)
                        gpr_data[quantity] = df
                    elif gpr_data['metadata']['type'] == 'Processed':
                        df.columns = ['x', 'qlk_x', 'y', 'y_sigma']
                        #print(df)
                        gpr_data[quantity] = df
            #print(gpr_data)
            if 'TIMP' in gpr_data:
                gpr_data['TI'] = gpr_data.pop('TIMP')
            return gpr_data
        
        else:
            print('No valid data location was specified!')
    
    # Filter functions
    def timeavg_filter(input_data=None, quantity_filter=None, source_filter=None):
        '''
        This function returns a dataframe containing the time averaged data for all the sources specified in the source_filter list

        :param input_data: a dict of the form raw_data[quantity][source][timeslice]

        :param source_filter: a list of strings indicating which sources to include in the returned time averaged data
        '''
        filtered_data = {}
        # Check that a data structure is provided to be time averaged
        if input_data != None and isinstance(input_data,dict):
            raw_data = input_data
            # If no quantity_filter list is specified, copy all the sources listed in the data structure
            if quantity_filter == None:
                quantity_filter = list(raw_data.keys())
                #print(quantity_filter)
            # If no source_filter list is specified, copy all the sources listed in the data structure
            if source_filter == None:
                source_filter = []
                for quantity in quantity_filter:
                    source_list = list(raw_data[quantity].keys())
                    for source in source_list:
                        if source not in source_filter:
                            source_filter.append(source)
                #print(source_filter)
                
            # Assuming the data structure has the raw_data[quantity][source][timeslice]= pandas.dataFrame(columns=['x', 'y', 'y_sigma', 'x_err', 'diagnostic']) structure
            for quantity in quantity_filter:
                df_dict = {}
                #print(raw_data)
                # Sequence through all the data source by source
                for source in raw_data[quantity].keys():
                    if source in source_filter:
                        # Specify the minimum number of samples in time to use scaling by number of data points in average input variance, does not affect population variance
                        if len(raw_data[quantity][source]) < 2:
                            use_n = False
                        else:
                            use_n = True

                        source_df = pd.concat(raw_data[quantity][source]).sort_values(by=['x']).reset_index(drop=True)
                        #print(source_df.to_string())
                        source_concat = pd.DataFrame(columns=raw_data[quantity][source][0].columns)

                        x_index = 0
                        x_ref = source_df.iloc[x_index]['x']
                        while x_ref < source_df.iloc[-1]['x']:
                            temp_concat = pd.DataFrame(columns=raw_data[quantity][source][0].columns)
                            #print('first x_ref: '+str(x_ref))
                            if x_ref < 0.01:
                                threshold = 50#5
                            elif x_ref < 0.06:
                                threshold = 25#3.25
                            elif x_ref < 0.1:
                                threshold = 18#2.5
                            elif x_ref < 0.2:
                                threshold = 7#2.5
                            else:
                                if source == "KG10":
                                    if x_ref > 1.02:
                                        threshold = 0.3
                                    elif x_ref > 1.005:
                                        threshold = 0.6
                                    elif x_ref > 0.9:
                                        threshold = 0.825
                                    else:
                                        threshold =3.5#1.5
                                else:
                                    if x_ref < 0.27:
                                        threshold = 1.25
                                    else:
                                        threshold = 1.1
                            while 100*((source_df.iloc[x_index]['x']/x_ref)-1) < threshold and x_index < source_df.shape[0]-1:
                                temp_concat.loc[temp_concat.shape[0]+1] = source_df.iloc[x_index]
                                x_index+=1
                            if temp_concat.shape[0] > 1:
                                temp_y, temp_yerr, temp_ystdm = ptools.calc_eb_error(list(temp_concat['y']),list(temp_concat['y_sigma']),use_n=use_n)
                                temp_concat = temp_concat.mean()
                                temp_concat['y'] = temp_y
                                temp_concat['y_sigma'] = temp_yerr
                                temp_concat['diagnostic'] = source
                            source_concat = source_concat.append(temp_concat,ignore_index=True).reset_index(drop=True)
                            #print(source_concat)
                            x_ref = source_df.iloc[x_index]['x']

                        df_dict[source] = source_concat
                        #if source == 'KG10':
                        #    print(source_concat.to_string())
                    else:
                        print('\t\t'+source+' is not in the source filter list')

                if len(df_dict) > 0:
                    filtered_data[quantity] = pd.concat(df_dict).sort_values('x')
                    filtered_data[quantity].reset_index(drop=True,inplace=True)
                else:
                    print('\t\tNo sources were selected for '+quantity+'!')
                #print(str(quantity)+": \n"+pm_data[quantity].to_string())
                print("\tFiltered custom data for: "+quantity)
            return filtered_data
        else:
            print('No valid input data structure was provided!')

    def copyandpaste_filter(input_data=None, copy_quantity=None, paste_quantity=None, source_filter=None, radial_filter=None):
        output_data = copy.deepcopy(input_data)
        if output_data != None and isinstance(output_data,dict):
            if copy_quantity != None and isinstance(copy_quantity,str) and paste_quantity != None and isinstance(paste_quantity,str):
                # If no source_filter list is specified, copy all the sources listed in the data structure
                if source_filter == None:
                    source_filter = set(output_data[copy_quantity]['diagnostic'])
                    #print(source_filter)
                if radial_filter == None:
                    radial_filter = [0, np.max(pd.Series(output_data[copy_quantity]['x']))]
                    #print(radial_filter)
                if output_data[copy_quantity]['diagnostic'].isin(source_filter).any():
                    #print('Selected diagnostic data is available for copy')
                    if output_data[copy_quantity]['x'].between(radial_filter[0],radial_filter[1]).any():
                        #print('Data is available for copy in the selected radial range')
                        output_data[paste_quantity] = output_data[paste_quantity].append(output_data[copy_quantity][output_data[copy_quantity]['diagnostic'].isin(source_filter) & output_data[copy_quantity]['x'].between(radial_filter[0],radial_filter[1])],ignore_index=True)
                        output_data[paste_quantity].sort_values(by=['x']).reset_index(drop=True,inplace=True)
                        #print(output_data[paste_quantity])

                        return output_data
                    else:
                        print('No data is available for copy in the selected radial range')
                else:
                    print('No data is available for copy for the selected diagnostic')

            else:
                print('Check proper definition of quantities to be copied and pasted from!')
        else:
            print('No valid input data structure was provided!')

    def delete_filter(input_data=None, delete_quantity=None, source_filter=None, radial_filter=None, value_filter=None):
        output_data = copy.deepcopy(input_data)
        if output_data != None and isinstance(output_data,dict):
            if delete_quantity != None and isinstance(delete_quantity,str):
                # If no source_filter list is specified, copy all the sources listed in the data structure
                if source_filter == None:
                    print('No sources were selected to delete data for in the input data structure')
                if radial_filter == None and value_filter == None:
                    print('Both no radial range and value range were selected to delete data for in the input data structure')
                if output_data[delete_quantity]['diagnostic'].isin(source_filter).any():
                    if radial_filter is not None:
                        if output_data[delete_quantity]['x'].between(radial_filter[0],radial_filter[1]).any():
                            output_data[delete_quantity] = output_data[delete_quantity].drop(output_data[delete_quantity][output_data[delete_quantity]['diagnostic'].isin(source_filter) & output_data[delete_quantity]['x'].between(radial_filter[0],radial_filter[1])].index)
                            output_data[delete_quantity].sort_values(by=['x']).reset_index(drop=True,inplace=True)
                            return output_data

                    elif value_filter is not None:
                        if output_data[delete_quantity]['y'].between(value_filter[0],value_filter[1]).any():
                            output_data[delete_quantity] = output_data[delete_quantity].drop(output_data[delete_quantity][output_data[delete_quantity]['diagnostic'].isin(source_filter) & output_data[delete_quantity]['y'].between(value_filter[0],value_filter[1])].index)
                            output_data[delete_quantity].sort_values(by=['x']).reset_index(drop=True,inplace=True)
                            return output_data
                    else:
                        print('No data is available for deletion in the selected radial range')
                else:
                    print('No data is available for deletion for the selected diagnostics for quantity: '+delete_quantity)

            else:
                print('Check proper definition of quantities to be deleted!')
        else:
            print('No valid input data structure was provided!')

## GENE
class GENE:
    def __init__(self):
        self.metadata = {}
        self.input = {}
        self.output = {}
    
    # I/O functions
    def write_input(self,rho=None,dataset=None,gene_config=None,diagdir=None,fname=None,imp_composite=False,miller=False):
        m_e = 9.109390E-31
        m_p = 1.672623E-27

        rho_idx = np.abs(dataset['rho']-rho).argmin()

        if miller:
            ne = dataset['ne'][rho_idx]*1e-19
            RLne = dataset['aLne'][rho_idx]
            ni = dataset['ni'][rho_idx]*1e-19
            RLni = dataset['aLni'][rho_idx]
            A_i = 2
            Z_i = 1

            Te = dataset['Te'][rho_idx]*1e-3
            RLTe = dataset['aLTe'][rho_idx]
            Ti = dataset['Ti'][rho_idx]*1e-3
            RLTi = dataset['aLTi'][rho_idx]

            n_LZ = dataset['n_LZ'][rho_idx]*1e-19
            RLn_LZ = dataset['aLn_LZ'][rho_idx]
            A_LZ = 9
            Z_L = dataset['Z_L']
            B0 = dataset['B0'][rho_idx]

            if imp_composite:
                n_comp = dataset['n_comp'][rho_idx]*1e-19
                RLn_comp = dataset['aLn_comp'][rho_idx]
                A_comp = dataset['A_comp'][rho_idx]
                Z_comp = dataset['Z_comp'][rho_idx]
            
            kappa = dataset['kappa'][rho_idx]
            delta = dataset['delta'][rho_idx]
            zeta = dataset['zeta'][rho_idx]
            s_kappa = dataset['s_kappa'][rho_idx]
            s_delta = dataset['s_delta'][rho_idx]
            s_zeta = dataset['s_zeta'][rho_idx]
            dRodr = dataset['dRodr'][rho_idx]
            dZodr = dataset['dZodr'][rho_idx]
        else:
            ne = dataset['ne'][rho_idx]*1e-19
            RLne = dataset['RLne'][rho_idx]
            ni = dataset['ni'][rho_idx]*1e-19
            RLni = dataset['RLni'][rho_idx]
            A_i = 2
            Z_i = 1

            Te = dataset['Te'][rho_idx]*1e-3
            RLTe = dataset['RLTe'][rho_idx]
            Ti = dataset['Ti'][rho_idx]*1e-3
            RLTi = dataset['RLTi'][rho_idx]

            n_LZ = dataset['n_LZ'][rho_idx]*1e-19
            RLn_LZ = dataset['RLn_LZ'][rho_idx]
            A_LZ = 9
            Z_L = dataset['Z_L']

            if imp_composite:
                n_comp = dataset['n_comp'][rho_idx]*1e-19
                RLn_comp = dataset['RLn_comp'][rho_idx]
                A_comp = dataset['A_comp'][rho_idx]
                Z_comp = dataset['Z_comp'][rho_idx]
            B0 = dataset['B0']

        q = dataset['q'][rho_idx]
        s = dataset['s'][rho_idx]
        alpha = dataset['alpha'][rho_idx]
        beta = dataset['beta'][rho_idx]
        trpeps = dataset['trpeps'][rho_idx]
        a = dataset['a']
        Ro = dataset['Ro'][rho_idx]
        R0 = dataset['R0']
        

        ## Species namelist
        species_nl = {
            "nl_name" : "species",
            'electrons' : {
                "nl_name" : "species",
                "name" : "'electrons'",
                "mass" : m_e/(A_i*m_p),
                "charge" : -1,
                "temp" : 1.0,
                "dens" : 1.0,
                "omt" : RLTe,
                "omn" : RLne,
            },
            'main_ion' : {
                "nl_name" : "species",
                "name" : "'deuterium'",
                "mass" : 1.0,
                "charge" : Z_i,
                "temp" : Ti/Te,
                "dens" : ni/ne,
                "omt" : RLTi,
                "omn" : RLni,
            },
            'impurity_1' : {
                "nl_name" : "species",
                "name" : "'beryllium'",
                "mass" : A_LZ/A_i,
                "charge" : Z_L,
                "temp" : Ti/Te,
                "dens" : n_LZ/ne,
                "omt" : RLTi,
                "omn" : RLn_LZ,
            },
        }
        if imp_composite:
            species_nl['impurity_2'] = {
                "nl_name" : "species",
                "name" : "'composite'",
                "mass" : A_comp/A_i,
                "charge" : Z_comp,
                "temp" : Ti/Te,
                "dens" : n_comp/ne,
                "omt" : RLTi,
                "omn" : RLn_comp,
            }

        ## Parallelization namelist
        parallel_nl = {
            "nl_name" : "parallelization", 
        }
        if gene_config['solver_type'] == 'IV':
            parallel_nl["n_parallel_sims"] = 2
        if gene_config['solver_type'] == 'EV':
            parallel_nl["n_parallel_sims"] = 8

        ## Box namelist
        box_nl = {
            "nl_name" : "box",
            "n_spec" : len(species_nl.keys())-1,
            "nx0" : gene_config['nx0'],
            "nky0" : 1,
            "nz0" : gene_config['nz0'],
            "nv0" : gene_config['nv0'],
            "nw0" : gene_config['nw0'],
            "kymin" : gene_config['kymin'],
            "lv" : gene_config['lv'],
            "lw" : gene_config['lw'],
            "mu_grid_type" : "'gau_lag'",
            "n0_global" : -1111,
        }

        ## I/O namelist
        io_nl = {
            "nl_name" : "in_out",
            "diagdir" : diagdir,
            "read_checkpoint" : '.F.',
            "istep_nrg" : 10,
            "istep_field" : 200,
            "istep_mom" : 400,
            "istep_energy" : 500,
            "istep_vsp" : 500,
            "istep_schpt" : 0,
        }

        ## General namelist
        general_nl = {
            "nl_name" : "general", 
            "nonlinear" : '.F.',
        }
        if gene_config['solver_type'] == 'IV':
            general_nl["comp_type"] = "'IV'"
        elif gene_config['solver_type'] == 'EV':
            general_nl["comp_type"] = "'EV'"
            general_nl["which_ev"] = "'mfn'"
            general_nl["n_ev"] = 2
            general_nl["taumfn"] = 0.2
        general_nl.update(
            {"calc_dt" : '.T.',
            "simtimelim" : 10000.0,
            "timelim" : 43200,
            "collision_op" : gene_config['coll_op'],
            "coll_on_h" : '.T.',
            "coll_f_fm_on" : '.T.',
            "coll_cons_model" : "'self_adj'",
            "coll" : gene_config['coll'],
            }
        )
        if gene_config['beta']:
            general_nl.update({"beta" : beta})
        general_nl.update(
            {"bpar" : '.F.',
            "debye2" : -1,
            "hyp_z" : gene_config['hyp_z'],
            "init_cond" : "'alm'",}
        )

        ## External contribution namelist
        external_nl = {
            "nl_name" : "external_contr",
        }

        ## Geometry namelist
        if miller:
            geo_nl = {
            "nl_name" : "geometry",
            "magn_geometry" : "'miller'",
            "trpeps" : str(trpeps)+" ! rho = "+str(rho),
            "q0" : q,
            "shat" : s,
            "amhd" : alpha,
            'drR' : dRodr,
            'drZ' : dZodr,
            'kappa' : kappa,
            's_kappa' : s_kappa,
            'delta' : delta,
            's_delta' : s_delta,
            'zeta' : zeta,
            's_zeta' : s_zeta,
            'minor_r' : 1.0,
            "major_R" : Ro/a,
            "norm_flux_projection" : '.F.',
            "rhostar" : -1,
            "dpdx_term" : "'full_drift'",
            "dpdx_pm" : -1,
            }
        else:
            geo_nl = {
                "nl_name" : "geometry",
                "magn_geometry" : "'s_alpha'",
                "trpeps" : str(trpeps)+" ! rho = "+str(rho),
                "q0" : q,
                "shat" : s,
                "major_R" : 1.0,
                "amhd" : alpha,
                "norm_flux_projection" : '.F.',
                "rhostar" : -1,
                "dpdx_term" : "'full_drift'",
                "dpdx_pm" : -1,
            }

        ## Info namelist
        info_nl = {
            "nl_name" : "info",
        }

        ## Units namelist
        units_nl = {
            "nl_name" : "units",
            "Tref" : Te,
            "nref" : ne,
            "Bref" : B0,
        }
        if miller:
            units_nl.update(
                {"Lref" : a,}
            )
        else:
            units_nl.update(
                {"Lref" : R0,}
            )
        units_nl.update(
            {"mref" : A_i,
            "omegatorref" : 0}
        )

        ## Complete GENE namelist
        gene_nl = {
            'meta' : {
                "path" : "./",
                "file" : fname,
            },
            'parallel' : parallel_nl,
            'box' : box_nl,
            'io' : io_nl,
            'general' : general_nl,
            'external' : external_nl,
            'geo' : geo_nl,
            'species' : species_nl,
            'info' : info_nl,
            'units' : units_nl
        }

        ## Print GENE namelist parameters.dat file
        for i in gene_nl.keys():
            if i == 'meta':
                pathlib.Path(gene_nl['meta']["path"]).mkdir(parents=True, exist_ok=True)
                f = open(gene_nl['meta']["path"]+gene_nl['meta']["file"],"w+")
            elif len(gene_nl[i].keys()) > 1:
                if gene_nl[i]['nl_name'] != "species":
                    #print("&{}".format(gene_nl[i]['nl_name']))
                    f.write("&{}\n".format(gene_nl[i]['nl_name']))
                    gene_nl[i].pop('nl_name')
                    for key,value in gene_nl[i].items():
                        #print("{} = {}".format(key,value))
                        f.write("{} = {}\n".format(key,value))
                    #print("/\n")
                    f.write("/\n\n")
                else:
                    gene_nl[i].pop('nl_name')
                    for key,value in gene_nl[i].items():
                        #print("&{}".format(gene_nl[i][key]['nl_name']))
                        f.write("&{}\n".format(gene_nl[i][key]['nl_name']))
                        gene_nl[i][key].pop('nl_name')
                        for k,v in gene_nl[i][key].items():
                            #print("{} = {}".format(k,v))
                            f.write("{} = {}\n".format(k,v))
                        #print("/\n")
                        f.write("/\n\n")
        f.close()
        print('Generated GENE input file at: '+gene_nl['meta']["path"]+gene_nl['meta']["file"])

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

## TGLF
class TGLF:
    def __init__(self):
        self.metadata = {}
        self.input = {}
        self.output = {}
    
    # I/O functions
    def write_inputs(fname=None,path=None,control=None,species=None,gaussian=None,geometry=None,expert=None):
        # default values for TGLF input namelists
        header_params = {
            'name':'input.tglf',
            'message':'See https://gafusion.github.io/doc/tglf/tglf_table.html'
        }
        control_params = {
            'name':'Control paramters',
            'UNITS':'GYRO',
            'NS':len(species),
            'USE_TRANSPORT_MODEL':True,
        }
        if geometry['name']=='s-alpha':
            control_params.update({'GEOMETRY_FLAG':0})
        elif geometry['name']=='miller' or geometry == None:
            control_params.update({'GEOMETRY_FLAG':1})
        elif geometry['name']=='fourier':
            control_params.update({'GEOMETRY_FLAG':2})
        elif geometry['name']=='elite':
            control_params.update({'GEOMETRY_FLAG':3}) 
        control_params.update(
            {
                'USE_BPER':False,
                'USE_BPAR':False,
                'USE_MHD_RULE':False,
                'USE_BISECTION':True,
                'USE_INBOARD_DETRAPPED':False,
                'SAT_RULE':2,
                'KYGRID_MODEL':0,
                'XNU_MODEL':2,
                'VPAR_MODEL':0,
                'VPAR_SHEAR_MODEL':1,
                'SIGN_BT':1.0,
                'SIGN_IT':1.0,
                'KY':None,
                'NEW_EIKONAL':True,
                'VEXB':0.0,
                'VEXB_SHEAR':0.0,
                'BETAE': None,
                'XNUE':None,
                'ZEFF':None,
                'DEBYE':None,
                'IFLUX':True,
                'IBRANCH':-1,
                'NMODES':2,
                'NBASIS_MAX':4,
                'NBASIS_MIN':2,
                'NXGRID':16,
                'NKY':12,
                'ADIABATIC_ELEC':False,
                'ALPHA_MACH':0.0,
                'ALPHA_E':1.0,
                'ALPHA_P':1.0,
                'ALPHA_QUENCH':1.0,
                'ALPHA_ZF':1.0,
                'XNU_FACTOR':1.0,
                'DEBYE_FACTOR':1.0,
                'ETG_FACTOR':1.25,
                'WRITE_WAVEFUNCTION_FLAG':1,
            }
        )
        species_vector = {
            'ZS':None,
            'MASS':None,
            'RLNS':None,
            'RLTS':None,
            'TAUS':None,
            'AS':None,
            'VPAR':None,
            'VPAR_SHEAR':None,
            'VNS_SHEAR':None,
            'VTS_SHEAR':None,
        }
        species_params = {
            # values need to be specified as a nested dict of species vectors according to the above species vector format!
            'name':'Species vectors',
        }
        gaussian_params = {
            # all values at default
            'name':'Gaussian width parameters',
            'WIDTH':1.65,
            'WIDTH_MIN':0.3,
            'NWIDTH':21,
            'FIND_WIDTH':True,
        }
        geometry_params = {
            'miller':{
                # values need to be specified!
                'name':'Miller geometry parameters',
                'RMIN_LOC':None,
                'RMAJ_LOC':None,
                'ZMAJ_LOC':None,
                'DRMINDX_LOC':None,
                'DRMAJDX_LOC':None,
                'DZMAJDX_LOC':None,
                'Q_LOC':None,
                'KAPPA_LOC':None,
                'S_KAPPA_LOC':None,
                'DELTA_LOC':None,
                'S_DELTA_LOC':None,
                'ZETA_LOC':None,
                'S_ZETA_LOC':None,
                'P_PRIME_LOC':None,
                'Q_PRIME_LOC':None,
                'KX0_LOC':None,
            },
            's-alpha':{
                # values need to be specified!
                'name':'s-alpha geometry parameters',
                'RMIN_SA':None,
                'RMAJ_SA':None,
                'Q_SA':None,
                'SHAT_SA':None,
                'ALPHA_SA':None,
                'XWELL_SA':None,
                'THETA0_SA':None,
                'B_MODEL_SA':None,
                'FT_MODEL_SA':None,
            }
        }

        expert_params = {
            # all values at default
            'name':'Expert parameters',
            'DAMP_PSI':0.0,
            'DAMP_SIG':0.0,
            'PARK':1.0,
            'GHAT':1.0,
            'GCHAT':1.0,
            'WD_ZERO':0.1,
            'LINSKER_FACTOR':0.0,
            'GRADB_FACTOR':0.0,
            'FILTER':2.0,
            'THETA_TRAPPED':0.7,
            'NN_MAX_ERROR':-1.0,
        }

        # modify the different parameter namelists with custom inputs
        if control != None:
            for key in control:
                control_params[key] = control[key]
            for key in control_params:
                    if control_params[key]==None:
                        exit("Control parameter: '"+key+"' has not been specified, please check your inputs!")
        if species != None:
            for key in species_vector:
                for index in species:
                    if key in species[index]:
                        species_params[key+'_'+str(index)] = species[index][key]
                    else:
                        species_params[key+'_'+str(index)] = 0.0
        else:
            exit('No species information specified!')
        if gaussian != None:
            for key in gaussian:
                gaussian_params[key] = gaussian[key]
        if geometry != None:
            if geometry['name']=='miller':
                for key in geometry:
                    if key != 'name':
                        geometry_params[geometry['name']][key] = geometry[key]
                for key in geometry_params[geometry['name']]:
                    if geometry_params[geometry['name']][key]==None:
                        exit(geometry['name']+" parameter: '"+key+"' has not been specified, please check your geometry inputs!")   
        if expert != None:
            for key in expert:
                expert_params[key] = expert[key]

        tglf_namelist = {
            'meta' : {
                "file":fname,
            },
            'header':header_params,
            'control':control_params,
            'species':species_params,
            'gaussian':gaussian_params,
            'geometry':geometry_params[geometry['name']],
            'expert':expert_params,
        }
        if path == None:
            tglf_namelist['meta'].update({'path':"./",})
        elif isinstance(path,str):
            tglf_namelist['meta'].update({'path':path,})
        else:
            exit("Invalid path string provided!")

        spacer = '#---------------------------------------------------\n'
        for namelist in tglf_namelist:
            if namelist == 'meta':
                if tglf_namelist[namelist]['file'] != None:
                    pathlib.Path(tglf_namelist[namelist]["path"]).mkdir(parents=True, exist_ok=True)
                    f = open(tglf_namelist[namelist]["path"]+tglf_namelist[namelist]["file"],"w+")
                    generated_file = True
                else:
                    exit('File name not specified!')
            elif namelist == 'header':
                for key in tglf_namelist[namelist]:
                    f.write('# '+tglf_namelist[namelist][key]+'\n')
            elif namelist != 'meta' and namelist != 'header':
                f.write(spacer)
                f.write('# '+tglf_namelist[namelist]['name']+'\n')
                f.write(spacer)
                for key in tglf_namelist[namelist]:
                    if key != 'name':
                        f.write("{}={}\n".format(key,tglf_namelist[namelist][key]))
                if namelist != 'expert':
                    f.write('\n')
        f.close()
        if generated_file:
            print('Generated TGLF input file at: '+tglf_namelist['meta']["path"]+tglf_namelist['meta']["file"])

## QuaLiKiz
class QLK:
    def __init__(self):
        self.metadata = {}
        self.input = {}
        self.output = {}
    
    # I/O functions
    def write_input(rho=None,dataset=None,output_loc=None,fname=None,imp_composite=False):
        rho_idx = np.abs(dataset['rho']-rho).argmin()

        ne = dataset['ne'][rho_idx]*1e-19
        RLne = dataset['RLne'][rho_idx]
        ni = dataset['ni'][rho_idx]*1e-19
        RLni = dataset['RLni'][rho_idx]
        A_i = 2
        Z_i = 1

        n_LZ = dataset['n_LZ'][rho_idx]*1e-19
        RLn_LZ = dataset['RLn_LZ'][rho_idx]
        A_LZ = 9
        Z_L = dataset['Z_L']

        if imp_composite:
            n_comp = dataset['n_comp'][rho_idx]*1e-19
            RLn_comp = dataset['RLn_comp'][rho_idx]
            A_comp = dataset['A_comp'][rho_idx]
            Z_comp = dataset['Z_comp'][rho_idx]

        Te = dataset['Te'][rho_idx]*1e-3
        RLTe = dataset['RLTe'][rho_idx]
        Ti = dataset['Ti'][rho_idx]*1e-3
        RLTi = dataset['RLTi'][rho_idx]

        q = dataset['q'][rho_idx]
        s = dataset['s'][rho_idx]
        B0 = dataset['B0']

        alpha = dataset['alpha'][rho_idx]
        a = dataset['a']
        x = dataset['x'][rho_idx]
        Ro = dataset['Ro'][rho_idx]
        R0 = dataset['R0']

        Mach_tor = 0
        Au_tor = 0
        Mach_par = 0
        Au_par = 0
        gamma_E = 0

        ## QuaLiKiz input preparation
        from qualikiz_tools.qualikiz_io.inputfiles import QuaLiKizXpoint, Electron, Ion, IonList
        from qualikiz_tools.qualikiz_io.inputfiles import QuaLiKizPlan

        kthetarhos = list(np.linspace(0.1,0.8,8))
        elec = Electron(T=Te,n=ne,At=RLTe,An=RLne,type=1,anis=1, danisdr=0)
        ion0 = Ion(T=Ti,n=ni,At=RLTi,An=RLni,A=A_i,Z=Z_i,type=1,anis=1,danisdr=0)
        ion1 = Ion(T=Ti,n=n_LZ,At=RLTi,An=RLn_LZ,A=A_LZ,Z=Z_L,type=1,anis=1,danisdr=0)
        if imp_composite:
            ion2 = Ion(T=Ti,n=n_comp,At=RLTi,An=RLn_comp,A=A_comp,Z=Z_comp,type=1,anis=1,danisdr=0)
            ions = IonList(ion0,ion1,ion2)
        else:
            ions = IonList(ion0,ion1)

        meta = {
            "phys_meth": 1,
            "coll_flag": 1,
            "rot_flag": 0,
            "verbose": True,
            "separateflux": False,
            "write_primi": True,
            "numsols": 2,
            "relacc1": 0.00001,
            "relacc2": 0.0002,
            "absacc1": 0,
            "absacc2": 0,
            "integration_routine": 1,
            "maxruns": 1,
            "maxpts": 500000000.0,
            "timeout": 60,
            "ETGmult": 1,
            "collmult": 0.5,
            "R0": R0
        }
        spatial = {
            "x": x,
            "rho": rho,
            "Ro": Ro,
            "Rmin": a,
            "Bo": B0,
            "q": q,
            "smag": s,
            "alpha": alpha,
            "Machtor": Mach_tor,
            "Autor": Au_tor,
            "Machpar": Mach_par, 
            "Aupar": Au_par,
            "gammaE": gamma_E
        }
        options = {
            "set_qn_normni": True,
            "set_qn_normni_ion": 0,
            "set_qn_An": True,
            "set_qn_An_ion": 0,
            "check_qn": True,
            "x_eq_rho": True,
            "recalc_Nustar": False,
            "recalc_Ti_Te_rel": False,
            "assume_tor_rot": True,
            "puretor_abs_var": "Machtor",
            "puretor_grad_var": "gammaE"
        }

        xpoint_base = QuaLiKizXpoint(**meta, kthetarhos=kthetarhos, electrons=elec, ions=ions, **spatial, **options)

        scan_dict = {'Ate': [xpoint_base['Ate']]}
        plan = QuaLiKizPlan(scan_dict=scan_dict,scan_type='hyperedge',xpoint_base=xpoint_base)
        plan.to_json(output_loc+fname)