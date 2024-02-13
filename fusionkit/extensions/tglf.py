'''
The TGLF class

For more information on the TGLF code see https://gafusion.github.io/doc/tglf.html
NOTE: TGLF does not allow (unescaped) spaces in the run path!
NOTE: run_path needs to end with a /
'''
import sys
import os, glob
import copy

import numpy as np
import matplotlib.pyplot as plt

from ..core.dataspine import DataSpine
from ..core.utils import *
from ..core.get_metric_fusionkit import *

from idspy_dictionaries import ids_gyrokinetics as gkids
import idspy_toolkit as idspy

# Import the file FS_param.py from the repository IMAS-GK 
# that can be found at https://gitlab.com/gkdb/imas-gk
# it can be obtained with the command "git clone https://gitlab.com/gkdb/imas-gk.git"
if not '/home/anass/codes/imas-gk/python/' in sys.path :
    sys.path.append('/home/anass/codes/imas-gk/python/')

if not '/home/anass/codes/imas-gk/gkids/tools' in sys.path :
    sys.path.append('/home/anass/codes/imas-gk/gkids/tools')
from FS_param import *

CURSOR_UP_ONE = '\x1b[1A'
if os.uname().sysname.lower() in ['darwin','linux','linux2']:
    ERASE_LINE = '\x1b[2K'
else:
    ERASE_LINE = '\x1b[1M' #for Windows

## TGLF
class TGLF(DataSpine):
    def __init__(self):
        DataSpine.__init__(self)
        self.input = {}
        self.output = {}
        self.collect = True
        self._species_vector = {
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
        self._input_defaults = {
            # control
            'UNITS':'GYRO',
            'NS':2,
            'USE_TRANSPORT_MODEL':'T',
            'GEOMETRY_FLAG':1,
            'USE_BPER':'F',
            'USE_BPAR':'F',
            'USE_BISECTION':'T',
            'USE_MHD_RULE':'T',
            'USE_INBOARD_DETRAPPED':'F',
            'SAT_RULE':0,
            'KYGRID_MODEL':1,
            'XNU_MODEL':2,
            'VPAR_MODEL':0,
            'VPAR_SHEAR_MODEL':0,
            'SIGN_BT':1.0,
            'SIGN_IT':1.0,
            'KY':0.3,
            'NEW_EIKONAL':'T',
            'VEXB':0.0,
            'VEXB_SHEAR':0.0,
            'BETAE': 0.0,
            'XNUE':0.0,
            'ZEFF':1.0,
            'DEBYE':0.0,
            'IFLUX':'T',
            'IBRANCH':-1,
            'NMODES':2,
            'NBASIS_MAX':4,
            'NBASIS_MIN':2,
            'NXGRID':16,
            'NKY':12,
            'ADIABATIC_ELEC':'F',
            'ALPHA_P':1.0,
            'ALPHA_MACH':0.0,
            'ALPHA_E':1.0,
            'ALPHA_QUENCH':0.0,
            'ALPHA_ZF':1.0,
            'XNU_FACTOR':1.0,
            'DEBYE_FACTOR':1.0,
            'ETG_FACTOR':1.25,
            # gaussian
            'WRITE_WAVEFUNCTION_FLAG':0,
            'WIDTH':1.65,
            'WIDTH_MIN':0.3,
            'NWIDTH':21,
            'FIND_WIDTH':'T',
            # miller
            'RMIN_LOC':0.5,
            'RMAJ_LOC':3.0,
            'ZMAJ_LOC':0.0,
            'Q_LOC':2.0,
            'Q_PRIME_LOC':16.0,
            'P_PRIME_LOC':0.0,
            'DRMINDX_LOC':1.0,
            'DRMAJDX_LOC':0.0,
            'DZMAJDX_LOC':0.0,
            'KAPPA_LOC':1.0,
            'S_KAPPA_LOC':0.0,
            'DELTA_LOC':0.0,
            'S_DELTA_LOC':0.0,
            'ZETA_LOC':0.0,
            'S_ZETA_LOC':0.0,
            'KX0_LOC':0.0,
            # s-alpha
            'RMIN_SA':0.5,
            'RMAJ_SA':3.0,
            'Q_SA':2.0,
            'SHAT_SA':1.0,
            'ALPHA_SA':0.0,
            'XWELL_SA':0.0,
            'THETA0_SA':0.0,
            'B_MODEL_SA':1,
            'FT_MODEL_SA':1,
            # expert
            'THETA_TRAPPED':0.7,
            'PARK':1.0,
            'GHAT':1.0,
            'GCHAT':1.0,
            'WD_ZERO':0.1,
            'LINSKER_FACTOR':0.0,
            'GRADB_FACTOR':0.0,
            'FILTER':2.0,
            'DAMP_PSI':0.0,
            'DAMP_SIG':0.0,
            'NN_MAX_ERROR':-1.0,
        }
        self._ids_map = {}

    # I/O functions  
    def read_ave_p0_spectrum(self,run_path=None,nky=None):
        ave_p0 = self.read_var_spectrum(run_path=run_path,file='out.tglf.ave_p0_spectrum',header=3,nky=nky,var='ave_p0')
        if not self.collect:
            return ave_p0
        
    def read_density_spectrum(self,run_path=None,nspecies=None):
        self.read_fluctuation_spectrum(run_path=run_path,file='out.tglf.density_spectrum',symbol='n',nspecies=nspecies)

    def read_eigenvalue_spectrum(self,run_path=None,nmodes=None,sign_convention=-1):
        """Read the eigenvalue spectra and store them per mode.
        Frequency convention is set to -1 (electron diamagnetic direction modes negative, ion diamagnetic direction positive), 1 is the TGLF default.

        Args:
            run_path (_type_, optional): _description_. Defaults to None.
            nmodes (int, optional): _description_. Defaults to 1.
            sign_convention (int, optional): _description_. Defaults to -1.

        Returns:
            _type_: _description_
        """
        # if unspecified, for convenience check for run path in metadata
        if not run_path and 'run_path' in self.metadata:
            run_path = self.metadata['run_path']

        lines = read_file(path=run_path,file='out.tglf.eigenvalue_spectrum')

        # if the file was successfully read
        if lines:
            # set file dependent variables
            header = 2
            ev = {'gamma':{},'omega':{}}
            # check if storing IO in the TGLF object
            if self.collect:
                if 'eigenvalues' not in self.output:
                    self.output['eigenvalues'] = {}
                eigenvalues = self.output['eigenvalues']
            # reader is used standalone
            else:
                eigenvalues = {}
            # read the file description
            description = lines[0].strip()
            # set and check index limits if applicable 
            _nmodes = int(len(lines[header].strip().split())/2)
            if not nmodes or not nmodes <= _nmodes:
                nmodes = _nmodes
            # prefill/merge trees
            merge_trees(ev,eigenvalues)
            for key_ev in ev:
                for i_mode in range(0,nmodes):
                    key_mode = i_mode+1
                    if key_mode not in eigenvalues[key_ev]:
                        eigenvalues[key_ev].update({key_mode:[]})
            # read the eigenvalue spectra for all modes line by line
            for line in lines[header:]:
                row = line.split()
                row = [float(value) for value in row]
                for i_mode in range(0,nmodes):
                    key_mode = i_mode + 1
                    if row[0+i_mode*2] != 0:
                        eigenvalues['gamma'][key_mode].append(row[0+i_mode*2])
                    else:
#                        eigenvalues['gamma'][key_mode].append(np.NaN)
                        eigenvalues['gamma'][key_mode].append(0.0)
                    if row[1+i_mode*2] != 0:
                        eigenvalues['omega'][key_mode].append(sign_convention*row[1+i_mode*2])
                    else:
#                        eigenvalues['omega'][key_mode].append(np.NaN)
                        eigenvalues['omega'][key_mode].append(0.0)

            eigenvalues = list_to_array(eigenvalues)
            
            if self.collect:
                eigenvalues['sign_convention'] = sign_convention
                merge_trees({'nmodes':nmodes},self.metadata)
            else:
                eigenvalue_spectrum = {'description':description, 'eigenvalues':eigenvalues, 'nmodes':nmodes, 'sign_convention':sign_convention}
                return eigenvalue_spectrum

    def read_field_spectrum(self,run_path=None,nky=None,nmodes=None):
        """Read the gyro-bohn normalised field fluctuation intensity spectra and store them per mode.

        Args:
            run_path (str): path to the output directory of the tglf run. Defaults to None.
            nky (int, optional): the desired number ky modes to be read from the field fluctuation spectra. Defaults to the total number of ky in the spectrum.
            nmodes (int, optional): the desired number of modes for which to read the field fluctuation spectra. Defaults to the total number modes in the file.

        Returns:
            dict: contains the file description, the field fluctuation spectra per mode and the number of modes.
        """
        # if unspecified, for convenience check for run path in metadata
        if not run_path and 'run_path' in self.metadata:
            run_path = self.metadata['run_path']

        # read the output.tglf.field_spectrum file
        lines = read_file(path=run_path,file='out.tglf.field_spectrum')

        # if the file was successfully read
        if lines:
            # set file dependent variables
            header = 6
            _fields = {'A':{}, 'phi':{},}
            # check if storing IO in the TGLF object
            if self.collect:
                if 'fields' not in self.output:
                    self.output['fields'] = {}
                fields = self.output['fields']
            # reader is used standalone
            else:
                fields = {}
            # read the file description
            description = lines[0].strip()
            # read the index limits
            [_nky,_nmodes] = [int(limit) for limit in lines[3].strip().split()]
            if not nky or not nky <= _nky:
                nky = _nky
            if not nmodes or not nmodes <= _nmodes:
                nmodes = _nmodes
            # check if A_par and B_par and if present add them to the fields dict
            a_par = bool(lines[:header][-2].strip().split('_')[-1])
            b_par = bool(lines[:header][-1].strip().split('_')[-1])
            if a_par:
                _fields.update({'A_par':{}})
            if b_par:
                _fields.update({'B_par':{}})
            # prefill/merge trees
            merge_trees(_fields,fields)
            for key_field in _fields:
                for i_mode in range(0,nmodes):
                    key_mode = i_mode+1
                    if key_mode not in fields[key_field]:
                        fields[key_field].update({key_mode:[]})
            
            for i_mode in range(0,nmodes):
                #print(lines[header+(i_mode*_nky):header+(i_mode*_nky)+nky])
                key_mode = i_mode+1
                # per mode read the field fluctuation amplitude spectrum line by line
                for i_ky in range(0,nky):
                    index = header+(i_ky*_nmodes)+i_mode
                    row = lines[index].strip().split()
                    row = [float(value) for value in row]
                    for i_field,key_field in enumerate(fields.keys()):
                        fields[key_field][key_mode].append(row[i_field])
            list_to_array(fields)

            if self.collect:
                merge_trees({'nmodes':nmodes},self.metadata)
            else:
                field_spectrum = {'description':description, 'fields':fields, 'nmodes':nmodes}
                return field_spectrum

    def read_fluctuation_spectrum(self,run_path=None,file=None,symbol=None,nspecies=None):
        """Read the output.tglf.density_spectrum or output.tglf.temperature_spectrum file.

        Args:
            run_path (_type_, optional): _description_. Defaults to None.
            nspecies (int, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # if unspecified, for convenience check for run path in metadata
        if not run_path and 'run_path' in self.metadata:
            run_path = self.metadata['run_path']

        lines = read_file(path=run_path,file=file)

        # if the file was successfully read
        if lines:
            # set file dependent variables
            header = 2
            _fluctuations = {'fluctuations':{'amplitude':{symbol:[]}}}
            # check if storing IO in the TGLF object
            if self.collect:
                if 'species' not in self.output:
                    self.output['species'] = {}
                species = self.output['species']
            # reader is used standalone
            else:
                species = {}
            # read the file description
            description = lines[0].strip()
            # set and check index limits if applicable 
            _nspecies = len(lines[header].strip().split())
            if not nspecies or not nspecies <= _nspecies:
                nspecies = _nspecies
            # read the fluctuation amplitude spectrum for each species line by line
            for line in lines[header:]:
                row = line.strip().split()
                row = [float(value) for value in row]
                for i_species in range(0,nspecies):
                    key_species = i_species+1
                    if key_species not in species:
                        species[key_species]={}
                    merge_trees(_fluctuations,species[key_species])
                    fluctuations = species[key_species]['fluctuations']['amplitude']
                    fluctuations[symbol].append(row[i_species])

            for key_species in species.keys():
                list_to_array(species[key_species]['fluctuations']['amplitude'])

            if self.collect:
                merge_trees({'nspecies':nspecies},self.metadata)
            else:
                fluctuation_spectrum = {'description':description, 'species':species, 'nspecies':nspecies}
                return fluctuation_spectrum

    def read_gbflux(self,run_path=None,nspecies=None):
        """Read the gyro-Bohm normalised fluxes averaged over radius and summed over mode number.

        Args:
            run_path (_type_, optional): _description_. Defaults to None.
            nspecies (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        # if unspecified, for convenience check for run path in metadata
        if not run_path and 'run_path' in self.metadata:
            run_path = self.metadata['run_path']

        # read the output.tglf.gbflux file
        lines = read_file(path=run_path,file='out.tglf.gbflux')

        species = {}

        # if the file was successfully read
        if lines:
            # check if storing IO in the TGLF object
            if self.collect:
                if 'species' not in self.output:
                    self.output['species'] = {}
                species = self.output['species']
            # reader is used standalone
            else:
                species = {}
            _fluxes = {'fluxes':{'Gamma':{'total':0.}, 'Q':{'total':0.}, 'Pi':{'total':0.}, 'S':{'total':0.}}}
            # convert the line of values into a list of floats
            fluxes_list = [float(value) for value in lines[0].split()]
            _nspecies = int(len(fluxes_list)/len(list(_fluxes['fluxes'].keys())))
            if not nspecies or not nspecies <= _nspecies:
                nspecies = _nspecies

            for i_species in range(0,nspecies):
                key_species = i_species+1
                if key_species not in species:
                    species.update({key_species:copy.deepcopy(_fluxes)})
                else:
                    merge_trees(_fluxes,species[key_species])
                fluxes = species[key_species]['fluxes']
                for i_flux,key_flux in enumerate(_fluxes['fluxes'].keys()):
                    fluxes[key_flux]['total'] = ((fluxes_list[i_flux*_nspecies:(i_flux+1)*_nspecies])[:nspecies])[i_species]

            if self.collect:
                merge_trees({'nspecies':nspecies},self.metadata)
            else:
                gbfluxes = {'description':'gyro-bohm normalized fluxes summed over mode number and ky spectrum', 
                            'species':species, 'nspecies':nspecies}
                return gbfluxes

    def read_grid(self,run_path=None):
        """Read the out.tglf.grid file

        Args:
            run_path (str): path to the output directory of the tglf run. Defaults to None.

        Returns:
            dict: {'nspecies':int,'nxgrid':int}
        """
        # if unspecified, for convenience check for run path in metadata
        if not run_path and 'run_path' in self.metadata:
            run_path = self.metadata['run_path']

        # read the out.tglf.grid file
        lines = read_file(path=run_path,file='out.tglf.grid')

        if lines:
            nspecies = autotype(lines[0].strip())
            nxgrid = autotype(lines[1].strip())

            grid = {'nspecies':nspecies, 'nxgrid':nxgrid}

            if self.collect:
                merge_trees(grid,self.metadata)
            else:
                return grid

    def read_input(self,run_path=None,file='input.tglf',overwrite=False):
        """Read the TGLF inputs from the specified run folder.

        Args:
            `run_path` (str): path to the TGLF run.
            `file` (str): filename. Defaults to 'input.tglf'.

        Returns:
            dict: all TGLF variables set in the input.tglf file as key:value pairs
        """
        # if unspecified, for convenience check for run path in metadata
        if not run_path and 'run_path' in self.metadata:
            run_path = self.metadata['run_path']

        # read the input.tglf file
        try:
            lines = read_file(path=run_path,file=file)
        except:
            assert False, (str(run_path) + file + ' does not exist')

        # if the file was successfully read
        if lines:
            # set file dependent variables
            description = ''
            # check if storing IO in the TGLF object
            if self.collect:
                if self.input and overwrite:
                    self.input = {}
                input = self.input
            # reader is used standalone
            else:
                input = {}
            # go line by line
            for line in lines:
                # check it is not a header line
                if '#' not in line and line.strip():
                    line = line.split("=")
                    key = line[0].strip()
                    value = line[1].strip()
                    # type the values, not using autotype to keep bool values in Fortran format
                    # TODO: change write_input to convert bools to Fortran form and switch to autotype
                    try:
                        value = int(value)
                    except:
                        try:
                            value = float(value)
                        except:
                            value = str(value)

                    input.update({key:value})
                elif line.strip():
                    line = line.strip().split('#')[1]
                    description += ', '+line

            if not self.collect:
                input.update({'description':description})
                return input

    def read_input_gen(self,run_path=None):
        """Read the ouput TGLF input_gen.

        Args:
            `run_path` (str): path to the TGLF run.

        Returns:
            dict: all the TGLF input variables used in the run output in input.tglf.gen.
        """
        Fortran_false_list = ['F','f','False','false','.False.','.false.']
        Fortran_true_list = ['T','t','True','true','.True.','.true.']        
        
        # if unspecified, for convenience check for run path in metadata
        if not run_path and 'run_path' in self.metadata:
            run_path = self.metadata['run_path']
    
        # read the output.tglf.input_gen file
        lines = read_file(path=run_path,file='input.tglf.gen')
        
        # if the file was successfully read
        if lines:
            # set file dependent variables
            if self.collect:
                self.output['input_gen'] = {}
                input_gen = self.output['input_gen']
            else:
                input_gen = {}
            # convert input.tglf.gen line by line into key:value pairs
            for line in lines:
                line = line.split()
                key = line[1].strip()
                value = line[0].strip()
                try:
                    value = int(value)
                except:
                    try:
                        value = float(value)
                    except:
                        value = str(value)
                        if (value in Fortran_false_list):
                            value = False
                        if (value in Fortran_true_list):
                            value = True
                input_gen.update({key:value})
        
            # if standalone reader use
            if not self.collect:
                return input_gen

    def read_intensity_spectrum(self,run_path=None,nmodes=None,nspecies=None,nky=None):
        # if unspecified, for convenience check for run path in metadata
        if not run_path and 'run_path' in self.metadata:
            run_path = self.metadata['run_path']

        # read the output.tglf.intensity_spectrum file
        lines = read_file(path=run_path,file='out.tglf.intensity_spectrum')

        if lines:
            # set file dependent variables
            header = 4
            _fluctuations = {'fluctuations':{'intensity':{'n':{}, 'T':{}, 'v_par':{}, 'E_par':{}}}}
            # check if storing IO in the TGLF object
            if self.collect:
                if 'species' not in self.output:
                    self.output['species'] = {}
                species = self.output['species']
            # reader is used standalone
            else:
                species = {}
            # read the file description
            description = lines[0].strip()
            # read the index limits
            [_nspecies,_nky,_nmodes] = [int(limit) for limit in lines[3].strip().split()]
            # set and check index limits if applicable 
            if not nspecies or not nspecies <= _nspecies:
                nspecies = _nspecies
            if not nky or not nky <= _nky:
                nky = _nky
            if not nmodes or not nmodes <= _nmodes:
                nmodes = _nmodes
            # per mode, per species read the field fluctuation amplitude spectrum line by line
            for i_species in range(0,nspecies):
                key_species = i_species + 1
                if key_species not in species:
                    species[key_species] = {}
                merge_trees(_fluctuations,species[key_species])
                fluctuations = species[key_species]['fluctuations']['intensity']
                for i_mode in range(0,nmodes):
                    key_mode = i_mode + 1
                    for i_ky in range(0,nky):
                        index = header+(i_species*_nmodes*_nky)+(i_ky*_nmodes)+i_mode
                        row = lines[index].strip().split()
                        row = [float(value) for value in row]
                        for i_fluctuation,key_fluctuation in enumerate(_fluctuations['fluctuations']['intensity'].keys()):
                            if key_mode not in fluctuations[key_fluctuation]:
                                fluctuations[key_fluctuation].update({key_mode:[]})
                            fluctuations[key_fluctuation][key_mode].append(row[i_fluctuation])
                list_to_array(fluctuations)
            
            if self.collect:
                merge_trees({'nspecies':nspecies, 'nky':nky, 'nmodes':nmodes},self.metadata)
            else:
                intensity_spectrum = {'description':description, 'species':species, 'nspecies':nspecies, 'nky':nky, 'nmodes':nmodes}
                return intensity_spectrum

    def read_ky_spectrum(self,run_path=None):
        # if unspecified, for convenience check for run path in metadata
        if not run_path and 'run_path' in self.metadata:
            run_path = self.metadata['run_path']

        # read the out.tglf.ky_spectrum file
        lines = read_file(path=run_path,file='out.tglf.ky_spectrum')

        # if the file was successfully read
        if lines:
            # set file dependent variables
            header = 2
            ky_list = []
            # get the number of ky in the spectrum and then read the spectrum into a list
            for i_line,line in enumerate(lines):
                if i_line == 1:
                    nky = int(line.strip())
                if header <= i_line <= nky+header:
                    ky_list.append(float(line.strip()))
        
            if self.collect:
                merge_trees({'nky':nky},self.metadata)
                self.output['ky'] = list_to_array(ky_list)
            else:
                ky_spectrum = {'nky':nky, 'ky':ky_list}
                return ky_spectrum

    def read_nete_crossphase_spectrum(self,run_path=None,nmodes=None,nky=None):
        # if unspecified, for convenience check for run path in metadata
        if not run_path and 'run_path' in self.metadata:
            run_path = self.metadata['run_path']
        
        # read the output.tglf.nete_crossphase_spectrum file
        lines = read_file(path=run_path,file='out.tglf.nete_crossphase_spectrum')

        if lines:
            # set file dependent variables
            header = 2
            key_species = 1 # since this is nete cross phase spectrum
            _crossphase = {'nt_cross_phase':{}}
            # check if storing IO in the TGLF object
            if self.collect:
                if 'species' not in self.output:
                    self.output['species'] = {}
                species = self.output['species']
            else:
                species = {}
            # read the file description
            description = lines[0].strip()
            # set and check index limits if applicable 
            _nmodes = len(lines[header].strip().split())
            _nky = len(lines[header:])
            if not nmodes or not nmodes <= _nmodes:
                nmodes = _nmodes
            if not nky or not nky <= _nky:
                nky = _nky
            if key_species not in species:
                species[key_species] = {}
            merge_trees(_crossphase,species[key_species])
            crossphase = species[key_species]['nt_cross_phase']
            for i_ky in range(0,nky):
                index = header + i_ky
                row = lines[index].strip().split()
                row = [float(value) for value in row]
                for i_mode in range(0,nmodes):
                    key_mode = i_mode + 1
                    if key_mode not in crossphase:
                        crossphase[key_mode] = []
                    crossphase[key_mode].append(row[i_mode])
            
            list_to_array(crossphase)

            if self.collect:
                merge_trees({'nmodes':nmodes, 'nky':nky},self.metadata)
            else:
                nete_crossphase_spectrum = {'description':description, 'species':species, 'nmodes':nmodes, 'nky':nky}
                return nete_crossphase_spectrum

    def read_nsts_crossphase_spectrum(self,run_path=None,nspecies=None,nmodes=None,nky=None):
        # if unspecified, for convenience check for run path in metadata
        if not run_path and 'run_path' in self.metadata:
            run_path = self.metadata['run_path']
        
        # read the output.tglf.nsts_crossphase_spectrum file
        lines = read_file(path=run_path,file='out.tglf.nsts_crossphase_spectrum')

        if lines:
            # set file dependent variables
            header = 1
            header_species = 2
            _crossphase = {'nt_cross_phase':{}}
            # check if storing IO in the TGLF object
            if self.collect:
                if 'species' not in self.output:
                    self.output['species'] = {}
                species = self.output['species']
            else:
                species = {}
            # read the file description
            description = ' '.join(lines[0].strip().split())
            # set and check index limits if applicable
            _nspecies = int(description.split()[-2])
            _nmodes = int(len(lines[header+header_species].strip().split()))
            _nky = int((len(lines)-header)/_nspecies)-header_species
            if not nspecies or not nspecies <= _nspecies:
                nspecies = _nspecies
            if not nmodes or not nmodes <= _nmodes:
                nmodes = _nmodes
            if not nky or not nky <= _nky:
                nky = _nky
            for i_line,line in enumerate(lines[header:]):
                row = line.strip().split()
                # check if the row is a header line
                if 'species' in row:
                    key_species = int(row[-1])
                    if key_species not in species:
                        species[key_species] = {}
                    merge_trees(_crossphase,species[key_species])
                    crossphase = species[key_species]['nt_cross_phase']
                elif '(nsts_phase_spectrum_out' in row[0].split(','):
                    if 'nsts_phase_spectrum_out' not in description:
                        description += ', '+''.join(row)
                else:
                    line_lb = (key_species*header_species)+((key_species-1)*_nky)-1
                    if  line_lb < i_line <= line_lb+nky:
                        row = [float(value) for value in row]
                        for i_mode in range(0,nmodes):
                            key_mode = i_mode + 1
                            if key_mode not in crossphase:
                                crossphase[key_mode] = []
                            crossphase[key_mode].append(row[i_mode])

            for key_species in species.keys():
                list_to_array(species[key_species]['nt_cross_phase'])

            if self.collect:
                merge_trees({'nmodes':nmodes, 'nky':nky},self.metadata)
            else:
                nsts_crossphase_spectrum = {'description':description, 'species':species, 'nmodes':nmodes, 'nky':nky}
                return nsts_crossphase_spectrum

    def read_prec(self,run_path=None):
        """Read the out.tglf.prec file.

        Args:
            run_path (str): the path to the TGLF run. Defaults to None.

        Returns:
            _type_: _description_
        """
        # if unspecified, for convenience check for run path in metadata
        if not run_path and 'run_path' in self.metadata:
            run_path = self.metadata['run_path']

        lines = read_file(path=run_path,file='out.tglf.prec')

        if lines:
            prec = float(lines[0].strip())

            if self.collect:
                self.output['prec'] = prec
            else:
                return prec

    def read_QL_flux_spectrum(self,run_path=None,nspecies=None,nfields=None,nky=None,nmodes=None):
        # if unspecified, for convenience check for run path in metadata
        if not run_path and 'run_path' in self.metadata:
            run_path = self.metadata['run_path']

        # read the output.tglf.QL_flux_spectrum file
        lines = read_file(path=run_path,file='out.tglf.QL_flux_spectrum')

        # if the file was successfully read
        if lines:
            # set file dependent variables
            header = 4
            weights = {'QL_weights':{'Gamma':{}, 'Q':{}, 'Pi_tor':{}, 'Pi_par':{}, 'S':{}}}
            fields = ['phi','Bper','Bpar']
            # check if storing IO in the TGLF object
            if self.collect:
                if 'species' not in self.output:
                    self.output['species'] = {}
                species = self.output['species']
            else:
                species = {}
            # read the file description
            description = ' '.join([line.strip() for line in lines[:2]])
            [_nfluxes,_nspecies,_nfields,_nky,_nmodes] = [autotype(value) for value in lines[header-1].strip().split()]
            if not nspecies or not nspecies <= _nspecies:
                nspecies = _nspecies
            if not nfields or not nfields <= _nfields:
                nfields = _nfields
            if not nky or not nky <= _nky:
                nky = _nky
            if not nmodes or not nmodes <= _nmodes:
                nmodes = _nmodes

            # go line by line
            for line in lines[header:]:
                row = line.strip().split()
                if 'species' in row:
                    key_species = int(row[2])
                    if key_species not in species:
                        species[key_species] = {}
                    merge_trees(weights,species[key_species])
                    ql_weigths = species[key_species]['QL_weights']
                    i_field = int(row[-1])
                    for key_flux in weights['QL_weights'].keys():
                        if key_flux not in ql_weigths:
                            ql_weigths[key_flux] = {}
                        if fields[i_field-1] not in ql_weigths[key_flux]:
                            ql_weigths[key_flux][fields[i_field-1]] = {}
                elif 'mode' in row:
                    key_mode = int(row[-1])
                else:
                    row = [float(value) for value in row]
                    for i_flux,key_flux in enumerate(weights['QL_weights'].keys()):
                        if key_mode not in ql_weigths[key_flux][fields[i_field-1]]:
                            ql_weigths[key_flux][fields[i_field-1]][key_mode] = []   
                        ql_weigths[key_flux][fields[i_field-1]][key_mode].append(row[i_flux])
            
            for key_species in species.keys():
                list_to_array(species[key_species]['QL_weights'])

            if self.collect:
                merge_trees({'nspecies':nspecies ,'nfields':nfields, 'nmodes':nmodes, 'nky':nky},self.metadata)
            else:
                QL_flux_spectrum = {'description':description, 'species':species, 
                                    'nspecies':nspecies ,'nfields':nfields, 'nmodes':nmodes, 'nky':nky}
                return QL_flux_spectrum

    def read_run(self,run_path=None):
        # if unspecified, for convenience check for run path in metadata
        if not run_path and 'run_path' in self.metadata:
            run_path = self.metadata['run_path']

        # read the out.tglf.run file
        lines = read_file(path=run_path,file='out.tglf.run')

        # set file dependent variables
        header = 1
        run = {}
        species = {}
        _species = 0
        GENE_units = False

        if lines:
            for line in lines[header:]:
                line = [autotype(value) for value in line.strip().split()]
                if 'mpi' in line:
                    run.update({'mpi':line[0]})
                elif 'GENE' in line:
                    GENE_units = True
                    run.update({'GENE_units':{}})
                elif 'Conversion' in line:
                    run['GENE_units'].update({'description':' '.join(line)})
                elif GENE_units and '/' in line[0] and '=' in line[1]:
                    run['GENE_units'].update({line[0]:line[-1]})
                elif '=' in str(line[1]):
                    run.update({line[0]:line[2],line[3]:line[5]})
                elif '=' in str(line[2]):
                    run.update({' '.join(line[:2]):line[3],' '.join(line[4:6]):line[-1]})
                elif '/' in line[0]:
                    fluxes = line
                elif len(line[1:])==len(fluxes):
                    _species += 1
                    for i_value,value in enumerate(line[1:]):
                        species.update({_species:{fluxes[i_value]:value}})               

            return run

    def read_sat_geo_spectrum(self,run_path=None,nmodes=None):
        # if unspecified, for convenience check for run path in metadata
        if not run_path and 'run_path' in self.metadata:
            run_path = self.metadata['run_path']

        # read the out.tglf.sat_geo_spectrum file
        lines = read_file(path=run_path,file='out.tglf.sat_geo_spectrum')

        if lines:
            # set file dependent variables
            header = 3
            if self.collect:
                if 'sat_geo' not in self.output:
                    self.output['sat_geo'] = {}
                sat_geo = self.output['sat_geo']
            else:
                sat_geo = {}
            description = lines[0].strip()
            # get/set the number modes
            _nmodes = int(lines[header-1].strip()[-1])
            if not nmodes or not nmodes <= _nmodes:
                nmodes = _nmodes
            # read the values row by row
            for line in lines[header:]:
                row = line.strip().split()

                for i_mode in range(0,nmodes):
                    key_mode = i_mode + 1
                    if key_mode not in sat_geo:
                        sat_geo[key_mode] = []
                    sat_geo[key_mode].append(float(row[i_mode]))
            list_to_array(sat_geo)
            
            if self.collect:
                merge_trees({'nmodes':nmodes},self.metadata)
            else:
                sat_geo_spectrum = {'description':description, 'sat_geo':sat_geo,'nmodes':nmodes}
                return sat_geo_spectrum

    def read_scalar_sat_parameters(self,run_path=None):
        # if unspecified, for convenience check for run path in metadata
        if not run_path and 'run_path' in self.metadata:
            run_path = self.metadata['run_path']

        # read the out.tglf.scalar_saturation_parameters file
        lines = read_file(path=run_path,file='out.tglf.scalar_saturation_parameters')

        if lines:
            if self.collect:
                if 'sat_scalar_params' not in self.output:
                    self.output['sat_scalar_params'] = {}
                scalar_sat_params = self.output['sat_scalar_params']
            else:
                scalar_sat_params = {}
            sat = None
            # check which version of the scalar_saturation_parameters file is present
            if '!' not in lines[0].strip():
                # set file dependent variables
                header = 2
                description = lines[0].strip()
            else:
                # set file dependent variables for old variant of the file
                header = 1
                description = lines[0].strip().split('!')[-1]
            # go line by line
            for i_line,line in enumerate(lines[header:]):
                # check if the line contains a SAT header
                if '!' in line and 'SAT' in line:
                    sat = ''
                    sats = [string.strip() for string in line.strip().split('SAT') if '!' not in string]
                    i_sats = [sat_int[0] for sat_int in sats if isinstance(autotype(sat_int[0]),int)]
                    for i_sat in i_sats:
                        if len(sat) > 0:
                            sat +='/'
                        sat += 'SAT{}'.format(i_sat)
                    if sat not in scalar_sat_params:
                        scalar_sat_params[sat] = {}
                # check if the line contains a key=value pair
                if '=' in line:
                    row = line.strip().split('=')
                    key = row[0]
                    value = autotype(row[-1])
                    # check if this a loose key=value pair or belongs under a SAT rule
                    if sat:
                        scalar_sat_params[sat][key] = value
                    else:
                        scalar_sat_params[key] = value
                # check if the line contains the csv keys/values
                elif ',' in line:
                    row = line.strip().split()
                    row = [autotype(value.split(',')[0]) for value in row]
                    # check if the line contains the variable names
                    if all(isinstance(value,str) for value in row):
                        for i_value,value in enumerate(row):
                            # get the next row containing the variable values
                            next_row = [autotype(value) for value in lines[header+i_line+1].strip().split()]
                            # automatically sort and collect all the values
                            scalar_sat_params.update({value:next_row[i_value]})

            if not self.collect:
                scalar_sat_params.update({'description':description})
                return scalar_sat_params

    def read_spectral_shift(self,run_path=None,nky=None):
        spectral_shift = self.read_var_spectrum(run_path=run_path,file='out.tglf.spectral_shift_spectrum',header=5,nky=nky,var='spectral_shift')
        if not self.collect:
            return spectral_shift

    def read_sum_flux_spectrum(self,run_path=None,nspecies=None,nfields=None):
        # if unspecified, for convenience check for run path in metadata
        if not run_path and 'run_path' in self.metadata:
            run_path = self.metadata['run_path']

        # read the output.tglf.sum_flux_spectrum file
        lines = read_file(path=run_path,file='out.tglf.sum_flux_spectrum')

        # if the file was successfully read
        if lines:
            # check if storing IO in the TGLF object
            if self.collect:
                if 'species' not in self.output:
                    self.output['species'] = {}
                species = self.output['species']
            # reader is used standalone
            else:
                species = {}
            _fluxes = {'fluxes':{'Gamma':{}, 'Q':{}, 'Pi_tor':{}, 'Pi_par':{}, 'S':{}}}
            _fields = ['phi','Bper','Bpar']
            # go line by line
            for line in lines:
                row = line.strip().split()
                # check if the line is a header line
                if 'species' in row:
                    key_species = int(row[2])
                    if not nspecies or key_species <= nspecies:
                        if key_species not in species:
                            species[key_species] = {}
                        merge_trees(_fluxes,species[key_species])
                        fluxes = species[key_species]['fluxes']
                        _key_field = int(row[-1])
                        if not nfields or _key_field <= nfields:
                            key_field = _key_field
                            for key_flux in fluxes.keys():
                                if _fields[key_field-1] not in fluxes[key_flux]:
                                    fluxes[key_flux].update({_fields[key_field-1]:[]})

                # check if the line is a description line
                elif 'particle' in row:
                    _description = line.strip().split(',')
                # automatically sort and collect all the values
                else:
                    row = [float(value) for value in row]
                    for i_flux,key_flux in enumerate(_fluxes['fluxes'].keys()):
                        fluxes[key_flux][_fields[key_field-1]].append(row[i_flux])
            
            #get the value for 'nfields'
            if not nfields:
                nfields = key_field
            # convert to arrays
            species = list_to_array(species)
            # add the total flux electrostatic and electromagnetic fluxes
            for key_species in species.keys():
                total = 0.
                fluxes = species[key_species]['fluxes']
                for key_flux in _fluxes['fluxes'].keys():
                    if len(fluxes[key_flux].keys()) > 1:
                        for key_field in fluxes[key_flux]:
                            total += fluxes[key_flux][key_field]
                        fluxes[key_flux].update({'sum':total})

            description = ', '.join(_description)+' by field per species'
            
            if self.collect:
                merge_trees({'nspecies':nspecies, 'nfields':nfields},self.metadata)
            else:
                flux_spectrum = {'description':description, 'species':species, 'nspecies':nspecies, 'nfields':nfields}
                return flux_spectrum

    def read_temperature_spectrum(self,run_path=None,nspecies=None):
        self.read_fluctuation_spectrum(run_path=run_path,file='out.tglf.temperature_spectrum',symbol='T',nspecies=nspecies)

    def read_version(self,run_path=None):
        # if unspecified, for convenience check for run path in metadata
        if not run_path and 'run_path' in self.metadata:
            run_path = self.metadata['run_path']

        # read the out.tglf.version file
        lines = read_file(path=run_path,file='out.tglf.version')

        # if the file was successfully read    
        if lines:
            # slice out.tglf.version into a searchable dict 
            version_commit = lines[0].split()[0].strip()
            version_date = lines[0].split()[1].strip().split('[')[1].split(']')[0]
            gacode_platform = lines[1].strip()
            run_date_time = lines[2].split()
            run_date = '{} {} {}'.format(run_date_time[1],run_date_time[2],run_date_time[3])
            run_time = '{} {}'.format(run_date_time[-2],run_date_time[-1])

            version = {'version':version_commit, 'version_date':version_date, 'platform':gacode_platform, 'run_date':run_date, 'run_time':run_time}

            if self.collect:
                merge_trees(version,self.metadata)
            else:
                return version

    def read_var_spectrum(self,run_path=None,file=None,header=None,nky=None,var=None):
        # if unspecified, for convenience check for run path in metadata
        if not run_path and 'run_path' in self.metadata:
            run_path = self.metadata['run_path']

        # read the out.tglf.width_spectrum file
        lines = read_file(path=run_path,file=file)

        # if the file was successfully read
        if lines and header:
            # set file dependent variables
            _list = []
            _nky = autotype(lines[header-1].strip().split()[-1])
            if not nky or not nky <= _nky:
                nky = _nky
            description = lines[0].strip()
            # read the spectrum into a list
            for line in lines[header:header+nky]:
                _list.append(float(line.strip()))
            
            if self.collect:
                merge_trees({var:list_to_array(_list)},self.output)
                merge_trees({'nky':nky},self.metadata)
            else:
                var_spectrum = {'description':description, 'nky':nky, var:list_to_array(_list)}
                return var_spectrum
    
    def read_wavefunction(self,run_path=None,file=None,nmodes=None,nfields=None,verbose=True):
        # if unspecified, for convenience check for run path in metadata
        if not run_path and 'run_path' in self.metadata:
            run_path = self.metadata['run_path']

        if not file:
            file = 'out.tglf.wavefunction'
        # read the output.tglf.wavefunction file
        lines = read_file(path=run_path,file=file)

        # if the file was successfully read
        if lines:
            # set file dependent variables
            header = 2
            fields_list = ['phi','Bper','Bpar']
            # check if storing IO in the TGLF object
            if self.collect:
                if 'eigenfunctions' not in self.output:
                    self.output['eigenfunctions'] = {}
                eigenfunctions = self.output['eigenfunctions']
            else:
                eigenfunctions = {}
            # get the index ranges
            [_nmodes,_nfields,ntheta] = [int(n) for n in lines[0].strip().split()]
            if not nmodes or not nmodes <= _nmodes:
                nmodes = _nmodes
            if not nfields or not nfields <= _nfields:
                nfields = _nfields
            # read column headers to store the field keys
            fields = {key:{} for key in lines[1].strip().split()}
            # update fields_list now knowing which fields are present in the file
            for key_field in fields_list:
                field_present = False
                for field in fields.keys():
                    if key_field in field:
                        field_present = True
                if not field_present:
                    fields_list.remove(key_field)

            # line by line read the field wavefunctions per mode from the file
            for line in lines[header:]:
                # strip, split and autotype the values in the current row
                row = [autotype(value) for value in line.strip().split()]
                # extract the values mode by mode, field by field
                for i_mode in range(0,nmodes):
                    key_mode = i_mode + 1
                    for i_field,key_field in enumerate(fields.keys()):
                        if key_mode not in fields[key_field]:
                            fields[key_field][key_mode] = []
                        mode = fields[key_field][key_mode]
                        # always get the theta value at the start of the line, regardless of mode number
                        if key_field == 'theta':
                            mode.append(row[i_field])
                        # otherwise using a sliding window on the row to get the appropriate mode data
                        else:
                            mode.append(row[i_field+(i_mode*(2*nfields))])
                if nmodes == 0:
                    key_mode = 0
                    for i_field,key_field in enumerate(fields.keys()):
                        if key_mode not in fields[key_field]:
                            fields[key_field][key_mode] = []
                        mode = fields[key_field][key_mode]
                        # always get the theta value at the start of the line, regardless of mode number
                        if key_field == 'theta':
                            mode.append(row[i_field])
                        # otherwise using a sliding window on the row to get the appropriate mode data
                        else:
                            mode.append(row[0])
            list_to_array(fields)
            if nmodes == 0:
                if verbose and '=' in file:
                    print('Warning, the number of modes is equal to 0 for ',file.split('_')[1])
                fields['theta'] = fields['theta'][0]
            else:
                fields['theta'] = fields['theta'][1]
            
            # prep the eigenfunction storage
            for key_field in ['theta']+fields_list:
                if key_field not in eigenfunctions:
                    eigenfunctions[key_field] = {}
                if key_field in fields_list:
                    real_part = [fields[_key_field] for _key_field in fields.keys() if key_field in _key_field and 'RE' in _key_field]
                    imag_part = [fields[_key_field] for _key_field in fields.keys() if key_field in _key_field and 'IM' in _key_field]
                    if real_part and imag_part:
                        real_part = real_part[0]
                        imag_part = imag_part[0]
                        for key_mode in real_part.keys():
                            if key_mode in imag_part.keys():
                                if key_mode not in eigenfunctions[key_field]:
                                    eigenfunctions[key_field][key_mode] = []
                                eigenfunctions[key_field][key_mode] = real_part[key_mode] + 1j*imag_part[key_mode]
                else:
                    eigenfunctions[key_field] = fields[key_field]

            if self.collect:
                merge_trees({'nmodes':nmodes, 'nfields':nfields, 'ntheta':ntheta},self.metadata)
            if not self.collect:
                eigenfunction = {'fields':list(fields.keys()), 'eigenfunction':eigenfunctions, 'nmodes':nmodes, 'nfields':nfields, 'ntheta':ntheta}
                return eigenfunction

    def read_width_spectrum(self,run_path=None,nky=None):
        gaussian_width = self.read_var_spectrum(run_path=run_path,file='out.tglf.width_spectrum',header=3,nky=nky,var='gaussian_width')
        if not self.collect:
            return gaussian_width

    def write_input(self,path=None,file='input.tglf',header=True,verbose=False,ignore_defaults=True,overwrite=True):
        # if unspecified, for convenience check for run path in metadata
        if not path and 'run_path' in self.metadata:
            path = self.metadata['run_path']
        
        header_list = [
            'input.tglf',
            'See https://gafusion.github.io/doc/tglf/tglf_table.html',
            '--------------------------------------------------------'
        ]
        if path:
            path_check = os.path.isdir(path)
            if not path_check:
                os.makedirs(path)
                if verbose:
                    print('Created TGLF run folder at: {}'.format(path))
            if file:
                if not os.path.isfile(path+file):
                    file_check = False
                else:
                    file_check = True
                f = open(path+file,"w+")

                if header:
                    for message in header_list:
                        f.write('# {}\n'.format(message))

                for key in self.input:
                    if key in self._input_defaults or key.split('_')[0] in self._species_vector:
                        write_key = True
                    if ignore_defaults:
                        if key in self._input_defaults and self.input[key] == self._input_defaults[key]:
                            write_key = False
                                
                    if write_key:
                        f.write("{}={}\n".format(key,self.input[key]))
                
                f.close()

                if not file_check and verbose:
                    print('Created TGLF input file at: {}'.format(path+file))
                elif file_check and overwrite and verbose:
                    print('Overwrote TGLF input file at: {}'.format(path+file))
                elif verbose:
                    print('Could not write TGLF input file! It already exists at: {}'.format(path+file))
        else:
            raise ValueError('No path for run was set!')

        return



    def _get_B_rat(self):
        '''
        Return B_rat, the ratio between the reference magnetic field of TGLF and IMAS
        '''
        # Nuber of points in the theta grid
        Nth=289
        # Flux surface geometry
        r0 = self.output['input_gen']['RMIN_LOC'] 
        R0 = self.output['input_gen']['RMAJ_LOC'] 
        Z0 = self.output['input_gen']['ZMAJ_LOC']
        k = self.output['input_gen']['KAPPA_LOC']
        d = self.output['input_gen']['DELTA_LOC']
        z = self.output['input_gen']['ZETA_LOC']
        dRmildr = self.output['input_gen']['DRMAJDX_LOC']
        dZmildr = self.output['input_gen']['DZMAJDX_LOC']
        drmildr = self.output['input_gen']['DRMINDX_LOC']
        sk = self.output['input_gen']['S_KAPPA_LOC']
        sd = self.output['input_gen']['S_DELTA_LOC']/np.sqrt(1.0-d**2)
        sz = self.output['input_gen']['S_ZETA_LOC']
        # define the fine theta grid on which the integrals are performed
        th=np.linspace(0,2*np.pi,Nth)
        
        arg_r = th + np.arcsin(d)*np.sin(th)
        darg_r = 1.0 + np.arcsin(d)*np.cos(th)
        arg_z = th + z*np.sin(2.0*th)
        darg_z = 1.0 + z*2.0*np.cos(2.0*th)
        
        R = R0 + r0*np.cos(arg_r)
        Z = Z0 + r0*k*np.sin(arg_z)
        
        dRdr = dRmildr + drmildr*np.cos(arg_r) - np.sin(arg_r)*sd*np.sin(th)
        dZdr = dZmildr + k*np.sin(arg_z)*(drmildr + sk) + k*np.cos(arg_z)*sz*np.sin(2.0*th)
        dRdth = -1*r0*np.sin(arg_r)*darg_r
        dZdth = k*r0*np.cos(arg_z)*darg_z
        r = 0.5*(max(R)-min(R))
        
        J_r = -R*(dRdr*dZdth - dRdth*dZdr)
        dldth = np.sqrt((dRdth)**2+(dZdth)**2)
        grad_r = R/J_r*dldth

        B_rat = R0/(2*np.pi*r)*np.trapz(dldth/(R*abs(grad_r)),th)

        return B_rat


            
            
    def _miller_to_mxh(self):
        '''
        Return a dict with Miller eXtended Harmonics (MXH) parametrisation parameters
        From the TGLF Miller input parameters 
        '''
        #Get the values of r, R and Z
        r, R, Z = miller2rz(self.output['input_gen']['RMIN_LOC'],self.output['input_gen']['RMAJ_LOC'],
                            self.output['input_gen']['ZMAJ_LOC'],self.output['input_gen']['KAPPA_LOC'],
                            self.output['input_gen']['DELTA_LOC'],self.output['input_gen']['ZETA_LOC'],
                            self.output['input_gen']['DRMAJDX_LOC']/self.output['input_gen']['DRMINDX_LOC'],
                            self.output['input_gen']['DZMAJDX_LOC']/self.output['input_gen']['DRMINDX_LOC'],
                            self.output['input_gen']['S_KAPPA_LOC'],self.output['input_gen']['S_DELTA_LOC'],
                            self.output['input_gen']['S_ZETA_LOC'],code='tglf',dr_frac=0.01,Nth=500)       
        #Get the mxh coeffs parameters       
        r_out,R0,dR0dr,Z0,dZ0dr,k,dkdr,c,dcdr,s,dsdr,R_out,Z_out,err_out = rz2mxh(R,Z,'imas',
                                                                                  self.output['input_gen']['RMIN_LOC'],
                                                                                  Nsh=10,doplots=False)

        #Check if there is an issue with the function
        if ((abs(r_out-self.output['input_gen']['RMIN_LOC']) > 1e-5)
            or (abs(R0-self.output['input_gen']['RMAJ_LOC']) > 1e-5)
            or (abs(Z0-self.output['input_gen']['ZMAJ_LOC']) > 1e-5)
            or (abs(dR0dr-self.output['input_gen']['DRMAJDX_LOC']) > 1e-5)):
            print('There is an issue with _miller_to_mxh(), different values are obtained for r0, R0 and Z0 from the input')
            print('Obtained values: r0 = ',r_out,', R0 = ',R0,', Z0 = ',Z0,', dR0dr = ',dR0dr)

        return { 'r':r_out, 'R0':R0  ,  'dR0dr':dR0dr  ,  'Z0':Z0  ,
                'dZ0dr':dZ0dr  ,  'k':k  ,  'dkdr':dkdr  ,  'c':c  ,
                'dcdr':dcdr  ,  's':s  ,  'dsdr':dsdr  ,  'R_out':R_out ,
                'Z_out':Z_out }

#     def _tglf_to_ids(self,verbose=True):
#         """Convert a TGLF object to IMAS gyrokinetic IDS/GKDB format.
#         """
#         # Check if the code has been launched, else, need to run the code first        
#         try:
#             self.output['input_gen']
#         except:
#             try:
#                 self.collect_output(eigenfunctions=True)
#             except:
#                 if verbose:
#                     print('There are no values in the TGLF object to convert to an ids \
#                           \n(input.tglf.gen file is missing and self.collect_output failed)')
#                     print('You need to launch the code first. \
#                           \nTry self.run() or read the help in fusionkit/extensions/Fusionkit.tglf_example')
#                 return
        
#         # Obtain the Miller eXtended Harmonics (MXH) parametrisation coefficients from the TGLF input values
#         mhx = self._miller_to_mxh()
        
#         # compute TGLF_ref/IMAS_ref ratio of reference quantities for TGLf to IMAS conversion
#         # reference charge ratio
#         q_tglf = 1.6021746e-19      #at: July 29, 2022 See: https://gafusion.github.io/doc/tglf.html
#         q_imas = 1.602176634e-19    #Last uptade: July 29, 2022 ; IMAS: 3.22.0
#         q_rat = q_tglf/q_imas
#         # reference mass ratio
#         m_tglf = 3.345e-27          #at: Last July 29, 2022 See: https://gafusion.github.io/doc/tglf.html
#         m_imas = 3.3435837724e-27   #Last uptade: July 29, 2022 ; IMAS: 3.22.0
#         m_rat = m_tglf/m_imas
#         # reference temperature ratio
#         T_rat = 1
#         # reference length ratio
#         L_rat =  1/self.output['input_gen']['RMAJ_LOC'] 
#         # reference magnetic field ratio
#         B_rat = self._get_B_rat()
#         # reference density ratio
#         n_rat = 1
#         # reference thermal velocity ratio
#         v_thrat = (1/np.sqrt(2))*(np.sqrt( T_rat/m_rat ))
#         # reference Larmor radius ratio
#         rho_rat = (m_rat*v_thrat)/(q_rat*B_rat)



#         self.ids = {}
        
#         self.ids['ids_properties'] = {}
#         self.ids['ids_properties']['comment'] = 'This gyrokinetic run performed with the code TGLF'
#         self.ids['ids_properties']['homogeneous_time'] = 2
#         self.ids['ids_properties']['provider'] = os.environ.get('USERNAME')
#         self.ids['ids_properties']['creation_date'] = self.metadata['run_date'] + '   ' + self.metadata['run_time']
#         self.ids['ids_properties']['version_put'] = {}
#         self.ids['ids_properties']['version_put']['data_dictionary'] = 'Last change occured on version: 3.32.0'
#         self.ids['ids_properties']['version_put']['access_layer'] = 'Last change occured on version: 3.32.0'
#         self.ids['ids_properties']['version_put']['access_layer_language'] = 'Python'
#         self.ids['ids_properties']['provenance'] = {}
#         self.ids['ids_properties']['provenance']['node'] = {}
#         self.ids['ids_properties']['provenance']['node'][1] = {}
#         self.ids['ids_properties']['provenance']['node'][1]['path'] = ''
#         self.ids['ids_properties']['provenance']['node'][1]['sources'] = ''
        
#         self.ids['tag'] = {}
#         self.ids['tag']['name'] = 'no tag'
#         self.ids['tag']['comment'] = 'no comment'
        
#         self.ids['normalizing_quantities'] = {}
#         self.ids['normalizing_quantities']['t_e'] = []
#         self.ids['normalizing_quantities']['n_e'] = []
#         self.ids['normalizing_quantities']['r'] = []
#         self.ids['normalizing_quantities']['b_field_tor'] = []

#         self.ids['flux_surface'] = {}
#         self.ids['flux_surface']['r_minor_norm'] = self.output['input_gen']['RMIN_LOC'] * L_rat
#         self.ids['flux_surface']['elongation'] = mhx['k']
#         self.ids['flux_surface']['delongation_dr_minor_norm'] = mhx['dkdr']
#         self.ids['flux_surface']['dgeometric_axis_r_dr_minor'] = mhx['dR0dr']
#         self.ids['flux_surface']['dgeometric_axis_z_dr_minor'] = mhx['dZ0dr']
#         self.ids['flux_surface']['q'] = self.output['input_gen']['SIGN_BT']*\
#             self.output['input_gen']['SIGN_IT']*self.output['input_gen']['Q_LOC']
#         self.ids['flux_surface']['magnetic_shear_r_minor'] = (self.output['input_gen']['RMIN_LOC']/\
#             abs(self.output['input_gen']['Q_LOC']))**2 * self.output['input_gen']['Q_PRIME_LOC']
#         self.ids['flux_surface']['pressure_gradient_norm'] = (-8*np.pi)*(B_rat**2/L_rat)*\
#             (self.output['input_gen']['RMIN_LOC']/(self.output['input_gen']['Q_LOC'])) *\
#             self.output['input_gen']['P_PRIME_LOC']        
#         self.ids['flux_surface']['ip_sign'] = self.output['input_gen']['SIGN_IT']
#         self.ids['flux_surface']['b_field_tor_sign'] = self.output['input_gen']['SIGN_BT']
#         self.ids['flux_surface']['shape_coefficients_c'] = mhx['c']
#         self.ids['flux_surface']['dc_dr_minor_norm'] = mhx['dcdr']
#         self.ids['flux_surface']['shape_coefficients_s'] = mhx['s']
#         self.ids['flux_surface']['ds_dr_minor_norm'] = mhx['dsdr']
        
#         self.ids['model'] = {}
#         self.ids['model']['include_centrifugal_effects'] = int(False)
#         self.ids['model']['include_a_field_parallel'] = int(self.output['input_gen']['USE_BPER'])
#         self.ids['model']['include_b_field_parallel'] = int(self.output['input_gen']['USE_BPAR'])
#         self.ids['model']['include_full_curvature_drift'] = int(not self.output['input_gen']['USE_MHD_RULE'])
#         self.ids['model']['collisions_pitch_only'] = int(True)
#         self.ids['model']['collisions_momentum_conservation'] = int(False)
#         self.ids['model']['collisions_energy_conservation'] = int(False)
#         self.ids['model']['collisions_finite_larmor_radius'] = int(False)
#         self.ids['model']['non_linear_run'] = []
#         self.ids['model']['time_interval_norm'] = []
        
#         self.ids['species_all'] = {}
#         self.ids['species_all']['beta_reference'] = self.output['input_gen']['BETAE']*(B_rat**2)/(1.5*n_rat*T_rat) 
        
#         number_of_species = self.output['input_gen']['NS']
        
        
#         Vpar = []
#         for i in range(number_of_species):
#             Vpar.append(self.output['input_gen']['VPAR_'+str(i+1)])
#         if all(x==Vpar[0] for x in Vpar):
#             self.ids['species_all']['velocity_tor_norm'] = self.output['input_gen']['SIGN_IT']*self.output['input_gen']['VPAR_1']*(v_thrat/L_rat)
#         else:
#             print('Warning: Wrong values for velocity_tor_norm. (All species are assumed to have\
#                   \n a purely toroidal flow with a COMMON toroidal angular frequency)')
#             self.ids['species_all']['velocity_tor_norm'] = []
        
#         self.ids['species_all']['zeff'] = self.output['input_gen']['ZEFF']


#         self.ids['species'] = []
#         for i in range(number_of_species):
#             dum = {}
#             dum['charge_norm'] = self.output['input_gen']['ZS_'+str(i+1)]*q_rat
#             dum['mass_norm'] = self.output['input_gen']['MASS_'+str(i+1)]*m_rat
#             dum['density_norm'] = self.output['input_gen']['AS_'+str(i+1)]*n_rat
#             dum['density_log_gradient_norm'] = self.output['input_gen']['RLNS_'+str(i+1)]/L_rat
#             dum['temperature_norm'] = self.output['input_gen']['TAUS_'+str(i+1)]*T_rat
#             dum['temperature_log_gradient_norm'] = self.output['input_gen']['RLTS_'+str(i+1)]/L_rat
#             dum['velocity_tor_gradient_norm'] = self.output['input_gen']['VPAR_SHEAR_'+str(i+1)]/(L_rat)*v_thrat*self.output['input_gen']['SIGN_IT']
#             self.ids['species'].append(dum)
            
#         self.ids['collisions'] = {}
#         # Find electron/main ion index
#         mass_max = 0.
#         for i in range(len(self.ids['species'])):
#             charge = self.ids['species'][i]['charge_norm']
#             mass = self.ids['species'][i]['mass_norm']
#             if charge < 0.:
#                 index_electron = i
#             if mass > mass_max:
#                 mass_max = mass
#                 index_main_ion = i
#         self.ids['collisions']['collisionality_norm'] = np.full((len(self.ids['species']),len(self.ids['species'])),0.)
#         self.ids['collisions']['collisionality_norm'][index_electron][index_main_ion] = self.output['input_gen']['XNUE']*(v_thrat/L_rat)
        
        
#         self.ids['wavevector'] = []
#         number_of_ky = len(self.output['ky'])
#         number_of_modes = self.output['input_gen']['NMODES']
#         Gq = get_metric(self.ids,np.linspace(0,2*np.pi,501))['Gq'][0]
#         gradr = get_metric(self.ids,np.linspace(0,2*np.pi,501))['grad_r'][0]
        
#         # compute theta_imas=f(theta_tglf)
#         try:
#             th_tglf = self.output['eigenfunctions'][list(self.output['eigenfunctions'].keys())[-1]]['theta']
#             nb_pol_trurns = (max(th_tglf)-min(th_tglf))/(2*np.pi)
#             R_dum = self.output['input_gen']['RMAJ_LOC'] + self.output['input_gen']['RMIN_LOC']* \
#                 np.cos(th_tglf + np.arcsin(self.output['input_gen']['DELTA_LOC'])*np.sin(th_tglf))
#             Z_dum = self.output['input_gen']['KAPPA_LOC']*self.output['input_gen']['RMIN_LOC']* \
#                 np.sin(th_tglf + self.output['input_gen']['ZETA_LOC']*np.sin(2*th_tglf))
#             th = np.arctan2((Z_dum),R_dum-self.output['input_gen']['RMAJ_LOC'])
#             dum = np.cumsum(np.concatenate(([False],abs(np.diff(th))>np.pi)))
#             dum = dum - ((nb_pol_trurns-1)/2. )
#             th_imas_of_tglf = th + 2*np.pi*dum
#             th_sorted = th_imas_of_tglf
#             fields_list = list(self.output['eigenfunctions'][list(self.output['eigenfunctions'].keys())[-1]].keys())
#             fields_list.remove('theta')
#         except:
#             th_sorted = []
#         # eigenfunctions normalisations
#         tglf_eigenfunctions = {}
#         imas_eigenfunctions = {}
#         imas_eigenfunctions['phi'], imas_eigenfunctions['Bper'], imas_eigenfunctions['Bpar'] = np.zeros((3,len(th_sorted)))
#         try:
#             for fields_ in fields_list:
#                 tglf_eigenfunctions[fields_] = self.output['eigenfunctions'][list(self.output['eigenfunctions'].keys())[-1]][fields_][1]
#                 if fields_ == 'phi':
#                     imas_eigenfunctions[fields_] = tglf_eigenfunctions[fields_] * ((T_rat*rho_rat )/(q_rat*L_rat))
#                 elif fields_ == 'Bper':
#                     imas_eigenfunctions[fields_] = tglf_eigenfunctions[fields_] * ((B_rat*rho_rat**2 )/(L_rat))
#                 elif fields_ == 'Bpar':
#                     imas_eigenfunctions[fields_] = tglf_eigenfunctions[fields_] * ((B_rat*rho_rat )/(L_rat))
#                 if self.output['input_gen']['SIGN_BT']*self.output['input_gen']['SIGN_IT']==1: # take complex conjugate if zeta and y in opposite directions
#                   imas_eigenfunctions[fields_]=np.conj(imas_eigenfunctions[fields_])
            
#             # rotate and normalise eigenfunctions
#             # rotate_factor=exp(i*alpha) in IMAS documentation
#             th_fine = np.linspace(th_sorted[0],th_sorted[-1],len(th_sorted)*10)
#             f_re=interp1d(th_sorted,np.real(imas_eigenfunctions['phi']),kind='cubic')
#             f_im=interp1d(th_sorted,np.imag(imas_eigenfunctions['phi']),kind='cubic')
#             Imax=np.argmax(np.abs(f_re(th_fine) + 1j*f_im(th_fine)))
#             rotate_factor = f_re(th_fine)[Imax] + 1j*f_im(th_fine)[Imax]
#             rotate_factor = abs(rotate_factor)/rotate_factor
#             # mode amplitude 
#             # amp_rat = np.sqrt(2*np.pi)/(np.sqrt(np.trapz(np.abs(imas_eigenfunctions['phi'])**2 
#             #                                               + np.abs(imas_eigenfunctions['Bper'])**2 
#             #                                               + np.abs(imas_eigenfunctions['Bpar'])**2,th_sorted)))
#             # amp_rat = np.sqrt(np.trapz(np.abs(tglf_eigenfunctions['phi'])**2,th_tglf)) * np.sqrt(self.output['gaussian_width'][-1])/(np.sqrt(np.trapz(np.abs(imas_eigenfunctions['phi'])**2 
#             #                                               + np.abs(imas_eigenfunctions['Bper'])**2 
#             #                                               + np.abs(imas_eigenfunctions['Bpar'])**2,th_sorted)))


#             # amp_rat = np.sqrt(np.trapz(np.abs(tglf_eigenfunctions['phi'])**2,th_tglf)) * np.sqrt(2)/(np.sqrt(np.trapz(np.abs(imas_eigenfunctions['phi'])**2 
#             #                                               + np.abs(imas_eigenfunctions['Bper'])**2 
#             #                                               + np.abs(imas_eigenfunctions['Bpar'])**2,th_sorted)))
            
                                                                              
#             amp_rat = np.sqrt(2*np.pi)/(np.sqrt(np.trapz(np.abs(imas_eigenfunctions['phi'])**2 
#                                                           + np.abs(imas_eigenfunctions['Bper'])**2 
#                                                           + np.abs(imas_eigenfunctions['Bpar'])**2,th_tglf)))
            
            
            
#         except:
#             if (not self.output['input_gen']['USE_TRANSPORT_MODEL']) :
#                 if verbose:
#                     print('Warning, no wavefunctions to read for the file: ' + self.metadata['run_path'].split('/')[-2])
#             rotate_factor = 1.
#             amp_rat = 1.

#         for i in range(number_of_ky):
#             wave = {}
#             wave['radial_component_norm'] = self.output['input_gen']['KX0_LOC']*self.output['ky'][i]*(1/rho_rat)*abs(gradr)
#             wave['binormal_component_norm'] = self.output['ky'][i]*abs(Gq)*(1/rho_rat) 
#             wave['eigenmode'] = []
#             for j in range(number_of_modes):
#                 mode = {}
#                 mode['growth_rate_norm'] = self.output['eigenvalues']['gamma'][j+1][i]*(v_thrat/L_rat)
#                 mode['frequency_norm'] = self.output['eigenvalues']['omega'][j+1][i]*(v_thrat/L_rat)
#                 mode['poloidal_angle'] = th_sorted
#                 mode['growth_rate_tolerance'] = []
#                 mode['phi_potential_perturbed_weight'] = []
#                 mode['phi_potential_perturbed_parity'] = []
#                 mode['a_field_parallel_perturbed_weight'] = []
#                 mode['a_field_parallel_perturbed_parity'] = []
#                 mode['b_field_parallel_perturbed_weight'] = []
#                 mode['b_field_parallel_perturbed_parity'] = []
#                 mode['phi_potential_perturbed_norm'] = imas_eigenfunctions['phi']*rotate_factor*amp_rat
#                 mode['a_field_parallel_perturbed_norm'] = -1*imas_eigenfunctions['Bper']*rotate_factor*amp_rat
#                 mode['b_field_parallel_perturbed_norm'] = -1*imas_eigenfunctions['Bpar']*rotate_factor*amp_rat
#                 mode['time_norm'] = []
                
#                 mode['fluxes_moments'] = []
#                 for k in range(number_of_species):
#                     particle_flux_factor =  2.*((n_rat*v_thrat*rho_rat**2)/(L_rat**2)) * amp_rat**2 
#                     energy_flux_factor = 2.*((n_rat*v_thrat*T_rat*rho_rat**2)/(L_rat**2)) * amp_rat**2 
#                     momentum_flux_factor = 2.*((m_rat*n_rat*v_thrat**2*rho_rat**2)/(L_rat**2)) * amp_rat**2 
                    
#                     if self.output['input_gen']['AS_'+str(k+1)] != 0.0:
#                         particle_flux_factor /= self.output['input_gen']['AS_'+str(k+1)]
#                         energy_flux_factor /= self.output['input_gen']['AS_'+str(k+1)]
#                         momentum_flux_factor /= self.output['input_gen']['AS_'+str(k+1)]
                    
                    
                    
#                     species = {}
                    
                    
                    
#                     species['fluxes_norm_gyrocenter_rotating_frame'] = {}
                    
#                     species['fluxes_norm_gyrocenter_rotating_frame']['particles_phi_potential'] = \
#                         particle_flux_factor * self.output['species'][k+1]['QL_weights']['Gamma']['phi'][j+1][-1]
                    
#                     if self.output['input_gen']['USE_BPER']:
#                         species['fluxes_norm_gyrocenter_rotating_frame']['particles_a_field_parallel'] = \
#                             particle_flux_factor * self.output['species'][k+1]['QL_weights']['Gamma']['Bper'][j+1][-1]
#                     else:
#                         species['fluxes_norm_gyrocenter_rotating_frame']['particles_a_field_parallel'] = []
                    
#                     if self.output['input_gen']['USE_BPAR']:
#                         species['fluxes_norm_gyrocenter_rotating_frame']['particles_b_field_parallel'] = \
#                             particle_flux_factor * self.output['species'][k+1]['QL_weights']['Gamma']['Bpar'][j+1][-1]
#                     else:
#                         species['fluxes_norm_gyrocenter_rotating_frame']['particles_b_field_parallel'] = []
                    
                    
                    
#                     species['fluxes_norm_gyrocenter_rotating_frame']['energy_phi_potential'] = energy_flux_factor *\
#                         self.output['species'][k+1]['QL_weights']['Q']['phi'][j+1][-1]
                    
#                     if self.output['input_gen']['USE_BPER']:
#                         species['fluxes_norm_gyrocenter_rotating_frame']['energy_a_field_parallel'] = energy_flux_factor *\
#                             self.output['species'][k+1]['QL_weights']['Q']['Bper'][j+1][-1]
#                     else:
#                         species['fluxes_norm_gyrocenter_rotating_frame']['energy_a_field_parallel'] = []
                        
#                     if self.output['input_gen']['USE_BPAR']:
#                         species['fluxes_norm_gyrocenter_rotating_frame']['energy_b_field_parallel'] = energy_flux_factor *\
#                             self.output['species'][k+1]['QL_weights']['Q']['Bpar'][j+1][-1]
#                     else:
#                         species['fluxes_norm_gyrocenter_rotating_frame']['energy_b_field_parallel'] = []
                    
                    
                    
#                     species['fluxes_norm_gyrocenter_rotating_frame']['momentum_tor_parallel_phi_potential'] = momentum_flux_factor *\
#                         self.output['species'][k+1]['QL_weights']['Pi_par']['phi'][j+1][-1]
                    
#                     if self.output['input_gen']['USE_BPER']:
#                         species['fluxes_norm_gyrocenter_rotating_frame']['momentum_tor_parallel_a_field_parallel'] = momentum_flux_factor *\
#                             self.output['species'][k+1]['QL_weights']['Pi_par']['Bper'][j+1][-1]
#                     else:
#                         species['fluxes_norm_gyrocenter_rotating_frame']['momentum_tor_parallel_a_field_parallel'] = []
                    
#                     if self.output['input_gen']['USE_BPAR']:
#                         species['fluxes_norm_gyrocenter_rotating_frame']['momentum_tor_parallel_b_field_parallel'] = momentum_flux_factor *\
#                             self.output['species'][k+1]['QL_weights']['Pi_par']['Bpar'][j+1][-1]
#                     else:
#                         species['fluxes_norm_gyrocenter_rotating_frame']['momentum_tor_parallel_b_field_parallel'] = []
                    
                    
                    
#                     species['fluxes_norm_gyrocenter_rotating_frame']['momentum_tor_perpendicular_phi_potential'] = momentum_flux_factor *\
#                         (self.output['species'][k+1]['QL_weights']['Pi_tor']['phi'][j+1][-1]-self.output['species'][k+1]['QL_weights']['Pi_par']['phi'][j+1][-1])
                    
#                     if self.output['input_gen']['USE_BPER']:
#                         species['fluxes_norm_gyrocenter_rotating_frame']['momentum_tor_perpendicular_a_field_parallel'] = momentum_flux_factor *\
#                             (self.output['species'][k+1]['QL_weights']['Pi_tor']['Bper'][j+1][-1]-self.output['species'][k+1]['QL_weights']['Pi_par']['Bper'][j+1][-1])
#                     else:
#                         species['fluxes_norm_gyrocenter_rotating_frame']['momentum_tor_perpendicular_a_field_parallel'] = []
                    
#                     if self.output['input_gen']['USE_BPAR']:
#                         species['fluxes_norm_gyrocenter_rotating_frame']['momentum_tor_perpendicular_b_field_parallel'] = momentum_flux_factor *\
#                             (self.output['species'][k+1]['QL_weights']['Pi_tor']['Bpar'][j+1][-1]-self.output['species'][k+1]['QL_weights']['Pi_par']['Bpar'][j+1][-1])
#                     else:
#                         species['fluxes_norm_gyrocenter_rotating_frame']['momentum_tor_perpendicular_b_field_parallel'] = []
                    
                    
#                     mode['fluxes_moments'].append(species)
#                 wave['eigenmode'].append(mode)
#             self.ids['wavevector'].append(wave)
        
#         self.ids['fluxes_integrated_norm'] = []
#         fluxes={'name': ('particles','momentum_parallel','heat'),
#                 'field': ('phi','apar','bpar'),
#                 'is_field': (True,self.output['input_gen']['USE_BPER'],self.output['input_gen']['USE_BPAR']),
#                 'imas_name': ('particles_phi_potential','momentum_tor_parallel_phi_potential','energy_phi_potential',
#                               'particles_a_field_parallel','momentum_tor_parallel_a_field_parallel','energy_a_field_parallel',
#                               'particles_b_field_parallel','momentum_tor_parallel_b_field_parallel','energy_b_field_parallel'),
#                 'common_norm_fac': (rho_rat/L_rat)**2*v_thrat*n_rat,
#                 'norm_fac': (np.ones(number_of_species),m_rat*L_rat*v_thrat*np.ones(number_of_species),T_rat*np.ones(number_of_species))
#                }
# #       ATTENTION!!!! POUR L INSTANT JE MET LA SOMME DE TOUT DANS phi bper et bpar
#         tglf_flux_name = ['Gamma','Pi','Q']
#         tglf_field_name = ['phi','Bper','Bpar']
#         for ii_sp in range(number_of_species):
#           dum={}
#           for ii, field in enumerate(tglf_field_name):
#             for jj, name in enumerate(tglf_flux_name):
#               dum[fluxes['imas_name'][ii*3+jj]]=[]
#               if fluxes['is_field'][ii]:
#                   tglf_flux = self.output['species'][ii_sp+1]['fluxes'][name]['total']
#                   #print('tglf_flux = ',tglf_flux)
#                   #print(fluxes['imas_name'][ii*3+jj])
#                   dum[fluxes['imas_name'][ii*3+jj]] = tglf_flux*fluxes['common_norm_fac']*fluxes['norm_fac'][jj][ii_sp]
#               else:
#                 print("Missing data, can not convert lab frame "+field+" "+name+" flux to IMAS for species "+str(ii_sp))
#           self.ids['fluxes_integrated_norm'].append(dum)
            
        
        
#         self.ids['code'] = {}
#         self.ids['code']['name'] = 'TGLF'
#         self.ids['code']['commit'] = self.metadata['version']
#         self.ids['code']['version'] = self.metadata['framework_version']
#         self.ids['code']['repository'] = 'https://gafusion.github.io/doc/tglf.html'
#         self.ids['code']['parameters'] = []
#         self.ids['code']['output_flag'] = []
#         self.ids['code']['library'] = []
        
#         self.ids['time'] = []
        
#         return
        
        

    def _tglf_to_idspy(self,verbose=True):
        """Convert a TGLF object to IMAS gyrokinetic IDS/GKDB format.
        """
        
        # initialise an empty GK IDS structure
        self.ids=gkids.Gyrokinetics()
        idspy.fill_default_values_ids(self.ids)
    
        # Check if the code has been launched, else, need to run the code first        
        try:
            self.output['input_gen']
        except:
            try:
                self.collect_output(eigenfunctions=True)
            except:
                if verbose:
                    print('There are no values in the TGLF object to convert to an ids \
                          \n(input.tglf.gen file is missing and self.collect_output failed)')
                    print('You need to launch the code first. \
                          \nTry self.run() or read the help in fusionkit/extensions/Fusionkit.tglf_example')
                return
        
        # Obtain the Miller eXtended Harmonics (MXH) parametrisation coefficients from the TGLF input values
        mhx = self._miller_to_mxh()
        
        # compute TGLF_ref/IMAS_ref ratio of reference quantities for TGLf to IMAS conversion
        # reference charge ratio
        q_tglf = 1.6021746e-19      #at: July 29, 2022 See: https://gafusion.github.io/doc/tglf.html
        q_imas = 1.602176634e-19    #Last uptade: July 29, 2022 ; IMAS: 3.22.0
        q_rat = q_tglf/q_imas
        # reference mass ratio
        m_tglf = 3.345e-27          #at: Last July 29, 2022 See: https://gafusion.github.io/doc/tglf.html
        m_imas = 3.3435837724e-27   #Last uptade: July 29, 2022 ; IMAS: 3.22.0
        m_rat = m_tglf/m_imas
        # reference temperature ratio
        T_rat = 1
        # reference length ratio
        L_rat =  1/self.output['input_gen']['RMAJ_LOC'] 
        # reference magnetic field ratio
        B_rat = self._get_B_rat()
        # reference density ratio
        n_rat = 1
        # reference thermal velocity ratio
        v_thrat = (1/np.sqrt(2))*(np.sqrt( T_rat/m_rat ))
        # reference Larmor radius ratio
        rho_rat = (m_rat*v_thrat)/(q_rat*B_rat)



        self.ids.ids_properties.comment = 'This gyrokinetic run performed with the code TGLF'
        self.ids.ids_properties.homogeneous_time = 2
        self.ids.ids_properties.provider = os.environ.get('USERNAME')
        self.ids.ids_properties.creation_date = self.metadata['run_date'] + '   ' + self.metadata['run_time']
        #self.ids.ids_properties.version_put.data_dictionary = 'Last change occured on version: 3.32.0'
        #self.ids.ids_properties.version_put.access_layer = 'Last change occured on version: 3.32.0'
        #self.ids.ids_properties.version_put.access_layer_language = 'Python'


        self.ids.flux_surface.r_minor_norm = self.output['input_gen']['RMIN_LOC'] * L_rat
        self.ids.flux_surface.elongation = mhx['k']
        self.ids.flux_surface.delongation_dr_minor_norm = mhx['dkdr']
        self.ids.flux_surface.dgeometric_axis_r_dr_minor = mhx['dR0dr']
        self.ids.flux_surface.dgeometric_axis_z_dr_minor = mhx['dZ0dr']
        self.ids.flux_surface.q = self.output['input_gen']['SIGN_BT']*\
            self.output['input_gen']['SIGN_IT']*self.output['input_gen']['Q_LOC']
        self.ids.flux_surface.magnetic_shear_r_minor = (self.output['input_gen']['RMIN_LOC']/\
            abs(self.output['input_gen']['Q_LOC']))**2 * self.output['input_gen']['Q_PRIME_LOC']
        self.ids.flux_surface.pressure_gradient_norm = (-8*np.pi)*(B_rat**2/L_rat)*\
            (self.output['input_gen']['RMIN_LOC']/(self.output['input_gen']['Q_LOC'])) *\
            self.output['input_gen']['P_PRIME_LOC']        
        self.ids.flux_surface.ip_sign = self.output['input_gen']['SIGN_IT']
        self.ids.flux_surface.b_field_tor_sign = self.output['input_gen']['SIGN_BT']
        self.ids.flux_surface.shape_coefficients_c = mhx['c']
        self.ids.flux_surface.dc_dr_minor_norm = mhx['dcdr']
        self.ids.flux_surface.shape_coefficients_s = mhx['s']
        self.ids.flux_surface.ds_dr_minor_norm = mhx['dsdr']
        
        self.ids.model.include_centrifugal_effects = False
        self.ids.model.include_a_field_parallel = self.output['input_gen']['USE_BPER']
        self.ids.model.include_b_field_parallel = self.output['input_gen']['USE_BPAR']
        self.ids.model.include_full_curvature_drift = not self.output['input_gen']['USE_MHD_RULE']
        self.ids.model.collisions_pitch_only = True
        self.ids.model.collisions_momentum_conservation = False
        self.ids.model.collisions_energy_conservation = False
        self.ids.model.collisions_finite_larmor_radius = False
        self.ids.model.non_linear_run = 0
        
        self.ids.species_all.beta_reference = self.output['input_gen']['BETAE']*(B_rat**2)/(1.5*n_rat*T_rat) 
        
        number_of_species = self.output['input_gen']['NS']
        
        
        Vpar = []
        for i in range(number_of_species):
            Vpar.append(self.output['input_gen']['VPAR_'+str(i+1)])
        if all(x==Vpar[0] for x in Vpar):
            self.ids.species_all.velocity_tor_norm = self.output['input_gen']['SIGN_IT']*self.output['input_gen']['VPAR_1']*(v_thrat/L_rat)
        else:
            print('Warning: Wrong values for velocity_tor_norm. (All species are assumed to have\
                  \n a purely toroidal flow with a COMMON toroidal angular frequency)')
            self.ids.species_all.velocity_tor_norm = []
        
        self.ids.species_all.zeff = self.output['input_gen']['ZEFF']


        for i in range(number_of_species):
            dum=gkids.Species()
            dum.charge_norm = self.output['input_gen']['ZS_'+str(i+1)]*q_rat
            dum.mass_norm = self.output['input_gen']['MASS_'+str(i+1)]*m_rat
            dum.density_norm = self.output['input_gen']['AS_'+str(i+1)]*n_rat
            dum.density_log_gradient_norm = self.output['input_gen']['RLNS_'+str(i+1)]/L_rat
            dum.temperature_norm = self.output['input_gen']['TAUS_'+str(i+1)]*T_rat
            dum.temperature_log_gradient_norm = self.output['input_gen']['RLTS_'+str(i+1)]/L_rat
            dum.velocity_tor_gradient_norm = self.output['input_gen']['VPAR_SHEAR_'+str(i+1)]/(L_rat)*v_thrat*self.output['input_gen']['SIGN_IT']
            self.ids.species.append(dum)
            
        # Find electron/main ion index
        density_max = 0.
        for i in range(len(self.ids.species)):
            charge = self.ids.species[i].charge_norm
            density = self.ids.species[i].density_norm
            if charge < 0.:
                index_electron = i
            if (density > density_max) and (charge > 0.):
                density_max = density
                index_main_ion = i
        self.ids.collisions.collisionality_norm = np.full((number_of_species,number_of_species),0.0)
        self.ids.collisions.collisionality_norm[index_electron][index_main_ion] = self.output['input_gen']['XNUE']*(v_thrat/L_rat)


        number_of_ky = len(self.output['ky'])
        number_of_modes = self.output['input_gen']['NMODES']
        
        metric = get_metric_idspy(self.ids,np.linspace(0,2*np.pi,501))
        Gq = metric['Gq'][0]
        gradr = metric['grad_r'][0]
        
        # compute theta_imas=f(theta_tglf)
        try:
            th_tglf = self.output['eigenfunctions'][list(self.output['eigenfunctions'].keys())[-1]]['theta']
            nb_pol_trurns = (max(th_tglf)-min(th_tglf))/(2*np.pi)
            R_dum = self.output['input_gen']['RMAJ_LOC'] + self.output['input_gen']['RMIN_LOC']* \
                np.cos(th_tglf + np.arcsin(self.output['input_gen']['DELTA_LOC'])*np.sin(th_tglf))
            Z_dum = self.output['input_gen']['KAPPA_LOC']*self.output['input_gen']['RMIN_LOC']* \
                np.sin(th_tglf + self.output['input_gen']['ZETA_LOC']*np.sin(2*th_tglf))
            th = np.arctan2((Z_dum),R_dum-self.output['input_gen']['RMAJ_LOC'])
            dum = np.cumsum(np.concatenate(([False],abs(np.diff(th))>np.pi)))
            dum = dum - ((nb_pol_trurns-1)/2. )
            th_imas_of_tglf = th + 2*np.pi*dum
            th_sorted = th_imas_of_tglf
            fields_list = list(self.output['eigenfunctions'][list(self.output['eigenfunctions'].keys())[-1]].keys())
            fields_list.remove('theta')
        except:
            th_sorted = []
        # eigenfunctions normalisations
        imas_eigenfunctions_mode=[]
        rotate_factor_mode=[]
        amp_rat_mode=[]
        for i in range(number_of_modes):
            tglf_eigenfunctions = {}
            imas_eigenfunctions = {}
            imas_eigenfunctions['phi'], imas_eigenfunctions['Bper'], imas_eigenfunctions['Bpar'] = np.zeros((3,len(th_sorted)))
            try:
                for fields_ in fields_list:
                    tglf_eigenfunctions[fields_] = self.output['eigenfunctions'][list(self.output['eigenfunctions'].keys())[-1]][fields_][i+1]
                    if fields_ == 'phi':
                        imas_eigenfunctions[fields_] = tglf_eigenfunctions[fields_] * ((T_rat*rho_rat )/(q_rat*L_rat))
                    elif fields_ == 'Bper':
                        imas_eigenfunctions[fields_] = tglf_eigenfunctions[fields_] * ((B_rat*rho_rat**2 )/(L_rat))
                    elif fields_ == 'Bpar':
                        imas_eigenfunctions[fields_] = tglf_eigenfunctions[fields_] * ((B_rat*rho_rat )/(L_rat))
                    if self.output['input_gen']['SIGN_BT']*self.output['input_gen']['SIGN_IT']==1: # take complex conjugate if zeta and y in opposite directions
                      imas_eigenfunctions[fields_]=np.conj(imas_eigenfunctions[fields_])
                
                # rotate and normalise eigenfunctions
                # rotate_factor=exp(i*alpha) in IMAS documentation
                th_fine = np.linspace(th_sorted[0],th_sorted[-1],len(th_sorted)*10)
                f_re=interp1d(th_sorted,np.real(imas_eigenfunctions['phi']),kind='cubic')
                f_im=interp1d(th_sorted,np.imag(imas_eigenfunctions['phi']),kind='cubic')
                Imax=np.argmax(np.abs(f_re(th_fine) + 1j*f_im(th_fine)))
                rotate_factor = f_re(th_fine)[Imax] + 1j*f_im(th_fine)[Imax]
                rotate_factor = abs(rotate_factor)/rotate_factor
                # mode amplitude 
    
                # amp_rat = np.sqrt(np.trapz(np.abs(tglf_eigenfunctions['phi'])**2,th_tglf)) * np.sqrt(2)/(np.sqrt(np.trapz(np.abs(imas_eigenfunctions['phi'])**2 
                #                                               + np.abs(imas_eigenfunctions['Bper'])**2 
                #                                               + np.abs(imas_eigenfunctions['Bpar'])**2,th_sorted)))
                
                                                                                  
                amp_rat = np.sqrt(2*np.pi)/(np.sqrt(np.trapz(np.abs(imas_eigenfunctions['phi'])**2 
                                                              + np.abs(imas_eigenfunctions['Bper'])**2 
                                                              + np.abs(imas_eigenfunctions['Bpar'])**2,th_tglf)))
                
                
                
            except:
                if (not self.output['input_gen']['USE_TRANSPORT_MODEL']) :
                    if verbose:
                        print('Warning, no wavefunctions to read for the file: ' + self.metadata['run_path'].split('/')[-2])
                rotate_factor = 1.
                amp_rat = 1.
            
            imas_eigenfunctions_mode.append(imas_eigenfunctions)
            rotate_factor_mode.append(rotate_factor)
            amp_rat_mode.append(amp_rat)

        for i in range(number_of_ky):
            wave = gkids.Wavevector()
            wave.radial_component_norm = self.output['input_gen']['KX0_LOC']*self.output['ky'][i]*(1/rho_rat)*abs(gradr)
            wave.binormal_component_norm = self.output['ky'][i]*abs(Gq)*(1/rho_rat) 
            for j in range(number_of_modes):
                mode = gkids.Eigenmode()
                mode.growth_rate_norm = self.output['eigenvalues']['gamma'][j+1][i]*(v_thrat/L_rat)
                mode.frequency_norm = self.output['eigenvalues']['omega'][j+1][i]*(v_thrat/L_rat)
                mode.poloidal_angle = th_sorted
                mode.phi_potential_perturbed_norm = imas_eigenfunctions_mode[j]['phi']*rotate_factor_mode[j]*amp_rat_mode[j]
                mode.a_field_parallel_perturbed_norm = -1*imas_eigenfunctions_mode[j]['Bper']*rotate_factor_mode[j]*amp_rat_mode[j]
                mode.b_field_parallel_perturbed_norm = -1*imas_eigenfunctions_mode[j]['Bpar']*rotate_factor_mode[j]*amp_rat_mode[j]

                for k in range(number_of_species):
                    particle_flux_factor =  2.*((n_rat*v_thrat*rho_rat**2)/(L_rat**2)) * amp_rat_mode[j]**2
                    energy_flux_factor = 2.*((n_rat*v_thrat*T_rat*rho_rat**2)/(L_rat**2)) * amp_rat_mode[j]**2
                    momentum_flux_factor = 2.*((m_rat*n_rat*v_thrat**2*rho_rat**2)/(L_rat**2)) * amp_rat_mode[j]**2
                    if self.output['input_gen']['AS_'+str(k+1)] != 0.0:
                        particle_flux_factor /= self.output['input_gen']['AS_'+str(k+1)]
                        energy_flux_factor /= self.output['input_gen']['AS_'+str(k+1)]
                        momentum_flux_factor /= self.output['input_gen']['AS_'+str(k+1)]
                        
                    dum = gkids.FluxesMoments()
                    mode.fluxes_moments.append(dum)
                    
                    mode.fluxes_moments[k].fluxes_norm_gyrocenter_rotating_frame = gkids.Fluxes()
                    
                    
                
                    mode.fluxes_moments[k].fluxes_norm_gyrocenter_rotating_frame.particles_phi_potential = \
                        particle_flux_factor * self.output['species'][k+1]['QL_weights']['Gamma']['phi'][j+1][i]
                    
                    if self.output['input_gen']['USE_BPER']:
                        mode.fluxes_moments[k].fluxes_norm_gyrocenter_rotating_frame.particles_a_field_parallel = \
                            particle_flux_factor * self.output['species'][k+1]['QL_weights']['Gamma']['Bper'][j+1][i]
                    else:
                        mode.fluxes_moments[k].fluxes_norm_gyrocenter_rotating_frame.particles_a_field_parallel = []
                    
                    if self.output['input_gen']['USE_BPAR']:
                        mode.fluxes_moments[k].fluxes_norm_gyrocenter_rotating_frame.particles_b_field_parallel = \
                            particle_flux_factor * self.output['species'][k+1]['QL_weights']['Gamma']['Bpar'][j+1][i]
                    else:
                        mode.fluxes_moments[k].fluxes_norm_gyrocenter_rotating_frame.particles_b_field_parallel = []
                    
                    
                    
                    mode.fluxes_moments[k].fluxes_norm_gyrocenter_rotating_frame.energy_phi_potential = energy_flux_factor *\
                        self.output['species'][k+1]['QL_weights']['Q']['phi'][j+1][i]
                    
                    if self.output['input_gen']['USE_BPER']:
                        mode.fluxes_moments[k].fluxes_norm_gyrocenter_rotating_frame.energy_a_field_parallel = energy_flux_factor *\
                            self.output['species'][k+1]['QL_weights']['Q']['Bper'][j+1][i]
                    else:
                        mode.fluxes_moments[k].fluxes_norm_gyrocenter_rotating_frame.energy_a_field_parallel = []
                        
                    if self.output['input_gen']['USE_BPAR']:
                        mode.fluxes_moments[k].fluxes_norm_gyrocenter_rotating_frame.energy_b_field_parallel = energy_flux_factor *\
                            self.output['species'][k+1]['QL_weights']['Q']['Bpar'][j+1][i]
                    else:
                        mode.fluxes_moments[k].fluxes_norm_gyrocenter_rotating_frame.energy_b_field_parallel = []
                    
                    
                    
                    mode.fluxes_moments[k].fluxes_norm_gyrocenter_rotating_frame.momentum_tor_parallel_phi_potential = momentum_flux_factor *\
                        self.output['species'][k+1]['QL_weights']['Pi_par']['phi'][j+1][i]
                    
                    if self.output['input_gen']['USE_BPER']:
                        mode.fluxes_moments[k].fluxes_norm_gyrocenter_rotating_frame.momentum_tor_parallel_a_field_parallel = momentum_flux_factor *\
                            self.output['species'][k+1]['QL_weights']['Pi_par']['Bper'][j+1][i]
                    else:
                        mode.fluxes_moments[k].fluxes_norm_gyrocenter_rotating_frame.momentum_tor_parallel_a_field_parallel = []
                    
                    if self.output['input_gen']['USE_BPAR']:
                        mode.fluxes_moments[k].fluxes_norm_gyrocenter_rotating_frame.momentum_tor_parallel_b_field_parallel = momentum_flux_factor *\
                            self.output['species'][k+1]['QL_weights']['Pi_par']['Bpar'][j+1][i]
                    else:
                        mode.fluxes_moments[k].fluxes_norm_gyrocenter_rotating_frame.momentum_tor_parallel_b_field_parallel = []
                    
                    
                    
                    mode.fluxes_moments[k].fluxes_norm_gyrocenter_rotating_frame.momentum_tor_perpendicular_phi_potential = momentum_flux_factor *\
                        (self.output['species'][k+1]['QL_weights']['Pi_tor']['phi'][j+1][i]-self.output['species'][k+1]['QL_weights']['Pi_par']['phi'][j+1][i])
                    
                    if self.output['input_gen']['USE_BPER']:
                        mode.fluxes_moments[k].fluxes_norm_gyrocenter_rotating_frame.momentum_tor_perpendicular_a_field_parallel = momentum_flux_factor *\
                            (self.output['species'][k+1]['QL_weights']['Pi_tor']['Bper'][j+1][i]-self.output['species'][k+1]['QL_weights']['Pi_par']['Bper'][j+1][i])
                    else:
                        mode.fluxes_moments[k].fluxes_norm_gyrocenter_rotating_frame.momentum_tor_perpendicular_a_field_parallel = []
                    
                    if self.output['input_gen']['USE_BPAR']:
                        mode.fluxes_moments[k].fluxes_norm_gyrocenter_rotating_frame.momentum_tor_perpendicular_b_field_parallel = momentum_flux_factor *\
                            (self.output['species'][k+1]['QL_weights']['Pi_tor']['Bpar'][j+1][i]-self.output['species'][k+1]['QL_weights']['Pi_par']['Bpar'][j+1][i])
                    else:
                        mode.fluxes_moments[k].fluxes_norm_gyrocenter_rotating_frame.momentum_tor_perpendicular_b_field_parallel = []
                    
                    
                wave.eigenmode.append(mode)
            self.ids.wavevector.append(wave)
        
        fluxes={'name': ('particles','momentum_parallel','heat'),
                'field': ('phi','apar','bpar'),
                'is_field': (True,self.output['input_gen']['USE_BPER'],self.output['input_gen']['USE_BPAR']),
                'imas_name': ('particles_phi_potential','momentum_tor_parallel_phi_potential','energy_phi_potential',
                              'particles_a_field_parallel','momentum_tor_parallel_a_field_parallel','energy_a_field_parallel',
                              'particles_b_field_parallel','momentum_tor_parallel_b_field_parallel','energy_b_field_parallel'),
                'common_norm_fac': (rho_rat/L_rat)**2*v_thrat*n_rat,
                'norm_fac': (np.ones(number_of_species),m_rat*L_rat*v_thrat*np.ones(number_of_species),T_rat*np.ones(number_of_species))
               }
#       ATTENTION!!!! POUR L INSTANT JE MET LA SOMME DE TOUT DANS phi
        n_name=len(fluxes['name'])
        tglf_flux_name = ['Gamma','Pi','Q']
        tglf_field_name = ['phi','Bper','Bpar']
        for ii_sp in range(number_of_species):
          dum = gkids.Fluxes()
          if self.output['input_gen']['AS_'+str(ii_sp+1)] != 0.0:
              density_norm_factor = 1.0#self.output['input_gen']['AS_'+str(ii_sp+1)]
          else:
              density_norm_factor = 1.0
          for ii, field in enumerate(tglf_field_name):
            for jj, name in enumerate(tglf_flux_name):
              if fluxes['is_field'][ii]:
                  tglf_flux = self.output['species'][ii_sp+1]['fluxes'][name]['total']
                  #setattr(dum,fluxes['imas_name'][ii*n_name+jj],tglf_flux*fluxes['common_norm_fac']*fluxes['norm_fac'][jj][ii_sp])
                  if ii == 0.: # Everything is stored in phi
                      setattr(dum,fluxes['imas_name'][jj],tglf_flux*fluxes['common_norm_fac']*fluxes['norm_fac'][jj][ii_sp]/density_norm_factor)
                  else:
                      setattr(dum,fluxes['imas_name'][ii*n_name+jj],0.0)
              else:
                print("Missing data, can not convert lab frame "+field+" "+name+" flux to IMAS for species "+str(ii_sp))
          self.ids.fluxes_integrated_norm.append(dum)
            
              
        self.ids.code.name = 'TGLF'
        self.ids.code.commit = self.metadata['version']
        self.ids.code.version = self.metadata['framework_version']
        self.ids.code.repository = 'https://gafusion.github.io/doc/tglf.html'       
        # ajouter les inputs de TGLF
        return



    # run functions
    def run(self,path=None,gacode_platform=None,gacode_root=None,init_gacode=False,verbose=False,collect=None,collect_essential=True,eigenfunctions=False):
        # initialise init GACODE bash commands
        if init_gacode:
            if not gacode_platform:
                if 'gacode_platform' in self.metadata:
                    gacode_platform = self.metadata['gacode_platform']
                else:
                    raise ValueError('No GACODE_PLATFORM was set!')
            if not gacode_root:
                if 'gacode_root' in self.metadata:
                    gacode_root = self.metadata['gacode_root']
                else:
                    raise ValueError('No GACODE_ROOT was set!')
            bash_init_gacode = [
                'export GACODE_PLATFORM={}'.format(gacode_platform),
                'export GACODE_ROOT={}'.format(gacode_root),
                '. $GACODE_ROOT/shared/bin/gacode_setup'
            ]

        # set the run path, if unspecified for convenience check metadata for a run path
        if not path:
            if 'run_path' in self.metadata:
                path = self.metadata['run_path']
            else:
                raise ValueError('No path for run was set!')

        bash = [
            'cd {}'.format(path),
            'tglf -e .'
        ]

        if verbose:
            print('Running TGLF at {} ...'.format(path))

        # init GACODE
        if init_gacode:
            commands = '; '.join(bash_init_gacode+bash)
        else:
            commands = '; '.join(bash)
        
        # delete the out.tglf.wavefunction files to prevent issues when overwriting data
        for filename in glob.glob(path+"out.tglf.wavefunction*"):
            os.remove(filename)
        # run TGLF
        execution = os.popen(commands)

        if verbose:
            print(execution.read())
        else:
            execution.read()

        # if collect is not specified default to the object setting
        if collect == None:
            collect = self.collect
        # if collect, collect all the specified output
        if collect:
            if not self.collect:
                self.collect = True
            self.collect_output(run_path=path,essential=collect_essential,eigenfunctions=eigenfunctions)

        return self

    def run_1d_scan(self,path=None,var=None,values=[],verbose=False,collect_essential=False,return_self=True):
        
        # if unspecified, for convenience check for run path in metadata
        if not path and 'run_path' in self.metadata:
            path = self.metadata['run_path']
        
        # check if scan variable was provided
        if var:
            # pre-fill a value for the scan variable in case it is not already in input
            if var not in self.input:
                if values.any():
                    self.input[var] = values[0]
                else:
                    raise ValueError('Specify scan values for {}!'.format(var))
        else:
            raise ValueError('Specify scan variable!')
        
        scan_output = {var:{}}
        _collect = copy.deepcopy(self.collect)
        _output = copy.deepcopy(self.output)
        self.collect = True

        if verbose:
            print('Running TGLF 1D scan...')
        for value in values:
            # update the scan variable value
            self.input[var] = float(value)
            # generate a new input.tglf file
            self.write_input(path=path,ignore_defaults=False,header=False)
            # reset output dictionary
            self.output = {}
            # run TGLF
            self.run(path=path,collect_essential=collect_essential)
            # store the results in the scan_output dict
            scan_output[var].update({value:copy.deepcopy(self.output)})
            # print a progress %
            print('{} TGLF 1D scan {}% complete \n'.format(ERASE_LINE,round(100*(find(value,values))/len(values))),flush=False,end='\r')
        #print(ERASE_LINE)

        # put back the output present before the scan
        self.collect = _collect
        self.output = _output
        
        if verbose:
            print('{}TGLF 1D scan complete...'.format(ERASE_LINE))
        if return_self:
            if 'scans' not in self.output:
                self.output['scans'] = {var:scan_output[var]}
            else:
                self.output['scans'].update({var:scan_output[var]})    
        else:
            return scan_output

    def run_2d_scan(self,path=None,var_y=None,values_y=[],var_x=None,values_x=[],verbose=False,collect_essential=False,return_self=True):
        if var_y not in self.input:
            if values_y.any():
                self.input[var_y]=values_y[0]
            else:
                raise ValueError('Specify scan values for {}!'.format(var_y))
        
        scan_output = {var_y:{}}

        if verbose:
            print('Running TGLF 2D scan...\n')
        for value in values_y:
            if verbose:
                # print a progress %, ANSI escape squences are used to move the cursor to update the multiline progress print
                print('{} TGLF 2D scan {}% complete\n'.format(CURSOR_UP_ONE + ERASE_LINE,round(100*(find(value,values_y))/len(values_y))),flush=False,end='\r')
            self.input[var_y] = float(value)
            self.run_1d_scan(path=path,var=var_x,values=values_x,collect_essential=collect_essential,return_self=True)
            scan_output[var_y].update({value:copy.deepcopy(self.output['scans'])})
            del self.output['scans'][var_x]
        
        if verbose:
            print('{}TGLF 2D scan complete...\n{}'.format(CURSOR_UP_ONE+ERASE_LINE,ERASE_LINE+CURSOR_UP_ONE))
        if return_self:
            self.output['scans'].update({var_y:scan_output[var_y]})
        else:
            return scan_output

    def collect_output(self,run_path=None,essential=True,eigenfunctions=False,verbose=False):
        # if unspecified, for convenience check for run path in metadata
        if not run_path and 'run_path' in self.metadata:
            run_path = self.metadata['run_path']
        
        # check for and store any previous runs/output
        default_keys = ['species', 'eigenvalues', 'eigenfunctions', 'fields', 'input_gen', 'ky', 'prec', 'sat_geo', 'sat_scalar_params', 'spectral_shift', 'gaussian_width']
        self.collate_run(select_keys=default_keys)

        output_run = [file for file in os.listdir(run_path) if os.path.isfile(os.path.join(run_path,file))]

        output_files = {'out.tglf.version':{'method':self.read_version,'essential':True},
                        'input.tglf.gen':{'method':self.read_input_gen,'essential':True},
                        'out.tglf.density_spectrum':{'method':self.read_density_spectrum,'essential':False},
                        'out.tglf.eigenvalue_spectrum':{'method':self.read_eigenvalue_spectrum,'essential':True},
                        'out.tglf.field_spectrum':{'method':self.read_field_spectrum,'essential':False},
                        'out.tglf.gbflux':{'method':self.read_gbflux,'essential':True},
                        'out.tglf.grid':{'method':self.read_grid,'essential':False},
                        'out.tglf.intensity_spectrum':{'method':self.read_intensity_spectrum,'essential':False},
                        'out.tglf.ky_spectrum':{'method':self.read_ky_spectrum,'essential':True},
                        'out.tglf.nsts_crossphase_spectrum':{'method':self.read_nsts_crossphase_spectrum,'essential':True},
                        'out.tglf.prec':{'method':self.read_prec,'essential':True},
                        'out.tglf.QL_flux_spectrum':{'method':self.read_QL_flux_spectrum,'essential':True},
                        'out.tglf.run':{'method':self.read_run,'essential':False},
                        'out.tglf.sat_geo_spectrum':{'method':self.read_sat_geo_spectrum,'essential':False},
                        'out.tglf.scalar_saturation_parameters':{'method':self.read_scalar_sat_parameters,'essential':True},
                        'out.tglf.spectral_shift':{'method':self.read_spectral_shift,'essential':False},
                        'out.tglf.sum_flux_spectrum':{'method':self.read_sum_flux_spectrum,'essential':True},
                        'out.tglf.temperature_spectrum':{'method':self.read_temperature_spectrum,'essential':False},
                        'out.tglf.width_spectrum':{'method':self.read_width_spectrum,'essential':True}}

        # read all the files present in the specified run path, taking into account essential status
        for file in output_files.keys():
            if file in output_run:
                if essential:
                    if output_files[file]['essential']:
                        try:
                            output_files[file]['method']()
                        except:
                            if verbose:
                                print('Method {} failed!'.format(output_files[file]['method']))
                            pass
                        if verbose:
                            print('Reading {}...'.format(file))
                else:
                    try:
                        output_files[file]['method']()
                    except:
                        if verbose:
                            print('Method {} failed!'.format(output_files[file]['method']))
                        pass
                    if verbose:
                            print('Reading {}...'.format(file))
        
        # generate the eigenfunctions output for all the wavenumbers in the run and store them in output
        if eigenfunctions:
            # read the current input file and store input and collect for later reference
            self.read_input(overwrite=True)
            _input = copy.deepcopy(self.input)
            _collect = copy.deepcopy(self.collect)
            # switch of collect to only return the eigenfunction dict
            self.collect = False
            ky_list = list(self.output['ky'])
            if essential:
                ky_list = [self.output['ky'][-1]]
            # modify the inputs to get the eigenfunctions output
            self.input['NKY']=1
            self.input['USE_TRANSPORT_MODEL']='F'
            # for each ky in the run modify the input, run TGLF, read the eigenfunctions and store them with the run
            for ky in ky_list:
                file = 'out.tglf.wavefunction_ky={:.2f}'.format(ky)
                if 'eigenfunctions' not in self.output:
                    self.output['eigenfunctions'] = {}
                if not os.path.isfile(run_path+file):
                    self.input['KY'] = ky
                    self.write_input()
                    self.run()
                    eigenfunction = self.read_wavefunction()['eigenfunction']
                else:
                    _eigenfunctions = self.read_wavefunction(file=file)
                    eigenfunction = _eigenfunctions['eigenfunction']
                    if _eigenfunctions['nmodes'] < self.metadata['nmodes'] or _eigenfunctions['nfields'] < self.metadata['nfields']:
                        self.input['KY'] = ky
                        self.write_input()
                        self.run()
                        eigenfunction = self.read_wavefunction()['eigenfunction']
                if ky not in self.output['eigenfunctions']:
                    self.output['eigenfunctions'][ky] = {}
                self.output['eigenfunctions'][ky].update(eigenfunction)
                if os.path.isfile(run_path+'/out.tglf.wavefunction'):
                    os.rename(run_path+'/out.tglf.wavefunction',run_path+file)
            # reset the input and collect to their original states
            self.input = copy.deepcopy(_input)
            self.write_input()
            self.collect = copy.deepcopy(_collect)
        
        # if there are any collated runs present in self.output, collate the current run as well
        output_keys = [key for key in self.output.keys() if 'run' in key]
        if output_keys:
            self.collate_run(select_keys=default_keys)
    
    def collate_run(self,output=None,select_keys=[],run_id=None):
        """Collate the select_keys in output in run_id

        Args:
            `output` (dict, optional): The dictionary to look in for `select_keys`. Defaults to self.output.
            `select_keys` (list): The keys to be collated in a `run_id`. Defaults to [].
            `run_id` (str, optional): A string specifying the run in some way. Defaults to 'run_x', where x is the highest run_id+1.

        Raises:
            ValueError: If no select_keys are specified this function cannot proceed!
        """
        # for convenience assume self.output if output not specified
        if not output:
            output = self.output
        _run = {}
        if output:
            if select_keys:
                # key by key copy the output to _run and remove the keys from output
                for key in select_keys:
                    if key in output:
                        _run.update({key:copy.deepcopy(output[key])})
                        output.pop(key,None)
                if _run:
                    # if no run_id is specified assume run_x as default format and compute the next run number if neccesary
                    if not run_id:
                        run_num = sorted([int(key.split('_')[-1]) for key in output.keys() if 'run' in key])
                        if not run_num:
                            run_num = 1
                        else:
                            run_num = run_num[-1]+1
                        run_id = 'run_{}'.format(run_num)
                    # store the collated run
                    output.update({run_id:_run})
            else:
                raise ValueError("No keys provided to collect into a run!")
    
    # plotting functions
    def plot_eigenvalue_spectra(self,run_path=None,output=None,modes=[1],figures=[None,None],labels=[None,None],show=True,save=False,files=[None,None]):
        if run_path and not self.output:
            self.collect_output(run_path=run_path)
        
        if not output:
            output = self.output

        if 'eigenvalues' in output:
            self.plot_gamma_spectrum(run_path=run_path,output=output,modes=modes,figure=figures[0],label=labels[0],show=False,save=save,file=files[0])
            self.plot_omega_spectrum(run_path=run_path,output=output,modes=modes,figure=figures[1],label=labels[1],show=show,save=save,file=files[1])

    def plot_field_spectra(self,run_path=None,output=None,modes=[1],figure=None,label=None,markers=[],show=True,save=False,file=None):
        if run_path and not self.output:
            self.collect_output(run_path=run_path)
        
        if not output:
            output = self.output

        if 'fields' in self.output:
            fields = self.output['fields']
            ky = self.output['ky']
        
            if figure:
                plt.figure(figure)
            else:
                plt.figure()
            
            if not markers:
                markers = ['o','.','s','^']
            
            for key_mode in modes:
                for key_field in fields.keys():
                    i_field = list(fields.keys()).index(key_field)
                    if label:
                        label_ = label + ', {}'.format(key_field)
                    else:
                        label_ = key_field
                    if modes[-1] > 1:
                        label_ += ', {}'.format()
                    plt.plot(ky,fields[key_field][key_mode],'{}-'.format(markers[i_field]),label=label_)
            plt.xlabel('ky')
            plt.ylim(bottom=0.)
            plt.legend()
                
            if show:
                plt.show()

    def plot_gamma_spectrum(self,run_path=None,output=None,modes=[1],figure=None,label=None,show=True,save=False,file=None):
        """Plot the growth rate (gamma) spectrum as a function of ky, as output by `read_eigenvalue_spectrum()`.

        Args:
            run_path (_type_, optional): _description_. Defaults to None.
            modes (list, optional): _description_. Defaults to [1].
            figure (_type_, optional): _description_. Defaults to None.
            label (_type_, optional): _description_. Defaults to None.
            show (bool, optional): _description_. Defaults to False.
            save (bool, optional): _description_. Defaults to False.
            file (_type_, optional): _description_. Defaults to None.
        """
        if run_path and not self.output:
            self.collect_output(run_path=run_path)
        
        if not output:
            output = self.output
        
        if 'eigenvalues' in output:
            eigenvalues = output['eigenvalues']
            ky = self.output['ky']
            gamma = eigenvalues['gamma']

            if not modes:
                modes = list(gamma.keys())
                
            if figure:
                plt.figure(figure)
            else:
                plt.figure()
            for key_mode in modes:
                if modes[-1] > 1 and label:
                    label_ = label + ', mode {}'.format(key_mode)
                else:
                    label_ = label
                plt.plot(ky,gamma[key_mode],'.-',label=label_)
            plt.xlabel('ky')
            plt.ylabel('$\\gamma$ [cs/a]')
            plt.ylim(bottom=0.)
            if label:
                plt.legend()
            
            if save:
                if not run_path and 'run_path' in self.metadata:
                    run_path = self.metadata['run_path']
                if file:
                    plt.savefig('{}{}'.format(run_path,file),format='pdf')
                else:
                    raise ValueError('TGLF.plot_gamma_spectrum: No file name was provided!')
            
            if show:
                plt.show()

    def plot_omega_spectrum(self,run_path=None,output=None,modes=[1],figure=None,label=None,align='right',show=True,save=False,file=None):
        """Plot the frequency (omega) spectrum as a function of ky, as output by `read_eigenvalue_spectrum()`.
        NOTE: this plotting routine plots the frequency using the frequency direction convention as specified during reading!

        Args:
            run_path (_type_, optional): _description_. Defaults to None.
            sign (int, optional): _description_. Defaults to -1.
            modes (list, optional): _description_. Defaults to [1].
            figure (_type_, optional): _description_. Defaults to None.
            label (_type_, optional): _description_. Defaults to None.
            show (bool, optional): _description_. Defaults to False.
            save (bool, optional): _description_. Defaults to False.
            file (_type_, optional): _description_. Defaults to None.
        """
        if run_path and not self.output:
            self.collect_output(run_path=run_path)
        
        if not output:
            output = self.output
        
        if 'eigenvalues' in output:
            eigenvalues = output['eigenvalues']
            sign_convention = eigenvalues['sign_convention']
            sign_labels = ['ion','electron']
            if not sign_convention < 0:
                sign_labels = list(reversed(sign_labels))
            align_x = 0.985
            if align != 'right':
                align_x = 0.015
            ky = output['ky']
            omega = eigenvalues['omega']
            mono = {'size':9,'weight':'bold'}

            if not modes:
                modes = list(omega.keys())
            
            axline = False
            if figure:
                plt.figure(figure)
            else:
                plt.figure()
            for key_mode in modes:
                if modes[-1] > 1 and label:
                    label_ = label + ', mode {}'.format(key_mode)
                else:
                    label_ = label
                plt.plot(ky,omega[key_mode],'.-',label=label_)
                if np.sign(min(omega[key_mode])) != np.sign(max(omega[key_mode])) and not axline:
                    plt.axhline(0,linewidth=0.75,color='black')
                    axline = True
                    ax = plt.gca()
                    y_lims = ax.get_ylim()
                    align_y = np.abs(y_lims[0])/np.sum(np.abs(np.array(y_lims)))
                    if align_y <= 0.94:
                        plt.text(align_x,align_y+0.015,sign_labels[0],ha=align,va='bottom',fontdict=mono,transform=ax.transAxes)
                    if align_y >= 0.05:
                        plt.text(align_x,align_y-0.015,sign_labels[1],ha=align,va='top',fontdict=mono,transform=ax.transAxes)
            plt.xlabel('ky')
            plt.ylabel('$\\omega$ [cs/a]')
            
            if label:
                plt.legend()
            
            if save:
                if not run_path and 'run_path' in self.metadata:
                    run_path = self.metadata['run_path']
                if file:
                    plt.savefig('{}{}'.format(run_path,file),format='pdf')
                else:
                    raise ValueError('TGLF.plot_omega_spectrum: No file name was provided!')

            if show:
                plt.show()

    def plot_eigenfunctions(self,run_path=None,output=None,modes=[1],label='',fields=['phi'],show=True,save=False,file=None):
        """Plot the growth rate (gamma) spectrum as a function of ky, as output by `read_eigenvalue_spectrum()`.

        Args:
            run_path (_type_, optional): _description_. Defaults to None.
            modes (list, optional): _description_. Defaults to [1].
            figure (_type_, optional): _description_. Defaults to None.
            label (_type_, optional): _description_. Defaults to None.
            show (bool, optional): _description_. Defaults to False.
            save (bool, optional): _description_. Defaults to False.
            file (_type_, optional): _description_. Defaults to None.
        """
        if run_path and not self.output:
            self.collect_output(run_path=run_path)
        
        if not output:
            output = self.output
        
        if 'eigenfunctions' in output:
            if 'ky' in self.output:
                ky_list = list(output['ky'])
            else:
                ky_list = list(output['eigenfunctions'].keys())
            for ky in ky_list:
                eigenfunctions = copy.deepcopy(output['eigenfunctions'][ky])
                theta = eigenfunctions['theta']
                _fields = [key for key in eigenfunctions.keys() if key != 'theta' and key in fields]
                reim_labels = ['Re','Im']
                axline = False

                for key_field in _fields:
                    eigenfunction = eigenfunctions[key_field]
                    if not modes or len(modes) > output['input_gen']['NMODES']:
                        modes = list(eigenfunction.keys())
                        print('Changed number of modes to be plotted to the available maximum!')

                    for key_mode in modes:
                        plt.figure()
                        for key_reim in reim_labels:
                            i_reim = reim_labels.index(key_reim)
                            color_reim = ['red','blue']
                            mode = [np.real(eigenfunction[key_mode]),np.imag(eigenfunction[key_mode])]
                            plt.plot(theta/np.pi,mode[i_reim],label=label+' {}({})'.format(key_reim,key_field),color=color_reim[i_reim])
                        plt.plot(theta/np.pi,np.sqrt(mode[0]**2+mode[1]**2),label=label+' ||{}||'.format(key_field),color='black')
                        if np.sign(min(mode[0])) != np.sign(max(mode[0])) or np.sign(min(mode[1])) != np.sign(max(mode[1])) and not axline:
                            plt.axhline(0,linewidth=0.75,color='black')
                            axline = True
                        plt.title('ky={:.2f}, mode: {}'.format(ky,key_mode))
                        plt.xlabel('$\\theta_p$/$\\pi$')
                        plt.ylabel('eigenfunction')
                        #plt.ylim(bottom=0.)
                        plt.legend()
            if show:
                plt.show()
        else:
            raise ValueError("No eigenfunctions found to plot, check your output!")

    # legacy functions
    def write_inputs_(self,path=None,file=None,control=None,species=None,gaussian=None,geometry=None,expert=None):
        # default values for TGLF input namelists
        header_params = {
            'name':'input.tglf',
            'message':'See https://gafusion.github.io/doc/tglf/tglf_table.html'
        }
        control_params = {
            'name':'Control paramters',
            'UNITS':'GYRO', #options: GYRO, CGYRO, GENE
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
                'DRMINDX_LOC':1.0,
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
            for key in self._species_vector:
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
                        exit(geometry['name']+" parameter: '{}' has not been specified, please check your geometry inputs!".format(key))   
        if expert != None:
            for key in expert:
                expert_params[key] = expert[key]

        self.tglf_namelist = {
            'meta' : {
                "file":file,
            },
            'header':header_params,
            'control':control_params,
            'species':species_params,
            'gaussian':gaussian_params,
            'geometry':geometry_params[geometry['name']],
            'expert':expert_params,
        }
        if path == None:
            self.tglf_namelist['meta'].update({'path':"./",})
        elif isinstance(path,str):
            self.tglf_namelist['meta'].update({'path':path,})
        else:
            exit("Invalid path string provided!")

        spacer = '#---------------------------------------------------\n'
        for namelist in self.tglf_namelist:
            if namelist == 'meta':
                if self.tglf_namelist[namelist]['file'] != None:
                    pathlib.Path(self.tglf_namelist[namelist]["path"]).mkdir(parents=True, exist_ok=True)
                    f = open(self.tglf_namelist[namelist]["path"]+self.tglf_namelist[namelist]["file"],"w+")
                    generated_file = True
                else:
                    exit('File name not specified!')
            elif namelist == 'header':
                for key in self.tglf_namelist[namelist]:
                    f.write('# '+self.tglf_namelist[namelist][key]+'\n')
            elif namelist != 'meta' and namelist != 'header':
                f.write(spacer)
                f.write('# '+self.tglf_namelist[namelist]['name']+'\n')
                f.write(spacer)
                for key in self.tglf_namelist[namelist]:
                    if key != 'name':
                        f.write("{}={}\n".format(key,self.tglf_namelist[namelist][key]))
                if namelist != 'expert':
                    f.write('\n')
        f.close()
        if generated_file:
            print('Generated TGLF input file at: '+self.tglf_namelist['meta']["path"]+self.tglf_namelist['meta']["file"])
