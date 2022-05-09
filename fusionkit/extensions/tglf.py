'''
The TGLF class

For more information on the TGLF code see https://gafusion.github.io/doc/tglf.html
NOTE: TGLF does not allow (unescaped) spaces in the run path!
NOTE: run_path needs to end with a /
'''

import os
import copy

import matplotlib.pyplot as plt
from numpy import sign
import numpy as np

from ..core.dataspine import DataSpine
from ..core.utils import *

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'
#ERASE_LINE = '\x1b[1M' #for Windows

## TGLF
class TGLF(DataSpine):
    def __init__(self):
        DataSpine.__init__(self)
        self.input = {}
        self.output = {}
        self.collect = False
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
                        eigenvalues['gamma'][key_mode].append(np.NaN)
                    if row[1+i_mode*2] != 0:
                        eigenvalues['omega'][key_mode].append(sign_convention*row[1+i_mode*2])
                    else:
                        eigenvalues['omega'][key_mode].append(np.NaN)

            eigenvalues = list_to_array(eigenvalues)
            
            if self.collect:
                eigenvalues['sign_convention'] = sign_convention
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
            _fields = {'A':{},
                      'phi':{},
            }
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
            if not nky:
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
                    for i_key,key_fields in enumerate(fields.keys()):
                        fields[key_fields][key_mode].append(row[i_key])
            list_to_array(fields)

            if not self.collect:
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
            fluctuations = {'fluctuations':{'amplitude':{symbol:[]}}}
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
            for i_species in range(0,nspecies):
                key_species = i_species+1
                if key_species not in species:
                    species[key_species]=copy.deepcopy(fluctuations)
                #elif symbol not in species[key_species]['fluctuations']['amplitude']:
                #    species[key_species]['fluctuations']['amplitude'][symbol] = []
                else:
                    merge_trees(fluctuations,species[key_species])

            #print(species)
            # read the fluctuation amplitude spectrum for each species line by line
            for line in lines[header:]:
                row = line.strip().split()
                row = [float(value) for value in row]
                for i_species in range(0,nspecies):
                    #print(i_species)
                    species[i_species+1]['fluctuations']['amplitude'][symbol].append(row[i_species])

            for key_species in range(1,nspecies+1):
                species[key_species]['fluctuations']['amplitude'][symbol] = list_to_array(species[key_species]['fluctuations']['amplitude'][symbol])

            if not self.collect:
                fluctuation_spectrum = {'description':description, 'species':species}
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
            fluxes = {'Gamma':{'total':0.},
                      'Q':{'total':0.},
                      'Pi':{'total':0.},
                      'S':{'total':0.}
            }
            # convert the line of values into a list of floats
            fluxes_list = [float(value) for value in lines[0].split()]
            _nspecies = int(len(fluxes_list)/len(list(fluxes.keys())))
            if not nspecies or not nspecies <= _nspecies:
                nspecies = _nspecies

            for key in range(0,nspecies):
                key_species = key+1
                if key_species not in species:
                    species.update({key_species:copy.deepcopy(fluxes)})
                for i_flux,key_flux in enumerate(list(fluxes.keys())):
                    if key_flux not in species[key_species]:
                        species[key_species][key_flux]['total'] = 0.
                    species[key_species][key_flux]['total'] = ((fluxes_list[i_flux*_nspecies:(i_flux+1)*_nspecies])[:nspecies])[key]

            if not self.collect:
                gbfluxes = {'description':'gyro-bohm normalized fluxes summed over mode number and ky spectrum', 
                            'species':species}
                return gbfluxes

    def read_grid(self,run_path=None):
        """Read the out.tglf.grid file

        Args:
            run_path (str): path to the output directory of the tglf run. Defaults to None.

        Returns:
            dict: {'NS':int,'NXGRID':int}
        """
        # if unspecified, for convenience check for run path in metadata
        if not run_path and 'run_path' in self.metadata:
            run_path = self.metadata['run_path']

        # read the out.tglf.grid file
        lines = read_file(path=run_path,file='out.tglf.grid')

        if lines:
            nspecies = lines[0].strip()
            nxgrid = lines[1].strip()

            grid = {'NS':nspecies, 'NXGRID':nxgrid}

            return grid

    def read_input(self,run_path=None,file='input.tglf'):
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
        lines = read_file(path=run_path,file=file)

        # if the file was successfully read
        if lines:
            # set file dependent variables
            input = {}
            # go line by line
            for line in lines:
                # check it is not a header line
                if '#' not in line:
                    line = line.split("=")
                    key = line[0].strip()
                    value = line[1].strip()
                    # type the values, not using autotype to keep bool values in Fortran format
                    try:
                        value = int(value)
                    except:
                        try:
                            value = float(value)
                        except:
                            value = str(value)
                    
                    # check if storing IO in the TGLF object
                    if self.collect:
                        self.input.update({key:value})
                    # reader is used standalone
                    else:
                        input.update({key:value})

            if not self.collect:
                return input

    def read_input_gen(self,run_path=None):
        """Read the ouput TGLF input_gen.

        Args:
            `run_path` (str): path to the TGLF run.

        Returns:
            dict: all the TGLF input variables used in the run output in input.tglf.gen.
        """
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
                if self.collect:
                    self.output['input_gen'].update({key:value})
                else:
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
            fluctuations = {'n':[],
                            'T':[],
                            'v_par':[],
                            'E_par':[]
            }
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
            for _species in range(0,nspecies):
                key_species = _species + 1
                if key_species not in species:
                    species[key_species] = {}
                if 'modes' not in species[key_species]:
                    species[key_species]['modes'] = {}
                modes = species[key_species]['modes']
                for mode in range(0,nmodes):
                    key_mode = mode + 1
                    if key_mode not in modes:
                        modes[key_mode] = {}
                    if 'fluctuation_intensity' not in modes[key_mode]:
                        modes[key_mode]['fluctuation_intensity'] = {}
                    _fluctuations = copy.deepcopy(fluctuations)
                    for ky in range(0,nky):
                        index = header+(_species*_nmodes*_nky)+(ky*_nmodes)+mode
                        row = lines[index].strip().split()
                        row = [float(value) for value in row]
                        for i_key,key in enumerate(_fluctuations.keys()):
                            _fluctuations[key].append(row[i_key])
                    _fluctuations = list_to_array(_fluctuations)
                    modes[key_mode].update({'fluctuation_intensity':_fluctuations})
            
            if not self.collect:
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
            if self.collect:
                if 'ky' not in self.output:
                    self.output['ky'] = []
                ky_list = self.output['ky']
            else:
                ky_list = []
            # get the number of ky in the spectrum and then read the spectrum into a list
            for i_line,line in enumerate(lines):
                if i_line == 1:
                    nky = int(line.strip())
                if header <= i_line <= nky+header:
                    ky_list.append(float(line.strip()))
            
            list_to_array(ky_list)
        
            if not self.collect:
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
            if 'modes' not in species[key_species]:
                    species[key_species]['modes'] = {}
            modes = species[key_species]['modes']
            for ky in range(0,nky):
                index = header + ky
                row = lines[index].strip().split()
                row = [float(value) for value in row]
                for mode in range(0,nmodes):
                    key_mode = mode + 1
                    if key_mode not in modes:
                        modes[key_mode] = {}
                    if 'nt_cross_phase' not in modes[key_mode]:
                        modes[key_mode].update({'nt_cross_phase':[]})
                    modes[key_mode]['nt_cross_phase'].append(row[mode])
            
            modes = list_to_array(modes)

            if not self.collect:
                nete_crossphase_spectrum = {'description':description, 'modes':modes, 'nmodes':nmodes, 'nky':nky}
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
                    if 'modes' not in species[key_species]:
                        species[key_species]['modes'] = {}
                    modes = species[key_species]['modes']
                elif '(nsts_phase_spectrum_out' in row[0].split(','):
                    if 'nsts_phase_spectrum_out' not in description:
                        description += ', '+''.join(row)
                else:
                    line_lb = (key_species*header_species)+((key_species-1)*_nky)-1
                    if  line_lb < i_line <= line_lb+nky:
                        row = [float(value) for value in row]
                        for mode in range(0,nmodes):
                            key_mode = mode + 1
                            if key_mode not in modes:
                                modes[key_mode] = {}
                            if 'nt_cross_phase' not in modes[key_mode]:
                                modes[key_mode].update({'nt_cross_phase':[]})
                            modes[key_mode]['nt_cross_phase'].append(row[mode])

            modes = list_to_array(modes)

            if not self.collect:
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
                        species.update({key_species:copy.deepcopy(weights)})
                    merge_trees(weights,species[key_species])
                    ql_weigths = species[key_species]['QL_weights']
                    i_field = int(row[-1])
                    for key_flux in weights['QL_weights'].keys():
                        if key_flux not in ql_weigths:
                            ql_weigths[key_flux] = {}
                        if fields[i_field-1] not in ql_weigths[key_flux]:
                            ql_weigths[key_flux].update({fields[i_field-1]:['list-of-arrays']})
                elif 'mode' in row:
                    i_mode = int(row[-1])
                else:
                    row = [float(value) for value in row]
                    for i_flux,key_flux in enumerate(weights['QL_weights'].keys()):
                        if i_mode+1 > len(ql_weigths[key_flux][fields[i_field-1]]):
                            ql_weigths[key_flux][fields[i_field-1]].append([])    
                        ql_weigths[key_flux][fields[i_field-1]][i_mode].append(row[i_flux])
            
            for key_species in species:
                for key_flux in weights['QL_weights']:
                    list_to_array(species[key_species]['QL_weights'][key_flux])

            if not self.collect:
                QL_flux_spectrum = {'description':description, 'species':species, 'nfields':nfields, 'nmodes':nmodes}
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

        # set file dependent variables
        header = 3
        modes = {}

        if lines:
            description = lines[0].strip()
            # get/set the number modes
            _nmodes = int(lines[header-1].strip()[-1])
            if not nmodes:
                nmodes = _nmodes
            # read the values row by row
            for line in lines[header:]:
                row = line.strip().split()

                for mode in range(0,nmodes):
                    key_mode = mode + 1
                    if key_mode not in modes:
                        modes[key_mode] = {'sat_geo':[]}
                    modes[key_mode]['sat_geo'].append(float(row[mode]))
            
            sat_geo_spectrum = {'description':description, 'modes':list_to_array(modes)}

            return sat_geo_spectrum

    def read_scalar_saturation_parameters(self,run_path=None):
        # if unspecified, for convenience check for run path in metadata
        if not run_path and 'run_path' in self.metadata:
            run_path = self.metadata['run_path']

        # read the out.tglf.scalar_saturation_parameters file
        lines = read_file(path=run_path,file='out.tglf.scalar_saturation_parameters')

        # set file dependent variables
        header = 2
        scalar_sat_params = {}

        if lines:
            description = lines[0].strip()
            scalar_sat_params.update({'description':description})
            # go line by line
            for i_line,line in enumerate(lines[header:]):
                row = line.strip().split()
                row = [autotype(value.split(',')[0]) for value in row]
                # check if the line contains the variable names
                if all(isinstance(value,str) for value in row):
                    for i_value,value in enumerate(row):
                        # get the next row containing the variable values
                        next_row = [autotype(value) for value in lines[header+i_line+1].strip().split()]
                        # automatically sort and collect all the values
                        scalar_sat_params.update({value:next_row[i_value]})

            return scalar_sat_params

    def read_spectral_shift(self,run_path=None):
        # if unspecified, for convenience check for run path in metadata
        if not run_path and 'run_path' in self.metadata:
            run_path = self.metadata['run_path']

        # read the out.tglf.spectral_shift file
        lines = read_file(path=run_path,file='out.tglf.spectral_shift')

        # set file dependent variables
        header = 5
        shift_list = []

        # if the file was successfully read
        if lines:
            description = lines[0].strip() + ', ' + ' '.join([line.strip() for line in lines[1:header-2]])
            # read the spectrum into a list
            for line in lines[header:]:
                shift_list.append(float(line.strip()))
        
            spectral_shift = {'description':description, 'spectral_shift':list_to_array(shift_list)}
        
            return spectral_shift

    def read_sum_flux_spectrum(self,run_path=None,nspecies=1,nfields=1):
        # if unspecified, for convenience check for run path in metadata
        if not run_path and 'run_path' in self.metadata:
            run_path = self.metadata['run_path']

        # read the output.tglf.sum_flux_spectrum file
        lines = read_file(path=run_path,file='out.tglf.sum_flux_spectrum')

        # if the file was successfully read
        if lines:
            # set file dependent variables
            # check if storing IO in the TGLF object
            if self.collect:
                if 'species' not in self.output:
                    self.output['species'] = {}
                species = self.output['species']
            # reader is used standalone
            else:
                species = {}
            fluxes = {'Gamma':{},
                      'Q':{},
                      'Pi_tor':{},
                      'Pi_par':{},
                      'S':{}
            }
            fields = ['phi','Bper','Bpar']
            # go line by line
            for line in lines:
                row = line.strip().split()
                # check if the line is a header line
                if 'species' in row:
                    key_species = int(row[2])
                    if key_species not in species:
                        species.update({key_species:copy.deepcopy(fluxes)})
                    i_field = int(row[-1])
                    for key_flux in fluxes.keys():
                        if key_flux not in species[key_species]:
                            species[key_species][key_flux] = {}
                        if fields[i_field-1] not in species[key_species][key_flux]:
                            species[key_species][key_flux].update({fields[i_field-1]:[]})
                    if nfields < i_field:
                        nfields = i_field
                # check if the line is a description line
                elif 'particle' in row:
                    _description = line.strip().split(',')
                # automatically sort and collect all the values
                else:
                    row = [float(value) for value in row]
                    for i_flux,key_flux in enumerate(fluxes.keys()):
                        species[key_species][key_flux][fields[i_field-1]].append(row[i_flux])
            
            # convert to arrays
            species = list_to_array(species)

            # add the total flux electrostatic and electromagnetic fluxes
            for _species in species.keys():
                total = 0.
                for key_flux in fluxes:
                    for key_field in species[_species][key_flux]:
                        total += species[_species][key_flux][key_field]
                    species[_species][key_flux].update({'sum':total})

            description = ', '.join(_description)+' by field per species'
            
            flux_spectrum = {'description':description, 'species':species, 'nfields':nfields}

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

            version = {'version':'TGLF '+version_commit, 'version_date':version_date, 'platform':gacode_platform, 'run_date':run_date, 'run_time':run_time}

            return version

    def read_wavefunction(self,run_path=None,nmodes=None):
        # if unspecified, for convenience check for run path in metadata
        if not run_path and 'run_path' in self.metadata:
            run_path = self.metadata['run_path']

        # read the output.tglf.wavefunction file
        lines = read_file(path=run_path,file='out.tglf.wavefunction')

        # set file dependent variables
        header = 2
        modes = {}

        # if the file was successfully read
        if lines:
            # get the index ranges
            [_nmodes,nfields,ntheta] = [int(n) for n in lines[0].strip().split()]
            if not nmodes:
                nmodes = _nmodes
            # read column headers to store the field keys
            fields = {key:[] for key in lines[1].strip().split()}
            # prep the mode storage
            for mode in range(0,nmodes):
                key_mode = mode + 1
                modes[key_mode] = copy.deepcopy(fields)
            # line by line read the field wavefunctions per mode
            for line in lines[header:]:
                row = line.split()
                row = [autotype(value) for value in row]
                for mode in range(0,nmodes):
                    key_mode = mode + 1
                    for i_key,key in enumerate(modes[key_mode].keys()):
                        # always get the theta value at the start of the line, regardless of mode number
                        if i_key == 0:
                            modes[key_mode][key].append(row[i_key])
                        # otherwise using a sliding window on the row to get the appropriate mode data
                        else:
                            modes[key_mode][key].append(row[i_key+(mode*(2*nfields))])

            field_spectrum = {'fields':fields, 'modes':list_to_array(modes), 'nmodes':nmodes, 'nfields':nfields, 'ntheta':ntheta}

            return field_spectrum

    def read_width_spectrum(self,run_path=None):
        # if unspecified, for convenience check for run path in metadata
        if not run_path and 'run_path' in self.metadata:
            run_path = self.metadata['run_path']

        # read the out.tglf.width_spectrum file
        lines = read_file(path=run_path,file='out.tglf.width_spectrum')

        # set file dependent variables
        header = 3
        width_list = []

        # if the file was successfully read
        if lines:
            description = lines[0].strip()
            # read the spectrum into a list
            for line in lines[header:]:
                width_list.append(float(line.strip()))

            width_spectrum = {'description':description, 'Gaussian_width':list_to_array(width_list)}

            return width_spectrum

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

    # adapter functions, code_x_to_tglf() or tglf_to_code_x()
    def _gene_to_tglf(self,input_path=None):
        """Adapat a GENE input file to a TGLF input file.

        Args:
            input_path (_type_, optional): _description_. Defaults to None.
        """
        return

    def _tglf_to_ids():
        """Convert a TGLF object to IMAS gyrokinetic IDS/GKDB format.
        """
        return

    def _ids_to_tglf():
        """Convert TGLF data in IMAS gyrokinetic IDS/GKDB format to a TGLF object.
        """
        return

    # run functions
    def run(self,path=None,gacode_platform=None,gacode_root=None,init_gacode=True,verbose=False,collect=True):
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

        if init_gacode:
            commands = '; '.join(bash_init_gacode+bash)
        else:
            commands = '; '.join(bash)

        execution = os.popen(commands)

        if verbose:
            print(execution.read())
        else:
            execution.read()

        if collect:
            self.collect_output(run_path=path)

        return

    def run_1d_scan(self,path=None,var=None,values=[],verbose=False,return_self=True):
        # check if scan variable was provided
        if var:
            # pre-fill a value for the scan variable in case it is not already in input
            if var not in self.input:
                if values:
                    self.input[var] = values[0]
                else:
                    raise ValueError('Specify scan values for {}!'.format(var))
        else:
            raise ValueError('Specify scan variable!')
        
        scan_output = {var:{}}

        if verbose:
            print('Running TGLF 1D scan...')
        for value in values:
            # print a progress %
            print('{} TGLF 1D scan {}% complete'.format(ERASE_LINE,round(100*(find(value,values))/len(values))),flush=False,end='\r')
            # update the scan variable value
            self.input[var] = float(value)
            # generate a new input.tglf file
            self.write_input(path=path,ignore_defaults=False,header=False)
            # run TGLF
            self.run(path=path)
            # store the results in the scan_output dict
            scan_output[var].update({value:copy.deepcopy(self.output)})
        #print(ERASE_LINE)
        
        if verbose:
            print('{}TGLF 1D scan complete...'.format(ERASE_LINE))
        if return_self:
            if 'scans' not in self.output:
                self.output['scans'] = {var:scan_output[var]}
            else:
                self.output['scans'].update({var:scan_output[var]})    
        else:
            return scan_output
    
    def run_2d_scan(self,path=None,var_y=None,values_y=[],var_x=None,values_x=[],verbose=False,return_self=True):
        if var_y not in self.input:
            if values_y:
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
            self.run_1d_scan(path=path,var=var_x,values=values_x,return_self=True)
            scan_output[var_y].update({value:copy.deepcopy(self.output['scans'])})
            del self.output['scans'][var_x]
        
        if verbose:
            print('{}TGLF 2D scan complete...\n{}'.format(CURSOR_UP_ONE+ERASE_LINE,ERASE_LINE+CURSOR_UP_ONE))
        if return_self:
            self.output['scans'].update({var_y:scan_output[var_y]})
        else:
            return scan_output

    def collect_output(self,run_path=None):
        if 'species' not in self.output:
            self.output['species'] = {}
        if 'ky' not in self.output:
            self.output['ky'] = []
        if 'modes' not in self.output:
            self.output['modes'] = {}
        
        self.read_input(run_path=run_path)
        self.output['inputs_gen'] = self.read_input_gen(run_path=run_path)

        # read all the output files
        density_fluctuations = self.read_density_spectrum(run_path=run_path,nspecies=self.output['inputs_gen']['NS'])
        eigenvalue_spectrum = self.read_eigenvalue_spectrum(run_path=run_path)
        gbfluxes = self.read_gbflux(run_path=run_path,nspecies=self.output['inputs_gen']['NS'])
        ky_spectrum = self.read_ky_spectrum(run_path=run_path)
        temperature_fluctuations = self.read_temperature_spectrum(run_path=run_path,nspecies=self.output['inputs_gen']['NS'])
        version = self.read_version(run_path=run_path)

        # process the output results for convenient retrieval
        self.output.update(version)
        self.output['ky']+=ky_spectrum['ky']

        for species in gbfluxes['species']:
            if species not in self.output['species']:
                self.output['species'][species] = {}
            if 'n_tilde' not in self.output['species'][species]:
                self.output['species'][species]['n_tilde'] = {}
            if 'T_tilde' not in self.output['species'][species]:
                self.output['species'][species]['T_tilde'] = {}
            self.output['species'][species].update(gbfluxes['species'][species])
            for i_ky,ky in enumerate(ky_spectrum['ky']):
                self.output['species'][species]['n_tilde'].update({ky:density_fluctuations['species'][species]['density_fluctuation'][i_ky]})
                self.output['species'][species]['T_tilde'].update({ky:temperature_fluctuations['species'][species]['temperature_fluctuation'][i_ky]})
    
        for mode in eigenvalue_spectrum['modes']:
            if mode not in self.output['modes']:
                self.output['modes'][mode] = {'x_variable':'ky', 'gamma':{}, 'omega':{}}
            for eigenvalue in ['gamma','omega']:
                for i_ky,ky in enumerate(ky_spectrum['ky']):
                    self.output['modes'][mode][eigenvalue].update({ky:eigenvalue_spectrum['modes'][mode][eigenvalue][i_ky]})
    
        del self.output['inputs_gen']
    
        return
    
    # plotting functions
    def plot_eigenvalue_spectra(self,run_path=None,modes=[1],figures=[None,None],labels=[None,None],show=True,save=False,files=[None,None]):
        if run_path and not self.output:
            self.collect_output(run_path=run_path)

        if 'eigenvalues' in self.output:
            self.plot_gamma_spectrum(run_path=run_path,modes=modes,figure=figures[0],label=labels[0],show=False,save=save,file=files[0])
            self.plot_omega_spectrum(run_path=run_path,modes=modes,figure=figures[1],label=labels[1],show=show,save=save,file=files[1])

    def plot_field_spectra(self,run_path=None,modes=[1],figure=None,label=None,markers=[],show=True,save=False,file=None):
        if run_path and not self.output:
            self.collect_output(run_path=run_path)

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


    def _plot_flux_spectrum():
        """Plot the particle flux, energy flux, toroidal stress, parallel stress, exchange spectra output by `read_sum_flux_spectrum()`.
        """
        return

    def plot_gamma_spectrum(self,run_path=None,modes=[1],figure=None,label=None,show=True,save=False,file=None):
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
        
        if 'eigenvalues' in self.output:
            eigenvalues = self.output['eigenvalues']
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
            
            if show:
                plt.show()

    def plot_omega_spectrum(self,run_path=None,modes=[1],figure=None,label=None,align='right',show=True,save=False,file=None):
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
        
        if 'eigenvalues' in self.output:
            eigenvalues = self.output['eigenvalues']
            sign_convention = eigenvalues['sign_convention']
            sign_labels = ['ion','electron']
            if not sign_convention < 0:
                sign_labels = list(reversed(sign_labels))
            align_x = 0.985
            if align != 'right':
                align_x = 0.015
            ky = self.output['ky']
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
                if sign(min(omega[key_mode])) != sign(max(omega[key_mode])) and not axline:
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

            if show:
                plt.show()

    def _plot_wavefunction(self,run_path=None,nfields=1):
        """Plot the wavefunctions as a function of ballooning angle for the specified number of fields, as output by `read_wavefunction()`.

        Args:
            run_path (_type_, optional): _description_. Defaults to None.
            nfields (int, optional): _description_. Defaults to 1.
        """

        plt.figure()
        plt.plot(self.output['modes'][1]['theta']/np.pi,self.output['modes'][1]['RE(phi)'],'r--',label='Re(phi)')
        plt.plot(self.output['modes'][1]['theta']/np.pi,self.output['modes'][1]['IM(phi)'],'b--',label='Im(phi)')
        plt.plot(self.output['modes'][1]['theta']/np.pi,np.sqrt(self.output['modes'][1]['RE(phi)']**2+self.output['modes'][1]['IM(phi)']**2),'k-',label='|phi|')
        plt.xlabel('$\\theta/\\pi$')
        plt.legend()

        if 'RE(Bper)' in self.output['modes'][1]:
            plt.figure()
            plt.plot(self.output['modes'][1]['theta']/np.pi,self.output['modes'][1]['RE(Bper)'],'r--',label='Re(Bper)')
            plt.plot(self.output['modes'][1]['theta']/np.pi,self.output['modes'][1]['IM(Bper)'],'b--',label='Im(Bper)')
            plt.plot(self.output['modes'][1]['theta']/np.pi,np.sqrt(self.output['modes'][1]['RE(Bper)']**2+self.output['modes'][1]['IM(Bper)']**2),'k-',label='|Bper|')
            plt.xlabel('$\\theta/\\pi$')
            plt.legend()

        if 'RE(Bpar)' in self.output['modes'][1]:
            plt.figure()
            plt.plot(self.output['modes'][1]['theta']/np.pi,self.output['modes'][1]['RE(Bpar)'],'r--',label='Re(Bpar)')
            plt.plot(self.output['modes'][1]['theta']/np.pi,self.output['modes'][1]['IM(Bpar)'],'b--',label='Im(Bpar)')
            plt.plot(self.output['modes'][1]['theta']/np.pi,np.sqrt(self.output['modes'][1]['RE(Bpar)']**2+self.output['modes'][1]['IM(Bpar)']**2),'k-',label='|Bpar|')
            plt.xlabel('$\\theta/\\pi$')
            plt.legend()
        plt.show()

        return

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
