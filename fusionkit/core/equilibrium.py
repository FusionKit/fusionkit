"""
Module to handle any and all methods related to magnetic equilibrium data

The Equilibrium class can read magnetic equilibria files (only edsk g-files for now),
add derived quantities (e.g. phi, rho_tor, rho_pol, etc.) to the Equilibrium, trace flux surfaces
and derive shaping parameters (for now only Miller parameters) from the flux surfaces.
"""

# general imports
import os
import re
import numpy as np
import json,codecs
import copy
# imports methods from
from operator import itemgetter
from scipy import interpolate, integrate
from sys import stdout
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from IPython import embed

# framework imports
from .utils import *
from .dataspine import DataSpine

# Common numerical data types, for ease of type-checking
np_itypes = (np.int8, np.int16, np.int32, np.int64)
np_utypes = (np.uint8, np.uint16, np.uint32, np.uint64)
np_ftypes = (np.float16, np.float32, np.float64)

number_types = (float, int, np_itypes, np_utypes, np_ftypes)
array_types = (list, tuple, np.ndarray)

class Equilibrium(DataSpine):
    """
    Class to handle any and all data related to the magnetic equilibrium in a magnetic confinement fusion device.
    """
    def __init__(self):
        DataSpine.__init__(self)
        self.raw = {} # storage for all raw eqdsk data
        self.derived = {} # storage for all data derived from eqdsk data
        self.fluxsurfaces = {} # storage for all data related to flux surfaces
        # specify the eqdsk file formate, based on 'G EQDSK FORMAT - L Lao 2/7/97'
        self._eqdsk_format = {
            0:{'vars':['case','idum','nw','nh'],'size':[4]},
            1:{'vars':['rdim', 'zdim', 'rcentr', 'rleft', 'zmid'],'size':[5]},
            2:{'vars':['rmaxis', 'zmaxis', 'simag', 'sibry', 'bcentr'],'size':[5]},
            3:{'vars':['current', 'simag2', 'xdum', 'rmaxis2', 'xdum'],'size':[5]},
            4:{'vars':['zmaxis2', 'xdum', 'sibry2', 'xdum', 'xdum'],'size':[5]},
            5:{'vars':['fpol'],'size':['nw']},
            6:{'vars':['pres'],'size':['nw']},
            7:{'vars':['ffprim'],'size':['nw']},
            8:{'vars':['pprime'],'size':['nw']},
            9:{'vars':['psirz'],'size':['nh','nw']},
            10:{'vars':['qpsi'],'size':['nw']},
            11:{'vars':['nbbbs','limitr'],'size':[2]},
            12:{'vars':['rbbbs','zbbbs'],'size':['nbbbs']},
            13:{'vars':['rlim','zlim'],'size':['limitr']},
        }
        self._sanity_values = ['rmaxis','zmaxis','simag','sibry'] # specify the sanity values used for consistency check of eqdsk file
        self._max_values = 5 # maximum number of values per line

    ## I/O functions
    def read_geqdsk(self,f_path=None,just_raw=False,add_derived=False):
        """Read an eqdsk g-file from file into `Equilibrium` object

        Args:
            `f_path` (str): the path to the eqdsk g-file, including the file name (!).
            `just_raw` (bool): [True] return only the raw dictionary, or [False, default] return the `Equilibrium` object.
            `add_derived` (bool): [True] also add derived quantities (e.g. phi, rho_tor) to the `Equilibrium` object upon reading the g-file, or [False, default] not.

        Returns:
            [default] self, or dict if `just_raw`
        
        Raises:
            ValueError: Raise an exception when no `f_path` is provided
        """
        print('Reading eqdsk g-file to Equilibrium...')

        # check if eqdsk file path is provided and if it exists
        if f_path is None or not os.path.isfile(f_path):
            raise ValueError('Invalid file or path provided!')
            return
        
        # read the g-file
        with open(f_path,'r') as file:
            lines = file.readlines()
        
        if lines:
            # start at the top of the file
            current_row = 0
            # go through the eqdsk format key by key and collect all the values for the vars in each format row
            for key in self._eqdsk_format:
                if current_row < len(lines):
                    # check if the var size is a string refering to a value to be read from the eqdsk file and backfill it, for loop for multidimensional vars
                    for i,size in enumerate(self._eqdsk_format[key]['size']):
                        if isinstance(size,str):
                            self._eqdsk_format[key]['size'][i] = self.raw[size]

                    # compute the row the current eqdsk format key ends
                    if len(self._eqdsk_format[key]['vars']) != np.prod(self._eqdsk_format[key]['size']):
                        end_row = current_row + int(np.ceil(len(self._eqdsk_format[key]['vars'])*np.prod(self._eqdsk_format[key]['size'])/self._max_values))
                    else:
                        end_row = current_row + int(np.ceil(np.prod(self._eqdsk_format[key]['size'])/self._max_values))

                    # check if there are values to be collected
                    if end_row > current_row:
                        _lines = lines[current_row:end_row]
                        for i_row, row in enumerate(_lines):
                            try:
                                # split the row string into separate values by ' ' as delimiter, adding a space before a minus sign if it is the delimiter
                                values = list(filter(None,re.sub(r'(?<![Ee])-',' -',row).rstrip('\n').split(' ')))
                                # select all the numerical values in the list of sub-strings of the current row, but keep them as strings so the fortran formatting remains
                                numbers = [j for i in [number for number in (re.findall(r'^(?![A-Z]).*-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[-+]?\ *[0-9]+)?', value) for value in values)] for j in i]
                                # select all the remaining sub-strings and store them in a separate list
                                strings = [value for value in values if value not in numbers]
                                # handle the exception of the first line where in the case description numbers and strings can be mixed
                                if current_row == 0:
                                    numbers = numbers[-3:]
                                    strings = [string for string in values if string not in numbers] 
                                # convert the list of numerical sub-strings to their actual int or float value and collate the strings in a single string
                                numbers = [number(value) for value in numbers]
                                strings = ' '.join(strings)
                                _values = numbers
                                if strings:
                                    _values.insert(0,strings)
                            except:
                                _values = row.strip()
                            _lines[i_row] = _values
                        # unpack all the values between current_row and end_row in the eqdsk file and flatten the resulting list of lists to a list
                        values = [value for row in _lines for value in row]

                        # handle the exception of len(eqdsk_format[key]['vars']) > 1 and the data being stored in value pairs 
                        if len(self._eqdsk_format[key]['vars']) > 1 and len(self._eqdsk_format[key]['vars']) != self._eqdsk_format[key]['size'][0]:
                            # make a shadow copy of values
                            _values = copy.deepcopy(values)
                            # empty the values list
                            values = []
                            # collect all the values belonging to the n-th variable in the format list and remove them from the shadow value list until empty
                            for j in range(len(self._eqdsk_format[key]['vars']),0,-1):
                                values.append(np.array(_values[0::j]))
                                _values = [value for value in _values if value not in values[-1]]
                        # store and reshape the values in a np.array() in case eqdsk_format[key]['size'] > max_values
                        elif self._eqdsk_format[key]['size'][0] > self._max_values:
                            values = [np.array(values).reshape(self._eqdsk_format[key]['size'])]
                        # store the var value pairs in the eqdsk dict
                        self.raw.update({var:values[k] for k,var in enumerate(self._eqdsk_format[key]['vars'])})
                    # update the current position in the 
                    current_row = end_row
            
            # store any remaining lines as a comment, in case of CHEASE/LIUQE
            if current_row < len(lines):
                comment_lines = []
                for line in lines[current_row+1:]:
                    if isinstance(line,list):
                        comment_lines.append(' '.join([str(text) for text in line]))
                    else:
                        if line.strip():
                            comment_lines.append(str(line))
                self.raw['comment'] = '\n'.join(comment_lines)

            # sanity check the eqdsk values
            for key in self._sanity_values:
                # find the matching sanity key in eqdsk
                sanity_pair = [keypair for keypair in self.raw.keys() if keypair.startswith(key)][1]
                #print(sanity_pair)
                if self.raw[key]!=self.raw[sanity_pair]:
                    raise ValueError('Inconsistent '+key+': %7.4g, %7.4g'%(self.raw[key], self.raw[sanity_pair])+'. CHECK YOUR EQDSK FILE!')

            if add_derived:
                self.add_derived()
            if just_raw:
                return self.raw
            else:
                return self
    
    def write_geqdsk(self,f_path=None):
        """ Write an `Equilibrium` object to an eqdsk g-file 

        Args:
            f_path (str): the target path of generated eqdsk g-file, including the file name (!).
        
        Returns:
            
        """
        print('Writing Equilibrium to eqdsk g-file...')

        if self.raw:
            if not isinstance(f_path, str):
                raise TypeError("filepath field must be a string. EQDSK file write aborted.")

            maxv = int(self._max_values)

            eqpath = Path(f_path)
            if eqpath.is_file():
                print("%s exists, overwriting file with EQDSK file!" % (str(eqpath)))
            eq = {"xdum": 0.0}
            for linenum in self._eqdsk_format:
                if "vars" in self._eqdsk_format[linenum]:
                    for key in self._eqdsk_format[linenum]["vars"]:
                        if key in self.raw:
                            eq[key] = copy.deepcopy(self.raw[key])
                        elif key in ["nbbbs","limitr","rbbbs","zbbbs","rlim","zlim"]:
                            eq[key] = None
                            if key in self.derived:
                                eq[key] = copy.deepcopy(self.derived[key])
                        else:
                            raise TypeError("%s field must be specified. EQDSK file write aborted." % (key))
            if eq["nbbbs"] is None or eq["rbbbs"] is None or eq["zbbbs"] is None:
                eq["nbbbs"] = 0
                eq["rbbbs"] = []
                eq["zbbbs"] = []
            if eq["limitr"] is None or eq["rlim"] is None or eq["zlim"] is None:
                eq["limitr"] = 0
                eq["rlim"] = []
                eq["zlim"] = []

            eq["xdum"] = 0.0
            with open(str(eqpath), 'w') as ff:
                gcase = ""
                '''if "code" in eq and eq["code"]:
                    gcase = gcase + eq["code"] + " "'''
                gcase = gcase + eq["case"][:48 - len(gcase)] if (len(eq["case"]) - len(gcase)) > 48 else gcase + eq["case"]
                ff.write("%-48s%4d%4d%4d\n" % (gcase, eq["idum"], eq["nw"], eq["nh"]))
                ff.write("%16.9E%16.9E%16.9E%16.9E%16.9E\n" % (eq["rdim"], eq["zdim"], eq["rcentr"], eq["rleft"], eq["zmid"]))
                ff.write("%16.9E%16.9E%16.9E%16.9E%16.9E\n" % (eq["rmaxis"], eq["zmaxis"], eq["simag"], eq["sibry"], eq["bcentr"]))
                ff.write("%16.9E%16.9E%16.9E%16.9E%16.9E\n" % (eq["current"], eq["simag"], eq["xdum"], eq["rmaxis"], eq["xdum"]))
                ff.write("%16.9E%16.9E%16.9E%16.9E%16.9E\n" % (eq["zmaxis"], eq["xdum"], eq["sibry"], eq["xdum"], eq["xdum"]))
                for ii in range(0, len(eq["fpol"])):
                    ff.write("%16.9E" % (eq["fpol"][ii]))
                    if (ii + 1) % maxv == 0 and (ii + 1) != len(eq["fpol"]):
                        ff.write("\n")
                ff.write("\n")
                for ii in range(0, len(eq["pres"])):
                    ff.write("%16.9E" % (eq["pres"][ii]))
                    if (ii + 1) % maxv == 0 and (ii + 1) != len(eq["pres"]):
                        ff.write("\n")
                ff.write("\n")
                for ii in range(0, len(eq["ffprim"])):
                    ff.write("%16.9E" % (eq["ffprim"][ii]))
                    if (ii + 1) % maxv == 0 and (ii + 1) != len(eq["ffprim"]):
                        ff.write("\n")
                ff.write("\n")
                for ii in range(0, len(eq["pprime"])):
                    ff.write("%16.9E" % (eq["pprime"][ii]))
                    if (ii + 1) % maxv == 0 and (ii + 1) != len(eq["pprime"]):
                        ff.write("\n")
                ff.write("\n")
                kk = 0
                for ii in range(0, eq["nh"]):
                    for jj in range(0, eq["nw"]):
                        ff.write("%16.9E" % (eq["psirz"][ii, jj]))
                        if (kk + 1) % maxv == 0 and (kk + 1) != (eq["nh"] * eq["nw"]):
                            ff.write("\n")
                        kk = kk + 1
                ff.write("\n")
                for ii in range(0, len(eq["qpsi"])):
                    ff.write("%16.9E" % (eq["qpsi"][ii]))
                    if (ii + 1) % maxv == 0 and (ii + 1) != len(eq["qpsi"]):
                        ff.write("\n")
                ff.write("\n")
                ff.write("%5d%5d\n" % (eq["nbbbs"], eq["limitr"]))
                kk = 0
                for ii in range(0, eq["nbbbs"]):
                    ff.write("%16.9E" % (eq["rbbbs"][ii]))
                    if (kk + 1) % maxv == 0 and (ii + 1) != eq["nbbbs"]:
                        ff.write("\n")
                    kk = kk + 1
                    ff.write("%16.9E" % (eq["zbbbs"][ii]))
                    if (kk + 1) % maxv == 0 and (ii + 1) != eq["nbbbs"]:
                        ff.write("\n")
                    kk = kk + 1
                ff.write("\n")
                kk = 0
                for ii in range(0, eq["limitr"]):
                    ff.write("%16.9E" % (eq["rlim"][ii]))
                    if (kk + 1) % maxv == 0 and (kk + 1) != eq["limitr"]:
                        ff.write("\n")
                    kk = kk + 1
                    ff.write("%16.9E" % (eq["zlim"][ii]))
                    if (kk + 1) % maxv == 0 and (kk + 1) != eq["limitr"]:
                        ff.write("\n")
                    kk = kk + 1
                ff.write("\n")
            print('Output EQDSK file saved as %s.' % (str(eqpath)))

        else:
            print("g-eqdsk could not be written")

        return
    
    def _read_json(self,f_path=None):
        """Read an `Equilibrium` object stored on disk in JSON into a callable `Equilibrium` object

        Args:
            `f_path` (str): path to the location of the desired file, including the file name (!).

        Returns:
            `Equilibrium` object containing the data from the specified JSON file.
        """
        print("Reading Equilibrium {}".format(f_path))
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
                        self.fluxsurfaces[key] = np.array(self.fluxsurfaces[key],dtype=object)

        if 'metadata' in equilibrium_json:
            self.metadata = equilibrium_json['metadata']

        return self

    def _write_json(self,path='./',f_name='Equilibrium.json',metadata=None):
        """Write an `Equilibrium` object to a JSON file on disk

        Args:
            `path` (str, optional): path to the desired location to store the JSON output, [default] the current folder.
            `f_name` (str, optional): the desired file name including the .json extension (!), [default] 'Equilibrium.json'.
            `metadata` (dict, optional): metadata relevant to the `Equilibrium` for later reference.

        Returns:
            
        """

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

        json.dump(equilbrium, codecs.open(path+f_name, 'w', encoding='utf-8'), separators=(',', ':'), indent=4)

        print('Generated fusionkit.Equilibrium file at: {}'.format(path+f_name))

        return

    ## physics functions
    def add_derived(self,f_path=None,resolution=129,_interp=None,_diag=None,just_derived=False,incl_fluxsurfaces=False,incl_miller_geo=False):
        """Add quantities derived from the raw `Equilibrium.read_geqdsk()` output, such as phi, rho_pol, rho_tor to the `Equilibrium` object.
        Can also be called directly if `f_path` is defined.

        Args:
            `f_path` (str): path to the eqdsk g-file, including the file name (!)
            `resolution` (int): the number of desired `Equilibrium` grid points, if the native resolution is lower than this value, it is refined using the `refine()` method
            `just_derived` (bool): [True] return only the derived quantities dictionary, or [False, default] return the `Equilibrium` object
            `incl_fluxsurfaces` (bool): include fluxsurface tracing output in the added derived quantities
            `incl_miller_geo` (bool): include the symmetrised fluxsurface Miller shaping parameters. Defaults to False.

        Returns:
            self or dict if just_derived

        Raises:
            ValueError: Raises an exception when `Equilibrium.raw` is empty and no `f_path` is provided
        """

        print('Adding derived quantities to Equilibrium...')

        if self.raw == {}:
            try:
                self.raw= self.read_eqdsk(f_path=f_path,just_data=True)
            except:
                raise ValueError('Unable to read provided EQDSK file, check file and/or path')

        # introduce shorthands for data and derived locations for increased readability
        raw = self.raw
        derived = self.derived
        fluxsurfaces = self.fluxsurfaces

        # compute R and Z grid vectors
        derived['R'] = np.array([raw['rleft'] + i*(raw['rdim']/(raw['nw']-1)) for i in range(raw['nw'])])
        derived['Z'] = np.array([raw['zmid'] - 0.5*raw['zdim'] + i*(raw['zdim']/(raw['nh']-1)) for i in range(raw['nh'])])

        # equidistant psi grid
        derived['psi'] = np.linspace(raw['simag'],raw['sibry'],raw['nw'])

        # corresponding rho_pol grid
        psi_norm = (derived['psi'] - raw['simag'])/(raw['sibry'] - raw['simag'])
        derived['rho_pol'] = np.sqrt(psi_norm)

        if 'rbbbs' in raw and 'zbbbs' in raw:
            # ensure the boundary coordinates are stored from midplane lfs to midplane hfs
            i_split = find(np.max(raw['rbbbs']),self.raw['rbbbs'])
            derived['rbbbs'] = np.hstack((raw['rbbbs'][i_split:],raw['rbbbs'][:i_split]))
            derived['zbbbs'] = np.hstack((raw['zbbbs'][i_split:],raw['zbbbs'][:i_split]))
            
            # find the indexes of 'zmaxis' on the high field side (hfs) and low field side (lfs) of the separatrix
            i_zmaxis_hfs = int(len(derived['zbbbs'])/3)+find(raw['zmaxis'],derived['zbbbs'][int(len(derived['zbbbs'])/3):int(2*len(derived['zbbbs'])/3)])
            i_zmaxis_lfs = int(2*len(derived['zbbbs'])/3)+find(raw['zmaxis'],derived['zbbbs'][int(2*len(derived['zbbbs'])/3):])
            
            # find the index of 'zmaxis' in the R,Z grid
            i_zmaxis = find(raw['zmaxis'],derived['Z'])

            # find indexes of separatrix on HFS, magnetic axis, separatrix on LFS in R
            i_R_hfs = find(derived['rbbbs'][i_zmaxis_hfs],derived['R'][:int(len(derived['R'])/2)])
            i_rmaxis = find(raw['rmaxis'],derived['R'])
            i_R_lfs = int(len(derived['R'])/2)+find(derived['rbbbs'][i_zmaxis_lfs],derived['R'][int(len(derived['R'])/2):])

            # HFS and LFS R and psirz
            R_hfs = derived['R'][i_R_hfs:i_rmaxis]
            R_lfs = derived['R'][i_rmaxis:i_R_lfs]
            psirzmaxis_hfs = raw['psirz'][i_zmaxis,i_R_hfs:i_rmaxis]
            psirzmaxis_lfs = raw['psirz'][i_zmaxis,i_rmaxis:i_R_lfs]

            # nonlinear R grid at 'zmaxis' based on equidistant psi grid for 'fpol', 'pres', 'ffprim', 'pprime' and 'qpsi'
            derived['R_psi_hfs'] = interpolate.interp1d(psirzmaxis_hfs,R_hfs,fill_value='extrapolate')(derived['psi'][::-1])
            derived['R_psi_lfs'] = interpolate.interp1d(psirzmaxis_lfs,R_lfs,fill_value='extrapolate')(derived['psi'])
        
            # find the R,Z values of the x-point, !TODO: should add check for second x-point in case of double-null equilibrium
            i_xpoint_Z = find(np.min(derived['zbbbs']),derived['zbbbs']) # assuming lower null, JET-ILW shape for now
            derived['R_x'] = derived['rbbbs'][i_xpoint_Z]
            derived['Z_x'] = derived['zbbbs'][i_xpoint_Z]

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
        psirz_norm = abs(raw['psirz'] - raw['simag'])/(raw['sibry'] - raw['simag'])
        derived['rhorz_pol'] = np.sqrt(psirz_norm)

        derived['phirz'] = interpolate.interp1d(derived['psi'],derived['phi'],kind=5,bounds_error=False)(raw['psirz'])
        """
        # repair nan values in phirz, first find the indexes of the nan values
        ij_nan = np.argwhere(np.isnan(derived['phirz']))
        print(ij_nan)
        for _nan in ij_nan:
            i_nan = _nan[0]
            j_nan = _nan[1]
            if j_nan !=0:
                j_nan_min = j_nan-1
            else:
                j_nan_min = j_nan+1
            if j_nan != raw['nh']-1:
                j_nan_plus = j_nan+1
            else:
                j_nan_plus = j_nan-1
            # cycle through the nan values and compute a weighted sum of the last and earliest non-nan values
            derived['phirz'][i_nan,j_nan] = 0.5*(derived['phirz'][i_nan,j_nan_min]+derived['phirz'][i_nan,j_nan_plus]) 
        """

        phirz_norm = abs(derived['phirz']/(derived['phi'][-1]))
        derived['rhorz_tor'] = np.sqrt(phirz_norm)

        # compute the toroidal magnetic field and current density
        derived['B_tor'] = raw['ffprim']/derived['R']
        derived['j_tor'] = derived['R']*raw['pprime']+derived['B_tor']

        if incl_fluxsurfaces:
            self.add_fluxsurfaces(raw=raw,derived=derived,fluxsurfaces=fluxsurfaces,resolution=resolution,_interp=_interp,_diag=_diag,incl_miller_geo=incl_miller_geo)
              
        if just_derived:
            return self.raw['derived']
        else:
            return self

    def add_fluxsurfaces(self,raw=None,derived=None,fluxsurfaces=None,resolution=None,_interp=None,_diag=None,incl_miller_geo=False):
        """Add flux surfaces to an `Equilibrium`.
        
        Args:
            `raw` (dict, optional):  the raw `Equilibrium` data, [default] self.raw if None is set.
            `derived` (dict, optional): the derived `Equilibrium` quantities, [default] self.derived if None is set.
            `fluxsurfaces` (dict, optional): the `Equilibrium` flux surface data, each key a variable containing an array, [default] self.fluxsurfaces if None is set.
            `incl_miller_geo` (bool, optional): [True] include the flux surface Miller shaping parameters delta, kappa and zeta, or [False, default] not.

        Returns:
            self.
        """
        print('Adding fluxsurfaces to Equilibrium...')

        # check if self.fluxsurfaces contains all the flux surfaces specified by derived['rho_tor'] already
        if self.fluxsurfaces and self.derived and len(self.fluxsurfaces['rho_tor']) == len(self.derived['rho_tor']):
            # skip
            print('Skipped adding flux surfaces to Equilibrium as it already contains fluxsurfaces')
        else:
            # set the default locations if None is specified
            if raw is None:
                raw = self.raw
            if derived is None:
                derived = self.derived
                if not self.derived:
                    self.add_derived()
            if fluxsurfaces is None:
                fluxsurfaces = self.fluxsurfaces

            if resolution is None:
                if 'nw' in self.raw:
                    resolution = self.raw['nw']
            if self.raw['nw']<resolution:
                self.refine(nw=resolution,self_consistent=False)
                self.add_derived()
            
            R = copy.deepcopy(derived['R'])
            Z = copy.deepcopy(derived['Z'])
            psirz = copy.deepcopy(raw['psirz'])

            if _interp:
                # refine the psi R,Z grid by <interp>x to get smooth(er) gradients for geometric quantities
                refine = _interp*self.raw['nw']
                R = np.linspace(self.derived['R'][0],self.derived['R'][-1],refine)
                Z = np.linspace(self.derived['Z'][0],self.derived['Z'][-1],refine)
                _R,_Z = np.meshgrid(R,Z)
                r,z = np.meshgrid(self.derived['R'],self.derived['Z'])
                points = np.column_stack((r.flatten(),z.flatten()))
                psirz = interpolate.griddata(points,self.raw['psirz'].flatten(),(_R,_Z),method='cubic')

                '''
                refine = _interp*self.raw['nw']
                R = np.linspace(self.derived['R'][0],self.derived['R'][-1],refine)
                Z = np.linspace(self.derived['Z'][0],self.derived['Z'][-1],refine)
                psirz = interpolate.interp2d(self.derived['R'],self.derived['Z'],self.raw['psirz'])(R,Z)
                '''

            if _diag == 'mesh':
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                R_,Z_ = np.meshgrid(R,Z)
                ax.plot_wireframe(R_,Z_,psirz, rstride=10, cstride=10)
                ax.set_xlabel('R [m]')
                ax.set_ylabel('Z [m]')
                ax.set_zlabel('$\\Psi$')
                plt.show()

            # find the approximate location of the magnetic axis on the psirz map
            if self.raw['sibry'] > self.raw['simag']:
                i_rmaxis = np.where(psirz == np.min(psirz))[1][0]
                i_zmaxis = np.where(psirz == np.min(psirz))[0][0]
            elif self.raw['sibry'] < self.raw['simag']:
                i_rmaxis = np.where(psirz == np.max(psirz))[1][0]
                i_zmaxis = np.where(psirz == np.max(psirz))[0][0]

            if _diag:
                plt.figure()
                if _diag == 'fs':
                    plt.plot(raw['rmaxis'],raw['zmaxis'],'bx')
            '''
            plt.imshow(psirz,origin='lower',extent=[R[0],R[-1],Z[0],Z[-1]],)
            ax = plt.gca()
            ax.set_xticks(np.arange(R[0], R[-1],R[1]-R[0]))
            ax.set_yticks(np.arange(Z[0], Z[-1],Z[1]-Z[0]))
            ax.grid(color='black', linestyle='-.', linewidth=1)
            '''
            # add the flux surface data for rho_tor > 0
            for rho_fs in self.derived['rho_tor'][1:]:
                # print a progress %
                stdout.write('\r {}% completed'.format(round(100*(find(rho_fs,self.derived['rho_tor'][1:])+1)/len(self.derived['rho_tor'][1:]))))
                stdout.flush()
                # check that rho stays inside the lcfs
                if rho_fs < 0.999:
                    self.fluxsurface_find(x_fs=rho_fs,R=R,Z=Z,psirz=psirz,i_maxis=[i_rmaxis,i_zmaxis],_diag=_diag,incl_miller_geo=incl_miller_geo,return_self=True)
            stdout.write('\n')
            if _diag == 'fs':
                plt.show()

            if 'rbbbs' in raw and 'zbbbs' in raw:
                # find the geometric center, minor radius and extrema of the lcfs manually
                lcfs = self.fluxsurface_center(psi_fs=raw['sibry'],R_fs=derived['rbbbs'],Z_fs=derived['zbbbs'],incl_extrema=True)
                lcfs.update({'R':derived['rbbbs'],'Z':derived['zbbbs']})
            else:
                lcfs = self.fluxsurface_find(x_fs=1.0,psi_fs=raw['sibry'],psirz=psirz,R=R,Z=Z,i_maxis=[i_rmaxis,i_zmaxis],interp_method='bounded_extrapolation',return_self=False)
                derived.update({'rbbbs':lcfs['R'],'zbbbs':lcfs['Z'],'nbbbs':len(lcfs['R'])})
            if incl_miller_geo:
                lcfs = self.fluxsurface_miller_geo(fs=lcfs)

            # add a zero at the start of all flux surface quantities and append the lcfs values to the end of the flux surface data
            for key in fluxsurfaces:
                if key in ['R','R0']:
                    fluxsurfaces[key].insert(0,raw['rmaxis'])
                elif key in ['Z','Z0']:
                    fluxsurfaces[key].insert(0,raw['zmaxis'])
                elif key in ['kappa','delta','zeta','s_kappa','s_delta','s_zeta','R_top','R_bottom','Z_top','Z_bottom','R_sym_top','R_sym_bottom','Z_sym_top','Z_sym_bottom']:
                    fluxsurfaces[key].insert(0,fluxsurfaces[key][0])
                else:
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
            derived['r/a'] = derived['r']/derived['a']

            # add the midplane average major radius and elevation derivatives to derived
            derived['dRodr'] = np.gradient(derived['Ro'],derived['r'])
            derived['dZodr'] = np.gradient(derived['Zo'],derived['r'])

            # add the magnetic shear to derived
            derived['s'] = derived['r']*np.gradient(np.log(raw['qpsi']),derived['r'],edge_order=2)

            # add several magnetic field quantities to derived
            derived['Bref_eqdsk'] = raw['fpol'][0]/raw['rmaxis']
            derived['Bref_miller'] = raw['fpol']/derived['Ro']
            #derived['B_unit'] = interpolate.interp1d(derived['r'],(1/derived['r'])*np.gradient(derived['phi'],derived['r'],edge_order=2))(derived['r'])
            with np.errstate(divide='ignore'):
                derived['B_unit'] = interpolate.interp1d(derived['r'],(raw['qpsi']/derived['r'])*np.gradient(derived['psi'],derived['r']))(derived['r'])
            
            # add beta and alpha, assuming the pressure profile included in the equilibrium and Bref=derived['Bref_eqdsk]
            derived['p'] = raw['pres']
            derived['beta'] = 8*np.pi*1E-7*derived['p']/(derived['Bref_eqdsk']**2)
            derived['alpha'] = -1*raw['qpsi']**2*derived['Ro']*np.gradient(derived['beta'],derived['r'])

            if incl_miller_geo:
                # add the symmetrised flux surface trace arrays to derived
                if 'R_sym' and 'Z_sym' in fluxsurfaces:
                    derived['R_sym'] = fluxsurfaces['R_sym']
                    derived['Z_sym'] = fluxsurfaces['Z_sym']

                # add the Miller shaping parameters to derived and smooth them using a Savitzky-Golay filter with a window length of 11 and second order polynomial 
                derived['kappa'] = np.array(fluxsurfaces['kappa'])
                derived['delta'] = np.array(fluxsurfaces['delta'])
                derived['zeta'] = np.array(fluxsurfaces['zeta'])

                # compute the shear of the Miller shaping parameters
                with np.errstate(divide='ignore'):
                    derived['s_kappa'] = derived['r']*np.gradient(np.log(derived['kappa']),derived['r'],edge_order=2)
                    derived['s_delta'] = (derived['r']/np.sqrt(1-derived['delta']**2))*np.gradient(derived['delta'],derived['r'],edge_order=2)
                derived['s_delta_ga'] = derived['r']*np.gradient(derived['delta'],derived['r'],edge_order=2)
                derived['s_zeta'] = derived['r']*np.gradient(derived['zeta'],derived['r'],edge_order=2)
            
            return self

    def fluxsurface_find(self,psi_fs=None,psi=None,x_fs=None,x=None,x_label='rho_tor',R=None,Z=None,psirz=None,i_maxis=None,interp_method='normal',_diag=None,incl_miller_geo=False,return_self=False):
        """Find the R,Z trace of a flux surface.

        Args:
            `psi_fs` (float, optional): the poloidal flux value of the flux surface.
            `psi` (array, optional): vector containing the poloidal flux psi from axis to separatrix.
            `x_fs` (float): the radial flux label of the flux surface, [default] assumed to be in rho_tor.
            `x` (array, optional): vector of the radial flux surface label on the same grid as psi, [default] assumed to be rho_tor.
            `x_label` (str, optional): the radial flux label, options: [default] 'rho_tor', 'rho_pol', 'psi' and 'r'.
            `R` (array): vector of R grid.
            `Z` (array): vector of Z grid.
            `psirz` (array): a R,Z map of the poloidal flux psi of the magnetic equilibrium.
            `i_maxis` (list or array, optional): the indexes of the approximate magnetic axis in psriz to speed up the tracing calculation in a loop.
            `incl_miller_geo` (bool, optional): [True] to include the symmetrised flux surface Miller shaping parameters delta, kappa and zeta, or [False default] not.
            `return_self` (bool, optional): [True] to return the result to the `Equilibrium` object, or [False, default] as a standalone dictionary.

        Returns:
            [default] self, or if `return_self` a dict containing the R, Z and psi values of the flux surface trace.

        """
        if x_label in self.derived  and 'psi' in self.derived:
            # find the flux of the selected flux surface
            x = self.derived[x_label]
            psi = self.derived['psi']
        else:
            raise SyntaxError('Equilibrium.fluxsurface_find error: Not enough inputs provided to determine psi of the flux surface, check your inputs!')

        if x_fs != None:
            if psi_fs == None:
                psi_fs = interpolate.interp1d(x,psi,kind='cubic')(x_fs)
        else:
            raise SyntaxError('Equilibrium.fluxsurface_find error: No radial position of the flux surface was specified, check your inputs!')

        # define storage for flux surface coordinates
        RZ_fs = {'hfs':{'top':[],'bottom':[]},'lfs':{'top':[],'bottom':[]}}

        # take/find the approximate location of the magnetic axis on the psirz map
        if i_maxis is not None:
            i_rmaxis = i_maxis[0]
            i_zmaxis = i_maxis[1]
        else:
            if self.raw['sibry'] > self.raw['simag']:
                i_rmaxis = np.where(psirz == np.min(psirz))[1][0]
                i_zmaxis = np.where(psirz == np.min(psirz))[0][0]
            elif self.raw['sibry'] < self.raw['simag']:
                i_rmaxis = np.where(psirz == np.max(psirz))[1][0]
                i_zmaxis = np.where(psirz == np.max(psirz))[0][0]

        # find the vertical extrema of the LCFS at the major radius of the magnetic axis
        if self.raw['sibry'] > self.raw['simag']:
            i_psiz_rmaxis_split = np.argmin(psirz[:,i_rmaxis])
            i_psiz_rmaxis_bottom_max = np.argmax(psirz[:i_psiz_rmaxis_split,i_rmaxis])
            i_psiz_rmaxis_top_max = i_psiz_rmaxis_split+np.argmax(psirz[i_psiz_rmaxis_split:,i_rmaxis])
        elif self.raw['sibry'] < self.raw['simag']:
            i_psiz_rmaxis_split = np.argmax(psirz[:,i_rmaxis])
            i_psiz_rmaxis_bottom_max = np.argmin(psirz[:i_psiz_rmaxis_split,i_rmaxis])
            i_psiz_rmaxis_top_max = i_psiz_rmaxis_split+np.argmin(psirz[i_psiz_rmaxis_split:,i_rmaxis])

        psiz_rmaxis_bottom = psirz[i_psiz_rmaxis_bottom_max:i_psiz_rmaxis_split,i_rmaxis]
        psiz_rmaxis_top = psirz[i_psiz_rmaxis_split:i_psiz_rmaxis_top_max,i_rmaxis]
        '''# patch for rare cases where the psirz map flattens off below the value of psi at the boundary (e.g. some ESCO EQDSKs)
        if not np.any(psiz_rmaxis_bottom >= psi[-1]):
            dpsiz_rmaxis_bottom_dr = np.gradient(psiz_rmaxis_bottom)
            if np.any(dpsiz_rmaxis_bottom_dr==0.):
                i_dpsiz_rmaxis_bottom_dr = np.where(dpsiz_rmaxis_bottom_dr==0.)[0][-1]+1
                i_psiz_rmaxis_bottom_max += i_dpsiz_rmaxis_bottom_dr
                psiz_rmaxis_bottom = psiz_rmaxis_bottom[i_dpsiz_rmaxis_bottom_dr:]

        # patch for rare cases where the psirz map flattens off below the value of psi at the boundary (e.g. some ESCO EQDSKs)
        if not np.any(psiz_rmaxis_top >= psi[-1]):
            dpsiz_rmaxis_top_dr = np.gradient(psiz_rmaxis_top)
            if np.any(dpsiz_rmaxis_top_dr==0.):
                i_dpsiz_rmaxis_top_dr = np.where(dpsiz_rmaxis_top_dr==0.)[0][0]-1
                i_psiz_rmaxis_top_max = i_psiz_rmaxis_split+i_dpsiz_rmaxis_top_dr
                psiz_rmaxis_top = psiz_rmaxis_top[:i_dpsiz_rmaxis_top_dr]'''

        Z_lcfs_max = interpolate.interp1d(psiz_rmaxis_top,Z[i_psiz_rmaxis_split:i_psiz_rmaxis_top_max],bounds_error=False,fill_value='extrapolate')(psi[-1])
        Z_lcfs_min = interpolate.interp1d(psiz_rmaxis_bottom,Z[i_psiz_rmaxis_bottom_max:i_psiz_rmaxis_split],bounds_error=False,fill_value='extrapolate')(psi[-1])

        if self.raw['sibry'] > self.raw['simag']:
            i_psiz_zmaxis_split = np.argmin(psirz[i_zmaxis,:])
            i_psiz_zmaxis_bottom_max = np.argmax(psirz[i_zmaxis,:i_psiz_zmaxis_split])
            i_psiz_zmaxis_top_max = i_psiz_rmaxis_split+np.argmax(psirz[i_zmaxis,i_psiz_zmaxis_split:])
        elif self.raw['sibry'] < self.raw['simag']:
            i_psiz_zmaxis_split = np.argmax(psirz[i_zmaxis,:])
            i_psiz_zmaxis_bottom_max = np.argmin(psirz[i_zmaxis,:i_psiz_zmaxis_split])
            i_psiz_zmaxis_top_max = i_psiz_rmaxis_split+np.argmin(psirz[i_zmaxis,i_psiz_zmaxis_split:])

        psiz_zmaxis_bottom = psirz[i_zmaxis,i_psiz_zmaxis_bottom_max:i_psiz_zmaxis_split]
        psiz_zmaxis_top = psirz[i_zmaxis,i_psiz_zmaxis_split:i_psiz_zmaxis_top_max]

        # diagnostic plot to check interpolation issues for the vertical bounds
        if _diag == 'bounds':
            plt.figure()
            plt.plot(Z[i_psiz_rmaxis_split:i_psiz_rmaxis_top_max],psiz_rmaxis_top,'.-')
            plt.plot(Z[i_psiz_rmaxis_bottom_max:i_psiz_rmaxis_split],psiz_rmaxis_bottom,'.-')
            plt.plot(R[i_psiz_zmaxis_split:i_psiz_zmaxis_top_max],psiz_zmaxis_top,'.-')
            plt.plot(R[i_psiz_zmaxis_bottom_max:i_psiz_zmaxis_split],psiz_zmaxis_bottom,'.-')
            plt.axhline(self.raw['sibry'],ls='dashed',color='black')
            plt.show()

        R_lcfs_max = interpolate.interp1d(psiz_zmaxis_top,R[i_psiz_zmaxis_split:i_psiz_zmaxis_top_max],bounds_error=False,fill_value='extrapolate')(psi[-1])
        R_lcfs_min = interpolate.interp1d(psiz_zmaxis_bottom,R[i_psiz_zmaxis_bottom_max:i_psiz_zmaxis_split],bounds_error=False,fill_value='extrapolate')(psi[-1])

        # set the starting coordinates for the flux surface tracing algorithm
        i, j = i_zmaxis, i_zmaxis
        k, l = i_rmaxis, i_rmaxis

        # find the psi value corresponding to the the current x coordinate
        psi_fs = interpolate.interp1d(x,psi,kind='cubic')(x_fs)

        # while the psi_fs intersects with the current psirz slice gather the intersection coordinates
        while (psi_fs > np.min(psirz[i]) and i < psirz.shape[0]-1):
            if Z[i] <= Z_lcfs_max:
                # find the split in the Z slice of psirz
                if self.raw['sibry'] > self.raw['simag']:
                    i_psir_slice_split = np.argmin(psirz[i])
                elif self.raw['sibry'] < self.raw['simag']:
                    i_psir_slice_split = np.argmax(psirz[i])

                # chop the psirz and R slices in two parts to separate the HFS and LFS
                psir_slice_top_hfs = psirz[i,:i_psir_slice_split]
                psir_slice_top_lfs = psirz[i,i_psir_slice_split:]

                # interpolate the R coordinate of the top half HFS and LFS
                if interp_method == 'normal':
                    R_fs_top_hfs = float(interpolate.interp1d(psir_slice_top_hfs,R[i_psir_slice_split-len(psir_slice_top_hfs):i_psir_slice_split],bounds_error=False)(psi_fs))
                    R_fs_top_lfs = float(interpolate.interp1d(psir_slice_top_lfs,R[i_psir_slice_split:i_psir_slice_split+len(psir_slice_top_lfs)],bounds_error=False)(psi_fs))
                # if the normal method provides spikey flux surface traces, bound the interpolation domain and extrapolate the intersection
                elif interp_method == 'bounded_extrapolation':
                    psir_slice_top_hfs = psir_slice_top_hfs[psir_slice_top_hfs<=self.raw['sibry']]
                    psir_slice_top_lfs = psir_slice_top_lfs[psir_slice_top_lfs<=self.raw['sibry']]
                    
                    R_fs_top_hfs = float(interpolate.interp1d(psir_slice_top_hfs,R[i_psir_slice_split-len(psir_slice_top_hfs):i_psir_slice_split][psir_slice_top_hfs<=self.raw['sibry']],bounds_error=False,fill_value='extrapolate')(psi_fs))
                    R_fs_top_lfs = float(interpolate.interp1d(psir_slice_top_lfs,R[i_psir_slice_split:i_psir_slice_split+len(psir_slice_top_lfs)][psir_slice_top_lfs<=self.raw['sibry']],bounds_error=False,fill_value='extrapolate')(psi_fs))
                # diagnostic plot to check interpolation issues for rows
                if _diag == 'interp':
                    plt.figure()
                    plt.plot(psir_slice_top_hfs,R[i_psir_slice_split-len(psir_slice_top_hfs):i_psir_slice_split],'b-',label='HFS, top')
                    plt.plot(psir_slice_top_lfs,R[i_psir_slice_split:i_psir_slice_split+len(psir_slice_top_lfs)],'r-',label='LFS, top')
                    plt.axvline(psi_fs,ls='dashed',color='black')
                
                # insert the coordinate pairs into the flux surface trace dict if not nan (bounds error) and order properly for merging later
                if not np.isnan(R_fs_top_hfs):
                    RZ_fs['hfs']['top'].append((R_fs_top_hfs,Z[i]))
                    if R_fs_top_hfs > R_lcfs_max:
                        R_lcfs_max = R_fs_top_hfs
                if not np.isnan(R_fs_top_lfs):
                    RZ_fs['lfs']['top'].append((R_fs_top_lfs,Z[i]))
                    if R_fs_top_lfs > R_lcfs_max:
                        R_lcfs_max = R_fs_top_lfs
            
            # update the slice coordinates
            if i < psirz.shape[0]-1:
                i+=1
            
        # while the psi_fs intersects with the current psirz slice gather the intersection coordinates
        while (psi_fs > np.min(psirz[:,k]) and k < psirz.shape[1]-1):
            if R[k] <= R_lcfs_max:
                # find the split and the extrema of the R slice of psirz
                if self.raw['sibry'] > self.raw['simag']:
                    k_psiz_slice_split = np.argmin(psirz[:,k])
                    k_psiz_slice_bottom_max = np.argmax(psirz[:k_psiz_slice_split,k])
                    k_psiz_slice_top_max = k_psiz_slice_split+np.argmax(psirz[k_psiz_slice_split:,k])
                elif self.raw['sibry'] < self.raw['simag']:
                    k_psiz_slice_split = np.argmax(psirz[:,k])
                    k_psiz_slice_bottom_max = np.argmin(psirz[:k_psiz_slice_split,k])
                    k_psiz_slice_top_max = k_psiz_slice_split+np.argmin(psirz[k_psiz_slice_split:,k])

                # chop the psirz slices in two parts to separate the top and bottom halves
                psiz_slice_bottom_lfs = psirz[k_psiz_slice_bottom_max:k_psiz_slice_split,k]
                psiz_slice_top_lfs = psirz[k_psiz_slice_split:k_psiz_slice_top_max,k]
                
                # interpolate the Z coordinate for the LFS top and bottom
                if interp_method == 'normal':
                    Z_fs_bottom_lfs = float(interpolate.interp1d(psiz_slice_bottom_lfs,Z[k_psiz_slice_bottom_max:k_psiz_slice_split],bounds_error=False)(psi_fs))
                    Z_fs_top_lfs = float(interpolate.interp1d(psiz_slice_top_lfs,Z[k_psiz_slice_split:k_psiz_slice_top_max],bounds_error=False)(psi_fs))
                elif interp_method == 'bounded_extrapolation':
                    Z_slice_bottom_lfs = Z[k_psiz_slice_bottom_max:k_psiz_slice_split][psiz_slice_bottom_lfs<=self.raw['sibry']]
                    Z_slice_top_lfs = Z[k_psiz_slice_split:k_psiz_slice_top_max][psiz_slice_top_lfs<=self.raw['sibry']]
                    psiz_slice_bottom_lfs = psiz_slice_bottom_lfs[psiz_slice_bottom_lfs<=self.raw['sibry']]
                    psiz_slice_top_lfs = psiz_slice_top_lfs[psiz_slice_top_lfs<=self.raw['sibry']]

                    Z_fs_bottom_lfs = float(interpolate.interp1d(psiz_slice_bottom_lfs,Z_slice_bottom_lfs,bounds_error=False,fill_value='extrapolate')(psi_fs))
                    Z_fs_top_lfs = float(interpolate.interp1d(psiz_slice_top_lfs,Z_slice_top_lfs,bounds_error=False,fill_value='extrapolate')(psi_fs))
                # diagnostic plot to check interpolation issues for columns
                if _diag == 'interp':
                    plt.plot(psiz_slice_bottom_lfs,Z[k_psiz_slice_bottom_max:k_psiz_slice_split],'b--',label='LFS, bottom')
                    plt.plot(psiz_slice_top_lfs,Z[k_psiz_slice_split:k_psiz_slice_top_max],'r--',label='LFS, top')
                    plt.legend()

                if not np.isnan(Z_fs_bottom_lfs):
                    RZ_fs['lfs']['bottom'].append((R[k],Z_fs_bottom_lfs))
                    # update the higher vertical bound if it is higher than the vertical bound found at the magnetic axis
                    if Z_fs_bottom_lfs < Z_lcfs_min:
                        Z_lcfs_min = Z_fs_bottom_lfs
                if not np.isnan(Z_fs_top_lfs):
                    RZ_fs['lfs']['top'].append((R[k],Z_fs_top_lfs))
                    # update the lower vertical bound if it is lower than the vertical bound found at the magnetic axis
                    if Z_fs_top_lfs > Z_lcfs_max:
                        Z_lcfs_max = Z_fs_top_lfs

            # update the slice coordinates
            if k < psirz.shape[1]-1:
                k+=1

        # while the psi_fs intersects with the current psirz slice gather the intersection coordinates
        while (psi_fs > np.min(psirz[j]) and j > 0):
            # interpolate the R coordinate of the bottom half of the flux surface on both the LFS and HFS
            if Z[j] >= Z_lcfs_min:
                # find the split and the extrema of the Z slice of psirz
                if self.raw['sibry'] > self.raw['simag']:
                    j_psir_slice_split = np.argmin(psirz[j])
                elif self.raw['sibry'] < self.raw['simag']:
                    j_psir_slice_split = np.argmax(psirz[j])

                psir_slice_bottom_hfs = psirz[j,:j_psir_slice_split]
                psir_slice_bottom_lfs = psirz[j,j_psir_slice_split:]

                # interpolate the R coordinate of the bottom half HFS and LFS
                if interp_method == 'normal':
                    R_fs_bottom_hfs = float(interpolate.interp1d(psir_slice_bottom_hfs,R[j_psir_slice_split-len(psir_slice_bottom_hfs):j_psir_slice_split],bounds_error=False)(psi_fs))
                    R_fs_bottom_lfs = float(interpolate.interp1d(psir_slice_bottom_lfs,R[j_psir_slice_split:j_psir_slice_split+len(psir_slice_bottom_lfs)],bounds_error=False)(psi_fs))
                elif interp_method == 'bounded_extrapolation':
                    psir_slice_bottom_hfs = psir_slice_bottom_hfs[psir_slice_bottom_hfs<=self.raw['sibry']]
                    psir_slice_bottom_lfs = psir_slice_bottom_lfs[psir_slice_bottom_lfs<=self.raw['sibry']]
                    R_fs_bottom_hfs = float(interpolate.interp1d(psir_slice_bottom_hfs,R[j_psir_slice_split-len(psir_slice_bottom_hfs):j_psir_slice_split][psir_slice_bottom_hfs<=self.raw['sibry']],bounds_error=False,fill_value='extrapolate')(psi_fs))
                    R_fs_bottom_lfs = float(interpolate.interp1d(psir_slice_bottom_lfs,R[j_psir_slice_split:j_psir_slice_split+len(psir_slice_bottom_lfs)][psir_slice_bottom_lfs<=self.raw['sibry']],bounds_error=False,fill_value='extrapolate')(psi_fs))
                # diagnostic plot to check interpolation issues for rows
                if _diag == 'interp':
                    plt.figure()
                    plt.plot(psir_slice_bottom_hfs,R[j_psir_slice_split-len(psir_slice_bottom_hfs):j_psir_slice_split],'b-',label='HFS, bottom')
                    plt.plot(psir_slice_bottom_lfs,R[j_psir_slice_split:j_psir_slice_split+len(psir_slice_bottom_lfs)],'r-',label='LFS, bottom')
                    plt.axvline(psi_fs,ls='dashed',color='black')

                if not np.isnan(R_fs_bottom_hfs):
                    RZ_fs['hfs']['bottom'].append((R_fs_bottom_hfs,Z[j]))
                    if R_fs_bottom_hfs < R_lcfs_min:
                        R_lcfs_min = R_fs_bottom_hfs
                if not np.isnan(R_fs_bottom_lfs):
                    RZ_fs['lfs']['bottom'].append((R_fs_bottom_lfs,Z[j]))
                    if R_fs_bottom_lfs < R_lcfs_min:
                        R_lcfs_min = R_fs_bottom_lfs
            # update the slice coordinates
            if j > 0:
                j-=1
                
        # while the psi_fs intersects with the current psirz slice gather the intersection coordinates
        while (psi_fs > np.min(psirz[:,l]) and l > 0):
            # interpolate the R coordinate of the bottom half of the flux surface on both the LFS and HFS
            if R[l] >= R_lcfs_min:
                # split the current column of the psirz map in top and bottom
                # find the split and the extrema of the Z slice of psirz
                if self.raw['sibry'] > self.raw['simag']:
                    l_psiz_slice_split = np.argmin(psirz[:,l])
                    l_psiz_slice_bottom_max = np.argmax(psirz[:l_psiz_slice_split,l])
                    l_psiz_slice_top_max = l_psiz_slice_split+np.argmax(psirz[l_psiz_slice_split:,l])
                elif self.raw['sibry'] < self.raw['simag']:
                    l_psiz_slice_split = np.argmax(psirz[:,l])
                    l_psiz_slice_bottom_max = np.argmin(psirz[:l_psiz_slice_split,l])
                    l_psiz_slice_top_max = l_psiz_slice_split+np.argmin(psirz[l_psiz_slice_split:,l])

                psiz_slice_bottom_hfs = psirz[l_psiz_slice_bottom_max:l_psiz_slice_split,l]
                psiz_slice_top_hfs = psirz[l_psiz_slice_split:l_psiz_slice_top_max,l]
                
                # interpolate the Z coordinate for the HFS top and bottom
                if interp_method == 'normal':
                    Z_fs_bottom_hfs = float(interpolate.interp1d(psiz_slice_bottom_hfs,Z[l_psiz_slice_bottom_max:l_psiz_slice_split],bounds_error=False)(psi_fs))
                    Z_fs_top_hfs = float(interpolate.interp1d(psiz_slice_top_hfs,Z[l_psiz_slice_split:l_psiz_slice_top_max],bounds_error=False)(psi_fs))
                elif interp_method == 'bounded_extrapolation':
                    Z_slice_bottom_hfs = Z[l_psiz_slice_bottom_max:l_psiz_slice_split][psiz_slice_bottom_hfs<=self.raw['sibry']]
                    Z_slice_top_hfs = Z[l_psiz_slice_split:l_psiz_slice_top_max][psiz_slice_top_hfs<=self.raw['sibry']]
                    psiz_slice_bottom_hfs = psiz_slice_bottom_hfs[psiz_slice_bottom_hfs<=self.raw['sibry']]
                    psiz_slice_top_hfs = psiz_slice_top_hfs[psiz_slice_top_hfs<=self.raw['sibry']]

                    Z_fs_bottom_hfs = float(interpolate.interp1d(psiz_slice_bottom_hfs,Z_slice_bottom_hfs,bounds_error=False,fill_value='extrapolate')(psi_fs))
                    Z_fs_top_hfs = float(interpolate.interp1d(psiz_slice_top_hfs,Z_slice_top_hfs,bounds_error=False,fill_value='extrapolate')(psi_fs))
                # diagnostic plot to check interpolation issues for columns
                if _diag == 'interp':
                    plt.plot(psiz_slice_bottom_hfs,Z[l_psiz_slice_bottom_max:l_psiz_slice_split],'b--',label='HFS, bottom')
                    plt.plot(psiz_slice_top_hfs,Z[l_psiz_slice_split:l_psiz_slice_top_max],'r--',label='HFS, top')
                    plt.legend()

                if not np.isnan(Z_fs_bottom_hfs):
                    RZ_fs['hfs']['bottom'].append((R[l],Z_fs_bottom_hfs))
                    # update the higher vertical bound if it is higher than the vertical bound found at the magnetic axis
                    if Z_fs_bottom_hfs < Z_lcfs_min:
                        Z_lcfs_min = Z_fs_bottom_hfs
                if not np.isnan(Z_fs_top_hfs):
                    RZ_fs['hfs']['top'].append((R[l],Z_fs_top_hfs))
                    # update the lower vertical bound if it is lower than the vertical bound found at the magnetic axis
                    if Z_fs_top_hfs > Z_lcfs_max:
                        Z_lcfs_max = Z_fs_top_hfs

            # update the slice coordinates
            if l > 0:
                l-=1

        # collate all the traced coordinates of the flux surface and sort by increasing R
        fs_RZ = sorted(RZ_fs['lfs']['top']+RZ_fs['hfs']['top']+RZ_fs['hfs']['bottom']+RZ_fs['lfs']['bottom'])
        
        # find the extrema of the traced coordinates
        Z_R_min = min(fs_RZ) # coordinates for min(R)
        Z_R_max = max(fs_RZ) # coordinates for max(R)
        R_Z_min = min(fs_RZ,key=itemgetter(1)) # coordinates for min(Z)
        R_Z_max = max(fs_RZ,key=itemgetter(1)) # coordinates for max(Z)

        # split all the flux surface coordinates in four sequential quadrants, all sorted by increasing R
        fs_lfs_top = [(r,z) for r,z in fs_RZ if r>=R_Z_max[0] and z>Z_R_max[1]]
        fs_hfs_top = [(r,z) for r,z in fs_RZ if r<R_Z_max[0] and z>=Z_R_min[1]]
        fs_lfs_bottom = [(r,z) for r,z in fs_RZ if r>R_Z_min[0] and z<=Z_R_max[1]]
        fs_hfs_bottom = [(r,z) for r,z in fs_RZ if r<=R_Z_min[0] and z<Z_R_min[1]]

        # diagnostic plot for checking the sorting and glueing of the 
        if _diag == 'trace':
            for x in [R_Z_min,R_Z_max,Z_R_min,Z_R_max]:
                plt.plot(*zip(x),'x')
            plt.plot(*zip(*sorted(fs_lfs_top)),'.-')
            plt.plot(*zip(*sorted(fs_hfs_top)),'.-')
            plt.plot(*zip(*sorted(fs_lfs_bottom)),'.-')
            plt.plot(*zip(*sorted(fs_hfs_bottom)),'.-')
            plt.plot(*zip(*fs_RZ))
            plt.xlabel('R [m]')
            plt.ylabel('Z [m]')
            plt.show()

        # merge the complete flux surface trace, starting at the LFS Rmax,Z_Rmax, 
        RZ_fs = fs_lfs_top[::-1]+fs_hfs_top[::-1]+fs_hfs_bottom+fs_lfs_bottom+[fs_lfs_top[-1]]

        # separate the R and Z coordinates in separate vectors
        R_fs = np.array([R for R,Z in RZ_fs])
        Z_fs = np.array([Z for R,Z in RZ_fs])

        fs = {'R':R_fs,'Z':Z_fs,'psi':psi_fs}

        # diagnostic plot for checking the complete flux surface trace, e.g. in combination with the flux surface extrema fitting
        if _diag == 'fs':
            #plt.figure()
            plt.plot(fs['R'],fs['Z'],'b-')
            if (len(R) > self.raw['nw'] or len(Z) > self.raw['nh']):
                if ((self.raw['simag'] < self.raw['sibry']) and (psi_fs < self.raw['sibry'])) or ((self.raw['simag'] > self.raw['sibry']) and (psi_fs > self.raw['sibry'])):
                    plt.contour(self.derived['R'],self.derived['Z'],self.raw['psirz'],[psi_fs],linestyles='dashed',colors='purple')
            plt.axis('scaled')
            plt.xlabel('R [m]')
            plt.ylabel('Z [m]')

        # find the flux surface center quantities and add them to the flux surface dict
        fs.update(self.fluxsurface_center(psi_fs=psi_fs,R_fs=fs['R'],Z_fs=fs['Z'],_diag=_diag,incl_extrema=True))

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

    def fluxsurface_center(self,psi_fs=None,R_fs=None,Z_fs=None,_diag='none',incl_extrema=False,return_self=False):
        """Find the geometric center of a flux surface trace defined by R_fs,Z_fs and psi_fs.

        Args:
            `psi_fs` (float): the poloidal flux value of the flux surface.
            `R_fs` (array): vector of the horizontal coordinates of the flux surface trace.
            `Z_fs` (array): vector of the vertical coordinates of the flux surface trace.
            `psirz` (array): the R,Z map of the poloidal flux psi of the magnetic equilibrium.
            `R` (array): vector of R grid.
            `Z` (array): vector of Z grid.
            `incl_extrema` (bool): [True] include the extrema data in the returned dict, or [False, default] to leave it separate.
            `return_self` (bool): [True] to return the result to the `Equilibrium`, or [False, default] as a standalone dictionary.

        Returns:
            [default] A dict with the flux surface R, Z, psi, r, R0, Z0 and (optional) extrema, or if `return_self` add the flux surface data to Equilibrium.fluxsurfaces
        """

        # create temporary flux surface storage dict
        fs = {}

        # find the average elevation (midplane) of the flux surface [Candy PPCF 51 (2009) 105009]
        fs['Z0'] = integrate.trapz(R_fs*Z_fs,Z_fs)/integrate.trapz(R_fs,Z_fs)
        #print(fs['Z0'])

        # find the extrema of the flux surface in the radial direction at the average elevation
        fs_extrema = self.fluxsurface_extrema(psi_fs=psi_fs,R_fs=R_fs,Z_fs=Z_fs,Z0_fs=fs['Z0'],_diag=_diag)
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

    def fluxsurface_extrema(self,psi_fs=None,R_fs=None,Z_fs=None,Z0_fs=None,_diag='none',return_self=False):
        """Find the extrema in R and Z of a flux surface trace defined by R_fs, Z_fs and psi_fs.

        Args:
            `psi_fs` (float): the poloidal flux value of the flux surface.
            `R_fs` (array): vector of the horizontal coordinates of the flux surface trace.
            `Z_fs` (array): vector of the vertical coordinates of the flux surface trace.
            `Z0_fs` (float): the average elevation of the flux surface.
            `return_self` (bool, optional): [True] boolean to return the result to the Equilibrium object, or [False, default] as a standalone dict.

        Returns:
            [default] dict with the flux surface, or if `return_self` append the flux surface data to Equilibrium.fluxsurfaces
        """

        # create temporary flux surface storage dict
        fs = {}

        # check if the poloidal flux value of the flux surface is provided
        if psi_fs != None:
            # check if the midplane of the flux surface is provided
            if Z0_fs != None:
                # restack R_fs and Z_fs to get a continuous midplane outboard trace
                R_fs_out = np.hstack((R_fs[int(0.9*len(Z_fs)):],R_fs[:int(0.1*len(Z_fs))]))
                Z_fs_out = np.hstack((Z_fs[int(0.9*len(Z_fs)):],Z_fs[:int(0.1*len(Z_fs))]))

                R_fs_in = R_fs[int(len(Z_fs)/2)-int(0.1*len(Z_fs)):int(len(Z_fs)/2)+int(0.1*len(Z_fs))]
                Z_fs_in = Z_fs[int(len(Z_fs)/2)-int(0.1*len(Z_fs)):int(len(Z_fs)/2)+int(0.1*len(Z_fs))]
                
                # find the extrema in R of the flux surface at the midplane
                fs['R_out'] = interpolate.interp1d(Z_fs_out,R_fs_out,bounds_error=False)(Z0_fs)
                fs['R_in'] = interpolate.interp1d(Z_fs_in,R_fs_in,bounds_error=False)(Z0_fs)

                # in case psi_fs is out of bounds in these interpolations
                if np.isnan(fs['R_out']) or np.isinf(fs['R_out']):
                    # restack R_fs to get continuous trace on outboard side
                    R_fs_ = np.hstack((R_fs[np.argmin(R_fs):],R_fs[:np.argmin(R_fs)]))
                    # take the derivative of R_fs
                    dR_fsdr_ = np.gradient(R_fs_)
                    # find R_out by interpolating the derivative of R_fs to 0.
                    dR_fsdr_out =  dR_fsdr_[np.argmax(dR_fsdr_):np.argmin(dR_fsdr_)]
                    R_fs_out = R_fs_[np.argmax(dR_fsdr_):np.argmin(dR_fsdr_)]
                    fs['R_out'] = float(interpolate.interp1d(dR_fsdr_out,R_fs_out,bounds_error=False)(0.))
                if np.isnan(fs['R_in']) or np.isinf(fs['R_in']):
                    dR_fsdr = np.gradient(R_fs,edge_order=2)
                    dR_fsdr_in =  dR_fsdr[np.argmin(dR_fsdr):np.argmax(dR_fsdr)]
                    R_fs_in = R_fs[np.argmin(dR_fsdr):np.argmax(dR_fsdr)]
                    fs['R_in'] = float(interpolate.interp1d(dR_fsdr_in,R_fs_in,bounds_error=False)(0.))
                
                # find the extrema in Z of the flux surface
                # find the approximate fluxsurface top and bottom
                Z_top = np.max(Z_fs)
                Z_bottom = np.min(Z_fs)

                x_fs = interpolate.interp1d(self.derived['psi'],self.derived['rho_tor'])(psi_fs)

                # generate filter lists that take a representative slice of the top and bottom of the flux surface coordinates around the approximate Z_top and Z_bottom
                alpha = (0.9+0.075*x_fs**2)
                top_filter = [z > alpha*(Z_top-Z0_fs) for z in Z_fs-Z0_fs]
                bottom_filter = [z < alpha*(Z_bottom-Z0_fs) for z in Z_fs-Z0_fs]

                # patch for the filter lists in case the filter criteria results in < 7 points (minimum of required for 5th order fit + 1)
                i_Z_max = np.argmax(Z_fs)
                i_Z_min = np.argmin(Z_fs)

                if np.array(top_filter).sum() < 7:
                    for i in range(i_Z_max-3,i_Z_max+4):
                        if Z_fs[i] >= (np.min(Z_fs)+np.max(Z_fs))/2:
                            top_filter[i] = True
                
                if np.array(bottom_filter).sum() < 7:
                    for i in range(i_Z_min-3,i_Z_min+4):
                        if Z_fs[i] <= (np.min(Z_fs)+np.max(Z_fs))/2:
                            bottom_filter[i] = True

                # fit the top and bottom slices of the flux surface, compute the gradient of these fits and interpolate to zero to find R_top, Z_top, R_bottom and Z_bottom
                R_top_fit = np.linspace(R_fs[top_filter][-1],R_fs[top_filter][0],5000)
                try:
                    Z_top_fit = interpolate.UnivariateSpline(R_fs[top_filter][::-1],Z_fs[top_filter][::-1],k=5)(R_top_fit)
                except:
                    Z_top_fit = np.poly1d(np.polyfit(R_fs[top_filter][::-1],Z_fs[top_filter][::-1],5))(R_top_fit)
                Z_top_fit_grad = np.gradient(Z_top_fit,R_top_fit)

                R_top = interpolate.interp1d(Z_top_fit_grad,R_top_fit,bounds_error=False)(0.)
                Z_top = interpolate.interp1d(R_top_fit,Z_top_fit,bounds_error=False)(R_top)

                R_bottom_fit = np.linspace(R_fs[bottom_filter][-1],R_fs[bottom_filter][0],5000)
                try:
                    Z_bottom_fit = interpolate.UnivariateSpline(R_fs[bottom_filter],Z_fs[bottom_filter],k=5)(R_bottom_fit)
                except:
                    Z_bottom_fit = np.poly1d(np.polyfit(R_fs[bottom_filter][::-1],Z_fs[bottom_filter][::-1],5))(R_bottom_fit)
                Z_bottom_fit_grad = np.gradient(Z_bottom_fit,R_bottom_fit)

                R_bottom = interpolate.interp1d(Z_bottom_fit_grad,R_bottom_fit,bounds_error=False)(0.)
                Z_bottom = interpolate.interp1d(R_bottom_fit,Z_bottom_fit,bounds_error=False)(R_bottom)

                if _diag =='fs':
                    # diagnostic plots
                    plt.plot(R_fs[top_filter],Z_fs[top_filter],'r.')
                    plt.plot(R_fs[bottom_filter],Z_fs[bottom_filter],'r.')
                    plt.plot(R_top_fit,Z_top_fit,'g-')
                    plt.plot(R_bottom_fit,Z_bottom_fit,'g-')
                    #plt.axis('equal')
                    #plt.show()

                fs.update({'R_top':R_top,'Z_top':Z_top,'R_bottom':R_bottom,'Z_bottom':Z_bottom})
            else:
                raise SyntaxError('Equilibibrium.fluxsurface_extrema error: No average elevation provided for the target flux surface, check your inputs!')
        else:
            raise SyntaxError('Equilibibrium.fluxsurface_extrema error: No poloidal flux value for target flux surface was provided, check your inputs!')

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
    
    def fluxsurface_miller_geo(self,fs=None,symmetrise=False):
        """Extract Miller geometry parameters [Turnbull PoP 6 1113 (1999)] from a (symmetrised) flux surface trace.

        Args:
            `fs` (dict): flux surface data containing R, Z, R0, Z0, r, R_in, R_out, Z_top and Z_bottom.
            `symmetrise` (bool, optional): [True] symmetrise the provided flux surface trace, or [False, default] not.

        Returns:
            dict: the fs dict supplied to the method with the Miller parameters appended.
        """

        if symmetrise:
            fs['R_sym'] = (fs['R']+fs['R'][::-1])/2
            fs['Z_sym'] = (fs['Z']-fs['Z'][::-1])/2+fs['Z0']
            R_fs = fs['R_sym']
            Z_fs = fs['Z_sym']
        else:
            R_fs = fs['R']
            Z_fs = fs['Z']

        # compute triangularity (delta) and elongation (kappa) of flux surface
        fs['delta_top'] = (fs['R0'] - fs['R_top'])/fs['r']
        fs['delta_bottom'] = (fs['R0'] - fs['R_bottom'])/fs['r']
        fs['delta'] = (fs['delta_top']+fs['delta_bottom'])/2
        x = np.arcsin(fs['delta'])
        fs['kappa'] = (fs['Z_top'] - fs['Z_bottom'])/(2*fs['r'])

        # generate theta grid and interpolate the flux surface trace to the Miller parameterisation
        fs['theta'] = np.linspace(0,2*np.pi,720)
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
        
        #plt.plot(fs['R_miller'],fs['Z_miller'],'g-')

        return fs

    def update_pressure(self,p=None,additive=False,self_consistent=True):
        """Update the pressure profile in `Equilibrium.derived`.

        Args:
            `p` (array): vector of the (new) pressure profile in units of Pa.
            `additive` (bool, optional): [True] add `p` to the existing profile in `Equilibrium.derived`, or [False, default] not.
            `self_consistent` (bool, optional): [True, default] update the pressure dependent alpha and beta profiles, or [False] not.
        
        Returns:
            self
        """
        # add or set the new pressure profile 
        if additive:
            self.derived['p'] += p
        else:
            self.derived['p'] = p
        # self-consistently update the pressure dependent alpha and beta profiles
        if self_consistent:
            self.update_beta()
            self.update_alpha()

        return self

    def update_beta(self,beta=None,Bref=None,self_consistent=True):
        """Update the beta profile in `Equilibrium.derived`.

        Args:
            `beta` (array): vector of the (new) beta profile.
            `Bref` (float, optional): the reference magnetic field value in units of T.
            `self_consistent` (bool, optional): [True, default] update the beta dependent alpha profile, or [False] not.
        
        Returns:
            self
        """
        # set the beta profile if provided, or calculate it self-consistently
        if beta:
            self.derived['beta'] = beta
        else:
            if not Bref:
                Bref = self.derived['Bref_eqdsk']
            self.derived['beta'] = 8*np.pi*1E-7*self.derived['p']/(Bref**2)
        
        # update the alpha profile
        if self_consistent:
            self.update_alpha()
        
        return self

    def update_alpha(self,alpha=None):
        """Update the alpha profile in `Equilibrium.derived`.

        Args:
            `alpha` (array, optional): vector of the (new) alpha profile.
   
        Returns:
            self
        """
        # set the alpha profile if provided, or calculate it self-consistently
        if alpha:
            self.derived['alpha'] = alpha
        else:
            self.derived['alpha'] = -1*self.raw['qpsi']**2*self.derived['Ro']*np.gradient(self.derived['beta'],self.derived['r'])
        
        return self
    
    def map_on_equilibrium(self,x=None,y=None,x_label=None,interp_order=9,extrapolate=False):
        """Map a 1D plasma profile on to the `x_label` radial coordinate basis of this `Equilibrium`.

        Args:
            `x` (array): vector of the radial basis of the existing profile in units of `x_label`.
            `y` (array): vector of the 1D profile of the plasma quantity that is mapped on this `Equilibrium`.
            `x_label` (str): label of the radial coordinate specification of `x`.
            `interp_order` (float): the interpolation order used in the remapping of the 1D profile, [default] 9 based on experience.
            `extrapolate` (bool): [True] use `fill_value` = 'extrapolate' in the interpolation, or [False, default] not.

        Returns:
            Two vectors, the x vector of `Equilibrium.derived` [`x_label`] and the y vector of the remapped 1D profile.
        """

        # remap the provided y profile, onto the x basis of x_label in this equilibrium
        if extrapolate:
            y_interpolated = interpolate.interp1d(x,y,kind=interp_order,bounds_error=False,fill_value='extrapolate')(self.derived[x_label])
        else:
            y_interpolated = interpolate.interp1d(x,y,kind=interp_order,bounds_error=False)(self.derived[x_label])
        
        return self.derived[x_label],y_interpolated
    
    def refine(self,nw=None,nbbbs=None,interp_order=9,retain_original=False,self_consistent=True):
        """Refine the R,Z resolution of the `Equilibrium` assuming a g-EQDSK file as origin.

        Args:
            `nw` (int): desired grid resolution of the 1D profiles and 2D psi(R,Z) map (assuming a nw x nw grid).
            `nbbbs` (int, optional): desired grid resolution of the last closed flux surface plasma boundary trace, currently not implemented!
            `interp_order` (int, optional): the interpolation order used in the remapping of the 1D profiles, [default] 9 based on experience.
            `retain_original` (bool, optional): [True] store the original raw g-EQDSK equilibrium data in `Equilibrium.original`, or [False, default] not.
            `self_consistent` (bool, optional): [True, default] re-derive and re-trace all the existing additional data when applied to an existing `Equilibrium`, or [False] not.

        Returns:
            self

        Raises:
            ValueError: if the provided `nw` is smaller than the native resolution of the `Equilibrium`.
        """
        print('Refining Equilibrium to {}x{}...'.format(nw,nw))
        if retain_original:
            self.original = copy.deepcopy(self.raw)

        if nw and nw > self.raw['nw']:
            old_x = np.linspace(0,1,self.raw['nw'])
            old_y = np.linspace(0,1,self.raw['nh'])
            new_x = np.linspace(0,1,nw)
            refinable = True
        else:
            refinable = False
            raise ValueError('Provided nw does not refine the equilibrium, provided:{} < exisiting:{}'.format(nw,self.raw['nw']))
        
        '''
        elif nbbbs and nbbbs > self.raw['nbbbs']:
            old_x = np.linspace(0,1,self.raw['nbbbs'])
            new_x = np.linspace(0,1,nbbbs)
            refinable = True
        elif limitr and limitr > self.raw['limitr']:
            old_x = np.linspace(0,1,self.raw['limitr'])
            new_x = np.linspace(0,1,limitr)
            refinable = True
        '''

        if refinable:
            for quantity in self.raw.keys():
                if isinstance(self.raw[quantity],np.ndarray):
                    if self.raw[quantity].size == old_x.size:
                        #print('quantity: {}'.format(quantity))
                        self.raw[quantity] = interpolate.interp1d(old_x,self.raw[quantity],kind=interp_order)(new_x)
                    elif self.raw[quantity].size == old_x.size*old_y.size:
                        #print('quantity: {}'.format(quantity))
                        self.raw[quantity] = interpolate.interp2d(old_x,old_y,self.raw[quantity],kind='quintic')(new_x,new_x)
            self.raw['nw'] = nw
            # assuming a square psi(R,Z) grid (nw x nw)
            self.raw['nh'] = nw
        
        if self_consistent:
            if self.derived and self.fluxsurfaces:
                self.derived = {}
                self.fluxsurfaces = {}
                self.add_derived(incl_fluxsurfaces=True)
            elif self.derived:
                self.derived = {}
                self.add_derived()

        return self