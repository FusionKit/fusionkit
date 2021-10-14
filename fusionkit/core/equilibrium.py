'''
The Equilibrium class can read magnetic equilibria files (only edsk g-files for now),
add derived quantities (e.g. phi, rho_tor, rho_pol, etc.) to the Equilibrium, trace flux surfaces
and derive shaping parameters (for now only Miller parameters) from the flux surfaces.
'''

# general imports
import os
import re
import numpy as np
import json,codecs
import copy
from pathlib import Path
from scipy import interpolate, integrate
from sys import stdout

# fusionkit dependencies
from .utils import find, number
from .dataspine import DataSpine

# Common numerical data types, for ease of type-checking
np_itypes = (np.int8, np.int16, np.int32, np.int64)
np_utypes = (np.uint8, np.uint16, np.uint32, np.uint64)
np_ftypes = (np.float16, np.float32, np.float64)

number_types = (float, int, np_itypes, np_utypes, np_ftypes)
array_types = (list, tuple, np.ndarray)

class Equilibrium:
    '''
    Class to handle any and all data related to the magnetic equilibrium in a magnetic confinement fusion device
    '''
    def __init__(self):
        self.raw = {} # storage for all raw eqdsk data
        self.derived = {} # storage for all data derived from eqdsk data
        self.fluxsurfaces = {} # storage for all data related to flux surfaces
        # specify the eqdsk file formate, based on 'G EQDSK FORMAT - L Lao 2/7/97'
        self.eqdsk_format = {
            0:{'vars':['code','case','idum','nw','nh'],'size':[5]},
            1:{'vars':['rdim', 'zdim', 'rcentr', 'rleft', 'zmid'],'size':[5]},
            2:{'vars':['rmaxis', 'zmaxis', 'simag', 'sibry', 'bcentr'],'size':[5]},
            3:{'vars':['current', 'simag2', 'xdum', 'rmaxis2', 'xdum'],'size':[5]},
            4:{'vars':['zmaxis2', 'xdum', 'sibry2', 'xdum', 'xdum'],'size':[5]},
            5:{'vars':['fpol'],'size':['nw']},
            6:{'vars':['pres'],'size':['nw']},
            7:{'vars':['ffprim'],'size':['nw']},
            8:{'vars':['pprime'],'size':['nw']},
            9:{'vars':['psirz'],'size':['nw','nh']},
            10:{'vars':['qpsi'],'size':['nw']},
            11:{'vars':['nbbbs','limitr'],'size':[2]},
            12:{'vars':['rbbbs','zbbbs'],'size':['nbbbs']},
            13:{'vars':['rlim','zlim'],'size':['limitr']},
        }
        self.sanity_values = ['rmaxis','zmaxis','simag','sibry'] # specify the sanity values used for consistency check of eqdsk file
        self.max_values = 5 # maximum number of values per line

    ## I/O functions
    def read_geqdsk(self,f_path=None,just_raw=False,add_derived=False):
        '''
        Function to convert an eqdsk g-file from file to Equilibrium() object

        :param f_path: string containing the path to the eqdsk g-file, including the file name (!)

        :param just_raw: boolean to return only the raw dictionary (True) or [default] return the Equilibrium() object (False)

        :param add_derived: boolean to directly add derived quantities (e.g. phi, rho_tor) to the Equilibrium() object upon reading the g-file

        :return: self or dict if just_raw
        '''
        print('Reading edsk g-file to Equilibrium...')

        # check if eqdsk file path is provided and if it exists
        if f_path is None or not os.path.isfile(f_path):
            print('Invalid file or path provided!')
            return
        
        # read the g-file
        with open(f_path,'r') as file:
            lines = file.readlines()
        
        # convert the line strings in the values list to lists of numerical values, while retaining potential character strings at the start of the file
        for i,line in enumerate(lines):
            # split the line string into separate values by ' ' as delimiter, adding a space before a minus sign if it is the delimiter
            values = list(filter(None,re.sub(r'(?<![Ee])-',' -',line).rstrip('\n').split(' ')))
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
            # handle the exception of the first line where in the case description numbers and strings can be mixed
            if i == 0:
                numbers = numbers[-3:]
                case = [string for string in list(line.rstrip('\n').split())[1:] if string not in numbers] 
                strings = [strings[0],' '.join(case)]
            # convert the list of numerical sub-strings to their actual int or float value
            numbers = [number(value) for value in numbers]
            #print('numbers: '+str(numbers))
            lines[i] = strings+numbers
            #print('line after: '+str(lines[i]))

        # start at the top of the file
        current_row = 0
        # go through the eqdsk format line by line and collect all the values for the vars in each format line
        for key in self.eqdsk_format:
            if current_row < len(lines):
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
                raise ValueError('Inconsistent '+key+': %7.4g, %7.4g'%(self.raw[key], self.raw[sanity_pair])+'. CHECK YOUR EQDSK FILE!')

        if add_derived:
            self.add_derived()
        if just_raw:
            return self.raw
        else:
            return self
    
    def write_geqdsk(self,f_path=None):
        '''
        Function to convert this Equilibrium() object to a eqdsk g-file 
        :param f_path: string containing the target path of generated eqdsk g-file, including the file name (!)
        :returns: none
        '''
        print('Writing Equilibrium to eqdsk g-file...')

        if self.raw:
            if not isinstance(f_path, str):
                raise TypeError("filepath field must be a string. EQDSK file write aborted.")

            maxv = int(self.max_values)

            eqpath = Path(f_path)
            if eqpath.is_file():
                print("%s exists, overwriting file with EQDSK file!" % (str(eqpath)))
            eq = {"xdum": 0.0}
            for linenum in self.eqdsk_format:
                if "vars" in self.eqdsk_format[linenum]:
                    for key in self.eqdsk_format[linenum]["vars"]:
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
                if "code" in eq and eq["code"]:
                    gcase = gcase + eq["code"] + " "
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
    
    def read_json(self,f_path=None):
        '''
        Function to read an Equilibrium object stored on disk in json into a callable Equilibrium object

        :param f_path: string path to the location the desired file, including the desired file name (!)

        :return: Equilibrium object containing the data from the json
        '''
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

    def write_json(self,f_path='./',f_name='Equilibrium.json',metadata=None):
        '''
        Function to write the Equilibrium object to a json file on disk

        :param f_path: string path to the location the desired file, [default] the current folder '.'  (optional) 

        :param f_name: string of the desired file name including the .json extension (!), [default] 'Equilibrium.json' (optional)

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

        json.dump(equilbrium, codecs.open(f_path+f_name, 'w', encoding='utf-8'), separators=(',', ':'), indent=4)

        print('Generated fusionkit.Equilibrium file at: {}'.format(f_path+f_name))

        return

    ## physics functions
    def add_derived(self,f_path=None,just_derived=False,incl_fluxsurfaces=False,resolution=None,incl_miller_geo=False):
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
        '''
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
        '''

        phirz_norm = abs(derived['phirz'])/(derived['phi'][-1])
        derived['rhorz_tor'] = np.sqrt(phirz_norm)

        # compute the toroidal magnetic field and current density
        derived['B_tor'] = raw['ffprim']/derived['R']
        derived['j_tor'] = derived['R']*raw['pprime']+derived['B_tor']

        if incl_fluxsurfaces:
            self.add_fluxsurfaces(raw=raw,derived=derived,fluxsurfaces=fluxsurfaces,resolution=resolution,incl_miller_geo=incl_miller_geo)
              
        if just_derived:
            return self.raw['derived']
        else:
            return self

    def add_fluxsurfaces(self,raw=None,derived=None,fluxsurfaces=None,resolution=None,incl_miller_geo=False):
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

            if resolution is None:
                if 'nw' in self.raw:
                    resolution = self.raw['nw']
            if self.raw['nw']<resolution:
                self.refine(nw=resolution,self_consistent=False)
                self.add_derived()

            # refine the psi R,Z grid by 4x to get smooth(er) gradients for geometric quantities
            refine = 4*self.raw['nw']
            R = np.linspace(self.derived['R'][0],self.derived['R'][-1],refine)
            Z = np.linspace(self.derived['Z'][0],self.derived['Z'][-1],refine)
            psirz = interpolate.interp2d(self.derived['R'],self.derived['Z'],self.raw['psirz'])(R,Z)

            # find the approximate location of the magnetic axis on the psirz map
            i_rmaxis = np.where(psirz == np.min(psirz))[1][0]
            i_zmaxis = np.where(psirz == np.min(psirz))[0][0]

            # add the flux surface data for rho_tor > 0
            for rho_fs in self.derived['rho_tor'][1:]:
                # print a progress %
                stdout.write('\r {}% completed'.format(round(100*(find(rho_fs,self.derived['rho_tor'][1:])+1)/len(self.derived['rho_tor'][1:]))))
                stdout.flush()
                # check that rho stays inside the lcfs
                if rho_fs < 0.999:
                    self.fluxsurface_find(x_fs=rho_fs,R=R,Z=Z,psirz=psirz,i_maxis=[i_rmaxis,i_zmaxis],incl_miller_geo=True,return_self=True)
            stdout.write('\n')

            if 'rbbbs' in raw and 'zbbbs' in raw:
                # find the geometric center, minor radius and extrema of the lcfs manually
                lcfs = self.fluxsurface_center(psi_fs=raw['sibry'],R_fs=derived['rbbbs'],Z_fs=derived['zbbbs'],psirz=raw['psirz'],R=derived['R'],Z=derived['Z'],incl_extrema=True)
                lcfs.update({'R':derived['rbbbs'],'Z':derived['zbbbs']})
            else:
                lcfs = self.fluxsurface_find(x_fs=1.0,psi_fs=raw['sibry'],psirz=psirz,R=R,Z=Z,i_maxis=[i_rmaxis,i_zmaxis],interp_method='bounded_extrapolation',return_self=False)
                derived.update({'rbbbs':lcfs['R'],'zbbbs':lcfs['Z'],'nbbbs':len(lcfs['R'])})
            if incl_miller_geo:
                lcfs = self.fluxsurface_miller_geo(fs=lcfs)
            
            # add a zero at the start of all fluxsurface quantities and append the lcfs values to the end of the flux surface data
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
    
    def fluxsurface_find(self,psi_fs=None,psi=None,x_fs=None,x=None,x_label='rho_tor',R=None,Z=None,psirz=None,i_maxis=None,interp_method='normal',incl_miller_geo=False,return_self=False):
        '''
        #Function to find the R,Z trace of a flux surface 

        :param psi_fs: (optional) float of the poloidal flux value of the flux surface

        :param psi: (optional) array vector containing the poloidal flux psi from axis to separatrix

        :param x_fs: float of the radial flux label of the flux surface, by default assumed to be in rho_tor

        :param x: (optional) array vector of the radial flux surface label on the same grid as psi, by default assume to be rho_tor

        :param x_label: string of the radial flux label, options (for now) are [default] 'rho_tor', 'rho_pol', 'psi' and 'r'

        :param R: array vector of R grid mesh

        :param Z: array vector of Z grid mesh

        :param psirz: array containing the R,Z map of the poloidal flux psi of the magnetic equilibrium

        :param i_maxis: (optional) list or array containing the indexes of the approximate magnetic axis in psriz to speed up the tracing calculation in a loop

        :param incl_miller_geo: boolean to include the symmetrised flux surface Miller shaping parameters delta, kappa and zeta (True) or [default] not (False)

        :param return_self: boolean to return the result to the Equilibrium() object (True) or [default] as a standalone dictionary (False)

        :return: dict with the flux surface [default] or add the fluxsurface data to Equilibrium.fluxsurfaces

        '''
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
            i_rmaxis = np.where(psirz == np.min(psirz))[1][0]
            i_zmaxis = np.where(psirz == np.min(psirz))[0][0]

        # find the vertical extrema of the LCFS at the major radius of the magnetic axis
        i_psiz_rmaxis_min = np.argmin(psirz[:,i_rmaxis])
        i_psiz_rmaxis_bottom_max = np.argmax(psirz[:i_psiz_rmaxis_min,i_rmaxis])
        i_psiz_rmaxis_top_max = i_psiz_rmaxis_min+np.argmax(psirz[i_psiz_rmaxis_min:,i_rmaxis])

        psiz_rmaxis_bottom = psirz[i_psiz_rmaxis_bottom_max:i_psiz_rmaxis_min,i_rmaxis]
        psiz_rmaxis_top = psirz[i_psiz_rmaxis_min:i_psiz_rmaxis_top_max,i_rmaxis]

        Z_lcfs_max = interpolate.interp1d(psiz_rmaxis_top,Z[i_psiz_rmaxis_min:i_psiz_rmaxis_top_max],bounds_error=False,fill_value='extrapolate')(psi[-1])
        Z_lcfs_min = interpolate.interp1d(psiz_rmaxis_bottom,Z[i_psiz_rmaxis_bottom_max:i_psiz_rmaxis_min],bounds_error=False,fill_value='extrapolate')(psi[-1])

        # set the starting coordinates for the fluxsurface tracing algorithm
        i, j = i_zmaxis, i_zmaxis

        # find the psi value corresponding to the the current x coordinate
        psi_fs = interpolate.interp1d(x,psi,kind='cubic')(x_fs)

        # while the psi_fs intersects with the current psirz slice gather the intersection coordinates
        while (psi_fs > np.min(psirz[i]) and i < psirz.shape[0]-1):
            top = False
            
            if Z[i] <= Z_lcfs_max:
                # find the minimum in the Z slice of psirz
                i_psir_slice_min = np.argmin(psirz[i])

                # chop the psirz and R slices in two parts to separate the HFS and LFS
                psir_slice_top_hfs = psirz[i,:i_psir_slice_min]
                psir_slice_top_lfs = psirz[i,i_psir_slice_min:]

                # interpolate the R coordinate of the top half HFS and LFS
                if interp_method == 'normal':
                    R_fs_top_hfs = float(interpolate.interp1d(psir_slice_top_hfs,R[i_psir_slice_min-len(psir_slice_top_hfs):i_psir_slice_min],bounds_error=False)(psi_fs))
                    R_fs_top_lfs = float(interpolate.interp1d(psir_slice_top_lfs,R[i_psir_slice_min:i_psir_slice_min+len(psir_slice_top_lfs)],bounds_error=False)(psi_fs))
                # if the normal method provides spikey flux surface traces, bound the interpolation domain and extrapolate the intersection
                elif interp_method == 'bounded_extrapolation':
                    psir_slice_top_hfs = psir_slice_top_hfs[psir_slice_top_hfs<=self.raw['sibry']]
                    psir_slice_top_lfs = psir_slice_top_lfs[psir_slice_top_lfs<=self.raw['sibry']]
                    
                    R_fs_top_hfs = float(interpolate.interp1d(psir_slice_top_hfs,R[i_psir_slice_min-len(psir_slice_top_hfs):i_psir_slice_min][psir_slice_top_hfs<=self.raw['sibry']],bounds_error=False,fill_value='extrapolate')(psi_fs))
                    R_fs_top_lfs = float(interpolate.interp1d(psir_slice_top_lfs,R[i_psir_slice_min:i_psir_slice_min+len(psir_slice_top_lfs)][psir_slice_top_lfs<=self.raw['sibry']],bounds_error=False,fill_value='extrapolate')(psi_fs))

                # insert the coordinate pairs into the fluxsurface trace dict if not nan (bounds error) and order properly for merging later
                if not np.isnan(R_fs_top_hfs):
                    RZ_fs['hfs']['top'].insert(0,(R_fs_top_hfs,Z[i]))
                if not np.isnan(R_fs_top_lfs):
                    RZ_fs['lfs']['top'].append((R_fs_top_lfs,Z[i]))
                
                top = True
            
            # interpolate the Z coordinate of the fluxsurface on the LFS for the top and bottom until crossing the +- 1/4 pi diagonals
            if top and R[i] < R_fs_top_lfs:
                # create a lockstepping index for vertical tracing
                k = (i-i_zmaxis)+i_rmaxis

                # find the minimum and the extrema of the R slice of psirz
                i_psiz_slice_min = np.argmin(psirz[:,k])
                i_psiz_slice_bottom_max = np.argmax(psirz[:i_psiz_slice_min,k])
                i_psiz_slice_top_max = i_psiz_slice_min+np.argmax(psirz[i_psiz_slice_min:,k])

                # chop the psirz slices in two parts to separate the top and bottom halves
                psiz_slice_bottom_lfs = psirz[i_psiz_slice_bottom_max:i_psiz_slice_min,k]
                psiz_slice_top_lfs = psirz[i_psiz_slice_min:i_psiz_slice_top_max,k]
                
                # interpolate the Z coordinate for the LFS top and bottom
                if interp_method == 'normal':
                    Z_fs_bottom_lfs = float(interpolate.interp1d(psiz_slice_bottom_lfs,Z[i_psiz_slice_bottom_max:i_psiz_slice_min],bounds_error=False)(psi_fs))
                    Z_fs_top_lfs = float(interpolate.interp1d(psiz_slice_top_lfs,Z[i_psiz_slice_min:i_psiz_slice_top_max],bounds_error=False)(psi_fs))
                elif interp_method == 'bounded_extrapolation':
                    Z_slice_bottom_lfs = Z[i_psiz_slice_bottom_max:i_psiz_slice_min][psiz_slice_bottom_lfs<=self.raw['sibry']]
                    Z_slice_top_lfs = Z[i_psiz_slice_min:i_psiz_slice_top_max][psiz_slice_top_lfs<=self.raw['sibry']]
                    psiz_slice_bottom_lfs = psiz_slice_bottom_lfs[psiz_slice_bottom_lfs<=self.raw['sibry']]
                    psiz_slice_top_lfs = psiz_slice_top_lfs[psiz_slice_top_lfs<=self.raw['sibry']]

                    Z_fs_bottom_lfs = float(interpolate.interp1d(psiz_slice_bottom_lfs,Z_slice_bottom_lfs,bounds_error=False,fill_value='extrapolate')(psi_fs))
                    Z_fs_top_lfs = float(interpolate.interp1d(psiz_slice_top_lfs,Z_slice_top_lfs,bounds_error=False,fill_value='extrapolate')(psi_fs))

                if not np.isnan(Z_fs_bottom_lfs):
                    RZ_fs['lfs']['bottom'].append((R[k],Z_fs_bottom_lfs))
                if not np.isnan(Z_fs_top_lfs):
                    RZ_fs['lfs']['top'].append((R[k],Z_fs_top_lfs))

            # update the slice coordinates
            if i < psirz.shape[0]-1:
                i+=1

        # while the psi_fs intersects with the current psirz slice gather the intersection coordinates
        while (psi_fs > np.min(psirz[j]) and j > 0):
            bottom = False

            # interpolate the R coordinate of the bottom half of the fluxsurface on both the LFS and HFS
            if Z[j] >= Z_lcfs_min:
                j_psir_slice_min = np.argmin(psirz[j])

                psir_slice_bottom_hfs = psirz[j,:j_psir_slice_min]
                psir_slice_bottom_lfs = psirz[j,j_psir_slice_min:]

                # interpolate the R coordinate of the bottom half HFS and LFS
                if interp_method == 'normal':
                    R_fs_bottom_hfs = float(interpolate.interp1d(psir_slice_bottom_hfs,R[j_psir_slice_min-len(psir_slice_bottom_hfs):j_psir_slice_min],bounds_error=False)(psi_fs))
                    R_fs_bottom_lfs = float(interpolate.interp1d(psir_slice_bottom_lfs,R[j_psir_slice_min:j_psir_slice_min+len(psir_slice_bottom_lfs)],bounds_error=False)(psi_fs))
                elif interp_method == 'bounded_extrapolation':
                    psir_slice_bottom_hfs = psir_slice_bottom_hfs[psir_slice_bottom_hfs<=self.raw['sibry']]
                    psir_slice_bottom_lfs = psir_slice_bottom_lfs[psir_slice_bottom_lfs<=self.raw['sibry']]
                    R_fs_bottom_hfs = float(interpolate.interp1d(psir_slice_bottom_hfs,R[j_psir_slice_min-len(psir_slice_bottom_hfs):j_psir_slice_min][psir_slice_bottom_hfs<=self.raw['sibry']],bounds_error=False,fill_value='extrapolate')(psi_fs))
                    R_fs_bottom_lfs = float(interpolate.interp1d(psir_slice_bottom_lfs,R[j_psir_slice_min:j_psir_slice_min+len(psir_slice_bottom_lfs)][psir_slice_bottom_lfs<=self.raw['sibry']],bounds_error=False,fill_value='extrapolate')(psi_fs))

                if not np.isnan(R_fs_bottom_hfs):
                    RZ_fs['hfs']['bottom'].append((R_fs_bottom_hfs,Z[j]))
                if not np.isnan(R_fs_bottom_lfs):
                    RZ_fs['lfs']['bottom'].insert(0,(R_fs_bottom_lfs,Z[j]))
                
                bottom = True

            # interpolate the Z coordinate of the fluxsurface on the HFS for the top and bottom until crossing the +- 3/4 pi diagonals
            if bottom and R[j] > R_fs_bottom_hfs:
                k = (j-i_zmaxis)+i_rmaxis

                i_psiz_slice_min = np.argmin(psirz[:,k])
                i_psiz_slice_bottom_max = np.argmax(psirz[:i_psiz_slice_min,k])
                i_psiz_slice_top_max = i_psiz_slice_min+np.argmax(psirz[i_psiz_slice_min:,k])

                psiz_slice_bottom_hfs = psirz[i_psiz_slice_bottom_max:i_psiz_slice_min,k]
                psiz_slice_top_hfs = psirz[i_psiz_slice_min:i_psiz_slice_top_max,k]
                
                # interpolate the Z coordinate for the HFS top and bottom
                if interp_method == 'normal':
                    Z_fs_bottom_hfs = float(interpolate.interp1d(psiz_slice_bottom_hfs,Z[i_psiz_slice_bottom_max:i_psiz_slice_min],bounds_error=False)(psi_fs))
                    Z_fs_top_hfs = float(interpolate.interp1d(psiz_slice_top_hfs,Z[i_psiz_slice_min:i_psiz_slice_top_max],bounds_error=False)(psi_fs))
                elif interp_method == 'bounded_extrapolation':
                    Z_slice_bottom_hfs = Z[i_psiz_slice_bottom_max:i_psiz_slice_min][psiz_slice_bottom_hfs<=self.raw['sibry']]
                    Z_slice_top_hfs = Z[i_psiz_slice_min:i_psiz_slice_top_max][psiz_slice_top_hfs<=self.raw['sibry']]
                    psiz_slice_bottom_hfs = psiz_slice_bottom_hfs[psiz_slice_bottom_hfs<=self.raw['sibry']]
                    psiz_slice_top_hfs = psiz_slice_top_hfs[psiz_slice_top_hfs<=self.raw['sibry']]

                    Z_fs_bottom_hfs = float(interpolate.interp1d(psiz_slice_bottom_hfs,Z_slice_bottom_hfs,bounds_error=False,fill_value='extrapolate')(psi_fs))
                    Z_fs_top_hfs = float(interpolate.interp1d(psiz_slice_top_hfs,Z_slice_top_hfs,bounds_error=False,fill_value='extrapolate')(psi_fs))

                if not np.isnan(Z_fs_bottom_hfs):
                    RZ_fs['hfs']['bottom'].append((R[k],Z_fs_bottom_hfs))
                if not np.isnan(Z_fs_top_hfs):
                    RZ_fs['hfs']['top'].append((R[k],Z_fs_top_hfs))

            # update the slice coordinates
            if j > 0:
                j-=1

        # zip each quadrant as a function of the Z coordinate, then sort, then re-zip as a function of the R coordinate
        RZ_fs_hfs_top_reverse = [(b, a) for a, b in sorted([(b, a) for a, b in RZ_fs['hfs']['top']])]
        RZ_fs_hfs_bottom_reverse = [(b, a) for a, b in sorted([(b, a) for a, b in RZ_fs['hfs']['bottom']])]
        RZ_fs_lfs_top_reverse = [(b, a) for a, b in sorted([(b, a) for a, b in RZ_fs['lfs']['top']])]
        RZ_fs_lfs_bottom_reverse = [(b, a) for a, b in sorted([(b, a) for a, b in RZ_fs['lfs']['bottom']])]

        # define the merge indexes at the +- 1/4 pi and +- 3/4 pi diagonals
        i_merge_hfs_top = int(len(RZ_fs['hfs']['top'])/2)
        i_merge_hfs_bottom = int(len(RZ_fs['hfs']['bottom'])/2)
        i_merge_lfs_top = int(len(RZ_fs['lfs']['top'])/2)
        i_merge_lfs_bottom = int(len(RZ_fs['lfs']['bottom'])/2)

        # sort each quadrant half as a function of the R coordinate and the other half as a function of the Z coordinate and then merge them
        RZ_fs['hfs']['top'] = (RZ_fs_hfs_top_reverse[:i_merge_hfs_top]+sorted(RZ_fs['hfs']['top'])[i_merge_hfs_top:])[::-1]
        RZ_fs['hfs']['bottom'] = (sorted(RZ_fs['hfs']['bottom'])[i_merge_hfs_bottom:][::-1]+RZ_fs_hfs_bottom_reverse[i_merge_hfs_bottom:])[::-1]
        RZ_fs['lfs']['top'] = RZ_fs_lfs_top_reverse[:i_merge_lfs_top]+sorted(RZ_fs['lfs']['top'])[:i_merge_lfs_top][::-1]
        RZ_fs['lfs']['bottom'] = sorted(RZ_fs['lfs']['bottom'])[:i_merge_lfs_bottom]+RZ_fs_lfs_bottom_reverse[i_merge_lfs_bottom:]

        # merge the different quadrants accounting for the direction of tracing for the different parts of each quadrant
        fs_start = RZ_fs['lfs']['top'][:i_merge_lfs_top]
        fs_middle_1 = sorted(RZ_fs['lfs']['top'][i_merge_lfs_top:]+RZ_fs['hfs']['top'][:i_merge_hfs_top])[::-1]
        fs_middle_2 = [(b, a) for a, b in sorted([(b, a) for a, b in RZ_fs['hfs']['top'][i_merge_hfs_top:]]+[(b, a) for a, b in RZ_fs['hfs']['bottom'][:i_merge_hfs_bottom]])][::-1]
        fs_middle_3 = sorted(RZ_fs['hfs']['bottom'][i_merge_hfs_bottom:]+RZ_fs['lfs']['bottom'][:i_merge_lfs_bottom])
        fs_end = RZ_fs['lfs']['bottom'][i_merge_lfs_bottom:]

        # merge the complete fluxsurface trace
        RZ_fs = fs_start + fs_middle_1 +fs_middle_2 + fs_middle_3 + fs_end

        # separate the R and Z coordinates in separate vectors
        R_fs = np.array([a for a,b in RZ_fs])
        Z_fs = np.array([b for a,b in RZ_fs])

        fs = {'R':R_fs,'Z':Z_fs,'psi':psi_fs}

        # find the flux surface center quantities and add them to the flux surface dict
        fs.update(self.fluxsurface_center(psi_fs=psi_fs,R_fs=fs['R'],Z_fs=fs['Z'],R=R,Z=Z,psirz=psirz,incl_extrema=True))

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

    def fluxsurface_center(self,psi_fs=None,R_fs=None,Z_fs=None,psirz=None,R=None,Z=None,incl_extrema=False,return_self=False):
        '''
        Function to find the geometric center of a flux surface trace defined by R_fs,Z_fs and psi_fs

        :param psi_fs: float of the poloidal flux value of the flux surface

        :param R_fs: array containing the horizontal coordinates of the flux surface trace

        :param Z_fs: array containing the vertical coordinates of the flux surface trace

        :param psirz: array containing the R,Z map of the poloidal flux psi of the magnetic equilibrium

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
        fs_extrema = self.fluxsurface_extrema(psi_fs=psi_fs,R_fs=R_fs,Z_fs=Z_fs,Z0_fs=fs['Z0'],psirz=psirz,R=R,Z=Z)
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

    def fluxsurface_extrema(self,psi_fs=None,R_fs=None,Z_fs=None,Z0_fs=None,psirz=None,R=None,Z=None,return_self=False):
        '''
        Function to find the extrema in R and Z of a flux surface trace defined by R_fs,Z_fs and psi_fs

        :param psi_fs: float of the poloidal flux value of the flux surface

        :param R_fs: array containing the horizontal coordinates of the flux surface trace

        :param Z_fs: array containing the vertical coordinates of the flux surface trace

        :param Z0_fs: float of the average elevation of the flux surface

        :param psirz: array containing the R,Z map of the poloidal flux psi of the magnetic equilibrium

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
                psirz0 = interpolate.interp2d(R,Z,psirz)(R,Z0_fs)
                #print(psirz0)

                # find the extrema in R of the flux surface at the midplane
                fs['R_out'] = float(interpolate.interp1d(psirz0[np.argmin(psirz0):],R[np.argmin(psirz0):],bounds_error=False)(psi_fs))
                fs['R_in'] = float(interpolate.interp1d(psirz0[:np.argmin(psirz0)],R[:np.argmin(psirz0)],bounds_error=False)(psi_fs))

                # find the extrema in Z of the flux surface
                # find the approximate fluxsurface top and bottom
                Z_top = np.max(Z_fs)
                Z_bottom = np.min(Z_fs)

                x_fs = interpolate.interp1d(self.derived['psi'],self.derived['rho_tor'])(psi_fs)

                # generate filter lists that take a representative slice of the top and bottom of the flux surface coordinates around the approximate Z_top and Z_bottom
                top_filter = [z > (0.90+0.075*x_fs**2)*Z_top for z in Z_fs]
                bottom_filter = [z < (0.90+0.075*x_fs**2)*(Z_bottom-Z0_fs) for z in Z_fs-Z0_fs]

                # fit the top and bottom slices of the flux surface, compute the gradient of these fits and interpolate to zero to find R_top, Z_top, R_bottom and Z_bottom
                R_top_fit = np.linspace(R_fs[top_filter][-1],R_fs[top_filter][0],5000)
                Z_top_fit = np.poly1d(np.polyfit(R_fs[top_filter][::-1],Z_fs[top_filter][::-1],5))(R_top_fit)
                Z_top_fit_grad = np.gradient(Z_top_fit,R_top_fit)

                R_top = interpolate.interp1d(Z_top_fit_grad,R_top_fit)(0)
                Z_top = interpolate.interp1d(R_top_fit,Z_top_fit)(R_top)

                R_bottom_fit = np.linspace(R_fs[bottom_filter][-1],R_fs[bottom_filter][0],5000)
                Z_bottom_fit = np.poly1d(np.polyfit(R_fs[bottom_filter][::-1],Z_fs[bottom_filter][::-1],5))(R_bottom_fit)
                Z_bottom_fit_grad = np.gradient(Z_bottom_fit,R_bottom_fit)

                R_bottom = interpolate.interp1d(Z_bottom_fit_grad,R_bottom_fit)(0)
                Z_bottom = interpolate.interp1d(R_bottom_fit,Z_bottom_fit)(R_bottom)

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

        # compute triangularity (delta) and elongation (kappa) of flux surface
        delta_top = (fs['R0'] - fs['R_top'])/fs['r']
        delta_bottom = (fs['R0'] - fs['R_bottom'])/fs['r']
        fs['delta'] = (delta_top+delta_bottom)/2
        x = np.arcsin(fs['delta'])
        fs['kappa'] = (fs['Z_top'] - fs['Z_bottom'])/(2*fs['r'])

        # generate theta grid and interpolate the flux surface trace to the Miller parameterisation
        fs['theta'] = np.linspace(0,2*np.pi,360)
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

    def update_pressure(self,p=None,additive=False,self_consistent=True):
        if additive:
            self.derived['p'] += p
        else:
            self.derived['p'] = p
        if self_consistent:
            self.update_beta()
            self.update_alpha()
        return self

    def update_beta(self,beta=None,Bref=None,self_consistent=True):
        if beta:
            self.derived['beta'] = beta
        else:
            if not Bref:
                Bref = self.derived['Bref_eqdsk']
            self.derived['beta'] = 8*np.pi*1E-7*self.derived['p']/(Bref**2)
        if self_consistent:
            self.update_alpha()
        return self

    def update_alpha(self,alpha=None):
        if alpha:
            self.derived['alpha'] = alpha
        else:
            self.derived['alpha'] = -1*self.raw['qpsi']**2*self.derived['Ro']*np.gradient(self.derived['beta'],self.derived['r'])
        return self
    
    def map_on_equilibrium(self,x=None,y=None,x_label=None,interp_order=9):
        y_interpolated = interpolate.interp1d(x,y,kind=interp_order,bounds_error=False)(self.derived[x_label])
        return self.derived[x_label],y_interpolated
    
    def refine(self,nw=None,nbbbs=None,limitr=None,interp_order=9,retain_original=False,self_consistent=True):
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
            print('Provided nw does not refine the equilibrium, provided:{} < exisiting:{}'.format(nw,self.raw['nw']))
        
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
                    elif self.raw[quantity].size == old_x.size**2:
                        #print('quantity: {}'.format(quantity))
                        self.raw[quantity] = interpolate.interp2d(old_x,old_y,self.raw[quantity],kind='quintic')(new_x,new_x)
            self.raw['nw'] = nw
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