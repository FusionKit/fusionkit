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
from scipy import interpolate, integrate
from sys import stdout

# fusionkit dependencies
from .utils import find, number, read_eqdsk, write_eqdsk
from .dataspine import DataSpine

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
        print('Reading eqdsk g-file to Equilibrium...')

        self.raw.update(read_eqdsk(f_path))

        # specify the sanity values used for consistency check of eqdsk file
        self.sanity_values = ['rmaxis','zmaxis','simag','sibry']

        # sanity check the eqdsk values
        for key in self.sanity_values:
            # find the matching sanity key in eqdsk
            sanity_pair = [keypair for keypair in self.raw.keys() if keypair.startswith(key)][1]
            #print(sanity_pair)
            if self.raw[key]!=self.raw[sanity_pair]:
                raise ValueError('Inconsistent '+key+': %7.4g, %7.4g'%(self.raw[key], self.raw[sanity_pair])+'. CHECK YOUR EQDSK FILE!')
        
        if 'rbbbs' in self.raw and 'zbbbs' in self.raw:
            # ensure the boundary coordinates are stored from midplane lfs to midplane hfs
            i_split = find(np.max(self.raw['rbbbs']),self.raw['rbbbs'])
            self.raw['rbbbs'] = np.hstack((self.raw['rbbbs'][i_split:],self.raw['rbbbs'][:i_split]))
            self.raw['zbbbs'] = np.hstack((self.raw['zbbbs'][i_split:],self.raw['zbbbs'][:i_split]))

        if add_derived:
            self.add_derived()
        if just_raw:
            return self.raw
        else:
            return self

    def write_geqdsk(self,f_path=None):
        '''
        Function to convert this Equilibrium() object to a eqdsk g-file 

        :param f_path: string containing the targett path of generated eqdsk g-file, including the file name (!)

        :return: none
        '''
        print('Writing Equilibrium to eqdsk g-file...')

        if self.raw:
            try:
                write_eqdsk(f_path,**self.raw)
            except TypeError:
                print("g-eqdsk file could not be written!")

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
        derived['R'] = np.array([raw['rleft'] + i*(raw['rdim']/(raw['nw']-1)) for i in range(raw['nw'])])
        derived['Z'] = np.array([raw['zmid'] - 0.5*raw['zdim'] + i*(raw['zdim']/(raw['nh']-1)) for i in range(raw['nh'])])

        # equidistant psi grid
        derived['psi'] = np.linspace(raw['simag'],raw['sibry'],raw['nw'])

        # corresponding rho_pol grid
        psi_norm = (derived['psi'] - raw['simag'])/(raw['sibry'] - raw['simag'])
        derived['rho_pol'] = np.sqrt(psi_norm)

        if 'rbbbs' in raw and 'zbbbs' in raw:
            # find the indexes of 'zmaxis' on the high field side (hfs) and low field side (lfs) of the separatrix
            i_zmaxis_hfs = int(len(raw['zbbbs'])/3)+find(raw['zmaxis'],raw['zbbbs'][int(len(raw['zbbbs'])/3):int(2*len(raw['zbbbs'])/3)])
            i_zmaxis_lfs = int(2*len(raw['zbbbs'])/3)+find(raw['zmaxis'],raw['zbbbs'][int(2*len(raw['zbbbs'])/3):])
            
            # find the index of 'zmaxis' in the R,Z grid
            i_zmaxis = find(raw['zmaxis'],derived['Z'])

            # find indexes of separatrix on HFS, magnetic axis, separatrix on LFS in R
            i_R_hfs = find(raw['rbbbs'][i_zmaxis_hfs],derived['R'][:int(len(derived['R'])/2)])
            i_rmaxis = find(raw['rmaxis'],derived['R'])
            i_R_lfs = int(len(derived['R'])/2)+find(raw['rbbbs'][i_zmaxis_lfs],derived['R'][int(len(derived['R'])/2):])

            # HFS and LFS R and psirz
            R_hfs = derived['R'][i_R_hfs:i_rmaxis]
            R_lfs = derived['R'][i_rmaxis:i_R_lfs]
            psirzmaxis_hfs = raw['psirz'][i_zmaxis,i_R_hfs:i_rmaxis]
            psirzmaxis_lfs = raw['psirz'][i_zmaxis,i_rmaxis:i_R_lfs]

            # nonlinear R grid at 'zmaxis' based on equidistant psi grid for 'fpol', 'pres', 'ffprime', 'pprime' and 'qpsi'
            derived['R_psi_hfs'] = interpolate.interp1d(psirzmaxis_hfs,R_hfs,fill_value='extrapolate')(derived['psi'][::-1])
            derived['R_psi_lfs'] = interpolate.interp1d(psirzmaxis_lfs,R_lfs,fill_value='extrapolate')(derived['psi'])
        
            # find the R,Z values of the x-point, !TODO: should add check for second x-point in case of double-null equilibrium
            i_xpoint_Z = find(np.min(raw['zbbbs']),raw['zbbbs']) # assuming lower null, JET-ILW shape for now
            derived['R_x'] = raw['rbbbs'][i_xpoint_Z]
            derived['Z_x'] = raw['zbbbs'][i_xpoint_Z]

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
                stdout.write('\r {}% completed'.format(round(100*(find(rho_fs,derived['rho_tor'][1:])+1)/len(derived['rho_tor'][1:]))))
                stdout.flush()
                # check that rho stays inside the lcfs
                if rho_fs < 0.999:
                    self.fluxsurface_find(x_fs=rho_fs,psirz=raw['psirz'],R=derived['R'],Z=derived['Z'],incl_miller_geo=incl_miller_geo,return_self=True)
            stdout.write('\n')

            # find the geometric center, minor radius and extrema of the lcfs manually
            lcfs = self.fluxsurface_center(psi_fs=raw['sibry'],R_fs=raw['rbbbs'],Z_fs=raw['zbbbs'],psirz=raw['psirz'],R=derived['R'],Z=derived['Z'],incl_extrema=True)
            lcfs.update({'R':raw['rbbbs'],'Z':raw['zbbbs']})
            if incl_miller_geo:
                lcfs = self.fluxsurface_miller_geo(fs=lcfs)
            
            # add a zero at the start of all fluxsurface quantities and append the lcfs values to the end of the flux surface data
            for key in fluxsurfaces:
                if key in ['R']:
                    fluxsurfaces[key].insert(0,raw['rmaxis'])
                elif key in ['Z']:
                    fluxsurfaces[key].insert(0,raw['zmaxis'])
                elif key in ['kappa','delta','zeta']:
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
    
    def fluxsurface_find(self,psi_fs=None,psi=None,x_fs=None,x=None,x_label='rho_tor',psirz=None,R=None,Z=None,incl_miller_geo=False,return_self=False):
        '''
        #Function to find the R,Z trace of a flux surface 

        :param psi_fs: (optional) float of the poloidal flux value of the flux surface

        :param psi: (optional) array vector containing the poloidal flux psi from axis to separatrix

        :param x_fs: float of the radial flux label of the flux surface, by default assumed to be in rho_tor

        :param x: (optional) array vector of the radial flux surface label on the same grid as psi, by default assume to be rho_tor

        :param x_label: string of the radial flux label, options (for now) are [default] 'rho_tor', 'rho_pol', 'psi' and 'r'

        :param psirz: array containing the R,Z map of the poloidal flux psi of the magnetic equilibrium

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
                    raise SyntaxError('Equilibrium.fluxsurface_find error: Did not receive enough inputs to deterlefte psi of the flux surface, check your inputs!')
                psi_fs = interpolate.interp1d(x,psi,kind='cubic')(x_fs)
        else:
            raise SyntaxError('Equilibrium.fluxsurface_find error: No radial position of the flux surface was specified, check your inputs!')
        
        fs['psi'] = float(psi_fs)

        refine=None
        # refine the R,Z and psirz grids if the eqdsk resolution is below 512x512
        if self.raw['nw'] < 512 or self.raw['nh'] < 512:
            refine = 512

        # refine the psi R,Z grid if refine
        if refine:
            R_fine = np.linspace(R[0],R[-1],refine)
            Z_fine = np.linspace(Z[0],Z[-1],refine)
            psirz = interpolate.interp2d(R,Z,psirz)(R_fine,Z_fine)
            R=R_fine
            Z=Z_fine

        # find the approximate magnetic axis in psirz
        i_rmaxis = np.where(psirz == np.min(psirz))[1][0]
        zmaxis = Z[np.where(psirz == np.min(psirz))[0][0]]

        # find the R values of the flux surface at 'zmaxis' and the corresponding closest indexes in derived['R']
        psirzmaxis = interpolate.interp2d(R,Z,psirz)(R,zmaxis)

        R_fs_hfs = interpolate.interp1d(psirzmaxis[:i_rmaxis],R[:i_rmaxis])(psi_fs)
        i_R_hfs = find(R_fs_hfs,R)

        R_fs_lfs = interpolate.interp1d(psirzmaxis[i_rmaxis:],R[i_rmaxis:])(psi_fs)
        i_R_lfs = find(R_fs_lfs,R)

        # setup arrays to store R,Z coordinates of the flux surface
        R_fs = R[i_R_hfs:i_R_lfs]
        Z_fs = np.zeros((len(R_fs),2))

        # find the top and bottom Z of the flux surface by slicing psirz by R between R_fs_hfs and R_fs_lfs to ensure max(Z_fs) and min(Z_fs) are included
        for i_R in range(i_R_hfs,i_R_lfs):
            i = i_R - i_R_hfs

            # take a slice of psirz
            psiZ = np.array(psirz[:,i_R])

            # find the minimum of psi in the slice to split the R,Z plane
            i_psiZ_min = find(np.min(psiZ),psiZ)

            # find the maximum of psi in the split slice to ensure not interpolating to a match in psi at a too high/low Z
            i_psiZ_upper_max = find(np.max(psiZ[i_psiZ_min:]),psiZ[i_psiZ_min:])
            i_psiZ_lower_max = find(np.max(psiZ[:i_psiZ_min]),psiZ[:i_psiZ_min])

            # find the upper and lower Z corresponding to the flux surface
            Z_fs[i,0] = interpolate.interp1d(psiZ[i_psiZ_min:i_psiZ_min+i_psiZ_upper_max],Z[i_psiZ_min:i_psiZ_min+i_psiZ_upper_max],bounds_error=False)(psi_fs)
            Z_fs[i,1] = interpolate.interp1d(psiZ[i_psiZ_lower_max:i_psiZ_min],Z[i_psiZ_lower_max:i_psiZ_min],bounds_error=False)(psi_fs)
        
        # find the top and bottom of the Z gap at the inner and outer sides as a consequence of assuming min(R_fs) and max(R_fs) to be on zmaxis
        i_Z_upper = find(np.max([Z_fs[np.where(~np.isnan(Z_fs[:,0]))[0][0],0],Z_fs[np.where(~np.isnan(Z_fs[:,0]))[0][-1],0]]),Z)
        i_Z_lower = find(np.min([Z_fs[np.where(~np.isnan(Z_fs[:,1]))[0][0],1],Z_fs[np.where(~np.isnan(Z_fs[:,1]))[0][-1],1]]),Z)

        # setup arrays to store R,Z coordinates of the missing slices of the flux surface
        Z_fs_ = Z[i_Z_lower:i_Z_upper]
        R_fs_ = np.zeros((len(Z_fs_),2))

        # find the inner and outer R of the flux surface by slicing psirz by Z between Z[i_Z_lower] and Z[i_Z_upper] to ensure min(R_fs) and max(R_fs) are included
        for i_Z in range(i_Z_lower,i_Z_upper):
            i = i_Z - i_Z_lower
            # take a slice of psirz
            psiR = np.array(psirz[i_Z,:])

            # find the minimum of psi in the slice to split the R,Z plane
            i_psiR_min = find(np.min(psiR),psiR)

            # find the inner and outer R corresponding to the flux surface
            R_fs_[i,0] = interpolate.interp1d(psiR[:i_psiR_min],R[:i_psiR_min],bounds_error=False)(psi_fs)
            R_fs_[i,1] = interpolate.interp1d(psiR[i_psiR_min:],R[i_psiR_min:],bounds_error=False)(psi_fs)

        # find the glue edges
        i_upper = sorted(find(Z[i_Z_upper],Z_fs[:,0],n=4))
        i_lower = sorted(find(Z[i_Z_lower],Z_fs[:,1],n=4))
        i_zmaxis_fs_ = find(zmaxis,Z_fs_)

        # merge the upper and lower halves of the flux surface coordinates with the side slices such that the trace starts and ends at the lfs mid-plane
        fs['R'] = np.hstack((R_fs_[i_zmaxis_fs_:,1],R_fs[i_upper[0]:i_upper[-1]][::-1],R_fs_[:-2,0][::-1],R_fs[i_lower[0]+1:i_lower[-1]],R_fs_[:i_zmaxis_fs_+1,1]))
        fs['Z'] = np.hstack((Z_fs_[i_zmaxis_fs_:],Z_fs[i_upper[0]:i_upper[-1],0][::-1],Z_fs_[:-2][::-1],Z_fs[i_lower[0]+1:i_lower[-1],1],Z_fs_[:i_zmaxis_fs_+1]))
        
        # find the flux surface center quantities and add them to the flux surface dict
        fs.update(self.fluxsurface_center(psi_fs=psi_fs,R_fs=fs['R'],Z_fs=fs['Z'],psirz=psirz,R=R,Z=Z,incl_extrema=True))

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
                fs['R_out'] = float(interpolate.interp1d(psirz0[int(len(psirz0)/2):],R[int(len(psirz0)/2):],bounds_error=False)(psi_fs))
                fs['R_in'] = float(interpolate.interp1d(psirz0[:int(len(psirz0)/2)],R[:int(len(psirz0)/2)],bounds_error=False)(psi_fs))

                # find the extrema in Z of the flux surface
                fs['Z_top'] = np.max(Z_fs)
                fs['R_top'] = R_fs[find(fs['Z_top'],Z_fs)]
                fs['Z_bottom'] = np.min(Z_fs)
                fs['R_bottom'] = R_fs[find(fs['Z_bottom'],Z_fs)]
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
        R_bottom = interpolate.interp1d(Z_fs,R_fs)(Z_bottom)
        Z_top = np.max(Z_fs)
        R_top = interpolate.interp1d(Z_fs,R_fs)(Z_top)

        # compute triangularity (delta) and elongation (kappa) of flux surface
        delta_top = (fs['R0'] - R_top)/fs['r']
        delta_bottom = (fs['R0'] - R_bottom)/fs['r']
        fs['delta'] = (delta_top+delta_bottom)/2
        x = np.arcsin(fs['delta'])
        fs['kappa'] = (Z_top - Z_bottom)/(2*fs['r'])

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

        fs['Z_sym_bottom'] = Z_bottom
        fs['R_sym_bottom'] = R_bottom
        fs['Z_sym_top'] = Z_top
        fs['R_sym_top'] = R_top

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
        print('Refining Equilibrium...')
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
