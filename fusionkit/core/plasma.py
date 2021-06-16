'''
The Plasma class is the core object of fusionkit.
'''

import copy
import numpy as np
from scipy import interpolate

from .equilibrium import Equilibrium
from .utils import calcz
from ..extensions.ex2gk import EX2GK
from ..extensions.jet_ppf import JET_PPF

## PLASMA
class Plasma:
    '''
    Class to handle any and all data related to the plasma in a magnetic confinement fusion device

    :self.metadata: dict for storage of all metadata related to the plasma state (e.g. device, shot, etc.) and the Plasma object

    :self.species: dict for storage of all species data (name, mass, charge-state and 1D profile data), 
                   any species profile data is automatically remapped onto the equilibrium
    
    :self.num_species: integer value of the total number of plasma particle species, can be updated with update_species()

    :self.equilibrium: Equilibrium object to store the magnetic equilibrium of the plasma state

    :self.diagnostics: dict containing any and all diagnostic data related to the plasma state
    '''
    def __init__(self):
        self.metadata = {}
        self.species = {}
        self.num_species = len(self.species)
        self.equilibrium = Equilibrium()
        self.diagnostics = {}
        self.dataset = {}
    
    def add_species(self,name=None,mass=None,charge=None,n=None,T=None,remap=True):
        '''
        Function to add a plasma species to a Plasma object, NOTE: when adding multiple species UNITS HAVE TO BE CONSISTENT/COMPATIBLE!

        :param name: string to identify the species

        :param mass: dict specifying the mass value and unit, {'value':float, 'unit':string}

        :param charge: dict specifying the species electric charge value/radial profile and unit, {'grid':array, 'value':float/array, 'unit':string}

        :param n: dict specifying the species density radial coordinate, profile and unit, {'x':array, 'x_label':string, 'value':array, 'unit':string}

        :param T: dict specifying the species temperature radial profile and unit, {'x':array, 'x_label':string, 'value':array, 'unit':string}

        :param remap: boolean specifying whether [default] remapping the radial profiles onto the equilibrium (True) or not (False)

        :return: self
        '''
        # set the species index to the current length, this starts indexing naturally at 0 and increases automatically
        i_species = len(self.species)
        
        # append the species to the Plasma species list, !TODO: add more properties e.g. omega
        self.species.update({i_species:{
            'name':name,
            'mass':mass,
            'charge':charge,
            'n':n,
            'T':T,
            }
        })

        # loop over all species properties
        for property in self.species[i_species]:
            # save the property string
            property_label = copy.deepcopy(property)
            # introduce shorthand
            property = self.species[i_species][property]
            # check if property is physics quantity and a profile array
            if 'value' in property and isinstance(property['value'],np.ndarray):
                # map the profile on the equilibrium, robust to multiple radial coordinate inputs by x_label attribute
                property['x'], property['value'] = self.equilibrium.map_on_equilibrium(x=property['x'],y=property['value'],x_label=property['x_label'])
                # if property in plasma kinetic profiles, compute normalised logarithmic gradients
                if property_label in ['n','T']:
                    property.update({'z':calcz(self.equilibrium.derived['r'],property['value'])})
        
        # update the species derived quantities p, beta and alpha in the Equilibrium
        self.update_species_derived()
        return self
    
    def update_species_derived(self,Bref=None):
        '''
        Function to update species derived plasma quantities p, beta, alpha, quasi-neutrality density and switch Bref for these quantities

        :param Bref: scalar or vector reference magnetic field used in calculating beta

        :return: self
        '''
        self.num_species = len(self.species)
        for i_species in self.species:
            #print('i_species: {}'.format(i_species))
            if i_species > 0:
                additive=True
            else:
                additive=False
            #print('n: {}'.format(self.species[i_species]['n']['value']))
            #print('T: {}'.format(self.species[i_species]['T']['value']))
            e = 1.602176E-19
            mu0 = 4*np.pi*1E-7
            p = self.species[i_species]['n']['value']*(e*self.species[i_species]['T']['value'])
            #print('p: {}'.format(p))
            if not Bref and self.equilibrium.derived:
                Bref = self.equilibrium.derived['Bref_eqdsk']
            beta = 2*mu0*(p/(Bref**2))
            #print('beta: {}'.format(beta))
            self.species[i_species].update({'p':{'value':p,'unit':'Pa'},'beta':{'value':beta,'unit':'-'}})
            self.equilibrium.update_pressure(p=p,additive=additive)
        return self
    
    def update_quasineutrality(self,name=None,density=True,gradient=True):
        '''
        Function to self-consistently update the quasi-neutrality of a Plasma object 
        '''
        # storage for all species except the quasi-neutrality target
        unmodified = {}
        # loop over all species
        for i_species in self.species:
            # find the target index
            if self.species[i_species]['name'] == name:
                i_target = copy.deepcopy(i_species)
            # store the rest in unmodified, incl. the maximum density to find the electrons
            else:
                unmodified.update({i_species:np.max(self.species[i_species]['n']['value'])})
        
        # find the electron species
        i_electrons = max(unmodified, key=unmodified.get)
        # remove the electrons from unmodified
        del(unmodified[i_electrons])

        # update the density to achieve (numerical) quasi-neutrality
        if density:
            # start with the electron density
            self.species[i_target]['n']['value'] = copy.deepcopy(self.species[i_electrons]['n']['value'])
            # subtract the density of all the remaining species
            for i_species in unmodified:
                self.species[i_target]['n']['value'] -= self.species[i_species]['charge']['value']*self.species[i_species]['n']['value']
            self.species[i_target]['n']['value'] /= self.species[i_target]['charge']['value']
            # update the normalised logarithmic gradient self-consistently
            if not gradient:
                self.species[i_target]['n']['z'] = calcz(self.equilibrium.derived['r'],self.species[i_target]['n']['value'])
        # compute the normalised logarithmic gradients to achieve (numerical) quasi-neutrality
        if gradient:
            # start with the electron density normalised gradient
            self.species[i_target]['n']['z'] = copy.deepcopy(self.species[i_electrons]['n']['value'])*copy.deepcopy(self.species[i_electrons]['n']['z'])
            # subtract the gradient of all the remaining species
            for i_species in unmodified:
                self.species[i_target]['n']['z'] -= self.species[i_species]['charge']['value']*self.species[i_species]['n']['value']*self.species[i_species]['n']['z']
            
            self.species[i_target]['n']['z'] /= self.species[i_target]['n']['value']*self.species[i_target]['charge']['value']

        return self

    def composite_species(self,species_names=[],pop=True):
        '''
        Function to create a composite species from two (or TODO more) species in a Plasma object
        '''
        # instantiate composite species properties
        #self.add_species(name='composite')

        # create storage for the properties of the selected species to composite
        composite_species = {}

        species_indexes = [i_species for i_species in self.species if self.species[i_species]['name'] in species_names]
        for i_species in species_indexes:
            #print(self.species[i_species])
            for key in self.species[i_species]:
                if key!='name' and 'value' in self.species[i_species][key].keys():
                    if key not in composite_species:
                        composite_species.update({key:[self.species[i_species][key]['value']]})
                    else:
                        # TODO add check that units of appended 
                        composite_species[key].append(self.species[i_species][key]['value'])

        weighted_sum_charge = composite_species['n'][0]*composite_species['charge'][0]+composite_species['n'][1]*composite_species['charge'][1]
        weighted_sum_charge_sqr = composite_species['n'][0]*composite_species['charge'][0]**2+composite_species['n'][1]*composite_species['charge'][1]**2
        weighted_sum_charge_mass = composite_species['n'][0]*composite_species['charge'][0]*composite_species['mass'][0]+composite_species['n'][1]*composite_species['charge'][1]*composite_species['mass'][1]

        charge_comp = (np.round(weighted_sum_charge_sqr/weighted_sum_charge)).astype(int)
        n_comp = weighted_sum_charge/charge_comp
        mass_comp = weighted_sum_charge_mass/(charge_comp*n_comp)
        T_comp = np.mean(np.array(composite_species['T']),axis=0)

        if pop:
            p_pop = -1*np.sum(np.array(composite_species['p']),axis=0)
            self.equilibrium.update_pressure(p=p_pop,additive=True)
            for i_species in species_indexes:
                del(self.species[i_species])
        x = copy.deepcopy(self.equilibrium.derived['rho_tor'])
        self.add_species(name='composite',
                         mass={'x':x,'x_label':'rho_tor','value':mass_comp,'unit':'u'},
                         charge={'x':x,'x_label':'rho_tor','value':charge_comp,'unit':'e'},
                         n={'x':x,'x_label':'rho_tor','value':n_comp,'unit':'m^{-3}'},
                         T={'x':x,'x_label':'rho_tor','value':T_comp,'unit':'eV'})

        return self

    def construct_legacy_plasma(self,tokamak=None,shot=None,imp_rescale=False,imp_composite=False,imp_sertoli=False,miller=False,gyro=False):
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