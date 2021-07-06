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
