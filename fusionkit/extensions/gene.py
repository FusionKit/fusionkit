'''
The GENE extension class allows for storing input data for GENE runs and writing input files.
TODO: retrieve subset of GENE output from remote, plot GENE scan.log output and ballooning representation
'''

from warnings import WarningMessage
from ..core.remote import Remote
from ..core.utils import *

import numpy as np
import pathlib

from scipy import interpolate

class GENE:
    def __init__(self):
        self.metadata = {}
        self.input = {}
        self.output = {}
    
    # I/O functions
    def write_input(self,plasma=None,rho_fs=None,gene_config=None,diagdir=None,f_path=None,f_name=None,imp_composite=False,geometry=False):
        q = interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.equilibrium.raw['qpsi'])(rho_fs)
        s = interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.equilibrium.derived['s'])(rho_fs)
        alpha = interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.equilibrium.derived['alpha'])(rho_fs)
        beta = interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.equilibrium.derived['beta'])(rho_fs)
        trpeps = interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.equilibrium.derived['epsilon'])(rho_fs)
        R0 = plasma.equilibrium.derived['R0']
        a = plasma.equilibrium.derived['a']
        Ro = interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.equilibrium.derived['Ro'])(rho_fs)

        # densities
        ne = 1e-19*interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[0]['n']['value'])(rho_fs)
        ni = 1e-19*interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[1]['n']['value'])(rho_fs)
        n_LZ = 1e-19*interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[2]['n']['value'])(rho_fs)
        mass_e = plasma.species[0]['mass']['value']
        mass_i = plasma.species[1]['mass']['value']
        mass_LZ = plasma.species[2]['mass']['value']
        charge_e = plasma.species[0]['charge']['value']
        charge_i = plasma.species[1]['charge']['value']
        charge_L = plasma.species[2]['charge']['value']
        if imp_composite:
            n_comp = 1e-19*interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[3]['n']['value'])(rho_fs)
            mass_comp = interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[3]['mass']['value'])(rho_fs)
            charge_comp = (interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[3]['charge']['value'])(rho_fs)).astype(int)
            #n_comp = (ne-ni*charge_i-n_LZ*charge_L)/charge_comp
            #print("quasi-neutrality check: "+str(ne-ni-(n_LZ*charge_L+n_comp*charge_comp)))
            ni = (ne-n_LZ*charge_L-n_comp*charge_comp)/charge_i
            #print("quasi-neutrality check: "+str(ne-ni-(n_LZ*charge_L+n_comp*charge_comp)))
        else:
            #n_LZ = (ne-ni*charge_i)/charge_L
            #print("quasi-neutrality check: "+str(ne-ni-n_LZ*charge_L))
            ni = (ne-n_LZ*charge_L)/charge_i
            #print("quasi-neutrality check: "+str(ne-ni-n_LZ*charge_L))

        # temperatures
        Te = 1e-3*interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[0]['T']['value'])(rho_fs)
        Ti = 1e-3*interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[1]['T']['value'])(rho_fs)
        
        if geometry=='miller':
            B0 = interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.equilibrium.derived['Bref_miller'])(rho_fs)
            kappa = interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.equilibrium.derived['kappa'])(rho_fs)
            delta = interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.equilibrium.derived['delta'])(rho_fs)
            zeta = interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.equilibrium.derived['zeta'])(rho_fs)
            s_kappa = interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.equilibrium.derived['s_kappa'])(rho_fs)
            s_delta = interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.equilibrium.derived['s_delta'])(rho_fs)
            s_zeta = interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.equilibrium.derived['s_zeta'])(rho_fs)
            dRodr = interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.equilibrium.derived['dRodr'])(rho_fs)
            dZodr = interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.equilibrium.derived['dZodr'])(rho_fs)

            Lref = a

        elif geometry=='s-alpha':
            B0 = plasma.equilibrium.derived['Bref_eqdsk']
            Lref = R0

        elif geometry=='tracer_efit':
            #B0 = plasma.equilibrium.derived['Bref_eqdsk']
            #Lref = np.sqrt(2*np.abs(plasma.equilibrium.derived['phi'][-1])/B0)
            Lref=1

        Lne = Lref*(interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[0]['n']['z'])(rho_fs))
        #Lni = Lref*(interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[1]['n']['z'])(rho_fs))
        Ln_LZ = Lref*(interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[2]['n']['z'])(rho_fs))
        if imp_composite:
            Ln_comp = Lref*(interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[3]['n']['z'])(rho_fs))
            #Ln_comp = ((ne*Lne)-(ni*charge_i*Lni)-(n_LZ*charge_L*Ln_LZ))/(n_comp*charge_comp)
            Lni = ((ne*Lne)-(n_LZ*charge_L*Ln_LZ)-(n_comp*charge_comp*Ln_comp))/(ni*charge_i)
        else:
            #Ln_LZ = ((ne*Lne)-(ni*charge_i*Lni))/(n_LZ*charge_L)
            Lni = ((ne*Lne)-(n_LZ*charge_L*Ln_LZ))/(ni*charge_i)
        LTe = Lref*(interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[0]['T']['z'])(rho_fs))
        LTi = Lref*(interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[1]['T']['z'])(rho_fs))
        
        if geometry=='tracer_efit':
            B0 = False
            Lref = False

        ## Species namelist
        species_nl = {
            "nl_name" : "species",
            'electrons' : {
                "nl_name" : "species",
                "name" : "'electrons'",
                "mass" : mass_e/mass_i,
                "charge" : charge_e,
                "temp" : 1.0,
                "dens" : 1.0,
                "omt" : LTe,
                "omn" : Lne,
            },
            'main_ion' : {
                "nl_name" : "species",
                "name" : "'deuterium'",
                "mass" : 1.0,
                "charge" : charge_i,
                "temp" : Ti/Te,
                "dens" : ni/ne,
                "omt" : LTi,
                "omn" : Lni,
            },
            'impurity_1' : {
                "nl_name" : "species",
                "name" : "'beryllium'",
                "mass" : mass_LZ/mass_i,
                "charge" : charge_L,
                "temp" : Ti/Te,
                "dens" : n_LZ/ne,
                "omt" : LTi,
                "omn" : Ln_LZ,
            },
        }
        if imp_composite:
            species_nl['impurity_2'] = {
                "nl_name" : "species",
                "name" : "'composite'",
                "mass" : mass_comp/mass_i,
                "charge" : charge_comp,
                "temp" : Ti/Te,
                "dens" : n_comp/ne,
                "omt" : LTi,
                "omn" : Ln_comp,
            }

        ## Parallelization namelist
        parallel_nl = {
            "nl_name" : "parallelization", 
        }
        if 'n_parallel_sims' in gene_config:
            parallel_nl.update({"n_parallel_sims" : gene_config['n_parallel_sims']})
        else:
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
            "nw0" : gene_config['nw0'],}
        if gene_config['kymin']:
            box_nl["kymin"] = gene_config['kymin']
        if geometry=='tracer_efit' and gene_config['x0']:
            box_nl['x0'] = gene_config['x0']
        box_nl.update({
            "lv" : gene_config['lv'],
            "lw" : gene_config['lw'],
            "mu_grid_type" : gene_config['mu_grid_type'],
            "n0_global" : gene_config['n0_global'],
        })
        if box_nl['n0_global'] != -1111:
            box_nl.update({'adapt_ly': '.T.'})

        ## I/O namelist
        io_nl = {
            "nl_name" : "in_out",
            "diagdir" : diagdir,
            "read_checkpoint" : '.F.',
            "istep_nrg" : 10,
            "istep_fe_time" : 10,
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
            "coll_cons_model" : gene_config['coll_cons_model'],
            "coll" : gene_config['coll'],
            }
        )
        #if general_nl['collision_op'] == "'sugama'":
        #    general_nl.update({"coll_FLR" : '.T.'})
        if gene_config['beta']:
            general_nl.update({"beta" : gene_config['beta']})
        if general_nl['beta'] == -1:
            general_nl.update({"bpar" : '.T.'})
        else:
            general_nl.update({"bpar" : '.F.'})
        general_nl.update({"debye2" : -1,
                            "hyp_z" : gene_config['hyp_z'],
                            "init_cond" : gene_config['init_cond'],
                            "diag_GyroLES" : '.T.'})

        ## External contribution namelist
        external_nl = {
            "nl_name" : "external_contr",
        }

        ## Geometry namelist
        if geometry=='miller':
            geo_nl = {
            "nl_name" : "geometry",
            "magn_geometry" : "'miller'",
            "trpeps" : str(trpeps)+" ! rho = "+str(rho_fs),
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
            'minor_r' : a/Lref,#1.0,
            "major_R" : Ro/Lref,#Ro/a,
            "norm_flux_projection" : '.F.',
            "rhostar" : -1,
            "dpdx_term" : "'full_drift'",
            "dpdx_pm" : -1,
            }
        elif geometry=='s-alpha':
            geo_nl = {
                "nl_name" : "geometry",
                "magn_geometry" : "'s_alpha'",
                "trpeps" : str(trpeps)+" ! rho = "+str(rho_fs),
                "q0" : q,
                "shat" : s,
                "major_R" : 1.0,
                "amhd" : alpha,
                "norm_flux_projection" : '.F.',
                "rhostar" : -1,
                "dpdx_term" : "'full_drift'",
                "dpdx_pm" : -1,}
        elif geometry=='tracer_efit':
            geo_nl = {
                "nl_name" : "geometry",
                "magn_geometry" : "'tracer_efit'",
                "geomfile":gene_config['geomfile'],
                "sign_Ip_CW":1,
                "sign_Bt_CW":1,
                "norm_flux_projection" : '.F.',
                "rhostar" : -1,
                "dpdx_term" : "'full_drift'",
                "dpdx_pm" : -1,}

        ## Info namelist
        info_nl = {
            "nl_name" : "info",
        }

        ## Units namelist
        units_nl = {
            "nl_name" : "units",
            "Tref" : Te,
            "nref" : ne
        }
        if B0:
            units_nl.update({"Bref" : abs(B0)})
        if Lref:
            units_nl.update({"Lref" : Lref})
        units_nl.update({
            "mref" : mass_i,
            "omegatorref" : 0,
        })

        ## Complete GENE namelist
        gene_nl = {
            'meta' : {
                "path" : f_path,
                "file" : f_name,
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

    def read_scan_log(f_path=None,any_sign=False,no_split=False):
        # setup storage
        scan_log = {}
        scan_log['x_var'] = [[]]
        scan_log['gamma'] = [[]]
        scan_log['omega'] = [[]]

        # read the scan.log
        f = open(f_path, 'r')
        lines = f.readlines()
        f.close()

        # get the scan variable name from the header line
        scan_log['x_label'] = lines[0].split()[2]

        # set the amount of sign splits
        i=0

        # go through the lines and append the values to the scan_log dict, split the scan based on sign changes
        for j,line in enumerate(lines[1:]):
            if j>= 1:
                if any_sign:
                    if np.sign(float(line.split()[-1])) != np.sign(float(lines[j].split()[-1])):
                        #print('sign change')
                        i+=1
                        scan_log['x_var'].append([])
                        scan_log['gamma'].append([])
                        scan_log['omega'].append([])
                elif no_split:
                    i=0
                elif np.sign(float(line.split()[-1])) < 0 and np.sign(float(lines[j].split()[-1])) > 0:
                    #print('sign change')
                    i+=1
                    scan_log['x_var'].append([])
                    scan_log['gamma'].append([])
                    scan_log['omega'].append([])
            scan_log['x_var'][i].append(float(line.split()[2]))
            scan_log['gamma'][i].append(float(line.split()[-2]))
            scan_log['omega'][i].append(float(line.split()[-1]))
        return scan_log

    def read_energy(self,output_path=None,file='energy_0001'):
        # read the energy file
        lines = read_file(path=output_path,file=file)

        # set file dependent variables
        header = 15
        energy = {
            'time':[],
            'Etot':[],
            'dEdt_tot':[],
            'dEdt_drive':[],
            'dEdt_source':[],
            'dEdt_coll':[],
            'dEdt_Dz':[],
            'dEdt_Dv':[],
            'dEdt_Dxy':[],
            'dEdt_nl':[],
            'dEdt_zv':[],
            'dEdt_rest':[],
            'dEdt_check':[],
            'dEdt_tot2':[]
        }

        descriptions = {}

        # line by line read the header with the descriptions and store them
        for i_value,value in enumerate(list(energy.keys())):
            line = lines[i_value].strip().split()
            description = ' '.join(line[2:])
            descriptions.update({value:description})

        # line by line starting after the header read the values and sort them to the predefined keys
        for line in lines[header:]:
            line = line.strip().split()
            for i_value,value in enumerate(list(energy.keys())):
                energy[value].append(float(line[i_value]))
        
        # convert every variable to 1d array for easy post processing
        for value in energy:
            if isinstance(energy[value],list):
                energy[value] = np.array(energy[value])

        energy.update({'descriptions':descriptions})

        return energy

    def read_fe_time(output_path=None,file=None):
        # read the e_time_spec/fe_time/fe_time_spec/w_time_spec file
        lines = read_file(path=output_path,file=file)

        # set file dependent variables
        descriptions = {}
        header = 0
        for line in lines:
            line = line.strip().split('#')
            if len(line)>1:
                header+=1

        fe_time = {
            'time':[],
        }
        # check if file is a potential energy file
        if file.strip().split('_')[0]=='e':
            fe_time.update({'FE_pot':[]})
        # check if file is a free energy file
        elif file.strip().split('_')[0]=='fe':
            fe_time.update({'FE':[]})
        # check if file is a kinetic energy file
        elif file.strip().split('_')[0]=='w':
            fe_time.update({'FE_kin':[]})
        fe_time.update({
            'dFEdt_drive':[],
            'dFEdt_disp_v':[],
            'dFEdt_disp_z':[],
            'dFEdt_disp_kperp':[],
            'dFEdt_coll':[],
            'dFEdt_par':[],
            'dFEdt_curv':[],
        })
        # check if file is a fe_time_total file
        if header == 13:
            fe_time.update({
                'dFEdt_rhs':[],
                'dFEdt_lhs':[],
                'dFEdt_lhs-rhs':[]
            })
        # else unknown file format
        elif header < 10 or header > 13:
            raise ValueError('Unknown fe_time file format, could not read the specified file!')
        
        if lines:
            # line by line read the header with the descriptions and store them
            for i_value,value in enumerate(list(fe_time.keys())):
                line = lines[i_value].strip().split()
                description = ' '.join(line[2:])
                descriptions.update({value:description})

            # line by line starting after the header read the values and sort them to the predefined keys
            for line in lines[header:]:
                line = [autotype(value) for value in line.strip().split()]
                for i_value,value in enumerate(list(fe_time.keys())):
                    fe_time[value].append(line[i_value])
            
            fe_time.update({'descriptions':descriptions})

            return list_to_array(fe_time)
    
    def read_nrg(self,output_path=None,file='nrg_0001',nspecies=1):
        # read the nrg file
        lines = read_file(path=output_path,file=file)

        # set file dependent variables
        header_lines = []
        species = {}
        time = []

        # prepare the data structure to slice by species
        for i_species in range(0,nspecies):
            if i_species not in species:
                species[i_species] = {
                    'n1':[],
                    'u1_par':[],
                    'T1_par':[],
                    'T1_perp':[],
                    'Gamma_es':[],
                    'Gamma_em':[],
                    'Q_es':[],
                    'Q_em':[],
                    'Pi_es':[],
                    'Pi_em':[],
                }

        # find all the lines with a timestamp
        for i_line,line in enumerate(lines):
            line = line.split()
            if len(line)==1:
                time.append(float(line[0]))
                header_lines.append(i_line)

        # for each timestamp slice the data per species, per variable and store 
        for i_line in header_lines:
            for i_species,line in enumerate(lines[i_line+1:i_line+1+nspecies]):
                line = line.split()
                for i_value,key in enumerate(species[i_species]):
                    species[i_species][key].append(float(line[i_value]))

        # convert all outputs to arrays for later ease of use
        time = np.array(time)
        for i_species in range(0,nspecies):
            for value in species[i_species]:
                if isinstance(species[i_species][value],list):
                    species[i_species][value] = np.array(species[i_species][value])

        nrg = {'time':time,'species':species}

        return nrg

    def read_omega(self,output_path=None,file='omega_0001'):
        # read the omega file
        line = read_file(path=output_path,file=file)
        # strip the line and split into the three separate values
        if line:
            line = line[0].strip().split()
            #print(line)
            # convert all str to float
            line = [float(value) for value in line]

            omega = {'ky':line[0],'gamma':line[1],'omega':line[2]}
        else:
            omega = {'ky':np.NaN,'gamma':np.NaN,'omega':np.NaN}
        
        return omega

    def read_parameters(self,output_path=None,file='parameters_0001'):
        # read the parameters file
        lines = read_file(path=output_path,file=file)

        # set file dependent variables
        parameters = {}
        namelist_list = []

        for line in lines:
            # check if the line is not empty whitespace
            if line.strip():
                # remove whitespace
                line = line.strip()
                # find the namelist header names by checking for &
                if '&' in line:
                    # keep track of the already checked off namelists
                    namelist_name = line.strip('&')
                    namelist_list.append(namelist_name)

                    # add the namelist as a key to the parameters dict if does not exist yet
                    if namelist_name not in parameters:
                        parameters.update({namelist_name:{}})
                    
                    # special treatment for the multiple species namelists
                    if namelist_name == 'species':
                        if parameters[namelist_list[-1]].keys():
                            i_species = len(list(parameters[namelist_list[-1]].keys()))
                        else:
                            i_species = 0
                        parameters[namelist_list[-1]].update({i_species:{}})

                # check if line is a variable line
                elif '/' not in line:
                    # process the variables and add them to the current namelist dict
                    line = [value.strip() for value in line.split('=')]
                    key = line[0]
                    value = autotype(line[1])

                    if namelist_list[-1]!='species':
                        parameters[namelist_list[-1]].update({key:value})
                    else:
                        parameters[namelist_list[-1]][i_species].update({key:value})

        return parameters

    def read_geometry(self,output_path=None,file=None,filter=True):
        if self.parameters:
            params = self.parameters
        else:
            filter = False
            params = {}
        # read the s-alpha/miller/tracer_efit file
        lines = read_file(path=output_path,file=file)

        # set file dependent variables
        parameters = {}
        geometry = {
            'g_xx':[],
            'g_xy':[],
            'g_xz':[],
            'g_yy':[],
            'g_yz':[],
            'g_zz':[],
            'B0':[],
            'dB0dx':[],
            'dB0dy':[],
            'dB0dz':[],
            'J':[],
            'R':[],
            'phi':[],
            'Z':[],
            'dxdR':[],
            'dxdZ':[]
        }
        header_lines = []

        # find the header lines for the parameters namelist at the top of the geometry file
        for i_line,line in enumerate(lines):
            line = line.strip()
            for char in ['&','/']:
                if char in line:
                    header_lines.append(i_line)
        
        # read, strip and store all the variables in the header parameters namelist
        for line in lines[header_lines[0]+1:header_lines[1]-1]:
            if line.strip():
                line = [value.strip() for value in line.split('=')]
                key = line[0]
                value = autotype(line[1])

                parameters.update({key:value})
        
        # line by line starting after the header read the geometry values and sort them to the predefined keys
        for line in lines[header_lines[-1]+1:]:
            line = line.strip().split()
            for i_value,value in enumerate(list(geometry.keys())):
                geometry[value].append(float(line[i_value]))
        
        # convert every variable to 1d array for easy post processing
        for value in geometry:
            if isinstance(geometry[value],list):
                geometry[value] = np.array(geometry[value])

        # if parameter filtering is on check if the parameter is an unnecessary duplicate of the input parameters before adding them to the output dict
        if filter:
            for parameter in parameters:
                if params and parameter not in list(params['geometry'].keys())+list(params['units'].keys())+list(params['general'].keys()):
                    geometry.update({parameter:parameters[parameter]})
        # or just add all the parameters to the output dict
        else:
            geometry.update({'parameters':parameters})

        return geometry

    def read_field(self,parameters,output_path=None,file='field_0001'):
        # set file dependent variables
        _fields = {
            'phi':[],
            'B_par':[],
            'B_perp':[]
        }
        n_fields = parameters['info']['n_fields']

        # get input parameter values
        # get local/global booleans
        local = {}
        for key in ['x_local','y_local']:
            if key not in parameters['general']:
                local[key] = True
            else:
                local[key] = parameters['general'][key]
        
        # set dimension sizes
        nx0 = parameters['box']['nx0']
        nky0 = parameters['box']['nky0']
        nz0 = parameters['box']['nz0']
        size_field = {'x':nx0,'y':nky0,'z':nz0}

        # specify byte precision and endianness
        end = str.lower(parameters['info']['ENDIANNESS'])
        prec = str.lower(parameters['info']['PRECISION'])

        fields = self.read_binary_file(path=output_path,file=file,endianness=end,precision=prec,n_arrays=n_fields,size=size_field,arrays=_fields,local=local)

        return fields

    def read_mom(self,parameters,output_path=None,file='mom_0001'):
        # set file dependent variables
        _moments = {
            'n1':[],
            'T1_par':[],
            'T1_perp':[],
            'q1_par':[],
            'q1_perp':[],
            'u1_par':[],
        }
        n_moms = parameters['info']['n_moms']

        # get input parameter values
        # get local/global booleans
        local = {}
        for key in ['x_local','y_local']:
            if key not in parameters['general']:
                local[key] = True
            else:
                local[key] = parameters['general'][key]
        
        # set dimension sizes
        nx0 = parameters['box']['nx0']
        nky0 = parameters['box']['nky0']
        nz0 = parameters['box']['nz0']
        size_mom = {'x':nx0,'y':nky0,'z':nz0}

        # specify byte precision and endianness
        end = str.lower(parameters['info']['ENDIANNESS'])
        prec = str.lower(parameters['info']['PRECISION'])

        moments = self.read_binary_file(path=output_path,file=file,endianness=end,precision=prec,n_arrays=n_moms,size=size_mom,arrays=_moments,local=local)

        return moments

    def read_binary_file(path=None,file=None,endianness='little',precision='double',n_arrays=None,size={},arrays={},local={}):
        # set byte order
        if endianness == 'native':
            _endianness = '='
        elif endianness == 'little':
            _endianness = '<'
        else:
            _endianness = '>'

        # set byte sizes
        byte_size_int = 4
        if precision == 'double':
            byte_size_float = 8
        else:
            byte_size_float = 4
        byte_size_complex = 2*byte_size_float

        # set numpy dtypes for the different bytes taking endianness and bitesizes in account
        dtype_real = np.dtype('{}f{}'.format(_endianness,str(byte_size_float)))
        dtype_complex = np.dtype('{}c{}'.format(_endianness,str(byte_size_complex)))

        # define GENE specific variables
        time = []
        try:
            size_array = size['x']*size['y']*size['z']
        except:
            raise ValueError('No information of the size of the x, y and/or z dimensions was provided, check your inputs!')

        # define byte sizes entries for GENE binary files
        # NOTE: Fortran binary output is of type: int(size_of_entry) float/complex(entry) int(size_of_entry), so appropriate byte_size_int is added to entry sizes
        byte_size_time = byte_size_float + 2*byte_size_int # time entry + the two Fortran entry size int's
        byte_size_array = size_array * byte_size_complex # number of grid points in an array entry * number of bytes in a complex entry
        byte_size_arrays = n_arrays * (byte_size_array + 2*byte_size_int) # number of arrays * (number of bytes in an array entry + the two Fortran entry size int's)

        # check if the provided path exists
        if os.path.isdir(path):
            # check if the provided file name exists in the output path
            if os.path.isfile(path+file):
                # read the GENE binary file
                with open(path+file,'rb') as _file:
                    # get the time and append it, find the number of time entries by taking the total number of bytes / number of bytes of one time + arrays entry
                    for i_time in range(int(os.path.getsize(path+file)/(byte_size_time + byte_size_arrays))):
                        start_byte = i_time*(byte_size_arrays + byte_size_time) + byte_size_int
                        # skip the appropriate number of bytes for a time entry + array entry + the first Fortran int entry
                        _file.seek(start_byte)
                        time.append(np.fromfile(_file,count=1,dtype=dtype_real)[0])
                        #print(time)
                        # get each array and append it
                        for i_array,key_array in enumerate(list(arrays.keys())[:n_arrays]):
                            #print(key_array)
                            start_byte = byte_size_time + i_time*(byte_size_arrays + byte_size_time) + i_array*(byte_size_array + 2*byte_size_int) + byte_size_int
                            _file.seek(start_byte)
                            array = np.fromfile(_file,count=size_array,dtype=dtype_complex)
                            #print(array)
                            if local:
                                if local['x_local'] and not local['y_local']:
                                    # y-global has yx order
                                    array = np.swapaxes(array.reshape(size['y'], size['x'], size['z'], order="F"),0,1)
                                else:
                                    array = array.reshape(size['x'], size['y'], size['z'], order="F")
                            arrays[key_array].append(array)
                            #print(arrays[key_array][-1])
            else:
                raise ValueError('The file {} does not exist!'.format(file))
        else:
            raise ValueError('{} is not a valid path to a directory!'.format(path))

        # remove any fields that are empty
        for key in list(arrays.keys()):
            if not arrays[key]:
                del arrays[key]

        # add the time array
        arrays.update({'time':np.array(time)})

        return arrays

    # remote run functions
    def search_input_var(host=None,search_dir=None,search_var=None,search_type=None,verbose=False):
        search_commands = ['cd {};'.format(search_dir),
                        'grep {} parameters_*;'.format(search_var)]
        response = Remote().send_commands(host=host,commands=''.join(search_commands),verbose=False)
        # create storage for all the partially matching grep results
        searchlist = {}
        # go through the remote response line by line
        for line in response:
            # find the partial grep match variable value
            line = line.strip().split('=')
            search_var_value = line[1].strip()
            # find the partial grep match variable name
            _search_var = (line[0].split(':')[1]).strip()
            # identify the original scan number 
            i_scan_var = line[0].split(':')[0].split('_00')[1]
            # remove _eff parameter file results to prevent duplication
            if '_eff' not in i_scan_var:
                if _search_var not in searchlist:
                    searchlist.update({_search_var:{}})
                try:
                    search_var_value = int(search_var_value)
                except:
                    try:
                        search_var_value = float(search_var_value)
                    except:
                        search_var_value = str(search_var_value)
                searchlist[_search_var].update({int(i_scan_var):search_var_value})
        if search_type == 'scanlist':
            _scanlist = []
            for i_scan_var in searchlist[search_var]:
                _scanlist.append(searchlist[search_var][i_scan_var])
            if verbose:
                print('{} = {} !scanlist: {}'.format(search_var,_scanlist[0],', '.join(map(str,_scanlist))))
            return _scanlist
        elif search_type == 'five_point_scan':
            _five_point_scan = []
            _five_point_scan.append(0.8*searchlist[search_var][1])
            _five_point_scan.append(0.9*searchlist[search_var][1])
            _five_point_scan.append(searchlist[search_var][1])
            _five_point_scan.append(1.1*searchlist[search_var][1])
            _five_point_scan.append(1.2*searchlist[search_var][1])
            if verbose:
                print('{}: {}'.format(search_var,_five_point_scan))
            return _five_point_scan

    def create_new_run(host=None,genecode_dir=None,input_dir=None,duration=None,nodes=2,queue='skl_fua_prod',budget=None,verbose=False):
        print('Creating new run directory on remote {}...'.format(host))
        create_run_commands = ['cd {};'.format(genecode_dir),
                            'if [ ! -d "{input_dir}" ]; then ./newprob; mv prob01 {input_dir}; fi;'.format(input_dir=input_dir),
                            'cd {};'.format(input_dir),
                            'sed -i "s/.\{2\}:00:00'+'/{}:00:00/g" submit.cmd;'.format(duration),
                            'sed -i "s/nodes=1/nodes={}/g" submit.cmd;'.format(nodes),
                            'sed -i "s/skl_fua_prod/{}/g" submit.cmd;'.format(queue),
                            'sed -i "s/FUSIO_ALL/{}/g" submit.cmd;'.format(budget),
                            'sed -i "0,/srun -l/s//#srun -l/" submit.cmd;',
                            'sed -i "s/#.\/scanscript/.\/scanscript/" submit.cmd;',
                            'sed -i "s/--mps 4/--mps {}/" submit.cmd;'.format(nodes)]

        response = Remote().send_commands(host=host,
                                        commands=''.join(create_run_commands),
                                        verbose=verbose)
        return response
    
    def submit_remote_run(host=None,genecode_dir=None,input_dir=None,user=None,verbose=True):
        print('Submitting job on remote {}...'.format(host))
        submit_run_commands = ['cd {}/{};'.format(genecode_dir,input_dir),
                               'sbatch submit.cmd;',
                               'squeue -u {};'.format(user)]

        response = Remote().send_commands(host=host,
                                        commands=''.join(submit_run_commands),
                                        verbose=verbose)
        return response
