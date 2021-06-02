'''
The GENE extension class allows for storing input data for GENE runs and writing input files.
TODO: refactor to take a Plasma object as input data, read input files, retrieve subset of GENE output from remote,
plot GENE scan.log output and ballooning representation
'''

import numpy as np
import pathlib

class GENE:
    def __init__(self):
        self.metadata = {}
        self.input = {}
        self.output = {}
    
    # I/O functions
    def write_input(self,rho=None,dataset=None,gene_config=None,diagdir=None,f_path=None,f_name=None,imp_composite=False,miller=False):
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
