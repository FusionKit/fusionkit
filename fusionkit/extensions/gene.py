'''
The GENE extension class allows for storing input data for GENE runs and writing input files.
TODO: refactor to take a Plasma object as input data, read input files, retrieve subset of GENE output from remote,
plot GENE scan.log output and ballooning representation
'''

from ..core.remote import Remote

import numpy as np
import pathlib

from scipy import interpolate

class GENE:
    def __init__(self):
        self.metadata = {}
        self.input = {}
        self.output = {}
    
    # I/O functions
    def write_input(self,plasma=None,rho_fs=None,gene_config=None,diagdir=None,f_path=None,f_name=None,imp_composite=False,miller=False):
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
            #n_comp = 1e-19*interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[3]['n']['value'])(rho_fs)
            mass_comp = interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[3]['mass']['value'])(rho_fs)
            charge_comp = (interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[3]['charge']['value'])(rho_fs)).astype(int)
            n_comp = (ne-ni*charge_i-n_LZ*charge_L)/charge_comp
        else:
            n_LZ = (ne-ni*charge_i)/charge_L

        # temperatures
        Te = 1e-3*interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[0]['T']['value'])(rho_fs)
        Ti = 1e-3*interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[1]['T']['value'])(rho_fs)
        
        if miller:
            B0 = interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.equilibrium.derived['Bref_miller'])(rho_fs)
            kappa = interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.equilibrium.derived['kappa'])(rho_fs)
            delta = interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.equilibrium.derived['delta'])(rho_fs)
            zeta = interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.equilibrium.derived['zeta'])(rho_fs)
            s_kappa = interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.equilibrium.derived['s_kappa'])(rho_fs)
            s_delta = interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.equilibrium.derived['s_delta'])(rho_fs)
            s_zeta = interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.equilibrium.derived['s_zeta'])(rho_fs)
            dRodr = interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.equilibrium.derived['dRodr'])(rho_fs)
            dZodr = interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.equilibrium.derived['dZodr'])(rho_fs)

            Lne = a*(interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[0]['n']['z'])(rho_fs))
            Lni = a*(interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[1]['n']['z'])(rho_fs))
            Ln_LZ = a*(interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[2]['n']['z'])(rho_fs))
            if imp_composite:
                #Ln_comp = a*(interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[3]['n']['z'])(rho_fs))
                Ln_comp = ((ne*Lne)-(ni*charge_i*Lni)-(n_LZ*charge_L*Ln_LZ))/(n_comp*charge_comp)
            else:
                Ln_LZ = ((ne*Lne)-(ni*charge_i*Lni))/(n_LZ*charge_L)
            LTe = a*(interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[0]['T']['z'])(rho_fs))
            LTi = a*(interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[1]['T']['z'])(rho_fs))

        else:
            B0 = plasma.equilibrium.derived['Bref_eqdsk']
            Lne = R0*(interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[0]['n']['z'])(rho_fs))
            Lni = R0*(interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[1]['n']['z'])(rho_fs))
            Ln_LZ = R0*(interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[2]['n']['z'])(rho_fs))
            if imp_composite:
                #Ln_comp = R0*(interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[3]['n']['z'])(rho_fs))
                Ln_comp = ((ne*Lne)-(ni*charge_i*Lni)-(n_LZ*charge_L*Ln_LZ))/(n_comp*charge_comp)
            else:
                Ln_LZ = ((ne*Lne)-(ni*charge_i*Lni))/(n_LZ*charge_L)
            LTe = R0*(interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[0]['T']['z'])(rho_fs))
            LTi = R0*(interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[1]['T']['z'])(rho_fs))
        
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
        if gene_config['beta']:
            general_nl.update({"beta" : gene_config['beta']})
        general_nl.update(
            {"bpar" : '.F.',
            "debye2" : -1,
            "hyp_z" : gene_config['hyp_z'],
            "init_cond" : gene_config['init_cond'],}
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
                "trpeps" : str(trpeps)+" ! rho = "+str(rho_fs),
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
            "Bref" : abs(B0),
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
            {"mref" : mass_i,
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

    def io_var_search(host=None,search_dir=None,search_var=None,search_type=None,verbose=False):
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
            i_scan_var = line[0].split(':')[0].split('_000')[1]
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

    def create_new_run(host=None,genecode_dir=None,input_dir=None,duration=None,nodes=2,budget=None,verbose=False):
        print('Creating new run directory on remote {}...'.format(host))
        create_run_commands = ['cd {};'.format(genecode_dir),
                            'if [ ! -d "{input_dir}" ]; then ./newprob; mv prob01 {input_dir}; fi;'.format(input_dir=input_dir),
                            'cd {};'.format(input_dir),
                            'sed -i "s/.\{2\}:00:00'+'/{}:00:00/g" submit.cmd;'.format(duration),
                            'sed -i "s/nodes=1/nodes={}/g" submit.cmd;'.format(nodes),
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