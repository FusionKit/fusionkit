'''
The TGLF class
'''

import pathlib

## TGLF
class TGLF:
    def __init__(self):
        self.metadata = {}
        self.input = {}
        self.output = {}
    
    # I/O functions
    def write_inputs(fname=None,path=None,control=None,species=None,gaussian=None,geometry=None,expert=None):
        # default values for TGLF input namelists
        header_params = {
            'name':'input.tglf',
            'message':'See https://gafusion.github.io/doc/tglf/tglf_table.html'
        }
        control_params = {
            'name':'Control paramters',
            'UNITS':'GYRO',
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
        species_vector = {
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
                'DRMINDX_LOC':None,
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
            for key in species_vector:
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
                        exit(geometry['name']+" parameter: '"+key+"' has not been specified, please check your geometry inputs!")   
        if expert != None:
            for key in expert:
                expert_params[key] = expert[key]

        tglf_namelist = {
            'meta' : {
                "file":fname,
            },
            'header':header_params,
            'control':control_params,
            'species':species_params,
            'gaussian':gaussian_params,
            'geometry':geometry_params[geometry['name']],
            'expert':expert_params,
        }
        if path == None:
            tglf_namelist['meta'].update({'path':"./",})
        elif isinstance(path,str):
            tglf_namelist['meta'].update({'path':path,})
        else:
            exit("Invalid path string provided!")

        spacer = '#---------------------------------------------------\n'
        for namelist in tglf_namelist:
            if namelist == 'meta':
                if tglf_namelist[namelist]['file'] != None:
                    pathlib.Path(tglf_namelist[namelist]["path"]).mkdir(parents=True, exist_ok=True)
                    f = open(tglf_namelist[namelist]["path"]+tglf_namelist[namelist]["file"],"w+")
                    generated_file = True
                else:
                    exit('File name not specified!')
            elif namelist == 'header':
                for key in tglf_namelist[namelist]:
                    f.write('# '+tglf_namelist[namelist][key]+'\n')
            elif namelist != 'meta' and namelist != 'header':
                f.write(spacer)
                f.write('# '+tglf_namelist[namelist]['name']+'\n')
                f.write(spacer)
                for key in tglf_namelist[namelist]:
                    if key != 'name':
                        f.write("{}={}\n".format(key,tglf_namelist[namelist][key]))
                if namelist != 'expert':
                    f.write('\n')
        f.close()
        if generated_file:
            print('Generated TGLF input file at: '+tglf_namelist['meta']["path"]+tglf_namelist['meta']["file"])
