'''
The QuaLiKiz class
'''

import numpy as np
from scipy import interpolate
from fusionkit.core.utils import *

## QuaLiKiz
class QLK:
    def __init__(self):
        self.metadata = {}
        self.input = {}
        self.output = {}
    
    # I/O functions
    def write_input(rho_fs=None,plasma=None,output_loc=None,fname=None,imp_composite=False,ktheta=None):
        q = interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.equilibrium.raw['qpsi'])(rho_fs)
        s = interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.equilibrium.derived['s'])(rho_fs)
        alpha = interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.equilibrium.derived['alpha'])(rho_fs)
        R0 = plasma.equilibrium.derived['R0']
        a = plasma.equilibrium.derived['a']
        x = interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.equilibrium.derived['r']/plasma.equilibrium.derived['a'])(rho_fs)
        Ro = interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.equilibrium.derived['Ro'])(rho_fs)
        B0 = abs(plasma.equilibrium.derived['Bref_eqdsk'])

        # densities
        ne = 1e-19*interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[0]['n']['value'])(rho_fs)
        ni = 1e-19*interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[1]['n']['value'])(rho_fs)
        n_LZ = 1e-19*interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[2]['n']['value'])(rho_fs)
        mass_i = plasma.species[1]['mass']['value']
        mass_LZ = plasma.species[2]['mass']['value']
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

        RLne = R0*(interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[0]['n']['z'])(rho_fs))
        RLni = R0*(interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[1]['n']['z'])(rho_fs))
        RLn_LZ = R0*(interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[2]['n']['z'])(rho_fs))
        if imp_composite:
            #Ln_comp = R0*(interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[3]['n']['z'])(rho_fs))
            RLn_comp = ((ne*RLne)-(ni*charge_i*RLni)-(n_LZ*charge_L*RLn_LZ))/(n_comp*charge_comp)
        else:
            RLn_LZ = ((ne*RLne)-(ni*charge_i*RLni))/(n_LZ*charge_L)
        RLTe = R0*(interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[0]['T']['z'])(rho_fs))
        RLTi = R0*(interpolate.interp1d(plasma.equilibrium.derived['rho_tor'],plasma.species[1]['T']['z'])(rho_fs))

        Mach_tor = 0
        Au_tor = 0
        Mach_par = 0
        Au_par = 0
        gamma_E = 0

        ## QuaLiKiz input preparation
        from qualikiz_tools.qualikiz_io.inputfiles import QuaLiKizXpoint, Electron, Ion, IonList
        from qualikiz_tools.qualikiz_io.inputfiles import QuaLiKizPlan

        if not ktheta:
            kthetarhos = list(np.linspace(0.1,0.8,8))
        else:
            kthetarhos=ktheta
        elec = Electron(T=Te,n=ne,At=RLTe,An=RLne,type=1,anis=1, danisdr=0)
        ion0 = Ion(T=Ti,n=ni,At=RLTi,An=RLni,A=mass_i,Z=charge_i,type=1,anis=1,danisdr=0)
        ion1 = Ion(T=Ti,n=n_LZ,At=RLTi,An=RLn_LZ,A=mass_LZ,Z=charge_L,type=1,anis=1,danisdr=0)
        if imp_composite:
            ion2 = Ion(T=Ti,n=n_comp,At=RLTi,An=RLn_comp,A=mass_comp,Z=charge_comp,type=1,anis=1,danisdr=0)
            ions = IonList(ion0,ion1,ion2)
        else:
            ions = IonList(ion0,ion1)

        meta = {
            "phys_meth": 1,
            "coll_flag": 1,
            "rot_flag": 0,
            "verbose": True,
            "separateflux": False,
            "write_primi": True,
            "numsols": 2,
            "relacc1": 0.00001,
            "relacc2": 0.0002,
            "absacc1": 0,
            "absacc2": 0,
            "integration_routine": 1,
            "maxruns": 1,
            "maxpts": 500000000.0,
            "timeout": 60,
            "ETGmult": 1,
            "collmult": 0.5,
            "R0": R0
        }
        spatial = {
            "x": x,
            "rho": rho_fs,
            "Ro": Ro,
            "Rmin": a,
            "Bo": B0,
            "q": q,
            "smag": s,
            "alpha": alpha,
            "Machtor": Mach_tor,
            "Autor": Au_tor,
            "Machpar": Mach_par, 
            "Aupar": Au_par,
            "gammaE": gamma_E
        }
        options = {
            "set_qn_normni": True,
            "set_qn_normni_ion": 0,
            "set_qn_An": True,
            "set_qn_An_ion": 0,
            "check_qn": True,
            "x_eq_rho": True,
            "recalc_Nustar": False,
            "recalc_Ti_Te_rel": False,
            "assume_tor_rot": True,
            "puretor_abs_var": "Machtor",
            "puretor_grad_var": "gammaE"
        }

        xpoint_base = QuaLiKizXpoint(**meta, kthetarhos=kthetarhos, electrons=elec, ions=ions, **spatial, **options)

        scan_dict = {'Ate': [xpoint_base['Ate']]}
        plan = QuaLiKizPlan(scan_dict=scan_dict,scan_type='hyperedge',xpoint_base=xpoint_base)
        plan.to_json(output_loc+fname)
    
    def read_output(data_loc=None,ky_list=None,no_split=None):
        qlk_dataset = {}
        qlk_dataset['x_var'] = [[]]
        qlk_dataset['gamma'] = [[]]
        qlk_dataset['omega'] = [[]]

        f_loc = data_loc+'ome_GB.dat'
        f = open(f_loc, 'r')
        omegas = f.readlines()
        f.close()

        f_loc = data_loc+'gam_GB.dat'
        f = open(f_loc, 'r')
        gammas = f.readlines()
        f.close()

        ome_list = omegas[0].split()
        gam_list = gammas[0].split()

        i=0
        if no_split:
            qlk_dataset['x_var'] = ky_list
            qlk_dataset['omega'] = [float(value) for value in ome_list]
            qlk_dataset['gamma'] = [float(value) for value in gam_list]
            for i,value in enumerate(qlk_dataset['gamma']):
                if value == 0.0:
                    qlk_dataset['gamma'][i] = np.nan
                    qlk_dataset['omega'][i] = np.nan

        else:
            for j,value in enumerate(ome_list):
                if j>=1:
                    if np.sign(float(value)) != np.sign(float(ome_list[j-1])):
                        #print('sign change')
                        i+=1
                        qlk_dataset['x_var'].append([])
                        qlk_dataset['omega'].append([])
                        qlk_dataset['gamma'].append([])
                if float(gam_list[j]) !=0.0:
                    qlk_dataset['x_var'][i].append(ky_list[j])
                    qlk_dataset['omega'][i].append(float(value))
                    qlk_dataset['gamma'][i].append(float(gam_list[j]))
        
        list_to_array(qlk_dataset)

        return qlk_dataset