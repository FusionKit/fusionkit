'''
The QuaLiKiz class
'''

import numpy as np

## QuaLiKiz
class QLK:
    def __init__(self):
        self.metadata = {}
        self.input = {}
        self.output = {}
    
    # I/O functions
    def write_input(rho=None,dataset=None,output_loc=None,fname=None,imp_composite=False):
        rho_idx = np.abs(dataset['rho']-rho).argmin()

        ne = dataset['ne'][rho_idx]*1e-19
        RLne = dataset['RLne'][rho_idx]
        ni = dataset['ni'][rho_idx]*1e-19
        RLni = dataset['RLni'][rho_idx]
        A_i = 2
        Z_i = 1

        n_LZ = dataset['n_LZ'][rho_idx]*1e-19
        RLn_LZ = dataset['RLn_LZ'][rho_idx]
        A_LZ = 9
        Z_L = dataset['Z_L']

        if imp_composite:
            n_comp = dataset['n_comp'][rho_idx]*1e-19
            RLn_comp = dataset['RLn_comp'][rho_idx]
            A_comp = dataset['A_comp'][rho_idx]
            Z_comp = dataset['Z_comp'][rho_idx]

        Te = dataset['Te'][rho_idx]*1e-3
        RLTe = dataset['RLTe'][rho_idx]
        Ti = dataset['Ti'][rho_idx]*1e-3
        RLTi = dataset['RLTi'][rho_idx]

        q = dataset['q'][rho_idx]
        s = dataset['s'][rho_idx]
        B0 = dataset['B0']

        alpha = dataset['alpha'][rho_idx]
        a = dataset['a']
        x = dataset['x'][rho_idx]
        Ro = dataset['Ro'][rho_idx]
        R0 = dataset['R0']

        Mach_tor = 0
        Au_tor = 0
        Mach_par = 0
        Au_par = 0
        gamma_E = 0

        ## QuaLiKiz input preparation
        from qualikiz_tools.qualikiz_io.inputfiles import QuaLiKizXpoint, Electron, Ion, IonList
        from qualikiz_tools.qualikiz_io.inputfiles import QuaLiKizPlan

        kthetarhos = list(np.linspace(0.1,0.8,8))
        elec = Electron(T=Te,n=ne,At=RLTe,An=RLne,type=1,anis=1, danisdr=0)
        ion0 = Ion(T=Ti,n=ni,At=RLTi,An=RLni,A=A_i,Z=Z_i,type=1,anis=1,danisdr=0)
        ion1 = Ion(T=Ti,n=n_LZ,At=RLTi,An=RLn_LZ,A=A_LZ,Z=Z_L,type=1,anis=1,danisdr=0)
        if imp_composite:
            ion2 = Ion(T=Ti,n=n_comp,At=RLTi,An=RLn_comp,A=A_comp,Z=Z_comp,type=1,anis=1,danisdr=0)
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
            "rho": rho,
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