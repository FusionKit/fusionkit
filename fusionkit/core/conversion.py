'''
Set of routines to convert an ids to a specific code input
Contains:
  ids2tglf(ids,tglf) convert the ids inputs in the tglf object
  tglf_2_ids(tglf) Takes a tglf object and return an ids dict in IMAS format
'''


import sys
import numpy as np
import matplotlib.pyplot as plt

from fusionkit.extensions.tglf import *

# Import the file FS_param.py from the repository IMAS-GK 
# that can be found at https://gitlab.com/gkdb/imas-gk
# it can be obtained with the command "git clone https://gitlab.com/gkdb/imas-gk.git"
if not '/home/anass/codes/imas-gk/python/' in sys.path :
    sys.path.append('/home/anass/codes/imas-gk/python/')
from FS_param import *

from fusionkit.core.conversion_tools import *




# def ids_2_tglf(gkids,tglf,RMAJ_LOC=None,verbose=True):
#     """Convert an IMAS input object to a TGLF input object.
#     """
    
#     # Take a value for RMAJ_LOC, very useful because this value will be 
#     # used to make links between TGLF and IMAS normalisations 
#     if not RMAJ_LOC:
#         try:
#             RMAJ_LOC = tglf.input['RMAJ_LOC']
#         except:
#             RMAJ_LOC = 1.0
#     else:
#         try:
#             RMAJ_LOC=float(RMAJ_LOC)
#         except:
#             print('Wrong value for RMAJ_LOC, should be a number')
#             return
#     if verbose:
#         print('For the conversion from an IMAS input object to a TGLF input file,\
#               \nWe set: RMAJ_LOC = ',RMAJ_LOC)

#     # Check if the ids can be converted in a TGLF input
#     if gkids['model']['include_centrifugal_effects'] == 1:
#         if verbose:
#             print("Warning: Centrifugal effects can't be simulated in TGLF \
#                   \n ids['model']['include_centrifugal_effects'] should be 0")
# #        return
#     if (gkids['model']['collisions_pitch_only'] == 0 \
#         or gkids['model']['collisions_momentum_conservation'] == 1 \
#         or gkids['model']['collisions_energy_conservation'] == 1 \
#         or gkids['model']['collisions_finite_larmor_radius'] == 1 ):
#         if verbose:
#             print("Warning: ids['model'] parameters can't be simulated with TGLF \
#                   \nIn TGLF, the values of:\
#                       \n collisions_momentum_conservation \
#                       \n collisions_energy_conservation \
#                       \n collisions_finite_larmor_radius \
#                   \nAre false, and 'collisions_pitch_only' is true")
# #        return

#     # Value of dr/dx used in tglf
#     DRMINDX_LOC = 1.0
    
#     # Obtain the Miller parametrisation coefficients from the ids input values
#     miller = mxh_to_miller(gkids)
        
#     # compute TGLF_ref/IMAS_ref ratio of reference quantities for TGLf to IMAS conversion
#     # reference charge ratio
#     q_tglf = 1.6021746e-19      #at: July 29, 2022 See: https://gafusion.github.io/doc/tglf.html
#     q_imas = 1.602176634e-19    #Last uptade: July 29, 2022 ; IMAS: 3.22.0
#     q_rat = q_tglf/q_imas
#     # reference mass ratio
#     m_tglf = 3.345e-27          #at: Last July 29, 2022 See: https://gafusion.github.io/doc/tglf.html
#     m_imas = 3.3435837724e-27   #Last uptade: July 29, 2022 ; IMAS: 3.22.0
#     m_rat = m_tglf/m_imas
#     # reference temperature ratio
#     T_rat = 1
#     # reference length ratio
#     L_rat = 1/RMAJ_LOC 
#     # reference density ratio
#     n_rat = 1
#     # reference thermal velocity ratio
#     v_thrat = (1/np.sqrt(2))*(np.sqrt( T_rat/m_rat ))
#     # reference magnetic field ratio
#     B_rat = get_Brat_ids_tglf(miller,gkids['flux_surface']['q'])
#     # reference Larmor radius ratio
#     rho_rat = (m_rat*v_thrat)/(q_rat*B_rat) 
    
#     #Calculation of gthth
#     metric = get_metric(gkids,np.linspace(0,2*np.pi,501))
#     Gq = metric['Gq'][0]
#     gradr = metric['grad_r'][0]
#     #Calculation of KY
#     ky_max = 0.
#     for i in range(len(gkids['wavevector'] )):
#         if(gkids['wavevector'][i]['binormal_component_norm'] > ky_max ):
#             ky_max = gkids['wavevector'][i]['binormal_component_norm']
#             kx_max = gkids['wavevector'][i]['radial_component_norm']
#     #Fill input dict
#     inputs = {'GEOMETRY_FLAG' : 1,
#               'WRITE_WAVEFUNCTION_FLAG' : 1,
#               'RMIN_LOC' : miller['r']/L_rat ,
#               'RMAJ_LOC' : miller['Rmil']/L_rat ,
#               'ZMAJ_LOC' : miller['Zmil']/L_rat ,
#               'Q_LOC' : abs(gkids['flux_surface']['q']) ,
#               'Q_PRIME_LOC' :gkids['flux_surface']['magnetic_shear_r_minor']* \
#                   (gkids['flux_surface']['q'] / (miller['r']/L_rat) )**2  ,
                  
#               'P_PRIME_LOC' :gkids['flux_surface']['pressure_gradient_norm']* \
#                   (abs(gkids['flux_surface']['q']))/((miller['r']/L_rat)*(-8*np.pi)*(B_rat**2/L_rat)) ,
                  
#               'DRMINDX_LOC' : DRMINDX_LOC ,
#               'DRMAJDX_LOC' : miller['dRmildr']*DRMINDX_LOC ,
#               'DZMAJDX_LOC' : miller['dZmildr']*DRMINDX_LOC ,
#               'KAPPA_LOC' : miller['k'] ,
#               'S_KAPPA_LOC' : miller['sk'] ,
#               'DELTA_LOC' : miller['d'] ,
#               'S_DELTA_LOC' : miller['sd'] ,
#               'ZETA_LOC' : miller['z'] ,
#               'S_ZETA_LOC' : miller['sz'] ,
#               'KY' : ky_max*rho_rat/abs(Gq), 
#               'KX0_LOC' : (kx_max/ky_max)*(abs(Gq)/abs(gradr)),
#               'SIGN_IT' : gkids['flux_surface']['ip_sign']  ,
#               'SIGN_BT' : gkids['flux_surface']['b_field_tor_sign']  ,
#               'USE_BPER' : bool(gkids['model']['include_a_field_parallel']) ,
#               'USE_BPAR' : bool(gkids['model']['include_b_field_parallel']) ,
#               'USE_MHD_RULE' : False , #bool( not gkids['model']['include_full_curvature_drift']) ,
#               'BETAE' : gkids['species_all']['beta_reference']*1.5*n_rat*T_rat/( B_rat**2),
#               'KYGRID_MODEL' : 0 , 
#               'NKY' : int(max(len(gkids['wavevector']),ky_max+10))  ,
#               'NMODES' : 2,
#               'SAT_RULE' : 2 ,
#               'NS' :len(gkids['species']),
#               'UNITS':'CGYRO',
#               'NBASIS_MIN': 2,
#               'NBASIS_MAX': 6,
#               'FILTER': 2.0,
#               }
#     # Find electron/ion collision frequency
#     density_max = 0.
#     for i in range(len(gkids['species'])):
#         charge = gkids['species'][i]['charge_norm']
#         density = gkids['species'][i]['density_norm']
#         if charge < 0.:
#             index_electron = i
#         if (density > density_max) and (charge > 0.):
#             density_max = density
#             index_main_ion = i
#     # inputs['XNUE'] = gkids['collisions']['collisionality_norm'][index_electron][index_main_ion]*(L_rat/v_thrat)
#     # inputs['ZEFF'] = 1.
    
#     # Computation of ZEFF and XNUE
#     zeff = 0.
#     xnue = 0.
#     for i in range(len(gkids['species'])):
#         if gkids['species'][i]['charge_norm'] > 0. :
#             zeff = zeff + gkids['species'][i]['density_norm']*gkids['species'][i]['charge_norm']**2
#             xnue = xnue + gkids['collisions']['collisionality_norm'][index_electron][i]
#     inputs['ZEFF'] = zeff
    
#     inputs['XNUE'] = (1/zeff)*(L_rat/v_thrat)*xnue
        
    
#     # Electron species should be first
#     list_species_index = np.insert(np.delete(np.arange(len(gkids['species'])),index_electron),0,index_electron)
#     # Compute the species inputs        
#     for i in range(len(gkids['species'])):
#         charge_name = 'ZS_'+str(i+1)
#         charge = gkids['species'][list_species_index[i]]['charge_norm']/q_rat
#         inputs[charge_name] = np.round(charge)
        
#         mass_name = 'MASS_'+str(i+1)
#         mass = gkids['species'][list_species_index[i]]['mass_norm']/m_rat
#         inputs[mass_name] = mass
        
#         density_name = 'AS_'+str(i+1)
#         density = gkids['species'][list_species_index[i]]['density_norm']/n_rat
#         inputs[density_name] = density
        
#         rln_name = 'RLNS_'+str(i+1)
#         rln = gkids['species'][list_species_index[i]]['density_log_gradient_norm']*L_rat
#         inputs[rln_name] = rln
        
#         temperature_name = 'TAUS_'+str(i+1)
#         temperature = gkids['species'][list_species_index[i]]['temperature_norm']/T_rat
#         inputs[temperature_name] = temperature
        
#         rlt_name = 'RLTS_'+str(i+1)
#         rlt = gkids['species'][list_species_index[i]]['temperature_log_gradient_norm']*L_rat
#         inputs[rlt_name] = rlt
        
#         Vpar_name = 'VPAR_'+str(i+1)
#         Vpar = gkids['flux_surface']['ip_sign']*gkids['species_all']['velocity_tor_norm']*L_rat/v_thrat
#         inputs[Vpar_name] = Vpar
        
#         VparS_name = 'VPAR_SHEAR_'+str(i+1)
#         VparS = gkids['flux_surface']['ip_sign']*gkids['species'][list_species_index[i]]['velocity_tor_gradient_norm']*(L_rat)/v_thrat
#         inputs[VparS_name] = VparS
        
        
    
    
    
#     tglf.input = {}
#     tglf.input = inputs
#     return


# def tglf_2_ids(tglf):
#     """Convert a TGLF object to IMAS gyrokinetic IDS/GKDB format.
#     """
#     tglf._tglf_to_ids()
#     return tglf.ids


###############################################################################



def idspy_2_tglf(gkids, tglf, RMAJ_LOC=None, verbose=True):
    """Convert an IMAS input object to a TGLF input object."""
    
    # Take a value for RMAJ_LOC, very useful because this value will be 
    # used to make links between TGLF and IMAS normalisations 
    if not RMAJ_LOC:
        try:
            RMAJ_LOC = tglf.input.RMAJ_LOC
        except AttributeError:
            RMAJ_LOC = 1.0
    else:
        try:
            RMAJ_LOC = float(RMAJ_LOC)
        except ValueError:
            print('Wrong value for RMAJ_LOC, should be a number')
            return
    
    if verbose:
        print('For the conversion from an IMAS input object to a TGLF input file,\nWe set: RMAJ_LOC =', RMAJ_LOC)

    # Check if the ids can be converted in a TGLF input
    if gkids.model.include_centrifugal_effects == 1:
        if verbose:
            print("Warning: Centrifugal effects can't be simulated in TGLF\nids.model.include_centrifugal_effects should be 0")
#         return
    if (
        gkids.model.collisions_pitch_only == 0
        or gkids.model.collisions_momentum_conservation == 1
        or gkids.model.collisions_energy_conservation == 1
        or gkids.model.collisions_finite_larmor_radius == 1
    ):
        if verbose:
            print(
                "Warning: ids.model parameters can't be simulated with TGLF\n"
                "In TGLF, the values of:\n"
                "collisions_momentum_conservation\n"
                "collisions_energy_conservation\n"
                "collisions_finite_larmor_radius\n"
                "Are false, and 'collisions_pitch_only' is true"
            )
#         return

    # Value of dr/dx used in tglf
    DRMINDX_LOC = 1.0
    
    # Obtain the Miller parametrisation coefficients from the ids input values
    miller = mxh_to_miller_idspy(gkids)

    # compute TGLF_ref/IMAS_ref ratio of reference quantities for TGLf to IMAS conversion
    # reference charge ratio
    q_tglf = 1.6021746e-19      # at: July 29, 2022 See: https://gafusion.github.io/doc/tglf.html
    q_imas = 1.602176634e-19    # Last update: July 29, 2022 ; IMAS: 3.22.0
    q_rat = q_tglf / q_imas
    # reference mass ratio
    m_tglf = 3.345e-27          # at: Last July 29, 2022 See: https://gafusion.github.io/doc/tglf.html
    m_imas = 3.3435837724e-27   # Last update: July 29, 2022 ; IMAS: 3.22.0
    m_rat = m_tglf / m_imas
    # reference temperature ratio
    T_rat = 1
    # reference length ratio
    L_rat = 1 / RMAJ_LOC
    # reference density ratio
    n_rat = 1
    # reference thermal velocity ratio
    v_thrat = (1 / np.sqrt(2)) * (np.sqrt(T_rat / m_rat))
    # reference magnetic field ratio
    B_rat = get_Brat_ids_tglf(miller, gkids.flux_surface.q)
    # reference Larmor radius ratio
    rho_rat = (m_rat * v_thrat) / (q_rat * B_rat)

    # Calculation of gthth
    metric = get_metric_idspy(gkids, np.linspace(0, 2 * np.pi, 501))
    Gq = metric['Gq'][0]
    gradr = metric['grad_r'][0]
    # Calculation of KY
    ky_max = 0.
    kx_max = 0.
    for i in range(len(gkids.wavevector)):
        if gkids.wavevector[i].binormal_component_norm > ky_max:
            ky_max = gkids.wavevector[i].binormal_component_norm
            kx_max = gkids.wavevector[i].radial_component_norm
    # Fill input dict
    inputs = {
        'GEOMETRY_FLAG': 1,
        'WRITE_WAVEFUNCTION_FLAG': 1,
        'RMIN_LOC': miller['r'] / L_rat,
        'RMAJ_LOC': miller['Rmil'] / L_rat,
        'ZMAJ_LOC': miller['Zmil'] / L_rat,
        'Q_LOC': abs(gkids.flux_surface.q),
        'Q_PRIME_LOC': gkids.flux_surface.magnetic_shear_r_minor
        * (gkids.flux_surface.q / (miller['r'] / L_rat)) ** 2,
        'P_PRIME_LOC': gkids.flux_surface.pressure_gradient_norm
        * (abs(gkids.flux_surface.q))
        / ((miller['r'] / L_rat) * (-8 * np.pi) * (B_rat ** 2 / L_rat)),
        'DRMINDX_LOC': DRMINDX_LOC,
        'DRMAJDX_LOC': miller['dRmildr'] * DRMINDX_LOC,
        'DZMAJDX_LOC': miller['dZmildr'] * DRMINDX_LOC,
        'KAPPA_LOC': miller['k'],
        'S_KAPPA_LOC': miller['sk'],
        'DELTA_LOC': miller['d'],
        'S_DELTA_LOC': miller['sd'],
        'ZETA_LOC': miller['z'],
        'S_ZETA_LOC': miller['sz'],
        'KY': ky_max * rho_rat / abs(Gq),
        'KX0_LOC': (kx_max / ky_max) * (abs(Gq) / abs(gradr)),
        'SIGN_IT': gkids.flux_surface.ip_sign,
        'SIGN_BT': gkids.flux_surface.b_field_tor_sign,
        'USE_BPER': bool(gkids.model.include_a_field_parallel),
        'USE_BPAR': False, # NOT in use in TGLF, The BPAR flutter is not well resolved by TGLF 
                           # (not enough moments) and does not reproduce GYRO linear results for NSTX-U
        'USE_MHD_RULE': False,  # bool( not gkids.model.include_full_curvature_drift),
        'BETAE': gkids.species_all.beta_reference * 1.5 * n_rat * T_rat / (B_rat ** 2),
        'KYGRID_MODEL': 0,
        'NKY': int(max(len(gkids.wavevector), ky_max + 10)), # Attention quand le ids est trop long avec plusieurs ky et kx !
        'NMODES': 2, 
        'SAT_RULE': 2,
        'NS': len(gkids.species),
        'UNITS': 'CGYRO',
        'NBASIS_MIN': 4,
        'NBASIS_MAX': 6,
        'FILTER': 2.0,
    }
    
    # Find electron/ion collision frequency
    density_max = 0.
    for i in range(len(gkids.species)):
        charge = gkids.species[i].charge_norm
        density = gkids.species[i].density_norm
        if charge < 0.:
            index_electron = i
        if (density > density_max) and (charge > 0.):
            density_max = density
            index_main_ion = i
            
    # Computation of ZEFF and XNUE
    zeff = 0.
    xnue = 0.
    for i in range(len(gkids.species)):
        if gkids.species[i].charge_norm > 0.:
            zeff = zeff + gkids.species[i].density_norm * gkids.species[i].charge_norm ** 2
            xnue = xnue + gkids.collisions.collisionality_norm[index_electron][i]
    inputs['ZEFF'] = zeff
    inputs['XNUE'] = (1 / zeff) * (L_rat / v_thrat) * xnue

    # Electron species should be first
    list_species_index = np.insert(np.delete(np.arange(len(gkids.species)), index_electron), 0, index_electron)
    # Compute the species inputs
    for i in range(len(gkids.species)):
        charge_name = 'ZS_' + str(i + 1)
        charge = gkids.species[list_species_index[i]].charge_norm / q_rat
        inputs[charge_name] = np.round(charge)

        mass_name = 'MASS_' + str(i + 1)
        mass = gkids.species[list_species_index[i]].mass_norm / m_rat
        inputs[mass_name] = mass

        density_name = 'AS_' + str(i + 1)
        density = gkids.species[list_species_index[i]].density_norm / n_rat
        inputs[density_name] = density

        rln_name = 'RLNS_' + str(i + 1)
        rln = gkids.species[list_species_index[i]].density_log_gradient_norm * L_rat
        inputs[rln_name] = rln

        temperature_name = 'TAUS_' + str(i + 1)
        temperature = gkids.species[list_species_index[i]].temperature_norm / T_rat
        inputs[temperature_name] = temperature

        rlt_name = 'RLTS_' + str(i + 1)
        rlt = gkids.species[list_species_index[i]].temperature_log_gradient_norm * L_rat
        inputs[rlt_name] = rlt

        Vpar_name = 'VPAR_' + str(i + 1)
        Vpar = gkids.flux_surface.ip_sign * gkids.species_all.velocity_tor_norm * L_rat / v_thrat
        inputs[Vpar_name] = Vpar

        VparS_name = 'VPAR_SHEAR_' + str(i + 1)
        VparS = gkids.flux_surface.ip_sign * gkids.species[list_species_index[i]].velocity_tor_gradient_norm * (
                L_rat) / v_thrat
        inputs[VparS_name] = VparS

    tglf.input = {}
    tglf.input = inputs
    return

def tglf_2_idspy(tglf):
    """Convert a TGLF object to IMAS gyrokinetic IDS/GKDB format.
    """
    tglf._tglf_to_idspy()
    return tglf.ids








