# Set of routines used in conversion.py to convert an ids to a specific code input
# Contains:
#   mxh_to_miller(ids) Take the ids mxh coeffs and return miller coeffs parameters
#   get_metric_ids_tglf(miller,q) Use miller coeffs (and safety factor) to return tglf Bunit and tglf/imas Brat

import sys
import numpy as np
import matplotlib.pyplot as plt

if not '/home/anass/codes/gkw-pythontools/' in sys.path :
    sys.path.append('/home/anass/codes/gkw-pythontools/')
from FS_param import *





def mxh_to_miller(ids):
     #Get the values of r, R and Z (R0 is supposed to be equal to 1 and Z0 to 0, IMAS normalisations)
     r, R, Z = mxh2rz(ids['flux_surface']['r_minor_norm'], 1 , 0 ,
                      ids['flux_surface']['elongation'],
                      ids['flux_surface']['shape_coefficients_c'],
                      ids['flux_surface']['shape_coefficients_s'],
                      ids['flux_surface']['dgeometric_axis_r_dr_minor'],
                      ids['flux_surface']['dgeometric_axis_z_dr_minor'],
                      ids['flux_surface']['delongation_dr_minor_norm'],
                      ids['flux_surface']['dc_dr_minor_norm'],
                      ids['flux_surface']['ds_dr_minor_norm'],
                      code='imas',dr_frac=0.01,Nth=500)
     #Get the miller coeffs parameters       
     k,d,z,sk,sd,sz,Rmil,Zmil,r,dRmildr,dZmildr = rz2miller(R,Z,code='tglf',doplots=False)
     return { 'k':k[1],'d':d[1], 'z':z[1], 'sk':sk[1], 'sd':sd[1], 'sz':sz[1], 'Rmil':Rmil[1], 
             'Zmil':Zmil[1], 'r':r[1], 'dRmildr':dRmildr[1], 'dZmildr':dZmildr[1] }

def get_metric_ids_tglf(miller,q):
    Nth=500
    k,d,z,sk,sd,sz,R0,Z0,r0,dRmildr,dZmildr = miller.values()
    drmildr = 1
    sd = sd/np.sqrt(1.0-d**2)
    sj = -1
    
    # define the fine theta grid on which the integrals are performed
    th=np.linspace(0,2*np.pi,Nth)
    
    arg_r = th + np.arcsin(d)*np.sin(th)
    darg_r = 1.0 + np.arcsin(d)*np.cos(th)
    arg_z = th + z*np.sin(2.0*th)
    darg_z = 1.0 + z*2.0*np.cos(2.0*th)
    
    R = R0 + r0*np.cos(arg_r)
    Z = Z0 + r0*k*np.sin(arg_z)
    
    dRdr = dRmildr + drmildr*np.cos(arg_r) - np.sin(arg_r)*sd*np.sin(th)
    dZdr = dZmildr + k*np.sin(arg_z)*(drmildr + sk) + k*np.cos(arg_z)*sz*np.sin(2.0*th)
    dRdth = -1*r0*np.sin(arg_r)*darg_r
    dZdth = k*r0*np.cos(arg_z)*darg_z
    r = 0.5*(max(R)-min(R))

    J_r = -R*(dRdr*dZdth - dRdth*dZdr)
    dldth = np.sqrt((dRdth)**2+(dZdth)**2)
    grad_r = R/J_r*dldth
    dpsidr = (sj/(q*2.0*np.pi)*np.trapz(J_r/R,th))
    B_unit = (q/r)*dpsidr
    
    B_rat = R0/(2*np.pi*r)*np.trapz(dldth/(R*abs(grad_r)),th)

    return {'B_unit':B_unit, 'B_rat':B_rat} 
