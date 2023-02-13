import numpy as np
from scipy.integrate import cumtrapz,trapz
from scipy.interpolate import interp1d
 
import matplotlib.pyplot as plt

def get_metric(ids,th_in,Nth=501):
  """ Compute the magnetic equilibrium and associated metric for a GKDB point
  
  Inputs:
    ids           dict with the content of a GKDB point, as obtained from loading a GKDB JSON file
    th_in        numpy array with the poloidal angles at which the metric terms will be computed (a finer theta grid is used internally)
                  If empty, uses the internal finer theta grids
    Nth           number of points for the fine theta grid used internally for integral computations (defautlt 501)
                  
  Outputs:    Magnetic equilibrium quantites, all normalised following the GKDB conventions
    th_out        poloidal angles at which the metric terms were computed
    amin          flux surface minor radius
    R             flux surface horizontal coordinates
    Z             flux surface vertical coordinates
    J_r           (r,theta,phi) Jacobian J_r=1/(grad_r x grad_theta . grad_phi)
    dVdr          radial derivative of the plasma volume
    grad_r        |grad_r|
    dpsidr        radial derivative of the poloidal magnetic flux (positive if s_j=1)
    bt            toroidal magnetic field (normalised to Bref, positive if s_b=1)
    bp            poloidal magnetic field (normalised to Bref, positive if s_j=1)
    grr           radial-radial metric tensor to compute kperp from kr* and kth*
    grth          radial-binormal metric tensor to compute kperp from kr* and kth*
    gthth         radial-binormal metric tensor to compute kperp from kr* and kth*

  See section 7.1 of the GKDB manual at https://gitlab.com/gkdb/gkdb/raw/master/doc/general/IOGKDB.pdf
  """
  # convert th_out into an ndarray if needed
  if type(th_in) is np.ndarray:
    th_out=th_in.copy()
  else:
    th_out=np.asarray(th_in)

  # retrieve the relevant information from the ids dict
  r=np.asarray(ids['flux_surface']['r_minor_norm'])
  k=np.asarray(ids['flux_surface']['elongation'])
  dkdr=np.asarray(ids['flux_surface']['delongation_dr_minor_norm'])
  dR0dr=np.asarray(ids['flux_surface']['dgeometric_axis_r_dr_minor'])
  dZ0dr=np.asarray(ids['flux_surface']['dgeometric_axis_z_dr_minor'])
  c=np.asarray(ids['flux_surface']['shape_coefficients_c'])
  s=np.asarray(ids['flux_surface']['shape_coefficients_s'])
  dcdr=np.asarray(ids['flux_surface']['dc_dr_minor_norm'])
  dsdr=np.asarray(ids['flux_surface']['ds_dr_minor_norm'])
  sb=np.asarray(ids['flux_surface']['b_field_tor_sign'])
  sj=np.asarray(ids['flux_surface']['ip_sign'])
  q=np.asarray(ids['flux_surface']['q'])
  shat=np.asarray(ids['flux_surface']['magnetic_shear_r_minor'])
  pprime=np.asarray(ids['flux_surface']['pressure_gradient_norm'])

      
  # define the fine theta grid on which the integrals are performed
  th=np.linspace(0,2*np.pi,Nth)

  
  # R, Z and their radial/poloidal derivatives
  Nsh=np.reshape(np.arange(len(c)),(1,-1))
  TH=np.reshape(th,(-1,1))
  THr = th + np.sum(np.reshape(c,(1,-1))*np.cos(Nsh*TH) + np.reshape(s,(1,-1))*np.sin(Nsh*TH),axis=1)
  dTHrdr = np.sum(np.reshape(dcdr,(1,-1))*np.cos(Nsh*TH) + np.reshape(dsdr,(1,-1))*np.sin(Nsh*TH),axis=1)
  dTHrdth = 1 + np.sum(Nsh*(np.reshape(s,(1,-1))*np.cos(Nsh*TH) - np.reshape(c,(1,-1))*np.sin(Nsh*TH)),axis=1)
  d2THrdth2 = -1*np.sum((Nsh**2)*(np.reshape(c,(1,-1))*np.cos(Nsh*TH) + np.reshape(s,(1,-1))*np.sin(Nsh*TH)),axis=1)
  R0 = 1.
  Z0 = 0.
  R = R0 + r*np.cos(THr)
  Z = Z0 + k*r*np.sin(th)
  dRdr = dR0dr + np.cos(THr) - r*np.sin(THr)*dTHrdr
  dZdr = dZ0dr + k*np.sin(th) + dkdr*r*np.sin(th)
  dRdth = -1.*r*np.sin(THr)*dTHrdth
  dZdth = k*r*np.cos(th)
  d2Rdth2 = -1*r*(np.cos(THr)*(dTHrdth**2) + np.sin(THr)*d2THrdth2)
  d2Zdth2 = -1*k*r*np.sin(th)
  
  
  # Jacobian, dldth and grad_r
  J_r = -R*(dRdr*dZdth - dRdth*dZdr) 
  dldth = np.sqrt((dRdth)**2+(dZdth)**2)
  grad_r = R/J_r*dldth

  # dpsidr and dVdr
  dpsidr=sj/abs(q)*trapz(J_r/R,th)
  dVdr=2*np.pi*trapz(J_r,th)

  # bt and bp
  bt=sb/R
  bp=grad_r/(2*np.pi*R)*dpsidr

  # intermediate quantities for the calculation of kperp
  cos_u=-dZdth/dldth
  r_c=-dldth**3/(dRdth*d2Zdth2-dZdth*d2Rdth2)
  E1or=2*cumtrapz(J_r/R**2*bt/bp*(1/r_c-1/R*cos_u),th,initial=0)
  E2=cumtrapz(J_r/R**2*(bp**2+bt**2)/bp**2,th,initial=0)
  bE3=0.5*cumtrapz(dldth/R*bt/bp**3*pprime,th,initial=0)
  fstar=(2*np.pi*q*shat/r-E1or[-1]+bE3[-1])/E2[-1]

  # interpolate on the output grid if needed
  if not th_out.any():
    th_out=th
  else:
    th_out_2pi = th_out % (2*np.pi)
    n_2pi = th_out // (2*np.pi)
    # deal first with the 2pi periodic quantities
    f=interp1d(th,R)
    R=f(th_out_2pi)
    f=interp1d(th,Z)
    Z=f(th_out_2pi)
    f=interp1d(th,J_r)
    J_r=f(th_out_2pi)
    f=interp1d(th,grad_r)
    grad_r=f(th_out_2pi)
    f=interp1d(th,bt)
    bt=f(th_out_2pi)
    f=interp1d(th,bp)
    bp=f(th_out_2pi)
    # then deal with the cumulative integrals 
    f=interp1d(th,E1or)
    E1or=f(th_out_2pi)+f(2*np.pi)*n_2pi
    f=interp1d(th,E2)
    E2=f(th_out_2pi)+f(2*np.pi)*n_2pi
    f=interp1d(th,bE3)
    bE3=f(th_out_2pi)+f(2*np.pi)*n_2pi

  # finally compute Gq, Theta and the metric tensors for the calculation of kperp
  Gq=r*np.sqrt(bp**2+bt**2)/(q*R*bp)
  Theta=R*bp*grad_r*(E1or+fstar*E2-bE3)/np.sqrt(bp**2+bt**2)
  grr=(grad_r/grad_r[0])**2
  grth=grad_r/grad_r[0]*Gq*Theta
  gthth=Gq**2*(1+Theta**2)

  metric = {'th_out':th_out,
            'R':R,
            'Z':Z,
            'J_r':J_r,
            'dVdr':dVdr,
            'grad_r':grad_r,
            'dpsidr':dpsidr,
            'bt':bt,
            'bp':bp,
            'Gq':Gq,
            'grr':grr,
            'grth':grth,
            'gthth':gthth}

  return metric


