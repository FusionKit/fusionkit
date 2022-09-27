# Set of routines to deal with flux surfaces parametrisation
# Contains:
#   fourier2rz    (R,Z) flux surface description from Fourier parametrisation
#   mxh2fourier   (R,Z) flux surface description from Miller eXtended Harmonic
#   miller2rz     (R,Z) flux surface description from Miller parametrisation
#   rz2fourier    Fourier parametrisation from (R,Z) flux surface description
#   rz2mxh        Miller eXtended Harmonic from (R,Z) flux surface description
#

import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import splrep
from scipy.interpolate import splev
import matplotlib.pyplot as plt
import pdb
def fourier2rz(c,s,dcdr,dsdr,code,dr_frac=0.01,R0=1,Z0=0,Nth=500):
  """ return r,R,Z
      Compute the (R,Z) description of a flux surface and two adjacent neighbours from its Fourier parametrisation.
      The flux surfaces are computed as:
          R(r,theta) = R0 + a(r,theta)*cos(theta)
          Z(r,theta) = Z0 + sign_theta*a(r,theta)*sin(theta)
      with
          a(r,theta) = sum_{n=0 to N} [ c_n(r)*cos(n*theta) + s_n*sin(n*theta) ]

      The definition of theta and r are:
          tan(theta) = sign_theta * (Z-Z0)/(R-R0)
             with sign_theta code dependent
          r=(max(R)-min(R))/2

      This parametrisation is close to that introduced in Candy PPCF 51, 105009 (2009)

      Inputs:
        c,s            Fourier parametrisation of the flux surface 
        dcdr, dsdr     Radial derivative of the coefficients above 
        code           Code convention: 'gkw' or 'imas'
                          gkw:  theta=0 at Z=Z0 on the LFS and increasing counter-clockwise, sign_theta=+1
                          imas: theta=0 at Z=Z0 on the LFS and increasing clockwise (GK IDS convention), sign_theta=-1
        R0, Z0         reference point, optional (default: R0=1, Z0=0)
        dr_frac        neighbouring flux surfaces will be computed at r0*(1-dr_frac) and r0*(1+dr_frec)
                       optional (default dr_frac=0.01)
        Nth            number of points to discretize the flux surfaces
                       optional (default: Nth=500)
      Outputs:
        r              Minor radius of the three FS, given at r0*[1-dr_frac, 1, 1 + dr_frac]
        R,Z            Flux surface description in cylindrical coordinates (no double points)
                       
      Note that c, s, R0, Z0 are assumed to all be given with the same normalisation/units, which 
      can be arbitrary. 
      The (R,Z) and r values will be given with the same normalisation/units as used in input.
  """  
  c=np.asarray(c)
  s=np.asarray(s)
  dcdr=np.asarray(dcdr)
  dsdr=np.asarray(dsdr)
  Nsh = c.size
  if not np.all(np.array([s.size,dcdr.size,dsdr.size])==Nsh):
   print("Fourier coefficient arrays should all have the same size")
   return

  if code=='gkw':
   sign_theta=1
  elif code=='imas':
   sign_theta=-1
  else:
   print("Unknown code convention. Available: 'gkw' or 'imas'")
   return

  Nr=3    # total number of FS: 1 + 2 neighbours  
  th_grid=np.linspace(0,2*np.pi,num=Nth,endpoint=False)
  n=np.arange(Nsh)
  
  THETA=np.tile(th_grid,(Nsh,1))
  N=np.transpose(np.tile(n,(Nth,1)))
  C=np.transpose(np.tile(c,(Nth,1)))
  S=np.transpose(np.tile(s,(Nth,1)))
  DCDR=np.transpose(np.tile(dcdr,(Nth,1)))
  DSDR=np.transpose(np.tile(dsdr,(Nth,1)))

  # distance to the reference point, as a function of theta
  aN = np.sum(C*np.cos(N*THETA)+S*np.sin(N*THETA),axis=0)
  daNdr = np.sum(DCDR*np.cos(N*THETA)+DSDR*np.sin(N*THETA),axis=0)
  
  # minor radius r0
  Rdum=aN*np.cos(th_grid)
  r0 = (np.max(Rdum)-np.min(Rdum))/2
  
  # neighbouring flux surfaces
  dr=r0*dr_frac
  aNp = aN + daNdr*dr
  aNm = aN - daNdr*dr

  R = R0 + np.stack((aNm,aN,aNp))*np.cos(np.tile(th_grid,(3,1)))
  Z = Z0 + sign_theta*np.stack((aNm,aN,aNp))*np.sin(np.tile(th_grid,(3,1)))
  r = r0*np.array([1-dr_frac,1,1+dr_frac])

  return r,R,Z

def mxh2rz(r0,R0,Z0,k,c,s,dR0dr,dZ0dr,dkdr,dcdr,dsdr,code='gkw',dr_frac=0.01,Nth=500):
  """ return r,R,Z
      Compute the (R,Z) description of a flux surface and two adjacent neighbours from 
      its MXH parametrisation, see Arbon PPCF 63, 012001 (2021).
      The flux surfaces are computed as:
          R(r,theta) = R0 + r*cos(theta_R)
          Z(r,theta) = Z0 + sign_theta*r*k*sin(theta)
      with
          theta_R(r,theta) = theta + sum_{n=0 to Nsh-1} [ c_n(r)*cos(n*theta) + s_n*sin(n*theta) ]

      The definition of theta and r are:
          tan(theta) = sign_theta * (Z-Z0)/(R-R0)
             with sign_theta code dependent (but assuming theta=0 at the LFS)
          r=(max(R)-min(R))/2

      Inputs:
        r0             minor radius
        R0, Z0         flux surface center
        k              elongation
        c,s            Miller harmonic coefficients
        dR0dr          radial derivative of R0
        dZ0dr          radial derivative of Z0
        dkdr           radial derivative of k (exact definition code dependent)
        dcdr,dsdr      radial derivative of c and s
        code           Code convention (for the sign of theta): 'gkw' or 'imas'
                          gkw:  theta=0 at Z=Z0 on the LFS and increasing counter-clockwise, sign_theta=+1
                          imas: theta=0 at Z=Z0 on the LFS and increasing clockwise (GK IDS convention), sign_theta=-1
        dr_frac        neighbouring flux surfaces will be computed at r0*(1-dr_frac) and r0*(1+dr_frec)
                       optional (default dr_frac=0.01)
        Nth            number of points to discretize the flux surfaces
                       optional (default: Nth=500)
      Outputs:
        r              Minor radius of the three FS, given at r0*[1-dr_frac, 1, 1 + dr_frac]
        R,Z            Flux surface description in cylindrical coordinates (no double points)
                       
      Note that r0,R0,Z0,k,c,s,dR0dr,dZ0dr,dkdr,dcdr,dsdr  are assumed to all be given with the same normalisation/units, 
      which can be arbitrary. 
      The (R,Z) and r values will be given with the same normalisation/units as used in input.
  """  
  c=np.asarray(c)
  s=np.asarray(s)
  dcdr=np.asarray(dcdr)
  dsdr=np.asarray(dsdr)
  Nsh = c.size
  if not np.all(np.array([s.size,dcdr.size,dsdr.size])==Nsh):
   print("MXH coefficient arrays should all have the same size")
   return

  if code=='gkw':
   sign_theta=1
  elif code=='imas':
   sign_theta=-1
  else:
   print("Unknown code convention. Available: 'gkw' or 'imas'")
   return

  Nr=3    # total number of FS: 1 + 2 neighbours  
  th_grid=np.linspace(0,2*np.pi,num=Nth,endpoint=False)
  dr=r0*np.array([-dr_frac, 0, dr_frac])
  r=r0+dr
  n=np.arange(Nsh)
  THETA=np.tile(th_grid,(Nsh,1))
  N=np.transpose(np.tile(n,(Nth,1)))
  C=np.transpose(np.tile(c,(Nth,1)))
  S=np.transpose(np.tile(s,(Nth,1)))
  DCDR=np.transpose(np.tile(dcdr,(Nth,1)))
  DSDR=np.transpose(np.tile(dsdr,(Nth,1)))

  # angle theta_R, as a function of theta
  thR = th_grid + np.sum(C*np.cos(N*THETA)+S*np.sin(N*THETA),axis=0)
  dthRdr = np.sum(DCDR*np.cos(N*THETA)+DSDR*np.sin(N*THETA),axis=0)

  # compute shape description at all radii
  r_all=np.transpose(np.tile(r,(Nth,1)))
  th_all=np.tile(th_grid,(Nr,1))
  thR_all=np.tile(thR,(Nr,1))+np.tile(dthRdr,(Nr,1))*np.transpose(np.tile(dr,(Nth,1)))
  R0_all=np.transpose(np.tile(R0+dR0dr*dr,(Nth,1)))
  Z0_all=np.transpose(np.tile(Z0+dZ0dr*dr,(Nth,1)))
  k_all=np.transpose(np.tile(k+dkdr*dr,(Nth,1)))

  R = R0_all + r_all*np.cos(thR_all)
  Z = Z0_all + sign_theta*r_all*k_all*np.sin(th_all)

  return r,R,Z


def miller2rz(r0,Rmil,Zmil,k,d,z,dRmildr,dZmildr,sk,sd,sz,code='gkw',dr_frac=0.01,Nth=500):
  """ return r,R,Z
      Compute the (R,Z) description of a flux surface and two adjacent neighbours from its Miller parametrisation.
      The flux surfaces are computed as:
          R(r,theta) = Rmil + r*cos(theta + arcsin(d*sin(theta)))
          Z(r,theta) = Zmil + r*k*sin(theta+z*sin(2*theta))

      The definition of theta and r are:
          tan(theta) = sign_theta * (Z-Z0)/(R-R0)
             with sign_theta code dependent
          r=(max(R)-min(R))/2

      Inputs:
        r0             minor radius
        Rmil, Zmil     flux surface center
        k,d,z          elongation, triangularity, squareness
        dRmildr        radial derivative of Rmil
        dZmildr        radial derivative of Zmil
        sk,sd,sz       radial derivatives of k, d, z (exact definition code dependent)
        code           Code convention: 'gkw' (default)
                          gkw:  see GKW manual
        dr_frac        neighbouring flux surfaces will be computed at r0*(1-dr_frac) and r0*(1+dr_frec)
                       optional (default dr_frac=0.01)
        Nth            number of points to discretize the flux surfaces
                       optional (default: Nth=500)
      Outputs:
        r              Minor radius of the three FS, given at r0*[1-dr_frac, 1, 1 + dr_frac]
        R,Z            Flux surface description in cylindrical coordinates (no double points)
                       
      Note that r0,Rmil,Zmil,k,d,z,dRmildr,dZmildr,sk,sd,sz are assumed to all be given with the same normalisation/units, 
      which can be arbitrary. 
      The (R,Z) and r values will be given with the same normalisation/units as used in input.
  """  
  if code=='gkw':
   sign_theta=1
  elif code=='tglf':
   sign_theta=1
   sd = sd/np.sqrt(1.0-d**2)
  else:
   print("Unknown code convention. Available: 'gkw' or 'tglf'")
   return

  Nr=3    # total number of FS: 1 + 2 neighbours  
  th_grid=np.linspace(0,2*np.pi,num=Nth,endpoint=False)
  dr=r0*np.array([-dr_frac, 0, dr_frac])
  r=r0+dr

  th_all=np.tile(th_grid,(Nr,1))
  r_all=np.transpose(np.tile(r,(Nth,1)))
  Rmil_all=np.transpose(np.tile(Rmil+dRmildr*dr,(Nth,1)))
  Zmil_all=np.transpose(np.tile(Zmil+dZmildr*dr,(Nth,1)))
  k_all=np.transpose(np.tile(k+sk*k/r*dr,(Nth,1)))
  d_all=np.transpose(np.tile(d+sd*np.sqrt(1-d**2)/r*dr,(Nth,1)))
  z_all=np.transpose(np.tile(z+sz/r*dr,(Nth,1)))

  R = Rmil_all + r_all*np.cos(th_all + np.arcsin(d_all)*np.sin(th_all))
  Z = Zmil_all + r_all*k_all*np.sin(th_all+z_all*np.sin(2*th_all))

  return r,R,Z


def rz2fourier(R,Z,r0,code,Nsh=10,doplots=True):
  """ return c,s,dcdr,dsdr,R0,Z0,R_out,Z_out,err_out
      Compute the Fourier parametrisation of flux surfaces at a given radial location.
      The radial coordinate is defined as r=(max(R)-min(R))/2
      The reference point R0,Z0 is computed as:
          R0 = (max(R(r0)) + min(R(r0)))/2
          Z0 = (max(Z(r0)) + min(Z(r0)))/2
      with interpolation since the R,Z description of the flux surface can be coarse.
      The distance to the reference point is then:
          a = sqrt((R-R0)**2+(Z-Z0)**2)
      From which the Fourier coefficients are obtained
          a(r,theta) = sum_{n=0 to Nsh} [ c_n(r)*cos(n*theta) + s_n*sin(n*theta) ]
      with the definition of theta generally code dependent:
          tan(theta) = sign_theta * (Z-Z0)/(R-R0)

      This parametrisation is close to that introduced in Candy PPCF 51, 105009 (2009)

      Inputs:
        R,Z            Flux surface description (nrho,ntheta) without double points
        r0             Minor radius at which the parametrisation is interpolated                         
        Nsh            Number of terms kept in the Fourier expansion
        code           Code convention: 'gkw' or 'imas'
                          gkw:  theta=0 at Z=Z0 on the LFS and increasing counter-clockwise, sign_theta=+1
                          imas: theta=0 at Z=Z0 on the LFS and increasing clockwise (GK IDS convention), sign_theta=-1
        doplots        Perform plots to check the parametrisation if True (default: True)

      Outputs:
        c,s            Fourier parametrisation of the flux surface 
        dcdr, dsdr     Radial derivative of the coefficients above 
        R0, Z0         Reference point used for the parametrisation
        R_out,Z_out    Flux surface description of the parametrised flux surface
        err_out        Average relative error on the parametrisation of a(r0,theta)

      Note that (R,Z) and r0  are assumed to all be given with the same normalisation/units, which 
      can be arbitrary. 
      The c, s, R0, Z0 values will be given with the same normalisation/units as used in input.
  """  
  if code=='gkw':
   sign_theta=1
  elif code=='imas':
   sign_theta=-1
  else:
   print("Unknown code convention. Available: 'gkw' or 'imas'")
   return

  (Nr,Nth)=R.shape
  err_thr = 0.003;  # threshold to warn for poor parametrisation accuracy

  # approximate minor radius and (R0,Z0) of all FS (no interpolation)
  r_coarse = (np.max(R,1)-np.min(R,1))/2
  R0_coarse = np.reshape((np.max(R,1)+np.min(R,1))/2,(Nr,1))
  Z0_coarse = np.reshape((np.max(Z,1)+np.min(Z,1))/2,(Nr,1))

  # check requested FS does not requires extrapolation
  if r0<np.min(r_coarse) or r0>np.max(r_coarse):
    print("No extrapolation allowed, check value of r0")
    return

  # interpolate for a more accurate calculation of r and the local value of (R0, Z0)
  th_coarse = np.arctan2(sign_theta*(Z-np.tile(Z0_coarse,(1,Nth))),R-np.tile(R0_coarse,(1,Nth))) 
  th_fine = np.linspace(-np.pi,np.pi,num=500,endpoint=False)
  r,R0_fine,Z0_fine=np.full((3,Nr),np.nan)
  for ii in range(Nr):
    th_sorted,Ith=np.unique(th_coarse[ii,:],return_index=True)
    thth=np.concatenate(([th_sorted[-1]-2*np.pi],th_sorted))
    RR=np.concatenate(([R[ii,Ith[-1]]],R[ii,Ith]))
    ZZ=np.concatenate(([Z[ii,Ith[-1]]],Z[ii,Ith]))
    Rspl=splrep(thth,RR,per=True)
    Zspl=splrep(thth,ZZ,per=True)
    R_fine=splev(th_fine,Rspl,ext=0)
    Z_fine=splev(th_fine,Zspl,ext=0)
    Rmax=np.max(R_fine)
    Rmin=np.min(R_fine)
    Zmax=np.max(Z_fine)
    Zmin=np.min(Z_fine)
    r[ii]=(Rmax-Rmin)/2
    R0_fine[ii]=(Rmax+Rmin)/2
    Z0_fine[ii]=(Zmax+Zmin)/2
  
  f = interp1d(r,R0_fine, kind='quadratic')
  R0=f(r0)
  f = interp1d(r,Z0_fine, kind='quadratic')
  Z0=f(r0)

  # distance of FS to the local reference point 
  a = np.sqrt((R-R0)**2+(Z-Z0)**2)

  # corresponding poloidal angle (from 0 to 2*pi)
  th = np.arctan2(sign_theta*(Z-Z0),R-R0) 
  th = np.mod(th+2*np.pi,2*np.pi)

  # interpolate on a regular theta grid to perform the Fourier transform
  th_grid = np.linspace(0,2*np.pi,num=Nth,endpoint=False)
  ath = np.full((Nr,Nth),np.nan)
  coeffs = np.full((Nr,Nth),np.nan*1j)
  for ii in range(Nr):
    th_sorted,Ith=np.unique(th[ii,:],return_index=True)
    thth=np.concatenate(([th_sorted[-1]-2*np.pi],th_sorted))
    aa=np.concatenate(([a[ii,Ith[-1]]],a[ii,Ith]))
    aspl=splrep(thth,aa,per=True)
    ath[ii,:]=splev(th_grid,aspl,ext=0)
    coeffs[ii,:]=np.fft.fft(ath[ii,:])
  c_all = coeffs[:,0:Nsh].real/Nth
  c_all[:,1:Nsh] = 2*c_all[:,1:Nsh]
  s_all = -coeffs[:,0:Nsh].imag/Nth
  s_all[:,1:Nsh] = 2*s_all[:,1:Nsh]

  # compute the parametrisation quality  
  THETA=np.tile(th_grid,(Nr,Nsh,1))
  N=np.moveaxis(np.tile(np.arange(Nsh),(Nr,Nth,1)),1,2)
  C=np.moveaxis(np.tile(c_all,(Nth,1,1)),0,2)
  S=np.moveaxis(np.tile(s_all,(Nth,1,1)),0,2)

  a_out = np.sum(C*np.cos(N*THETA)+S*np.sin(N*THETA),1)
  R_out = R0 + a_out*np.cos(THETA[:,0,:]) 
  Z_out = Z0 + sign_theta*a_out*np.sin(THETA[:,0,:]) 

  err=np.sum(abs((ath-a_out)/ath),1)/Nth
  f = interp1d(r,err, kind='quadratic')
  err_out=f(r0)
  if err_out>err_thr:
    print("Warning, parametrisation quality is poor at r=r0, consider increasing Nsh")

  # interpolate the shaping coefficients at the desired location r0
  c,s,dcdr,dsdr=np.full((4,Nsh),np.nan)
  k=np.min([3, Nr-1]) # to deal with cases with 3 radial grid points
  for ii in range(Nsh):
    c_spl=splrep(r,c_all[:,ii],k=k)
    s_spl=splrep(r,s_all[:,ii],k=k)
    c[ii]=splev(r0,c_spl)
    dcdr[ii]=splev(r0,c_spl,der=1)
    s[ii]=splev(r0,s_spl)
    dsdr[ii]=splev(r0,s_spl,der=1)

  # perform a few plots
  if doplots:
    plt.figure()
    plt.plot([0, 1],[err_thr, err_thr],'r')
    plt.plot(r/np.max(r),err)
    plt.xlabel('$r/r_{max}$')
    plt.ylabel('relative error on the parametrisation')

    plt.figure()
    plt.plot(np.transpose(R),np.transpose(Z),'b')
    plt.plot(np.transpose(R_out[3:,:]),np.transpose(Z_out[3:,:]),'r--')

  return c,s,dcdr,dsdr,R0,Z0,R_out,Z_out,err_out


def rz2mxh(R,Z,code,r0=None,Nsh=10,doplots=True):
  """ return r0, R0, dR0dr, Z0, dZ0dr, k, dkdr, c, dcdr, s,dsdr, R_out, Z_out, err_out
      Compute the Miller extended harmonic parametrisation of flux surfaces for the input flux surfaces
      The radial coordinate is defined as r=(max(R)-min(R))/2
      The reference point R0,Z0 is computed for each flux surface as:
          R0 = (max(R(r)) + min(R(r)))/2
          Z0 = (max(Z(r)) + min(Z(r)))/2
      with interpolation since the R,Z description of the flux surface can be coarse.
      The parametrisation of the flux surfaces is defined as:
         R(r,theta)=R0(r) + r*cos(theta_R(r,theta))
         Z(r,theta)=Z0(r) + sign_theta*r*k*sin(theta)
      with
         theta_R(r,theta) = theta + sum_{n=0 to Nsh-1} [ c_n(r)*cos(n*theta) + s_n*sin(n*theta) ]
      The direction of increasing theta is generally code dependent, but we always assume theta=0 at the LFS:
          tan(theta) = sign_theta * (Z-Z0)/(R-R0)

      This parametrisation is described in Arbon PPCF 63, 012001 (2021).

      Inputs:
        R,Z            Flux surface description (nrho,ntheta) without double points
        Nsh            Number of terms kept in the Fourier expansion
        code           Code convention (for sign_theta): 'gkw' or 'imas'
                          gkw:  theta=0 at Z=Z0 on the LFS and increasing counter-clockwise, sign_theta=+1
                          imas: theta=0 at Z=Z0 on the LFS and increasing clockwise (GK IDS convention), sign_theta=-1
        r0             Minor radius at which the parametrisation coefficients in output are given 
                       If r0=None (default), give the coefficients for all flux surfaces in input
        doplots        Perform plots to check the parametrisation if True (default: True)

      Outputs:
        R0, Z0         Reference point used for the parametrisation
        dR0dr, dZ0dr   Radial derivative of the coefficients above
        k              Elongation
        dkdr           Radial derivative of the elongation
        c,s            extended Miller harmonics (from n=0)
        dcdr, dsdr     Radial derivative of the coefficients above 
        R_out,Z_out    Flux surface description of the parametrised flux surface
        err_out        Average relative error on the parametrisation of a(r0,theta)

      Note that (R,Z) and r0  are assumed to all be given with the same normalisation/units, which 
      can be arbitrary. 
      The c, s, k, R0, Z0 values will be given with the same normalisation/units as used in input.
  """  
  if code=='gkw':
   sign_theta=1
  elif code=='imas':
   sign_theta=-1
  elif code=='tglf':
   sign_theta=1
  else:
   print("Unknown code convention. Available: 'gkw' or 'imas'")
   return

  (Nr,Nth)=R.shape
  err_thr = 0.003;  # threshold to warn for poor parametrisation accuracy

  # regular theta grid used for Fourier transform and to check parametrisation quality
  th_grid = np.linspace(0,2*np.pi,num=Nth,endpoint=False)

  # approximate minor radius and (R0,Z0) of all FS (no interpolation)
  r_coarse = (np.max(R,1)-np.min(R,1))/2
  R0_coarse = np.reshape((np.max(R,1)+np.min(R,1))/2,(Nr,1))
  Z0_coarse = np.reshape((np.max(Z,1)+np.min(Z,1))/2,(Nr,1))

  # interpolate for a more accurate calculation of r and the local value of (R0, Z0)
  th_coarse = np.arctan2((Z-np.tile(Z0_coarse,(1,Nth))),R-np.tile(R0_coarse,(1,Nth))) 
  Nth_fine=500
  th_fine = np.linspace(-np.pi,np.pi,num=Nth_fine,endpoint=False)
  r,R0,Z0,k=np.full((4,Nr),np.nan)
  coeffs=np.full((Nr,Nth),np.nan*1j)
  a_check=np.full((Nr,Nth),np.nan)
  for ii in range(Nr):
    th_sorted,Ith=np.unique(th_coarse[ii,:],return_index=True)
    thth=np.concatenate(([th_sorted[-1]-2*np.pi],th_sorted))
    RR=np.concatenate(([R[ii,Ith[-1]]],R[ii,Ith]))
    ZZ=np.concatenate(([Z[ii,Ith[-1]]],Z[ii,Ith]))
    Rspl=splrep(thth,RR,per=True)
    Zspl=splrep(thth,ZZ,per=True)
    R_fine=splev(th_fine,Rspl,ext=0)
    Z_fine=splev(th_fine,Zspl,ext=0)
    Rmax=np.max(R_fine)
    th_Rmax=th_fine[Rmax==R_fine]
    Rmin=np.min(R_fine)
    th_Rmin=th_fine[Rmin==R_fine]
    Zmax=np.max(Z_fine)
    th_Zmax=th_fine[Zmax==Z_fine]
    Zmin=np.min(Z_fine)
    th_Zmin=th_fine[Zmin==Z_fine]
    r[ii]=(Rmax-Rmin)/2
    R0[ii]=(Rmax+Rmin)/2
    Z0[ii]=(Zmax+Zmin)/2
    k[ii]=(Zmax-Zmin)/2/r[ii]
    # distance of FS to the local reference point (used to check parametrisation quality)
    a_fine=np.sqrt((R_fine-R0[ii])**2+(Z_fine-Z0[ii])**2) 
    # compute the angles theta_R with cos(theta_R) = (R-R0)/r
    th_R=np.arccos(np.round((2*R_fine-Rmax-Rmin)/(Rmax-Rmin),14))  
    II=(th_fine<th_Rmax) & (th_fine>np.mod(th_Rmin,2*np.pi)-2*np.pi)
    th_R[II]=2*np.pi-th_R[II]
    # compute the angle theta with sin(theta) = (Z-Z0)/(k*r)
    th=np.arcsin(np.round((2*Z_fine-Zmax-Zmin)/(Zmax-Zmin),14))
    II=(th_fine>th_Zmax) | (th_fine<th_Zmin)
    th[II]=np.pi-th[II]
    th=np.mod(th,2*np.pi)
    # angle difference on which to perform the fourier transform
    delta_theta=-np.mod(th-th_R+np.pi,2*np.pi)+np.pi
    # interpolate on a regular theta grid to perform the Fourier transform
    th_sorted,Ith=np.unique(th,return_index=True)
    thth=np.concatenate(([th_sorted[-1]-2*np.pi],th_sorted))
    dthdth=np.concatenate(([delta_theta[Ith[-1]]],delta_theta[Ith]))
    dthspl=splrep(thth,dthdth,per=True)
    delta_theta_fourier=splev(th_grid,dthspl,ext=0)
    aa=np.concatenate(([a_fine[Ith[-1]]],a_fine[Ith]))
    aspl=splrep(thth,aa,per=True)
    a_check[ii,:]=splev(th_grid,aspl,ext=0)
    # perform the Fourier transform of theta_R-theta
    coeffs[ii,:]=np.fft.fft(delta_theta_fourier)/Nth

  # fill in the Fourier coefficients
  c_all = sign_theta*coeffs[:,0:Nsh].real
  c_all[:,1:Nsh] = 2*c_all[:,1:Nsh]
  s_all = -coeffs[:,0:Nsh].imag
  s_all[:,1:Nsh] = 2*s_all[:,1:Nsh]

  # check the parametrisation quality  
  THETA=np.tile(th_grid,(Nr,Nsh,1))
  N=np.moveaxis(np.tile(np.arange(Nsh),(Nr,Nth,1)),1,2)
  C=np.moveaxis(np.tile(c_all,(Nth,1,1)),0,2)
  S=np.moveaxis(np.tile(s_all,(Nth,1,1)),0,2)

  theta_R_out = np.tile(th_grid,(Nr,1)) + np.sum(sign_theta*C*np.cos(N*THETA)+S*np.sin(N*THETA),1)
  R_out =  np.moveaxis(np.tile(R0,(Nth,1)),0,1) +  np.moveaxis(np.tile(r,(Nth,1)),0,1)*np.cos(theta_R_out)
  Z_out =  np.moveaxis(np.tile(Z0,(Nth,1)),0,1) +  np.moveaxis(np.tile(k*r,(Nth,1)),0,1)*np.sin(np.tile(th_grid,(Nr,1)))
  a_out = np.sqrt((np.moveaxis(np.tile(r,(Nth,1)),0,1)*np.cos(theta_R_out))**2 + 
                  (np.moveaxis(np.tile(k*r,(Nth,1)),0,1)*np.sin(np.tile(th_grid,(Nr,1))))**2)

  err=np.sum(abs((a_check-a_out)/a_check),1)/Nth

  # interpolate the shaping coefficients and compute the radial derivatives
  if r0 is not None:
    r_out = r0
    Nr_out = np.array(r0).size
  else:
    r_out = r
    Nr_out = Nr
  kk=np.min([3, Nr-1]) # to deal with cases with 3 radial grid points
  R0_spl=splrep(r,R0,k=kk)
  Z0_spl=splrep(r,Z0,k=kk)
  k_spl=splrep(r,k,k=kk)
  err_spl=splrep(r,err,k=kk)
  R0=splev(r_out,R0_spl)
  dR0dr=splev(r_out,R0_spl,der=1)
  Z0=splev(r_out,Z0_spl)
  dZ0dr=splev(r_out,Z0_spl,der=1)
  k=splev(r_out,k_spl)
  dkdr=splev(r_out,k_spl,der=1)
  err_out=splev(r_out,err_spl)
  c,s,dcdr,dsdr=np.full((4,Nr_out,Nsh),np.nan)
  for ii in range(0,Nsh):
    c_spl=splrep(r,c_all[:,ii],k=kk)
    s_spl=splrep(r,s_all[:,ii],k=kk)
    c[:,ii]=splev(r_out,c_spl)
    dcdr[:,ii]=splev(r_out,c_spl,der=1)
    s[:,ii]=splev(r_out,s_spl)
    dsdr[:,ii]=splev(r_out,s_spl,der=1)

  # perform a few plots
  if doplots:
    plt.figure()
    plt.plot([0, 1],[err_thr, err_thr],'r')
    plt.plot(r/np.max(r),err)
    plt.xlabel('$r/r_{max}$')
    plt.ylabel('relative error on the parametrisation')

    plt.figure()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot(np.transpose(R),np.transpose(Z),'b')
    plt.plot(np.transpose(R_out),np.transpose(Z_out),'r--')

  return r_out, R0, dR0dr, Z0, dZ0dr, k, dkdr, np.squeeze(c), np.squeeze(dcdr), np.squeeze(s),np.squeeze(dsdr), R_out, Z_out, err_out
