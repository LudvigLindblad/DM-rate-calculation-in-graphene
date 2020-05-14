import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spint
import numpy.linalg as linalg
import scipy.special as spc
from mpmath import mp
from mpmath import fp

import constants as ct
import loadtables
import integration_methods as mint

#overlap parameters
s=0.129;
s2=0;    #Higher order overlap not concidered
t=-3.03*ct.q;

#overlap parameters for sigma wavefunctions
psx=0.16;
psy=0.16;
pss=0.21;
pxx=0.15;
pyy=0.15;
pxy=0.13;

#DM velocity disturubution parameters
v0=220e3;
vesc=550e3;
vobs=244e3;

#Nearest-neighbor vectors
R_1=[ct.a, 0, 0];      
R_2=[-ct.a/2, 3**(1/2)*ct.a/2, 0];
R_3=[-ct.a/2, -3**(1/2)*ct.a/2, 0];

#Phase factor
def fl(lx, ly, lz):
    l=[lx, ly, lz];

    out=mp.exp(1j*mp.fdot(l, R_1))+mp.exp(1j*mp.fdot(l, R_2))+mp.exp(1j*mp.fdot(l, R_3));
    return out;


#Momentum space orbitals (normalized)
def w2pz(kx, ky, kz):
    return ct.nrm_pz*ct.a0**(5/2)*(kz/ct.hbar)/((ct.a0*mp.norm([kx, ky, kz])/ct.hbar)**2+(ct.Zeff2pz/2)**2)**3;

def w2px(kx, ky, kz):
    return ct.nrm_px*ct.a0**(5/2)*(kx/ct.hbar)/((ct.a0*mp.norm([kx, ky, kz])/ct.hbar)**2+(ct.Zeff2pxy/2)**2)**3;

def w2py(kx, ky, kz):
    return ct.nrm_py*ct.a0**(5/2)*(ky/ct.hbar)/((ct.a0*mp.norm([kx, ky, kz])/ct.hbar)**2+(ct.Zeff2pxy/2)**2)**3;
    
def w2s(kx, ky, kz):
    return ct.nrm_s*ct.a0**(3/2)*(((ct.a0*mp.norm([kx, ky, kz]))/ct.hbar)**2-(ct.Zeff2s/2)**2)/(((ct.a0*mp.norm([kx, ky, kz]))/ct.hbar)**2+(ct.Zeff2s/2)**2)**3;


#bloch subfunctions for sigma orbitals
def fi_As(lx, ly, kx, ky, kz):
    return w2s(kx, ky, kz);

def fi_Ax(lx, ly, kx, ky, kz):
    return w2px(kx, ky, kz);

def fi_Ay(lx, ly, kx, ky, kz):
    return w2py(kx, ky, kz);

def fi_Bs(lx, ly, kx, ky, kz):
    return (mp.exp(1j*mp.fdot([lx+kx/ct.hbar, ly+ky/ct.hbar, kz/ct.hbar], R_1))+mp.exp(1j*mp.fdot([lx+kx/ct.hbar, ly+ky/ct.hbar, kz/ct.hbar], R_2))+mp.exp(1j*mp.fdot([lx+kx/ct.hbar, ly+ky/ct.hbar, kz/ct.hbar], R_3)))*w2s(kx, ky, kz);

def fi_Bx(lx, ly, kx, ky, kz):
    return (mp.exp(1j*mp.fdot([lx+kx/ct.hbar, ly+ky/ct.hbar, kz/ct.hbar], R_1))+mp.exp(1j*mp.fdot([lx+kx/ct.hbar, ly+ky/ct.hbar, kz/ct.hbar], R_2))+mp.exp(1j*mp.fdot([lx+kx/ct.hbar, ly+ky/ct.hbar, kz/ct.hbar], R_3)))*w2px(kx, ky, kz);

def fi_By(lx, ly, kx, ky, kz):
    return (mp.exp(1j*mp.fdot([lx+kx/ct.hbar, ly+ky/ct.hbar, kz/ct.hbar], R_1))+mp.exp(1j*mp.fdot([lx+kx/ct.hbar, ly+ky/ct.hbar, kz/ct.hbar], R_2))+mp.exp(1j*mp.fdot([lx+kx/ct.hbar, ly+ky/ct.hbar, kz/ct.hbar], R_3)))*w2py(kx, ky, kz);

table=loadtables.Table(); 

#returns lambda function for normalized pi/sigma electron wavefunction
def createP_pi(lx, ly):
    fle=fl(lx, ly, 0);
    phi_l=-mp.atan(mp.im(fle)/mp.re(fle));

    f=lambda kx,ky,kz: (1+1/(3**(1/2))*mp.exp(1j*phi_l)*fl(lx+kx/ct.hbar, ly+ky/ct.hbar, kz/ct.hbar))*w2pz(kx, ky, kz);
    
    #analytical
    #differential vectors, not used in nearest neighbor approximation
    R_12=[R_1[0]-R_2[0], R_1[1]-R_2[1], R_1[2]-R_2[2]];
    R_23=[R_2[0]-R_3[0], R_2[1]-R_3[1], R_2[2]-R_3[2]];
    R_31=[R_3[0]-R_1[0], R_3[1]-R_1[1], R_3[2]-R_1[2]];

    nrm=mp.sqrt(1/(2+2*s/(3**(1/2))*(mp.cos(phi_l+mp.fdot([lx, ly, 0], R_1))+mp.cos(phi_l+mp.fdot([lx, ly, 0], R_2))+mp.cos(phi_l+mp.fdot([lx, ly, 0], R_3)))+2*s2*(mp.cos(mp.fdot([lx, ly, 0], R_12))+mp.cos(mp.fdot([lx, ly, 0], R_23))+mp.cos(mp.fdot([lx, ly, 0], R_31)))));
    
    return lambda kx, ky, kz: f(kx, ky, kz)*nrm;

def createP_sigma(lx, ly, band): 
    C=table.C_Sigma(ly/(2/mp.sqrt(3)),lx/(mp.sqrt(3)/2),band);   #Values tabulated on a rectangular grid, flipped and compensated for.
    
    C=[C[0], C[1], C[2], C[3]/(3**(1/2)), C[4]/(3**(1/2)), C[5]/(3**(1/2))];  #For sqrt(3) normalisation, see paper for reference.

    f=lambda kx, ky, kz: C[0]*fi_As(lx, ly, kx, ky, kz)+C[1]*fi_Ax(lx, ly, kx, ky, kz)+C[2]*fi_Ay(lx, ly, kx, ky, kz)+C[3]*fi_Bs(lx, ly, kx, ky, kz)+C[4]*fi_Bx(lx, ly, kx, ky, kz)+C[5]*fi_By(lx, ly, kx, ky, kz);
    
    nrm=1/mp.sqrt(mp.conj(C[0])*C[0]+mp.conj(C[1])*C[1]+mp.conj(C[2])*C[2]+3*(mp.conj(C[3])*C[3]+mp.conj(C[4])*C[4]+mp.conj(C[5])*C[5])+2*mp.re((mp.conj(C[0])*C[3]*pss+mp.conj(C[0])*C[4]*psx+mp.conj(C[0])*C[5]*psy+mp.conj(C[1])*C[3]*psx+mp.conj(C[1])*C[4]*pxx+mp.conj(C[1])*C[5]*pxy+mp.conj(C[2])*C[3]*psy+mp.conj(C[2])*C[4]*pxy+mp.conj(C[2])*C[5]*pyy)*(mp.exp(1j*mp.fdot([lx, ly, 0], R_1))+mp.exp(1j*mp.fdot([lx, ly, 0], R_2))+mp.exp(1j*mp.fdot([lx, ly, 0], R_3)))));
   
    return lambda kx, ky, kz: f(kx, ky, kz)*nrm;  

#binding energy functions

def E_pi_minus(lx, ly):
    fle=float(mp.re(mp.sqrt(mp.conj(fl(lx, ly, 0)*fl(lx, ly, 0)))));
    return -1*t*fle/(1+s*fle);

def E_sigma(lx, ly, band):
    E=table.Ematrix(lx, ly, band);
    return -1*ct.q*E;

#form factor of interaction
def F_DM(qr, fdm):
    # 3 cases
    if fdm==1:
        return ct.sigma_e;  
    elif fdm==2:
        return ct.sigma_e*qr;
    elif fdm==3:
        return ct.sigma_e*qr**2;
    else:
        raise Exception("fdm error");       
        return 0;
    
class calculatormk3():
    lastlx=-1;
    lastly=-1;
    P_pi=0;
    P_sigma=0;
    
    #band energy, saved to avoid calling loader more than necessary
    E_b=0;

    #default mp-gl
    method="mp-gl";

    def __init__(self, method):
        self.method=method;

    def vmin(self, q, kf, E, local_mchi):  
        out=(1/q)*(kf**2/(2*ct.me)+E+ct.fi)+q/(2*local_mchi);
        
        if isinstance(out, mp.mpc):
            return -1;
        elif out < 0:
            return -1;
 
        return out;

   #velocity distrubution of incoming DM
    def g(self, v):
        if v+self.vobs < self.vesc:
            return mp.exp((ct.vobs**2+ct.vesc**2)/v0)/ct.K;
        else:
            return 0;
    
    #Analytical expresion of eta
    def eta(self, qr, kr, E, local_mchi):
        vm=self.vmin(qr, kr, E, local_mchi);
        if vm==-1:
            return 0;

        K=ct.K;

        if vm < ct.vesc-ct.vobs:
            out=ct.v0**2*mp.pi/(2*ct.vobs*K)*(-4*mp.exp(-(ct.vesc/ct.v0)**2)*ct.vobs+mp.pi**(1/2)*ct.v0*(mp.erf((vm+ct.vobs)/ct.v0)-mp.erf((vm-ct.vobs)/ct.v0)));
        elif vm < ct.vesc+ct.vobs:
            out=ct.v0**2*mp.pi/(2*ct.vobs*K)*(-2*mp.exp(-(ct.vesc/ct.v0)**2)*(ct.vesc-vm+ct.vobs)+mp.pi**(1/2)*ct.v0*(mp.erf(ct.vesc/ct.v0)-mp.erf((vm-ct.vobs)/ct.v0)));
        else: 
            out=0;

        return out;

    #q momentum integral
    def dR(self, lx, ly, kr, kt, kp, qmin, qmax, band, fdm, local_mchi):
        if band=="pi":
            if not self.lastlx==lx or not self.lastly==ly:
                self.P_pi=createP_pi(lx, ly);
                self.lastlx=lx;              
                self.lastly=ly;
                self.E_b=E_pi_minus(lx, ly);

                 
            sqP=lambda qr, qt, qp: mp.re(self.P_pi(qr*mp.sin(qt)*mp.cos(qp)-kr*mp.sin(kt)*mp.cos(kp), qr*mp.sin(qt)*mp.sin(qp)-kr*mp.sin(kt)*mp.sin(kp), qr*mp.cos(qt)-kr*mp.cos(kt)))**2+mp.im(self.P_pi(qr*mp.sin(qt)*mp.cos(qp)-kr*mp.sin(kt)*mp.cos(kp), qr*mp.sin(qt)*mp.sin(qp)-kr*mp.sin(kt)*mp.sin(kp), qr*mp.cos(qt)-kr*mp.cos(kt)))**2;
        else:
            if not self.lastlx==lx or not self.lastly==ly:
                self.P_sigma=createP_sigma(lx, ly, band);
                self.lastlx=lx;
                self.lastly=ly;
                self.E_b=E_sigma(lx, ly, band);             

            sqP=lambda qr, qt, qp: mp.re(self.P_sigma(qr*mp.sin(qt)*mp.cos(qp)-kr*mp.sin(kt)*mp.cos(kp), qr*mp.sin(qt)*mp.sin(qp)-kr*mp.sin(kt)*mp.sin(kp), qr*mp.cos(qt)-kr*mp.cos(kt)))**2+mp.im(self.P_sigma(qr*mp.sin(qt)*mp.cos(qp)-kr*mp.sin(kt)*mp.cos(kp), qr*mp.sin(qt)*mp.sin(qp)-kr*mp.sin(kt)*mp.sin(kp), qr*mp.cos(qt)-kr*mp.cos(kt)))**2;

        integrand=lambda qr, qt, qp: (qr)*mp.sin(qt)/(4*mp.pi)*self.eta(qr, kr, self.E_b, local_mchi)*F_DM(qr, fdm)*sqP(qr, qt, qp);

        result=mint.integrate(integrand, [qmin, qmax], [0, mp.pi], [0, 2*mp.pi], self.method);
        return result;


    #Solid angle integral over k
    def dRdEe(self, lx, ly, band, fdm, mchi_index=0):
        integrand=self.dRdEe_noint(lx, ly, band, fdm, mchi_index);

        return lambda kr: (1/ct.hbar)**3*((kr)**3)/(32*mp.pi**4)*mint.integrate(lambda kt, kp: integrand(kr, kt, kp), [0, mp.pi], [0, 2*mp.pi], 0, self.method);
    
    #m_chi index only used in pi band. set to 0 to disable m_chi looping
    def dRdEe_noint(self, lx, ly, band, fdm, mchi_index):
       if band=="pi":
            local_mchi=ct.mchi;   
            if mchi_index!=0:  
                local_mchi=ct.m_chi_list[mchi_index-1];
            
            qsqrt=lambda kr: mp.sqrt(local_mchi**2*ct.vmax**2-(local_mchi/ct.me)*kr**2-2*local_mchi*(E_pi_minus(lx, ly)+ct.fi));

            qmini=lambda kr: local_mchi*ct.vmax-qsqrt(kr);
            qmaxi=lambda kr: local_mchi*ct.vmax+qsqrt(kr);            

       else:
            local_mchi=ct.mchi;
            qsqrt=lambda kr: mp.sqrt(ct.mchi**2*ct.vmax**2-(ct.mchi/ct.me)*kr**2-2*ct.mchi*(E_sigma(lx, ly,band)+ct.fi));

            qmini=lambda kr: ct.mchi*ct.vmax-qsqrt(kr);
            qmaxi=lambda kr: ct.mchi*ct.vmax+qsqrt(kr);
 

       return lambda kr, kt, kp: (1/2)*mp.sin(kt)*self.dR(lx, ly, kr, kt, kp, qmini(kr), qmaxi(kr), band, fdm, local_mchi);
